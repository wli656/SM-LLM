import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import logging as hf_logging

from .config import LLM_REGISTRY

hf_logging.set_verbosity_error()

class GaussianDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        assert 0 <= p < 1
        self.sigma = p / (1 - p)

    def forward(self, x):
        if self.training and self.sigma > 0:
            return x * (1.0 + self.sigma * torch.randn_like(x))
        return x


def mlp(in_dim, hidden_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads'
        self.n_heads = n_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        b, l, _ = x.shape
        q, k, v = (proj(x).view(b, l, self.n_heads, -1).transpose(1, 2)
                   for proj in (self.q_proj, self.k_proj, self.v_proj))
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.out_proj(out.transpose(1, 2).reshape(b, l, -1))


class AttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff = mlp(d_model, d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attn(x)))
        return self.norm2(x + self.dropout(self.ff(x)))

class DimensionSegmentEmbedding(nn.Module):
    def __init__(self, seg_len, d_e):
        super().__init__()
        self.seg_len = seg_len
        self.proj = nn.Linear(seg_len, d_e)

    def forward(self, x):
        b, t, d = x.shape
        x = x.transpose(1, 2).reshape(b, d, t // self.seg_len, self.seg_len)
        return self.proj(x)


class SegmentMerging(nn.Module):
    def __init__(self, d_e, win_size):
        super().__init__()
        self.win_size = win_size
        self.norm = nn.LayerNorm(win_size * d_e)
        self.proj = nn.Linear(win_size * d_e, d_e)

    def forward(self, z):
        pad = (-z.shape[2]) % self.win_size
        if pad:
            z = torch.cat([z, z[:, :, -pad:]], dim=2)
        z = torch.cat([z[:, :, i::self.win_size] for i in range(self.win_size)], dim=-1)
        return self.proj(self.norm(z))


class DualAttentionLayer(nn.Module):
    def __init__(self, d_e, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.temporal_attention = AttentionBlock(d_e, n_heads, d_ff, dropout)
        self.cross_feature_attention = AttentionBlock(d_e, n_heads, d_ff, dropout)
        self.fusion = mlp(2 * d_e, d_ff, d_e)

    def forward(self, z):
        b, d, n, e = z.shape
        z_time = self.temporal_attention(z.reshape(b * d, n, e)).reshape(b, d, n, e)
        z_dim = self.cross_feature_attention(z.transpose(1, 2).reshape(b * n, d, e))
        z_dim = z_dim.reshape(b, n, d, e).transpose(1, 2)
        return self.fusion(torch.cat([z_time, z_dim], dim=-1))


class TemporalEncoder(nn.Module):
    def __init__(self, n_vars, in_len, seg_len=1, d_e=128, n_heads=8, d_ff=1024,
                 n_scales=3, win_size=2, dropout=0.0):
        super().__init__()
        n_segments = math.ceil(in_len / seg_len)
        self.front_pad = n_segments * seg_len - in_len

        self.embedding = DimensionSegmentEmbedding(seg_len, d_e)
        self.pos_embedding = nn.Parameter(torch.randn(1, n_vars, n_segments, d_e))
        self.norm = nn.LayerNorm(d_e)

        self.merges = nn.ModuleList(
            [nn.Identity()] + [SegmentMerging(d_e, win_size) for _ in range(n_scales - 1)])
        self.layers = nn.ModuleList(
            [DualAttentionLayer(d_e, n_heads, d_ff, dropout) for _ in range(n_scales)])

    def forward(self, x):
        if self.front_pad:
            x = torch.cat([x[:, :1].expand(-1, self.front_pad, -1), x], dim=1)
        z = self.norm(self.embedding(x) + self.pos_embedding)

        scales = []
        for merge, layer in zip(self.merges, self.layers):
            z = layer(merge(z))
            scales.append(z)
        return scales

class NumericStaticEncoder(nn.Module):
    def __init__(self, n_static, hidden=(64, 128), d_s=128, dropout=0.2):
        super().__init__()
        dims = [n_static, *hidden, d_s]
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out), nn.ReLU(), GaussianDropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x_s):
        return self.net(x_s)


class LLMEncoder(nn.Module):
    def __init__(self, llm='BERT', prompt_len=64, mode='embedding', llm_layers=3, llm_path=None):
        super().__init__()
        if mode not in ('embedding', 'hidden'):
            raise ValueError("mode must be 'embedding' or 'hidden'")
        self.mode = mode
        self.prompt_len = prompt_len
        self.tokenizer = load_tokenizer(llm, llm_path)

        source = llm_path or LLM_REGISTRY[llm]
        config = AutoConfig.from_pretrained(source)
        config.num_hidden_layers = llm_layers
        backbone = AutoModel.from_pretrained(source, config=config)
        backbone.requires_grad_(False)

        self.backbone = backbone.get_input_embeddings() if mode == 'embedding' else backbone
        self.dim = backbone.get_input_embeddings().embedding_dim
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, prompts, device):
        tokens = self.tokenizer(list(prompts), padding='max_length', truncation=True,
                                max_length=self.prompt_len, return_tensors='pt')
        ids = tokens['input_ids'].to(device)
        mask = tokens['attention_mask'].to(device)
        if self.mode == 'embedding':
            h = self.backbone(ids)
        else:
            h = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        return h, mask


def load_tokenizer(llm='BERT', llm_path=None):
    tokenizer = AutoTokenizer.from_pretrained(llm_path or LLM_REGISTRY[llm])
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(f'Tokenizer of {llm} defines neither a pad nor an EOS token.')
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'
    return tokenizer

class PredictionHead(nn.Module):
    def __init__(self, in_dim, hidden, out_len, dropout=0.2):
        super().__init__()
        layers = []
        for d_in, d_out in zip([in_dim, *hidden[:-1]], hidden):
            layers += [nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out), nn.ReLU(), GaussianDropout(dropout)]
        layers.append(nn.Linear(hidden[-1], out_len))
        self.net = nn.Sequential(*layers)

    def forward(self, h):
        return self.net(h)

class SMLLM(nn.Module):
    def __init__(self, n_vars, n_static, in_len, out_len, prompt_len,
                 llm='BERT', llm_mode='embedding', llm_layers=3, llm_path=None,
                 seg_len=1, win_size=2, n_scales=3, d_e=128, n_heads=8, d_ff=1024, attn_dropout=0.0,
                 d_s=128, d_model=128, head_hidden=None, dropout=0.2):
        super().__init__()
        self.temporal_encoder = TemporalEncoder(n_vars, in_len, seg_len, d_e, n_heads, d_ff,
                                                n_scales, win_size, attn_dropout)
        self.numeric_encoder = NumericStaticEncoder(n_static, (64, 128), d_s, dropout)
        self.llm_encoder = LLMEncoder(llm, prompt_len, llm_mode, llm_layers, llm_path)
        self.temporal_proj = mlp(d_e, d_model, d_model)
        self.numeric_proj = mlp(d_s, d_model, d_model)
        self.llm_proj = mlp(self.llm_encoder.dim, d_model, d_model)
        if head_hidden is None:
            head_hidden = (d_model, 64, 32) if out_len <= 5 else (d_model, 128, 64)
        self.head = PredictionHead((n_vars + 1 + prompt_len) * d_model, head_hidden, out_len, dropout)

    def forward(self, x_d, x_s, prompts):
        scales = self.temporal_encoder(x_d)
        z_temporal = torch.stack([z.mean(dim=2) for z in scales]).mean(dim=0)
        z_temporal = self.temporal_proj(z_temporal)

        h_mlp = self.numeric_proj(self.numeric_encoder(x_s)).unsqueeze(1)
        h_llm, mask = self.llm_encoder(prompts, x_d.device)
        h_llm = self.llm_proj(h_llm) * mask.unsqueeze(-1).to(h_llm.dtype)

        h_fused = torch.cat([z_temporal, h_mlp, h_llm], dim=1)
        return self.head(h_fused.flatten(1))

    def trainable_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith('llm_encoder.')}

    def load_trainable_state_dict(self, state_dict):
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        missing = [k for k in missing if not k.startswith('llm_encoder.')]
        if missing or unexpected:
            raise RuntimeError(f'State dict mismatch. Missing: {missing}. Unexpected: {unexpected}.')
