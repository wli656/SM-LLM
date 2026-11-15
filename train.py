import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sm_llm.config import EXPERIMENTS, LLM_REGISTRY, STATIC_FEATURES, TASKS, TS_VARIABLES
from sm_llm.data import Normalizer, SoilMoistureDataset, load_table, split_train_val, to_arrays
from sm_llm.model import SMLLM, load_tokenizer
from sm_llm.prompts import PROMPT_TYPES


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group('experiment')
    g.add_argument('--experiment', choices=sorted(EXPERIMENTS),
                   help='preset for --task/--train_years/--targets (explicit flags take precedence)')
    g.add_argument('--task', choices=sorted(TASKS), help='15_5 (short-term) or 90_30 (subseasonal)')
    g.add_argument('--train_years', type=int, help='number of years preceding the target used for training')
    g.add_argument('--targets', type=int, nargs='+', help='held-out target (test) years')
    g.add_argument('--data_dir', default='data', help='folder containing New_data_<task>.csv')
    g.add_argument('--data_path', default=None, help='explicit path to the data table (.csv or .xlsx)')
    g.add_argument('--out_dir', default='runs', help='root folder for checkpoints and logs')
    g.add_argument('--run_name', default=None, help='default: <task>_train<N>y_<llm>_<prompt>')

    g = p.add_argument_group('prompt and LLM')
    g.add_argument('--prompt', default='P2', choices=PROMPT_TYPES, help='prompt template (Fig. 7)')
    g.add_argument('--prompt_values', default='raw', choices=['raw', 'normalized'],
                   help='serialize physical values or the z-scored model inputs')
    g.add_argument('--llm', default='BERT', choices=sorted(LLM_REGISTRY), help='pretrained LLM backbone')
    g.add_argument('--llm_path', default=None, help='local folder / HF id overriding the registry')
    g.add_argument('--llm_mode', default='embedding', choices=['embedding', 'hidden'],
                   help="'embedding': frozen token embeddings; 'hidden': frozen LLM last hidden states")
    g.add_argument('--llm_layers', type=int, default=3, help="LLM layers kept with --llm_mode hidden")

    g = p.add_argument_group('model')
    g.add_argument('--seg_len', type=int, default=1, help='segment (patch) length P')
    g.add_argument('--win_size', type=int, default=2, help='segments merged between scales')
    g.add_argument('--n_scales', type=int, default=3, help='number of scales S + 1')
    g.add_argument('--d_e', type=int, default=128, help='temporal embedding dimension')
    g.add_argument('--n_heads', type=int, default=8)
    g.add_argument('--d_ff', type=int, default=1024, help='hidden size of the attention-block MLPs')
    g.add_argument('--attn_dropout', type=float, default=0.0)
    g.add_argument('--d_s', type=int, default=128, help='output size of the numeric static encoder')
    g.add_argument('--d_model', type=int, default=128, help='dimension of the fused representation')
    g.add_argument('--dropout', type=float, default=0.2, help='Gaussian dropout of static encoder and head')

    g = p.add_argument_group('optimization')
    g.add_argument('--epochs', type=int, default=500)
    g.add_argument('--patience', type=int, default=50, help='early-stopping patience (epochs)')
    g.add_argument('--batch_size', type=int, default=32)
    g.add_argument('--lr', type=float, default=1e-4)
    g.add_argument('--weight_decay', type=float, default=1e-5)
    g.add_argument('--val_ratio', type=float, default=0.1)
    g.add_argument('--seed', type=int, default=64)
    g.add_argument('--num_workers', type=int, default=0)
    g.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    args = p.parse_args()
    if args.experiment:
        for key, value in EXPERIMENTS[args.experiment].items():
            if getattr(args, key) is None:
                setattr(args, key, value)
    if args.task is None or args.train_years is None or args.targets is None:
        p.error('give --experiment, or all of --task, --train_years and --targets')

    args.data_path = Path(args.data_path or Path(args.data_dir) / TASKS[args.task]['file'])
    args.run_name = args.run_name or f'{args.task}_train{args.train_years}y_{args.llm}_{args.prompt}'
    args.run_dir = Path(args.out_dir) / args.run_name
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prompt_length(tokenizer, prompts, margin=8, chunk=4096):
    unique = list(set(prompts))
    longest = 1
    for i in range(0, len(unique), chunk):
        ids = tokenizer(unique[i:i + chunk])['input_ids']
        longest = max(longest, max(len(x) for x in ids))
    return longest + margin


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for x_d, x_s, prompts, y in tqdm(loader, desc='train' if training else 'val', leave=False):
            x_d, x_s, y = x_d.to(device), x_s.to(device), y.to(device)
            loss = F.mse_loss(model(x_d, x_s, prompts), y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(y)
            count += len(y)
    return total / count


def train_target(df, target, args):
    in_len, out_len = TASKS[args.task]['in_len'], TASKS[args.task]['out_len']
    train_years = list(range(target - args.train_years, target))
    out_dir = args.run_dir / f'target_{target}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'\n========== target {target} | training years {train_years} ==========')

    source = df[df['Year'].isin(train_years)].reset_index(drop=True)
    missing = sorted(set(train_years) - set(source['Year'].unique()))
    if missing:
        raise ValueError(f'No data for training year(s) {missing}.')

    train_idx, val_idx = split_train_val(source, args.val_ratio, args.seed)
    x_d, x_s, y = to_arrays(source, in_len, out_len)
    normalizer = Normalizer.fit(x_d[train_idx], x_s[train_idx])

    def make_dataset(idx):
        return SoilMoistureDataset(x_d[idx], x_s[idx], y[idx], normalizer,
                                   args.prompt, args.prompt_values, in_len, out_len)

    train_set, val_set = make_dataset(train_idx), make_dataset(val_idx)
    print(f'train samples: {len(train_set):,} | validation samples: {len(val_set):,}')

    tokenizer = load_tokenizer(args.llm, args.llm_path)
    prompt_len = prompt_length(tokenizer, train_set.prompts + val_set.prompts)
    print(f'prompt {args.prompt}: L = {prompt_len} tokens | example: "{train_set.prompts[0][:120]}..."')

    model_kwargs = dict(
        n_vars=len(TS_VARIABLES), n_static=len(STATIC_FEATURES), in_len=in_len, out_len=out_len,
        prompt_len=prompt_len, llm=args.llm, llm_mode=args.llm_mode, llm_layers=args.llm_layers,
        llm_path=args.llm_path, seg_len=args.seg_len, win_size=args.win_size, n_scales=args.n_scales,
        d_e=args.d_e, n_heads=args.n_heads, d_ff=args.d_ff, attn_dropout=args.attn_dropout,
        d_s=args.d_s, d_model=args.d_model, dropout=args.dropout,
    )
    set_seed(args.seed)
    device = torch.device(args.device)
    model = SMLLM(**model_kwargs).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f'trainable parameters: {sum(p.numel() for p in params):,}')

    pin = device.type == 'cuda'
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    best_loss, best_epoch, log = float('inf'), 0, []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device)
        log.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        pd.DataFrame(log).to_csv(out_dir / 'train_log.csv', index=False)

        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch = val_loss, epoch
            torch.save({
                'model_kwargs': model_kwargs,
                'model_state': model.trainable_state_dict(),
                'normalizer': normalizer.state_dict(),
                'task': args.task,
                'prompt': args.prompt,
                'prompt_values': args.prompt_values,
                'data_path': str(args.data_path),
                'train_years': train_years,
                'target_year': target,
                'epoch': epoch,
                'val_loss': val_loss,
            }, out_dir / 'checkpoint.pt')
        print(f'epoch {epoch:3d} | train {train_loss:.5f} | val {val_loss:.5f} | '
              f'{time.time() - start:5.1f}s' + (' | saved' if improved else ''))

        if epoch - best_epoch >= args.patience:
            print(f'Early stopping: no improvement for {args.patience} epochs.')
            break
    print(f'Best epoch {best_epoch} (val MSE {best_loss:.5f}) -> {out_dir / "checkpoint.pt"}')


def main():
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.run_dir / 'config.json').write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2))

    task = TASKS[args.task]
    df = load_table(args.data_path, task['in_len'], task['out_len'])
    for target in args.targets:
        train_target(df, target, args)


if __name__ == '__main__':
    main()
