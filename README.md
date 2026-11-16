# SM-LLM

Code for **"SM-LLM: A Deep Learning Framework Integrating Pretrained Language Models for Soil Moisture Forecasting"**.

SM-LLM forecasts in-situ surface (0–5 cm) soil moisture from meteorological and remote-sensing time series together with static geophysical context. Static features are encoded twice: numerically by an MLP and semantically by a **frozen** pretrained LLM that reads them as a structured text prompt.

```
time series  X ∈ R^{T×D} ──► Temporal encoder (multi-scale dual attention) ──┐
static data  X ∈ R^{K}   ──► MLP encoder ────────────────────────────────────├──► Fusion ──► Prediction head ──► Ŷ ∈ R^{H}
             └─ prompt ───► frozen LLM (BERT, SmolLM, GPT-2, TinyLlama, LLaMA)┘
```

## Repository structure

| File | Content |
|---|---|
| `sm_llm/model.py` | SM-LLM model: temporal encoder, static encoder (MLP + LLM), fusion module, prediction head |
| `sm_llm/prompts.py` | Prompt templates P0–P3 (Fig. 7) |
| `sm_llm/data.py` | Data loading, station selection, train/validation split, normalization, `Dataset` |
| `sm_llm/metrics.py` | R, RMSE, bias, ubRMSE, KGE |
| `sm_llm/config.py` | Variables, land-cover codes, LLM backbones, experiment presets |
| `sm_llm/stations.py` | The 200 SCAN/USCRN stations used in the paper |
| `train.py` | Training under the out-of-year protocol |
| `make_results.py` | Inference on held-out years and result tables |

Mapping of the code to the paper (Section III):

| Paper | Code (`sm_llm/model.py`) |
|---|---|
| Dimension–segment-wise embedding, Eq. (2) | `DimensionSegmentEmbedding` |
| Temporal attention, Eqs. (3)–(5) | `DualAttentionLayer.temporal_attention` |
| Cross-feature attention, Eqs. (6)–(8) | `DualAttentionLayer.cross_feature_attention` |
| Temporal / cross-feature fusion | `DualAttentionLayer.fusion` |
| Temporal aggregation P(·), Eq. (9); multi-scale encoding, Eq. (10) | `SegmentMerging`, `TemporalEncoder` |
| MLP static encoder, Eq. (11) | `NumericStaticEncoder` |
| Prompt serialization, Eq. (12) | `sm_llm/prompts.py` |
| Frozen LLM encoder, Eq. (13) | `LLMEncoder` |
| Fusion module, Eqs. (14)–(19) | `SMLLM.forward` |
| Prediction head, Eqs. (20)–(21) | `PredictionHead` |

## Installation

Tested with Python 3.11, PyTorch 2.7.1 (CUDA 11.8) on an NVIDIA RTX 4090.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Pretrained LLMs are downloaded from the Hugging Face Hub on first use. LLaMA-7B (`huggyllama/llama-7b`) is large; use `--llm_path` to point to a local copy of any backbone.

## Data

Prepare the dataset and place the tables in `data/`:

```
data/
├── New_data_15_5_200loc.csv     # 15-day input -> 5-day forecast
└── New_data_90_30_200loc.csv    # 90-day input -> 30-day forecast
```

If you start from the full tables (`New_data_15_5.csv`, `New_data_90_30.csv`), create the 200-station versions with

```bash
python prepare_data.py                     # reads data/New_data_*.csv, writes data/New_data_*_200loc.csv
python prepare_data.py --drop_duplicates   # additionally removes repeated (ID, Date) rows
```

Each row is one forecast sample (`ID`, `Date`, `Year`) with:

- time series `<var>_<k>` for `var ∈ {eto, etr, pr, sph, srad, vpd, vs, LST, SMAP}` and `k = T, …, 1` days before the forecast date;
- static features (land cover `LC_name`, terrain, soil, SMAP snapshot, vegetation and SAR indices, see `sm_llm/config.py`);
- targets `VWC_5<h>`: in-situ soil moisture at 5 cm for lead day `h = 1, …, H`.

Only the 200 stations listed in `sm_llm/stations.py` are used (`load_table` applies the same filter, so the full tables also work via `--data_path`). `.xlsx` files are also accepted.

## Reproducing the results

`train.py` trains one model per held-out target year on the preceding `N` years. 10 % of the station-month blocks of the training years are held out for validation (early stopping and checkpoint selection); target-year data are never used during training. `make_results.py` then predicts each target year and builds the table.

| Table | Setting | Training | Results |
|---|---|---|---|
| II | 15 → 5 days, single-year training, targets 2017–2022 | `python train.py --experiment table2` | `python make_results.py --run_dir runs/15_5_train1y_BERT_P2` |
| III | 15 → 5 days, three-year training, targets 2019–2022 | `python train.py --experiment table3` | `python make_results.py --run_dir runs/15_5_train3y_BERT_P2` |
| IV | 90 → 30 days, three-year training, targets 2019–2022 | `python train.py --experiment table4` | `python make_results.py --run_dir runs/90_30_train3y_BERT_P2` |

For every run, `make_results.py` writes to the run folder:

- `target_<year>/predictions_<year>.csv`: observed and predicted soil moisture for every station and day (lead days 1…H);
- `metrics_by_lead.csv`: R, RMSE, bias, ubRMSE and KGE for every target year and lead day;
- `table.csv` / `table.md`: the reported lead days per target year and their average across target years.

Metrics of a target year are computed over all samples of that year (all stations pooled); bias is mean(prediction − observation).

### Prompt and LLM sensitivity (Tables V and VI)

Change the prompt template or the backbone, then read the lead-1 `Average` row of the resulting table:

```bash
python train.py --experiment table3 --prompt P3                 # P0 | P1 | P2 | P3
python train.py --experiment table3 --llm SMOLLM                # BERT | SMOLLM | GPT2 | TINYLLAMA | LLAMA
python make_results.py --run_dir runs/15_5_train3y_SMOLLM_P2
```

### Main options

| Option | Default | Description |
|---|---|---|
| `--prompt` | `P2` | P0 no prompt, P1 task description, P2 static data, P3 variable description + static data |
| `--prompt_values` | `raw` | serialize physical values (`raw`) or the z-scored model inputs (`normalized`) |
| `--llm` | `BERT` | pretrained backbone (`sm_llm/config.py`) |
| `--llm_mode` | `embedding` | `embedding`: frozen pretrained token embeddings of the LLM; `hidden`: last hidden states of the frozen LLM (first `--llm_layers` layers) |
| `--seg_len` / `--win_size` / `--n_scales` | 1 / 2 / 3 | segment length P, merge window, number of scales S+1 |
| `--d_e` / `--d_model` / `--n_heads` / `--d_ff` | 128 / 128 / 8 / 1024 | model sizes |
| `--epochs` / `--patience` / `--batch_size` / `--lr` | 500 / 50 / 32 / 1e-4 | optimization (AdamW, MSE loss) |
| `--seed` | 64 | random seed |

Run `python train.py -h` for all options.

## Citation

```bibtex
@article{li_smllm,
  title  = {SM-LLM: A Deep Learning Framework Integrating Pretrained Language Models for Soil Moisture Forecasting},
  author = {Li, Wanru and Cai, Shuohao and Yang, Zhengwei and Zhang, Zhou and Huang, Jingyi},
  year   = {to be updated}
}
```

## Acknowledgment

Supported by the USDA NIFA Agriculture and Food Research Initiative: Foundational Program (Grant 2023-67021-40007). The dimension–segment-wise embedding and segment merging follow Crossformer (Zhang & Yan, ICLR 2023).
