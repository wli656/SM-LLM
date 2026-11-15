import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sm_llm.config import REPORTED_LEADS, TASKS
from sm_llm.data import Normalizer, SoilMoistureDataset, load_table, to_arrays
from sm_llm.metrics import METRIC_NAMES, compute_metrics
from sm_llm.model import SMLLM


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', required=True, help='folder created by train.py')
    p.add_argument('--leads', type=int, nargs='+', default=None,
                   help='lead days in the table (default: 1-5 for 15_5, 1/7/14/21/30 for 90_30)')
    p.add_argument('--data_path', default=None, help='override the data path stored in the checkpoints')
    p.add_argument('--llm_path', default=None, help='override the LLM source stored in the checkpoints')
    p.add_argument('--recompute', action='store_true', help='re-run inference even if predictions exist')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


@torch.no_grad()
def predict_target_year(ckpt, df, args):
    task = TASKS[ckpt['task']]
    in_len, out_len = task['in_len'], task['out_len']
    test = df[df['Year'] == ckpt['target_year']].reset_index(drop=True)
    if test.empty:
        raise ValueError(f'No samples for target year {ckpt["target_year"]}.')

    x_d, x_s, y = to_arrays(test, in_len, out_len)
    normalizer = Normalizer.from_state_dict(ckpt['normalizer'])
    dataset = SoilMoistureDataset(x_d, x_s, y, normalizer, ckpt['prompt'], ckpt['prompt_values'],
                                  in_len, out_len)

    model_kwargs = dict(ckpt['model_kwargs'])
    if args.llm_path:
        model_kwargs['llm_path'] = args.llm_path
    model = SMLLM(**model_kwargs)
    model.load_trainable_state_dict(ckpt['model_state'])
    model.to(args.device).eval()

    tokenizer, prompt_len = model.llm_encoder.tokenizer, model_kwargs['prompt_len']
    lengths = [len(ids) for ids in tokenizer(sorted(set(dataset.prompts)))['input_ids']]
    n_long = sum(n > prompt_len for n in lengths)
    if n_long:
        print(f'  note: {n_long} distinct prompts exceed L = {prompt_len} tokens and are truncated')

    preds = []
    for xd, xs, prompts, _ in tqdm(DataLoader(dataset, args.batch_size), desc='predict', leave=False):
        preds.append(model(xd.to(args.device), xs.to(args.device), list(prompts)).cpu().numpy())
    pred = np.concatenate(preds)

    out = test[['ID', 'Date']].copy()
    for h in range(out_len):
        out[f'obs_{h + 1}'] = y[:, h]
    for h in range(out_len):
        out[f'pred_{h + 1}'] = pred[:, h]
    return out


def fmt(value):
    text = f'{value:.3f}'
    return '0.000' if text == '-0.000' else text


def to_markdown(table, title):
    header = ['Target year', 'Lead time (day)', *METRIC_NAMES]
    lines = [f'**{title}**', '', '| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    previous = None
    for _, row in table.iterrows():
        year = str(row['target_year']) if row['target_year'] != previous else ''
        previous = row['target_year']
        lines.append('| ' + ' | '.join([year, str(row['lead'])] + [fmt(row[m]) for m in METRIC_NAMES]) + ' |')
    return '\n'.join(lines)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    ckpt_paths = sorted(run_dir.glob('target_*/checkpoint.pt'))
    if not ckpt_paths:
        raise FileNotFoundError(f'No target_*/checkpoint.pt found in {run_dir}')

    df, rows = None, []
    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        year, task = ckpt['target_year'], ckpt['task']
        pred_path = ckpt_path.parent / f'predictions_{year}.csv'
        print(f'target {year} (trained on {ckpt["train_years"]}, best epoch {ckpt["epoch"]})')

        if pred_path.exists() and not args.recompute:
            predictions = pd.read_csv(pred_path)
        else:
            if df is None:
                df = load_table(args.data_path or ckpt['data_path'],
                                TASKS[task]['in_len'], TASKS[task]['out_len'])
            predictions = predict_target_year(ckpt, df, args)
            predictions.to_csv(pred_path, index=False)

        for h in range(1, TASKS[task]['out_len'] + 1):
            scores = compute_metrics(predictions[f'obs_{h}'], predictions[f'pred_{h}'])
            rows.append({'target_year': year, 'lead': h, **scores})

    metrics = pd.DataFrame(rows)
    metrics.to_csv(run_dir / 'metrics_by_lead.csv', index=False)

    leads = args.leads or REPORTED_LEADS[task]
    table = metrics[metrics['lead'].isin(leads)]
    average = table.groupby('lead', as_index=False)[list(METRIC_NAMES)].mean()
    average.insert(0, 'target_year', 'Average')
    table = pd.concat([table, average], ignore_index=True)
    table.to_csv(run_dir / 'table.csv', index=False)

    in_len, out_len = TASKS[task]['in_len'], TASKS[task]['out_len']
    n_years = len(ckpt['train_years'])
    title = (f'SM-LLM: {out_len}-day ahead forecasting from the preceding {in_len} days, '
             f'{n_years}-year training ({ckpt["prompt"]}, {ckpt["model_kwargs"]["llm"]})')
    markdown = to_markdown(table, title)
    (run_dir / 'table.md').write_text(markdown + '\n')
    print('\n' + markdown)
    print(f'\nSaved metrics_by_lead.csv, table.csv and table.md to {run_dir}')


if __name__ == '__main__':
    main()
