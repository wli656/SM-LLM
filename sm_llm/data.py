from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import (LC_CODES, LC_COLUMNS, PROMPT_EXCLUDE, STATIC_FEATURES, STATIC_NUMERIC,
                     TARGET_PREFIX, TS_VARIABLES)
from .prompts import build_prompt
from .stations import STATIONS

def ts_columns(in_len):
    return [f'{v}_{k}' for v in TS_VARIABLES for k in range(in_len, 0, -1)]


def target_columns(out_len):
    return [f'{TARGET_PREFIX}{h}' for h in range(1, out_len + 1)]

def load_table(path, in_len, out_len, stations=STATIONS):
    path = Path(path)
    numeric = ts_columns(in_len) + list(STATIC_NUMERIC) + target_columns(out_len)
    usecols = ['ID', 'Date', 'Year', 'LC_name'] + numeric

    if path.suffix.lower() in ('.xlsx', '.xls'):
        header = pd.read_excel(path, nrows=0).columns
    else:
        header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in usecols if c not in header]
    if missing:
        raise KeyError(f'{path.name} is missing {len(missing)} expected columns, e.g. {missing[:5]}')

    if path.suffix.lower() in ('.xlsx', '.xls'):
        df = pd.read_excel(path, usecols=usecols)
        df[numeric] = df[numeric].astype(np.float32)
    else:
        df = pd.read_csv(path, usecols=usecols, dtype={c: np.float32 for c in numeric})

    df = df.drop_duplicates(subset=['ID', 'Date'])
    df = df[df['ID'].isin(stations)]

    unknown = set(df['LC_name'].unique()) - set(LC_CODES)
    if unknown:
        raise ValueError(f'Unknown land-cover classes: {sorted(unknown)}')
    if df[numeric].isna().any().any():
        raise ValueError('The table contains missing values in model inputs or targets.')

    df = df.sort_values(['ID', 'Date']).reset_index(drop=True)
    print(f'Loaded {len(df):,} samples from {df["ID"].nunique()} stations ({path.name}).')
    return df


def to_arrays(df, in_len, out_len):
    n = len(df)
    x_d = df[ts_columns(in_len)].to_numpy(np.float32)
    x_d = x_d.reshape(n, len(TS_VARIABLES), in_len).transpose(0, 2, 1).copy()

    lc = np.zeros((n, len(LC_COLUMNS)), dtype=np.float32)
    lc[np.arange(n), df['LC_name'].map(LC_CODES).to_numpy()] = 1.0
    x_s = np.concatenate([lc, df[list(STATIC_NUMERIC)].to_numpy(np.float32)], axis=1)

    y = df[target_columns(out_len)].to_numpy(np.float32)
    return x_d, x_s, y

def split_train_val(df, val_ratio=0.1, seed=0):
    month = pd.to_datetime(df['Date'].astype(str)).dt.strftime('%Y%m')
    block = (df['ID'].astype(str) + '|' + month).to_numpy()
    blocks = np.unique(block)
    rng = np.random.default_rng(seed)
    rng.shuffle(blocks)
    val_blocks = blocks[:int(round(val_ratio * len(blocks)))]
    is_val = np.isin(block, val_blocks)
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)

class Normalizer:

    def __init__(self, ts_mean, ts_std, static_mean, static_std):
        self.ts_mean = np.asarray(ts_mean, dtype=np.float32)
        self.ts_std = np.asarray(ts_std, dtype=np.float32)
        self.static_mean = np.asarray(static_mean, dtype=np.float32)
        self.static_std = np.asarray(static_std, dtype=np.float32)

    @classmethod
    def fit(cls, x_d, x_s):
        ts_mean, ts_std = x_d.mean(axis=(0, 1)), x_d.std(axis=(0, 1))
        static_mean, static_std = x_s.mean(axis=0), x_s.std(axis=0)
        n_lc = len(LC_COLUMNS)
        static_mean[:n_lc], static_std[:n_lc] = 0.0, 1.0
        return cls(ts_mean, np.maximum(ts_std, 1e-6), static_mean, np.maximum(static_std, 1e-6))

    def transform(self, x_d, x_s):
        return ((x_d - self.ts_mean) / self.ts_std).astype(np.float32), \
               ((x_s - self.static_mean) / self.static_std).astype(np.float32)

    def state_dict(self):
        return {k: getattr(self, k).tolist() for k in ('ts_mean', 'ts_std', 'static_mean', 'static_std')}

    @classmethod
    def from_state_dict(cls, state):
        return cls(**state)

class SoilMoistureDataset(Dataset):
    def __init__(self, x_d, x_s, y, normalizer, prompt_type, prompt_values, in_len, out_len):
        self.x_d, self.x_s = normalizer.transform(x_d, x_s)
        self.y = y.astype(np.float32)

        keep = [i for i, name in enumerate(STATIC_FEATURES) if name not in PROMPT_EXCLUDE]
        names = [STATIC_FEATURES[i] for i in keep]
        source = x_s if prompt_values == 'raw' else self.x_s
        self.prompts = [build_prompt(prompt_type, names, row[keep], in_len, out_len) for row in source]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.x_d[idx]), torch.from_numpy(self.x_s[idx]),
                self.prompts[idx], torch.from_numpy(self.y[idx]))
