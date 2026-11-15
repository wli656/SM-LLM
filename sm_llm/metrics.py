import numpy as np

METRIC_NAMES = ('R', 'RMSE', 'bias', 'ubRMSE', 'KGE')


def compute_metrics(obs, pred):
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    r = np.corrcoef(obs, pred)[0, 1]
    rmse = np.sqrt(np.mean((pred - obs) ** 2))
    bias = np.mean(pred - obs)
    ubrmse = np.sqrt(max(rmse ** 2 - bias ** 2, 0.0))
    alpha = pred.std() / obs.std()
    beta = pred.mean() / obs.mean()
    kge = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return {'R': r, 'RMSE': rmse, 'bias': bias, 'ubRMSE': ubrmse, 'KGE': kge}
