# Selected original analysis excerpt from build_track_filtered_enformer_comparison.py.
# Imports and constants included for context; no source patient data is included.
import numpy as np
import pandas as pd

N_PERMUTATIONS = 1000
RANDOM_SEED = 42

def permute(df: pd.DataFrame, metric: str) -> tuple[dict, pd.DataFrame]:
    x = df[metric].to_numpy(dtype=float)
    y = df['expression_mean'].to_numpy(dtype=float)
    observed = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for i in range(N_PERMUTATIONS):
        r = float(np.corrcoef(x, rng.permutation(y))[0, 1])
        rows.append({'iteration': i + 1, 'metric': metric, 'pearson_r': r})
    perm = pd.DataFrame(rows)
    values = perm['pearson_r'].to_numpy(dtype=float)
    return {
        'metric': metric,
        'observed_pearson_r': observed,
        'n_permutations': N_PERMUTATIONS,
        'null_mean': float(values.mean()),
        'null_std': float(values.std(ddof=1)),
        'null_q025': float(np.quantile(values, 0.025)),
        'null_q975': float(np.quantile(values, 0.975)),
        'two_sided_empirical_p_value': float((np.sum(np.abs(values) >= abs(observed)) + 1) / (len(values) + 1)),
        'observed_percentile_in_null': float((values <= observed).mean() * 100.0),
    }, perm
