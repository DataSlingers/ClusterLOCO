""" Created by : Claire He 
    12.04.24

    Generate minipatches 
functions: 
    - get_minipatch
    - visualise_minipatch
              
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import seaborn as sns
import math

palette = sns.color_palette([
    "#7fbf7b",  # Light Green
    "#af8dc3",  # Lavender
    "#e7d4e8",  # Light Purple
    "#fdc086",  # Light Orange
    "#ff9896",  # Light Red
    "#c5b0d5"   # Light Blue
])


def _resolve_patch_param(alpha, patch_size, total, alpha_name, size_name):
    """
    Resolve a patch parameter from either:
      - alpha in (0, 1] as a fraction of total
      - patch_size in [1, total] as an explicit size

    Explicit patch_size takes precedence if provided.
    """
    if patch_size is not None:
        if not isinstance(patch_size, int):
            raise TypeError(f"{size_name} must be an int, got {type(patch_size).__name__}")
        if not (1 <= patch_size <= total):
            raise ValueError(f"{size_name} must be in [1, {total}], got {patch_size}")
        return patch_size

    if alpha is None:
        raise ValueError(f"Must provide either {alpha_name} or {size_name}")

    if not isinstance(alpha, (int, float)):
        raise TypeError(f"{alpha_name} must be numeric, got {type(alpha).__name__}")

    alpha = float(alpha)
    if not (0 < alpha <= 1):
        raise ValueError(f"{alpha_name} must be in (0, 1], got {alpha}")

    return max(1, int(math.ceil(alpha * total)))

        
def iter_minipatches(N: int, M: int, B: int, n: int, m: int, rng=None, sort_indices: bool = False,):
    """
    Yield minipatches one at a time as (b, I_t, F_t).

    I_t: observation indices
    F_t: feature indices
    """
    rng = np.random.RandomState() if rng is None else rng
    q = max(1, n)
    r = max(1, m)
    choice = rng.choice
    for b in range(B):
        I_t = choice(N, size=q, replace=False)
        F_t = choice(M, size=r, replace=False)
        if sort_indices:
            I_t.sort()
            F_t.sort()

        yield b, I_t.astype(np.int32, copy=False), F_t.astype(np.int32, copy=False)


def get_minipatch(X_arr, y_arr, ratio_x, ratio_y, seed=None):
    """ Generate a minipatch from a dataset with covariates X, with obs size controled by ratio parameters
    Input: 
        X_arr: original feature dataset
        y_arr: target covariate/cluster labels. If y_arr is None, skip.
        ratio_y: features' ratio
        ratio_x: observations' ratio
    -------
    Outputs: 
        x_mp
        y_mp
        idx_I
        idx_F """
    N = X_arr.shape[0]
    M = X_arr.shape[1]
    m = int(ratio_y*M)
    n = int(ratio_x*N)
    assert n*N > m # verify that enough observations are sampled
    if isinstance(seed, np.random.RandomState):
        r = seed
    elif seed is None:
        r = np.random.RandomState()
    else:
        r = np.random.RandomState(seed)
    ## index of minipatch
    idx_I = r.choice(N, size=n, replace=False) # uniform sampling of subset of observations
    idx_F = r.choice(M, size=m, replace=False) # uniform sampling of subset of features
    ## record which obs/features are subsampled 
    x_mp = X_arr[np.ix_(idx_I, idx_F)]
    if y_arr is None:
        return x_mp, None, idx_I, idx_F
    else:
        y_mp = y_arr[np.ix_(idx_I)]
        return x_mp, y_mp, idx_I, idx_F


def fast_minipatches(N, M, B, alpha_N, alpha_M, rng=None, print_patch_size=True):
    """Return list of (I_t, F_t) index arrays (obs, feats). Does not pass X or y. 
    """
    m = max(1, int(np.floor(alpha_N * N)))
    r = max(1, int(np.floor(alpha_M * M)))
    if print_patch_size:
        print(f'feature size: {r}, obs size: {m}')
    patches = []
    for _ in range(B):
        I_t = rng.choice(N, size=m, replace=False)
        F_t = rng.choice(M, size=r, replace=False)
        patches.append((I_t, F_t))
    return patches

    
def adaptive_minipatch(X, p_item, p_feature, weights_item = None, weights_feature = None, pi_item = 1.0, pi_feature = 1.0, qI = 0.95, qF = 0.95):
    """
    EE+Prob style sampling. If pi_* < 1, split draw between "upper" set (exploitation)
    and complement (exploration). Upper = items with weights >= percentile (qI or qF).
    If pi_* == 1, draw purely by probability weights (or uniformly if None).

    X: (n_features, n_samples)
    """
    n_feat, n_samp = X.shape
    n_col = max(1, int(np.floor(n_samp * p_item)))
    n_row = max(1, int(np.floor(n_feat * p_feature)))

    # ---- columns (samples / observations)
    if weights_item is None:
        weights_item = np.ones(n_samp) / n_samp
    weights_item = np.array(weights_item, dtype=float)
    weights_item = np.maximum(weights_item, 0)
    if weights_item.sum() == 0:
        weights_item[:] = 1.0
    weights_item /= weights_item.sum()

    if pi_item < 1.0:
        thr = np.quantile(weights_item, qI)
        upper = np.flatnonzero(weights_item >= thr)
        # exploitation draw
        n1 = int(np.ceil(min(n_col, pi_item * len(upper))))
        n1 = min(n1, len(upper))
        if n1 > 0:
            p_u = weights_item[upper]
            p_u = p_u / p_u.sum()
            cols1 = np.random.choice(upper, size=n1, replace=False, p=p_u if len(upper) > 1 else None)
        else:
            cols1 = np.array([], dtype=int)
        # exploration draw
        remain = np.setdiff1d(np.arange(n_samp), cols1, assume_unique=False)
        n2 = n_col - len(cols1)
        if n2 > 0:
            cols2 = np.random.choice(remain, size=n2, replace=False)
        else:
            cols2 = np.array([], dtype=int)
        col_idx = np.sort(np.concatenate([cols1, cols2]))
    else:
        col_idx = np.random.choice(np.arange(n_samp), size=n_col, replace=False, p=weights_item)

    # ---- rows (features)
    if weights_feature is None:
        weights_feature = np.ones(n_feat) / n_feat
    weights_feature = np.array(weights_feature, dtype=float)
    weights_feature = np.maximum(weights_feature, 0)
    if weights_feature.sum() == 0:
        weights_feature[:] = 1.0
    weights_feature /= weights_feature.sum()

    if pi_feature < 1.0:
        if 0 < qF < 1:
            thr = np.quantile(weights_feature, qF)
        else:
            thr = weights_feature.mean() + qF * weights_feature.std(ddof=0)
        upper = np.flatnonzero(weights_feature >= thr)
        n1 = int(np.ceil(min(n_row, pi_feature * len(upper))))
        n1 = min(n1, len(upper))
        if n1 > 0:
            p_u = weights_feature[upper]
            p_u = p_u / p_u.sum()
            rows1 = np.random.choice(upper, size=n1, replace=False, p=p_u if len(upper) > 1 else None)
        else:
            rows1 = np.array([], dtype=int)
        remain = np.setdiff1d(np.arange(n_feat), rows1, assume_unique=False)
        n2 = n_row - len(rows1)
        if n2 > 0:
            rows2 = np.random.choice(remain, size=n2, replace=False)
        else:
            rows2 = np.array([], dtype=int)
        row_idx = np.sort(np.concatenate([rows1, rows2]))
    else:
        row_idx = np.random.choice(np.arange(n_feat), size=n_row, replace=False, p=weights_feature)

    submat = X[np.ix_(row_idx, col_idx)]
    return dict(submat=submat, row_idx=row_idx, col_idx=col_idx)


def visualize_minipatch(in_mp_obs, in_mp_feature, color_palette = palette, type='sorted'):
    
    B = in_mp_obs.shape[0]
    matrix = np.zeros((in_mp_obs.shape[1],in_mp_feature.shape[1]))
    for i in range(B):
        matrix += (in_mp_obs[i][:, np.newaxis] & in_mp_feature[i]).astype(int)
    df = pd.DataFrame(matrix)
    if type =='sorted':
        sns.heatmap(df[df.mean().sort_values().index].sort_values(by=df[df.mean().sort_values().index].columns[-1], axis=0), cmap=palette)
    else:
        sns.heatmap(df, cmap=palette)
    plt.title('Patch selection frequency')