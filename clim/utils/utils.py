import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
import seaborn as sns
from scipy.sparse import csr_array

def transform_scores_to_ranking(scores: np.ndarray):
    """
    Given a 1D array of scores (higher = better), produce:
      - tau: indices sorted by descending score (permutation)
      - ranks: 1-based ranks aligned to original indices (ties broken by stable order)
    """
    scores = np.asarray(scores).astype(float)
    # Sort descending; stable to preserve input order on ties
    tau = np.argsort(-scores, kind='mergesort')
    ranks = np.empty_like(tau)
    ranks[tau] = np.arange(0, len(scores))
    return tau, ranks

def _set_if_exists(estimator, **kwargs):
    """
    Checks if parameters exists for the estimator, if they do, reset arguments from passed kwargs
    """
    params = estimator.get_params()
    valid = {k: v for k, v in kwargs.items() if k in params}
    if valid:
        estimator.set_params(**valid)
        
def _resolve_patch_param(alpha, patch_size, total, alpha_name, size_name):
    """
    Resolves patch_size issues, prioritize explicit minipatch size patch_size vs minipatch ratio alpha
    """
    if patch_size is not None:
        if not isinstance(patch_size, int):
            raise TypeError(f"{size_name} must be int, got {type(patch_size).__name__}")
        if not (1 <= patch_size <= total):
            raise ValueError(f"{size_name} must be in [1, {total}], got {patch_size}")
        return patch_size

    if alpha is None:
        raise ValueError(f"Must provide either {alpha_name} or {size_name}")

    alpha = float(alpha)
    if not (0 < alpha <= 1):
        raise ValueError(f"{alpha_name} must be in (0,1], got {alpha}")

    return max(1, int(np.ceil(alpha * total)))


def _apply_label_mapping(z_local, mapping):
    """
    Apply local-label -> global-label mapping.
    """
    z_local = np.asarray(z_local, dtype=np.int32)
    mapping = np.asarray(mapping, dtype=np.int32)
    z_aligned = np.full_like(z_local, -1, dtype=np.int32)
    ok = ((z_local >= 0) & (z_local < mapping.size) & (mapping[z_local] >= 0))
    z_aligned[ok] = mapping[z_local[ok]]
    return z_aligned
    
def _label_mapping_from_overlap(K, z_ref_overlap, z_local_overlap):
    """
    Compute local-label -> reference-label mapping using a Hungarian
    assignment on the overlap labels.

    Parameters
    ----------
    z_ref_overlap : ndarray, shape (n_overlap,)
        Reference/global labels on overlapping observations.
    z_local_overlap : ndarray, shape (n_overlap,)
        Local minipatch labels on the same observations.

    Returns
    -------
    mapping : ndarray, shape (K,)
        mapping[local_label] = reference_label.
    """
    K = int(K)
    z_ref_overlap = np.asarray(z_ref_overlap, dtype=np.int32)
    z_local_overlap = np.asarray(z_local_overlap, dtype=np.int32)
    C = np.zeros((K, K), dtype=np.int32)
    ok = ((z_ref_overlap >= 0) & (z_ref_overlap < K) & (z_local_overlap >= 0) & (z_local_overlap < K))
    if not np.any(ok):
        return np.arange(K, dtype=np.int32)

    # C[global_label, local_label]
    np.add.at(C, (z_ref_overlap[ok], z_local_overlap[ok]), 1)
    row_ind, col_ind = linear_sum_assignment(-C)
    # Default identity protects against unmatched labels.
    mapping = np.arange(K, dtype=np.int32)
    mapping[col_ind] = row_ind
    return mapping

    
def standardize_cols(X: np.ndarray) -> np.ndarray:
    """Column-wise standardization (each feature -> mean 0, std 1)."""
    X = X.astype(float, copy=False)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0] = 1e-3
    return (X - mu) / sd


def match_labels(y_true, y_pred,return_map=False):
    """
    Hungarian alignment algorithm between true and prediction.
    This version is slower than label_alignment but supports generalized linear sum assignment ie even when label sets might differ. 
    """
    if len(y_true)==len(y_pred):
        D = confusion_matrix(y_true, y_pred)
        row_ind, col_ind = linear_sum_assignment(D.max() - D)
        mapping = dict(zip(col_ind, row_ind))
        if len(row_ind) == len(col_ind):
            y_pred_aligned = np.array([mapping[label] for label in y_pred])
            if return_map:
                return mapping, y_pred_aligned
            else:
                return y_pred_aligned
    else: # generalized linear sum assignment
        if return_map:
            return align_labels_confusion(y_true, y_pred)
        else:
            y_pred_aligned, _ = align_labels_confusion(y_true, y_pred)
            return y_pred_aligned
            
def label_alignment(t, s, K):
    """ Alternative (faster when N big) to match_labels that returns the permutation. To get y_pred, use aligned_s = col_ind[s] """
    if len(t) != len(s):
        raise ValueError("Labels must have same length")
    n = len(t)
    all_ones, arange_n = np.ones(n), np.arange(n)
    # sparse matrices 
    t_matrix, s_matrix = csr_array((all_ones, (arange_n, t)), shape=(n, K)), csr_array((all_ones, (arange_n, s)), shape=(n, K))
    # solve linear assignment
    Q = (-s_matrix.T @t_matrix).toarray()
    _, col_ind = linear_sum_assignment(Q)
    return col_ind

def hungarian_align(t, s, K=None):
    """
    Align predicted labels s to target labels t using Hungarian assignment.
    Returns aligned labels (same shape as s).
    """
    t = np.asarray(t, dtype=int)
    s = np.asarray(s, dtype=int)

    if K is None:
        # safest: include all labels that appear in either vector
        K = int(max(t.max(initial=0), s.max(initial=0)) + 1)

    perm = label_alignment(t, s, K)
    return perm[s]

def proba_alignment(proba, classes, col_ind):
    proba_aligned = np.zeros_like(proba)
    for i, pred_label in enumerate(classes):
        target_col = col_ind[pred_label]
        proba_aligned[:, target_col] = proba[:, i]
    return proba_aligned


def align_labels_confusion(y_true, y_pred, return_map=False):
    """
    Align y_pred labels to y_true labels using Hungarian matching on the
    confusion matrix. Works even when the label sets differ.

    Parameters
    ----------
    y_true : array-like
        Reference labels.
    y_pred : array-like
        Predicted labels to align.
    return_map : bool, default=False
        If True, also return the mapping {pred_label -> true_label}.

    Returns
    -------
    y_pred_aligned : np.ndarray
        y_pred relabeled to best match y_true.
    mapping : dict, optional
        Returned only if return_map=True.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)

    # confusion matrix with explicit label order
    C = confusion_matrix(y_true, y_pred, labels=true_labels)

    # If label sets differ in size, pad to square for Hungarian assignment
    n_true, n_pred = C.shape
    n = max(n_true, n_pred)
    C_pad = np.zeros((n, n), dtype=C.dtype)
    C_pad[:n_true, :n_pred] = C

    row_ind, col_ind = linear_sum_assignment(C_pad.max() - C_pad)

    mapping = {}
    for r, c in zip(row_ind, col_ind):
        if r < n_true and c < n_pred:
            mapping[pred_labels[c]] = true_labels[r]

    # Keep unmatched labels as themselves
    y_pred_aligned = np.array([mapping.get(lbl, lbl) for lbl in y_pred])

    if return_map:
        return y_pred_aligned, mapping
    return y_pred_aligned
    
