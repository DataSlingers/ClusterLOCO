import numpy as np


def hinge_score(x, y, pi, margin=0.0):
    """
    Multiclass hinge-style conformity score / error.

    s((x,y); pi) = max(0, max_{k != y} pi_k - pi_y + margin)
    Smaller is better:
    - score = 0 if pi_y is at least margin larger than every other class prob.
    - score is positive if some other class has probability close to or larger than pi_y.

    Parameters
    ----------
    x : array-like
        Feature vector. Included for API consistency. Not used directly if pi is already evaluated at x.
    y : int
        Label index.
    pi : array-like or callable
        Either a probability vector pi(x), shape (K,),
        or a callable such that pi(x) returns shape (K,).
    margin : float
        Required probability margin. Use margin=0.0 for probability ranking loss.
    Returns
    -------
    float
        Hinge-style score.
    """
    probs = pi(x) if callable(pi) else pi
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 1:
        raise ValueError("pi(x) must be a one-dimensional probability vector.")
    K = probs.shape[0]
    if not (0 <= y < K):
        raise ValueError(f"y must be an integer in {{0, ..., {K - 1}}}.")

    p_y = probs[y]
    other_probs = np.delete(probs, y)
    max_other = np.max(other_probs)
    score = max(0.0, max_other - p_y + margin)
    return float(score)


def giq_score(x, y, pi):
    """
    Generalized inverse quantile score.

    s((x,y); pi) = sum_{k=0}^{r(y)-1} pi(x)_{(k)}

    where pi(x)_{(0)} = 0 and pi(x)_{(1)} >= ... >= pi(x)_{(K)}
    are the sorted class probabilities, and r(y) is the rank of the
    y-th entry of pi(x).
    """
    probs = pi(x) if callable(pi) else pi
    probs = np.asarray(probs, dtype=float)

    if probs.ndim != 1:
        raise ValueError("pi(x) must be a one-dimensional probability vector.")
    K = probs.shape[0]
    if not (0 <= y < K):
        raise ValueError(f"y must be an integer in {{0, ..., {K - 1}}}.")

    p_y = probs[y]
    rank_y = 1 + np.sum(probs > p_y) # Rank under descending probabilities
    sorted_probs = np.sort(probs)[::-1]
    score = np.sum(sorted_probs[:rank_y - 1]) # Sum pi_(1), ..., pi_(r(y)-1)
    return float(score)


def brier_score(x, y, pi):
    """Multiclass Brier loss.

    Computes ``sum_k (pi_k(x) - 1{y = k})^2``.  Unlike the rank-based GIQ
    score, this remains sensitive to probability calibration when the correct
    class is already ranked first.
    """
    probs = pi(x) if callable(pi) else pi
    probs = np.asarray(probs, dtype=float)

    if probs.ndim != 1:
        raise ValueError("pi(x) must be a one-dimensional probability vector.")
    K = probs.shape[0]
    if not (0 <= y < K):
        raise ValueError(f"y must be an integer in {{0, ..., {K - 1}}}.")
    if np.any(~np.isfinite(probs)):
        raise ValueError("pi(x) must contain only finite probabilities.")

    truth = np.zeros(K, dtype=float)
    truth[y] = 1.0
    return float(np.sum((probs - truth) ** 2))
