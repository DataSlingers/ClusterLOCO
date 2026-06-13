"""
RAMPART wrappers for efficient computation of LOCO and SHAP feature importance scores

Author: Claire He
RAMPART code is based on description and pseudo code in Chen (TMLR 2025)

1. RAMPART algorithm 
2. Cluster LOCO RAMPART wrapper (global and local scores)
"""
from collections.abc import Mapping
import numpy as np
from clim.minipatches.generalizability import ClusterLOCOMP
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.multiclass import type_of_target


def _extract_phi_std(res, M: int | None = None):
    """
    RAMPART runs on the old version of Cluster LOCO (generalizability file) and Shapley MP
    
    Accepts either:
      - array-like phi
      - dict with keys {"phi", optional "std"}
      - dict with keys {"delta", optional "delta_se", optional "features"}  (ClusterLOCOMP.score output)
    Returns:
      phi: (m,) float
      std: (m,) float or None
      extras: dict (passes through z_ref/z_i if present)
    """
    # Plain array -> phi
    if not isinstance(res, Mapping):
        phi = np.asarray(res, dtype=float).reshape(-1)
        return phi, None, {}

    # Pass through some extras if present
    extras = {k: res[k] for k in ("z_ref", "z_i") if k in res}

    # Old-style output
    if "phi" in res:
        phi = np.asarray(res["phi"], dtype=float).reshape(-1)
        std = np.asarray(res["std"], dtype=float).reshape(-1) if "std" in res and res["std"] is not None else None
        return phi, std, extras

    # New-style output from ClusterLOCOMP.score
    if "delta" in res:
        delta = np.asarray(res["delta"], dtype=float).reshape(-1)
        delta_se = np.asarray(res["delta_se"], dtype=float).reshape(-1) if "delta_se" in res and res["delta_se"] is not None else None

        # If features are provided, map sparse outputs into dense vectors of length M
        if "features" in res and M is not None:
            feats = np.asarray(res["features"], dtype=int).reshape(-1)
            phi = np.full(M, np.nan, dtype=float)
            std = np.full(M, np.nan, dtype=float) if delta_se is not None else None

            phi[feats] = delta
            if std is not None:
                std[feats] = delta_se
            return phi, std, extras

        # Otherwise treat delta as already aligned
        return delta, delta_se, extras

    # Fallback: try common names
    if "score" in res:
        phi = np.asarray(res["score"], dtype=float).reshape(-1)
        return phi, None, extras

    raise KeyError("Unrecognized generalizability_fn output format (expected phi/std or delta/delta_se).")
    
##### Main RAMPART wrapper ######

def RAMPART(
    X,
    generalizability_fn,
    B,
    ranking_fn,
    B_schedule=True,
    alpha_N=0.25,
    alpha_M=0.1,
    top_k=None,
    rampart_verbose=True,
    save=False,
    gen_kwargs=None,
):
    """
    RAMPART wrapper for a generalizability function and ranking function.

    Important:
    ----------
    `index_set` stores the current global feature indices.
    At each round, ClusterLOCOMP sees only X_sub = X[:, index_set],
    so its scores are local to X_sub. RAMPART maps local rankings back
    to global feature indices using index_set.

    The optional `feature_list=index_set.copy()` is passed only for
    bookkeeping/debugging in the generalizability function.
    """
    if gen_kwargs is None:
        gen_kwargs = {}

    X = np.asarray(X)
    N, M = X.shape

    if top_k is None:
        top_k = 1

    if not (1 <= top_k <= M):
        raise ValueError(f"top_k must be in [1, {M}]")

    T = max(1, int(np.floor(np.log2(M)) - np.ceil(np.log2(top_k)) + 1))

    index_set = np.arange(M, dtype=int)
    history = []

    for t in range(1, T + 1):
        B_t = int(B * (0.5 + 0.5 * t / T)) if B_schedule else B

        # Current reduced feature matrix.
        X_sub = X[:, index_set]

        # Pass current global feature ids for bookkeeping/debugging.
        res_t, _ = generalizability_fn(
            X_sub,
            B=B_t,
            alpha_N=alpha_N,
            alpha_M=alpha_M,
            feature_list=index_set.copy(),
            **gen_kwargs,
        )

        # phi_t is local to X_sub, length len(index_set).
        phi_t, std_t, _ = _extract_phi_std(res_t, M=len(index_set))

        # Local ranking among current features.
        tau_local, _ = ranking_fn(phi_t)

        # Map local ranking back to global feature ids.
        tau_global = index_set[tau_local]

        half = max(1, int(np.ceil(len(index_set) / 2)))
        next_set = tau_global[:half]

        if save:
            history.append(
                {
                    "round": t,
                    "B_t": B_t,
                    "index_set_in": index_set.copy(),
                    "phi": phi_t.copy(),
                    "std": std_t.copy() if std_t is not None else None,
                    "tau_local": tau_local.copy(),
                    "tau_global": tau_global.copy(),
                    "kept_indices": next_set.copy(),
                }
            )

        if rampart_verbose:
            best_global = tau_global[0]
            best_score = phi_t[tau_local[0]]
            print(
                f"[RAMPART] round {t}/{T}: "
                f"|C_t|={len(index_set)} → |C_{t+1}|={len(next_set)} "
                f"(best feat={best_global}, score={best_score:.4g})"
            )

        index_set = next_set

        if len(index_set) <= top_k:
            break

    # Final evaluation on surviving set.
    X_final = X[:, index_set]

    res_final, g = generalizability_fn(
        X_final,
        B=B,
        alpha_N=alpha_N,
        alpha_M=alpha_M,
        feature_list=index_set.copy(),
        **gen_kwargs,
    )

    phi_final, std_final, extras = _extract_phi_std(res_final, M=len(index_set))

    tau_local_final, _ = ranking_fn(phi_final)
    tau_global_final = index_set[tau_local_final]

    k = min(top_k, len(tau_global_final))

    selected_indices = tau_global_final[:k]
    selected_scores = phi_final[tau_local_final[:k]]
    selected_std = (
        std_final[tau_local_final[:k]]
        if std_final is not None
        else None
    )

    result = {
        "selected_indices": selected_indices,
        "selected_scores": selected_scores,
        "selected_std": selected_std,
        "tau_last": tau_global_final,
        "phi_last": phi_final,
        "std_last": std_final,
        "history": history,
        "final_model": g,
        "final_index_set": index_set.copy(),
        "tau_local_final": tau_local_final.copy(),
    }

    result.update(extras)
    return result


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


##### RAMPART pipe to Cluster LOCO  (global AND local) ######

def ClusterLOCO_RAMPART(
    *,
    K: int,
    error_metric=None,
    proba_error: bool = True,
    base_clusterer=None,
    base_classifier=None,
    random_state: int = 0,
    parallel_MP: bool = True,
    parallel: dict | None = None,
    alpha_N=0.1,
    alpha_M=0.4,
    standardize: bool = False,
    reference: str = "full",
    agg: str = "mean",
):
    """
    Return a generalizability function compatible with RAMPART.

    This wrapper is adapted to the refactored ClusterLOCOMP.

    Scoring modes
    -------------
    error_metric=None:
        Uses ARI mode inside ClusterLOCOMP.score().
        This uses hard-label LOO / LOCO-LOO predictions.

    error_metric is not None and proba_error=True:
        Uses probability-based pointwise errors, e.g.
            hinge_error(z, P)
            margin_error(z, P)
            brier_error(z, P)

    error_metric is not None and proba_error=False:
        Uses hard-label errors, e.g.
            hamming_distance(z, z_hat)

    Notes
    -----
    RAMPART repeatedly fits ClusterLOCOMP on feature subsets X_sub.
    Therefore returned feature scores are local to X_sub's columns.
    The outer RAMPART function maps them back to global indices.
    """

    def generalizability_fn(
        X_sub,
        *,
        B,
        alpha_N=alpha_N,
        alpha_M=alpha_M,
        **_ignored,
    ):
        X_sub = np.asarray(X_sub)

        model = ClusterLOCOMP(
            K=K,
            B=B,
            base_clusterer=base_clusterer,
            base_classifier=base_classifier,
            random_state=random_state,
        )

        model.fit(
            X_sub,
            y=None,
            alpha_N=alpha_N,
            alpha_M=alpha_M,
            parallel_MP=parallel_MP,
            parallel=parallel,
            standardize=standardize,
            reference=reference,
        )

        # Refactored ClusterLOCOMP uses z_ref as the training/reference labels.
        z = model.z_ref

        sc = model.score(
            error_metric=error_metric,
            z=z,
            agg=agg,
            features=None,
            proba_error=proba_error,
            parallel_features=False,
            par={
                "n_jobs": 1,
                "backend": "loky",
                "prefer": "processes",
                "verbose": 0,
            },
        )

        # Return in RAMPART's old expected shape: phi/std aligned to X_sub columns.
        M_sub = X_sub.shape[1]

        phi = np.full(M_sub, np.nan, dtype=float)
        std = (
            np.full(M_sub, np.nan, dtype=float)
            if sc.get("delta_se", None) is not None
            else None
        )

        feats = np.asarray(sc["features"], dtype=int)
        phi[feats] = np.asarray(sc["delta"], dtype=float)

        if std is not None:
            std[feats] = np.asarray(sc["delta_se"], dtype=float)

        out = {
            "phi": phi,
            "std": std,
            "z_ref": model.z_ref,
            "score_raw": sc,
        }

        return out, model

    return generalizability_fn