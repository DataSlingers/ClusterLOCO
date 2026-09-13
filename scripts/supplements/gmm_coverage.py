#!/usr/bin/env python3
"""
Run feature-specific Cluster-LOCO coverage experiments and save CSV results.

For each fitted interval-producing method, evaluate coverage against all three
oracle population targets:

    Interval methods:
        1. Stochastic Cluster LOCO
        2. Hard Cluster LOCO
        3. Supervised LOCO

    Oracle targets:
        1. Stochastic Cluster LOCO target
        2. Hard Cluster LOCO target
        3. Supervised LOCO target

This produces 3 x 3 interval-target comparisons per feature and replication.

Supports two sweep modes:
    1. variance: vary a covariance multiplier while holding N fixed
    2. sample_size: vary N while holding the covariance multiplier fixed
    3. alpha: scale the configured means while holding N and variance fixed

The GMM configuration is loaded from an NPZ file containing:
    means              shape (K, p)
    base_covariances   shape (K, p, p)
    weights            shape (K,)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import sys

sys.path.append("./")

from src.non_conformity_scores import brier_score, giq_score, hinge_score
from src.conformal_generalizability import Conformal_Cluster_LOCO


ERROR_METRICS = {
    "giq": giq_score,
    "brier": brier_score,
    "hinge": hinge_score,
}


# ---------------------------------------------------------------------
# Interval-producing methods
# ---------------------------------------------------------------------

METHOD_STOCHASTIC = "Stochastic Cluster LOCO"
METHOD_CLUSTER = "Cluster LOCO"
METHOD_LOCO = "LOCO"


# ---------------------------------------------------------------------
# Oracle population targets
# ---------------------------------------------------------------------

TARGET_STOCHASTIC = "Stochastic Cluster LOCO target"
TARGET_CLUSTER = "Cluster LOCO target"
TARGET_LOCO = "LOCO target"


def sample_gmm(means, covariances, weights, n_samples, seed):
    """Sample observations and latent component labels from a finite GMM."""
    rng = np.random.default_rng(seed)

    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    weights = np.asarray(weights, dtype=float)

    k = len(weights)

    if means.shape[0] != k or covariances.shape[0] != k:
        raise ValueError("means, covariances, and weights must use the same K.")

    weights = weights / weights.sum()

    y = rng.choice(k, size=n_samples, p=weights)
    x = np.empty((n_samples, means.shape[1]), dtype=float)

    for component in range(k):
        idx = np.flatnonzero(y == component)

        if idx.size:
            x[idx] = rng.multivariate_normal(mean=means[component], cov=covariances[component], size=idx.size)
    return x, y


def as_feature_array(value, p, name):
    """Validate and return a one-dimensional result vector of length p."""
    arr = np.asarray(value)
    if arr.ndim == 0:
        arr = np.repeat(arr, p)
    arr = arr.reshape(-1)
    if arr.size != p:
        raise ValueError(f"{name} must have one value per feature. "
            f"Expected {p}, received shape "
            f"{np.asarray(value).shape}.")
    return arr


def extract_interval(result, method, p):
    """
    Extract the interval estimate, standard error, bounds, and length
    corresponding to one interval-producing method.
    """
    if method in {METHOD_STOCHASTIC, METHOD_CLUSTER}:
        estimate = as_feature_array(result["cluster_mean"], p, f"{method}.cluster_mean").astype(float)
        standard_error = as_feature_array(result["cluster_se"], p, f"{method}.cluster_se").astype(float)
        
        lower = as_feature_array(result["lower"], p, f"{method}.lower").astype(float)
        upper = as_feature_array(result["upper"], p, f"{method}.upper").astype(float)
        length = as_feature_array(result["length"], p, f"{method}.length").astype(float)
    
    elif method == METHOD_LOCO:
         estimate = as_feature_array(result["loco_mean"], p, "LOCO.loco_mean").astype(float)
         standard_error = as_feature_array(result["loco_se"], p, "LOCO.loco_se").astype(float)
         lower = as_feature_array(result["loco_low"], p, "LOCO.loco_low").astype(float)
         upper = as_feature_array(result["loco_up"], p, "LOCO.loco_up").astype(float)
         length = as_feature_array(result["loco_length"], p, "LOCO.loco_length").astype(float)
    
    else:
        raise ValueError(f"Unknown interval-producing method: {method!r}")

    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "lower": lower,
        "upper": upper,
        "length": length,
    }


def add_interval_target_rows(rows, result, interval_method, oracle_target, oracle_value, oracle_standard_error,
    sweep_mode, sweep_value, n, variance, replication, fit_seed, split_seed, test_seed, p,
    error_metric):
    """
    Append feature-level rows for one interval method evaluated against
    one oracle target.
    """
    interval = extract_interval(result=result, method=interval_method, p=p)

    oracle = as_feature_array(oracle_value, p, f"{oracle_target}.oracle_value").astype(float)
    oracle_se = as_feature_array(oracle_standard_error, p, f"{oracle_target}.oracle_standard_error").astype(float)

    lower = interval["lower"]
    upper = interval["upper"]

    covered = ((lower <= oracle) & (oracle <= upper))

    matched_target = ((
            interval_method == METHOD_STOCHASTIC
            and oracle_target == TARGET_STOCHASTIC
        ) or (
            interval_method == METHOD_CLUSTER
            and oracle_target == TARGET_CLUSTER
        ) or (
            interval_method == METHOD_LOCO
            and oracle_target == TARGET_LOCO
        ))

    for j in range(p):
        rows.append({
                "sweep_mode": sweep_mode,
                "sweep_value": float(sweep_value),
                "N": int(n),
                "variance": float(variance),
                "replication": int(replication),
                "feature": int(j),
                "error_metric": error_metric,
                "method": interval_method, # method to produce intervals (Cluster-LOCO, Stochastic Cluster LOCO or LOCO)
                "oracle_target": oracle_target, # Population target checked for coverage (Cluster-LOCO, Stochastic Cluster LOCO or LOCO)
                "matched_target": int(matched_target), # True only when an interval is checked against the population target corresponding to its own procedure.
                "covered": int(covered[j]),
                "interval_length": float(interval["length"][j]),
                "estimate": float(interval["estimate"][j]),
                "standard_error": float(interval["standard_error"][j]),
                "oracle_loco": float(oracle[j]),
                "oracle_standard_error": float(oracle_se[j]),
                "lower": float(lower[j]),
                "upper": float(upper[j]),
                "fit_seed": int(fit_seed),
                "split_seed": int(split_seed),
                "test_seed": int(test_seed),
            })


def summarize_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Create feature-specific Monte Carlo summaries."""
    group_cols = ["sweep_mode", "sweep_value", "N", "variance", "feature", "error_metric", "method", "oracle_target", "matched_target"]

    summary = raw_df.groupby(group_cols, as_index=False).agg(
                    coverage=("covered", "mean"),
                    coverage_count=("covered", "sum"),
                    replications=("covered", "size"),
                    mean_length=("interval_length", "mean"),
                    length_sd=("interval_length", "std"),
                    mean_estimate=("estimate", "mean"),
                    estimate_sd=("estimate", "std"),
                    mean_model_se=("standard_error", "mean"),
                    mean_oracle=("oracle_loco", "mean"),
                    oracle_sd=("oracle_loco", "std"),
                    mean_oracle_se=(
                        "oracle_standard_error",
                        "mean",
                    ))

    # Monte Carlo standard error for empirical coverage.
    summary["coverage_mcse"] = np.sqrt(summary["coverage"] * (1.0 - summary["coverage"]) / summary["replications"])
    summary["length_mcse"] = (summary["length_sd"] / np.sqrt(summary["replications"]))
    summary["estimate_mcse"] = (summary["estimate_sd"] / np.sqrt(summary["replications"]))
    summary["oracle_mcse"] = (summary["oracle_sd"] / np.sqrt(summary["replications"]))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("variance", "sample_size", "alpha"), required=True, help="Quantity varied by the outer simulation loop.")
    parser.add_argument("--config-npz", type=Path, required=True, help=("NPZ containing means, base_covariances, and weights."))
    parser.add_argument("--output-dir", type=Path, required=True, help=("Directory for raw_results.csv and summary_results.csv."))
    parser.add_argument("--replications", type=int, default=20)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--test-size", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--base-seed", type=int, default=234)
    parser.add_argument("--test-seed-base", type=int, default=432)
    parser.add_argument("--split-seed-base", type=int, default=634)
    parser.add_argument("--n", type=int, default=1000, help="Per-cluster N used in variance mode.")
    parser.add_argument("--variance", type=float, default=1.5, help=("Fixed covariance multiplier used in sample_size mode."))
    parser.add_argument("--variance-values", type=float, nargs="+", default=None, help=("Variance multipliers for variance mode."))
    parser.add_argument("--n-values", type=int, nargs="+", default=None, help=("Per-component sample sizes for sample_size mode."))
    parser.add_argument("--alpha-values", type=float, nargs="+", default=None, help=("Mean multipliers for alpha mode."))
    parser.add_argument(
        "--error-metric",
        choices=tuple(ERROR_METRICS),
        default="giq",
        help="Nonconformity loss used to form each LOCO difference.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.replications < 2:
        raise ValueError("--replications must be at least 2 to estimate MCSE.")
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must lie strictly between 0 and 1.")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must lie strictly between 0 and 1.")

    config = np.load(args.config_npz)
    required = {"means", "base_covariances", "weights"}
    missing = required.difference(config.files)

    if missing:
        raise KeyError(f"{args.config_npz} is missing required arrays: {sorted(missing)}")

    means = np.asarray(config["means"], dtype=float)
    base_covariances = np.asarray(config["base_covariances"], dtype=float)
    weights = np.asarray(config["weights"], dtype=float)
    k = len(weights)

    if args.mode == "variance":
        if not args.variance_values:
            raise ValueError("--variance-values is required when --mode variance.")
        sweep = [(args.n, float(variance), 1.0) for variance in args.variance_values]

    elif args.mode == "sample_size":
        if not args.n_values:
            raise ValueError("--n-values is required when --mode sample_size.")

        sweep = [(int(n), args.variance, 1.0) for n in args.n_values]

    else:
        if not args.alpha_values:
            raise ValueError("--alpha-values is required when --mode alpha.")
        sweep = [
            (args.n, args.variance, float(mean_scale))
            for mean_scale in args.alpha_values
        ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = (args.output_dir / "raw_results.csv")
    summary_path = (args.output_dir / "summary_results.csv")
    rows: list[dict[str, Any]] = []

    for n, variance, mean_scale in sweep:
        if args.mode == "variance":
            sweep_value = variance
            description = f"variance={variance:g}"
        elif args.mode == "sample_size":
            sweep_value = n
            description = f"N={n}"
        else:
            sweep_value = mean_scale
            description = f"alpha={mean_scale:g}"

        for replication in tqdm(range(args.replications), desc=description):
            fit_seed = args.base_seed + replication
            test_seed = args.test_seed_base + replication
            split_seed = args.split_seed_base + replication
            covariances = base_covariances * variance
            
            # Total generated size is K * N.
            scaled_means = mean_scale * means
            x, y = sample_gmm(means=scaled_means, covariances=covariances, weights=weights, n_samples=k * n, seed=fit_seed)
            x_tr, x_ca, y_tr, y_ca = train_test_split(x, y, test_size=args.test_size, random_state=split_seed, stratify=y)
            x_test, y_test = sample_gmm(means=scaled_means, covariances=covariances, weights=weights, n_samples=args.n_test, seed=test_seed)
            p = x_tr.shape[1]

            model_kwargs = {
                "X_tr": x_tr,
                "X_ca": x_ca,
                "model": GaussianMixture(n_components=k, random_state=fit_seed),
                "seed": fit_seed,
                "error_metric": ERROR_METRICS[args.error_metric],
                "n_jobs": args.n_jobs,
            }

            validation_kwargs = {
                "X_test": x_test,
                "y_test": y_test,
                "X_al": x_tr,
                "y_al": y_tr,
                "X_ca": x_ca,
                "y_ca": y_ca,
                "alpha": args.alpha,
            }

            # -------------------------------------------------
            # Stochastic Cluster-LOCO
            # -------------------------------------------------
            stochastic_model = Conformal_Cluster_LOCO(**model_kwargs)
            stochastic_result = stochastic_model.validate(**validation_kwargs, stochastic=True, method="stochastic")

            # -------------------------------------------------
            # Supervised LOCO oracle
            # -------------------------------------------------
            # fit_loco() was already called by validate().
            oracle_loco, oracle_loco_se = stochastic_model.predict_oracle(X_test=x_test, y_test=y_test, X_al=x_tr, y_al=y_tr, method="LOCO")

            # -------------------------------------------------
            # Cluster-LOCO
            # -------------------------------------------------
            cluster_model = Conformal_Cluster_LOCO(**model_kwargs)
            cluster_result = cluster_model.validate(**validation_kwargs, stochastic=False, method="Cluster-LOCO")

            # -------------------------------------------------
            # Collect targets
            # -------------------------------------------------
            oracle_targets = {
                TARGET_STOCHASTIC: {
                    "value": stochastic_result["oracle_loco"],
                    "standard_error": stochastic_result["oracle_se"],
                },
                TARGET_CLUSTER: {
                    "value": cluster_result["oracle_loco"],
                    "standard_error": cluster_result["oracle_se"],
                },
                TARGET_LOCO: {
                    "value": oracle_loco,
                    "standard_error": oracle_loco_se,
                },
            }

            interval_results = {
                METHOD_STOCHASTIC: stochastic_result,
                METHOD_CLUSTER: cluster_result,
                METHOD_LOCO: stochastic_result,
            }

            for interval_method, interval_result in interval_results.items():
                for oracle_target, oracle_result in oracle_targets.items():
                    add_interval_target_rows(
                        rows=rows, result=interval_result, interval_method=interval_method,
                        oracle_target=oracle_target, oracle_value=oracle_result["value"],
                        oracle_standard_error=oracle_result["standard_error"],
                        sweep_mode=args.mode, sweep_value=sweep_value,
                        n=n, variance=variance, replication=replication, p=p,
                        fit_seed=fit_seed, split_seed=split_seed, test_seed=test_seed,
                        error_metric=args.error_metric)
        # Checkpoint after each sweep value.
        raw_df = pd.DataFrame(rows)
        raw_df.to_csv(raw_path, index=False)
        summary_df = summarize_results(raw_df)
        summary_df.to_csv(summary_path, index=False)

    print(f"Saved raw replication results to {raw_path}")
    print(f"Saved feature-specific summaries to {summary_path}")
    


if __name__ == "__main__":
    main()
