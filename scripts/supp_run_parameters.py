"""
Parameter sweep experiments: various p setting (p_features = 10, p_noise = 10, 40) 

Gets run time + performance via top k overlap in GMM setting 
Keep track of ARI of base simulation and H-D simulation for assessing task's difficulty.
Varying N, p, alpha_N, alpha_M, B

Author: Claire He
"""

import numpy as np
import time
import os
import json
import argparse
from copy import deepcopy
from pathlib import Path
from joblib import Parallel, delayed
from itertools import product

from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler

from clim import ClusterLOCOMP, ClusterLOCO_RAMPART, GlobalStability_MP, RAMPART
from clim.data_splitting import Cluster_LOCO_Split
from clim.utils import hinge_error, transform_scores_to_ranking
from clim.models import BaseSpectralClustering, GammaMixture
import sys
sys.path.append('./')
from benchmarking import LRP_score, PBFI, c_SHAP, feature_imp_cluster
from simulations import *


N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))


def topk_overlap(scores, true_idx, k, signed=False):
    s = np.asarray(scores, dtype=float).reshape(-1)
    topk = np.argsort(s)[::-1][:k]
    hits = np.intersect1d(topk, true_idx, assume_unique=False).size # intersection selection and top k
    return hits/k

def generate_dataset_for_one_run(*, sim_method: str, sim_seed: int, embed_seed: int, K: int,
    n_per_cluster: int, alpha: float, d0: int, gaps, shape_probs, oversample: float, informative_d: int, noise_plan: list,):
    """
    noise_plan: list of dicts like:
      [{"type":"gaussian","d":10},
       {"type":"student-t","d":10,"df":10},
       {"type":"triangular","d":10,"low":-1,"high":1},
       {"type":"laplace","d":10,"scale":1.0}]
    """
    # Adds correlation in signal
    Cov_k = [GenerateCovariances(dim=d0, covMethod='onion', eta=1/(k+1)).covGen()[0] for k in range(K)]

    sim = BaseSimulator(K=K, n_samples_per_cluster=n_per_cluster, alpha=alpha, d_0=d0, Cov_k=Cov_k,
        method=sim_method, gaps=gaps, shape_probs=shape_probs, oversample=oversample, random_state=sim_seed)
    X, y = sim.generate_data()

    # project to informative dims with embed_seed
    rng_saved = sim.rng
    sim.rng = np.random.default_rng(embed_seed)
    if informative_d - d0 > 0:
        sim.project_higher_dim(embed_dim=informative_d-d0, method='orthogonal', gamma=1, degree=5, label_aware=True)
    sim.rng = rng_saved

    # sequentially append noise blocks
    for block in noise_plan:
        nt = block["type"]
        d = int(block["d"])
        kw = {k: v for k, v in block.items() if k not in ("type", "d")}
        if nt=='permuted':
            random_features = np.random.choice(np.arange(sim.X.shape[1]), size=d)
            X_noise = np.array([permute_feature(sim.X, rd_feat) for rd_feat in random_features]).T
            sim.X = np.concatenate([sim.X, X_noise], axis=1)
        elif nt =='uniform':
            sim.add_noise(noise_d=d, noise_type=nt, low=np.min(sim.X), high=np.max(sim.X))
        else:
            sim.add_noise(noise_d=d, noise_type=nt, **kw)
    X_aug = sim.X

    if sim_method == 'moon-donut':
        X_aug = np.concatenate([X, X_aug], axis=1) 
    elif sim_method == 'swiss-roll':
        X_aug = np.concatenate([X, X_aug], axis=1) 
    return X_aug, y
    

def run_one_simulation(*, X_aug, y, K: int, noise_d: int, informative_d: int = 10, base_clusterer=None,
        B: int = 100, standardize: bool = True, topk: int | None = None, alpha_N=0.2, alpha_M=0.2):
    """
    Returns dict with score vectors for benchmark and our methods
    """
    p = X_aug.shape[1]
    expected_p = informative_d + noise_d
    if p != expected_p:
        raise ValueError(f"Expected p={expected_p} features, got {p}")

    if base_clusterer is None:
        base_clusterer = BaseSpectralClustering(n_clusters=K)

    true_idx = np.arange(informative_d)
    if topk is None:
        topk = informative_d  

    topk_hits = {} 
    times = {} 
    ari = {}
        
    # ---- Score 4: Cluster LOCO hinge_error ----
    print("======== Split Cluster LOCO ========")
    X_tr, X_va, y_tr, y_va = train_test_split(X_aug, y, test_size=0.5, stratify=y)
    t0 = time.perf_counter()
    split_cloc, _ = Cluster_LOCO_Split(X_tr, X_va, model=base_clusterer, clf = RandomForestClassifier(), K=K, error_metric=hinge_error, n_jobs=N_JOBS)
    times['split_cloc']=time.perf_counter()-t0
    if split_cloc.size != p:
        raise ValueError(f"Cluster LOCO Split returned {split_cloc.size} features, expected {p}")
    ari['split_cloc'] = adjusted_rand_score(y_va, base_clusterer.fit_predict(X_va)) 
    
    # ---- Score 5: ClusterLOCOMP hinge_error ----
    print("======== Cluster LOCOMP ========")
    g = ClusterLOCOMP(base_clusterer=clone(base_clusterer), K=K, B=B)
    t0 = time.perf_counter()
    g.fit(X_aug, alpha_M=alpha_M, alpha_N=alpha_N, standardize=standardize, parallel={"n_jobs_features": N_JOBS},)
    cloc_out = g.score(hinge_error)
    times['cloc'] = time.perf_counter() - t0
    
    # adjust extraction for return type
    if isinstance(cloc_out, dict) and "delta" in cloc_out:
        cloc_raw = np.asarray(cloc_out["delta"], dtype=float).reshape(-1)
    else:
        cloc_raw = np.asarray(cloc_out, dtype=float).reshape(-1)

    if cloc_raw.size != p:
        raise ValueError(f"ClusterLOCOMP returned {cloc_raw.size} features, expected {p}")

    ari['cloc'] = adjusted_rand_score(y, g.z_ref)

    # ---- Score 6: RAMPART Cluster LOCO hinge_error ----
    print('======= ClusterLOCO RAMPART =======')
    gen_fn = ClusterLOCO_RAMPART(
        K=K,
        base_clusterer=clone(base_clusterer),
        error_metric=hinge_error,
        alpha_N =alpha_N,
        alpha_M =alpha_M,
        parallel_MP=True,
        parallel={"n_jobs_features": N_JOBS, "backend": "loky", "prefer": "processes", "verbose": 0},
        standardize=standardize,
    )
    t0 = time.perf_counter()
    out = RAMPART(
        X_aug,
        generalizability_fn=gen_fn,
        B=B,
        ranking_fn=transform_scores_to_ranking,
        top_k=topk,
        gen_kwargs={},  
    )
    times['rampart']=time.perf_counter()-t0
    rampart_raw = np.zeros(p,)
    rampart_raw[out['selected_indices']] = out['selected_scores']
    ramp_pos = out['selected_indices']
    
    ari['rampart'] = adjusted_rand_score(y, out['z_ref'])

    # ---- Compute comparison metrics ---- 
    # Top k recall using raw scores
    topk_hits["split_cloc"] = topk_overlap(split_cloc, true_idx, k=topk)
    topk_hits["cloc"] = topk_overlap(cloc_raw, true_idx, k=topk)
    topk_hits['rampart'] = np.intersect1d(ramp_pos, true_idx, assume_unique=False).size/topk
    
    return {
        "true_features":np.array([1.0]*informative_d + [0.0]*noise_d),
        "times":times,
        "topk_recall": topk_hits,
        "ari": ari,
    }

def run_chunk(
    *,
    cfg: dict,
    n_sims: int,
    setting_index: int,
    global_seed: int,
):

    sim_method = cfg.get("sim_method", "non-gaussian")
    K = int(cfg["K"])
    informative_d = int(cfg["informative_d"])
    topk = int(cfg.get("topk", informative_d))

    noise_plan = list(cfg["noise_plan"])
    noise_d = int(sum(int(block["d"]) for block in noise_plan))
    p = informative_d + noise_d

    # Important: use `elif`, otherwise moon-donut gets overwritten by KMeans.
    if sim_method in {"moon-donut", "swiss-roll"}:
        base_clusterer = BaseSpectralClustering(n_clusters=K)
    else:
        base_clusterer = KMeans(n_clusters=K)

    methods = [
        "split_cloc",
        "cloc",
        "rampart",
    ]

    times = {m: np.zeros(n_sims, dtype=float) for m in methods}
    topk_hits = {m: np.zeros(n_sims, dtype=float) for m in methods}
    ari = {m: np.zeros(n_sims, dtype=float) for m in methods}

    for t in range(n_sims):
        # Same task/replicate receives the same random seeds across
        # experiment settings when --seed is kept fixed.
        sim_seed = global_seed + 1_000_000 * setting_index + t
        embed_seed = global_seed + 2_000_000 * setting_index + t

        X_aug, y = generate_dataset_for_one_run(
            sim_method=sim_method,
            sim_seed=sim_seed,
            embed_seed=embed_seed,
            K=K,
            n_per_cluster=int(cfg["n_per_cluster"]),
            alpha=float(cfg["alpha"]),
            d0=int(cfg["d0"]),
            gaps=cfg["gaps"],
            shape_probs=cfg["shape_probs"],
            oversample=float(cfg["oversample"]),
            informative_d=informative_d,
            noise_plan=noise_plan,
        )

        out = run_one_simulation(
            X_aug=X_aug,
            y=y,
            K=K,
            informative_d=informative_d,
            noise_d=noise_d,
            base_clusterer=base_clusterer,
            B=int(cfg["B"]),
            standardize=bool(cfg.get("standardize", True)),
            topk=topk,
            alpha_N=float(cfg["alpha_N"]),
            alpha_M=float(cfg["alpha_M"]),
        )

        for method in methods:
            times[method][t] = float(out["times"][method])
            topk_hits[method][t] = float(out["topk_recall"][method])
            ari[method][t] = float(out["ari"][method])

    return {
    "setting_index": setting_index,
    "alpha_N": float(cfg["alpha_N"]),
    "alpha_M": float(cfg["alpha_M"]),
    "B": int(cfg["B"]),
    "N": int(K * cfg["n_per_cluster"]),
    "p": int(p),
    "alpha": float(cfg["alpha"]),
    "informative_d": informative_d,
    "noise_d": noise_d,
    **{f"ari_{m}": ari[m] for m in methods},
    **{f"time_{m}": times[m] for m in methods},
    **{f"topk_hits_{m}": topk_hits[m] for m in methods},
    }


def worker(task_id, cfg, experiment_name, difficulty_name, n_sims, seed, out_dir):
    run_chunk(cfg=cfg, experiment_name=experiment_name, difficulty_name=difficulty_name,
        n_sims=n_sims, task_id=task_id, global_seed=seed, out_dir=out_dir)


def save_chunk(records, out_path):
    """
    Save many grid settings into one compressed .npz file.

    Scalar metadata has shape (n_settings_in_chunk,).
    Metric arrays have shape (n_settings_in_chunk, n_sims).
    """
    scalar_keys = [
        "setting_index",
        "alpha_N",
        "alpha_M",
        "B",
        "N",
        "p",
        "alpha",
        "informative_d",
        "noise_d",
    ]

    payload = {
        key: np.asarray([record[key] for record in records])
        for key in scalar_keys
    }

    for method in METHODS:
        for prefix in ("ari", "time", "topk_hits"):
            key = f"{prefix}_{method}"
            payload[key] = np.stack(
                [record[key] for record in records],
                axis=0,
            )

    np.savez_compressed(out_path, **payload)


def main():
    ap = argparse.ArgumentParser()

    # Slurm array index: selects a chunk, not an individual setting.
    ap.add_argument("--chunk-index", type=int, required=True)
    ap.add_argument("--settings-per-chunk", type=int, default=16)

    ap.add_argument("--n-sims", type=int, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=123)

    args = ap.parse_args()
    METHODS = ["split_cloc", "cloc", "rampart"]
    BASE_CFG = {
        "sim_method": "gaussian",
        "K": 7,
        "alpha": 2.2,
        "d0": 10,
        "informative_d": 10,
        "topk": 10,
        "gaps": [0.02] * 7,
        "shape_probs": {"moon": 0.5, "donut": 0.5},
        "oversample": 10,
        "standardize": False,
    }

    GRID = {
        "N": [700, 1400, 2800],
        "p": [50, 200, 500, 1000],
        "alpha_N": [0.1, 0.2, 0.3, 0.4],
        "alpha_M": [0.1, 0.2, 0.3, 0.4],
        "B": [500, 1000, 2000, 5000],
    }

    K = BASE_CFG["K"]

    if any(N % K != 0 for N in GRID["N"]):
        raise ValueError(f"All N values must be divisible by K={K}.")

    settings = []

    for N, p, alpha_N, alpha_M, B in product(
        GRID["N"],
        GRID["p"],
        GRID["alpha_N"],
        GRID["alpha_M"],
        GRID["B"],
    ):
        cfg = deepcopy(BASE_CFG)
        cfg.update(
            {
                "n_per_cluster": N // K,
                "noise_plan": [
                    {
                        "type": "gaussian",
                        "d": p - BASE_CFG["informative_d"],
                    }
                ],
                "alpha_N": alpha_N,
                "alpha_M": alpha_M,
                "B": B,
            }
        )

        settings.append(cfg)

    start = args.chunk_index * args.settings_per_chunk
    stop = min(start + args.settings_per_chunk, len(settings))

    if start >= len(settings):
        raise ValueError(
            f"Chunk {args.chunk_index} is outside the grid. "
            f"There are {len(settings)} settings."
        )

    print(
        f"Chunk {args.chunk_index}: settings {start} through {stop - 1} "
        f"out of {len(settings)} total settings."
    )

    records = []

    for setting_index in range(start, stop):
        cfg = settings[setting_index]

        print(
            f"\nSetting {setting_index}: "
            f"N={K * cfg['n_per_cluster']}, "
            f"p={cfg['informative_d'] + sum(x['d'] for x in cfg['noise_plan'])}, "
            f"alpha_N={cfg['alpha_N']}, "
            f"alpha_M={cfg['alpha_M']}, "
            f"B={cfg['B']}"
        )

        record = run_chunk(
            cfg=cfg,
            setting_index=setting_index,
            n_sims=args.n_sims,
            global_seed=args.seed,
        )
        records.append(record)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"chunk_{args.chunk_index:04d}.npz"
    save_chunk(records, out_path)

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
