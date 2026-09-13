"""
Comparative experiments: medium p setting (p_features = 10, p_noise = 10, 40) 

Gets run time + performance via top k overlap in 3 different settings. 
Keep track of ARI of base simulation and H-D simulation for assessing task's difficulty.
1) Concentric circles and moons with random noise (medium-hard)
2) Gamma mixture (hard)
3) Gaussian mixture (easy)

Author: Claire He

Runtime controls: --outer-jobs 1 isolates simulations; --inner-jobs controls
workers within each simulation. --prediction-cache auto spills to temporary
storage above --cache-mib (256 MiB per simulation by default). JSON experiment
settings B, B_ramp, B_shapley, patch_batch_size, prediction_cache,
max_cache_bytes and cache_dir are honored; explicit cache CLI options override
JSON settings. B_shapley defaults to the previous effective c-SHAP budget, 100.
Output retains the Figure 2 keys and adds time_cloc_fit, time_cloc_score and
runtime_config. Both cloc and rampart now use ClusterLOCOMPStream.
"""

import numpy as np
import time
import os
import json
import argparse
from functools import wraps
from pathlib import Path
from joblib import Parallel, delayed, parallel_config, cpu_count
from threadpoolctl import threadpool_limits

from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clim import RAMPART, GlobalStability_MP
from clim.data_splitting import Cluster_LOCO_Split
from clim.minipatches import ClusterLOCOMPStream
from clim.utils import hinge_error, transform_scores_to_ranking
from clim.models import BaseSpectralClustering
from simulations import BaseSimulator, GenerateCovariances
from simulations.simulators import permute_feature


def _load_benchmarks():
    # Keep --help and the LOCO helper usable without optional PyTorch.
    from benchmarking import LRP_score, PBFI, c_SHAP, feature_imp_cluster
    return LRP_score, PBFI, c_SHAP, feature_imp_cluster


def _limit_benchmark_threads(function):
    @wraps(function)
    def run(*args, **kwargs):
        # Discover native pools after importing optional libraries such as torch.
        _load_benchmarks()
        with threadpool_limits(limits=1):
            return function(*args, **kwargs)
    return run


def fit_cached_cloc(X, *, K, B, base_clusterer, standardize=True,
                    alpha_N=0.2, alpha_M=0.2, inner_n_jobs=1,
                    patch_batch_size=16, prediction_cache="auto",
                    max_cache_bytes=256 * 1024**2, cache_dir=None):
    """Shared streamed/cached evaluator for ordinary LOCO-MP and RAMPART."""
    model = ClusterLOCOMPStream(K=K, B=B, base_clusterer=base_clusterer)
    t0 = time.perf_counter()
    model.fit_stream(X, alpha_N=alpha_N, alpha_M=alpha_M,
                     standardize=standardize, n_jobs=inner_n_jobs,
                     patch_batch_size=patch_batch_size)
    fit_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    scores = model.score(hinge_error, cache=prediction_cache,
                         max_cache_bytes=max_cache_bytes, cache_dir=cache_dir)
    score_seconds = time.perf_counter() - t0
    return scores, model, fit_seconds, score_seconds

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

    sim = BaseSimulator(K=K, n_samples_per_cluster=n_per_cluster, alpha=alpha, d_0=d0,
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
    

@_limit_benchmark_threads
def run_one_simulation(*, X_aug, y, K: int, noise_d: int, informative_d: int = 10, base_clusterer=None,
        B: int = 5000, B_ramp: int = 1000, B_shapley: int = 100,
        standardize: bool = True, topk: int | None = None, inner_n_jobs: int = 1,
        patch_batch_size: int = 16, prediction_cache: str = "auto",
        max_cache_bytes: int = 256 * 1024**2, cache_dir: str | None = None):
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

    LRP_score, PBFI, c_SHAP, feature_imp_cluster = _load_benchmarks()
    topk_hits = {} 
    times = {} 
    ari = {}
    phase_times = {}
    cache_options = dict(inner_n_jobs=inner_n_jobs, patch_batch_size=patch_batch_size,
                         prediction_cache=prediction_cache,
                         max_cache_bytes=max_cache_bytes, cache_dir=cache_dir)
        
    print("======== Compute PBFI ========")
    # ---- Score 1: PBFI ----
    t0 = time.perf_counter()
    pbfi_raw = np.asarray(PBFI(X_aug, KMeans(n_clusters=K)), dtype=float).reshape(-1)
    times['pbfi'] = time.perf_counter()-t0
    if pbfi_raw.size != p:
        raise ValueError(f"PBFI returned {pbfi_raw.size} features, expected {p}")

    ari['pbfi'] = adjusted_rand_score(y, KMeans(n_clusters=K).fit_predict(X_aug))
        
    print("======== Compute LRP ========")
    # ---- Score 2: LRP ----
    t0 = time.perf_counter()
    lrp_raw = np.asarray(LRP_score(X_aug, K=K), dtype=float).reshape(-1)
    times['lrp'] = time.perf_counter()-t0
    if lrp_raw.size != p:
        raise ValueError(f"LRP returned {lrp_raw.size} features, expected {p}")
    ari['lrp'] = ari['pbfi']
    
    print("======== Compute IMPACC ========")
    # ---- Score 3: IMPACC ----
    t0 = time.perf_counter()
    impacc = GlobalStability_MP(X_aug, base_clusterer, n_clusters=K)
    impacc_res = impacc.impacc(X_aug.T, K=K)
    times['impacc'] = time.perf_counter()-t0
    impacc_raw = np.asarray(impacc_res["feature_importance"], dtype=float).reshape(-1)
    if impacc_raw.size != p:
        raise ValueError(f"IMPACC returned {impacc_raw.size} features, expected {p}")

    ari['impacc'] = adjusted_rand_score(y, impacc_res['labels'])

    # ---- Score 4: Cluster LOCO hinge_error ----
    print("======== Split Cluster LOCO ========")
    X_tr, X_va, y_tr, y_va = train_test_split(X_aug, y, test_size=0.5, stratify=y)
    t0 = time.perf_counter()
    split_cloc, _ = Cluster_LOCO_Split(X_tr, X_va, model=base_clusterer, clf = RandomForestClassifier(), K=K,error_metric=None, n_jobs=inner_n_jobs)
    times['split_cloc']=time.perf_counter()-t0
    if split_cloc.size != p:
        raise ValueError(f"Cluster LOCO Split returned {split_cloc.size} features, expected {p}")
    ari['split_cloc'] = ari['pbfi']
    
    # ---- Score 5: ClusterLOCOMP hinge_error ----
    print("======== Cluster LOCOMP ========")
    t0 = time.perf_counter()
    cloc_out, g, fit_seconds, score_seconds = fit_cached_cloc(
        X_aug, K=K, B=B, base_clusterer=base_clusterer,
        standardize=standardize, **cache_options)
    phase_times.update(cloc_fit=fit_seconds, cloc_score=score_seconds)
    times['cloc']=time.perf_counter() - t0
    
    # adjust extraction for return type
    if isinstance(cloc_out, dict) and "delta" in cloc_out:
        cloc_raw = np.asarray(cloc_out["delta"], dtype=float).reshape(-1)
    else:
        cloc_raw = np.asarray(cloc_out, dtype=float).reshape(-1)

    if cloc_raw.size != p:
        raise ValueError(f"ClusterLOCOMP returned {cloc_raw.size} features, expected {p}")

    ari['cloc'] = adjusted_rand_score(y, g.z_ref)
    del g, cloc_out  # Release the ensemble before RAMPART fits more models.

    # ---- Score 6: RAMPART Cluster LOCO hinge_error ----
    print('======= ClusterLOCO RAMPART =======')
    def gen_fn(X_sub, *, B, alpha_N, alpha_M, **_ignored):
        scores, model, _, _ = fit_cached_cloc(
            X_sub, K=K, B=B, base_clusterer=base_clusterer,
            alpha_N=alpha_N, alpha_M=alpha_M,
            standardize=standardize, **cache_options)
        return dict(scores, z_ref=model.z_ref), model

    t0 = time.perf_counter()
    out = RAMPART(
        X_aug,
        generalizability_fn=gen_fn,
        B=B_ramp,
        ranking_fn=transform_scores_to_ranking,
        top_k=topk,
        gen_kwargs={},  
    )
    times['rampart']=time.perf_counter()-t0
    rampart_raw = np.zeros(p,)
    rampart_raw[out['selected_indices']] = out['selected_scores']
    ramp_pos = out['selected_indices']
    
    ari['rampart'] = adjusted_rand_score(y, out['z_ref'])
    del out

    # ---- Score 7: Permutation -----
    print("======== Permutation ========")
    t0 = time.perf_counter()
    fitted_clusterer = clone(base_clusterer).fit(X_aug) 
    perm_raw = feature_imp_cluster(fitted_clusterer, X_aug)['featureImp'].values 
    times['perm']=time.perf_counter() - t0
    ari['perm'] = adjusted_rand_score(y, fitted_clusterer.predict(X_aug))
    
    # ---- Score 8: c-SHAP ----
    print("======== c-SHAP ========")
    t0 = time.perf_counter()
    cshap_raw, shap_labels = c_SHAP(X=X_aug, K=K, method='kernel', X_reference=X_aug, M=B_shapley, n_jobs=inner_n_jobs).get_model_wide_importance()
    times['cshap'] = time.perf_counter() - t0
    ari['cshap'] = adjusted_rand_score(y, shap_labels)


    # ---- Compute comparison metrics ---- 
    # Top k recall using raw scores
    topk_hits["pbfi"] = topk_overlap(pbfi_raw, true_idx, k=topk)
    topk_hits["lrp"] = topk_overlap(lrp_raw,  true_idx, k=topk)
    topk_hits["impacc"] = topk_overlap(impacc_raw,true_idx,k=topk)
    topk_hits["split_cloc"] = topk_overlap(split_cloc, true_idx, k=topk)
    topk_hits["cloc"] = topk_overlap(cloc_raw, true_idx, k=topk)
    topk_hits["perm"] = topk_overlap(perm_raw, true_idx, k=topk)
    topk_hits['cshap'] = topk_overlap(cshap_raw, true_idx, k=topk)
    topk_hits['rampart'] = np.intersect1d(ramp_pos, true_idx, assume_unique=False).size/topk
    
    return {
        "true_features":np.array([1.0]*informative_d + [0.0]*noise_d),
        "times":times,
        "phase_times": phase_times,
        "topk_recall": topk_hits,
        "ari": ari,
    }

def run_chunk(*, cfg: dict, cfg_id: int, n_sims: int, task_id: int, global_seed: int, out_dir: str, inner_n_jobs: int = 1):
    os.makedirs(out_dir, exist_ok=True)
    cfg = dict(B=5000, B_ramp=1000, B_shapley=100, patch_batch_size=16,
               prediction_cache="auto", max_cache_bytes=256 * 1024**2,
               cache_dir=None, standardize=True) | cfg
    # experiment constants
    sim_method = cfg.get("sim_method", "non-gaussian")
    K = int(cfg["K"])
    informative_d = int(cfg["informative_d"])
    topk = int(cfg.get("topk", informative_d))

    noise_plan = list(cfg["noise_plan"])
    noise_d = int(sum(int(b["d"]) for b in noise_plan))
    p = informative_d + noise_d

    # base clusterer (create once per task)  
    if sim_method in {'moon-donut', 'swiss-roll'}:
        base_clusterer = BaseSpectralClustering(n_clusters=K)
    else:
    # For GMM and Gamma
        base_clusterer = KMeans(n_clusters=K)

    methods = ["pbfi", "lrp", "impacc", "split_cloc", "cloc", "rampart","perm", "cshap"] # , "rampshap"]

    phase_times = {m: np.zeros(n_sims) for m in ("cloc_fit", "cloc_score")}
    times = {m: np.zeros((n_sims,), dtype=float) for m in methods}
    topk_hits = {m: np.zeros((n_sims,), dtype=float) for m in methods}
    difficulty = {m: np.zeros(n_sims) for m in methods} #  ["spectral","fast-spectral","kmeans","gmm"]

    for t in range(n_sims):
        sim_seed = global_seed + 1_000_000 * cfg_id + 10_000 * task_id + t
        embed_seed = global_seed + 2_000_000 * cfg_id + 10_000 * task_id + t

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
            B=int(cfg.get("B", 5000)),
            B_ramp=int(cfg.get("B_ramp", 1000)),
            B_shapley=int(cfg.get("B_shapley", 100)),
            standardize=bool(cfg.get("standardize", True)),
            topk=topk,
            inner_n_jobs=inner_n_jobs,
            patch_batch_size=int(cfg.get("patch_batch_size", 16)),
            prediction_cache=cfg.get("prediction_cache", "auto"),
            max_cache_bytes=int(cfg.get("max_cache_bytes", 256 * 1024**2)),
            cache_dir=cfg.get("cache_dir"),
        )

        for phase in phase_times:
            phase_times[phase][t] = out["phase_times"][phase]

        for m in methods:
            times[m][t] = float(out["times"][m])
            topk_hits[m][t] = float(out["topk_recall"][m])
            difficulty[m][t] = float(out['ari'][m])

    out_path = os.path.join(out_dir, f"results_task{task_id:05d}.npz")
    np.savez_compressed(
        out_path,
        cfg_id=np.array([cfg_id], dtype=np.int32),
        runtime_config=np.array(json.dumps(dict(
            cfg, inner_n_jobs=inner_n_jobs, native_threads=1,
            cloc_implementation="ClusterLOCOMPStream"))),
        **{f"time_{phase}": values for phase, values in phase_times.items()},
        task_id=np.array([task_id], dtype=np.int32),
        global_seed=np.array([global_seed], dtype=np.int64),
        informative_d=np.array([informative_d], dtype=np.int32),
        noise_d=np.array([noise_d], dtype=np.int32),
        **{f"ari_{m}": difficulty[m] for m in methods},
        **{f"time_{m}": times[m] for m in methods},
        **{f"topk_hits_{m}": topk_hits[m] for m in methods},
    )
    print(f"Wrote {out_path}")

#  Parallelize on simulations 
def worker(task_id, cfg, n_sims, seed, out_dir, inner_n_jobs):
    run_chunk(
        cfg=cfg,
        cfg_id=0,
        n_sims=n_sims,
        task_id=task_id,
        global_seed=seed,
        out_dir=out_dir,
        inner_n_jobs=inner_n_jobs,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--n-sims", type=int, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n-tasks", type=int, required=True)
    ap.add_argument("--outer-jobs", type=int, default=1,
                    help="Concurrent simulations; use 1 for isolated runtime comparisons")
    ap.add_argument("--inner-jobs", type=int, default=1,
                    help="Workers per simulation; native threads per worker are capped at 1")
    ap.add_argument("--patch-batch-size", type=int, default=None)
    ap.add_argument("--prediction-cache", choices=("auto", "memory", "disk"), default=None)
    ap.add_argument("--cache-mib", type=int, default=None,
                    help="RAM cache budget per simulation; excess spills to disk in auto mode")
    ap.add_argument("--cache-dir", type=str, default=None)
    args = ap.parse_args()
    if min(args.outer_jobs, args.inner_jobs, args.n_tasks, args.n_sims) < 1:
        ap.error("job, task and simulation counts must be positive")
    if min(args.outer_jobs, args.n_tasks) * args.inner_jobs > cpu_count():
        ap.error("outer-jobs * inner-jobs exceeds the available CPU budget")
    if args.patch_batch_size is not None and args.patch_batch_size < 1:
        ap.error("patch-batch-size must be positive")
    if args.cache_mib is not None and args.cache_mib < 0:
        ap.error("cache-mib must be nonnegative")

    with open(args.config, "r") as f:
        cfg_all = json.load(f)

    cfg = dict(cfg_all["experiment"])
    for key in ("patch_batch_size", "prediction_cache", "cache_dir"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.cache_mib is not None:
        cfg["max_cache_bytes"] = args.cache_mib * 1024**2
    cfg["outer_jobs"] = args.outer_jobs
    cfg["gaps"] = cfg.get("gaps", [0.2] * int(cfg["K"]))
    cfg["shape_probs"] = cfg.get("shape_probs", {"donut": 0.5, "moon": 0.5})
    cfg["oversample"] = cfg.get("oversample", 10)
    os.makedirs(args.out_dir, exist_ok=True)

    with threadpool_limits(limits=1), parallel_config(backend="loky", inner_max_num_threads=1):
        Parallel(n_jobs=args.outer_jobs, verbose=10)(
            delayed(worker)(
                task_id=task_id,
                cfg=cfg,
                n_sims=args.n_sims,
                seed=args.seed,
                out_dir=args.out_dir,
                inner_n_jobs=args.inner_jobs,
            )
            for task_id in range(args.n_tasks)
        )

if __name__ == "__main__":
    main()
