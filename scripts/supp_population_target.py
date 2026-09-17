#!/usr/bin/env python3
"""Monte Carlo Cluster LOCO-MP target from Supp_population_target.ipynb.

See supp_population_target.md for SLURM, restart, and output instructions.
"""
import argparse
from contextlib import redirect_stderr
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile

# Limit native threads before importing NumPy/sklearn; parallelism is across fits.
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits
from clim import ClusterLOCOMP
from clim.utils import hinge_error
from simulations.simulators import BaseSimulator


def atomic_save(path, **arrays):
    """Publish only complete files, including on shared cluster filesystems."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as f:
        temporary = Path(f.name)
        try:
            if path.suffix == '.npy':
                np.save(f, arrays['data'], allow_pickle=False)
            else:
                np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def generate_data(config, seed):
    # The notebook computes Cov_k but never passes it to BaseSimulator.
    # Preserve its actual identity-covariance model, omitting the unused draws.
    sim = BaseSimulator(K=5, n_samples_per_cluster=config['n_per_cluster'],
                        alpha=1.5, d_0=10, method='gaussian', random_state=int(seed))
    sim.generate_data()
    sim.add_noise(noise_d=190, noise_type='gaussian')
    return sim.X


def validate(delta, n_eff, config):
    if (delta.shape != (200,) or n_eff.shape != (200,)
            or not np.all(np.isfinite(delta)) or not np.all(np.isfinite(n_eff))
            or np.any(n_eff <= 0) or np.any(n_eff > 5 * config['n_per_cluster'])):
        raise RuntimeError('Undefined/invalid MP score. Increase the fixed patch budget '
                           'and start a new run; do not drop or resample failed replicates.')


def run_replicate(index, seed, config, checkpoint_dir):
    path = checkpoint_dir / f'replicate_{index:05d}.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            validate(saved['delta'], saved['n_eff'], config)
            if int(saved['seed']) != int(seed):
                raise ValueError(f'Seed mismatch in {path}')
        return index
    data_seed, fit_seed = [int(s.generate_state(1)[0])
                           for s in np.random.SeedSequence(int(seed)).spawn(2)]
    with threadpool_limits(limits=1):
        X = generate_data(config, data_seed)
        g = ClusterLOCOMP(K=5, B=config['patches'], random_state=fit_seed,
                          base_clusterer=KMeans(n_clusters=5, n_init=10),
                          base_classifier=DecisionTreeClassifier())
        g.fit(X, patch_n=max(5, int(0.2 * len(X))), patch_m=100,
              reference='full', standardize=False, parallel_MP=False, pprint=False)
        with redirect_stderr(StringIO()):
            out = g.score(error_metric=hinge_error, agg='mean', parallel_features=False)
    delta, n_eff = np.asarray(out['delta']), np.asarray(out['n_eff'])
    validate(delta, n_eff, config)
    atomic_save(path, delta=delta, n_eff=n_eff, seed=np.uint64(seed),
                data_seed=np.uint64(data_seed), fit_seed=np.uint64(fit_seed))
    print(f'Replicate {index + 1}/{config["replicates"]} saved', flush=True)
    return index


def merge(config, seeds, checkpoint_dir, out_dir):
    paths = [checkpoint_dir / f'replicate_{i:05d}.npz' for i in range(len(seeds))]
    missing = [i for i, p in enumerate(paths) if not p.exists()]
    if missing:
        raise RuntimeError(f'{len(missing)} replicates missing; first indices: {missing[:10]}')
    deltas, counts = [], []
    for path, seed in zip(paths, seeds):
        with np.load(path, allow_pickle=False) as saved:
            validate(saved['delta'], saved['n_eff'], config)
            if int(saved['seed']) != int(seed):
                raise ValueError(f'Seed mismatch in {path}')
            deltas.append(saved['delta'])
            counts.append(saved['n_eff'])
    deltas = np.stack(deltas)
    atomic_save(out_dir / 'cluster_locomp_mc.npy', data=deltas)
    atomic_save(out_dir / 'n_eff_mc_mp.npy', data=np.stack(counts))
    atomic_save(out_dir / 'mc_summary.npz', Delta_mp=deltas.mean(axis=0),
                mc_se_mp=deltas.std(axis=0, ddof=1) / np.sqrt(len(deltas)), seeds=seeds,
                config_json=np.array(json.dumps(config, sort_keys=True)))
    print(f'Saved {out_dir / "cluster_locomp_mc.npy"}: shape={deltas.shape}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replicates', type=int, default=5000)
    parser.add_argument('--patches', type=int, default=5000)
    parser.add_argument('--n-per-cluster', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--jobs', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))
    parser.add_argument('--task-index', type=int, default=0)
    parser.add_argument('--num-tasks', type=int, default=1)
    parser.add_argument('--out-dir', type=Path, default=Path('scripts/results/population_target'))
    parser.add_argument('--merge-only', action='store_true')
    args = parser.parse_args()
    if (args.replicates < 2 or args.patches < 1 or args.n_per_cluster < 2
            or args.seed < 0 or args.jobs < 1 or args.num_tasks < 1
            or not 0 <= args.task_index < args.num_tasks):
        parser.error('Invalid count, seed, or task index; require >=2 replicates and >=2 samples/cluster.')
    config = dict(version=1, replicates=args.replicates, patches=args.patches,
                  n_per_cluster=args.n_per_cluster, seed=args.seed,
                  K=5, signal_dim=10, noise_dim=190, alpha=1.5,
                  patch_n=max(5, int(0.2 * 5 * args.n_per_cluster)), patch_m=100,
                  reference='full', standardize=False, covariance='identity',
                  fresh_data=True)
    # Configuration-specific checkpoints prevent accidentally reusing another run.
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    checkpoint_dir = args.out_dir / f'checkpoints_{digest}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    seeds = np.random.default_rng(args.seed).choice(2**32, size=args.replicates, replace=False)
    if not args.merge_only:
        indices = range(args.task_index, args.replicates, args.num_tasks)
        print(f'Task {args.task_index}/{args.num_tasks}: {len(indices)} replicates; '
              f'{args.jobs} workers; checkpoints={checkpoint_dir}', flush=True)
        Parallel(n_jobs=args.jobs, backend='loky', verbose=10)(
            delayed(run_replicate)(i, seeds[i], config, checkpoint_dir) for i in indices)
    if args.merge_only or args.num_tasks == 1:
        merge(config, seeds, checkpoint_dir, args.out_dir)


if __name__ == '__main__':
    main()
