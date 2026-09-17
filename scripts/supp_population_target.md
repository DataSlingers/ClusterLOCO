# Cluster LOCO-MP population target

`supp_population_target.py` extracts the MP experiment from
`paper_figures/Supplementary/Supp_population_target.ipynb`.
Defaults: 5,000 independent MC replicates, 5,000 minipatches per fit,
500 observations (5 clusters × 100), 10 signal + 190 Gaussian noise features,
alpha=1.5, patch_n=100, patch_m=100, KMeans(n_init=10), decision tree,
full-data reference, no standardization, and mean hinge-error differences.

Two details of the notebook matter:

- Its MC generator defaults to seed 42 even after setting NumPy's global seed.
  This script passes the replicate data seed directly to BaseSimulator, producing
  fresh data and separate MP-fit randomness. Seeds are reproducible across worker
  counts, task partitions, and restarts.
- Its computed `Cov_k` is never passed to BaseSimulator. This script preserves
  the actual identity-covariance model; it does not add signal correlation.
  As in BaseSimulator, fresh seeds also redraw random cluster centers, so this
  averages over that simulator's random centers as well as its observations.

The notebook itself is not modified. No observed-data fit is required for saving
Deltas_mp; this script only runs the Monte Carlo fits.

## Single node

From the repository root, with the project dependencies installed (including a
current joblib supporting `parallel_config`):

```bash
python scripts/supp_population_target.py --jobs 4
```

Parallelism is across independent fits; inner MP/feature parallelism and native
BLAS threads are disabled to avoid oversubscribing cluster CPUs.

## SLURM array

The launcher follows the account, checkout path, and conda environment used by
this repository's existing cluster scripts. Adjust those paths/resources if
needed. Time and memory requests are starting estimates, not measured full-run
requirements; pilot a task before committing all 25 million minipatch fits.

```bash
mkdir -p logs
sbatch scripts/supp_population_target_slurm.sh
```

This submits 100 tasks with 50 replicates each, up to 8 concurrent tasks, and
4 processes per task. If changing the array range, also change `--num-tasks`.
After all tasks succeed, activate the same environment and merge:

```bash
python scripts/supp_population_target.py --merge-only
```

Alternatively submit a dependent merge job (replace JOB_ID with the array job ID):

```bash
sbatch --dependency=afterok:JOB_ID --account=stats --cpus-per-task=1 --mem=2G \
  --time=00:10:00 --output=logs/mp_target_merge_%j.out \
  --wrap='source /burg-archive/stats/users/cmh2277/miniforge3/etc/profile.d/conda.sh; conda activate ficluster; cd /burg-archive/stats/users/cmh2277/ClusterLOCO; python scripts/supp_population_target.py --merge-only'
```

Each replicate is saved atomically in a configuration-specific checkpoint folder.
Rerunning a task skips completed replicates. Undefined scores stop the task rather
than silently dropping/resampling replicates. Merge requires every replicate and
preserves seed/index order. Use the same statistical flags for run and merge;
use a separate output directory for each experiment. Checkpoints are keyed by
settings, not source code or dependency versions: do not change code/environment
mid-run or reuse checkpoints after such changes.

## Outputs

In `scripts/results/population_target/`:

- `cluster_locomp_mc.npy`: float array `(5000, 200)`, directly equivalent in layout
  to `Deltas_mp`; no pickled Python objects.
- `n_eff_mc_mp.npy`: valid observation counts with the same shape.
- `mc_summary.npz`: `Delta_mp`, `mc_se_mp` (between-replicate SD / sqrt(R)),
  ordered replicate `seeds`, and `config_json`.
- `checkpoints_*/replicate_*.npz`: scores, counts, and replicate/data/fit seeds.

```python
Deltas_mp = np.load("scripts/results/population_target/cluster_locomp_mc.npy")
Delta_mp = Deltas_mp.mean(axis=0)
mc_se_mp = Deltas_mp.std(axis=0, ddof=1) / np.sqrt(len(Deltas_mp))
```

For a cheap execution check (not a scientific run):

```bash
python scripts/supp_population_target.py --replicates 2 --patches 20 \
    --n-per-cluster 5 --jobs 2 --out-dir /tmp/clusterloco-mp-smoke
```
