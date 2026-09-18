# Instructions for experiment scripts

This folder contains simulation runners, JSON configuration generators, PBMC
analyses, and SLURM launchers. Python files perform the analyses; `.sh` files
submit them to the compute cluster. Configuration generators only write JSON.

## Environment and working directory

Run analysis commands from the **repository root**, unless a command below
explicitly changes directory. Activate an environment containing the project and
its dependencies. Set the repository on the import path for the older scripts:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs
```

The benchmark runners also import PyTorch and other dependencies used by
`benchmarking`; the PBMC scripts require Scanpy and AnnData. The population-target
runner requires a joblib version supporting `parallel_config`.

The SLURM launchers currently use account `stats`, the checkout at
`/burg-archive/stats/users/cmh2277/ClusterLOCO`, and the `ficluster` conda
environment under `/burg-archive/stats/users/cmh2277/miniforge3`. Check these
settings before submission. Create `logs/` **before** calling `sbatch`, since
SLURM opens log files before the job body runs. Some older launchers also contain
email notification settings.

## Figure 2 simulations

### `figure2_run_configs.py`

Writes the combined configuration consumed by `figure2_experiments.py`. It defines
15 settings: Gaussian, moon/donut, and gamma data, each at nominal feature counts
20, 50, 200, 500, and 1,000, with easy/hard regimes. Current defaults include
7 clusters, **500 observations per cluster** (3,500 total), and 5,000 patches
for both MP and RAMPART. The `N500` filename refers to per-cluster size.

The output path is relative to the working directory, so run this generator
from `scripts/` to write `scripts/cfgs/experiments_N500.json`:

```bash
(cd scripts && python figure2_run_configs.py)
```

This overwrites that configuration. Edit `N`, `BASE`, or `DIFFICULTY_REGIMES`
in the generator when constructing a different experiment. Existing configs
are `cfgs/experiments.json` and `cfgs/experiments_N500.json`.

### `figure2_experiments.py`

Runs one setting and difficulty from a combined JSON config. Compares prototype
FI, NEON/LRP, importance accuracy, Cluster LOCO-Split, Cluster LOCO-MP, RAMPART,
permutation FI, and fuzzy c-means SHAP. Saves ARI, timing, and top-k recovery
metrics; MP and RAMPART use the streaming/cached implementation.

Example for the first Gaussian setting:

```bash
python scripts/figure2_experiments.py \
  --config scripts/cfgs/experiments_N500.json \
  --setting-index 0 --difficulty easy \
  --n-sims 10 --n-tasks 1 --outer-jobs 1 --inner-jobs 4 \
  --out-dir scripts/results_N500 --seed 123
```

`--setting-index` follows the JSON experiment order (0–14). `--n-tasks` is the
number of output chunks, each containing `--n-sims` simulations. `--outer-jobs`
controls concurrent chunks and `--inner-jobs` controls workers within a fit;
keep outer jobs at 1 for isolated timing comparisons. Cache controls include
`--patch-batch-size`, `--prediction-cache {auto,memory,disk}`, `--cache-mib`,
and `--cache-dir`.

Outputs are `<out-dir>/<setting>/<difficulty>/results_task00000.npz` and so on,
containing `ari_*`, `time_*`, `topk_hits_*`, phase timings, and runtime metadata.
Repeated runs using the same destination and task numbers overwrite these files.

### `figure2_slurm.sh`

Submits the Figure 2 runner with the N500 config, 10 simulations per setting,
4 CPUs, 8 GB memory, and a 12-hour limit per task. The current array `0-14`
runs **easy settings only**. Indices `15-29` select hard settings; use an override
to include both:

```bash
sbatch scripts/figure2_slurm.sh                  # easy only
sbatch --array=0-29 scripts/figure2_slurm.sh     # easy and hard
```

Results go to `scripts/results_N500/`; logs go to `logs/N500_all_*`.

### `run_configs.py`

Older configuration generator. Writes six separate JSON files (`moon_20`,
`moon_50`, `moon_200`, `gamma_20`, `gamma_50`, `gamma_200`) with 7 clusters and
200 observations per cluster:

```bash
(cd scripts && python run_configs.py)
```

These files use a single `experiment` object. They are **not compatible with
the current Figure 2 runner**, which requires combined `experiments` and
`difficulty_regimes` objects. Use `figure2_run_configs.py` for that runner.

## Figure 3 / PBMC analyses

### `figure3_PBMC68k_models.py` and `PBMC68k_models.py`

These two scripts currently have identical contents. They load Scanpy's
`pbmc68k_reduced` dataset, use `bulk_labels` to determine cluster count, and
compute cluster-level MP, RAMPART, and NEON/LRP results. The MP/RAMPART
clusterer is agglomerative clustering; LRP uses KMeans. Settings are hardcoded:
5,000 MP patches, 1,000 RAMPART patches, patch sizes 300 observations × 200
features, and top-k=200.

**Current code issue:** both scripts reference `RAMPART` and
`transform_scores_to_ranking` without importing them. Those imports need to be
added before a complete run. Their intended invocation is:

```bash
python scripts/figure3_PBMC68k_models.py --out scripts/results/pbmc68k_models.pkl
```

`PBMC68k_models.py` accepts the same `--out` argument. Include a parent directory
in the output path. The pickle contains `data`, `cluster_loco`, `rampart`, and,
if its calculation succeeds, `lrp`. LRP exceptions are currently suppressed,
so its key may be absent. There is no dedicated PBMC SLURM launcher here;
review the hardcoded parallel settings before scheduling it.

## Supplementary parameter sweep

### `supp_run_parameters.py`

Compares Cluster LOCO-Split, MP, and RAMPART over a Gaussian simulation grid with
7 clusters, 10 informative features, and alpha=2.2. The grid varies total sample
size (700, 1,400, 2,800), total features (50, 200, 500, 1,000), observation and
feature patch fractions (each 0.1, 0.2, 0.3, 0.4), and patch budget
(500, 1,000, 2,000, 5,000): **768 settings** in total.

`--chunk-index` selects a contiguous block of settings; the default block size
is 16. Intended invocation:

```bash
python scripts/supp_run_parameters.py \
  --chunk-index 0 --settings-per-chunk 16 --n-sims 5 \
  --out-dir scripts/results/parameter_sweep --seed 123
```

Each chunk writes `chunk_0000.npz` (and so on), with setting metadata and
`ari_*`, `time_*`, and `topk_hits_*` arrays of shape `(settings_in_chunk, n_sims)`.
Chunks are saved after all their settings finish; there is no per-replicate
restart mechanism here.

**Current code issue:** `save_chunk()` refers to `METHODS`, but `METHODS` is
only defined locally inside `main()`. It must be passed to `save_chunk()` or
moved to module scope before this runner can save results successfully.

### `supp_parameters.sh`

Launches the parameter sweep with 16 settings per chunk, 5 simulations per
setting, 4 CPUs, 8 GB memory, a 12-hour limit, and up to 8 concurrent tasks.
The current array `0-47` covers all **768 settings**. After resolving the
runner issue above, submit it with:

```bash
sbatch scripts/supp_parameters.sh
```

Outputs go to `scripts/results/parameter_sweep/`; logs use `logs/param_sweep_*`.

## Population target: `supp_population_target.py` and `supp_population_target_slurm.sh`

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

### Single node

From the repository root, with the project dependencies installed (including a
current joblib supporting `parallel_config`):

```bash
python scripts/supp_population_target.py --jobs 4
```

Parallelism is across independent fits; inner MP/feature parallelism and native
BLAS threads are disabled to avoid oversubscribing cluster CPUs.

### SLURM array

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

### Outputs

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
