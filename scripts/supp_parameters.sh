#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=param_sweep
#SBATCH --output=logs/param_sweep_%A_%a.out
#SBATCH --error=logs/param_sweep_%A_%a.err
#SBATCH --array=0-47%8
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --chdir=/burg-archive/stats/users/cmh2277/ClusterLOCO
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cmh2277@columbia.edu

set -euo pipefail

source "/burg-archive/stats/users/cmh2277/miniforge3/etc/profile.d/conda.sh"
conda activate ficluster

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-4}"

N_SIMS=5
SEED=$((123 + SLURM_ARRAY_TASK_ID))

mkdir -p logs

python scripts/supp_run_parameters.py \
    --chunk-index "$SLURM_ARRAY_TASK_ID" \
    --settings-per-chunk 16 \
    --n-sims 5 \
    --out-dir "./scripts/results/parameter_sweep" \
    --seed 123

# python scripts/supp_run_parameters.py \
#     --chunk-index 0 \
#     --settings-per-chunk 1 \
#     --n-sims 1 \
#     --out-dir "./scripts/results_test/parameter_sweep" \
#     --seed 123