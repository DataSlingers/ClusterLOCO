#!/bin/bash
#SBATCH --account=morphogenomics-lab
#SBATCH --job-name=f2_moons
#SBATCH --output=logs/fig2_all_%A_%a.out
#SBATCH --error=logs/fig2_all_%A_%a.err
#SBATCH --array=16-19 # 0-9 for Gaussian, 10-19 for Moon-Donut, 20-29 for Gamma
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --chdir=/burg-archive/stats/users/cmh2277/ClusterLOCO/
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cmh2277@columbia.edu

set -euo pipefail

source "/burg-archive/stats/users/cmh2277/miniforge3/etc/profile.d/conda.sh"
conda activate ficluster || {
    echo "[ERROR] conda environment 'ficluster' not found"
    exit 1
}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-4}"

CONFIG="./scripts/cfgs/experiments.json"
N_SIMS=5

SEED=$((123 + SLURM_ARRAY_TASK_ID))

mkdir -p logs

python scripts/figure2_experiments.py \
    --config "$CONFIG" \
    --setting-index "$SLURM_ARRAY_TASK_ID" \
    --n-sims "$N_SIMS" \
    --n-tasks 1 \
    --outer-jobs 1 \
    --out-dir "./scripts/results" \
    --seed "$SEED"
