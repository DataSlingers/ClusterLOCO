#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=n500
#SBATCH --output=logs/N500_all_%A_%a.out
#SBATCH --error=logs/N500_all_%A_%a.err
#SBATCH --array=0-14 # 0-14 easy, 15-29 hard with 0: gaussian_20, 1: gaussian_50, 2: gaussian_200, 3: gaussian_500, 4: gaussian_1000, 5: moon_20, 6: moon_50, 7: moon_200, 8: moon_500, 9: moon_1000, 10: gamma_20, 11: gamma_50, 12: gamma_200, 13: gamma_500, 14: gamma_1000
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

CONFIG="./scripts/cfgs/experiments_N500.json" # "./scripts/cfgs/experiments.json" # original run
N_SIMS=10
mkdir -p logs

N_SETTINGS=15
SETTING_INDEX=$(( SLURM_ARRAY_TASK_ID % N_SETTINGS ))
DIFFICULTY_INDEX=$(( SLURM_ARRAY_TASK_ID / N_SETTINGS ))

if [ "$DIFFICULTY_INDEX" -eq 0 ]; then
    DIFFICULTY="easy"
else
    DIFFICULTY="hard"
fi

SEED=$((123 + SETTING_INDEX))

echo "Array task:    $SLURM_ARRAY_TASK_ID"
echo "Setting index: $SETTING_INDEX"
echo "Difficulty:    $DIFFICULTY"
echo "Seed:          $SEED"
echo "CPUs available: ${SLURM_CPUS_PER_TASK}"

python scripts/figure2_experiments.py \
    --config "$CONFIG" \
    --setting-index "$SETTING_INDEX" \
    --difficulty "$DIFFICULTY" \
    --n-sims "$N_SIMS" \
    --n-tasks 1 \
    --outer-jobs 1 \
    --inner-jobs "$SLURM_CPUS_PER_TASK" \
    --out-dir "./scripts/results_N500" \
    --seed "$SEED"
    