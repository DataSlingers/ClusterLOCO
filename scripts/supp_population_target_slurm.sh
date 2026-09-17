#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=mp_target
#SBATCH --output=logs/mp_target_%A_%a.out
#SBATCH --error=logs/mp_target_%A_%a.err
#SBATCH --array=0-99%8
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --chdir=/burg-archive/stats/users/cmh2277/ClusterLOCO

set -euo pipefail
source "/burg-archive/stats/users/cmh2277/miniforge3/etc/profile.d/conda.sh"
conda activate ficluster
export PYTHONUNBUFFERED=1

# 100 tasks x 50 replicates = 5000; each task runs four independent fits at once.
# Requeue/rerun the same command to reuse completed replicate checkpoints.
python scripts/supp_population_target.py \
    --replicates 5000 --patches 5000 \
    --jobs "${SLURM_CPUS_PER_TASK:-4}" \
    --task-index "${SLURM_ARRAY_TASK_ID}" --num-tasks 100 \
    --out-dir scripts/results/population_target
