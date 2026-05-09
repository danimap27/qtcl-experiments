#!/bin/bash
#SBATCH --job-name=qtcl_lambda
#SBATCH --output=logs/lambda_%A_%a.out
#SBATCH --error=logs/lambda_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-20%6           # 3 heads × 7 lambdas × 1 seed = 21 runs
#SBATCH --partition=short

source /home/quantum-nas/.bashrc
conda activate qtcl
cd /path/to/QTCL/code

CMD=$(python runner.py --config config.yaml --study lambda_sensitivity --dry-run --export-commands \
    | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")

echo "Running: $CMD"
eval "$CMD --machine-id hercules --overwrite"
