#!/bin/bash
#SBATCH --job-name=qtcl_ablation
#SBATCH --output=logs/ablation_%A_%a.out
#SBATCH --error=logs/ablation_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-59%8           # 2 heads × 4 qubits × 3 depths (excl default) × 2 seeds ≈ 60
#SBATCH --partition=standard

source /home/quantum-nas/.bashrc
conda activate qtcl
cd /path/to/QTCL/code

CMD=$(python runner.py --config config.yaml --study ablation --dry-run --export-commands \
    | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")

echo "Running: $CMD"
eval "$CMD --machine-id hercules --overwrite"
