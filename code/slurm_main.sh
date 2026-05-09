#!/bin/bash
#SBATCH --job-name=qtcl_main
#SBATCH --output=logs/main_%A_%a.out
#SBATCH --error=logs/main_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-89%10          # 2 datasets × 3 backbones × 3 heads × 5 seeds = 90 runs
#SBATCH --partition=medium

# ── Environment ────────────────────────────────────────────────────────────────
source /home/quantum-nas/.bashrc
conda activate qtcl
cd /path/to/QTCL/code   # <-- update this path

# ── Export commands and pick the SLURM_ARRAY_TASK_ID-th one ───────────────────
CMD=$(python runner.py --config config.yaml --dry-run --export-commands \
    | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")

echo "Running: $CMD"
eval "$CMD --machine-id hercules --overwrite"
