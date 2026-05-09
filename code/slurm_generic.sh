#!/bin/bash
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=medium

# CMD_FILE and EXTRA_ARGS are injected via --export by manager.py
# CMD_FILE: path to the .txt file with one command per line
# EXTRA_ARGS: optional flags like --overwrite

source /home/quantum-nas/.bashrc
conda activate qtcl
cd /path/to/QTCL/code   # <-- update this path

CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CMD_FILE")

if [ -z "$CMD" ]; then
    echo "[ERROR] No command found for task index $SLURM_ARRAY_TASK_ID in $CMD_FILE"
    exit 1
fi

echo "[INFO] Running: $CMD $EXTRA_ARGS"
eval "$CMD --machine-id hercules $EXTRA_ARGS"
