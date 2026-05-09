#!/bin/bash
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard

# Variables injected via --export by manager.py:
#   CMD_FILE  — absolute path to the .txt command file
#   CODE_DIR  — absolute path to the code/ directory
#   EXTRA_ARGS — optional extra flags (e.g. --overwrite), empty by default

EXTRA_ARGS="${EXTRA_ARGS:-}"

module load Miniconda3
source activate qtcl

# Change to code directory so relative imports work
cd "$CODE_DIR" || { echo "[ERROR] Cannot cd to CODE_DIR=$CODE_DIR"; exit 1; }

# Read the N-th command from CMD_FILE (1-indexed by SLURM_ARRAY_TASK_ID)
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CMD_FILE")

if [ -z "$CMD" ]; then
    echo "[ERROR] No command for task index $SLURM_ARRAY_TASK_ID in $CMD_FILE"
    exit 1
fi

echo "[INFO] CODE_DIR=$CODE_DIR"
echo "[INFO] Running: $CMD $EXTRA_ARGS"
eval "$CMD $EXTRA_ARGS"
