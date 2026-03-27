#!/bin/bash
#SBATCH --job-name=train-continual
#SBATCH --gres=gpu:2
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
set -e

if [ -n "$SLURM_SUBMIT_DIR" ]; then
  REPO_DIR="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

# Configuration — adjust these as needed
NUM_GPUS="${NUM_GPUS:-2}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
PHASE1_CACHE_PATH="${PHASE1_CACHE_PATH:-/data/workspace/phudish/gated-continual-cartridges/outputs/2026-03-24-09-00-38-train_initial/36e44208-7d68-40cd-b4f7-a3d821c7e22e/cache_last.pt}"
SYNTH_DATA_PATH="${SYNTH_DATA_PATH:-/data/workspace/phudish/gated-continual-cartridges/outputs/2026-03-18-08-58-19-synthesize_self_study_data/synthesize_amd_2022_Qwen/Qwen3-4B_n8192-0/artifact/dataset.parquet}"
NUM_TOKENS="${NUM_TOKENS:-2048}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
COMPANY="${COMPANY:-AMD}"
YEAR_Y="${YEAR_Y:-2022}"
CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$REPO_DIR/outputs}"
CARTRIDGES_DIR="${CARTRIDGES_DIR:-$REPO_DIR}"
#PROVENANCE_TAG="${PROVENANCE_TAG:-PEPSI 2021}"
MASTER_PORT="${MASTER_PORT:-29506}"

if [ -z "$PHASE1_CACHE_PATH" ]; then
  echo "Error: PHASE1_CACHE_PATH is required"
  exit 1
fi

if [ -z "$SYNTH_DATA_PATH" ]; then
  echo "Error: SYNTH_DATA_PATH is required"
  exit 1
fi

source "$REPO_DIR/.venv/bin/activate"
if [ -f "$REPO_DIR/.env" ]; then
  source "$REPO_DIR/.env"
fi

echo "=== Training continual cartridge ==="
echo "Model:       $MODEL_NAME"
echo "GPUs:        $NUM_GPUS"
echo "Phase1 cache: $PHASE1_CACHE_PATH"
echo "Data:        $SYNTH_DATA_PATH"
echo "Tokens:      $NUM_TOKENS"
echo "Epochs:      $EPOCHS  LR: $LR"
echo "Company:     $COMPANY  Year: $YEAR_Y"
echo "===================================="

CARTRIDGES_DIR="$CARTRIDGES_DIR" \
PHASE1_CACHE_PATH="$PHASE1_CACHE_PATH" \
SYNTH_DATA_PATH="$SYNTH_DATA_PATH" \
NUM_TOKENS="$NUM_TOKENS" \
EPOCHS="$EPOCHS" \
LR="$LR" \
GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
COMPANY="$COMPANY" \
YEAR_Y="$YEAR_Y" \
MODEL_NAME="$MODEL_NAME" \
CARTRIDGES_OUTPUT_DIR="$CARTRIDGES_OUTPUT_DIR" \
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
  "$REPO_DIR/experiments/train/continual.py"

echo "=== Done ==="
