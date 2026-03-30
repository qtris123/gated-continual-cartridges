#!/bin/bash
#SBATCH --job-name=train-initial
#SBATCH --gres=gpu:1
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
NUM_GPUS="${NUM_GPUS:-1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
TEXT_PATH="${TEXT_PATH:-$REPO_DIR/data/financebench/texts/AMD_2021_10K.txt}"
SYNTH_DATA_PATH="${SYNTH_DATA_PATH:-/data/workspace/phudish/gated-continual-cartridges/outputs/2026-03-27-13-59-00-self_study/synthesize_amd_2021_Qwen/Qwen3-4B-Instruct-2507_n8192-0/artifact/dataset.parquet}"
NUM_TOKENS="${NUM_TOKENS:-2048}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
COMPANY="${COMPANY:-AMD}"
YEAR_Y="${YEAR_Y:-2021}"
CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$REPO_DIR/outputs}"
CARTRIDGES_DIR="${CARTRIDGES_DIR:-$REPO_DIR}"
#PROVENANCE_TAG="${PROVENANCE_TAG:-AMD 2021}"
MASTER_PORT="${MASTER_PORT:-29507}"

if [ -z "$SYNTH_DATA_PATH" ]; then
  echo "Error: SYNTH_DATA_PATH is required"
  exit 1
fi

source "$REPO_DIR/.venv/bin/activate"
if [ -f "$REPO_DIR/.env" ]; then
  source "$REPO_DIR/.env"
fi

echo "=== Training initial cartridge ==="
echo "Model:    $MODEL_NAME"
echo "GPUs:     $NUM_GPUS"
echo "Text:     $TEXT_PATH"
echo "Data:     $SYNTH_DATA_PATH"
echo "Tokens:   $NUM_TOKENS"
echo "Epochs:   $EPOCHS"
echo "Company:  $COMPANY  Year: $YEAR_Y"
echo "=================================="

CARTRIDGES_DIR="$CARTRIDGES_DIR" \
TEXT_PATH="$TEXT_PATH" \
SYNTH_DATA_PATH="$SYNTH_DATA_PATH" \
NUM_TOKENS="$NUM_TOKENS" \
EPOCHS="$EPOCHS" \
LR="$LR" \
GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
COMPANY="$COMPANY" \
YEAR_Y="$YEAR_Y" \
MODEL_NAME="$MODEL_NAME" \
CARTRIDGES_OUTPUT_DIR="$CARTRIDGES_OUTPUT_DIR" \
PROVENANCE_TAG="$PROVENANCE_TAG" \
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
  "$REPO_DIR/experiments/train/initial.py"

echo "=== Done ==="
