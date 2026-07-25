#!/usr/bin/env bash
# Phase 2 baseline training — dense gradient continual update, no sparse/AM.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CARTRIDGES_DIR="${CARTRIDGES_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$CARTRIDGES_DIR/outputs}"
export PATH=${CUDA_HOME:+$CUDA_HOME/bin:}$PATH
export LD_LIBRARY_PATH=${CUDA_HOME:+$CUDA_HOME/lib64:}$LD_LIBRARY_PATH

if [ -z "${NUM_GPUS:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_GPUS=$(awk -F',' '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
  else
    NUM_GPUS=2
  fi
fi

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}"
PHASE1_CACHE_PATH="${PHASE1_CACHE_PATH:-}"
SYNTH_DATA_PATH="${SYNTH_DATA_PATH:-$CARTRIDGES_DIR/data/qasper/train/qasper_MT_task_8192_no-cartridge.parquet}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-$CARTRIDGES_DIR/examples/qasper2/qasper_eval_MT.parquet}"
NUM_TOKENS="${NUM_TOKENS:-512}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MASTER_PORT="${MASTER_PORT:-29508}"
EVAL_EVERY_N_STEPS="${EVAL_EVERY_N_STEPS:-15}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-256}"
RUN_NAME="${RUN_NAME:-qasper_baseline_phase2}"
DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-nccl}"

if [[ "$MODEL_NAME" == *"qwen"* || "$MODEL_NAME" == *"Qwen"* ]]; then
  SYNTH_DATA_PATH="${SYNTH_DATA_PATH:-$CARTRIDGES_DIR/data/qasper/train/qwen_qasper_MT_task_8192.parquet}"
fi

echo "=========================================="
echo "Qasper Baseline Phase 2 — Dense Gradient"
echo "=========================================="
echo "Model:           $MODEL_NAME"
echo "GPUs:            $NUM_GPUS"
echo "Phase1 cache:    $PHASE1_CACHE_PATH"
echo "Train data:      $SYNTH_DATA_PATH"
echo "Eval data:       $EVAL_DATA_PATH"
echo "Tokens:          $NUM_TOKENS"
echo "Epochs:          $EPOCHS"
echo "Run name:        $RUN_NAME"
echo "=========================================="

if [ -z "$PHASE1_CACHE_PATH" ] || [ ! -f "$PHASE1_CACHE_PATH" ]; then
  echo "Error: PHASE1_CACHE_PATH is required and must exist"
  exit 1
fi

if [ ! -f "$SYNTH_DATA_PATH" ]; then
  echo "Error: SYNTH_DATA_PATH not found: $SYNTH_DATA_PATH"
  exit 1
fi

source "$CARTRIDGES_DIR/.venv/bin/activate"

CARTRIDGES_DIR="$CARTRIDGES_DIR" \
PHASE1_CACHE_PATH="$PHASE1_CACHE_PATH" \
SYNTH_DATA_PATH="$SYNTH_DATA_PATH" \
EVAL_DATA_PATH="$EVAL_DATA_PATH" \
NUM_TOKENS="$NUM_TOKENS" \
EPOCHS="$EPOCHS" \
LR="$LR" \
GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
MODEL_NAME="$MODEL_NAME" \
CARTRIDGES_OUTPUT_DIR="$CARTRIDGES_OUTPUT_DIR" \
EVAL_EVERY_N_STEPS="$EVAL_EVERY_N_STEPS" \
SAVE_EVERY_N_STEPS="$SAVE_EVERY_N_STEPS" \
RUN_NAME="$RUN_NAME" \
DISTRIBUTED_BACKEND="$DISTRIBUTED_BACKEND" \
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
  "$CARTRIDGES_DIR/examples/qasper2/train/baseline_continual.py"

echo "=== Done ==="
