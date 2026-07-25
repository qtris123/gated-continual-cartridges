#!/usr/bin/env bash
# Phase 1 baseline training — dense gradient, 10 epochs, no sparse/AM.
# Reproduces published HF cartridges for Qasper QA -> MT benchmark.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CARTRIDGES_DIR="${CARTRIDGES_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$CARTRIDGES_DIR/outputs}"
export PATH=${CUDA_HOME:+$CUDA_HOME/bin:}$PATH
export LD_LIBRARY_PATH=${CUDA_HOME:+$CUDA_HOME/lib64:}$LD_LIBRARY_PATH

NUM_GPUS="${NUM_GPUS:-2}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}"
NUM_TOKENS="${NUM_TOKENS:-512}"
TEXT_PATH="${TEXT_PATH:-$CARTRIDGES_DIR/data/qasper/init_text/qasper_init_${NUM_TOKENS}.txt}"
SYNTH_DATA_PATH="${SYNTH_DATA_PATH:-$CARTRIDGES_DIR/data/qasper/train/qasper_QA_task_8192_no-cartridge.parquet}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-$CARTRIDGES_DIR/examples/qasper2/qasper_eval_QA.parquet}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MASTER_PORT="${MASTER_PORT:-29507}"
EVAL_EVERY_N_STEPS="${EVAL_EVERY_N_STEPS:-50}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-256}"
RUN_NAME="${RUN_NAME:-qasper_baseline_phase1}"

# Qwen defaults
if [[ "$MODEL_NAME" == *"qwen"* || "$MODEL_NAME" == *"Qwen"* ]]; then
  TEXT_PATH="${TEXT_PATH:-$CARTRIDGES_DIR/examples/qasper2/train/qwen_qasper_init_${NUM_TOKENS}.txt}"
  SYNTH_DATA_PATH="${SYNTH_DATA_PATH:-$CARTRIDGES_DIR/data/qasper/train/qwen_qasper_QA_task_8192.parquet}"
fi

echo "=========================================="
echo "Qasper Baseline Phase 1 — Dense Gradient"
echo "=========================================="
echo "Model:       $MODEL_NAME"
echo "GPUs:        $NUM_GPUS"
echo "Tokens:      $NUM_TOKENS"
echo "Text:        $TEXT_PATH"
echo "Train data:  $SYNTH_DATA_PATH"
echo "Eval data:   $EVAL_DATA_PATH"
echo "Epochs:      $EPOCHS"
echo "Run name:    $RUN_NAME"
echo "=========================================="

if [ -z "$SYNTH_DATA_PATH" ] || [ ! -f "$SYNTH_DATA_PATH" ]; then
  echo "Error: SYNTH_DATA_PATH not found: $SYNTH_DATA_PATH"
  exit 1
fi

source "$CARTRIDGES_DIR/.venv/bin/activate"

CARTRIDGES_DIR="$CARTRIDGES_DIR" \
TEXT_PATH="$TEXT_PATH" \
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
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
  "$CARTRIDGES_DIR/examples/qasper2/train/baseline_initial.py"

echo "=== Done ==="
