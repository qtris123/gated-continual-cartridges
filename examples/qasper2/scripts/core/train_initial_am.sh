#!/usr/bin/env bash
# Phase 1: AM-based cartridge initialization (backprop-free).
set -e

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CARTRIDGES_DIR="${CARTRIDGES_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$CARTRIDGES_DIR/outputs}"

NUM_GPUS="${NUM_GPUS:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
NUM_TOKENS="${NUM_TOKENS:-512}"
TEXT_PATH="${TEXT_PATH:-$CARTRIDGES_DIR/data/qasper/init_text/qasper_init_${NUM_TOKENS}.txt}"
SYNTH_DATA_PATH="${SYNTH_DATA_PATH:-$CARTRIDGES_DIR/data/qasper/train/qwen_qasper_QA_task_8192.parquet}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-$CARTRIDGES_DIR/data/qasper/eval/qasper_eval_QA.parquet}"
AM_PASSES="${AM_PASSES:-3}"
AM_MAX_BATCHES="${AM_MAX_BATCHES:-50}"
GRANULARITY="${GRANULARITY:-per_layer}"
DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-nccl}"
RUN_NAME="${RUN_NAME:-qasper_phase1_am_${NUM_TOKENS}_${GRANULARITY}}"

if [ -f "$CARTRIDGES_DIR/.venv/bin/activate" ]; then
  source "$CARTRIDGES_DIR/.venv/bin/activate"
fi

echo "=== AM Phase 1 Init ==="
echo "GPU:         $CUDA_VISIBLE_DEVICES"
echo "Model:       $MODEL_NAME"
echo "Tokens:      $NUM_TOKENS"
echo "AM passes:   $AM_PASSES"
echo "AM batches:  $AM_MAX_BATCHES"
echo "Granularity: $GRANULARITY"
echo "========================"

export CUDA_VISIBLE_DEVICES
CARTRIDGES_DIR="$CARTRIDGES_DIR" \
CARTRIDGES_OUTPUT_DIR="$CARTRIDGES_OUTPUT_DIR" \
TEXT_PATH="$TEXT_PATH" \
SYNTH_DATA_PATH="$SYNTH_DATA_PATH" \
EVAL_DATA_PATH="$EVAL_DATA_PATH" \
NUM_TOKENS="$NUM_TOKENS" \
MODEL_NAME="$MODEL_NAME" \
AM_PASSES="$AM_PASSES" \
AM_MAX_BATCHES="$AM_MAX_BATCHES" \
GRANULARITY="$GRANULARITY" \
DISTRIBUTED_BACKEND="$DISTRIBUTED_BACKEND" \
RUN_NAME="$RUN_NAME" \
WANDB_DISABLED="${WANDB_DISABLED:-1}" \
python "$CARTRIDGES_DIR/examples/qasper2/train/initial_am.py"

echo "=== AM Phase 1 Done ==="
