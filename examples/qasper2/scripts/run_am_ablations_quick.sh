#!/usr/bin/env bash
# Quick ablation subset: A3 (update_interval), A4 (top_t), A5 (no IDF).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CARTRIDGES_DIR="${CARTRIDGES_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$CARTRIDGES_DIR/outputs}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export WANDB_DISABLED="${WANDB_DISABLED:-1}"

PHASE1_AM_PATH="${PHASE1_AM_PATH:?Set PHASE1_AM_PATH}"
BG_AM="${BG_AM:-$(dirname "$PHASE1_AM_PATH")/bg_stats.pt}"
MAX_STEPS="${MAX_STEPS:-5}"
EPOCHS=1
GLOBAL_BATCH_SIZE=16
TOP_T=64
GRANULARITY=per_layer

RESULTS_DIR="$CARTRIDGES_OUTPUT_DIR/am_ablations_quick_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
source "$CARTRIDGES_DIR/.venv/bin/activate"

log() { echo "$1" | tee -a "$RESULTS_DIR/ablation_results.log"; }

run_am() {
  local NAME="$1"; shift
  log "=== $NAME ==="
  T0=$(date +%s)
  env CUDA_VISIBLE_DEVICES=1 \
    PHASE1_CACHE_PATH="$PHASE1_AM_PATH" BG_STATS_PATH="$BG_AM" \
    EPOCHS="$EPOCHS" MAX_STEPS="$MAX_STEPS" TOP_T="$TOP_T" \
    GRANULARITY="$GRANULARITY" GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
    RUN_NAME="abl_${NAME}" "$@" \
    bash "$SCRIPT_DIR/train_continual_am_sparse.sh" \
    2>&1 | tee "$RESULTS_DIR/${NAME}.log"
  log "$NAME wall-clock: $(($(date +%s)-T0))s"
}

run_am A3_interval_1 TARGET_MODE=self UPDATE_INTERVAL=1 USE_IDF=1
run_am A3_interval_5 TARGET_MODE=self UPDATE_INTERVAL=5 USE_IDF=1
run_am A4_top_t_32 TARGET_MODE=self TOP_T=32 USE_IDF=1
run_am A4_top_t_128 TARGET_MODE=self TOP_T=128 USE_IDF=1
run_am A5_no_idf TARGET_MODE=self USE_IDF=0

log "=== Quick ablation evals (MT) ==="
export MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507
for CKPT in $(find "$CARTRIDGES_OUTPUT_DIR" -path "*abl_*" -name "cache_last.pt" -newer "$RESULTS_DIR" 2>/dev/null); do
  EXP_DIR=$(basename "$(dirname "$CKPT")")
  LOSS=$(CUDA_VISIBLE_DEVICES=0 CHECKPOINT_PATH="$CKPT" \
    EVAL_DATA_PATH="$CARTRIDGES_DIR/data/qasper/eval/qasper_eval_MT.parquet" \
    python "$CARTRIDGES_DIR/examples/qasper2/train/eval_forgetting.py" 2>&1 \
    | grep "Eval loss" | tail -1 || echo "eval_failed")
  log "$EXP_DIR MT: $LOSS"
done
log "Done: $RESULTS_DIR"
