#!/usr/bin/env bash
# Systematic AM-Sparse ablations (A1–A4) concurrent with gradient baseline on 2 GPUs.
#
# Usage:
#   PHASE1_AM_PATH=/path/to/am/cache_last.pt \
#   PHASE1_GRAD_PATH=/path/to/grad/cache.pt \
#   bash examples/qasper2/scripts/sweeps/run_am_ablations.sh
#
# Env:
#   QUICK=1       — MAX_STEPS=20, EPOCHS=1 (default for ablations)
#   MAX_STEPS     — override step cap per ablation run

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CARTRIDGES_DIR="${CARTRIDGES_DIR:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$CARTRIDGES_DIR/outputs}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export WANDB_DISABLED="${WANDB_DISABLED:-1}"

PHASE1_AM_PATH="${PHASE1_AM_PATH:?Set PHASE1_AM_PATH}"
PHASE1_GRAD_PATH="${PHASE1_GRAD_PATH:-}"
BG_AM="${BG_AM:-$(dirname "$PHASE1_AM_PATH")/bg_stats.pt}"
BG_GRAD="${BG_GRAD:-$(dirname "$PHASE1_GRAD_PATH")/bg_stats.pt}"

if [ "${QUICK:-1}" = "1" ]; then
  EPOCHS=1
  MAX_STEPS="${MAX_STEPS:-20}"
  GLOBAL_BATCH_SIZE=16
else
  EPOCHS="${EPOCHS:-1}"
  MAX_STEPS="${MAX_STEPS:-50}"
  GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
fi

TOP_T="${TOP_T:-64}"
GRANULARITY="${GRANULARITY:-per_layer}"
RESULTS_DIR="$CARTRIDGES_OUTPUT_DIR/am_ablations_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

if [ -f "$CARTRIDGES_DIR/.venv/bin/activate" ]; then
  source "$CARTRIDGES_DIR/.venv/bin/activate"
fi

log() { echo "$1" | tee -a "$RESULTS_DIR/ablation_results.log"; }

run_am_ablation() {
  local NAME="$1"
  local GPU="$2"
  shift 2
  local EXTRA=("$@")

  log "=== ABLATION: $NAME (GPU $GPU) ==="
  T0=$(date +%s)
  env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    PHASE1_CACHE_PATH="$PHASE1_AM_PATH" \
    BG_STATS_PATH="$BG_AM" \
    EPOCHS="$EPOCHS" \
    MAX_STEPS="$MAX_STEPS" \
    TOP_T="$TOP_T" \
    GRANULARITY="$GRANULARITY" \
    GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
    RUN_NAME="abl_${NAME}" \
    "${EXTRA[@]}" \
    bash "$SCRIPT_DIR/../core/train_continual_am_sparse.sh" \
    2>&1 | tee "$RESULTS_DIR/${NAME}.log"
  T1=$(date +%s)
  log "$NAME wall-clock: $((T1-T0))s"
}

# GPU 0: gradient sparse control (if grad checkpoint available)
if [ -n "$PHASE1_GRAD_PATH" ] && [ -f "$PHASE1_GRAD_PATH" ]; then
  log "=== CONTROL: gradient sparse freeze (GPU 0) ==="
  T0=$(date +%s)
  env \
    CUDA_VISIBLE_DEVICES=0 \
    NUM_GPUS=1 \
    DISTRIBUTED_BACKEND=gloo \
    PHASE1_CACHE_PATH="$PHASE1_GRAD_PATH" \
    BG_STATS_PATH="$BG_GRAD" \
    EPOCHS="$EPOCHS" \
    MAX_STEPS="$MAX_STEPS" \
    TOP_T="$TOP_T" \
    GRANULARITY="$GRANULARITY" \
    GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
    MOMENTUM_MASKING=freeze \
    RUN_NAME="abl_control_grad_sparse" \
    bash "$SCRIPT_DIR/../core/train_continual_sparse.sh" \
    2>&1 | tee "$RESULTS_DIR/control_grad_sparse.log" &
  CTRL_PID=$!
  T_CTRL=$T0
else
  CTRL_PID=""
  log "Skipping gradient control (PHASE1_GRAD_PATH not set)"
fi

# GPU 1: ablation variants (sequential on GPU 1)
# A1: target_mode
run_am_ablation "A1_target_self" 1 TARGET_MODE=self USE_IDF=1
run_am_ablation "A1_target_teacher_attn" 1 TARGET_MODE=teacher_attention USE_IDF=1

# A2: queries_per_batch
run_am_ablation "A2_queries_last_token" 1 TARGET_MODE=self QUERIES_PER_BATCH=last_token USE_IDF=1
run_am_ablation "A2_queries_all_tokens" 1 TARGET_MODE=self QUERIES_PER_BATCH=all_tokens USE_IDF=1

# A3: update_interval
run_am_ablation "A3_interval_1" 1 TARGET_MODE=self UPDATE_INTERVAL=1 USE_IDF=1
run_am_ablation "A3_interval_5" 1 TARGET_MODE=self UPDATE_INTERVAL=5 USE_IDF=1

# A4: top_t
run_am_ablation "A4_top_t_32" 1 TARGET_MODE=self TOP_T=32 USE_IDF=1
run_am_ablation "A4_top_t_64" 1 TARGET_MODE=self TOP_T=64 USE_IDF=1
run_am_ablation "A4_top_t_128" 1 TARGET_MODE=self TOP_T=128 USE_IDF=1

# A5: no IDF
run_am_ablation "A5_no_idf" 1 TARGET_MODE=self USE_IDF=0

wait ${CTRL_PID:-} 2>/dev/null || true
if [ -n "${CTRL_PID:-}" ]; then
  T1=$(date +%s)
  log "control_grad_sparse wall-clock: $((T1-T_CTRL))s"
fi

log "=== Ablation evals ==="
for CKPT in $(find "$CARTRIDGES_OUTPUT_DIR" -path "*abl_*" -name "cache_last.pt" 2>/dev/null); do
  EXP_DIR=$(basename "$(dirname "$CKPT")")
  for EVAL in QA MT; do
    EVAL_PATH="$CARTRIDGES_DIR/data/qasper/eval/qasper_eval_${EVAL}.parquet"
    [ -f "$EVAL_PATH" ] || continue
    LOSS=$(CUDA_VISIBLE_DEVICES=0 CHECKPOINT_PATH="$CKPT" \
      EVAL_DATA_PATH="$EVAL_PATH" WANDB_DISABLED=1 \
      python "$CARTRIDGES_DIR/examples/qasper2/train/eval_forgetting.py" 2>&1 \
      | grep "Eval loss" | tail -1 || echo "eval_failed")
    log "$EXP_DIR $EVAL: $LOSS"
  done
done

log "=== Ablation sweep complete: $RESULTS_DIR ==="
