#!/usr/bin/env bash
# Run AM-Sparse experiment suite: baselines + ablations on 2× GH200 GPUs.
#
# Usage:
#   bash examples/qasper2/scripts/sweeps/run_am_experiments.sh
#
# Experiments:
#   GPU 0: EXP-BASE (gradient sparse baseline) — if PHASE1_GRAD_PATH set
#   GPU 1: EXP-AM2, ablations
#
# Env overrides:
#   QUICK=1          — reduced epochs/steps for smoke test
#   SKIP_PHASE1=1    — skip Phase 1 training (use existing checkpoints)
#   PHASE1_GRAD_PATH — existing gradient Phase 1 cache for baseline Phase 2

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CARTRIDGES_DIR="${CARTRIDGES_DIR:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$CARTRIDGES_DIR/outputs}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export WANDB_DISABLED="${WANDB_DISABLED:-1}"

NUM_TOKENS="${NUM_TOKENS:-512}"
GRANULARITY="${GRANULARITY:-per_layer}"
TOP_T="${TOP_T:-64}"

if [ "${QUICK:-0}" = "1" ]; then
  EPOCHS=1
  MAX_STEPS=50
  AM_MAX_BATCHES=10
  AM_PASSES=1
  GLOBAL_BATCH_SIZE=16
else
  EPOCHS="${EPOCHS:-10}"
  MAX_STEPS="${MAX_STEPS:-550}"
  AM_MAX_BATCHES="${AM_MAX_BATCHES:-50}"
  AM_PASSES="${AM_PASSES:-3}"
  GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
fi

RESULTS_DIR="$CARTRIDGES_OUTPUT_DIR/am_experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
echo "Results dir: $RESULTS_DIR"

if [ -f "$CARTRIDGES_DIR/.venv/bin/activate" ]; then
  source "$CARTRIDGES_DIR/.venv/bin/activate"
fi

log_result() {
  echo "$1" | tee -a "$RESULTS_DIR/results.log"
}

# --- Phase 1: AM init (GPU 1) ---
if [ "${SKIP_PHASE1:-0}" != "1" ]; then
  log_result "=== EXP: Phase 1 AM init (GPU 1) ==="
  T0=$(date +%s)
  CUDA_VISIBLE_DEVICES=1 \
  NUM_TOKENS="$NUM_TOKENS" GRANULARITY="$GRANULARITY" \
  AM_PASSES="$AM_PASSES" AM_MAX_BATCHES="$AM_MAX_BATCHES" \
  RUN_NAME="exp_am1_${NUM_TOKENS}_${GRANULARITY}" \
  bash "$SCRIPT_DIR/../core/train_initial_am.sh" 2>&1 | tee "$RESULTS_DIR/phase1_am.log"
  T1=$(date +%s)
  log_result "Phase 1 AM wall-clock: $((T1-T0))s"

  PHASE1_AM_PATH=$(find "$CARTRIDGES_OUTPUT_DIR" -name "cache_last.pt" -newer "$RESULTS_DIR" 2>/dev/null | head -1)
  PHASE1_AM_BG=$(dirname "$PHASE1_AM_PATH")/bg_stats.pt 2>/dev/null || true
  log_result "Phase 1 AM cache: $PHASE1_AM_PATH"
fi

# --- Phase 1: Gradient baseline (GPU 0) — optional ---
if [ "${SKIP_PHASE1:-0}" != "1" ] && [ "${SKIP_GRAD_PHASE1:-0}" != "1" ]; then
  log_result "=== EXP: Phase 1 gradient baseline (GPU 0) ==="
  T0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 \
  NUM_TOKENS="$NUM_TOKENS" GRANULARITY="$GRANULARITY" EPOCHS="${EPOCHS:-10}" \
  RUN_NAME="exp_grad_phase1_${NUM_TOKENS}_${GRANULARITY}" \
  bash "$SCRIPT_DIR/../core/train_initial_sparse.sh" 2>&1 | tee "$RESULTS_DIR/phase1_grad.log" &
  GRAD_P1_PID=$!

  wait $GRAD_P1_PID || true
  T1=$(date +%s)
  log_result "Phase 1 gradient wall-clock: $((T1-T0))s"
  PHASE1_GRAD_PATH=$(find "$CARTRIDGES_OUTPUT_DIR" -path "*exp_grad_phase1*" -name "cache_last.pt" 2>/dev/null | head -1)
  log_result "Phase 1 grad cache: $PHASE1_GRAD_PATH"
fi

PHASE1_GRAD_PATH="${PHASE1_GRAD_PATH:-}"
PHASE1_AM_PATH="${PHASE1_AM_PATH:-}"

# Defaults when skipping Phase 1 training
if [ "${SKIP_PHASE1:-0}" = "1" ]; then
  PHASE1_AM_PATH="${PHASE1_AM_PATH:-/localhome/local-triv/gated-continual-cartridges/outputs/2026-07-10-06-22-23-initial_am/ba072f19-1d45-4f7c-bbb8-e2f98ea123d9/cache_last.pt}"
  PHASE1_GRAD_PATH="${PHASE1_GRAD_PATH:-/localhome/local-triv/gated-continual-cartridges/outputs/2026-06-22-08-31-41-initial_sparse_qwen_qasper_per-layer_max-tokens-512_all-reduce/ef9e7f9d-7b66-4369-86d6-bdecd9db9efd/cache-step622.pt}"
  log_result "SKIP_PHASE1=1: AM=$PHASE1_AM_PATH"
  log_result "SKIP_PHASE1=1: GRAD=$PHASE1_GRAD_PATH"
fi

# --- Phase 2 experiments ---
run_phase2() {
  local EXP_NAME="$1"
  local GPU="$2"
  local CACHE_PATH="$3"
  local BG_PATH="$4"
  local SCRIPT="$5"
  shift 5
  local EXTRA_ENV=("$@")

  log_result "=== EXP: $EXP_NAME (GPU $GPU) ==="
  T0=$(date +%s)
  env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    PHASE1_CACHE_PATH="$CACHE_PATH" \
    BG_STATS_PATH="$BG_PATH" \
    EPOCHS="$EPOCHS" \
    MAX_STEPS="$MAX_STEPS" \
    TOP_T="$TOP_T" \
    GRANULARITY="$GRANULARITY" \
    GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
    RUN_NAME="$EXP_NAME" \
    "${EXTRA_ENV[@]}" \
    bash "$SCRIPT" 2>&1 | tee "$RESULTS_DIR/${EXP_NAME}.log"
  T1=$(date +%s)
  log_result "$EXP_NAME wall-clock: $((T1-T0))s"
  find "$CARTRIDGES_OUTPUT_DIR" -path "*${EXP_NAME}*" -name "cache_last.pt" 2>/dev/null | tail -1
}

if [ -n "$PHASE1_GRAD_PATH" ]; then
  # EXP-BASE: gradient sparse Phase 2 on GPU 0
  run_phase2 "exp_base_grad_sparse" 0 "$PHASE1_GRAD_PATH" \
    "$(dirname "$PHASE1_GRAD_PATH")/bg_stats.pt" \
    "$SCRIPT_DIR/../core/train_continual_sparse.sh" \
    NUM_GPUS=1 DISTRIBUTED_BACKEND=gloo MOMENTUM_MASKING=freeze &
  BASE_PID=$!
fi

if [ -n "$PHASE1_GRAD_PATH" ]; then
  # EXP-AM2: gradient Phase 1 + AM Phase 2 on GPU 1
  run_phase2 "exp_am2_grad_p1" 1 "$PHASE1_GRAD_PATH" \
    "$(dirname "$PHASE1_GRAD_PATH")/bg_stats.pt" \
    "$SCRIPT_DIR/../core/train_continual_am_sparse.sh" \
    TARGET_MODE=self USE_IDF=1 &
  AM2_PID=$!
fi

if [ -n "$PHASE1_GRAD_PATH" ]; then
  # EXP-AM-noIDF: gradient Phase 1 + AM Phase 2 without IDF on GPU 1
  wait ${AM2_PID:-} 2>/dev/null || true
  run_phase2 "exp_am_noidf" 1 "$PHASE1_GRAD_PATH" \
    "$(dirname "$PHASE1_GRAD_PATH")/bg_stats.pt" \
    "$SCRIPT_DIR/../core/train_continual_am_sparse.sh" \
    TARGET_MODE=self USE_IDF=0
fi

if [ -n "$PHASE1_AM_PATH" ]; then
  # EXP-AM1+2: full AM pipeline on GPU 1 (after AM2 if concurrent)
  wait ${AM2_PID:-} 2>/dev/null || true
  run_phase2 "exp_am1plus2_full" 1 "$PHASE1_AM_PATH" \
    "$(dirname "$PHASE1_AM_PATH")/bg_stats.pt" \
    "$SCRIPT_DIR/../core/train_continual_am_sparse.sh" \
    TARGET_MODE=self USE_IDF=1
fi

wait ${BASE_PID:-} 2>/dev/null || true

# --- Eval all checkpoints ---
log_result "=== Running perplexity evals ==="
for CKPT in $(find "$CARTRIDGES_OUTPUT_DIR" -path "*exp_*" -name "cache_last.pt" 2>/dev/null); do
  EXP_DIR=$(basename "$(dirname "$CKPT")")
  for EVAL in QA MT; do
    EVAL_PATH="$CARTRIDGES_DIR/data/qasper/eval/qasper_eval_${EVAL}.parquet"
    if [ -f "$EVAL_PATH" ]; then
      LOSS=$(CUDA_VISIBLE_DEVICES=0 CHECKPOINT_PATH="$CKPT" \
        EVAL_DATA_PATH="$EVAL_PATH" WANDB_DISABLED=1 \
        MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
        python "$CARTRIDGES_DIR/examples/qasper2/train/eval_forgetting.py" 2>&1 \
        | grep "Eval loss" | tail -1 || echo "eval_failed")
      log_result "$EXP_DIR $EVAL: $LOSS"
    fi
  done
done

log_result "=== Experiment suite complete ==="
log_result "Results: $RESULTS_DIR/results.log"
