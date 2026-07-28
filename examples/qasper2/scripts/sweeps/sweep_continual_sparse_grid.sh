#!/usr/bin/env bash
# sweep_continual_sparse_grid.sh — Phase 2 sweep over (GRANULARITY x TOP_T)
# for Llama-3.2-3B-Instruct @ NUM_TOKENS=512, IDF_TOP_K=64.
#
# Default grid (8 experiments):
#   granularity in {per_head, per_layer}
#   top_t       in {32, 64, 128, 256}
#
# Override the grid via env to run any subset:
#   GRANULARITIES="global"           bash sweep_continual_sparse_grid.sh   # 4 expts
#   GRANULARITIES="global per_layer" TOP_T_VALUES="64 128" bash ...        # 4 expts
#   TOP_T_VALUES="256 512"           bash sweep_continual_sparse_grid.sh   # 4 expts
#
# Supported granularities: per_head, per_layer, global. Each needs a
# Phase 1 cache + bg_stats pair (paths set below; overridable per granularity).
#
# Each job runs on 1 GPU. We keep up to MAX_CONCURRENT (default 4) running at
# all times, launching the next one as soon as a GPU lock frees up. The actual
# atomic GPU claim is done inside train_continual_sparse.sh (flock under
# $GPU_LOCK_DIR), so we never race over the same device.
#
# USAGE
# -----
#   bash sweep_continual_sparse_grid.sh                  # default 8-cell grid
#   GRANULARITIES=global bash sweep_continual_sparse_grid.sh   # global x 4 top_t
#
#   # Unattended:
#   nohup bash sweep_continual_sparse_grid.sh > sweep_queue.log 2>&1 & disown
#
# ENV KNOBS (all optional; defaults match the recipe checked in this file)
# ------------------------------------------------------------------------
#   GRANULARITIES    space-separated list (default "per_head per_layer")
#   TOP_T_VALUES     space-separated list (default "32 64 128 256")
#   MAX_CONCURRENT   max concurrent 1-GPU jobs (default 4 — matches box)
#   POLL_INTERVAL    seconds between free-GPU probes (default 15)
#   LAUNCH_STAGGER   seconds between consecutive launches once a slot opens
#                    (default 15) — avoids two DDP bootstraps racing
#   GPU_LOCK_DIR     dir where train_continual_sparse.sh keeps GPU flocks
#                    (default /tmp/gpu_locks_${USER}/)
#   QUEUE_LOG_DIR    per-job stdout/stderr (default ./queue_logs/)
#
#   Any of the training constants further down can also be overridden the
#   same way (they all use ${VAR:-default} expansion), e.g.
#     LR=1e-2 MAX_STEPS=400 bash sweep_continual_sparse_grid.sh
#
# IMPLEMENTATION NOTES
# --------------------
# We never grab GPU locks ourselves — we only probe them with `flock -n`
# (immediately released) to decide when to launch the next job. The atomic
# claim happens in train_continual_sparse.sh on launch, so two concurrent
# pool processes (or a pool + manual run) can't double-claim a device.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_continual_sparse.sh"
CARTRIDGES_DIR="${CARTRIDGES_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

# ---- pool/runtime knobs ------------------------------------------------------
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
POLL_INTERVAL="${POLL_INTERVAL:-15}"
LAUNCH_STAGGER="${LAUNCH_STAGGER:-15}"
GPU_LOCK_DIR="${GPU_LOCK_DIR:-/tmp/gpu_locks_${USER:-$(id -un)}}"
QUEUE_LOG_DIR="${QUEUE_LOG_DIR:-$SCRIPT_DIR/queue_logs}"

# ---- Phase 1 artifacts (granularity -> cache + bg_stats) ---------------------
# Verified existing on disk for Llama-3.2-3B-Instruct @ NUM_TOKENS=512.
# We point at the explicit cache-stepNNN.pt files because the cache_last.pt
# symlinks in these dirs are broken (parent dir was renamed after Phase 1).
PER_LAYER_DIR="$CARTRIDGES_DIR/outputs/2026-06-22-09-44-22-initial_sparse-per-layer/02870162-68d4-4063-b4ef-05e2ec5817b6"
PER_LAYER_CACHE="${PER_LAYER_CACHE:-$PER_LAYER_DIR/cache-step534.pt}"
PER_LAYER_BG="${PER_LAYER_BG:-$PER_LAYER_DIR/bg_stats.pt}"

PER_HEAD_DIR="$CARTRIDGES_DIR/outputs/2026-06-22-09-51-45-initial_sparse-per-head/1c7aa8e2-dbde-420d-877f-d8b9825ced2c"
PER_HEAD_CACHE="${PER_HEAD_CACHE:-$PER_HEAD_DIR/cache-step534.pt}"
PER_HEAD_BG="${PER_HEAD_BG:-$PER_HEAD_DIR/bg_stats.pt}"

GLOBAL_DIR="$CARTRIDGES_DIR/outputs/2026-06-22-14-34-24-initial_sparse-global/43f97f6a-e2f8-456c-a5b0-af7a77ddc3f8"
GLOBAL_CACHE="${GLOBAL_CACHE:-$GLOBAL_DIR/cache-step535.pt}"
GLOBAL_BG="${GLOBAL_BG:-$GLOBAL_DIR/bg_stats.pt}"

# ---- Phase 2 data (Llama-compatible — no `qwen_` prefix) ---------------------
SYNTH_DATA="${SYNTH_DATA_PATH:-$CARTRIDGES_DIR/data/qasper/train/qasper_MT_task_8192.parquet}"
EVAL_DATA="${EVAL_DATA_PATH:-$CARTRIDGES_DIR/examples/qasper2/qasper_eval_MT.parquet}"

# ---- constants forwarded to every job ---------------------------------------
# All use ${VAR:-default} so pre-exported env wins.
export MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}"
export NUM_TOKENS="${NUM_TOKENS:-512}"
export IDF_TOP_K="${IDF_TOP_K:-64}"
export IDF_SMOOTHING="${IDF_SMOOTHING:-1.0}"
export LR="${LR:-2e-2}"
export EPOCHS="${EPOCHS:-10}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
export MAX_STEPS="${MAX_STEPS:-500}"      # tuned for GLOBAL_BATCH_SIZE=32 + EPOCHS=10
export MOMENTUM_MASKING="${MOMENTUM_MASKING:-freeze}"
export FREEZE_KEYS="${FREEZE_KEYS:-0}"
export NUM_GPUS="${NUM_GPUS:-1}"          # 1 GPU per job — keep this 1
export EVAL_EVERY_N_STEPS="${EVAL_EVERY_N_STEPS:-8}"
export SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-256}"
export DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-gloo}"
export SYNTH_DATA_PATH="$SYNTH_DATA"
export EVAL_DATA_PATH="$EVAL_DATA"

# ---- the (granularity, top_t) grid -------------------------------------------
# Build the grid dynamically from env so we can run any subset of granularities
# and top_t values without editing this file. Defaults preserve the original
# 8-cell (per_head, per_layer) x (32, 64, 128, 256) sweep.
GRANULARITIES="${GRANULARITIES:-per_head per_layer}"
TOP_T_VALUES="${TOP_T_VALUES:-32 64 128 256}"

# Resolve {cache, bg_stats} for a single granularity. Echoes
# "<cache_path>\t<bg_path>", or exits 1 if granularity is unsupported.
resolve_phase1_paths() {
  case "$1" in
    per_head)  printf '%s\t%s\n' "$PER_HEAD_CACHE"  "$PER_HEAD_BG"  ;;
    per_layer) printf '%s\t%s\n' "$PER_LAYER_CACHE" "$PER_LAYER_BG" ;;
    global)    printf '%s\t%s\n' "$GLOBAL_CACHE"    "$GLOBAL_BG"    ;;
    *)
      echo "Error: unsupported granularity '$1' (use per_head|per_layer|global)" >&2
      exit 1
      ;;
  esac
}

GRID=()
for _gran in $GRANULARITIES; do
  for _t in $TOP_T_VALUES; do
    GRID+=("$_gran $_t")
  done
done

if [ "${#GRID[@]}" -eq 0 ]; then
  echo "Error: empty grid (check GRANULARITIES / TOP_T_VALUES)" >&2
  exit 1
fi

# ---- sanity check inputs before launching anything --------------------------
# Only validate Phase 1 artifacts for the granularities we're actually using,
# so e.g. a global-only sweep doesn't fail if per-head bg_stats is missing.
REQUIRED_PATHS=("$SYNTH_DATA" "$EVAL_DATA" "$TRAIN_SCRIPT")
for _gran in $GRANULARITIES; do
  IFS=$'\t' read -r _cache _bg < <(resolve_phase1_paths "$_gran")
  REQUIRED_PATHS+=("$_cache" "$_bg")
done
for p in "${REQUIRED_PATHS[@]}"; do
  if [ ! -e "$p" ]; then
    echo "Error: missing required path: $p" >&2
    exit 1
  fi
done

mkdir -p "$QUEUE_LOG_DIR" "$GPU_LOCK_DIR"

TS="$(date +%Y%m%d-%H%M%S)"
ts() { date +'%H:%M:%S'; }

# Probe how many GPUs are currently claimable (lockfile missing OR flock -n ok).
# Never grabs the lock — releases immediately by `true` exiting under flock.
count_free_gpus() {
  local count=0 idx lock
  while read -r idx; do
    lock="$GPU_LOCK_DIR/gpu${idx}.lock"
    if [ ! -e "$lock" ]; then
      count=$((count + 1))
    elif flock -n "$lock" true 2>/dev/null; then
      count=$((count + 1))
    fi
  done < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null)
  echo "$count"
}

wait_for_free_gpus() {
  local need="$1" free
  while true; do
    free=$(count_free_gpus)
    if [ "$free" -ge "$need" ]; then
      return 0
    fi
    echo "[$(ts)]   waiting: $free/$need GPUs free"
    sleep "$POLL_INTERVAL"
  done
}

# Cap concurrency to whatever's smaller: MAX_CONCURRENT or live free-GPU count.
# We still let train_continual_sparse.sh do the atomic per-GPU claim; this is
# just a backstop so we don't pile up 8 children in 1 second when only 4 GPUs
# are actually idle.
wait_for_slot() {
  local running=0
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      running=$((running + 1))
    fi
  done
  while [ "$running" -ge "$MAX_CONCURRENT" ]; do
    # wait -n: return as soon as any tracked child exits
    if ! wait -n 2>/dev/null; then
      :  # ignore exit code; we'll re-check the wait/exit table at the end
    fi
    running=0
    for pid in "${PIDS[@]:-}"; do
      if kill -0 "$pid" 2>/dev/null; then
        running=$((running + 1))
      fi
    done
  done
}

echo "=========================================="
echo "sweep_continual_sparse_grid"
echo "  model:          $MODEL_NAME"
echo "  num_tokens:     $NUM_TOKENS"
echo "  idf_top_k:      $IDF_TOP_K"
echo "  granularities:  $GRANULARITIES"
echo "  top_t values:   $TOP_T_VALUES"
echo "  grid:           ${#GRID[@]} experiments"
echo "  max concurrent: $MAX_CONCURRENT  (1 GPU per job)"
echo "  poll interval:  ${POLL_INTERVAL}s"
echo "  launch stagger: ${LAUNCH_STAGGER}s"
echo "  log dir:        $QUEUE_LOG_DIR"
for _gran in $GRANULARITIES; do
  IFS=$'\t' read -r _cache _bg < <(resolve_phase1_paths "$_gran")
  printf "  phase1 (%-9s): %s\n" "$_gran" "$_cache"
  printf "  bg_stats (%-7s): %s\n" "$_gran" "$_bg"
done
echo "  synth data:     $SYNTH_DATA"
echo "  eval  data:     $EVAL_DATA"
echo "  started:        $(date)"
echo "=========================================="

PIDS=()
LABELS=()

for cell in "${GRID[@]}"; do
  read -r gran top_t <<<"$cell"

  IFS=$'\t' read -r cache bg < <(resolve_phase1_paths "$gran")

  echo "[$(ts)] queued: GRANULARITY=$gran TOP_T=$top_t"

  # Two backstops:
  #   1) Don't oversubscribe our own pool (wait until <MAX_CONCURRENT tracked
  #      children are alive)
  #   2) Don't launch unless NUM_GPUS lockfiles are claimable right now
  #      (matters for multi-GPU jobs; for NUM_GPUS=1 this is the original
  #       "at least 1 free" check)
  wait_for_slot
  wait_for_free_gpus "$NUM_GPUS"

  label="${gran}_top${top_t}"
  log="$QUEUE_LOG_DIR/sweep_grid_${TS}_${label}.log"
  run_name="qasper_phase2_grid_${label}_idfk${IDF_TOP_K}_${TS}"

  # Subshell isolates env per launch. train_continual_sparse.sh handles the
  # actual GPU flock + MASTER_PORT picking based on which device it claims.
  (
    GRANULARITY="$gran" \
    TOP_T="$top_t" \
    PHASE1_CACHE_PATH="$cache" \
    BG_STATS_PATH="$bg" \
    RUN_NAME="$run_name" \
    bash "$TRAIN_SCRIPT"
  ) >"$log" 2>&1 &
  pid=$!
  PIDS+=("$pid")
  LABELS+=("$label (pid=$pid)")
  echo "[$(ts)] launched $label  pid=$pid  log=$log"

  if [ "$LAUNCH_STAGGER" -gt 0 ]; then
    sleep "$LAUNCH_STAGGER"
  fi
done

echo ""
echo "[$(ts)] all ${#GRID[@]} jobs submitted; waiting for completion..."
echo "  progress: tail -f $QUEUE_LOG_DIR/sweep_grid_${TS}_*.log"

RC=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  label="${LABELS[$i]}"
  if wait "$pid"; then
    echo "[$(ts)] DONE  $label  exit=0"
  else
    rc=$?
    echo "[$(ts)] FAIL  $label  exit=$rc"
    RC=1
  fi
done

echo ""
echo "[$(ts)] sweep complete (overall RC=$RC)"
exit "$RC"
