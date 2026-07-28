#!/usr/bin/env bash
# Sweep continual_sparse over {per_layer, per_head} x top_t ∈ {64,128,256,512}.
#
# Layout:
#   - 8 jobs total, split across 2 parallel workers
#       Worker A: GPUs 0,1   (4 sequential jobs)
#       Worker B: GPUs 2,3   (4 sequential jobs)
#   - All jobs share fixed settings: GLOBAL_BATCH_SIZE=32, MOMENTUM_MASKING=freeze,
#     FREEZE_KEYS=0 (key+value updated), Qwen3-4B-Instruct, qasper MT data.
#   - Granularity-specific Phase-1 cache and bg_stats paths (see PER_HEAD_*/PER_LAYER_*
#     below). Edit those if the source runs change.
#   - Each torchrun call passes pydrantic CLI override `script_id=<descriptive name>`
#     so the local output dir is "<time>-continual_sparse_qwen_<gran>_top-<T>_key-value_all-reduce/<uuid>".
#   - WandB group is set via WANDB_RUN_GROUP env var (wandb.init(group=None) falls
#     back to it; verified in this repo's prepare_wandb).
#
# Usage:
#   # Foreground, see logs in this terminal (handy for a smoke check):
#   bash sweep_continual_sparse_granularity.sh
#
#   # Fully detached — survives logout, ctrl-C, closing the laptop, etc:
#   bash sweep_continual_sparse_granularity.sh --detach
#   # then:   tail -f <printed log dir>/sweep.log
#
# Stopping a detached sweep:
#   ps -fU "$USER" | grep sweep_continual_sparse_granularity
#   kill <pid_of_sweep_script>           # SIGTERM propagates to workers + torchrun
#
# Re-running a single (granularity, top_t) cell manually:
#   See run_one() below — it's a thin wrapper around train_continual_sparse.sh.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CARTRIDGES_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# ---------------------------------------------------------------------------
# Detach mode: re-exec under setsid+nohup so the sweep keeps running after
# the user logs out / closes the laptop. The child re-runs this same script
# without --detach, with SWEEP_LOG_DIR pre-populated so logs go to the same
# place the parent told the user about.
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--detach" ]]; then
  shift
  STAMP="$(date '+%Y-%m-%d-%H-%M-%S')"
  LOG_DIR="$CARTRIDGES_DIR/sweep_logs/${STAMP}-granularity-sparsity"
  mkdir -p "$LOG_DIR"
  echo "Detaching sweep to background."
  echo "  Log dir : $LOG_DIR"
  echo "  Main log: $LOG_DIR/sweep.log"
  echo "  Tail    : tail -f $LOG_DIR/sweep.log"
  SWEEP_LOG_DIR="$LOG_DIR" \
    setsid nohup bash "$0" "$@" </dev/null >>"$LOG_DIR/sweep.log" 2>&1 &
  SWEEP_PID=$!
  echo "  PID     : $SWEEP_PID"
  echo "$SWEEP_PID" > "$LOG_DIR/sweep.pid"
  exit 0
fi

# ---------------------------------------------------------------------------
# Foreground / detached-child path.
# ---------------------------------------------------------------------------
STAMP="$(date '+%Y-%m-%d-%H-%M-%S')"
LOG_DIR="${SWEEP_LOG_DIR:-$CARTRIDGES_DIR/sweep_logs/${STAMP}-granularity-sparsity}"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%F %T')] $*"; }

# ---- Fixed paths (granularity-specific) -----------------------------------
# Per-head Phase-1 outputs (from train_initial_sparse.sh with GRANULARITY=per_head).
PER_HEAD_CACHE="/localhome/local-triv/gated-continual-cartridges/outputs/2026-06-21-16-58-50-initial_sparse_qwen_qasper_per-head_all-reduce/951e1f20-b6dc-4794-87b7-0b1b544858b1/cache-step621.pt"
PER_HEAD_BG_STATS="/localhome/local-triv/gated-continual-cartridges/outputs/2026-06-21-16-58-50-initial_sparse_qwen_qasper_per-head_all-reduce/951e1f20-b6dc-4794-87b7-0b1b544858b1/bg_stats.pt"

# Per-layer Phase-1 outputs (from train_initial_sparse.sh with GRANULARITY=per_layer).
PER_LAYER_CACHE="/localhome/local-triv/gated-continual-cartridges/outputs/2026-06-21-16-59-14-initial_sparse_qwen_qasper_per-layer_all-reduce/9bf64e0e-eb3c-49ca-b9a2-e200fdb89e18/cache-step621.pt"
PER_LAYER_BG_STATS="/localhome/local-triv/gated-continual-cartridges/outputs/2026-06-21-16-59-14-initial_sparse_qwen_qasper_per-layer_all-reduce/9bf64e0e-eb3c-49ca-b9a2-e200fdb89e18/bg_stats.pt"

for f in "$PER_HEAD_CACHE" "$PER_HEAD_BG_STATS" "$PER_LAYER_CACHE" "$PER_LAYER_BG_STATS"; do
  if [ ! -f "$f" ]; then
    log "ERROR: required input not found: $f"
    exit 1
  fi
done

# ---- Fixed knobs ----------------------------------------------------------
# WandB group name. Matches the existing group at
# https://wandb.ai/vqtri-purdue-university/SEACrowd/groups/qasper%20-%20[qwen]%20-%20granularity%20-%20training
WANDB_GROUP_NAME="qasper - [qwen] - granularity - training"

GRANULARITIES=(per_layer per_head)
TOP_T_VALUES=(64 128 256 512)

# ---- Build job list and split across the two GPU pairs --------------------
JOBS=()
for gran in "${GRANULARITIES[@]}"; do
  for top_t in "${TOP_T_VALUES[@]}"; do
    JOBS+=("${gran}:${top_t}")
  done
done

QUEUE_A=()  # GPUs 0,1
QUEUE_B=()  # GPUs 2,3
for i in "${!JOBS[@]}"; do
  if (( i % 2 == 0 )); then
    QUEUE_A+=("${JOBS[$i]}")
  else
    QUEUE_B+=("${JOBS[$i]}")
  fi
done

# ---- Single-job runner ----------------------------------------------------
# Resolves granularity-specific paths, sets env, picks a unique MASTER_PORT,
# and passes `script_id=<run name>` as a pydrantic CLI override so the local
# folder is "<time>-<script_id>/<uuid>".
run_one() {
  local gpus="$1"     # e.g. "0,1"
  local job="$2"      # e.g. "per_head:128"
  local granularity="${job%%:*}"
  local top_t="${job##*:}"
  local gran_dash="${granularity//_/-}"
  local script_id="continual_sparse_qwen_${gran_dash}_top-${top_t}_key-value_all-reduce"
  local first_gpu="${gpus%%,*}"
  local master_port=$((29500 + 10#$first_gpu))
  local run_log="$LOG_DIR/${script_id}_gpus-${gpus//,/_}.log"

  local cache bg_stats
  case "$granularity" in
    per_head)  cache="$PER_HEAD_CACHE";  bg_stats="$PER_HEAD_BG_STATS"  ;;
    per_layer) cache="$PER_LAYER_CACHE"; bg_stats="$PER_LAYER_BG_STATS" ;;
    *) log "ERROR: unknown granularity '$granularity'"; return 2 ;;
  esac

  log "START gpus=$gpus  gran=$granularity  top_t=$top_t  port=$master_port"
  log "  cache    : $cache"
  log "  bg_stats : $bg_stats"
  log "  run log  : $run_log"

  CUDA_VISIBLE_DEVICES="$gpus" \
  NUM_GPUS=2 \
  MASTER_PORT="$master_port" \
  GRANULARITY="$granularity" \
  TOP_T="$top_t" \
  PHASE1_CACHE_PATH="$cache" \
  BG_STATS_PATH="$bg_stats" \
  RUN_NAME="$script_id" \
  WANDB_RUN_GROUP="$WANDB_GROUP_NAME" \
  GLOBAL_BATCH_SIZE=32 \
  MAX_STEPS=550 \
  MOMENTUM_MASKING=freeze \
  FREEZE_KEYS=0 \
  GPU_LOG="$LOG_DIR/gpu_${script_id}_gpus-${gpus//,/_}.log" \
    bash "$SCRIPT_DIR/../core/train_continual_sparse.sh" \
      "script_id=$script_id" \
      >>"$run_log" 2>&1
  local rc=$?
  if (( rc == 0 )); then
    log "DONE  gpus=$gpus  gran=$granularity  top_t=$top_t  rc=0"
  else
    log "FAIL  gpus=$gpus  gran=$granularity  top_t=$top_t  rc=$rc  (continuing sweep)"
  fi
  return 0  # never let a single failure abort the whole sweep
}

# ---- Worker: walks a queue sequentially on a fixed GPU pair ---------------
worker() {
  local gpus="$1"; shift
  local -a queue=("$@")
  log "worker[$gpus] start  jobs=${#queue[@]}: ${queue[*]}"
  for job in "${queue[@]}"; do
    run_one "$gpus" "$job"
  done
  log "worker[$gpus] done"
}

log "=================================================="
log "Sweep: continual_sparse — granularity × top_t"
log "  Host         : $(hostname)"
log "  Log dir      : $LOG_DIR"
log "  WandB group  : $WANDB_GROUP_NAME"
log "  Jobs (A 0,1) : ${QUEUE_A[*]}"
log "  Jobs (B 2,3) : ${QUEUE_B[*]}"
log "=================================================="

worker "0,1" "${QUEUE_A[@]}" >>"$LOG_DIR/worker_A.log" 2>&1 &
PID_A=$!
worker "2,3" "${QUEUE_B[@]}" >>"$LOG_DIR/worker_B.log" 2>&1 &
PID_B=$!

log "worker A PID=$PID_A  (log: $LOG_DIR/worker_A.log)"
log "worker B PID=$PID_B  (log: $LOG_DIR/worker_B.log)"

# If we get SIGTERM/SIGINT, knock down both workers (and their torchrun trees).
cleanup() {
  log "received signal — terminating workers"
  for pid in "$PID_A" "$PID_B"; do
    if kill -0 "$pid" 2>/dev/null; then
      pkill -TERM -P "$pid" 2>/dev/null || true
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup INT TERM

wait "$PID_A"
RC_A=$?
wait "$PID_B"
RC_B=$?

log "=================================================="
log "Sweep complete.  worker A rc=$RC_A  worker B rc=$RC_B"
log "Logs: $LOG_DIR"
log "=================================================="
