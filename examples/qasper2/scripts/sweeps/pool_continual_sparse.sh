#!/usr/bin/env bash
# pool_continual_sparse.sh — submit a list of TOP_T values, each as a 2-GPU
# DDP run, with at most floor(NUM_VISIBLE_GPUS / GPUS_PER_JOB) running
# concurrently. New jobs launch automatically as previous ones finish and
# release their GPU flocks.
#
# USAGE
# -----
#   bash pool_continual_sparse.sh 256 512
#   nohup bash pool_continual_sparse.sh 256 512 > queue.log 2>&1 & disown
#
#   # Override per-job env (forwarded to train_continual_sparse.sh):
#   MOMENTUM_MASKING=hard FREEZE_KEYS=0 \
#     bash pool_continual_sparse.sh 64 128 256 512
#
# ENV KNOBS
# ---------
#   GPUS_PER_JOB     GPUs per training job (default 2; matches existing DDP setup)
#   POLL_INTERVAL    seconds between free-GPU checks (default 15)
#   LAUNCH_STAGGER   seconds between consecutive launches once a slot opens
#                    (default 20). A small stagger keeps two simultaneous
#                    DDP bootstraps from racing each other on edge cases.
#   GPU_LOCK_DIR     where train_continual_sparse.sh keeps GPU flocks
#                    (default /tmp/gpu_locks_${USER}/)
#   QUEUE_LOG_DIR    where to write per-job stdout/stderr
#                    (default ./queue_logs/)
#   RUN_NAME_PREFIX  prefix for auto-generated W&B run names
#                    (default qasper_phase2_TOP_T)
#
# IMPLEMENTATION NOTES
# --------------------
# We never grab GPU locks ourselves — we only *probe* whether N GPUs are
# claimable, then immediately release the probe. The actual atomic claim is
# done inside train_continual_sparse.sh on launch, so two pool processes
# (or a pool + manual invocation) can't race-claim the same pair.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_continual_sparse.sh"

GPUS_PER_JOB="${GPUS_PER_JOB:-2}"
POLL_INTERVAL="${POLL_INTERVAL:-15}"
LAUNCH_STAGGER="${LAUNCH_STAGGER:-20}"
GPU_LOCK_DIR="${GPU_LOCK_DIR:-/tmp/gpu_locks_${USER:-$(id -un)}}"
QUEUE_LOG_DIR="${QUEUE_LOG_DIR:-$SCRIPT_DIR/queue_logs}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-qasper_phase2_TOP_T}"

mkdir -p "$QUEUE_LOG_DIR" "$GPU_LOCK_DIR"

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <TOP_T> [<TOP_T> ...]" >&2
  echo "Example: $0 256 512" >&2
  exit 1
fi

if [ ! -x "$TRAIN_SCRIPT" ] && [ ! -r "$TRAIN_SCRIPT" ]; then
  echo "Error: cannot find inner script at $TRAIN_SCRIPT" >&2
  exit 1
fi

VALUES=("$@")
TS="$(date +%Y%m%d-%H%M%S)"

ts() { date +'%H:%M:%S'; }

# Count idle GPUs by *probing* each lockfile with a non-blocking flock then
# releasing it. We treat a GPU index as free if either the lockfile doesn't
# exist (the inner script has never run) or flock succeeds (lock available).
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

echo "=========================================="
echo "pool_continual_sparse"
echo "  values:        ${VALUES[*]}"
echo "  GPUs per job:  $GPUS_PER_JOB"
echo "  log dir:       $QUEUE_LOG_DIR"
echo "  poll interval: ${POLL_INTERVAL}s"
echo "  launch stagger: ${LAUNCH_STAGGER}s"
echo "  started:       $(date)"
echo "=========================================="

PIDS=()
LABELS=()

for top_t in "${VALUES[@]}"; do
  echo "[$(ts)] next: TOP_T=$top_t"
  wait_for_free_gpus "$GPUS_PER_JOB"

  log="$QUEUE_LOG_DIR/pool_${TS}_top${top_t}.log"
  run_name="${RUN_NAME_PREFIX}_${top_t}_${TS}"

  # Launch in a subshell so its env doesn't leak between iterations.
  # The inner script grabs the actual GPU flocks atomically.
  (
    TOP_T="$top_t" RUN_NAME="$run_name" \
      bash "$TRAIN_SCRIPT"
  ) >"$log" 2>&1 &
  pid=$!
  PIDS+=("$pid")
  LABELS+=("TOP_T=$top_t (pid=$pid)")
  echo "[$(ts)] launched TOP_T=$top_t  pid=$pid  log=$log"

  # Brief stagger so two DDP bootstraps don't fire in the same millisecond.
  # Cheap insurance — gloo doesn't race the way NCCL did, but no reason to
  # tempt fate.
  if [ "$LAUNCH_STAGGER" -gt 0 ]; then
    sleep "$LAUNCH_STAGGER"
  fi
done

echo ""
echo "[$(ts)] all ${#VALUES[@]} jobs submitted; now waiting for completion..."
echo "  (check progress: ls -lat $QUEUE_LOG_DIR/  ;  tail -f $QUEUE_LOG_DIR/pool_${TS}_top*.log)"

# Wait for each job and report exit status, but don't fail-fast on one broken
# job — the others should still complete.
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
echo "[$(ts)] pool complete (overall RC=$RC)"
exit "$RC"
