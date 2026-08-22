#!/usr/bin/env bash
# Sweep over TOP_T sparsity values, running up to MAX_CONCURRENT experiments at once.
# Each iteration auto-claims free GPUs via the flock machinery in
# train_continual_sparse.sh, so concurrent invocations grab disjoint pairs.
#
# Usage:
#   bash examples/qasper/sweeps/sweep_top_t.sh
#
# Common overrides (all forwarded as env vars to train_continual_sparse.sh):
#   MAX_CONCURRENT=2          number of experiments to keep running at once
#                              (default 2; pair with NUM_GPUS=2 to fill a 4-GPU box)
#   NUM_GPUS=2                GPUs per experiment (default 2)
#   WANDB_GROUP="..."         cluster all iterations under one W&B group
#
# To run unattended (survives terminal close):
#   nohup bash examples/qasper/sweeps/sweep_top_t.sh \
#     > sweep_top_t_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -u  # NB: not -e — a single iter crashing should not abort the whole sweep.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/../pipelines/train_continual_sparse.sh"

TOP_T_VALUES=(32 64 128 256)
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
SWEEP_TAG="sweep_top_t_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/sweep_logs/${SWEEP_TAG}}"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "TOP_T sweep: ${TOP_T_VALUES[*]}"
echo "Max concurrent: $MAX_CONCURRENT"
echo "Per-iter logs:  $LOG_DIR"
echo "WANDB_GROUP:    ${WANDB_GROUP:-<unset>}"
echo "Started at: $(date)"
echo "Sweep PID:  $$"
echo "=========================================="
echo ""

# Track active background iterations.
active=0
declare -A pid_to_top_t   # pid -> TOP_T value, for logging

# Ensure all child jobs die if this sweep is killed (Ctrl-C / SIGTERM / nohup HUP).
trap 'echo "[sweep] caught signal — killing children"; kill 0 2>/dev/null; exit 130' INT TERM HUP

launch_iter() {
    local top_t="$1"
    local log_file="$LOG_DIR/top_t_${top_t}.log"
    echo "[sweep $(date +%H:%M:%S)] launching TOP_T=$top_t  -> $log_file"
    (
        TOP_T="$top_t" \
        RUN_NAME="" \
        bash "$TRAIN_SCRIPT" >"$log_file" 2>&1
        echo "[child] TOP_T=$top_t exited with code $? at $(date)"
    ) &
    local pid=$!
    pid_to_top_t[$pid]="$top_t"
    active=$((active + 1))
    # Stagger launches by a few seconds so the two iterations don't race
    # for the same flock + master_port slot at exactly the same instant.
    sleep 3
}

for TOP_T in "${TOP_T_VALUES[@]}"; do
    if (( active >= MAX_CONCURRENT )); then
        # Block until ANY backgrounded iteration completes, then launch the next.
        if wait -n; then
            ec=0
        else
            ec=$?
        fi
        active=$((active - 1))
        echo "[sweep $(date +%H:%M:%S)] one iter finished (exit=$ec); freeing slot"
    fi
    launch_iter "$TOP_T"
done

echo ""
echo "[sweep $(date +%H:%M:%S)] all iterations submitted; waiting for last $active to finish"
wait

echo ""
echo "=========================================="
echo "All TOP_T experiments complete: $(date)"
echo "Per-iter logs: $LOG_DIR"
echo "=========================================="
