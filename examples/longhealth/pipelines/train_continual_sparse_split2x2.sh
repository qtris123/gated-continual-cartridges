# Launch two parallel Phase 2 continual_sparse runs in a 2x2 GPU split.
#
#   Run A -> CUDA_VISIBLE_DEVICES=0,1 (MASTER_PORT default 29507)
#   Run B -> CUDA_VISIBLE_DEVICES=2,3 (MASTER_PORT default 29508)
#
# Each side calls train_continual_sparse.sh, which:
#   - auto-detects NUM_GPUS from CUDA_VISIBLE_DEVICES (so 2 here),
#   - defaults DISTRIBUTED_BACKEND=gloo (NCCL deadlocks on this host's PCIe
#     topology — see notes/2026-06-20-ddp-gloo-flock-pool.md),
#   - filters its GPU monitor to its own devices,
#   - writes a unique GPU usage log per side.
#
# Per-run overrides
# -----------------
# Any var that train_continual_sparse.sh accepts can be set globally (applies
# to both sides) or, prefixed with A_ / B_, only to one side. Per-run
# overrides win over the shared default. Examples:
#
#   # Same config on both sides (sanity check / variance):
#   ./train_continual_sparse_split2x2.sh
#
#   # Sweep TOP_T on the two sides:
#   A_TOP_T=128 B_TOP_T=512 ./train_continual_sparse_split2x2.sh
#
#   # Different masking strategies, shared LR:
#   LR=2e-2 \
#   A_MOMENTUM_MASKING=freeze \
#   B_MOMENTUM_MASKING=hard \
#     ./train_continual_sparse_split2x2.sh
#
# To watch progress:
#   tail -f $LOG_DIR/run_A_*.log
#   tail -f $LOG_DIR/run_B_*.log

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_SCRIPT="$SCRIPT_DIR/train_continual_sparse.sh"

if [ ! -x "$INNER_SCRIPT" ] && [ ! -r "$INNER_SCRIPT" ]; then
  echo "Error: cannot find inner script at $INNER_SCRIPT" >&2
  exit 1
fi

# GPU partitioning + ports — override on the command line if the host has a
# different topology.
GPUS_A="${GPUS_A:-0,1}"
GPUS_B="${GPUS_B:-2,3}"
MASTER_PORT_A="${MASTER_PORT_A:-29507}"
MASTER_PORT_B="${MASTER_PORT_B:-29508}"

# Variables that can be overridden per-run via A_<NAME> / B_<NAME>. Anything
# already in the environment without a prefix is forwarded to both sides.
PER_RUN_VARS=(
  MODEL_NAME
  PHASE1_CACHE_PATH
  SYNTH_DATA_PATH
  EVAL_DATA_PATH
  BG_STATS_PATH
  NUM_TOKENS
  EPOCHS
  LR
  GLOBAL_BATCH_SIZE
  PATIENT_IDS
  EVAL_EVERY_N_STEPS
  SAVE_EVERY_N_STEPS
  TOP_T
  MOMENTUM_MASKING
  FREEZE_KEYS
  GRANULARITY
  IDF_TOP_K
  IDF_SMOOTHING
  RUN_NAME
  DISTRIBUTED_BACKEND
  CARTRIDGES_DIR
  CARTRIDGES_OUTPUT_DIR
)

# Build a NAME=VALUE list for one side. A_<VAR>/B_<VAR> wins over <VAR>.
build_env_array() {
  local prefix="$1"  # "A_" or "B_"
  RUN_ENV=()
  local var pvar
  for var in "${PER_RUN_VARS[@]}"; do
    pvar="${prefix}${var}"
    if [ -n "${!pvar+x}" ]; then
      RUN_ENV+=("$var=${!pvar}")
    elif [ -n "${!var+x}" ]; then
      RUN_ENV+=("$var=${!var}")
    fi
  done
}

LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/split2x2_logs}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
LOG_A="$LOG_DIR/run_A_${TS}_gpus${GPUS_A//,/_}.log"
LOG_B="$LOG_DIR/run_B_${TS}_gpus${GPUS_B//,/_}.log"

echo "=========================================="
echo "LongHealth Phase 2 — 2x2 split launcher"
echo "=========================================="
echo "Host:         $(hostname)"
echo "Started at:   $(date)"
echo "Inner script: $INNER_SCRIPT"
echo "Log dir:      $LOG_DIR"
echo ""
echo "Run A: GPUs=$GPUS_A  MASTER_PORT=$MASTER_PORT_A  log=$LOG_A"
echo "Run B: GPUs=$GPUS_B  MASTER_PORT=$MASTER_PORT_B  log=$LOG_B"
echo ""

launch_side() {
  local label="$1" gpus="$2" port="$3" log="$4" prefix="$5"
  build_env_array "$prefix"
  # Header for the per-run log so it's clear what this run is.
  {
    echo "=========================================="
    echo "Run $label  (split2x2 wrapper)"
    echo "GPUs=$gpus  MASTER_PORT=$port  prefix=$prefix"
    echo "Started at: $(date)"
    echo "Per-run env overrides:"
    printf '  %s\n' "${RUN_ENV[@]}"
    echo "=========================================="
  } >"$log"
  # NOTE: we deliberately avoid `env "${RUN_ENV[@]}" bash …` here because some
  # users have a `~/.local/bin/env` PATH-manipulation shim earlier in $PATH that
  # silently eats its arguments and exits 0, making the inner script never run.
  # Using a subshell + bash's builtin `export` is shim-immune and portable.
  (
    export CUDA_VISIBLE_DEVICES="$gpus"
    export MASTER_PORT="$port"
    export NUM_GPUS=2
    local kv
    for kv in "${RUN_ENV[@]}"; do
      export "$kv"
    done
    exec bash "$INNER_SCRIPT"
  ) >>"$log" 2>&1 &
}

launch_side A "$GPUS_A" "$MASTER_PORT_A" "$LOG_A" "A_"
PID_A=$!
launch_side B "$GPUS_B" "$MASTER_PORT_B" "$LOG_B" "B_"
PID_B=$!

echo "Launched: PID_A=$PID_A  PID_B=$PID_B"
echo "Tail with:"
echo "  tail -f $LOG_A"
echo "  tail -f $LOG_B"
echo ""

# Best-effort cleanup: TERM bash subshell + its descendants (torchrun, the GPU
# monitor, Python workers), then KILL after a short grace period.
cleanup() {
  echo ""
  echo "[split2x2] Caught signal — terminating both runs..." >&2
  local pid
  for pid in "$PID_A" "$PID_B"; do
    if kill -0 "$pid" 2>/dev/null; then
      pkill -TERM -P "$pid" 2>/dev/null || true
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  sleep 5
  for pid in "$PID_A" "$PID_B"; do
    if kill -0 "$pid" 2>/dev/null; then
      pkill -KILL -P "$pid" 2>/dev/null || true
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup INT TERM

wait "$PID_A"
RC_A=$?
wait "$PID_B"
RC_B=$?

echo ""
echo "Run A exit code: $RC_A"
echo "Run B exit code: $RC_B"
echo "Logs:"
echo "  A: $LOG_A"
echo "  B: $LOG_B"

if [ "$RC_A" -ne 0 ] || [ "$RC_B" -ne 0 ]; then
  exit 1
fi
