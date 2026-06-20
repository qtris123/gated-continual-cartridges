# Phase 2: Continual qasper cartridge WITH TF-IDF sparse cache finetuning.
# Uses train/continual_sparse.py (SparseCacheFinetuningConfig enabled).
#
# Prerequisites:
#   - Run train_initial_sparse.sh first; set PHASE1_CACHE_PATH to its cache_last.pt.
#   - Optionally set BG_STATS_PATH to the same checkpoint (or a dedicated bg_stats.pt)
#     to enable IDF-weighted slot selection. Defaults to PHASE1_CACHE_PATH if unset.

set -e

# Configuration — adjust these as needed
export TORCH_CUDA_ARCH_LIST="8.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CARTRIDGES_DIR="${CARTRIDGES_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$CARTRIDGES_DIR/outputs}"
export PATH=${CUDA_HOME:+$CUDA_HOME/bin:}$PATH
export LD_LIBRARY_PATH=${CUDA_HOME:+$CUDA_HOME/lib64:}$LD_LIBRARY_PATH

# GPU selection
# -------------
# Three regimes, in priority order:
#   1) CUDA_VISIBLE_DEVICES set explicitly  -> honour it; NUM_GPUS = its cardinality
#      unless the user also set NUM_GPUS.
#   2) NUM_GPUS set, CUDA_VISIBLE_DEVICES unset  -> auto-claim that many free GPUs.
#   3) Neither set -> default to NUM_GPUS=2 and auto-claim two free GPUs.
#
# Auto-claim uses per-GPU lockfiles under /tmp/gpu_locks_${USER}/. We hold an
# flock on each claimed GPU's lockfile for the lifetime of this script, so a
# second invocation in another terminal (or a third, etc.) sees those GPUs
# as taken and picks the next free pair. Locks auto-release when this script
# exits for any reason (clean exit, Ctrl-C, OOM, kill -9 of the bash pid).
if [ -z "${NUM_GPUS:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_GPUS=$(awk -F',' '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
  else
    NUM_GPUS=2
  fi
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Error: CUDA_VISIBLE_DEVICES is unset and nvidia-smi is not on PATH." >&2
    exit 1
  fi

  GPU_LOCK_DIR="${GPU_LOCK_DIR:-/tmp/gpu_locks_${USER:-$(id -un)}}"
  mkdir -p "$GPU_LOCK_DIR"

  # Order GPUs by memory used (ascending) so flock attempts hit idle ones first.
  # The flock is what enforces mutual exclusion; the sort is just a tiebreaker.
  mapfile -t _GPU_PREF_ORDER < <(
    nvidia-smi --query-gpu=index,memory.used \
               --format=csv,noheader,nounits \
      | sort -t',' -k2 -n \
      | awk -F', *' '{print $1}'
  )

  if [ "${#_GPU_PREF_ORDER[@]}" -lt "$NUM_GPUS" ]; then
    echo "Error: only ${#_GPU_PREF_ORDER[@]} GPU(s) visible to nvidia-smi; need $NUM_GPUS." >&2
    exit 1
  fi

  # Try to claim NUM_GPUS lockfiles non-blockingly. Each successful flock holds
  # an open FD in this shell; closing the script (any exit path) releases it.
  _CLAIMED_GPUS=()
  for _gpu_idx in "${_GPU_PREF_ORDER[@]}"; do
    [ "${#_CLAIMED_GPUS[@]}" -ge "$NUM_GPUS" ] && break
    _lockfile="$GPU_LOCK_DIR/gpu${_gpu_idx}.lock"
    # `exec {_fd}>"$path"` allocates a fresh FD into $_fd; we never close it
    # explicitly on the success path, so the OS releases the flock on exit.
    exec {_fd}>"$_lockfile"
    if flock -n "$_fd"; then
      _CLAIMED_GPUS+=("$_gpu_idx")
    else
      exec {_fd}>&-
    fi
  done

  if [ "${#_CLAIMED_GPUS[@]}" -lt "$NUM_GPUS" ]; then
    echo "Error: could not lock $NUM_GPUS free GPUs (claimed: ${_CLAIMED_GPUS[*]:-none})." >&2
    echo "  Lock dir: $GPU_LOCK_DIR" >&2
    echo "  Other runs are holding the rest. Wait, or set CUDA_VISIBLE_DEVICES manually." >&2
    exit 1
  fi

  CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${_CLAIMED_GPUS[*]}")"
  export CUDA_VISIBLE_DEVICES
  echo "Auto-claimed GPUs: $CUDA_VISIBLE_DEVICES  (flocks under $GPU_LOCK_DIR)"

  # Pick a deterministic, GPU-set-specific MASTER_PORT so two concurrent runs
  # in different terminals don't have to coordinate ports either.
  if [ -z "${MASTER_PORT:-}" ]; then
    MASTER_PORT=$((29500 + 10#${_CLAIMED_GPUS[0]}))
    export MASTER_PORT
    echo "Auto-set MASTER_PORT=$MASTER_PORT"
  fi
fi
# Distributed backend.
#
# We default to **gloo** because NCCL deadlocks during DDP._sync_module_states
# on this host's PCIe topology (RTX 5880 Ada, no NVLink). Every successful
# Phase-1 / Phase-2 run on this box from 06-17 onward used gloo; every
# attempt that flipped this to nccl hung in DDP.__init__ for the full
# 600 s NCCL watchdog window before timing out.
#
# If you want to try NCCL again later (it does win throughput on hosts where
# it works), the usual escape hatch is:
#     DISTRIBUTED_BACKEND=nccl NCCL_P2P_DISABLE=1 ./train_continual_sparse.sh
# which forces NCCL through host memory instead of PCIe peer-to-peer. Verify
# with a single-run smoke test before launching a sweep.
DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-gloo}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}"
# TODO: set to the cache_last.pt produced by train_initial_sparse.sh
PHASE1_CACHE_PATH="${PHASE1_CACHE_PATH:-/localhome/local-triv/gated-continual-cartridges/outputs/qasper-initial-per-layer-all-reduce/d8103e75-4886-47a0-8af0-286ca4bec665/cache-step534.pt}"
SYNTH_DATA_PATH="${SYNTH_DATA_PATH:-/localhome/local-triv/gated-continual-cartridges/data/qasper/train/qasper_MT_task_8192_no-cartridge.parquet}"
# BG_STATS_PATH defaults to PHASE1_CACHE_PATH inside continual_sparse.py if unset
BG_STATS_PATH="${BG_STATS_PATH:-/localhome/local-triv/gated-continual-cartridges/outputs/qasper-initial-per-layer-all-reduce/d8103e75-4886-47a0-8af0-286ca4bec665/bg_stats.pt}"
NUM_TOKENS="${NUM_TOKENS:-1024}"        # must match Phase 1 cache size
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-2}"                        # adam: 2e-2
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"   # 4 GPU: 16/GPU; 2 GPU: 32/GPU
# Cosine-schedule horizon. Rule of thumb on this dataset @ EPOCHS=10:
#   GLOBAL_BATCH_SIZE=64 -> MAX_STEPS=250
#   GLOBAL_BATCH_SIZE=32 -> MAX_STEPS=500
# If you change GLOBAL_BATCH_SIZE or EPOCHS, scale MAX_STEPS to match the actual
# number of optimizer steps so the LR doesn't sit at its floor mid-run.
MAX_STEPS="${MAX_STEPS:-250}"
MASTER_PORT="${MASTER_PORT:-29507}"
# Optional: set to a parquet to log perplexity in W&B during training.
EVAL_DATA_PATH="${EVAL_DATA_PATH:-$CARTRIDGES_DIR/examples/qasper2/qasper_eval_MT.parquet}"
EVAL_EVERY_N_STEPS="${EVAL_EVERY_N_STEPS:-8}"   # halved from 15 to match doubled batch size
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-256}"
TOP_T="${TOP_T:-64}" # 128 256 512
MOMENTUM_MASKING="${MOMENTUM_MASKING:-freeze}"  # soft | hard | freeze | decouple
FREEZE_KEYS="${FREEZE_KEYS:-1}"                 # 1=freeze keys, 0=update keys
GRANULARITY="${GRANULARITY:-per_layer}"          # must match Phase 1 GRANULARITY
IDF_TOP_K="${IDF_TOP_K:-128}"                   # top-k positions per bg batch for df
IDF_SMOOTHING="${IDF_SMOOTHING:-1.0}"           # Laplace smoothing for IDF denominator
RUN_NAME="${RUN_NAME:-qasper_phase2_sparse_${MOMENTUM_MASKING}_value-only_adam_top-${TOP_T}_${GRANULARITY}_lr${LR}_all-reduce}"

echo "=========================================="
echo "Qasper Phase 2 — TF-IDF Sparse Continual Cartridge"
echo "=========================================="
echo "Host=$(hostname)"
echo "Started at: $(date)"
echo ""

# Load modules only when running under a module system (e.g. SLURM cluster)
if command -v module >/dev/null 2>&1; then
  echo "Loading GCC 11.4..."
  module load gcc/11.4.1 2>/dev/null || true
  echo "GCC version: $(gcc --version | head -1)"
  echo "Loading CUDA module..."
  module load cuda/12.1.0 2>/dev/null || true
  echo "CUDA version: $(nvcc --version | grep release 2>/dev/null || echo 'nvcc not found')"
  echo ""
fi

echo "=== GPU configuration ($(hostname)) ==="
echo "CUDA_HOME=${CUDA_HOME:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "NUM_GPUS=$NUM_GPUS"
echo "DISTRIBUTED_BACKEND=$DISTRIBUTED_BACKEND"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L
  echo "--- nvidia-smi (full snapshot) ---"
  nvidia-smi
else
  echo "(nvidia-smi not found — no NVIDIA driver in PATH)"
fi

# GPU monitor: include the visible-device list and master port in the filename
# so concurrent runs (e.g. a 2x2 split) don't clobber each other's logs.
_GPU_LOG_TAG="${CUDA_VISIBLE_DEVICES:-all}"
_GPU_LOG_TAG="${_GPU_LOG_TAG//,/_}"
GPU_LOG="${GPU_LOG:-train_continual_sparse_gpu_usage_${_GPU_LOG_TAG}_${MASTER_PORT}.log}"
mkdir -p "$(dirname "$GPU_LOG")"
echo "=== GPU monitor started: $(date) on $(hostname) (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}) ===" >"$GPU_LOG"
# Limit nvidia-smi to this run's devices when possible. --id accepts the same
# global indices that CUDA_VISIBLE_DEVICES uses.
NVSMI_ID_OPTS=()
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  NVSMI_ID_OPTS=(--id="$CUDA_VISIBLE_DEVICES")
fi
(
  while true; do
    echo "index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw"
    date "+[%F %T]"
    nvidia-smi "${NVSMI_ID_OPTS[@]}" --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
      --format=csv,noheader,nounits 2>/dev/null || echo "(nvidia-smi query failed)"
    echo "-----"
    sleep 5
  done
) >>"$GPU_LOG" 2>&1 &
GPU_MON_PID=$!

cleanup() {
  echo ""
  echo "Cleaning up..."
  if [ -n "${GPU_MON_PID:-}" ]; then
    kill "$GPU_MON_PID" 2>/dev/null || true
    wait "$GPU_MON_PID" 2>/dev/null || true
    echo "=== GPU monitor stopped: $(date) ===" >>"${GPU_LOG:-/dev/null}"
  fi
  echo "Job finished at: $(date)"
}
trap cleanup EXIT

if [ -z "$PHASE1_CACHE_PATH" ] || [ "$PHASE1_CACHE_PATH" = "PLACEHOLDER_set_PHASE1_CACHE_PATH" ]; then
  echo "Error: PHASE1_CACHE_PATH is required — set it to the cache_last.pt from Phase 1"
  exit 1
fi

if [ -z "$SYNTH_DATA_PATH" ]; then
  echo "Error: SYNTH_DATA_PATH is required"
  exit 1
fi

# Activate the repo virtual environment (pydrantic + cartridges live in .venv)
echo "Activating .venv..."
source "$CARTRIDGES_DIR/.venv/bin/activate"
echo "Python: $(which python3)"
echo ""

echo "=== Training continual (sparse TF-IDF) cartridge ==="
echo "Model:           $MODEL_NAME"
echo "GPUs:            $NUM_GPUS  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
echo "Backend:         $DISTRIBUTED_BACKEND"
echo "Master port:     $MASTER_PORT"
echo "Phase1 cache:    $PHASE1_CACHE_PATH"
echo "BG stats:        ${BG_STATS_PATH:-<same as PHASE1_CACHE_PATH>}"
echo "Train data:      $SYNTH_DATA_PATH"
echo "Eval data:       $EVAL_DATA_PATH"
echo "Tokens:          $NUM_TOKENS"
echo "Epochs:          $EPOCHS"
echo "GLOBAL_BATCH:    $GLOBAL_BATCH_SIZE"
echo "MAX_STEPS:       $MAX_STEPS"
echo "TOP_T:           $TOP_T"
echo "Momentum:        $MOMENTUM_MASKING"
echo "Freeze keys:     $FREEZE_KEYS"
echo "Granularity:     $GRANULARITY"
echo "IDF top-k:       $IDF_TOP_K"
echo "IDF smoothing:   $IDF_SMOOTHING"
echo "Run name:        $RUN_NAME"
echo "====================================================="


CARTRIDGES_DIR="$CARTRIDGES_DIR" \
PHASE1_CACHE_PATH="$PHASE1_CACHE_PATH" \
SYNTH_DATA_PATH="$SYNTH_DATA_PATH" \
EVAL_DATA_PATH="$EVAL_DATA_PATH" \
BG_STATS_PATH="$BG_STATS_PATH" \
NUM_TOKENS="$NUM_TOKENS" \
EPOCHS="$EPOCHS" \
LR="$LR" \
GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
MAX_STEPS="$MAX_STEPS" \
MODEL_NAME="$MODEL_NAME" \
CARTRIDGES_OUTPUT_DIR="$CARTRIDGES_OUTPUT_DIR" \
EVAL_EVERY_N_STEPS="$EVAL_EVERY_N_STEPS" \
SAVE_EVERY_N_STEPS="$SAVE_EVERY_N_STEPS" \
TOP_T="$TOP_T" \
MOMENTUM_MASKING="$MOMENTUM_MASKING" \
FREEZE_KEYS="$FREEZE_KEYS" \
GRANULARITY="$GRANULARITY" \
IDF_TOP_K="$IDF_TOP_K" \
IDF_SMOOTHING="$IDF_SMOOTHING" \
RUN_NAME="$RUN_NAME" \
DISTRIBUTED_BACKEND="$DISTRIBUTED_BACKEND" \
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
  "$CARTRIDGES_DIR/examples/qasper2/train/continual_sparse.py"

echo "=== Training Done ==="
echo ""
echo "Next step: run eval_forgetting.sh with the checkpoint paths from this run"
echo "  to measure Phase 1 retention (QA) and Phase 2 acquisition (MT)."
