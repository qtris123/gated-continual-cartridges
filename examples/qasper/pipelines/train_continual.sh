# Phase 2: Continual qasper cartridge (NON-SPARSE — dense cache update).
# For sparse Phase 2 with TF-IDF, use train_continual_sparse.sh instead.

set -e

# Configuration — adjust these as needed
export TORCH_CUDA_ARCH_LIST="8.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CARTRIDGES_DIR="${CARTRIDGES_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$CARTRIDGES_DIR/outputs}"
export PATH=${CUDA_HOME:+$CUDA_HOME/bin:}$PATH
export LD_LIBRARY_PATH=${CUDA_HOME:+$CUDA_HOME/lib64:}$LD_LIBRARY_PATH

# If CUDA_VISIBLE_DEVICES is set, default NUM_GPUS to its cardinality (so this
# script DTRT in a 2x2 split). Otherwise default to 4. Explicit NUM_GPUS wins.
if [ -z "${NUM_GPUS:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_GPUS=$(awk -F',' '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
  else
    NUM_GPUS=4
  fi
fi
DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-nccl}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}"
# TODO: set to the cache_last.pt produced by train_initial.sh (or _sparse)
PHASE1_CACHE_PATH="${PHASE1_CACHE_PATH:-PLACEHOLDER_set_PHASE1_CACHE_PATH}"
SYNTH_DATA_PATH="${SYNTH_DATA_PATH:-/localhome/local-triv/gated-continual-cartridges/data/qasper/train/qasper_MT_task_8192_no-cartridge.parquet}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-$CARTRIDGES_DIR/data/qasper/eval/qasper_eval_MT.parquet}"
NUM_TOKENS="${NUM_TOKENS:-1024}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MASTER_PORT="${MASTER_PORT:-29507}"
EVAL_EVERY_N_STEPS="${EVAL_EVERY_N_STEPS:-15}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-256}"
RUN_NAME="${RUN_NAME:-qasper_phase2}"

echo "=========================================="
echo "Qasper Phase 2 — Continual Cartridge"
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
GPU_LOG="${GPU_LOG:-train_continual_gpu_usage_${_GPU_LOG_TAG}_${MASTER_PORT}.log}"
mkdir -p "$(dirname "$GPU_LOG")"
echo "=== GPU monitor started: $(date) on $(hostname) (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}) ===" >"$GPU_LOG"
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

echo "=== Training continual cartridge ==="
echo "Model:           $MODEL_NAME"
echo "GPUs:            $NUM_GPUS  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
echo "Backend:         $DISTRIBUTED_BACKEND"
echo "Master port:     $MASTER_PORT"
echo "Phase1 cache:    $PHASE1_CACHE_PATH"
echo "Train data:      $SYNTH_DATA_PATH"
echo "Eval data:       $EVAL_DATA_PATH"
echo "Tokens:          $NUM_TOKENS"
echo "Epochs:          $EPOCHS"
echo "Run name:        $RUN_NAME"
echo "====================================================="


CARTRIDGES_DIR="$CARTRIDGES_DIR" \
PHASE1_CACHE_PATH="$PHASE1_CACHE_PATH" \
SYNTH_DATA_PATH="$SYNTH_DATA_PATH" \
EVAL_DATA_PATH="$EVAL_DATA_PATH" \
NUM_TOKENS="$NUM_TOKENS" \
EPOCHS="$EPOCHS" \
LR="$LR" \
GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
MODEL_NAME="$MODEL_NAME" \
CARTRIDGES_OUTPUT_DIR="$CARTRIDGES_OUTPUT_DIR" \
EVAL_EVERY_N_STEPS="$EVAL_EVERY_N_STEPS" \
SAVE_EVERY_N_STEPS="$SAVE_EVERY_N_STEPS" \
RUN_NAME="$RUN_NAME" \
DISTRIBUTED_BACKEND="$DISTRIBUTED_BACKEND" \
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
  "$CARTRIDGES_DIR/examples/shared/train/continual_perplexity.py"

echo "=== Training Done ==="
