# Phase 1: Train initial qasper cartridge WITH background-stats collection.
# Uses train/initial_sparse.py so that Phase 2 can apply IDF-weighted sparse finetuning.
# The GRANULARITY here MUST match the GRANULARITY used in train_continual_sparse.sh.

set -e

# Configuration — adjust these as needed
export TORCH_CUDA_ARCH_LIST="8.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CARTRIDGES_DIR="${CARTRIDGES_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$CARTRIDGES_DIR/outputs}"
export PATH=${CUDA_HOME:+$CUDA_HOME/bin:}$PATH
export LD_LIBRARY_PATH=${CUDA_HOME:+$CUDA_HOME/lib64:}$LD_LIBRARY_PATH

NUM_GPUS="${NUM_GPUS:-2}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}"
NUM_TOKENS="${NUM_TOKENS:-1024}"
TEXT_PATH="${TEXT_PATH:-$CARTRIDGES_DIR/examples/qasper2/train/qasper_init_${NUM_TOKENS}.txt}"
SYNTH_DATA_PATH="${SYNTH_DATA_PATH:-/localhome/local-triv/gated-continual-cartridges/data/qasper/train/qasper_QA_task_8192_no-cartridge.parquet}"
# Optional: set to a parquet to log perplexity in W&B
EVAL_DATA_PATH="${EVAL_DATA_PATH:-$CARTRIDGES_DIR/examples/qasper2/qasper_eval_QA.parquet}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"   # for 4 GPUs: ~16/GPU; for 2 GPUs: ~32/GPU
MASTER_PORT="${MASTER_PORT:-29507}"
EVAL_EVERY_N_STEPS="${EVAL_EVERY_N_STEPS:-25}"  # halved from 50 to match doubled batch size
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-256}"
# bg_stats granularity — must match GRANULARITY in train_continual_sparse.sh
GRANULARITY="${GRANULARITY:-per_head}"    # global | per_layer | per_head
RUN_NAME="${RUN_NAME:-qasper_phase1_${GRANULARITY}_all_reduced}"

echo "=========================================="
echo "Qasper Phase 1 — Sparse Initial Cartridge"
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
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L
  echo "--- nvidia-smi (full snapshot) ---"
  nvidia-smi
else
  echo "(nvidia-smi not found — no NVIDIA driver in PATH)"
fi

# GPU monitoring (separate file)
GPU_LOG="train_initial_sparse_gpu_usage.log"
mkdir -p "$(dirname "$GPU_LOG")"
echo "=== GPU monitor started: $(date) on $(hostname) ===" >"$GPU_LOG"
(
  while true; do
    echo "index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw"
    date "+[%F %T]"
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
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

if [ -z "$SYNTH_DATA_PATH" ]; then
  echo "Error: SYNTH_DATA_PATH is required"
  exit 1
fi

# Activate the repo virtual environment (pydrantic + cartridges live in .venv)
echo "Activating .venv..."
source "$CARTRIDGES_DIR/.venv/bin/activate"
echo "Python: $(which python3)"
echo ""

echo "=== Training initial (sparse) cartridge ==="
echo "Model:       $MODEL_NAME"
echo "GPUs:        $NUM_GPUS"
echo "Text:        $TEXT_PATH"
echo "Train data:  $SYNTH_DATA_PATH"
echo "Eval data:   $EVAL_DATA_PATH"
echo "Tokens:      $NUM_TOKENS"
echo "Epochs:      $EPOCHS"
echo "Granularity: $GRANULARITY  (bg_stats for Phase 2 IDF)"
echo "Run name:    $RUN_NAME"
echo "============================================"


CARTRIDGES_DIR="$CARTRIDGES_DIR" \
TEXT_PATH="$TEXT_PATH" \
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
GRANULARITY="$GRANULARITY" \
RUN_NAME="$RUN_NAME" \
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
  "$CARTRIDGES_DIR/examples/qasper2/train/initial_sparse.py"

echo "=== Done ==="
echo ""
echo "Next step: copy the output cache_last.pt path and set it as"
echo "  PHASE1_CACHE_PATH (and optionally BG_STATS_PATH) in train_continual_sparse.sh"
