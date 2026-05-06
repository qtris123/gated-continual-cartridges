#!/usr/bin/env bash
# Qasper self-study synthesis — SLURM driver: Tokasaurus (tksrs) + synthesis.
#
# Uses conda env `cartridges` by default (override with CONDA_ENV).
#
#   sbatch examples/qasper2/scripts/synthesize_self_study.sh
#   bash examples/qasper2/scripts/synthesize_self_study.sh
#
# Env: CONDA_BASE, CONDA_ENV (default cartridges), CARTRIDGES_* (see README),
#      CUDA_HOME (+ optional TORCH_CUDA_ARCH_LIST) for Tokasaurus / FlashInfer JIT on GPU nodes.

#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G 
#SBATCH --time=45:00
#SBATCH --job-name=qasper_synthesize
#SBATCH --output=qasper_synthesize.out
#SBATCH --error=qasper_synthesize.err

set -e 

echo "=========================================="
echo "Qasper Synthesis with Tokasaurus Server"
echo "=========================================="
echo "JobID=$SLURM_JOB_ID"
echo "Partition=$SLURM_JOB_PARTITION"
echo "NodeList=$SLURM_JOB_NODELIST"
echo "Started at: $(date)"
echo ""

# Activate conda environment
echo "Activating cartridges conda environment..."
CONDA_ENV="${CONDA_ENV:-cartridges}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
echo "Python: $(which python3)"
echo ""

# Load compatible GCC for CUDA (GCC 14 causes compilation issues)
echo "Loading GCC 11.4..."
module load gcc/11.4.1
echo "GCC version: $(gcc --version | head -1)"
echo ""


# Load CUDA module for nvcc compiler (needed for flashinfer)
echo "Loading CUDA module..."
module load cuda/12.1.0
echo "CUDA version: $(nvcc --version | grep release)"
echo ""

### CUSTOMIZE YOUR SETTING ###
export TORCH_CUDA_ARCH_LIST="8.0"
export CARTRIDGES_DIR=/home/vo43/cartridges
export CARTRIDGES_OUTPUT_DIR=/home/vo43/cartridges/outputs
export TOKA_ROOT=/home/vo43/tokasaurus
BATCH_SIZE="${BATCH_SIZE:-32}"
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-2}"
###--------------------------###



export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

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
GPU_LOG="synthesize_self_study_gpu_usage.log"  
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
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  echo "Job finished at: $(date)"
}
trap cleanup EXIT


PORT="${PORT:-8000}"
NUM_SAMPLES="${NUM_SAMPLES:-65536}"
MAX_NUM_BATCHES="${MAX_NUM_BATCHES:-64}"
PROB_THINKING="${PROB_THINKING:-0.2}"
RUN_NAME="${RUN_NAME:-qasper_self_study_65K}"

export CARTRIDGES_TOKASAURUS_URL="http://127.0.0.1:${PORT}"

# Models to iterate over — one Tokasaurus server per model.
MODELS=(
  "Qwen/Qwen3-4B-Instruct-2507"
  "meta-llama/Llama-3.2-1B-Instruct"
)

start_server() {
  local model_name=$1
  echo "=== Starting Tokasaurus on :$PORT model=$model_name ==="
  tksrs \
    model="$model_name" \
    kv_cache_num_tokens='(128 * 1024)' \
    max_topk_logprobs=20 \
    dp_size="$DP_SIZE" \
    tp_size="$TP_SIZE" \
    port="$PORT" &
  SERVER_PID=$!

  local max_wait=3600 waited=0
  until curl -so /dev/null "http://127.0.0.1:${PORT}/ping" 2>/dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Tokasaurus exited unexpectedly"
      exit 1
    fi
    if [ "$waited" -ge "$max_wait" ]; then
      echo "Tokasaurus did not become ready within ${max_wait}s"
      exit 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
  echo "=== Tokasaurus ready ==="
}

stop_server() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    unset SERVER_PID
  fi
}

echo "=== Qasper self_study synthesis (conda: $CONDA_ENV) ==="
for model in "${MODELS[@]}"; do
  echo "=== Processing model: $model ==="
  # Short tag for output naming, e.g. "Qwen3-4B-Instruct-2507" or "Llama-3.2-1B-Instruct"
  model_tag="${model##*/}"
  start_server "$model"

  for topic in QA; do # MT SA; do
    echo "=== Processing topic: $topic ==="
    python "$CARTRIDGES_DIR/examples/qasper2/synthesize/self_study.py" \
      --model "$model" \
      --tokasaurus-url "$CARTRIDGES_TOKASAURUS_URL" \
      --num-samples "$NUM_SAMPLES" \
      --batch-size "$BATCH_SIZE" \
      --max-num-batches "$MAX_NUM_BATCHES" \
      --prob-thinking "$PROB_THINKING" \
      --topic "$topic" \
      --run-name "${RUN_NAME}_${model_tag}_${topic}"
  done

  stop_server
  echo "=== Done with model: $model ==="
done
echo "=== All done ==="
