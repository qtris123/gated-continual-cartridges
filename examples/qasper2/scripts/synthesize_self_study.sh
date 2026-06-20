#!/usr/bin/env bash
# Qasper self-study synthesis — Tokasaurus (tksrs) + synthesis.
#
# Runs locally; SLURM directives removed (use `sbatch` only on a cluster that
# accepts the script as-is). Use the .venv interpreter at $CARTRIDGES_DIR/.venv,
# matching the longhealth pattern.
#
# Usage:
#   bash examples/qasper2/scripts/synthesize_self_study.sh
#
# Env: CARTRIDGES_DIR (auto-detected), CARTRIDGES_OUTPUT_DIR, CUDA_HOME
#      (+ optional TORCH_CUDA_ARCH_LIST) for Tokasaurus / FlashInfer JIT on GPU nodes.

set -e

echo "=========================================="
echo "Qasper Synthesis with Tokasaurus Server"
echo "=========================================="
echo "Host=$(hostname)"
echo "Started at: $(date)"
echo ""

### CUSTOMIZE YOUR SETTING ###
export TORCH_CUDA_ARCH_LIST="8.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CARTRIDGES_DIR="${CARTRIDGES_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$CARTRIDGES_DIR/outputs}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-2}"
###--------------------------###

export PATH=${CUDA_HOME:+$CUDA_HOME/bin:}$PATH
export LD_LIBRARY_PATH=${CUDA_HOME:+$CUDA_HOME/lib64:}$LD_LIBRARY_PATH

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

# Activate the repo virtual environment (pydrantic + cartridges live in .venv)
echo "Activating .venv..."
source "$CARTRIDGES_DIR/.venv/bin/activate"
echo "Python: $(which python3)"
echo ""

PORT="${PORT:-8000}"
NUM_SAMPLES="${NUM_SAMPLES:-65536}"
MAX_NUM_BATCHES="${MAX_NUM_BATCHES:-64}"
PROB_THINKING="${PROB_THINKING:-0.2}"
RUN_NAME="${RUN_NAME:-qasper_self_study_65K}"

export CARTRIDGES_TOKASAURUS_URL="http://127.0.0.1:${PORT}"

# Models to iterate over — one Tokasaurus server per model.
MODELS=(
  "Qwen/Qwen3-4B-Instruct-2507"
  "meta-llama/Llama-3.2-3B-Instruct"
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

echo "=== Qasper self_study synthesis ==="
for model in "${MODELS[@]}"; do
  echo "=== Processing model: $model ==="
  # Short tag for output naming, e.g. "Qwen3-4B-Instruct-2507" or "Llama-3.2-3B-Instruct"
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
