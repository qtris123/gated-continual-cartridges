#!/bin/bash
#SBATCH --job-name=self-study-qwen3
#SBATCH --gres=gpu:1
#SBATCH --output=logs/server_%j.out
#SBATCH --error=logs/server_%j.err
set -e

export TORCH_CUDA_ARCH_LIST="9.0"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"
export LD_PRELOAD="/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12"

# Use SLURM_SUBMIT_DIR when running via sbatch (Slurm copies the script to a temp path)
if [ -n "$SLURM_SUBMIT_DIR" ]; then
  REPO_DIR="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
DP_SIZE="${DP_SIZE:-1}"
PORT="${PORT:-8001}"
TOKA_DIR="${TOKA_DIR:-/data/workspace/phudish/tokasaurus}"
NUM_SAMPLES="${NUM_SAMPLES:-8192}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NUM_BATCHES="${MAX_NUM_BATCHES:-128}"
INCLUDE_YEAR="${INCLUDE_YEAR:-false}"
CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$REPO_DIR/outputs}"

# Install Tokasaurus if not found
if [ ! -d "$TOKA_DIR/.venv" ]; then
  echo "=== Tokasaurus not found, installing... ==="
  if [ ! -d "$TOKA_DIR" ]; then
    git clone https://github.com/ScalingIntelligence/tokasaurus "$TOKA_DIR"
  fi
  cd "$TOKA_DIR"
  git checkout --track origin/sabri/batch 2>/dev/null || git checkout geoff/cartridges
  uv venv
  uv sync
  echo "=== Tokasaurus installed ==="
fi

# Activate Tokasaurus venv
source "$TOKA_DIR/.venv/bin/activate"

echo "=== Starting Tokasaurus server ==="
echo "Model:   $MODEL_NAME"
echo "GPUs:    $DP_SIZE"
echo "Port:    $PORT"
echo "=================================="

tksrs \
  model=$MODEL_NAME \
  kv_cache_num_tokens='(256 * 1024)' \
  max_topk_logprobs=20 \
  dp_size=$DP_SIZE \
  port=$PORT &
SERVER_PID=$!

MAX_WAIT=36000
WAITED=0
until curl -so /dev/null "http://localhost:$PORT/ping" 2>/dev/null; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Error: Tokasaurus server exited unexpectedly"
    exit 1
  fi
  if [ "$WAITED" -ge "$MAX_WAIT" ]; then
    echo "Error: Tokasaurus server did not start within ${MAX_WAIT}s"
    exit 1
  fi
  sleep 2
  WAITED=$((WAITED + 2))
  if [ $((WAITED % 30)) -eq 0 ]; then
    echo "Waiting for Tokasaurus server... (${WAITED}s elapsed)"
  fi
done
echo "=== Tokasaurus server ready (waited ${WAITED}s) ==="

source "$REPO_DIR/.venv/bin/activate"
if [ -f "$REPO_DIR/.env" ]; then
  source "$REPO_DIR/.env"
fi

INCLUDE_YEAR_FLAG=""
if [ "$INCLUDE_YEAR" = "true" ]; then
  INCLUDE_YEAR_FLAG="--include-year"
fi

run_synthesis() {
  local company="$1" year="$2" prob_thinking="$3" max_tokens_per_chunk="$4"
  echo "=== Running synthesis: company=$company year=$year prob_thinking=$prob_thinking max_tokens_per_chunk=$max_tokens_per_chunk ==="
  CARTRIDGES_DIR="$REPO_DIR" \
  CARTRIDGES_OUTPUT_DIR="$CARTRIDGES_OUTPUT_DIR" \
  CARTRIDGES_TOKASAURUS_URL="http://localhost:$PORT" \
  DATA_DIR="$REPO_DIR/data/financebench" \
  python "$REPO_DIR/experiments/synthesize/self_study.py" \
    --company "$company" \
    --year "$year" \
    --model "$MODEL_NAME" \
    --num_samples "$NUM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --max_num_batches "$MAX_NUM_BATCHES" \
    --prob_thinking "$prob_thinking" \
    --max_tokens_per_chunk "$max_tokens_per_chunk" \
    $INCLUDE_YEAR_FLAG
  echo "=== Done: company=$company year=$year prob_thinking=$prob_thinking max_tokens_per_chunk=$max_tokens_per_chunk ==="
}

# chunk=1024, thinking=0.2: AMD 2021, AMD 2022, PepsiCo 2021
run_synthesis "AMD"    "2021" "0.2" "1024"
run_synthesis "AMD"    "2022" "0.2" "1024"
run_synthesis "PepsiCo" "2021" "0.2" "1024"
