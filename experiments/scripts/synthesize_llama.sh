#!/bin/bash
#SBATCH --job-name=self-study-generation-llama
#SBATCH --gres=gpu:3
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

# Configuration — adjust these for your cluster
# MODEL_NAME is required — no default
MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
DP_SIZE="${DP_SIZE:-3}"          # number of GPUs (data parallel)
PORT="${PORT:-8000}"
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
  kv_cache_num_tokens='(512 * 1024)' \
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

# All 12 combinations: 3 datasets x 4 combos (max_tokens:prob_thinking)
DATASETS="AMD:2021 AMD:2022 PepsiCo:2021"
COMBOS="8192:1.0 8192:0.2 1024:1.0 1024:0.2"

for DATASET in $DATASETS; do
  RUN_COMPANY="${DATASET%%:*}"
  RUN_YEAR="${DATASET##*:}"
  for COMBO in $COMBOS; do
    MAX_TOKENS_PER_CHUNK="${COMBO%%:*}"
    PROB_THINKING="${COMBO##*:}"
    echo "=== Running synthesis: company=$RUN_COMPANY year=$RUN_YEAR prob_thinking=$PROB_THINKING max_tokens_per_chunk=$MAX_TOKENS_PER_CHUNK ==="
    CARTRIDGES_DIR="$REPO_DIR" \
    CARTRIDGES_OUTPUT_DIR="$CARTRIDGES_OUTPUT_DIR" \
    CARTRIDGES_TOKASAURUS_URL="http://localhost:$PORT" \
    DATA_DIR="$REPO_DIR/data/financebench" \
    python "$REPO_DIR/experiments/synthesize/self_study.py" \
      --company "$RUN_COMPANY" \
      --year "$RUN_YEAR" \
      --model "$MODEL_NAME" \
      --num_samples "$NUM_SAMPLES" \
      --batch_size "$BATCH_SIZE" \
      --max_num_batches "$MAX_NUM_BATCHES" \
      --prob_thinking "$PROB_THINKING" \
      --max_tokens_per_chunk "$MAX_TOKENS_PER_CHUNK" \
      $INCLUDE_YEAR_FLAG
    echo "=== Done: company=$RUN_COMPANY year=$RUN_YEAR prob_thinking=$PROB_THINKING max_tokens_per_chunk=$MAX_TOKENS_PER_CHUNK ==="
  done
done
