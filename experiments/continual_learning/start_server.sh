#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Configuration — adjust these for your cluster
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}"
DP_SIZE="${DP_SIZE:-1}"          # number of GPUs (data parallel)
PORT="${PORT:-8000}"
TOKA_DIR="${TOKA_DIR:-$HOME/tokasaurus}"

# Install Tokasaurus if not found
if [ ! -d "$TOKA_DIR/.venv" ]; then
  echo "=== Tokasaurus not found, installing... ==="
  if [ ! -d "$TOKA_DIR" ]; then
    git clone https://github.com/ScalingIntelligence/tokasaurus "$TOKA_DIR"
  fi
  cd "$TOKA_DIR"
  git checkout --track origin/sabri/batch 2>/dev/null || git checkout sabri/batch
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
  port=$PORT
