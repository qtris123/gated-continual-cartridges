#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Configuration ──────────────────────────────────────────────────────
CARTRIDGES_DIR="${CARTRIDGES_DIR:-$HOME/continual-cartridges}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}"
DP_SIZE="${DP_SIZE:-1}"
PORT="${PORT:-8000}"
TOKA_DIR="${TOKA_DIR:-$HOME/tokasaurus}"

# ── Cleanup: kill the server when the script exits ─────────────────────
SERVER_PID=""
cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo ""
    echo "=== Stopping Tokasaurus server (PID $SERVER_PID) ==="
    kill "$SERVER_PID"
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# ── 1. Install & start Tokasaurus server ──────────────────────────────
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

echo "=== Starting Tokasaurus server ==="
echo "Model:   $MODEL_NAME"
echo "GPUs:    $DP_SIZE"
echo "Port:    $PORT"
echo "=================================="

# Launch server in the background using its own venv
(
  source "$TOKA_DIR/.venv/bin/activate"
  tksrs \
    model=$MODEL_NAME \
    kv_cache_num_tokens='(512 * 1024)' \
    max_topk_logprobs=20 \
    dp_size=$DP_SIZE \
    port=$PORT
) &
SERVER_PID=$!

# Wait for the server to become healthy
echo "=== Waiting for Tokasaurus server on port $PORT... ==="
MAX_WAIT=300
WAITED=0
until curl -so /dev/null -w '' "http://localhost:$PORT/" 2>/dev/null; do
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
done
echo "=== Tokasaurus server is ready (waited ${WAITED}s) ==="

# ── 2. Set up continual-cartridges ────────────────────────────────────
if [ ! -d "$CARTRIDGES_DIR/.git" ]; then
  echo "=== continual-cartridges repo not found, cloning... ==="
  git clone https://github.com/faridlazuarda/continual-cartridges.git "$CARTRIDGES_DIR"
  cd "$CARTRIDGES_DIR"
  git checkout --track origin/stress_test 2>/dev/null || git checkout stress_test
  echo "=== continual-cartridges cloned ==="
fi

if [ ! -d "$CARTRIDGES_DIR/.venv" ]; then
  echo "=== Setting up cartridges venv... ==="
  cd "$CARTRIDGES_DIR"
  uv venv
  uv sync
  echo "=== cartridges venv created ==="
fi

cd "$CARTRIDGES_DIR"
source "$CARTRIDGES_DIR/.venv/bin/activate"

# ── 3. Export environment variables ────────────────────────────────────
export CARTRIDGES_DIR="$CARTRIDGES_DIR"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$CARTRIDGES_DIR/outputs}"
export CARTRIDGES_WANDB_PROJECT="${CARTRIDGES_WANDB_PROJECT:-cartridges}"
export CARTRIDGES_WANDB_ENTITY="${CARTRIDGES_WANDB_ENTITY:-phudishp-scb-datax}"
export CARTRIDGES_TOKASAURUS_URL="${CARTRIDGES_TOKASAURUS_URL:-http://localhost:$PORT}"

# ── 4. Run synthesis for all companies ─────────────────────────────────
declare -A COMPANY_YEARS
COMPANY_YEARS=(
  ["AMD"]="2015 2016 2017 2018 2019 2020"
  ["Pepsi"]="2015 2016 2017 2018 2019 2020 2021"
  ["Apple"]="2015 2016 2017 2018 2019 2020 2021 2022"
  ["Netflix"]="2015 2016 2017 2018 2019 2020 2021 2022"
)

echo ""
echo "=== Continual Learning Synthesis ==="
echo "Companies: ${!COMPANY_YEARS[*]}"
echo "Output:    $CARTRIDGES_OUTPUT_DIR"
echo "====================================="

for COMPANY in "${!COMPANY_YEARS[@]}"; do
  YEARS="${COMPANY_YEARS[$COMPANY]}"
  echo ""
  echo "=== Company: $COMPANY | Years: $YEARS ==="
  for year in $YEARS; do
    echo ""
    echo "=== Synthesizing $COMPANY $year ==="
    python experiments/continual_learning/synthesize_data.py \
      --company $COMPANY --year $year
    echo "=== Done: $COMPANY $year ==="
  done
done

echo ""
echo "=== All synthesis complete ==="
