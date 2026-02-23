#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CARTRIDGES_DIR="${CARTRIDGES_DIR:-$SCRIPT_DIR/../..}"

# Clone continual-cartridges repo if not found
if [ ! -d "$CARTRIDGES_DIR/.git" ]; then
  echo "=== continual-cartridges repo not found, cloning... ==="
  git clone https://github.com/faridlazuarda/continual-cartridges.git "$CARTRIDGES_DIR"
  cd "$CARTRIDGES_DIR"
  git checkout --track origin/stress_test 2>/dev/null || git checkout stress_test
  echo "=== continual-cartridges cloned ==="
fi

# Set up and activate venv
if [ ! -d "$CARTRIDGES_DIR/.venv" ]; then
  echo "=== Setting up venv... ==="
  cd "$CARTRIDGES_DIR"
  uv venv
  uv sync
  echo "=== venv created ==="
fi

cd "$CARTRIDGES_DIR"
source "$CARTRIDGES_DIR/.venv/bin/activate"

# Export environment variables (override with env vars if needed)
export CARTRIDGES_DIR="$CARTRIDGES_DIR"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$CARTRIDGES_DIR/outputs}"
export CARTRIDGES_WANDB_PROJECT="${CARTRIDGES_WANDB_PROJECT:-cartridges}"
export CARTRIDGES_WANDB_ENTITY="${CARTRIDGES_WANDB_ENTITY:-phudishp-scb-datax}"
export CARTRIDGES_TOKASAURUS_URL="${CARTRIDGES_TOKASAURUS_URL:-http://localhost:8000}"

COMPANY="${COMPANY:-AMD}"
YEARS="${YEARS:-2015 2016 2017 2018 2019 2020 2021 2022}"

# Make sure CARTRIDGES_OUTPUT_DIR is set
if [ -z "$CARTRIDGES_OUTPUT_DIR" ]; then
  echo "Error: CARTRIDGES_OUTPUT_DIR is not set"
  echo "Usage: export CARTRIDGES_OUTPUT_DIR=/path/to/outputs"
  exit 1
fi

echo "=== Continual Learning Synthesis ==="
echo "Company: $COMPANY"
echo "Years:   $YEARS"
echo "Output:  $CARTRIDGES_OUTPUT_DIR"
echo "====================================="

for year in $YEARS; do
  echo ""
  echo "=== Synthesizing $COMPANY $year ==="
  python experiments/continual_learning/synthesize_data.py \
    --company $COMPANY --year $year
  echo "=== Done: $COMPANY $year ==="
done

echo ""
echo "=== All synthesis complete ==="
