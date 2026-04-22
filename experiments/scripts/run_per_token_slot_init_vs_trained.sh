#!/bin/bash
# Per-token-slot V-swap for AMD-2021 trained vs step-0 Qwen3 carts.
# Runs both directions (sufficiency + necessity) for toks=1024 first.
set -euo pipefail

export CARTRIDGES_DIR=/home/phudishp/continual-cartridges
export CARTRIDGES_OUTPUT_DIR=/home/phudishp/continual-cartridges/outputs
source "$CARTRIDGES_DIR/.venv/bin/activate"
cd "$CARTRIDGES_DIR"

EVAL="amd_2021=data/financebench/eval/amd_2021_120_original.parquet"

declare -A TRAINED
TRAINED[512]=outputs/2026-03-28-00-59-25-initial/11859665-b964-4ce5-afac-cc96123fd443/cache_last.pt
TRAINED[1024]=outputs/2026-03-28-01-50-39-initial/d66546bb-e862-4771-b0ae-8ab1e54888d1/cache_last.pt
TRAINED[2048]=outputs/2026-03-28-12-48-24-initial/7bd35e62-3a4a-47ed-a41c-2ef0db6b044f/cache_last.pt

for TOKS in "$@"; do
  TR="${TRAINED[$TOKS]}"
  UNTR="outputs/amd_2021_untrained_qwen3_toks${TOKS}/cache_last.pt"

  echo "=== toks=${TOKS}  A_into_B (sufficiency: trained V into untrained base) ==="
  python experiments/swapping/per_slot_swap.py \
    --model qwen3-4b-instruct \
    --cache-a "$TR" \
    --cache-b "$UNTR" \
    --eval "$EVAL" \
    --batch-size 32 \
    --output-dir "outputs/per_token_slot_qwen3_toks${TOKS}_init_vs_trained_A_into_B"

done
echo "### DONE sizes: $@"
