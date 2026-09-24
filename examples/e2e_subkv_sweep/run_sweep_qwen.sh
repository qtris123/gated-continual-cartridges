#!/usr/bin/env bash
# ==============================================================================
# E2E Sub-KV Cache Compaction Sweep for Qwen-3-4B-Instruct-2507
#
# Convenience wrapper around run_sweep.sh pre-configured for Qwen 3 4B:
#   - Model: Qwen/Qwen3-4B-Instruct-2507
#   - RoPE Base: Automatically resolved to 5,000,000.0
#   - Budgets & Top-Ts: 512/32, 1024/64, 2048/128, 4096/256
#
# Usage:
#   bash examples/e2e_subkv_sweep/run_sweep_qwen.sh [options]
#
# Examples:
#   # Run LongHealth ablation (Group A: 64 queries, delta_weight=0.01):
#   bash examples/e2e_subkv_sweep/run_sweep_qwen.sh --dataset longhealth --budget 512 --top-ts 32,64 --max-queries-per-head 64 --delta-weight 0.01 --tag-suffix _q64 --eval-accuracy --gpu 0
#
#   # Run LongHealth ablation (Group B: 1024 queries, delta_weight=0.16):
#   bash examples/e2e_subkv_sweep/run_sweep_qwen.sh --dataset longhealth --budget 512 --top-ts 32,64 --max-queries-per-head 1024 --delta-weight 0.16 --tag-suffix _q1024 --eval-accuracy --gpu 0
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"

BUDGETS_DEFAULT="512,1024,2048,4096"
TOP_TS_DEFAULT="32,64,128,256"

EXTRA_ARGS=()
HAS_BUDGETS=0
HAS_TOP_TS=0
for arg in "$@"; do
  if [[ "$arg" == "--budgets" || "$arg" == "--budget" ]]; then HAS_BUDGETS=1; fi
  if [[ "$arg" == "--top-ts" || "$arg" == "--top-t" ]]; then HAS_TOP_TS=1; fi
done

if [[ $HAS_BUDGETS -eq 0 ]]; then
  EXTRA_ARGS+=(--budgets "$BUDGETS_DEFAULT")
fi
if [[ $HAS_TOP_TS -eq 0 ]]; then
  EXTRA_ARGS+=(--top-ts "$TOP_TS_DEFAULT")
fi

exec bash "$SCRIPT_DIR/run_sweep.sh" --model "$MODEL_NAME" "${EXTRA_ARGS[@]}" "$@"
