#!/usr/bin/env bash
# ==============================================================================
# E2E Sub-KV Cache Compaction Sweep for Llama-3.2-3B-Instruct
#
# Convenience wrapper around run_sweep.sh pre-configured for Llama 3.2 3B:
#   - Model: meta-llama/Llama-3.2-3B-Instruct
#   - RoPE Base: Automatically resolved to 500,000.0 (model theta)
#   - Reference Queries: Reuses existing self-study datasets
#   - Budgets & Top-Ts: 1024/64, 2048/128, 4096/256 (6.25% proportional write)
#
# Usage:
#   bash examples/e2e_subkv_sweep/run_sweep_llama.sh [options]
#
# Examples:
#   # Run default sweep on GPU 0 for QASPER:
#   bash examples/e2e_subkv_sweep/run_sweep_llama.sh --dataset qasper --gpu 0
#
#   # Run with separate eval GPU:
#   bash examples/e2e_subkv_sweep/run_sweep_llama.sh --dataset quality --gpu 0 --eval-gpus 1
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}"

exec bash "$SCRIPT_DIR/run_sweep.sh" --model "$MODEL_NAME" "$@"
