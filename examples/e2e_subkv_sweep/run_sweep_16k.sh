#!/usr/bin/env bash
# ==============================================================================
# Top-t Capacity Sweep for Sub-KV Cache Budget = 16,384 Slots
#
# Sweeps over top-t in {1024, 4096, 8192} for a fixed 16,384-slot cartridge across
# 5 continual training stages (p01 initial compaction -> p02-p05 continual AM writes -> 5x5 evals).
#
# Phase 1 Optimization:
#   Phase 1 initial compaction (16,384 slots) is built once during the first arm (top-t=1024)
#   and automatically detected and reused by subsequent arms (top-t=4096, 8192),
#   saving massive compute and GPU hours.
#
# Supported Architectures:
#   - Qwen 3 4B (default: Qwen/Qwen3-4B-Instruct-2507, RoPE theta = 5,000,000)
#   - Llama 3.2 3B (meta-llama/Llama-3.2-3B-Instruct, RoPE theta = 500,000)
#
# Usage:
#   bash examples/e2e_subkv_sweep/run_sweep_16k.sh [options]
#
# Examples:
#   # Qwen on default dataset (qasper) on GPU 0:
#   bash examples/e2e_subkv_sweep/run_sweep_16k.sh --gpu 0
#
#   # Llama 3.2 3B on QASPER:
#   bash examples/e2e_subkv_sweep/run_sweep_16k.sh --model meta-llama/Llama-3.2-3B-Instruct --gpu 0
#
#   # Multi-dataset sequential execution:
#   bash examples/e2e_subkv_sweep/run_sweep_16k.sh --datasets qasper,quality --gpu 0
#
#   # Dry run:
#   bash examples/e2e_subkv_sweep/run_sweep_16k.sh --dry-run
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUDGET_VAL="16384"
TOP_TS_VAL="1024,4096,8192"

exec bash "$SCRIPT_DIR/run_sweep.sh" \
  --budget "$BUDGET_VAL" \
  --top-ts "$TOP_TS_VAL" \
  "$@"
