#!/usr/bin/env bash
# Chain the RoPE-corrected Phase-1 QASPER cartridge through p02-p05 and emit the
# 5x5 stage-by-eval grid. Stage-resumable: rerunning skips completed stages.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${CARTRIDGES_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export CARTRIDGES_DIR="$ROOT"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$ROOT/outputs}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_DISABLED="${WANDB_DISABLED:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Arm D of the Phase-1 RoPE sweep: rebake at the model's own theta (5e6) with
# global teacher positions. Best of the six arms on QA logppl.
P01_CACHE="${P01_CACHE:-$ROOT/outputs/p1_rope_sweep/D_b0_rebake5e6_globalpos/2026-08-27-05-20-18-initial_am_compaction/97fde689-ebd8-4ca7-804c-cf21212dcc75/cache_last.pt}"
TAG="${TAG:-delta_ha_b0_idf0_ropefix_p01D}"
GPU="${GPU:-0}"

PY_BIN="${CARTRIDGES_PYTHON:-${PY:-python}}"
export CARTRIDGES_PYTHON="$PY_BIN"

exec "$PY_BIN" "$ROOT/examples/shared/am/continual_chain.py" \
  --dataset qasper \
  --p01-cache "$P01_CACHE" \
  --tag "$TAG" \
  --gpu "$GPU" \
  "$@"
