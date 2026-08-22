#!/usr/bin/env bash
# Continue all 24 completed QASPER P3 AM methods through ASR and KG.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${CARTRIDGES_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export CARTRIDGES_DIR="$ROOT"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$ROOT/outputs}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_DISABLED="${WANDB_DISABLED:-1}"

GPUS="${GPUS:-0,1,2}"
exec "$ROOT/.venv/bin/python" \
  "$ROOT/examples/shared/am/qasper_five_phase.py" \
  --gpus "$GPUS" "$@"
