#!/usr/bin/env bash
# QuALITY phase-1 self-distilled cartridge (1 GPU, W&B online).
#
# Same recipe as the healthy QASPER p01: KVFromText init + 10-epoch Adam on
# assistant top-20 logprobs. Does not use classic AM compaction.
#
# Usage:
#   bash examples/quality/pipelines/train_initial_selfdistill.sh
#   GPU=3 EPOCHS=10 bash examples/quality/pipelines/train_initial_selfdistill.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${CARTRIDGES_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PY="${AM_PY:-$ROOT/.venv/bin/python}"

export CARTRIDGES_DIR="$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

GPU="${GPU:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
NUM_TOKENS="${NUM_TOKENS:-512}"
NUM_FROZEN_TOKENS="${NUM_FROZEN_TOKENS:-1}"
TEXT_PATH="${TEXT_PATH:-$ROOT/data/quality/init_text/quality_p1.txt}"
SYNTH_DATA_PATH="${SYNTH_DATA_PATH:-$ROOT/data/quality/train/qwen_quality_p1_task_8192.parquet}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-$ROOT/data/quality/phases/phase1_eval.parquet}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
EVAL_EVERY_N_STEPS="${EVAL_EVERY_N_STEPS:-50}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-256}"
DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-nccl}"
MASTER_PORT="${MASTER_PORT:-29611}"
RUN_NAME="${RUN_NAME:-quality_p1_selfdistill_qwen512}"
CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$ROOT/outputs/quality_selfdistill_runs}"

export CARTRIDGES_WANDB_PROJECT="${CARTRIDGES_WANDB_PROJECT:-SEACrowd}"
export CARTRIDGES_WANDB_ENTITY="${CARTRIDGES_WANDB_ENTITY:-vqtri-purdue-university}"
unset WANDB_DISABLED || true
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"

if [ -z "${WANDB_API_KEY:-}" ]; then
  SECRETS="${WANDB_SECRETS_FILE:-$HOME/incoming/holders/research-backup-for-mac/secrets/.bashrc}"
  if [ -f "$SECRETS" ]; then
    WANDB_API_KEY="$(sed -n 's/^export WANDB_API_KEY=//p' "$SECRETS" | head -1)"
    export WANDB_API_KEY
  fi
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "FATAL: WANDB_API_KEY is not set and no secrets file was found." >&2
  exit 2
fi

[ -x "$PY" ] || { echo "FATAL: python not found at $PY" >&2; exit 1; }
[ -f "$SYNTH_DATA_PATH" ] || { echo "FATAL: missing $SYNTH_DATA_PATH" >&2; exit 1; }
[ -f "$EVAL_DATA_PATH" ] || { echo "FATAL: missing $EVAL_DATA_PATH" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export MODEL_NAME TEXT_PATH SYNTH_DATA_PATH EVAL_DATA_PATH
export NUM_TOKENS NUM_FROZEN_TOKENS EPOCHS LR GLOBAL_BATCH_SIZE
export EVAL_EVERY_N_STEPS SAVE_EVERY_N_STEPS DISTRIBUTED_BACKEND RUN_NAME
export CARTRIDGES_OUTPUT_DIR

mkdir -p "$CARTRIDGES_OUTPUT_DIR" "$(dirname "$TEXT_PATH")"
LOGDIR="${LOGDIR:-$ROOT/outputs/quality_selfdistill_logs}"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOGDIR/${RUN_NAME}_${STAMP}.log"

echo "=========================================="
echo "QuALITY p01 self-distill | GPU $GPU | $(date)"
echo "model=$MODEL_NAME tokens=$NUM_TOKENS frozen=$NUM_FROZEN_TOKENS epochs=$EPOCHS"
echo "synth=$SYNTH_DATA_PATH"
echo "eval=$EVAL_DATA_PATH"
echo "text=$TEXT_PATH"
echo "wandb=$CARTRIDGES_WANDB_ENTITY/$CARTRIDGES_WANDB_PROJECT name=$RUN_NAME"
echo "log=$LOG"
echo "=========================================="

"$PY" - <<'PY'
import os
from pathlib import Path
from cartridges.data.quality.resources import QuALITYResource

path = Path(os.environ["TEXT_PATH"])
if not path.is_file() or path.stat().st_size == 0:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(QuALITYResource(QuALITYResource.Config(phase=1)).to_string())
    print(f"wrote init text {path} ({path.stat().st_size} bytes)")
else:
    print(f"reusing init text {path} ({path.stat().st_size} bytes)")
PY

"$PY" - <<'PY'
import os, wandb
wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
print("wandb login ok")
PY

if [ "$NUM_GPUS" = "1" ]; then
  "$PY" "$ROOT/examples/quality/pipelines/train_initial_selfdistill.py"
else
  torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    "$ROOT/examples/quality/pipelines/train_initial_selfdistill.py"
fi

echo "=== done $(date) ==="
