#!/usr/bin/env bash
# Unattended global AM sweep: collect global bg_stats, then queue one job/GPU.
set -euo pipefail

REPO=/localhome/local-triv/gated-continual-cartridges
P1="$REPO/outputs/e2e_qa_to_mt_20260721/2026-07-21-04-55-45-initial_am/bf89229f-ddf7-46f8-8619-b2d1065da1cf"
OUT="$REPO/outputs/e2e_am_sweep"
LOG="$OUT/tmux_claude_global.log"

cd "$REPO"
source .venv/bin/activate

export CARTRIDGES_DIR="$REPO"
export CARTRIDGES_OUTPUT_DIR="$OUT"
export WANDB_DISABLED="${WANDB_DISABLED:-0}"
export CARTRIDGES_WANDB_PROJECT=cartridges
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export TORCH_CUDA_ARCH_LIST=9.0
export PYTHONUNBUFFERED=1

mkdir -p "$OUT"

if [ ! -f "$P1/bg_stats_global.pt" ]; then
  echo "[$(date -Is)] Collecting global bg_stats on GPU 0..."
  PHASE1_CACHE_PATH="$P1/cache_last.pt" \
  QA_DATA_PATH="$REPO/data/qasper/train/qwen_qasper_QA_task_8192.parquet" \
  GRANULARITY=global \
  OUT_PATH="$P1/bg_stats_global.pt" \
  NUM_BG_BATCHES=1000 \
  CUDA_VISIBLE_DEVICES=0 \
  python examples/shared/am/collect_background_stats.py
  echo "[$(date -Is)] global bg_stats ready: $P1/bg_stats_global.pt"
else
  echo "[$(date -Is)] Reusing existing $P1/bg_stats_global.pt"
fi

echo "[$(date -Is)] Launching global AM sweep queue on GPUs 0,1..."
python -u examples/shared/am/sweeps/launch_am_sweep_e2e.py \
  --phase1-cache "$P1/cache_last.pt" \
  --bg-global "$P1/bg_stats_global.pt" \
  --output-root "$OUT" \
  --gpus "${LAUNCH_GPUS:-0,1}" \
  --phase1-qa-loss 3.821 \
  --poll-s 20 \
  --granularities global
