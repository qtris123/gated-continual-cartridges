#!/usr/bin/env bash
# Wrapper to run the Phase-2 AM granularity x sparsity sweep (delta=1e-2).
# Intended to be launched inside a tmux session.
set -euo pipefail

REPO=/localhome/local-triv/gated-continual-cartridges
P1="$REPO/outputs/e2e_qa_to_mt_20260721/2026-07-21-04-55-45-initial_am/bf89229f-ddf7-46f8-8619-b2d1065da1cf"

cd "$REPO"
source .venv/bin/activate

export CARTRIDGES_DIR="$REPO"
export CARTRIDGES_OUTPUT_DIR="$REPO/outputs/e2e_am_sweep"
export WANDB_DISABLED="${WANDB_DISABLED:-0}"
export CARTRIDGES_WANDB_PROJECT=cartridges
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export TORCH_CUDA_ARCH_LIST=9.0
export PYTHONUNBUFFERED=1

python -u examples/shared/am/sweeps/launch_am_sweep_e2e.py \
  --phase1-cache "$P1/cache_last.pt" \
  --bg-per-layer "$P1/bg_stats.pt" \
  --bg-per-head "$P1/bg_stats_per_head.pt" \
  --output-root "$CARTRIDGES_OUTPUT_DIR" \
  --gpus "${LAUNCH_GPUS:-0,1}" \
  --phase1-qa-loss 3.821 \
  --poll-s 20
