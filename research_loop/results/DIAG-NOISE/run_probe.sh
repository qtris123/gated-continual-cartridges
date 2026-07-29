#!/usr/bin/env bash
set -u
export CARTRIDGES_DIR=/tmp/amsnap_DIAG-NOISE
export CARTRIDGES_OUTPUT_DIR=/localhome/local-triv/gated-continual-cartridges_explore/outputs
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export PYTHONPATH=/tmp/amsnap_DIAG-NOISE
export WANDB_DISABLED=0
export RUN_NAME=DIAG-NOISE_perexample-losses-pass4-curves
export WANDB_GROUP=VERIFY
PY=/localhome/local-triv/gated-continual-cartridges_explore/.venv/bin/python
mkdir -p /tmp/gpu_locks_$USER
export PEROUT=/tmp/diag_noise/perexample_curves.json
exec flock -w 7200 /tmp/gpu_locks_$USER/gpu0.lock bash -c '
  export CUDA_VISIBLE_DEVICES=0
  cd /tmp
  echo "GPU0 lock acquired $(date -u +%FT%TZ)"
  '"$PY"' /tmp/diag_noise/perex_probe4.py
'
