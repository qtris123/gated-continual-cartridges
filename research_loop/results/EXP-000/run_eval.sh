#!/bin/bash
# EXP-000 REF-CART final-checkpoint dual-split eval (EVAL ONLY, no train).
set -uo pipefail
cd /localhome/local-triv/gated-continual-cartridges_explore
export CARTRIDGES_DIR=$PWD
export CARTRIDGES_OUTPUT_DIR=$PWD/outputs
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
PY=$PWD/.venv/bin/python

GPU=0
LOCKDIR="/tmp/gpu_locks_${USER}"
LOCK="${LOCKDIR}/gpu${GPU}.lock"
mkdir -p "$LOCKDIR"

# Claim exactly one GPU via the repo flock; hold fd 200 for script lifetime.
exec 200>"$LOCK"
if ! flock -n 200; then
  echo "GPU${GPU} BUSY — could not acquire flock; aborting"; exit 3
fi
echo "Acquired flock on gpu${GPU} at $(date)"

export CUDA_VISIBLE_DEVICES=$GPU
export MASTER_PORT=$((29500 + GPU))
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES MASTER_PORT=$MASTER_PORT"

CKPT="outputs/2026-07-26-01-07-07-baseline_continual/10554903-c539-4056-8962-29f4fd91baee/cache-step624.pt"
OUTDIR="research_loop/results/EXP-000"

run_one () {
  local split="$1" data="$2"
  local log="$OUTDIR/eval_${split}.log"
  echo "=== [$split] launch $(date) ==="
  CHECKPOINT_PATH="$CKPT" EVAL_DATA_PATH="$data" \
    MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 RUN_NAME="refcart_10ep_${split}" \
    WANDB_DISABLED=1 \
    "$PY" examples/qasper2/train/eval_forgetting.py >"$log" 2>&1 &
  local pid=$!
  echo "[$split] pid=$pid log=$log"
  local found=""
  # poll up to ~10 min (cold CUDA init can be >2 min)
  for i in $(seq 1 600); do
    if grep -q "Eval loss" "$log" 2>/dev/null; then
      found=1; echo "[$split] Eval loss found at ${i}s"; break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[$split] process exited before Eval loss (crash?)"; break
    fi
    sleep 1
  done
  # eval_forgetting.py HANGS after printing Eval loss -> kill it, never wait-for-exit
  kill -TERM "$pid" 2>/dev/null; sleep 2; kill -KILL "$pid" 2>/dev/null
  echo "[$split] killed pid=$pid"
  echo "--- [$split] Eval loss line ---"
  grep -i "Eval loss" "$log" | tail -3
  echo "--------------------------------"
}

run_one qa data/qasper/eval/qasper_eval_QA.parquet
run_one mt data/qasper/eval/qasper_eval_MT.parquet

echo "=== ALL DONE $(date) ==="
