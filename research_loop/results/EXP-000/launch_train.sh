#!/usr/bin/env bash
# EXP-000 REF-CART detached launcher. Holds a GPU flock for the run lifetime,
# runs the dense self-distillation Phase-2 baseline to completion (10 epochs).
set -uo pipefail
cd /localhome/local-triv/gated-continual-cartridges_explore
export CARTRIDGES_DIR=$PWD
export CARTRIDGES_OUTPUT_DIR=$PWD/outputs
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
LOG=research_loop/results/EXP-000/train.log
GPU_LOCK_DIR=/tmp/gpu_locks_${USER}
mkdir -p "$GPU_LOCK_DIR"

CLAIMED=""
for idx in 0 1; do
  lock="$GPU_LOCK_DIR/gpu${idx}.lock"
  exec {fd}>"$lock"
  if flock -n "$fd"; then CLAIMED=$idx; break; else exec {fd}>&-; fi
done
if [ -z "$CLAIMED" ]; then echo "NO_FREE_GPU" > "$LOG"; exit 3; fi
export CUDA_VISIBLE_DEVICES=$CLAIMED
export MASTER_PORT=$((29500+CLAIMED))
{
  echo "=== EXP-000 REF-CART launch $(date -Is) ==="
  echo "CLAIMED_GPU=$CLAIMED MASTER_PORT=$MASTER_PORT"
  echo "TRAIN_START_EPOCH=$(date +%s)"
} > "$LOG"

PHASE1_CACHE_PATH=outputs/phase1_selfdistill_qwen512/cache_last.pt \
SYNTH_DATA_PATH=data/qasper/train/qwen_qasper_MT_task_8192.parquet \
EVAL_DATA_PATH=data/qasper/eval/qasper_eval_QA.parquet \
MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
NUM_TOKENS=512 EPOCHS=10 GLOBAL_BATCH_SIZE=32 LR=2e-2 \
EVAL_EVERY_N_STEPS=15 SAVE_EVERY_N_STEPS=256 \
DISTRIBUTED_BACKEND=gloo WANDB_MODE=disabled RUN_NAME=refcart_phase2_EXP000 \
.venv/bin/torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
  examples/qasper2/train/baseline_continual.py >> "$LOG" 2>&1
echo "TRAIN_END_EPOCH=$(date +%s) rc=$?" >> "$LOG"
