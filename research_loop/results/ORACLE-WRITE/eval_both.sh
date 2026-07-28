#!/usr/bin/env bash
# Eval one cartridge checkpoint on BOTH splits with eval_forgetting.py.
# RUNBOOK §1: the script HANGS after printing `Eval loss` -> poll the log, then kill the PGID.
# Usage: eval_both.sh <label> <checkpoint.pt>
set -u
cd /localhome/local-triv/gated-continual-cartridges_explore || exit 9
export CARTRIDGES_DIR=$PWD
export CARTRIDGES_OUTPUT_DIR=$PWD/outputs
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export PYTHONPATH="$CARTRIDGES_DIR:${PYTHONPATH:-}"

LABEL="$1"
CKPT="$2"
RESDIR="$CARTRIDGES_DIR/research_loop/results/ORACLE-WRITE"
GPU="${EVAL_GPU:-1}"
export CUDA_VISIBLE_DEVICES="$GPU"

for SPLIT in QA MT; do
  LOG="$RESDIR/eval_${LABEL}_${SPLIT}.log"
  : > "$LOG"
  CHECKPOINT_PATH="$CKPT" \
  EVAL_DATA_PATH="data/qasper/eval/qasper_eval_${SPLIT}.parquet" \
  MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  RUN_NAME="ORACLE-WRITE_${LABEL}_eval${SPLIT}" \
  WANDB_GROUP=B-ROUTE \
  WANDB_DISABLED=0 \
  setsid python examples/qasper2/train/eval_forgetting.py >"$LOG" 2>&1 &
  PID=$!
  echo "[$LABEL/$SPLIT] pid=$PID ckpt=$CKPT"
  for _ in $(seq 1 90); do
    if grep -q "Eval loss - " "$LOG"; then break; fi
    if ! kill -0 "$PID" 2>/dev/null; then break; fi
    sleep 5
  done
  grep -oE "Eval loss - [0-9.]+" "$LOG" | tail -1
  grep -oE "View run at https://\S+" "$LOG" | tail -1
  kill -TERM -"$PID" 2>/dev/null
  sleep 3
  kill -KILL -"$PID" 2>/dev/null
done
echo "[$LABEL] eval done"
