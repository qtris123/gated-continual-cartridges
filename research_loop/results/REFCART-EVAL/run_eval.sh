#!/usr/bin/env bash
# REFCART-EVAL: eval dense self-distill Phase-2 ckpt (cache-step256.pt) on QA + MT.
# Holds gpu1 flock for its whole lifetime; for each split launches eval_forgetting.py,
# polls its log for the mean-CE "Eval loss" line, then KILLS the PID (script hangs after
# printing Eval loss -- never wait-for-exit). Releases lock on exit (FD9 close).
set -uo pipefail

cd /localhome/local-triv/gated-continual-cartridges_explore
export CARTRIDGES_DIR=$PWD
export CARTRIDGES_OUTPUT_DIR=$PWD/outputs
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
PY=$PWD/.venv/bin/python

RESDIR=research_loop/results/REFCART-EVAL
CKPT=outputs/2026-07-25-18-02-35-baseline_continual/dfed5e8b-a264-4cbf-8b9b-aa868e789ae5/cache-step256.pt

GPU=1
LOCKDIR=/tmp/gpu_locks_$USER
mkdir -p "$LOCKDIR"
LOCKFILE=$LOCKDIR/gpu${GPU}.lock

: > "$RESDIR/RUNNER_STATUS"

# --- claim exactly one GPU (gpu1) via non-blocking flock, with retries ---
exec 9>"$LOCKFILE"
acquired=0
for i in $(seq 1 30); do
  if flock -n 9; then acquired=1; break; fi
  echo "[runner] gpu$GPU busy, retry $i/30..."
  sleep 10
done
if [ "$acquired" != "1" ]; then
  echo "status=FAILED_LOCK" >> "$RESDIR/RUNNER_STATUS"
  echo "DONE" >> "$RESDIR/RUNNER_STATUS"
  echo "[runner] FAILED to claim gpu$GPU after retries"
  exit 3
fi
echo "[runner] claimed gpu$GPU via flock (MASTER_PORT convention 29501)"
echo "gpu=$GPU" >> "$RESDIR/RUNNER_STATUS"

run_one() {
  local tag="$1" evaldata="$2"
  local log="$RESDIR/eval_${tag}.log"
  echo "[runner] launching $tag eval on $evaldata -> $log"
  CUDA_VISIBLE_DEVICES=$GPU \
  CHECKPOINT_PATH="$CKPT" \
  EVAL_DATA_PATH="$evaldata" \
  MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  NUM_TOKENS=512 \
  RUN_NAME=refcart_eval_${tag} \
  WANDB_DISABLED=1 \
  "$PY" examples/qasper2/train/eval_forgetting.py > "$log" 2>&1 &
  local pid=$!
  echo "[runner] $tag PID=$pid"
  local loss=""
  # poll up to 120*5 = 600s (cold CUDA init ~1-2 min, eval a few s)
  for i in $(seq 1 120); do
    loss=$(grep -oP 'Eval loss - \K[0-9.]+' "$log" 2>/dev/null | tail -1)
    if [ -n "$loss" ]; then
      echo "[runner] $tag Eval loss = $loss -> killing PID $pid"
      kill -TERM "$pid" 2>/dev/null
      sleep 2
      kill -KILL "$pid" 2>/dev/null
      break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[runner] $tag PID $pid exited before Eval loss appeared (check log for traceback)"
      break
    fi
    sleep 5
  done
  loss=$(grep -oP 'Eval loss - \K[0-9.]+' "$log" 2>/dev/null | tail -1)
  echo "${tag}_loss=${loss:-NONE}" >> "$RESDIR/RUNNER_STATUS"
  # make sure nothing lingers
  kill -KILL "$pid" 2>/dev/null || true
}

START=$(date +%s)
run_one qa data/qasper/eval/qasper_eval_QA.parquet
run_one mt data/qasper/eval/qasper_eval_MT.parquet
END=$(date +%s)
echo "runtime_s=$((END-START))" >> "$RESDIR/RUNNER_STATUS"
echo "status=DONE" >> "$RESDIR/RUNNER_STATUS"
echo "DONE" >> "$RESDIR/RUNNER_STATUS"
echo "[runner] complete"
