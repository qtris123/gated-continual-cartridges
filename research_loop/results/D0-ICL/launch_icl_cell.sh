#!/usr/bin/env bash
# D0-ICL — measure full-context ICL on the LOOP'S OWN harness (eval_forgetting.py EVAL_MODE=icl).
# One cell per invocation:  launch_icl_cell.sh <gpu_idx> <ICL_TOPIC> <EVAL_SPLIT>
#   e.g. launch_icl_cell.sh 0 QA QA   -> icl_QA|QA   (diagonal)
#        launch_icl_cell.sh 0 QA MT   -> icl_QA|MT   (off-diagonal)
# Holds the repo flock on the chosen GPU for the whole run (RUNBOOK §5).
# Imports are PINNED to the frozen snapshot /tmp/amsnap_d0 (RUNBOOK §9c-bis) because
# another worker (MECH-KEYS) is editing cartridges/am/key_select.py.
set -uo pipefail

GPU="${1:?gpu idx}"
TOPIC="${2:?ICL_TOPIC (QA|MT)}"
SPLIT="${3:?EVAL split (QA|MT)}"

REPO=/localhome/local-triv/gated-continual-cartridges_explore
SNAP=/tmp/amsnap_d0
CELL="icl_${TOPIC}|${SPLIT}"
SLUG="${TOPIC}on${SPLIT}"
OUT="$REPO/research_loop/results/D0-ICL"
LOG="$REPO/logs/d0_icl_${SLUG}.log"
mkdir -p "$OUT" "$REPO/logs"

export CARTRIDGES_DIR="$REPO"
export CARTRIDGES_OUTPUT_DIR="$REPO/outputs"
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export PYTHONPATH="$SNAP:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
PY="$REPO/.venv/bin/python"

LOCKDIR="/tmp/gpu_locks_${USER}"
mkdir -p "$LOCKDIR"
LOCK="$LOCKDIR/gpu${GPU}.lock"

{
  echo "=== D0-ICL cell $CELL | gpu $GPU | $(date -Is) ==="
  echo "snapshot HEAD: $(cat $SNAP/.snapshot_head)"
  echo "repo HEAD now: $(git -C "$REPO" rev-parse HEAD)"
} >>"$LOG" 2>&1

exec 9>"$LOCK"
if ! flock -w 10800 9; then
  echo "FAILED to claim gpu${GPU} lock within 3h" >>"$LOG"; exit 3
fi
echo "claimed flock on $LOCK (pid $$)" >>"$LOG"

# provenance probe, run from /tmp so cwd never shadows the pin (RUNBOOK §9c-bis)
( cd /tmp && "$PY" -c "import cartridges,os;print('RESOLVED cartridges ->',os.path.dirname(cartridges.__file__))" ) >>"$LOG" 2>&1

START=$(date +%s)
(
  cd /tmp
  CUDA_VISIBLE_DEVICES="$GPU" \
  EVAL_MODE=icl \
  ICL_TOPIC="$TOPIC" \
  EVAL_DATA_PATH="$REPO/data/qasper/eval/qasper_eval_${SPLIT}.parquet" \
  MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  PREFILL_CHUNK_SIZE=2048 \
  WANDB_DISABLED=0 \
  RUN_NAME="D0-ICL_${TOPIC}" \
  WANDB_GROUP=REF-ICL \
  "$PY" "$SNAP/examples/qasper2/train/eval_forgetting.py"
) >>"$LOG" 2>&1
RC=$?
END=$(date +%s)
echo "=== eval rc=$RC elapsed_s=$((END-START)) $(date -Is) ===" >>"$LOG"

# Parse the metrics dict the ICL path prints, dump to JSON, and log to wandb.
CELL="$CELL" TOPIC="$TOPIC" SPLIT="$SPLIT" LOG="$LOG" OUT="$OUT" \
ELAPSED="$((END-START))" GPU="$GPU" SNAPHEAD="$(cat $SNAP/.snapshot_head)" \
"$PY" "$OUT/log_cell_to_wandb.py" >>"$LOG" 2>&1
echo "=== D0-ICL cell $CELL DONE (wandb step rc=$?) $(date -Is) ===" >>"$LOG"
