#!/usr/bin/env bash
# DIAG-SEQUENCE (B-OVERWRITE): the acquisition curve across the 16-document sequence.
# Pure eval — no training, no source edits. Evaluates the per-document cache snapshots
# (cache-after-doc-*.pt) that the canonical DIAG-OVERWRITE run already saved
# (SAVE_AFTER_EACH_DOCUMENT=1) on BOTH eval splits.
#
# Imports pinned to a frozen snapshot (RUNBOOK 9c-bis) because MECH-BETA is editing
# cartridges/am/key_select.py + finetune.py concurrently.
set -u
REPO=/localhome/local-triv/gated-continual-cartridges_explore
SNAP=/tmp/amsnap_seq
PY="$REPO/.venv/bin/python"

export CARTRIDGES_DIR="$SNAP"
export CARTRIDGES_OUTPUT_DIR="$REPO/outputs"
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export PYTHONPATH="$SNAP:${PYTHONPATH:-}"

RESDIR="$REPO/research_loop/results/DIAG-SEQUENCE"
LOGDIR="$RESDIR/evals"
mkdir -p "$LOGDIR"
exec > "$RESDIR/wrapper.log" 2>&1

echo "SEQ_START_TS=$(date -Is)"
echo "REPO_HEAD_AT_LAUNCH=$(git -C "$REPO" rev-parse HEAD)"
echo "SNAPSHOT_PATH=$SNAP"
echo "SNAPSHOT_MANIFEST=$(cd "$SNAP" && find cartridges examples -type f -name '*.py' | LC_ALL=C sort | xargs sha256sum | awk '{print $1"  "$2}' | LC_ALL=C sort | sha256sum | cut -d' ' -f1)"

# ---- claim exactly one GPU via flock -----------------------------------------
GPU_LOCK_DIR="/tmp/gpu_locks_${USER:-$(id -un)}"
mkdir -p "$GPU_LOCK_DIR"
CLAIMED=""
for attempt in $(seq 1 60); do
  for idx in 0 1; do
    exec {fd}>"$GPU_LOCK_DIR/gpu${idx}.lock"
    if flock -n "$fd"; then CLAIMED="$idx"; break; else exec {fd}>&-; fi
  done
  [ -n "$CLAIMED" ] && break
  echo "SEQ_WAITING_FOR_GPU attempt=$attempt ts=$(date -Is)"
  sleep 30
done
if [ -z "$CLAIMED" ]; then echo "SEQ_NO_FREE_GPU" >&2; exit 3; fi
export CUDA_VISIBLE_DEVICES="$CLAIMED"
echo "SEQ_CLAIMED_GPU=$CLAIMED"

# ---- MANDATORY: prove the pin, from /tmp (NOT the repo root) -----------------
(cd /tmp && "$PY" -c "import cartridges,os;print('CARTRIDGES_IMPORT_PATH='+os.path.dirname(cartridges.__file__))")

RUN_DIR="$REPO/outputs/2026-07-28-21-43-17-continual_am_sparse/4b0edcab-d1c5-4313-bc43-309d7ff77097"
echo "SEQ_RUN_DIR=$RUN_DIR"

SUMMARY="$RESDIR/curve.tsv"
: > "$SUMMARY"
printf "k\tsplit\tloss\tckpt\twandb_url\n" >> "$SUMMARY"

eval_one() {   # k split ckpt
  local k="$1" split="$2" ckpt="$3"
  local log="$LOGDIR/eval_doc${k}_${split}.log"
  : > "$log"
  echo "--- [$(date -Is)] k=${k} split=${split} ckpt=${ckpt}"
  CHECKPOINT_PATH="$ckpt" \
  EVAL_DATA_PATH="$REPO/data/qasper/eval/qasper_eval_${split}.parquet" \
  MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  RUN_NAME="DIAG-SEQUENCE_doc${k}_${split}" \
  WANDB_GROUP=B-OVERWRITE \
  WANDB_DISABLED=0 \
  "$PY" "$SNAP/examples/qasper2/train/eval_forgetting.py" >"$log" 2>&1 &
  local pid=$!
  local i
  for i in $(seq 1 180); do
    grep -q "Eval loss - " "$log" && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 3
  done
  local loss url
  loss=$(grep -oE 'Eval loss - [0-9.]+' "$log" | tail -1 | awk '{print $4}')
  url=$(grep -oE 'https://wandb\.ai/[^ ]+/runs/[A-Za-z0-9]+' "$log" | tail -1)
  echo "RESULT k=${k} ${split}: loss=${loss:-MISSING} url=${url:-MISSING}"
  printf "%s\t%s\t%s\t%s\t%s\n" "$k" "$split" "${loss:-MISSING}" "$ckpt" "${url:-MISSING}" >> "$SUMMARY"
  # RUNBOOK 1: the script HANGS after printing Eval loss -> kill the PID.
  kill -TERM "$pid" 2>/dev/null; sleep 4; kill -KILL "$pid" 2>/dev/null
  sleep 2
}

# Phase-1 anchors (k=0) are already known (QA 2.23880672454834 / MT 3.7825491428375244,
# ORACLE-WRITE, same harness) -- re-measured here as a same-session control.
eval_one 0 QA "$REPO/outputs/phase1_selfdistill_qwen512/cache_last.pt"
eval_one 0 MT "$REPO/outputs/phase1_selfdistill_qwen512/cache_last.pt"

# Priority order: the brief's k first (16 is the correctness gate), then fill in the rest.
for K in 16 1 2 4 8 12 3 5 6 7 9 10 11 13 14 15; do
  IDX=$(printf "%03d" $((K-1)))
  CKPT=$(ls "$RUN_DIR"/cache-after-doc-${IDX}-*.pt 2>/dev/null | head -1)
  if [ -z "$CKPT" ]; then echo "MISSING_SNAPSHOT k=$K idx=$IDX"; continue; fi
  eval_one "$K" QA "$CKPT"
  eval_one "$K" MT "$CKPT"
done

echo "SEQ_DONE_TS=$(date -Is)"
echo "SNAPSHOT_MANIFEST_AFTER=$(cd "$SNAP" && find cartridges examples -type f -name '*.py' | LC_ALL=C sort | xargs sha256sum | awk '{print $1"  "$2}' | LC_ALL=C sort | sha256sum | cut -d' ' -f1)"
echo "REPO_HEAD_AFTER=$(git -C "$REPO" rev-parse HEAD)"
cat "$SUMMARY"
