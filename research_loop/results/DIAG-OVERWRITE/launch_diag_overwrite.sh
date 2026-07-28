#!/usr/bin/env bash
# DIAG-OVERWRITE (B-OVERWRITE): do the 16 documents overwrite each other at top-32?
#
# ONE canonical AM top-32 run (EXP-007-top32 config, must reproduce
# QA 2.1766157150268555 / MT 2.5483615398406982) whose per-document artefacts
# (am_doc_*.pt = per-doc selections; cache-after-*.pt = per-doc cache snapshots)
# are the measurement, plus ONE forward-pass-only slot-mass diagnostic.
#
# NO source file is edited. Imports are pinned to a FROZEN SNAPSHOT of committed
# code (RUNBOOK §9c-bis) because DIAG-ROPE is editing cartridges/am/*.py concurrently.
set -u
REPO=/localhome/local-triv/gated-continual-cartridges_explore
SNAP=/tmp/amsnap_ovw

export CARTRIDGES_DIR="$SNAP"                       # driver + package both from HEAD
export CARTRIDGES_OUTPUT_DIR="$REPO/outputs"
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export PYTHONPATH="$SNAP:${PYTHONPATH:-}"

RESDIR="$REPO/research_loop/results/DIAG-OVERWRITE"
mkdir -p "$RESDIR"
exec > "$RESDIR/wrapper.log" 2>&1

echo "DIAG_OVERWRITE_START_TS=$(date -Is)"
echo "SNAPSHOT_GIT_HEAD=$(git -C "$REPO" rev-parse HEAD)"
echo "SNAPSHOT_PY_SHA256=$(cd "$SNAP" && find cartridges examples -type f -name '*.py' | sort | xargs sha256sum | sha256sum | cut -d' ' -f1)"

GPU_LOCK_DIR="/tmp/gpu_locks_${USER:-$(id -un)}"
mkdir -p "$GPU_LOCK_DIR"
CLAIMED=""
for idx in 0 1; do
  lockfile="$GPU_LOCK_DIR/gpu${idx}.lock"
  exec {fd}>"$lockfile"
  if flock -n "$fd"; then CLAIMED="$idx"; break; else exec {fd}>&-; fi
done
if [ -z "$CLAIMED" ]; then echo "DIAG_OVERWRITE_NO_FREE_GPU" >&2; exit 3; fi
export CUDA_VISIBLE_DEVICES="$CLAIMED"
export MASTER_PORT=$((29500 + CLAIMED))
echo "DIAG_OVERWRITE_CLAIMED_GPU=$CLAIMED MASTER_PORT=$MASTER_PORT"

# §9c-bis: the probe is a FALSE NEGATIVE from the repo root -> run it from /tmp.
cd /tmp && python -c "import cartridges,os;print('CARTRIDGES_IMPORT_PATH='+os.path.dirname(cartridges.__file__))"
cd "$REPO" || exit 9

echo "DIAG_OVERWRITE_TRAIN_START_EPOCH=$(date +%s) TS=$(date -Is)"
USE_IDF=0 \
SLOT_SELECTION=tfidf \
GRANULARITY=per_layer \
TOP_T=32 \
TARGET_MODE=cartridge_plus_doc \
KEY_MODE=freeze \
RIDGE_LAMBDA=1e-4 \
RIDGE_SCALE=spectral \
DELTA_WEIGHT=1e-2 \
AM_EXECUTION_MODE=per_document \
SAVE_AFTER_EACH_DOCUMENT=1 \
PHASE1_CACHE_PATH="$REPO/outputs/phase1_selfdistill_qwen512/cache_last.pt" \
SYNTH_DATA_PATH="$REPO/data/qasper/train/qwen_qasper_MT_task_8192.parquet" \
MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
NUM_TOKENS=512 \
EVAL_QA_PATH="$REPO/data/qasper/eval/qasper_eval_QA.parquet" \
EVAL_MT_PATH="$REPO/data/qasper/eval/qasper_eval_MT.parquet" \
RUN_NAME=DIAG-OVERWRITE_top32 \
WANDB_DISABLED=0 \
WANDB_GROUP=B-OVERWRITE \
WANDB_NOTES="per-document slot collision / survival at the canonical top-32 operating point (EXP-007-top32)" \
bash "$SNAP/examples/qasper2/scripts/core/train_continual_am_sparse.sh" >"$RESDIR/run_top32.log" 2>&1
RC_TRAIN=$?
echo "DIAG_OVERWRITE_TRAIN_END_EPOCH=$(date +%s) TS=$(date -Is) RC=$RC_TRAIN"

RUN_DIR=$(grep -oE "Saved to [^ ]+" "$RESDIR/run_top32.log" | tail -1 | awk '{print $3}')
echo "DIAG_OVERWRITE_RUN_DIR=$RUN_DIR"
echo "$RUN_DIR" > "$RESDIR/run_dir.txt"

if [ -n "$RUN_DIR" ] && [ -d "$RUN_DIR" ]; then
  echo "DIAG_OVERWRITE_MASS_START_EPOCH=$(date +%s) TS=$(date -Is)"
  PHASE1_CACHE="$REPO/outputs/phase1_selfdistill_qwen512/cache_last.pt" \
  EVAL_SPECS="QA:$REPO/data/qasper/eval/qasper_eval_QA.parquet,MT:$REPO/data/qasper/eval/qasper_eval_MT.parquet" \
  SLOTS_RUN_DIR="$RUN_DIR" \
  OUT_JSON="$REPO/research_loop/state/diagnostics/DIAG-OVERWRITE_slot_mass.json" \
  OUT_NPZ="$REPO/research_loop/state/diagnostics/DIAG-OVERWRITE_slot_mass.npz" \
  python "$RESDIR/measure_slot_mass.py" >"$RESDIR/slot_mass.log" 2>&1
  RC_MASS=$?
  echo "DIAG_OVERWRITE_MASS_END_EPOCH=$(date +%s) TS=$(date -Is) RC=$RC_MASS"
else
  RC_MASS=99
  echo "DIAG_OVERWRITE_NO_RUN_DIR"
fi

echo "SNAPSHOT_PY_SHA256_AFTER=$(cd "$SNAP" && find cartridges examples -type f -name '*.py' | sort | xargs sha256sum | sha256sum | cut -d' ' -f1)"
echo "REPO_GIT_HEAD_AFTER=$(git -C "$REPO" rev-parse HEAD)"
echo "DIAG_OVERWRITE_RC_TRAIN=$RC_TRAIN DIAG_OVERWRITE_RC_MASS=$RC_MASS"
echo "DIAG_OVERWRITE_ALL_DONE_TS=$(date -Is)"
exit $(( RC_TRAIN != 0 || RC_MASS != 0 ))
