#!/usr/bin/env bash
# DIAG-OBJ-c (board B-OBJ) — "remove the trust region: is the solve finally free to drive its
# own objective toward zero, and does token CE follow?"
#
# ONE run. Single variable vs DIAG-OBJ-b (run `b_mse-oracle`): DELTA_WEIGHT 1e-2 -> 0.
# Everything else held: TOP_T=511 (= all trainable slots, same effective support as b's clamped 512),
# RIDGE_LAMBDA=0, GRANULARITY=per_layer, SLOT_SELECTION=tfidf, USE_IDF=0, KEY_MODE=freeze,
# TARGET_MODE=cartridge_plus_doc, RIDGE_SCALE=spectral, AM_EXECUTION_MODE=per_document.
#
# Env-knobs ONLY. NO source edits. A concurrent ORACLE-WRITE worker is editing
# cartridges/am/{value_solve,finetune,continual}.py and examples/qasper2/train/continual_am_sparse.py
# in the live tree, so BOTH the `cartridges` package AND the driver/launcher script are taken from a
# frozen `git archive HEAD` snapshot at /tmp/amsnap_diagobjc:
#   * PYTHONPATH=/tmp/amsnap_diagobjc  -> `import cartridges` resolves to the snapshot
#   * CARTRIDGES_DIR=/tmp/amsnap_diagobjc -> the snapshot's copy of
#       examples/qasper2/scripts/core/train_continual_am_sparse.sh + train/continual_am_sparse.py
#     is what runs. CARTRIDGES_DIR is used ONLY for (a) a presence check in cartridges/__init__.py
#     and (b) resolving those two file paths; every data/cache/output path below is an absolute
#     path into the real repo, so no numerics depend on it.
#   * the frozen HEAD driver differs from the live driver ONLY by the opt-in AM_ORACLE_WRITE block,
#     which contributes {} and no extra tags when AM_ORACLE_WRITE is unset (as it is here).
#   * PATH is prepended with the repo venv bin so the snapshot script's bare `python` is the venv one
#     (the snapshot has no .venv/bin/activate to source).
# ONE flock-claimed GPU held for the whole wrapper lifetime; the train + both evals run on it.
set -u

REPO=/localhome/local-triv/gated-continual-cartridges_explore
SNAP=/tmp/amsnap_diagobjc
cd "$REPO" || exit 9

export CARTRIDGES_DIR="$SNAP"
export CARTRIDGES_OUTPUT_DIR="$REPO/outputs"
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export PYTHONPATH="$SNAP:${PYTHONPATH:-}"
export PATH="$REPO/.venv/bin:$PATH"
export TOKENIZERS_PARALLELISM=false

RESDIR="$REPO/research_loop/results/DIAG-OBJ-c"
mkdir -p "$RESDIR"
exec > "$RESDIR/wrapper.log" 2>&1

PY="$REPO/.venv/bin/python"

snap_hash() {
  find "$SNAP/cartridges" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1
}

echo "DIAGOBJC_START_TS=$(date -Is)"
echo "DIAGOBJC_SNAP_CARTRIDGES_HASH_BEFORE=$(snap_hash)"
sha256sum "$SNAP/cartridges/am/value_solve.py" "$SNAP/cartridges/am/finetune.py" \
          "$SNAP/cartridges/am/continual.py" \
          "$SNAP/examples/qasper2/train/continual_am_sparse.py" \
          "$SNAP/examples/qasper2/scripts/core/train_continual_am_sparse.sh"
echo "DIAGOBJC_WHICH_PYTHON=$(command -v python)"

# ---------------- claim EXACTLY ONE GPU (never steal) ----------------
GPU_LOCK_DIR="/tmp/gpu_locks_${USER:-$(id -un)}"
mkdir -p "$GPU_LOCK_DIR"
CLAIMED=""
for attempt in $(seq 1 180); do
  for idx in 0 1; do
    lockfile="$GPU_LOCK_DIR/gpu${idx}.lock"
    exec {fd}>"$lockfile"
    if flock -n "$fd"; then
      CLAIMED="$idx"; break
    else
      exec {fd}>&-
    fi
  done
  [ -n "$CLAIMED" ] && break
  echo "DIAGOBJC_WAITING_FOR_GPU attempt=$attempt TS=$(date -Is)"
  sleep 20
done
if [ -z "$CLAIMED" ]; then
  echo "DIAGOBJC_NO_FREE_GPU: could not flock a GPU under $GPU_LOCK_DIR" >&2
  exit 3
fi
export CUDA_VISIBLE_DEVICES="$CLAIMED"
export MASTER_PORT=$((29500 + CLAIMED))
echo "DIAGOBJC_CLAIMED_GPU=$CLAIMED MASTER_PORT=$MASTER_PORT PYTHONPATH_head=${PYTHONPATH%%:*} TS=$(date -Is)"

# sanity: which cartridges does a child python actually import?  (run from /tmp so that cwd,
# which is sys.path[0] for `python -c`, cannot shadow PYTHONPATH -- the real driver is launched by
# file path so its sys.path[0] is the script dir and PYTHONPATH wins there.)
printf 'import cartridges,os,sys\nprint("DIAGOBJC_CARTRIDGES_RESOLVED="+os.path.dirname(cartridges.__file__))\n' > /tmp/_diagobjc_probe.py
( cd /tmp && "$PY" /tmp/_diagobjc_probe.py )

# ---------------- the ONE AM-sparse phase-2 train run ----------------
LOG="$RESDIR/run_c_no_trust_region.log"
echo "DIAGOBJC_TRAIN_START_EPOCH=$(date +%s) TS=$(date -Is)"
USE_IDF=0 \
SLOT_SELECTION=tfidf \
GRANULARITY=per_layer \
TOP_T=511 \
TARGET_MODE=cartridge_plus_doc \
KEY_MODE=freeze \
RIDGE_LAMBDA=0 \
RIDGE_SCALE=spectral \
DELTA_WEIGHT=0 \
AM_EXECUTION_MODE=per_document \
PHASE1_CACHE_PATH="$REPO/outputs/phase1_selfdistill_qwen512/cache_last.pt" \
SYNTH_DATA_PATH="$REPO/data/qasper/train/qwen_qasper_MT_task_8192.parquet" \
MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
NUM_TOKENS=512 \
EVAL_DATA_PATH="$REPO/data/qasper/eval/qasper_eval_MT.parquet" \
EVAL_QA_PATH="$REPO/data/qasper/eval/qasper_eval_QA.parquet" \
EVAL_MT_PATH="$REPO/data/qasper/eval/qasper_eval_MT.parquet" \
RUN_NAME=DIAG-OBJ-c_no-trust-region \
WANDB_DISABLED=0 \
WANDB_GROUP=B-OBJ \
WANDB_NOTES="DELTA_WEIGHT=0 vs 1e-2 (DIAG-OBJ-b): true MSE->0 test" \
timeout 3000 bash "$SNAP/examples/qasper2/scripts/core/train_continual_am_sparse.sh" >"$LOG" 2>&1
RC=$?
echo "DIAGOBJC_TRAIN_END_EPOCH=$(date +%s) TS=$(date -Is) RC=$RC"
echo "DIAGOBJC_SNAP_CARTRIDGES_HASH_AFTER_TRAIN=$(snap_hash)"

RUNDIR=$(grep -ohE "Saved to /localhome[^ ]*" "$LOG" | tail -1 | sed 's/Saved to //')
echo "DIAGOBJC_RUNDIR=$RUNDIR RC=$RC"

# ---------------- standalone eval (parse `Eval loss`, then KILL: it hangs) ----------------
run_eval() {
  local tag="$1" ckpt="$2" evalpath="$3" runname="$4" log="$5"
  echo "DIAGOBJC_EVAL_${tag}_START TS=$(date -Is) ckpt=$ckpt eval=$evalpath"
  CHECKPOINT_PATH="$ckpt" \
  EVAL_DATA_PATH="$evalpath" \
  MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  RUN_NAME="$runname" \
  WANDB_GROUP=B-OBJ \
  WANDB_DISABLED=0 \
  "$PY" "$SNAP/examples/qasper2/train/eval_forgetting.py" >"$log" 2>&1 &
  local pid=$!
  local i
  for i in $(seq 1 1200); do
    if grep -q "Eval loss - " "$log" 2>/dev/null; then
      sleep 8   # let wandb flush the run
      break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "DIAGOBJC_EVAL_${tag}_PROC_EXITED_EARLY"
      break
    fi
    sleep 1
  done
  kill -TERM "$pid" 2>/dev/null
  sleep 3
  kill -KILL "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  echo "DIAGOBJC_EVAL_${tag}_LINE: $(grep -h 'Eval loss - ' "$log" | tail -1)"
  echo "DIAGOBJC_EVAL_${tag}_END TS=$(date -Is)"
}

if [ -n "$RUNDIR" ] && [ -f "$RUNDIR/cache_last.pt" ]; then
  run_eval QA "$RUNDIR/cache_last.pt" "$REPO/data/qasper/eval/qasper_eval_QA.parquet" \
      DIAG-OBJ-c_eval_QA "$RESDIR/eval_c_qa.log"
  run_eval MT "$RUNDIR/cache_last.pt" "$REPO/data/qasper/eval/qasper_eval_MT.parquet" \
      DIAG-OBJ-c_eval_MT "$RESDIR/eval_c_mt.log"
else
  echo "DIAGOBJC_EVAL_SKIPPED: no cache_last.pt at '$RUNDIR'"
fi

echo "DIAGOBJC_SNAP_CARTRIDGES_HASH_AFTER_ALL=$(snap_hash)"
echo "DIAGOBJC_RC=$RC"
echo "DIAGOBJC_ALL_DONE_TS=$(date -Is)"
exit $RC
