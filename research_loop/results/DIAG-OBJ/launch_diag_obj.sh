#!/usr/bin/env bash
# DIAG-OBJ (board B-OBJ) — "is the closed-form solve already near its own optimum?"
#
#   (a) canonical best gradient-free point   : TOP_T=32,  RIDGE_LAMBDA=1e-4  (reproduce EXP-007-top32)
#   (b) MSE->0 oracle / maximal fitting room : TOP_T=512, RIDGE_LAMBDA=0     (everything else held)
#
# Env-knobs ONLY. No source edits. `import cartridges` is PYTHONPATH-pinned to a FROZEN snapshot
# of the explore package (/tmp/DIAG-OBJ_frozen) so a concurrent code-editing worker cannot change
# the numerics between run (a) and run (b).
# ONE flock-claimed GPU held for the whole wrapper lifetime; both trains + all 4 evals run on it.
set -u

REPO=/localhome/local-triv/gated-continual-cartridges_explore
cd "$REPO" || exit 9
export CARTRIDGES_DIR=$REPO
export CARTRIDGES_OUTPUT_DIR=$REPO/outputs
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export PYTHONPATH="/tmp/DIAG-OBJ_frozen:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

RESDIR="$REPO/research_loop/results/DIAG-OBJ"
mkdir -p "$RESDIR"
exec > "$RESDIR/wrapper.log" 2>&1

PY="$REPO/.venv/bin/python"

# ---------------- claim EXACTLY ONE GPU (never steal) ----------------
GPU_LOCK_DIR="/tmp/gpu_locks_${USER:-$(id -un)}"
mkdir -p "$GPU_LOCK_DIR"
CLAIMED=""
for attempt in $(seq 1 60); do
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
  echo "DIAGOBJ_WAITING_FOR_GPU attempt=$attempt TS=$(date -Is)"
  sleep 20
done
if [ -z "$CLAIMED" ]; then
  echo "DIAGOBJ_NO_FREE_GPU: could not flock a GPU under $GPU_LOCK_DIR" >&2
  exit 3
fi
export CUDA_VISIBLE_DEVICES="$CLAIMED"
export MASTER_PORT=$((29500 + CLAIMED))
echo "DIAGOBJ_CLAIMED_GPU=$CLAIMED MASTER_PORT=$MASTER_PORT PYTHONPATH_head=${PYTHONPATH%%:*} TS=$(date -Is)"

# sanity: which cartridges does a child python actually import?
"$PY" -c "import cartridges,os;print('DIAGOBJ_CARTRIDGES_RESOLVED='+os.path.dirname(cartridges.__file__))"

# ---------------- one AM-sparse phase-2 train run ----------------
run_train() {
  local tag="$1" top_t="$2" ridge="$3" runname="$4" log="$5"
  echo "DIAGOBJ_${tag}_TRAIN_START_EPOCH=$(date +%s) TS=$(date -Is) TOP_T=$top_t RIDGE_LAMBDA=$ridge"
  sha256sum examples/qasper2/train/continual_am_sparse.py \
            examples/qasper2/scripts/core/train_continual_am_sparse.sh
  USE_IDF=0 \
  SLOT_SELECTION=tfidf \
  GRANULARITY=per_layer \
  TOP_T="$top_t" \
  TARGET_MODE=cartridge_plus_doc \
  KEY_MODE=freeze \
  RIDGE_LAMBDA="$ridge" \
  RIDGE_SCALE=spectral \
  DELTA_WEIGHT=1e-2 \
  AM_EXECUTION_MODE=per_document \
  PHASE1_CACHE_PATH="$REPO/outputs/phase1_selfdistill_qwen512/cache_last.pt" \
  SYNTH_DATA_PATH="$REPO/data/qasper/train/qwen_qasper_MT_task_8192.parquet" \
  MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  NUM_TOKENS=512 \
  EVAL_QA_PATH="$REPO/data/qasper/eval/qasper_eval_QA.parquet" \
  EVAL_MT_PATH="$REPO/data/qasper/eval/qasper_eval_MT.parquet" \
  RUN_NAME="$runname" \
  WANDB_DISABLED=0 \
  WANDB_GROUP=B-OBJ \
  WANDB_NOTES="fitting freedom vs realized CE" \
  timeout 1500 bash examples/qasper2/scripts/core/train_continual_am_sparse.sh >"$log" 2>&1
  local rc=$?
  echo "DIAGOBJ_${tag}_TRAIN_END_EPOCH=$(date +%s) TS=$(date -Is) RC=$rc"
  sha256sum examples/qasper2/train/continual_am_sparse.py \
            examples/qasper2/scripts/core/train_continual_am_sparse.sh
  return $rc
}

# ---------------- standalone eval (parse `Eval loss`, then KILL: it hangs) ----------------
run_eval() {
  local tag="$1" ckpt="$2" evalpath="$3" runname="$4" log="$5"
  echo "DIAGOBJ_EVAL_${tag}_START TS=$(date -Is) ckpt=$ckpt eval=$evalpath"
  CHECKPOINT_PATH="$ckpt" \
  EVAL_DATA_PATH="$evalpath" \
  MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  RUN_NAME="$runname" \
  WANDB_GROUP=B-OBJ \
  WANDB_DISABLED=0 \
  "$PY" examples/qasper2/train/eval_forgetting.py >"$log" 2>&1 &
  local pid=$!
  local i
  for i in $(seq 1 900); do   # up to 15 min
    if grep -q "Eval loss - " "$log" 2>/dev/null; then
      sleep 8   # let wandb flush the run
      break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "DIAGOBJ_EVAL_${tag}_PROC_EXITED_EARLY"
      break
    fi
    sleep 1
  done
  kill -TERM "$pid" 2>/dev/null
  sleep 3
  kill -KILL "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  echo "DIAGOBJ_EVAL_${tag}_LINE: $(grep -h 'Eval loss - ' "$log" | tail -1)"
  echo "DIAGOBJ_EVAL_${tag}_END TS=$(date -Is)"
}

resolve_run_dir() {
  grep -ohE "Saved to /localhome[^ ]*" "$1" | tail -1 | sed 's/Saved to //'
}

# ================= RUN (a): canonical top32 =================
run_train A 32 1e-4 DIAG-OBJ_a_canonical-top32 "$RESDIR/run_a_top32.log"
RC_A=$?
DIR_A=$(resolve_run_dir "$RESDIR/run_a_top32.log")
echo "DIAGOBJ_RUN_A_RUNDIR=$DIR_A RC=$RC_A"

# ================= RUN (b): MSE->0 oracle, TOP_T=512 lambda=0 =================
TOPB=512
run_train B "$TOPB" 0 DIAG-OBJ_b_mse-oracle "$RESDIR/run_b_oracle.log"
RC_B=$?
if [ $RC_B -ne 0 ]; then
  echo "DIAGOBJ_RUN_B_TOP512_FAILED RC=$RC_B -> FALLING BACK TO TOP_T=256"
  cp "$RESDIR/run_b_oracle.log" "$RESDIR/run_b_oracle_top512_FAILED.log"
  TOPB=256
  run_train B256 "$TOPB" 0 DIAG-OBJ_b_mse-oracle-top256 "$RESDIR/run_b_oracle.log"
  RC_B=$?
fi
DIR_B=$(resolve_run_dir "$RESDIR/run_b_oracle.log")
echo "DIAGOBJ_RUN_B_TOPT=$TOPB RUNDIR=$DIR_B RC=$RC_B"

# ================= standalone confirmation evals on the saved checkpoints =================
if [ -n "$DIR_A" ] && [ -f "$DIR_A/cache_last.pt" ]; then
  run_eval A_QA "$DIR_A/cache_last.pt" "$REPO/data/qasper/eval/qasper_eval_QA.parquet" \
      DIAG-OBJ_a_eval_QA "$RESDIR/eval_a_qa.log"
  run_eval A_MT "$DIR_A/cache_last.pt" "$REPO/data/qasper/eval/qasper_eval_MT.parquet" \
      DIAG-OBJ_a_eval_MT "$RESDIR/eval_a_mt.log"
else
  echo "DIAGOBJ_EVAL_A_SKIPPED: no cache_last.pt at '$DIR_A'"
fi
if [ -n "$DIR_B" ] && [ -f "$DIR_B/cache_last.pt" ]; then
  run_eval B_QA "$DIR_B/cache_last.pt" "$REPO/data/qasper/eval/qasper_eval_QA.parquet" \
      DIAG-OBJ_b_eval_QA "$RESDIR/eval_b_qa.log"
  run_eval B_MT "$DIR_B/cache_last.pt" "$REPO/data/qasper/eval/qasper_eval_MT.parquet" \
      DIAG-OBJ_b_eval_MT "$RESDIR/eval_b_mt.log"
else
  echo "DIAGOBJ_EVAL_B_SKIPPED: no cache_last.pt at '$DIR_B'"
fi

echo "DIAGOBJ_RC_A=$RC_A DIAGOBJ_RC_B=$RC_B DIAGOBJ_TOPB=$TOPB"
echo "DIAGOBJ_ALL_DONE_TS=$(date -Is)"
exit $(( RC_A != 0 || RC_B != 0 ))
