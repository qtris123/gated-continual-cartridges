#!/usr/bin/env bash
# ORACLE-WRITE-512 (board B-ROUTE): the write-ceiling oracle at FULL support.
#
# ORACLE-WRITE measured the value-only write ceiling on the tf-idf top-32 slots, whose union carries
# only ~9% of eval-time attention (mass_on_S 0.0827 QA / 0.0896 MT) and reached MT 2.3810.
# DIAG-ROUTING measured mass_on_S at top_t=512 = 0.5697 QA / 0.5879 MT — a 6.6x bandwidth gain.
# This run reads the SAME ceiling at ~59% bandwidth instead of ~9%.
#
# TWO runs, SEQUENTIAL, on ONE flock-claimed GPU held for trains + evals + diagnostic. Env-only,
# NO source edits:
#   A) ORACLE   AM_ORACLE_WRITE=1  TOP_T=512  -> teacher's own doc values into every selected slot
#   B) CONTROL  AM_ORACLE_WRITE=0  TOP_T=512  -> the ordinary closed-form solve at the same support
# Everything else is exactly ORACLE-WRITE's config (GRANULARITY=per_layer, SLOT_SELECTION=tfidf,
# USE_IDF=0, KEY_MODE=freeze, RIDGE_LAMBDA=1e-4, RIDGE_SCALE=spectral, DELTA_WEIGHT=1e-2).
#
# IMPORT PIN (RUNBOOK §9c-bis): a concurrent worker (MECH-QUERIES) is editing cartridges/am/*.py AND
# examples/qasper2/train/continual_am_sparse.py, so BOTH the package and the DRIVER come from a frozen
# `git archive HEAD` snapshot at /tmp/amsnap_ow512. Unpinned, `import cartridges` on this box resolves
# to the SIBLING repo (verified). The resolved path is printed in this log, from /tmp (§9c-bis probe).
set -u

REPO=/localhome/local-triv/gated-continual-cartridges_explore
SNAP=/tmp/amsnap_ow512
cd "$REPO" || exit 9
export CARTRIDGES_DIR=$REPO
export CARTRIDGES_OUTPUT_DIR=$REPO/outputs
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export PYTHONPATH="$SNAP:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

RESDIR="$REPO/research_loop/results/ORACLE-WRITE-512"
mkdir -p "$RESDIR"
exec > "$RESDIR/wrapper.log" 2>&1

PY="$REPO/.venv/bin/python"
DRIVER="$SNAP/examples/qasper2/train/continual_am_sparse.py"
EVALER="$SNAP/examples/qasper2/train/eval_forgetting.py"
PHASE1="$REPO/outputs/phase1_selfdistill_qwen512/cache_last.pt"

echo "OW512_SNAPSHOT_HEAD=$(cat $SNAP/SNAPSHOT_HEAD.txt)"
echo "OW512_DRIVER_SHA=$(sha256sum $DRIVER)"
echo "OW512_EVALER_SHA=$(sha256sum $EVALER)"

# ---------------- claim EXACTLY ONE GPU (never steal) ----------------
GPU_LOCK_DIR="/tmp/gpu_locks_${USER:-$(id -un)}"
mkdir -p "$GPU_LOCK_DIR"
CLAIMED=""
for attempt in $(seq 1 90); do
  for idx in 0 1; do
    lockfile="$GPU_LOCK_DIR/gpu${idx}.lock"
    exec {fd}>"$lockfile"
    if flock -n "$fd"; then CLAIMED="$idx"; break; else exec {fd}>&-; fi
  done
  [ -n "$CLAIMED" ] && break
  echo "OW512_WAITING_FOR_GPU attempt=$attempt TS=$(date -Is)"
  sleep 20
done
[ -z "$CLAIMED" ] && { echo "OW512_NO_FREE_GPU" >&2; exit 3; }
export CUDA_VISIBLE_DEVICES="$CLAIMED"
export MASTER_PORT=$((29500 + CLAIMED))
echo "OW512_CLAIMED_GPU=$CLAIMED MASTER_PORT=$MASTER_PORT TS=$(date -Is)"

# the FALSE-NEGATIVE-free probe: run it from /tmp, not the repo root (RUNBOOK §9c-bis)
(cd /tmp && "$PY" -c "import cartridges,os;print('OW512_CARTRIDGES_RESOLVED='+os.path.dirname(cartridges.__file__))")

run_one() {
  local tag="$1" oracle="$2" runname="$3" log="$4"
  echo "OW512_${tag}_START_EPOCH=$(date +%s) TS=$(date -Is)"
  AM_ORACLE_WRITE="$oracle" \
  AM_ORACLE_WRITE_ASSIGN=mass_ranked \
  USE_IDF=0 \
  SLOT_SELECTION=tfidf \
  GRANULARITY=per_layer \
  TOP_T=512 \
  TARGET_MODE=cartridge_plus_doc \
  KEY_MODE=freeze \
  ENABLE_BETA=0 \
  ENABLE_OLD_REFERENCE_GUARD=0 \
  OLD_REFERENCE_WEIGHT=1.0 \
  MAX_REF_EXAMPLES_PER_DOC=32 \
  QUERIES_PER_BATCH=all_tokens \
  UPDATE_INTERVAL=1 \
  RIDGE_LAMBDA=1e-4 \
  RIDGE_SCALE=spectral \
  RIDGE_LAMBDA_MIN=0.0 \
  DELTA_WEIGHT=1e-2 \
  AM_EXECUTION_MODE=per_document \
  EPOCHS=10 \
  GLOBAL_BATCH_SIZE=32 \
  MAX_STEPS=550 \
  DISTRIBUTED_BACKEND=nccl \
  PHASE1_CACHE_PATH="$PHASE1" \
  SYNTH_DATA_PATH="$REPO/data/qasper/train/qwen_qasper_MT_task_8192.parquet" \
  EVAL_DATA_PATH="$REPO/data/qasper/eval/qasper_eval_MT.parquet" \
  MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  NUM_TOKENS=512 \
  EVAL_QA_PATH="$REPO/data/qasper/eval/qasper_eval_QA.parquet" \
  EVAL_MT_PATH="$REPO/data/qasper/eval/qasper_eval_MT.parquet" \
  RUN_NAME="$runname" \
  WANDB_DISABLED=0 \
  WANDB_GROUP=B-ROUTE \
  WANDB_NOTES="write ceiling at 59% bandwidth vs 9% (ORACLE-WRITE top32)" \
  timeout 3600 "$PY" "$DRIVER" >"$log" 2>&1
  local rc=$?
  echo "OW512_${tag}_END_EPOCH=$(date +%s) TS=$(date -Is) RC=$rc"
  return $rc
}

# eval_forgetting.py HANGS after printing `Eval loss` (RUNBOOK §1) -> parse, then KILL the PID.
run_eval() {
  local tag="$1" ckpt="$2" evalpath="$3" runname="$4" log="$5"
  echo "OW512_EVAL_${tag}_START TS=$(date -Is) ckpt=$ckpt eval=$evalpath"
  : > "$log"
  CHECKPOINT_PATH="$ckpt" \
  EVAL_DATA_PATH="$evalpath" \
  MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  RUN_NAME="$runname" \
  WANDB_GROUP=B-ROUTE \
  WANDB_NOTES="write ceiling at 59% bandwidth vs 9% (ORACLE-WRITE top32)" \
  WANDB_DISABLED=0 \
  "$PY" "$EVALER" >"$log" 2>&1 &
  local pid=$!
  local i
  for i in $(seq 1 900); do
    if grep -q "Eval loss - " "$log" 2>/dev/null; then sleep 8; break; fi
    kill -0 "$pid" 2>/dev/null || { echo "OW512_EVAL_${tag}_PROC_EXITED_EARLY"; break; }
    sleep 1
  done
  kill -TERM "$pid" 2>/dev/null; sleep 3; kill -KILL "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  echo "OW512_EVAL_${tag}_LINE: $(grep -h 'Eval loss - ' "$log" | tail -1)"
  echo "OW512_EVAL_${tag}_CKPT_LOADED: $(grep -h -oE '/localhome[^ ]*cache_last.pt' "$log" | tail -1)"
  echo "OW512_EVAL_${tag}_WANDB: $(grep -h -oE 'https://wandb.ai/\S+/runs/\S+' "$log" | tail -1)"
  echo "OW512_EVAL_${tag}_END TS=$(date -Is)"
}

resolve_run_dir() { grep -ohE "Saved to /localhome[^ ]*" "$1" | tail -1 | sed 's/Saved to //'; }

# ================= RUN A: the oracle (decisive) =================
run_one ORACLE 1 ORACLE-WRITE-512_oracle "$RESDIR/run_oracle.log"
RC_O=$?
ORAC_DIR=$(resolve_run_dir "$RESDIR/run_oracle.log")
echo "OW512_ORACLE_RUNDIR=$ORAC_DIR RC=$RC_O"

# ================= RUN B: the like-for-like control at the same support =================
run_one CONTROL 0 ORACLE-WRITE-512_control "$RESDIR/run_control.log"
RC_C=$?
CTRL_DIR=$(resolve_run_dir "$RESDIR/run_control.log")
echo "OW512_CONTROL_RUNDIR=$CTRL_DIR RC=$RC_C"

# ================= standalone confirmation evals on the saved checkpoints =================
for SP in QA MT; do
  if [ -n "$ORAC_DIR" ] && [ -f "$ORAC_DIR/cache_last.pt" ]; then
    run_eval "oracle_$SP" "$ORAC_DIR/cache_last.pt" "$REPO/data/qasper/eval/qasper_eval_${SP}.parquet" \
      "ORACLE-WRITE-512_oracle_eval${SP}" "$RESDIR/eval_oracle_${SP}.log"
  else
    echo "OW512_EVAL_oracle_${SP}_SKIPPED: no cache_last.pt at '$ORAC_DIR'"
  fi
  if [ -n "$CTRL_DIR" ] && [ -f "$CTRL_DIR/cache_last.pt" ]; then
    run_eval "control_$SP" "$CTRL_DIR/cache_last.pt" "$REPO/data/qasper/eval/qasper_eval_${SP}.parquet" \
      "ORACLE-WRITE-512_control_eval${SP}" "$RESDIR/eval_control_${SP}.log"
  else
    echo "OW512_EVAL_control_${SP}_SKIPPED: no cache_last.pt at '$CTRL_DIR'"
  fi
done

# ================= eval-time routing diagnostic (standalone, touches nothing in cartridges/) ======
if [ -n "$ORAC_DIR" ] && [ -n "$CTRL_DIR" ]; then
  echo "=== OW512 route-mass diagnostic ==="
  CACHE_SPECS="phase1_S511:$PHASE1:$ORAC_DIR,oracle:$ORAC_DIR/cache_last.pt:-,control:$CTRL_DIR/cache_last.pt:$CTRL_DIR" \
  EVAL_SPECS="QA:$REPO/data/qasper/eval/qasper_eval_QA.parquet,MT:$REPO/data/qasper/eval/qasper_eval_MT.parquet" \
  OUT_JSON="$REPO/research_loop/state/diagnostics/ORACLE-WRITE-512_route_mass.json" \
  timeout 3600 "$PY" "$RESDIR/measure_route_mass.py"
  echo "OW512_DIAG_RC=$?"
fi

echo "OW512_RC_ORACLE=$RC_O OW512_RC_CONTROL=$RC_C"
echo "OW512_SNAP_MANIFEST_AFTER=$(find $SNAP/cartridges -name '*.py' -type f | sort | xargs sha256sum | sha256sum)"
echo "OW512_ALL_DONE_TS=$(date -Is)"
exit $(( RC_O != 0 || RC_C != 0 ))
