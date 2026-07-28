#!/usr/bin/env bash
# MECH-BETA (B-SOLVE primary, B-ROUTE): does more bandwidth convert to acquisition?
#
#   arm control : beta off, AM_ROPE_THETA=1e4  -> MUST reproduce QA 2.1766157150268555
#                                                 / MT 2.5483615398406982 (in-run).
#   arm rope    : beta off, AM_ROPE_THETA=5e6  -> should reproduce DIAG-ROPE arm B.
#   arm beta    : ENABLE_BETA=1, AM_ROPE_THETA=5e6, boxed NNLS (box 3.0, gelsd, 2 PGD).
#
# All three sequential on ONE flock-claimed GPU. PYTHONPATH pinned to _explore
# (RUNBOOK §6.10/§9c-bis): unpinned, `import cartridges` resolves to the SIBLING
# repo, which has neither `rope_theta` nor the beta-fit fields.
set -u
cd /localhome/local-triv/gated-continual-cartridges_explore || exit 9
export CARTRIDGES_DIR=$PWD
export CARTRIDGES_OUTPUT_DIR=$PWD/outputs
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export PYTHONPATH="$CARTRIDGES_DIR:${PYTHONPATH:-}"
PY="$CARTRIDGES_DIR/.venv/bin/python"

RESDIR="$CARTRIDGES_DIR/research_loop/results/MECH-BETA"
mkdir -p "$RESDIR"
exec > "$RESDIR/wrapper.log" 2>&1

GPU_LOCK_DIR="/tmp/gpu_locks_${USER:-$(id -un)}"
mkdir -p "$GPU_LOCK_DIR"
CLAIMED=""
for attempt in $(seq 1 240); do
  for idx in 1 0; do
    lockfile="$GPU_LOCK_DIR/gpu${idx}.lock"
    exec {fd}>"$lockfile"
    if flock -n "$fd"; then CLAIMED="$idx"; break; else exec {fd}>&-; fi
  done
  [ -n "$CLAIMED" ] && break
  sleep 15
done
if [ -z "$CLAIMED" ]; then echo "MECH_BETA_NO_FREE_GPU"; exit 3; fi
export CUDA_VISIBLE_DEVICES="$CLAIMED"
export MASTER_PORT=$((29500 + CLAIMED))
echo "MECH_BETA_CLAIMED_GPU=$CLAIMED MASTER_PORT=$MASTER_PORT TS=$(date -Is)"
(cd /tmp && "$PY" -c "import cartridges,os;print('CARTRIDGES_IMPORT_PATH='+os.path.dirname(cartridges.__file__))")

echo "=== PART A: CUDA UNIT SANITY (gels NaN vs gelsd+box) ==="
"$PY" "$RESDIR/sanity_beta.py" --cuda > "$RESDIR/sanity_cuda.json" 2> "$RESDIR/sanity_cuda.err"
echo "SANITY_RC=$?"
tail -c 4000 "$RESDIR/sanity_cuda.json"

run_arm() {   # tag  theta  enable_beta  runname  log
  local tag="$1" theta="$2" beta="$3" runname="$4" log="$5"
  echo "MECH_BETA_${tag}_START_EPOCH=$(date +%s) TS=$(date -Is)"
  export AM_ROPE_THETA="$theta"
  export ENABLE_BETA="$beta"
  if [ "$beta" = "1" ]; then
    export AM_BETA_BOX=3.0 AM_NNLS_ITERS=2 AM_NNLS_DRIVER=gelsd AM_BETA_TARGET=residual
  else
    unset AM_BETA_BOX AM_NNLS_ITERS AM_NNLS_DRIVER AM_BETA_TARGET
  fi
  USE_IDF=0 \
  SLOT_SELECTION=tfidf \
  GRANULARITY=per_layer \
  TOP_T=32 \
  TARGET_MODE=cartridge_plus_doc \
  KEY_MODE=freeze \
  BETA_FIT_SCOPE=selected \
  RIDGE_LAMBDA=1e-4 \
  RIDGE_SCALE=spectral \
  RIDGE_LAMBDA_MIN=0.0 \
  DELTA_WEIGHT=1e-2 \
  MAX_QUERIES_PER_HEAD=64 \
  AM_EXECUTION_MODE=per_document \
  PHASE1_CACHE_PATH=outputs/phase1_selfdistill_qwen512/cache_last.pt \
  SYNTH_DATA_PATH=data/qasper/train/qwen_qasper_MT_task_8192.parquet \
  MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  NUM_TOKENS=512 \
  EVAL_QA_PATH=data/qasper/eval/qasper_eval_QA.parquet \
  EVAL_MT_PATH=data/qasper/eval/qasper_eval_MT.parquet \
  RUN_NAME="$runname" \
  WANDB_DISABLED=0 \
  WANDB_GROUP=B-SOLVE \
  WANDB_NOTES="boxed NNLS + gelsd + correct rope: does bandwidth convert to acquisition?" \
  bash examples/qasper2/scripts/core/train_continual_am_sparse.sh >"$log" 2>&1
  local rc=$?
  unset AM_ROPE_THETA ENABLE_BETA AM_BETA_BOX AM_NNLS_ITERS AM_NNLS_DRIVER AM_BETA_TARGET
  echo "MECH_BETA_${tag}_END_EPOCH=$(date +%s) TS=$(date -Is) RC=$rc"
  local dir
  dir=$(grep -oE "Saved to /localhome/\S+" "$log" | tail -1 | awk '{print $3}')
  echo "MECH_BETA_${tag}_RUNDIR=$dir"
  echo "MECH_BETA_${tag}_WANDB=$(grep -oE 'https://wandb.ai/\S+/runs/[a-z0-9]+' "$log" | tail -1)"
  echo "$dir" > "$RESDIR/rundir_${tag}.txt"
  return $rc
}

echo "=== PART B: THREE ARMS ==="
run_arm control "10000.0"   0 MECH-BETA_control  "$RESDIR/run_control.log";  RC_C=$?
run_arm rope    "5000000.0" 0 MECH-BETA_rope     "$RESDIR/run_rope.log";     RC_R=$?
run_arm beta    "5000000.0" 1 MECH-BETA_beta     "$RESDIR/run_beta.log";     RC_B=$?
echo "MECH_BETA_RC control=$RC_C rope=$RC_R beta=$RC_B"

DIR_C=$(cat "$RESDIR/rundir_control.txt" 2>/dev/null)
DIR_R=$(cat "$RESDIR/rundir_rope.txt" 2>/dev/null)
DIR_B=$(cat "$RESDIR/rundir_beta.txt" 2>/dev/null)
echo "DIR_control=$DIR_C"
echo "DIR_rope=$DIR_R"
echo "DIR_beta=$DIR_B"

# ---------------------------------------------------------------- independent evals
eval_one() {  # label ckpt split
  local label="$1" ckpt="$2" split="$3"
  local log="$RESDIR/eval_${label}_${split}.log"
  : > "$log"
  CHECKPOINT_PATH="$ckpt" \
  EVAL_DATA_PATH="data/qasper/eval/qasper_eval_${split}.parquet" \
  MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  RUN_NAME="MECH-BETA_${label}_eval${split}" \
  WANDB_GROUP=B-SOLVE \
  WANDB_NOTES="boxed NNLS + gelsd + correct rope: does bandwidth convert to acquisition?" \
  WANDB_DISABLED=0 \
  "$PY" examples/qasper2/train/eval_forgetting.py >"$log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 180); do
    grep -q "Eval loss - " "$log" && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 5
  done
  echo "IMPORT ${label} ${split}: $(grep -oE '/localhome/\S*_explore/cartridges' "$log" | head -1)"
  echo "RESULT ${label} ${split}: $(grep -oE 'Eval loss - [0-9.]+' "$log" | tail -1)  ckpt=$ckpt"
  echo "WANDB  ${label} ${split}: $(grep -oE 'https://wandb.ai/\S+/runs/[a-z0-9]+' "$log" | tail -1)"
  kill -TERM "$pid" 2>/dev/null; sleep 4; kill -KILL "$pid" 2>/dev/null
  sleep 2
}

echo "=== INDEPENDENT EVALS ==="
for SPLIT in QA MT; do
  [ -n "$DIR_C" ] && eval_one control "$DIR_C/cache_last.pt" "$SPLIT"
  [ -n "$DIR_R" ] && eval_one rope    "$DIR_R/cache_last.pt" "$SPLIT"
  [ -n "$DIR_B" ] && eval_one beta    "$DIR_B/cache_last.pt" "$SPLIT"
done

# --------------------------------------------- eval-time mass_on_S (beta-aware) + ratio
echo "=== ROUTE-MASS DIAGNOSTIC (beta-aware) ==="
PHASE1="$CARTRIDGES_DIR/outputs/phase1_selfdistill_qwen512/cache_last.pt"
SPECS=""
[ -n "$DIR_C" ] && SPECS="${SPECS}control:$DIR_C/cache_last.pt:$DIR_C,"
[ -n "$DIR_R" ] && SPECS="${SPECS}rope:$DIR_R/cache_last.pt:$DIR_R,"
[ -n "$DIR_B" ] && SPECS="${SPECS}beta:$DIR_B/cache_last.pt:$DIR_B,"
# Phase-1 reference measured against the CONTROL arm's slot set (as MECH-QUERIES-B did)
SPECS="${SPECS}phase1:$PHASE1:$DIR_C"
echo "CACHE_SPECS=$SPECS"
CACHE_SPECS="$SPECS" \
EVAL_SPECS="QA:data/qasper/eval/qasper_eval_QA.parquet,MT:data/qasper/eval/qasper_eval_MT.parquet" \
OUT_JSON="$CARTRIDGES_DIR/research_loop/state/diagnostics/MECH-BETA_route_mass.json" \
"$PY" "$RESDIR/measure_route_mass_beta.py"
echo "ROUTE_MASS_RC=$?"

# --------------------------------------------------------------- per-layer collection
echo "=== COLLECT PER-DOC / PER-LAYER STATS ==="
ARMS="control:$DIR_C,rope:$DIR_R,beta:$DIR_B" \
CONTROL_ARM=control \
OUT_JSON="$CARTRIDGES_DIR/research_loop/state/diagnostics/MECH-BETA.json" \
"$PY" "$RESDIR/collect_stats.py"
echo "COLLECT_RC=$?"

echo "MECH_BETA_ALL_DONE_TS=$(date -Is)"
exit $(( RC_C != 0 || RC_R != 0 || RC_B != 0 ))
