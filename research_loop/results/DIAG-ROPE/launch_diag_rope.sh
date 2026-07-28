#!/usr/bin/env bash
# DIAG-ROPE (B-ROPE): rotary base of the AM teacher path — 10000.0 (hard-coded) vs
# 5e6 (the model's actual config), SINGLE VARIABLE, canonical top32 operating point.
#
#   arm A (control): AM_ROPE_THETA unset  -> historical 10000.0.
#                    MUST reproduce QA 2.1766157150268555 / MT 2.5483615398406982.
#   arm B          : AM_ROPE_THETA=5000000.0.
#
# Both arms sequential on ONE flock-claimed GPU. PYTHONPATH is pinned to the _explore
# repo (RUNBOOK §6.10) — unpinned, `import cartridges` resolves to the SIBLING repo,
# which has no `rope_theta` field (verified 2026-07-28).
set -u
cd /localhome/local-triv/gated-continual-cartridges_explore || exit 9
export CARTRIDGES_DIR=$PWD
export CARTRIDGES_OUTPUT_DIR=$PWD/outputs
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export PYTHONPATH="$CARTRIDGES_DIR:${PYTHONPATH:-}"
PY="$CARTRIDGES_DIR/.venv/bin/python"

RESDIR="$CARTRIDGES_DIR/research_loop/results/DIAG-ROPE"
mkdir -p "$RESDIR"
exec > "$RESDIR/wrapper.log" 2>&1

GPU_LOCK_DIR="/tmp/gpu_locks_${USER:-$(id -un)}"
mkdir -p "$GPU_LOCK_DIR"
CLAIMED=""
for idx in 0 1; do
  lockfile="$GPU_LOCK_DIR/gpu${idx}.lock"
  exec {fd}>"$lockfile"
  if flock -n "$fd"; then CLAIMED="$idx"; break; else exec {fd}>&-; fi
done
if [ -z "$CLAIMED" ]; then echo "DIAG_ROPE_NO_FREE_GPU"; exit 3; fi
export CUDA_VISIBLE_DEVICES="$CLAIMED"
export MASTER_PORT=$((29500 + CLAIMED))
echo "DIAG_ROPE_CLAIMED_GPU=$CLAIMED MASTER_PORT=$MASTER_PORT TS=$(date -Is)"
(cd /tmp && "$PY" -c "import cartridges,os;print('CARTRIDGES_IMPORT_PATH='+os.path.dirname(cartridges.__file__))")

run_arm() {   # tag  theta("" = unset)  runname  log
  local tag="$1" theta="$2" runname="$3" log="$4"
  echo "DIAG_ROPE_${tag}_START_EPOCH=$(date +%s) TS=$(date -Is)"
  if [ -z "$theta" ]; then
    unset AM_ROPE_THETA
  else
    export AM_ROPE_THETA="$theta"
  fi
  USE_IDF=0 \
  SLOT_SELECTION=tfidf \
  GRANULARITY=per_layer \
  TOP_T=32 \
  TARGET_MODE=cartridge_plus_doc \
  KEY_MODE=freeze \
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
  WANDB_GROUP=B-ROPE \
  WANDB_NOTES="rotary base 10000 (hard-coded) vs 5e6 (model config), single variable" \
  bash examples/qasper2/scripts/core/train_continual_am_sparse.sh >"$log" 2>&1
  local rc=$?
  unset AM_ROPE_THETA
  echo "DIAG_ROPE_${tag}_END_EPOCH=$(date +%s) TS=$(date -Is) RC=$rc"
  local dir
  dir=$(grep -oE "Saved to /localhome/\S+" "$log" | tail -1 | awk '{print $3}')
  echo "DIAG_ROPE_${tag}_RUNDIR=$dir"
  echo "DIAG_ROPE_${tag}_WANDB=$(grep -oE 'https://wandb.ai/\S+/runs/[a-z0-9]+' "$log" | tail -1)"
  echo "$dir" > "$RESDIR/rundir_${tag}.txt"
  return $rc
}

run_arm A "" DIAG-ROPE_theta1e4 "$RESDIR/run_theta1e4.log"; RC_A=$?
run_arm B "5000000.0" DIAG-ROPE_theta5e6 "$RESDIR/run_theta5e6.log"; RC_B=$?
echo "DIAG_ROPE_RC_A=$RC_A RC_B=$RC_B"

DIR_A=$(cat "$RESDIR/rundir_A.txt" 2>/dev/null)
DIR_B=$(cat "$RESDIR/rundir_B.txt" 2>/dev/null)
echo "DIR_A=$DIR_A"
echo "DIR_B=$DIR_B"

# ---------------------------------------------------------------- independent evals
eval_one() {  # label ckpt split
  local label="$1" ckpt="$2" split="$3"
  local log="$RESDIR/eval_${label}_${split}.log"
  : > "$log"
  CHECKPOINT_PATH="$ckpt" \
  EVAL_DATA_PATH="data/qasper/eval/qasper_eval_${split}.parquet" \
  MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  RUN_NAME="DIAG-ROPE_${label}_eval${split}" \
  WANDB_GROUP=B-ROPE \
  WANDB_DISABLED=0 \
  "$PY" examples/qasper2/train/eval_forgetting.py >"$log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 150); do
    grep -q "Eval loss - " "$log" && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 5
  done
  echo "RESULT ${label} ${split}: $(grep -oE 'Eval loss - [0-9.]+' "$log" | tail -1)  ckpt=$ckpt"
  echo "WANDB  ${label} ${split}: $(grep -oE 'https://wandb.ai/\S+/runs/[a-z0-9]+' "$log" | tail -1)"
  kill -TERM "$pid" 2>/dev/null; sleep 4; kill -KILL "$pid" 2>/dev/null
  sleep 2
}

echo "=== INDEPENDENT EVALS ==="
for SPLIT in QA MT; do
  [ -n "$DIR_A" ] && eval_one theta1e4 "$DIR_A/cache_last.pt" "$SPLIT"
  [ -n "$DIR_B" ] && eval_one theta5e6 "$DIR_B/cache_last.pt" "$SPLIT"
done

# --------------------------------------------------- eval-time cartridge routing mass
echo "=== ROUTE-MASS DIAGNOSTIC ==="
if [ -n "$DIR_A" ] && [ -n "$DIR_B" ]; then
  CACHE_SPECS="theta1e4:$DIR_A/cache_last.pt:$DIR_A,theta5e6:$DIR_B/cache_last.pt:$DIR_B" \
  EVAL_SPECS="QA:data/qasper/eval/qasper_eval_QA.parquet,MT:data/qasper/eval/qasper_eval_MT.parquet" \
  OUT_JSON="$CARTRIDGES_DIR/research_loop/state/diagnostics/DIAG-ROPE_route_mass.json" \
  "$PY" "$RESDIR/../ORACLE-WRITE/measure_route_mass.py"
  echo "ROUTE_MASS_RC=$?"
fi

# --------------------------------------------------------------- per-layer collection
echo "=== COLLECT PER-DOC / PER-LAYER STATS ==="
RUNDIR_A="$DIR_A" RUNDIR_B="$DIR_B" \
OUT_JSON="$CARTRIDGES_DIR/research_loop/state/diagnostics/DIAG-ROPE.json" \
"$PY" "$RESDIR/collect_stats.py"
echo "COLLECT_RC=$?"

echo "DIAG_ROPE_ALL_DONE_TS=$(date -Is)"
exit $(( RC_A != 0 || RC_B != 0 ))
