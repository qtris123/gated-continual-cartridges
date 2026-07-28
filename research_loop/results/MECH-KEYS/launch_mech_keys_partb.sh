#!/usr/bin/env bash
# MECH-KEYS (B-ROUTE, key side): the FIRST KEY_MODE != freeze experiment in this loop.
#
#   arm control        : KEY_MODE=freeze                        -> MUST reproduce
#                        DIAG-ROPE arm B: QA 2.15320086479187 / MT 2.5296061038970947
#   arm keys_norepos   : KEY_MODE=highest_attention, AM_KEY_REPOSITION=0  (the folklore test)
#   arm keys_repos     : KEY_MODE=highest_attention, AM_KEY_REPOSITION=1  (H2 corrected)
#   arm omp            : KEY_MODE=omp, reposition = whichever of the two did better on MT
#
# Everything else canonical top32 at theta=5e6. ENABLE_BETA=0 EXPLICITLY (hazard H4):
# `_should_fit_beta` no longer couples beta to key_mode (MECH-004) but the flag is set
# anyway so the keys-only arm is unambiguous.
#
# All arms sequential on ONE flock-claimed GPU. PYTHONPATH pinned to _explore
# (RUNBOOK 6.10 / 9c-bis): unpinned, `import cartridges` resolves to the SIBLING repo,
# which has neither `rope_theta` nor `key_reposition`.
set -u
cd /localhome/local-triv/gated-continual-cartridges_explore || exit 9
export CARTRIDGES_DIR=$PWD
export CARTRIDGES_OUTPUT_DIR=$PWD/outputs
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export PYTHONPATH="$CARTRIDGES_DIR:${PYTHONPATH:-}"
PY="$CARTRIDGES_DIR/.venv/bin/python"

RESDIR="$CARTRIDGES_DIR/research_loop/results/MECH-KEYS"
mkdir -p "$RESDIR"
exec >> "$RESDIR/wrapper_partb.log" 2>&1

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
if [ -z "$CLAIMED" ]; then echo "MECH_KEYS_NO_FREE_GPU"; exit 3; fi
export CUDA_VISIBLE_DEVICES="$CLAIMED"
export MASTER_PORT=$((29500 + CLAIMED))
echo "MECH_KEYS_CLAIMED_GPU=$CLAIMED MASTER_PORT=$MASTER_PORT TS=$(date -Is)"
(cd /tmp && "$PY" -c "import cartridges,os;print('CARTRIDGES_IMPORT_PATH='+os.path.dirname(cartridges.__file__))")

# PART A sanity already completed (sanity_cpu.json + sanity_cuda.json, both green).
# The CUDA interpreter hung in `nv_kthread_q_flush` at exit AFTER writing its JSON
# (same class as the eval_forgetting hang, RUNBOOK 1/6.7), so this restart skips it.
echo "=== PART A: SKIPPED (already done; see sanity_cuda.json) ==="
echo "SANITY_RC=0"

run_arm() {   # tag  key_mode  reposition  runname  log
  local tag="$1" kmode="$2" repos="$3" runname="$4" log="$5"
  echo "MECH_KEYS_${tag}_START_EPOCH=$(date +%s) TS=$(date -Is)"
  export AM_ROPE_THETA=5000000
  export MAX_QUERIES_PER_HEAD=64
  export SLOT_SELECTION=tfidf
  if [ "$repos" = "-" ]; then unset AM_KEY_REPOSITION; else export AM_KEY_REPOSITION="$repos"; fi
  USE_IDF=0 \
  GRANULARITY=per_layer \
  TOP_T=32 \
  TARGET_MODE=cartridge_plus_doc \
  KEY_MODE="$kmode" \
  ENABLE_BETA=0 \
  BETA_FIT_SCOPE=selected \
  RIDGE_LAMBDA=1e-4 \
  RIDGE_SCALE=spectral \
  RIDGE_LAMBDA_MIN=0.0 \
  DELTA_WEIGHT=1e-2 \
  AM_EXECUTION_MODE=per_document \
  SAVE_AFTER_EACH_DOCUMENT=1 \
  PHASE1_CACHE_PATH=outputs/phase1_selfdistill_qwen512/cache_last.pt \
  SYNTH_DATA_PATH=data/qasper/train/qwen_qasper_MT_task_8192.parquet \
  MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  NUM_TOKENS=512 \
  EVAL_QA_PATH=data/qasper/eval/qasper_eval_QA.parquet \
  EVAL_MT_PATH=data/qasper/eval/qasper_eval_MT.parquet \
  RUN_NAME="$runname" \
  WANDB_DISABLED=0 \
  WANDB_GROUP=B-ROUTE \
  WANDB_NOTES="first key-side arm; selectivity (MT/QA mass ratio) is the target metric" \
  bash examples/qasper2/scripts/core/train_continual_am_sparse.sh >"$log" 2>&1
  local rc=$?
  unset AM_KEY_REPOSITION AM_ROPE_THETA
  echo "MECH_KEYS_${tag}_END_EPOCH=$(date +%s) TS=$(date -Is) RC=$rc"
  local dir
  dir=$(grep -oE "Saved to /localhome/\S+" "$log" | tail -1 | awk '{print $3}')
  echo "MECH_KEYS_${tag}_RUNDIR=$dir"
  echo "MECH_KEYS_${tag}_WANDB=$(grep -oE 'https://wandb.ai/\S+/runs/[a-z0-9]+' "$log" | tail -1)"
  echo "MECH_KEYS_${tag}_INRUN=$(grep -E 'eval_(qa_forgetting|mt_acquisition)/loss ' "$log" | tr -s ' ' | tr '\n' ' ')"
  [ -n "$dir" ] && echo "MECH_KEYS_${tag}_SUMMARY=$(cat "$dir/phase2_summary.json" 2>/dev/null | tr -d '\n ')"
  echo "$dir" > "$RESDIR/rundir_${tag}.txt"
  return $rc
}

inrun_mt() {  # run_dir -> in-run MT loss (or 1e9 when missing)
  "$PY" - "$1" <<'EOF'
import json, os, sys
p = os.path.join(sys.argv[1], "phase2_summary.json")
try:
    print(json.load(open(p))["eval_metrics"]["mt_acquisition"]["loss"])
except Exception:
    print(1e9)
EOF
}

echo "=== PART B: ARMS ==="
run_arm control      freeze            -  MECH-KEYS_control      "$RESDIR/run_control.log";      RC_C=$?
run_arm keys_norepos highest_attention 0  MECH-KEYS_keys-norepos "$RESDIR/run_keys_norepos.log"; RC_N=$?
run_arm keys_repos   highest_attention 1  MECH-KEYS_keys-repos   "$RESDIR/run_keys_repos.log";   RC_R=$?

DIR_C=$(cat "$RESDIR/rundir_control.txt" 2>/dev/null)
DIR_N=$(cat "$RESDIR/rundir_keys_norepos.txt" 2>/dev/null)
DIR_R=$(cat "$RESDIR/rundir_keys_repos.txt" 2>/dev/null)

# arm 4: OMP with the reposition setting of whichever key arm had the lower in-run MT
MT_N=$(inrun_mt "$DIR_N"); MT_R=$(inrun_mt "$DIR_R")
BEST_REPOS=$("$PY" -c "print('1' if float('$MT_R') <= float('$MT_N') else '0')")
echo "MECH_KEYS_OMP_CHOICE inrun_MT_norepos=$MT_N inrun_MT_repos=$MT_R -> AM_KEY_REPOSITION=$BEST_REPOS"
run_arm omp omp "$BEST_REPOS" "MECH-KEYS_omp-repos$BEST_REPOS" "$RESDIR/run_omp.log"; RC_O=$?
DIR_O=$(cat "$RESDIR/rundir_omp.txt" 2>/dev/null)

echo "MECH_KEYS_RC control=$RC_C keys_norepos=$RC_N keys_repos=$RC_R omp=$RC_O"
echo "DIR_control=$DIR_C"
echo "DIR_keys_norepos=$DIR_N"
echo "DIR_keys_repos=$DIR_R"
echo "DIR_omp=$DIR_O"

# ---------------------------------------------------------------- independent evals
eval_one() {  # label ckpt split
  local label="$1" ckpt="$2" split="$3"
  local log="$RESDIR/eval_${label}_${split}.log"
  : > "$log"
  CHECKPOINT_PATH="$ckpt" \
  EVAL_DATA_PATH="data/qasper/eval/qasper_eval_${split}.parquet" \
  MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  RUN_NAME="MECH-KEYS_${label}_eval${split}" \
  WANDB_GROUP=B-ROUTE \
  WANDB_NOTES="first key-side arm; selectivity (MT/QA mass ratio) is the target metric" \
  WANDB_DISABLED=0 \
  "$PY" examples/qasper2/train/eval_forgetting.py >"$log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 180); do
    grep -q "Eval loss - " "$log" && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 5
  done
  echo "IMPORT ${label} ${split}: $(grep -oE '/localhome/\S*_explore/cartridges' "$log" | head -1)"
  echo "CKPTLINE ${label} ${split}: $(grep -oE 'path=.*cache_last.pt|Loading.*cache_last.pt' "$log" | head -1)"
  echo "RESULT ${label} ${split}: $(grep -oE 'Eval loss - [0-9.]+' "$log" | tail -1)  ckpt=$ckpt"
  echo "WANDB  ${label} ${split}: $(grep -oE 'https://wandb.ai/\S+/runs/[a-z0-9]+' "$log" | tail -1)"
  kill -TERM "$pid" 2>/dev/null; sleep 4; kill -KILL "$pid" 2>/dev/null
  sleep 2
}

echo "=== INDEPENDENT EVALS ==="
for SPLIT in QA MT; do
  [ -n "$DIR_C" ] && eval_one control      "$DIR_C/cache_last.pt" "$SPLIT"
  [ -n "$DIR_N" ] && eval_one keys_norepos "$DIR_N/cache_last.pt" "$SPLIT"
  [ -n "$DIR_R" ] && eval_one keys_repos   "$DIR_R/cache_last.pt" "$SPLIT"
  [ -n "$DIR_O" ] && eval_one omp          "$DIR_O/cache_last.pt" "$SPLIT"
done

# --------------------------------------------- eval-time mass_on_S + MT/QA ratio
echo "=== ROUTE-MASS DIAGNOSTIC ==="
PHASE1="$CARTRIDGES_DIR/outputs/phase1_selfdistill_qwen512/cache_last.pt"
SPECS=""
[ -n "$DIR_C" ] && SPECS="${SPECS}control:$DIR_C/cache_last.pt:$DIR_C,"
[ -n "$DIR_N" ] && SPECS="${SPECS}keys_norepos:$DIR_N/cache_last.pt:$DIR_N,"
[ -n "$DIR_R" ] && SPECS="${SPECS}keys_repos:$DIR_R/cache_last.pt:$DIR_R,"
[ -n "$DIR_O" ] && SPECS="${SPECS}omp:$DIR_O/cache_last.pt:$DIR_O,"
SPECS="${SPECS}phase1:$PHASE1:$DIR_C"
echo "CACHE_SPECS=$SPECS"
CACHE_SPECS="$SPECS" \
EVAL_SPECS="QA:data/qasper/eval/qasper_eval_QA.parquet,MT:data/qasper/eval/qasper_eval_MT.parquet" \
OUT_JSON="$CARTRIDGES_DIR/research_loop/state/diagnostics/MECH-KEYS_route_mass.json" \
"$PY" "$RESDIR/measure_route_mass.py" > "$RESDIR/route_mass.log" 2>&1
echo "ROUTE_MASS_RC=$?"
tail -30 "$RESDIR/route_mass.log"

# --------------------------------------------------------------- per-layer collection
echo "=== COLLECT PER-DOC / PER-LAYER STATS ==="
ARMS="control:$DIR_C,keys_norepos:$DIR_N,keys_repos:$DIR_R,omp:$DIR_O" \
CONTROL_ARM=control \
OUT_JSON="$CARTRIDGES_DIR/research_loop/state/diagnostics/MECH-KEYS.json" \
"$PY" "$RESDIR/collect_stats.py"
echo "COLLECT_RC=$?"

echo "MECH_KEYS_ALL_DONE_TS=$(date -Is)"
exit $(( RC_C != 0 || RC_N != 0 || RC_R != 0 ))
