#!/usr/bin/env bash
# MECH-QUERIES-B part 2: independent eval_forgetting.py on every arm's checkpoint x
# BOTH splits, then the route-mass diagnostic (total cartridge attention mass +
# eval-time mass_on_S), then the |dV| / per-layer collector. One flock-claimed GPU.
#
# Imports pinned to the frozen snapshot /tmp/amsnap_mqb (RUNBOOK 9c-bis); the driver
# and the route-mass diagnostic are run from the snapshot copy too.
#
# RUN_SPECS="n64_dw1e-2:<run_dir>,n1024_dw0.16:<run_dir>,n16384_dw2.56:<run_dir>"
set -u
REPO=/localhome/local-triv/gated-continual-cartridges_explore
SNAP=/tmp/amsnap_mqb
cd "$REPO" || exit 9
export CARTRIDGES_DIR=$REPO
export CARTRIDGES_OUTPUT_DIR=$REPO/outputs
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export PYTHONPATH="$SNAP:${PYTHONPATH:-}"
PY="$REPO/.venv/bin/python"

RESDIR="$CARTRIDGES_DIR/research_loop/results/MECH-QUERIES-B"
exec > "$RESDIR/wrapper_evals.log" 2>&1

RUN_SPECS="${RUN_SPECS:?set RUN_SPECS}"
PHASE1="$CARTRIDGES_DIR/outputs/phase1_selfdistill_qwen512/cache_last.pt"

GPU_LOCK_DIR="/tmp/gpu_locks_${USER:-$(id -un)}"
CLAIMED=""
for attempt in $(seq 1 240); do
  for idx in 1 0; do
    exec {fd}>"$GPU_LOCK_DIR/gpu${idx}.lock"
    if flock -n "$fd"; then CLAIMED="$idx"; break; else exec {fd}>&-; fi
  done
  [ -n "$CLAIMED" ] && break
  sleep 15
done
[ -z "$CLAIMED" ] && { echo "NO_FREE_GPU"; exit 3; }
export CUDA_VISIBLE_DEVICES="$CLAIMED"
echo "EVALS_CLAIMED_GPU=$CLAIMED"
(cd /tmp && "$PY" -c "import cartridges,os;print('CARTRIDGES_IMPORT_PATH='+os.path.dirname(cartridges.__file__))")

eval_one() {  # label ckpt split
  local label="$1" ckpt="$2" split="$3"
  local log="$RESDIR/eval_${label}_${split}.log"
  : > "$log"
  CHECKPOINT_PATH="$ckpt" \
  EVAL_DATA_PATH="data/qasper/eval/qasper_eval_${split}.parquet" \
  MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  RUN_NAME="MECH-QUERIES-B_${label}_eval${split}" \
  WANDB_GROUP=B-CASCADE \
  WANDB_NOTES="n-scaled trust region: DELTA_WEIGHT proportional to n" \
  WANDB_DISABLED=0 \
  "$PY" "$SNAP/examples/qasper2/train/eval_forgetting.py" >"$log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 180); do
    grep -q "Eval loss - " "$log" && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 5
  done
  echo "IMPORT ${label} ${split}: $(grep -oE '/(tmp|localhome)\S*/cartridges' "$log" | head -1)"
  echo "CKPT   ${label} ${split}: intended_path_mentions=$(grep -c -- "$ckpt" "$log") path=$ckpt"
  echo "RESULT ${label} ${split}: $(grep -oE 'Eval loss - [0-9.]+' "$log" | tail -1)  ckpt=$ckpt"
  echo "WANDB  ${label} ${split}: $(grep -oE 'https://wandb.ai/\S+/runs/\S+' "$log" | tail -1)"
  kill -TERM "$pid" 2>/dev/null; sleep 4; kill -KILL "$pid" 2>/dev/null
  sleep 2
}

CACHE_SPECS=""
COLLECT_SPECS=""
for spec in ${RUN_SPECS//,/ }; do
  LBL="${spec%%:*}"; DIR="${spec#*:}"
  for SPLIT in QA MT; do eval_one "$LBL" "$DIR/cache_last.pt" "$SPLIT"; done
  CACHE_SPECS="${CACHE_SPECS}${LBL}:$DIR/cache_last.pt:$DIR,"
  COLLECT_SPECS="${COLLECT_SPECS}${LBL}:$DIR,"
done
# Phase-1 reference measured against the FIRST arm's (= the n=64 control's) slot set,
# exactly as MECH-QUERIES did, so the 0.588 total-cartridge-mass number is comparable.
FIRST_DIR="$(echo "$RUN_SPECS" | cut -d, -f1)"; FIRST_DIR="${FIRST_DIR#*:}"
CACHE_SPECS="${CACHE_SPECS}phase1:$PHASE1:$FIRST_DIR"
COLLECT_SPECS="${COLLECT_SPECS%,}"

echo "=== EVALS DONE, route-mass diagnostic ==="
CACHE_SPECS="$CACHE_SPECS" \
EVAL_SPECS="QA:data/qasper/eval/qasper_eval_QA.parquet,MT:data/qasper/eval/qasper_eval_MT.parquet" \
OUT_JSON="$CARTRIDGES_DIR/research_loop/state/diagnostics/MECH-QUERIES-B_route_mass.json" \
"$PY" "$SNAP/measure_route_mass.py"
echo "ROUTE_MASS_RC=$?"

echo "=== ARTIFACT COLLECTOR (incl. |dV| vs Phase-1) ==="
RUNS="$COLLECT_SPECS" \
PHASE1="$PHASE1" \
OUT_JSON="$CARTRIDGES_DIR/research_loop/state/diagnostics/MECH-QUERIES-B_solve.json" \
"$PY" "$RESDIR/collect_diag_b.py"
echo "COLLECT_RC=$?"
echo "MQB_EVALS_ALL_DONE_TS=$(date -Is)"
