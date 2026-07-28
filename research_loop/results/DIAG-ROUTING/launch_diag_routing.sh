#!/usr/bin/env bash
# DIAG-ROUTING (W1 MEASURE): routing-geometry diagnostic. Forward passes only, no
# training, no cache writes, no source edits. ONE flock-claimed GPU held for the whole
# lifetime. Imports pinned to a frozen git-archive snapshot (RUNBOOK 9c-bis) because
# MECH-QUERIES is editing cartridges/am/*.py concurrently.
set -u
cd /localhome/local-triv/gated-continual-cartridges_explore || exit 9
export CARTRIDGES_DIR=$PWD
export CARTRIDGES_OUTPUT_DIR=$PWD/outputs
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university

export AMSNAP=/tmp/amsnap_diagrouting
export PYTHONPATH="$AMSNAP:${PYTHONPATH:-}"
export SNAP_GIT_HEAD="$(git -C "$CARTRIDGES_DIR" rev-parse HEAD)"
export SNAP_SHA="$(find "$AMSNAP" -name '*.py' | sort | xargs sha256sum | sha256sum | cut -d' ' -f1)"

PY="$CARTRIDGES_DIR/.venv/bin/python"
RESDIR="$CARTRIDGES_DIR/research_loop/results/DIAG-ROUTING"
exec > "$RESDIR/wrapper.log" 2>&1

echo "START_TS=$(date -Is)"
echo "SNAP_GIT_HEAD=$SNAP_GIT_HEAD"
echo "SNAP_SHA=$SNAP_SHA"

GPU_LOCK_DIR="/tmp/gpu_locks_${USER:-$(id -un)}"
mkdir -p "$GPU_LOCK_DIR"
CLAIMED=""
for attempt in $(seq 1 240); do
  for idx in 0 1; do
    exec {fd}>"$GPU_LOCK_DIR/gpu${idx}.lock"
    if flock -n "$fd"; then CLAIMED="$idx"; break; else exec {fd}>&-; fi
  done
  [ -n "$CLAIMED" ] && break
  sleep 15
done
[ -z "$CLAIMED" ] && { echo "DIAGROUTING_NO_FREE_GPU"; exit 3; }
export CUDA_VISIBLE_DEVICES="$CLAIMED"
echo "DIAGROUTING_CLAIMED_GPU=$CLAIMED (flock held on $GPU_LOCK_DIR/gpu${CLAIMED}.lock)"

# import-path probe MUST run from /tmp (RUNBOOK 9c-bis: cwd is sys.path[0] for python -c)
(cd /tmp && "$PY" -c "import cartridges,os;print('CARTRIDGES_IMPORT_PATH='+os.path.dirname(cartridges.__file__))")

PHASE1_CACHE="$CARTRIDGES_DIR/outputs/phase1_selfdistill_qwen512/cache_last.pt" \
EVAL_SPECS="QA:$CARTRIDGES_DIR/data/qasper/eval/qasper_eval_QA.parquet,MT:$CARTRIDGES_DIR/data/qasper/eval/qasper_eval_MT.parquet" \
SLOTS_RUN_DIR="$CARTRIDGES_DIR/outputs/2026-07-28-19-27-08-continual_am_sparse/43a46f4b-5341-4a0c-98f3-2305daa8247f" \
OUT_JSON="$CARTRIDGES_DIR/research_loop/state/diagnostics/DIAG-ROUTING.json" \
OUT_NPZ="$CARTRIDGES_DIR/research_loop/state/diagnostics/DIAG-ROUTING.npz" \
MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507" \
QCHUNK=256 \
WANDB_DISABLED=0 \
RUN_NAME="DIAG-ROUTING_geometry" \
WANDB_GROUP="B-ROUTE" \
WANDB_NOTES="QA routing Gram spectrum, MT null-space energy, mass_on_S, QA/MT overlap" \
"$PY" "$RESDIR/measure_routing_geometry.py"
RC=$?
echo "DIAG_RC=$RC"

echo "POSTRUN_SNAP_SHA=$(find "$AMSNAP" -name '*.py' | sort | xargs sha256sum | sha256sum | cut -d' ' -f1)"
echo "POSTRUN_GIT_HEAD=$(git -C "$CARTRIDGES_DIR" rev-parse HEAD)"
echo "END_TS=$(date -Is)"
