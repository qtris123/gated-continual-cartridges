#!/usr/bin/env bash
# DIAG-KEYSPACE (W1 MEASURE): 128-dim per-head QUERY-space geometry. Forward passes only,
# no training, no solve, no cache writes, NO source edits. ONE flock-claimed GPU held for
# the whole lifetime. Imports pinned to a frozen git-archive snapshot (RUNBOOK 9c-bis)
# because MECH-KEYS is editing cartridges/am/key_select.py concurrently.
set -u
cd /localhome/local-triv/gated-continual-cartridges_explore || exit 9
export CARTRIDGES_DIR=$PWD
export CARTRIDGES_OUTPUT_DIR=$PWD/outputs
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university

export AMSNAP=/tmp/amsnap_kspace
export PYTHONPATH="$AMSNAP:${PYTHONPATH:-}"
export SNAP_GIT_HEAD="$(git -C "$CARTRIDGES_DIR" rev-parse HEAD)"
export SNAP_SHA="$(find "$AMSNAP" -name '*.py' | sort | xargs sha256sum | sha256sum | cut -d' ' -f1)"

PY="$CARTRIDGES_DIR/.venv/bin/python"
RESDIR="$CARTRIDGES_DIR/research_loop/results/DIAG-KEYSPACE"
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
[ -z "$CLAIMED" ] && { echo "KEYSPACE_NO_FREE_GPU"; exit 3; }
export CUDA_VISIBLE_DEVICES="$CLAIMED"
echo "KEYSPACE_CLAIMED_GPU=$CLAIMED (flock held on $GPU_LOCK_DIR/gpu${CLAIMED}.lock)"

# import-path probe MUST run from /tmp (RUNBOOK 9c-bis: cwd is sys.path[0] for python -c)
(cd /tmp && "$PY" -c "import cartridges,os;print('CARTRIDGES_IMPORT_PATH='+os.path.dirname(cartridges.__file__))")

PHASE1_CACHE="$CARTRIDGES_DIR/outputs/phase1_selfdistill_qwen512/cache_last.pt" \
EVAL_SPECS="QA:$CARTRIDGES_DIR/data/qasper/eval/qasper_eval_QA.parquet,MT:$CARTRIDGES_DIR/data/qasper/eval/qasper_eval_MT.parquet" \
SLOTS_RUN_DIR="$CARTRIDGES_DIR/outputs/2026-07-28-19-27-08-continual_am_sparse/43a46f4b-5341-4a0c-98f3-2305daa8247f" \
OUT_JSON="$CARTRIDGES_DIR/research_loop/state/diagnostics/DIAG-KEYSPACE.json" \
OUT_NPZ="$CARTRIDGES_DIR/research_loop/state/diagnostics/DIAG-KEYSPACE.npz" \
MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507" \
QCHUNK=256 N_SUB=8192 OPT_STEPS=300 DIAG_SEED=0 \
WANDB_DISABLED=0 \
RUN_NAME="DIAG-KEYSPACE_query-geometry" \
WANDB_GROUP="B-ROUTE" \
WANDB_NOTES="128-dim per-head query second moments Q0 (QA vs MT): rho_key, eff rank, principal angles, mean-query separation, achievable key selectivity bound" \
"$PY" "$RESDIR/measure_query_geometry.py"
RC=$?
echo "DIAG_RC=$RC"

echo "POSTRUN_SNAP_SHA=$(find "$AMSNAP" -name '*.py' | sort | xargs sha256sum | sha256sum | cut -d' ' -f1)"
echo "POSTRUN_GIT_HEAD=$(git -C "$CARTRIDGES_DIR" rev-parse HEAD)"
echo "END_TS=$(date -Is)"
