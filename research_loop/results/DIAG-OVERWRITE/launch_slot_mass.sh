#!/usr/bin/env bash
# DIAG-OVERWRITE Q3: forward-pass-only slot-mass diagnostic on the untouched Phase-1
# cartridge, using the written union from the DIAG-OVERWRITE top-32 run.
set -u
REPO=/localhome/local-triv/gated-continual-cartridges_explore
SNAP=/tmp/amsnap_ovw
PY="$REPO/.venv/bin/python"

export CARTRIDGES_DIR="$SNAP"
export CARTRIDGES_OUTPUT_DIR="$REPO/outputs"
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export PYTHONPATH="$SNAP:${PYTHONPATH:-}"

RESDIR="$REPO/research_loop/results/DIAG-OVERWRITE"
exec > "$RESDIR/wrapper_mass.log" 2>&1
echo "MASS_START_TS=$(date -Is)"
echo "SNAPSHOT_MANIFEST=$(cd "$SNAP" && find cartridges examples -type f -name '*.py' | LC_ALL=C sort | xargs sha256sum | awk '{print $1\"  \"$2}' | LC_ALL=C sort | sha256sum | cut -d' ' -f1)"

GPU_LOCK_DIR="/tmp/gpu_locks_${USER:-$(id -un)}"
mkdir -p "$GPU_LOCK_DIR"
CLAIMED=""
for idx in 0 1; do
  lockfile="$GPU_LOCK_DIR/gpu${idx}.lock"
  exec {fd}>"$lockfile"
  if flock -n "$fd"; then CLAIMED="$idx"; break; else exec {fd}>&-; fi
done
if [ -z "$CLAIMED" ]; then echo "MASS_NO_FREE_GPU" >&2; exit 3; fi
export CUDA_VISIBLE_DEVICES="$CLAIMED"
echo "MASS_CLAIMED_GPU=$CLAIMED"
cd /tmp && "$PY" -c "import cartridges,os;print('CARTRIDGES_IMPORT_PATH='+os.path.dirname(cartridges.__file__))"

RUN_DIR=$(cat "$RESDIR/run_dir.txt")
echo "MASS_RUN_DIR=$RUN_DIR"
PHASE1_CACHE="$REPO/outputs/phase1_selfdistill_qwen512/cache_last.pt" \
EVAL_SPECS="QA:$REPO/data/qasper/eval/qasper_eval_QA.parquet,MT:$REPO/data/qasper/eval/qasper_eval_MT.parquet" \
SLOTS_RUN_DIR="$RUN_DIR" \
OUT_JSON="$REPO/research_loop/state/diagnostics/DIAG-OVERWRITE_slot_mass.json" \
OUT_NPZ="$REPO/research_loop/state/diagnostics/DIAG-OVERWRITE_slot_mass.npz" \
"$PY" "$RESDIR/measure_slot_mass.py" >"$RESDIR/slot_mass.log" 2>&1
RC=$?
echo "MASS_RC=$RC"
echo "SNAPSHOT_MANIFEST_AFTER=$(cd "$SNAP" && find cartridges examples -type f -name '*.py' | LC_ALL=C sort | xargs sha256sum | awk '{print $1\"  \"$2}' | LC_ALL=C sort | sha256sum | cut -d' ' -f1)"
echo "MASS_ALL_DONE_TS=$(date -Is)"
exit $RC
