#!/usr/bin/env bash
# MECH-QUERIES (B-CASCADE): reference-query count n per KV head, at fixed TOP_T=32.
#
# ONE variable: MAX_QUERIES_PER_HEAD in {64, 256, 1024}. Everything else canonical
# EXP-007-top32. n=64 is the CONTROL and must reproduce EXP-007-top32 exactly
# (QA 2.1766157150268555 / MT 2.5483615398406982).
#
# PYTHONPATH is pinned to the _explore repo (RUNBOOK §6.10): without it
# `import cartridges` resolves to the SIBLING repo, which HAS max_queries_per_head
# (so the guard would not fire) but LACKS the mass_on_S / |v| instrumentation.
set -u
cd /localhome/local-triv/gated-continual-cartridges_explore || exit 9
export CARTRIDGES_DIR=$PWD
export CARTRIDGES_OUTPUT_DIR=$PWD/outputs
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export PYTHONPATH="$CARTRIDGES_DIR:${PYTHONPATH:-}"

RESDIR="$CARTRIDGES_DIR/research_loop/results/MECH-QUERIES"
mkdir -p "$RESDIR"
exec > "$RESDIR/wrapper.log" 2>&1

GPU_LOCK_DIR="/tmp/gpu_locks_${USER:-$(id -un)}"
mkdir -p "$GPU_LOCK_DIR"
CLAIMED=""
for idx in 1 0; do
  lockfile="$GPU_LOCK_DIR/gpu${idx}.lock"
  exec {fd}>"$lockfile"
  if flock -n "$fd"; then CLAIMED="$idx"; break; else exec {fd}>&-; fi
done
if [ -z "$CLAIMED" ]; then echo "MECH_QUERIES_NO_FREE_GPU" >&2; exit 3; fi
export CUDA_VISIBLE_DEVICES="$CLAIMED"
export MASTER_PORT=$((29500 + CLAIMED))
echo "MECH_QUERIES_CLAIMED_GPU=$CLAIMED MASTER_PORT=$MASTER_PORT"

# Hard pre-flight: the import MUST be _explore AND must carry the new instrumentation.
"$CARTRIDGES_DIR/.venv/bin/python" - <<'PY' || exit 8
import os, sys, inspect
import cartridges
p = os.path.dirname(cartridges.__file__)
print("CARTRIDGES_IMPORT_PATH=" + p)
assert p.endswith("gated-continual-cartridges_explore/cartridges"), "WRONG cartridges repo"
from cartridges.am.finetune import AttentionMatchingFinetuningConfig as C
assert "max_queries_per_head" in C.model_fields
src = inspect.getsource(__import__("cartridges.am.value_solve", fromlist=["x"]).guarded_sparse_am_value_update)
assert "mass_on_S_mean" in src, "guarded solve lacks mass_on_S instrumentation"
print("PREFLIGHT_OK")
PY

run_one() {  # n
  local n="$1"
  local log="$RESDIR/run_n${n}.log"
  echo "MECH_QUERIES_n${n}_START_EPOCH=$(date +%s) TS=$(date -Is)"
  MAX_QUERIES_PER_HEAD="$n" \
  USE_IDF=0 \
  SLOT_SELECTION=tfidf \
  GRANULARITY=per_layer \
  TOP_T=32 \
  TARGET_MODE=cartridge_plus_doc \
  KEY_MODE=freeze \
  RIDGE_LAMBDA=1e-4 \
  RIDGE_SCALE=spectral \
  DELTA_WEIGHT=1e-2 \
  AM_EXECUTION_MODE=per_document \
  PHASE1_CACHE_PATH=outputs/phase1_selfdistill_qwen512/cache_last.pt \
  SYNTH_DATA_PATH=data/qasper/train/qwen_qasper_MT_task_8192.parquet \
  MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  NUM_TOKENS=512 \
  EVAL_QA_PATH=data/qasper/eval/qasper_eval_QA.parquet \
  EVAL_MT_PATH=data/qasper/eval/qasper_eval_MT.parquet \
  RUN_NAME="MECH-QUERIES_n${n}" \
  WANDB_DISABLED=0 \
  WANDB_GROUP=B-CASCADE \
  WANDB_NOTES="reference-query count n vs |v| collapse and MT" \
  bash examples/qasper2/scripts/core/train_continual_am_sparse.sh >"$log" 2>&1
  local rc=$?
  echo "MECH_QUERIES_n${n}_END_EPOCH=$(date +%s) TS=$(date -Is) RC=$rc"
  echo "MECH_QUERIES_n${n}_RUNDIR=$(grep -oE 'outputs/[0-9-]+-continual_am_sparse/[0-9a-f-]+' "$log" | tail -1)"
  return $rc
}

RCS=""
for N in ${NS:-64 256 1024}; do
  run_one "$N"; RCS="$RCS n${N}=$?"
done
echo "MECH_QUERIES_RCS:$RCS"
echo "MECH_QUERIES_ALL_DONE_TS=$(date -Is)"
