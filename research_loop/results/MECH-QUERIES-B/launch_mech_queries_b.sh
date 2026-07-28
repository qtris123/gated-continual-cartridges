#!/usr/bin/env bash
# MECH-QUERIES-B (B-CASCADE): resolve MECH-QUERIES' own confound.
#
# MECH-QUERIES swept n = MAX_QUERIES_PER_HEAD at FIXED DELTA_WEIGHT=1e-2. The guarded
# solve stacks [X_new (n x t); sqrt(w) I (t x t)], so the data Gram scales with n while
# the trust region contributes a fixed t rows => its relative pull decays like 1/n.
# Here we hold the relative pull CONSTANT by scaling w with n:  w(n) = 1e-2 * n/64.
#
#   n=64    -> DELTA_WEIGHT=1e-2   (CONTROL, must reproduce EXP-007-top32 exactly)
#   n=1024  -> DELTA_WEIGHT=0.16
#   n=16384 -> DELTA_WEIGHT=2.56
#
# Everything else canonical EXP-007-top32 / MECH-QUERIES.
#
# IMPORT PIN (RUNBOOK 9c-bis): both the `cartridges` package AND the driver come from a
# frozen `git archive HEAD` snapshot at /tmp/amsnap_mqb, so a concurrent source-editing
# worker cannot leak half-finished edits into these numbers. The shell wrapper is the
# snapshot's own, with only the final `python <driver>` line repointed at the snapshot
# driver + the repo venv interpreter (diff = exactly 1 line, see wrapper.log).
set -u
REPO=/localhome/local-triv/gated-continual-cartridges_explore
SNAP=/tmp/amsnap_mqb
cd "$REPO" || exit 9
export CARTRIDGES_DIR=$REPO           # data/ + .venv live here (same as MECH-QUERIES)
export CARTRIDGES_OUTPUT_DIR=$REPO/outputs
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export PYTHONPATH="$SNAP:${PYTHONPATH:-}"   # snapshot WINS over repo and sibling repo
export AM_PYBIN="$REPO/.venv/bin/python"
export AM_DRIVER="$SNAP/examples/qasper2/train/continual_am_sparse.py"

RESDIR="$CARTRIDGES_DIR/research_loop/results/MECH-QUERIES-B"
mkdir -p "$RESDIR"
exec > "$RESDIR/wrapper.log" 2>&1

echo "SNAPSHOT_HEAD=$(cat /tmp/amsnap_mqb_HEAD.txt)"
echo "SNAPSHOT_MANIFEST=$(cat /tmp/amsnap_mqb_manifest.txt)"
echo "REPO_HEAD_NOW=$(git -C "$REPO" rev-parse HEAD)"
echo "PINNED_WRAPPER_DIFF:"; diff "$SNAP/examples/qasper2/scripts/core/train_continual_am_sparse.sh" "$SNAP/train_am_sparse_pinned.sh"

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
if [ -z "$CLAIMED" ]; then echo "MECH_QUERIES_B_NO_FREE_GPU" >&2; exit 3; fi
export CUDA_VISIBLE_DEVICES="$CLAIMED"
export MASTER_PORT=$((29500 + CLAIMED))
echo "MECH_QUERIES_B_CLAIMED_GPU=$CLAIMED MASTER_PORT=$MASTER_PORT"

# Hard pre-flight: import MUST be the SNAPSHOT and must carry MECH-002's instrumentation.
# Probe from /tmp, never the repo root (RUNBOOK 9c-bis false-negative trap).
(cd /tmp && "$AM_PYBIN" - <<'PY') || exit 8
import os, sys, inspect
import cartridges
p = os.path.dirname(cartridges.__file__)
print("CARTRIDGES_IMPORT_PATH=" + p)
assert p == "/tmp/amsnap_mqb/cartridges", "NOT the frozen snapshot: " + p
from cartridges.am.finetune import AttentionMatchingFinetuningConfig as C
assert "max_queries_per_head" in C.model_fields
src = inspect.getsource(__import__("cartridges.am.value_solve", fromlist=["x"]).guarded_sparse_am_value_update)
assert "mass_on_S_mean" in src, "guarded solve lacks mass_on_S instrumentation"
assert "v_delta_absmax" in src, "guarded solve lacks v_delta instrumentation"
print("PREFLIGHT_OK")
PY

run_one() {  # n  delta_weight  slug
  local n="$1" dw="$2" slug="$3"
  local log="$RESDIR/run_n${n}_dw${slug}.log"
  echo "MQB_n${n}_dw${slug}_START_EPOCH=$(date +%s) TS=$(date -Is)"
  MAX_QUERIES_PER_HEAD="$n" \
  DELTA_WEIGHT="$dw" \
  USE_IDF=0 \
  SLOT_SELECTION=tfidf \
  GRANULARITY=per_layer \
  TOP_T=32 \
  TARGET_MODE=cartridge_plus_doc \
  KEY_MODE=freeze \
  RIDGE_LAMBDA=1e-4 \
  RIDGE_SCALE=spectral \
  AM_EXECUTION_MODE=per_document \
  PHASE1_CACHE_PATH=outputs/phase1_selfdistill_qwen512/cache_last.pt \
  SYNTH_DATA_PATH=data/qasper/train/qwen_qasper_MT_task_8192.parquet \
  MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  NUM_TOKENS=512 \
  EVAL_QA_PATH=data/qasper/eval/qasper_eval_QA.parquet \
  EVAL_MT_PATH=data/qasper/eval/qasper_eval_MT.parquet \
  RUN_NAME="MECH-QUERIES-B_n${n}_dw${slug}" \
  WANDB_DISABLED=0 \
  WANDB_GROUP=B-CASCADE \
  WANDB_NOTES="n-scaled trust region: DELTA_WEIGHT proportional to n" \
  bash "$SNAP/train_am_sparse_pinned.sh" >"$log" 2>&1
  local rc=$?
  echo "MQB_n${n}_dw${slug}_END_EPOCH=$(date +%s) TS=$(date -Is) RC=$rc"
  echo "MQB_n${n}_dw${slug}_RUNDIR=$(grep -oE 'outputs/[0-9-]+-continual_am_sparse/[0-9a-f-]+' "$log" | tail -1)"
  echo "MQB_n${n}_dw${slug}_EVALS=$(grep -oE 'Eval loss - [0-9.]+' "$log" | tr '\n' ' ')"
  echo "MQB_n${n}_dw${slug}_WANDB=$(grep -oE 'https://wandb.ai/\S+/runs/\S+' "$log" | tail -1)"
  return $rc
}

RCS=""
# CONTROL FIRST: if it does not reproduce, everything else is void.
run_one 64    1e-2  "1e-2"; RCS="$RCS n64=$?"
run_one 1024  0.16  "0.16"; RCS="$RCS n1024=$?"
run_one 16384 2.56  "2.56"; RCS="$RCS n16384=$?"
echo "MQB_RCS:$RCS"
echo "SNAPSHOT_MANIFEST_AFTER=$(find /tmp/amsnap_mqb -type f -name '*.py' | sort | xargs sha256sum | sha256sum)"
echo "REPO_HEAD_AFTER=$(git -C "$REPO" rev-parse HEAD)"
echo "MQB_ALL_DONE_TS=$(date -Is)"
