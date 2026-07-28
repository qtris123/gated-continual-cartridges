#!/usr/bin/env bash
# ORACLE-WRITE (B-ROUTE): write-ceiling oracle vs solved values, canonical top32 config.
#
# TWO runs, SEQUENTIAL, on ONE flock-claimed GPU:
#   A) CONTROL  AM_ORACLE_WRITE=0  -> must reproduce EXP-007 top32 (QA 2.1766 / MT 2.5484)
#                                     => proves the new flag is bit-identical when off,
#                                        and removes the sibling-import confound.
#   B) ORACLE   AM_ORACLE_WRITE=1  -> teacher's own document values written into the
#                                     selected slots instead of solving for them.
# Everything else is held at the canonical EXP-007 top32 config.
# PYTHONPATH is pinned to the _explore repo (RUNBOOK §6.10): without it `import cartridges`
# resolves to the SIBLING repo, which has no `oracle_write` field (verified).
set -u
cd /localhome/local-triv/gated-continual-cartridges_explore || exit 9
export CARTRIDGES_DIR=$PWD
export CARTRIDGES_OUTPUT_DIR=$PWD/outputs
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export PYTHONPATH="$CARTRIDGES_DIR:${PYTHONPATH:-}"

RESDIR="$CARTRIDGES_DIR/research_loop/results/ORACLE-WRITE"
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
if [ -z "$CLAIMED" ]; then
  echo "ORACLE_WRITE_NO_FREE_GPU" >&2
  exit 3
fi
export CUDA_VISIBLE_DEVICES="$CLAIMED"
export MASTER_PORT=$((29500 + CLAIMED))
echo "ORACLE_WRITE_CLAIMED_GPU=$CLAIMED MASTER_PORT=$MASTER_PORT"
python -c "import cartridges,os;print('CARTRIDGES_IMPORT_PATH='+os.path.dirname(cartridges.__file__))"

run_one() {
  local tag="$1" oracle="$2" runname="$3" log="$4"
  echo "ORACLE_WRITE_${tag}_START_EPOCH=$(date +%s) TS=$(date -Is)"
  AM_ORACLE_WRITE="$oracle" \
  AM_ORACLE_WRITE_ASSIGN=mass_ranked \
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
  RUN_NAME="$runname" \
  WANDB_DISABLED=0 \
  WANDB_GROUP=B-ROUTE \
  WANDB_NOTES="write-ceiling oracle vs solved values (EXP-007-top32)" \
  bash examples/qasper2/scripts/core/train_continual_am_sparse.sh >"$log" 2>&1
  local rc=$?
  echo "ORACLE_WRITE_${tag}_END_EPOCH=$(date +%s) TS=$(date -Is) RC=$rc"
  return $rc
}

run_one CONTROL 0 ORACLE-WRITE_control-solved-top32 "$RESDIR/run_control.log"
RC_C=$?
run_one ORACLE  1 ORACLE-WRITE_teacher-kv-top32     "$RESDIR/run_oracle.log"
RC_O=$?

echo "ORACLE_WRITE_RC_CONTROL=$RC_C ORACLE_WRITE_RC_ORACLE=$RC_O"
echo "ORACLE_WRITE_ALL_DONE_TS=$(date -Is)"
exit $(( RC_C != 0 || RC_O != 0 ))
