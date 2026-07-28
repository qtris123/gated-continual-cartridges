#!/usr/bin/env bash
# EXP-008 (HYP-T1): TARGET_MODE sweep vs canonical cartridge_plus_doc (EXP-001).
# ONE variable under test = TARGET_MODE; all else held at the EXP-001 no-IDF
# canonical (USE_IDF=0, tfidf, per_layer, top_t=64, freeze, ridge 1e-4 spectral,
# delta 1e-2, per_document). The enum (cartridges/am/finetune.py L65) has exactly
# 3 values: cartridge_plus_doc (=EXP-001, skipped), self, teacher_attention.
# Runs the 2 non-canonical modes SEQUENTIALLY on ONE flock-claimed GPU (flock FD
# held for the wrapper lifetime, auto-released on exit). No BG_STATS (USE_IDF=0).
# PYTHONPATH pinned to the explore repo so `import cartridges` resolves to it
# (byte-identical to sibling for this code path => EXP-001-comparable).
set -u
cd /localhome/local-triv/gated-continual-cartridges_explore || exit 9
export CARTRIDGES_DIR=$PWD
export CARTRIDGES_OUTPUT_DIR=$PWD/outputs
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export PYTHONPATH="$CARTRIDGES_DIR:${PYTHONPATH:-}"

RESDIR="$CARTRIDGES_DIR/research_loop/results/EXP-008"
mkdir -p "$RESDIR"
exec > "$RESDIR/wrapper.log" 2>&1

GPU_LOCK_DIR="/tmp/gpu_locks_${USER:-$(id -un)}"
mkdir -p "$GPU_LOCK_DIR"

# Claim EXACTLY ONE GPU non-blockingly; prefer gpu1 (gpu0 holds a REF-ICL job).
CLAIMED=""
for idx in 1 0; do
  lockfile="$GPU_LOCK_DIR/gpu${idx}.lock"
  exec {fd}>"$lockfile"
  if flock -n "$fd"; then
    CLAIMED="$idx"
    break
  else
    exec {fd}>&-
  fi
done
if [ -z "$CLAIMED" ]; then
  echo "EXP008_NO_FREE_GPU: could not flock a GPU under $GPU_LOCK_DIR" >&2
  exit 3
fi
export CUDA_VISIBLE_DEVICES="$CLAIMED"
export MASTER_PORT=$((29500 + CLAIMED))
echo "EXP008_CLAIMED_GPU=$CLAIMED MASTER_PORT=$MASTER_PORT PYTHONPATH_head=${PYTHONPATH%%:*} (flock held on $GPU_LOCK_DIR/gpu${CLAIMED}.lock)"

# ---- one run = the no-IDF canonical (EXP-001), everything held except TARGET_MODE ----
run_one() {
  local tag="$1" target_mode="$2" runname="$3" log="$4"
  echo "EXP008_${tag}_START_EPOCH=$(date +%s) TS=$(date -Is) TARGET_MODE=$target_mode"
  USE_IDF=0 \
  SLOT_SELECTION=tfidf \
  GRANULARITY=per_layer \
  TOP_T=64 \
  TARGET_MODE="$target_mode" \
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
  WANDB_DISABLED=1 \
  bash examples/qasper2/scripts/core/train_continual_am_sparse.sh >"$log" 2>&1
  local rc=$?
  echo "EXP008_${tag}_END_EPOCH=$(date +%s) TS=$(date -Is) RC=$rc"
  return $rc
}

# ---- Run 1: TARGET_MODE=self ----
run_one SELF self exp008_self "$RESDIR/run_self.log"
RC_SELF=$?

# ---- Run 2: TARGET_MODE=teacher_attention ---- (sequential, same GPU)
run_one TEACHER teacher_attention exp008_teacher_attention "$RESDIR/run_teacher_attention.log"
RC_TEACHER=$?

echo "EXP008_RC_SELF=$RC_SELF EXP008_RC_TEACHER=$RC_TEACHER"
echo "EXP008_ALL_DONE_TS=$(date -Is)"
exit $(( RC_SELF != 0 || RC_TEACHER != 0 ))
