#!/usr/bin/env bash
# EXP-005 launcher (HYP-T2): IDENTICAL to EXP-001 (no-IDF canonical) except the ONE
# variable under test: ENABLE_BETA off(0) -> 1 (AM per-token beta mass-bias, NNLS,
# w_j=exp(beta_j)>=0, fit on selected slots). Claim EXACTLY ONE GPU via repo flock;
# GPU0 is held by a concurrent dense job -> prefer GPU 1. The flock FD is held for
# the lifetime of THIS process (stays alive through the whole job) -> auto-releases
# on exit. No BG_STATS_PATH (USE_IDF=0, pure TF ranking).
cd /localhome/local-triv/gated-continual-cartridges_explore || exit 9
export CARTRIDGES_DIR=$PWD
export CARTRIDGES_OUTPUT_DIR=$PWD/outputs
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university

GPU_LOCK_DIR="/tmp/gpu_locks_${USER:-$(id -un)}"
mkdir -p "$GPU_LOCK_DIR"

# Try to flock a GPU non-blockingly; prefer GPU 1 (dense job holds GPU 0).
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
  echo "EXP005_NO_FREE_GPU: could not flock a GPU under $GPU_LOCK_DIR" >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES="$CLAIMED"
export MASTER_PORT=$((29500 + CLAIMED))
echo "EXP005_CLAIMED_GPU=$CLAIMED MASTER_PORT=$MASTER_PORT (flock held on $GPU_LOCK_DIR/gpu${CLAIMED}.lock)"
echo "EXP005_E2E_START_EPOCH=$(date +%s)"
echo "EXP005_START_TS=$(date -Is)"

# --- ONE variable vs EXP-001: ENABLE_BETA=1 (was 0). BETA_FIT_SCOPE=selected explicit.
ENABLE_BETA=1 \
BETA_FIT_SCOPE=selected \
USE_IDF=0 \
SLOT_SELECTION=tfidf \
GRANULARITY=per_layer \
TOP_T=64 \
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
RUN_NAME=exp005_am_sparse_noidf_beta_top64_perlayer \
WANDB_DISABLED=1 \
bash examples/qasper2/scripts/core/train_continual_am_sparse.sh
RC=$?

echo "EXP005_E2E_END_EPOCH=$(date +%s)"
echo "EXP005_END_TS=$(date -Is)"
echo "EXP005_PYTHON_RC=$RC"
exit $RC
