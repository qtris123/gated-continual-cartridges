#!/usr/bin/env bash
# EXP-005b launcher (HYP-T2 retry): ENABLE_BETA=1 WITH RIDGE_LAMBDA=0.
# EXP-005 (ENABLE_BETA=1, RIDGE_LAMBDA=1e-4) CRASHED with cholesky not-positive-definite
# in _ridge_lstsq. RIDGE_LAMBDA=0 forces effective_ridge_lambda=0 -> _ridge_lstsq takes
# the robust OLS gels branch (NOT cholesky), avoiding the crash (per EXP-004).
# ONE variable vs EXP-004 (lambda=0 baseline, beta OFF): ENABLE_BETA off->1, holding lambda=0.
# Claim EXACTLY ONE GPU via repo flock; prefer GPU 1. flock FD held for process lifetime.
# No BG_STATS_PATH (USE_IDF=0, pure TF ranking).
cd /localhome/local-triv/gated-continual-cartridges_explore || exit 9
export CARTRIDGES_DIR=$PWD
export CARTRIDGES_OUTPUT_DIR=$PWD/outputs
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university

GPU_LOCK_DIR="/tmp/gpu_locks_${USER:-$(id -un)}"
mkdir -p "$GPU_LOCK_DIR"

# Try to flock a GPU non-blockingly; prefer GPU 1.
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
  echo "EXP005B_NO_FREE_GPU: could not flock a GPU under $GPU_LOCK_DIR" >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES="$CLAIMED"
export MASTER_PORT=$((29500 + CLAIMED))
echo "EXP005B_CLAIMED_GPU=$CLAIMED MASTER_PORT=$MASTER_PORT (flock held on $GPU_LOCK_DIR/gpu${CLAIMED}.lock)"
echo "EXP005B_E2E_START_EPOCH=$(date +%s)"
echo "EXP005B_START_TS=$(date -Is)"

# --- ONE variable vs EXP-004: ENABLE_BETA off->1, holding RIDGE_LAMBDA=0.
ENABLE_BETA=1 \
BETA_FIT_SCOPE=selected \
USE_IDF=0 \
SLOT_SELECTION=tfidf \
GRANULARITY=per_layer \
TOP_T=64 \
TARGET_MODE=cartridge_plus_doc \
KEY_MODE=freeze \
RIDGE_LAMBDA=0 \
RIDGE_SCALE=spectral \
RIDGE_LAMBDA_MIN=0 \
DELTA_WEIGHT=1e-2 \
AM_EXECUTION_MODE=per_document \
PHASE1_CACHE_PATH=outputs/phase1_selfdistill_qwen512/cache_last.pt \
SYNTH_DATA_PATH=data/qasper/train/qwen_qasper_MT_task_8192.parquet \
MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
NUM_TOKENS=512 \
EVAL_QA_PATH=data/qasper/eval/qasper_eval_QA.parquet \
EVAL_MT_PATH=data/qasper/eval/qasper_eval_MT.parquet \
RUN_NAME=exp005b_am_sparse_noidf_beta_top64_perlayer_ridge0 \
WANDB_DISABLED=1 \
bash examples/qasper2/scripts/train_continual_am_sparse.sh
RC=$?

echo "EXP005B_E2E_END_EPOCH=$(date +%s)"
echo "EXP005B_END_TS=$(date -Is)"
echo "EXP005B_PYTHON_RC=$RC"
exit $RC
