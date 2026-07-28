#!/usr/bin/env bash
# EXP-006 launcher (HYP-T2 answered): ENABLE_BETA=1 WITH the new opt-in beta
# magnitude clamp BETA_CLAMP_ABS=3.0, holding RIDGE_LAMBDA=0.
# EXP-005 (beta, lambda=1e-4) crashed cholesky-not-PD; EXP-005b (beta, lambda=0)
# crashed with "NaNs in ridge lstsq solution" -- both from HUGE fitted beta
# log-weights (max|beta|~66-68) collapsing the pre-softmax attention and making
# the value-solve design matrix rank-deficient. The AM paper clamps beta to a
# bounded range for exactly this reason. This run adds that clamp (opt-in).
# ONE variable vs EXP-004 (lambda=0 baseline, beta OFF): ENABLE_BETA off->1 WITH
# BETA_CLAMP_ABS=3.0, holding lambda=0.
# Claim EXACTLY ONE GPU via repo flock; prefer GPU 1. flock FD held for lifetime.
# No BG_STATS_PATH (USE_IDF=0, pure TF ranking).
cd /localhome/local-triv/gated-continual-cartridges_explore || exit 9
export CARTRIDGES_DIR=$PWD
export CARTRIDGES_OUTPUT_DIR=$PWD/outputs
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
# IMPORT-RESOLUTION FIX: the venv's editable install maps `cartridges` to the
# SIBLING repo (/localhome/local-triv/gated-continual-cartridges), so a
# `python /abs/script.py` invocation (script-dir on sys.path, not cwd) would
# import the sibling package and MISS this branch's edits (e.g. beta_clamp_abs).
# Prepend this repo so `import cartridges` resolves to _explore/cartridges.
export PYTHONPATH="$CARTRIDGES_DIR:${PYTHONPATH:-}"

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
  echo "EXP006_NO_FREE_GPU: could not flock a GPU under $GPU_LOCK_DIR" >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES="$CLAIMED"
export MASTER_PORT=$((29500 + CLAIMED))
echo "EXP006_CLAIMED_GPU=$CLAIMED MASTER_PORT=$MASTER_PORT (flock held on $GPU_LOCK_DIR/gpu${CLAIMED}.lock)"
echo "EXP006_E2E_START_EPOCH=$(date +%s)"
echo "EXP006_START_TS=$(date -Is)"

# --- ONE variable vs EXP-004: ENABLE_BETA off->1 WITH BETA_CLAMP_ABS=3.0, lambda=0.
ENABLE_BETA=1 \
BETA_CLAMP_ABS=3.0 \
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
RUN_NAME=exp006_am_sparse_noidf_beta_clamp3_top64_perlayer_ridge0 \
WANDB_DISABLED=1 \
bash examples/qasper2/scripts/core/train_continual_am_sparse.sh
RC=$?

echo "EXP006_E2E_END_EPOCH=$(date +%s)"
echo "EXP006_END_TS=$(date -Is)"
echo "EXP006_PYTHON_RC=$RC"
exit $RC
