#!/usr/bin/env bash
# EXP-007 launcher (HYP-S1): support sweep TOP_T in {32,128} on the no-IDF
# canonical (EXP-001), to map the acquisition/forgetting knee. ONE variable under
# test = TOP_T; all else held at EXP-001 (no-IDF canonical, top_t=64 already exists).
# Runs BOTH configs SEQUENTIALLY on ONE flock-claimed GPU. The flock FD is held for
# the lifetime of THIS wrapper (stays alive through both runs), auto-releasing on exit.
# No BG_STATS_PATH (USE_IDF=0, pure TF ranking). ENABLE_BETA unset.
set -u
cd /localhome/local-triv/gated-continual-cartridges_explore || exit 9
export CARTRIDGES_DIR=$PWD
export CARTRIDGES_OUTPUT_DIR=$PWD/outputs
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university

RESDIR="$CARTRIDGES_DIR/research_loop/results/EXP-007"
# Self-capture wrapper markers so a bare `nohup bash <script>` needs no redirect.
exec > "$RESDIR/wrapper.log" 2>&1
GPU_LOCK_DIR="/tmp/gpu_locks_${USER:-$(id -un)}"
mkdir -p "$GPU_LOCK_DIR"

# Claim EXACTLY ONE GPU non-blockingly; prefer whichever is free (try 0 then 1).
CLAIMED=""
for idx in 0 1; do
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
  echo "EXP007_NO_FREE_GPU: could not flock a GPU under $GPU_LOCK_DIR" >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES="$CLAIMED"
export MASTER_PORT=$((29500 + CLAIMED))
echo "EXP007_CLAIMED_GPU=$CLAIMED MASTER_PORT=$MASTER_PORT (flock held on $GPU_LOCK_DIR/gpu${CLAIMED}.lock)"

# ---- shared (held) config: the no-IDF canonical (EXP-001), everything except TOP_T ----
run_one() {
  local tag="$1" top_t="$2" runname="$3" log="$4"
  echo "EXP007_${tag}_E2E_START_EPOCH=$(date +%s)"
  echo "EXP007_${tag}_START_TS=$(date -Is)"
  USE_IDF=0 \
  SLOT_SELECTION=tfidf \
  GRANULARITY=per_layer \
  TOP_T="$top_t" \
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
  WANDB_DISABLED=1 \
  bash examples/qasper2/scripts/train_continual_am_sparse.sh >"$log" 2>&1
  local rc=$?
  echo "EXP007_${tag}_E2E_END_EPOCH=$(date +%s)"
  echo "EXP007_${tag}_END_TS=$(date -Is)"
  echo "EXP007_${tag}_PYTHON_RC=$rc"
  return $rc
}

# ---- Run 1: TOP_T=32 ----
run_one TOP32 32 exp007_am_sparse_noidf_top32_perlayer "$RESDIR/run_top32.log"
RC32=$?

# ---- Run 2: TOP_T=128 ---- (runs regardless of run1 outcome, sequential, same GPU)
run_one TOP128 128 exp007_am_sparse_noidf_top128_perlayer "$RESDIR/run_top128.log"
RC128=$?

echo "EXP007_RC32=$RC32 EXP007_RC128=$RC128"
echo "EXP007_ALL_DONE_TS=$(date -Is)"
# non-zero if either failed, but both were attempted
exit $(( RC32 != 0 || RC128 != 0 ))
