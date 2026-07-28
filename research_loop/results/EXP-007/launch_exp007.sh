#!/usr/bin/env bash
# EXP-007 (HYP-S1): support sweep TOP_T in {32,128} on the no-IDF canonical
# (EXP-001). ONE variable under test = TOP_T; all else held at EXP-001 (no-IDF
# canonical, top_t=64 already exists => not re-run here). Runs BOTH configs
# SEQUENTIALLY on ONE flock-claimed GPU (flock FD held for the wrapper lifetime,
# auto-released on exit). No BG_STATS_PATH (USE_IDF=0, pure TF ranking).
# ENABLE_BETA unset; BETA_CLAMP_ABS not set (defaults 0.0 = inert / no clamp).
# PYTHONPATH pinned to the explore repo so `import cartridges` resolves to it
# (explore am/*.py is byte-identical to the sibling install => EXP-001-comparable;
# and the explore config accepts the beta_clamp_abs field if a concurrent editor
# has re-applied the driver's unconditional pass). Retries a run ONLY on the
# beta_clamp_abs/"Extra inputs are not permitted" race crash.
set -u
cd /localhome/local-triv/gated-continual-cartridges_explore || exit 9
export CARTRIDGES_DIR=$PWD
export CARTRIDGES_OUTPUT_DIR=$PWD/outputs
export CARTRIDGES_WANDB_PROJECT=SEACrowd
export CARTRIDGES_WANDB_ENTITY=vqtri-purdue-university
export PYTHONPATH="$CARTRIDGES_DIR:${PYTHONPATH:-}"

RESDIR="$CARTRIDGES_DIR/research_loop/results/EXP-007"
mkdir -p "$RESDIR"
exec > "$RESDIR/wrapper.log" 2>&1

GPU_LOCK_DIR="/tmp/gpu_locks_${USER:-$(id -un)}"
mkdir -p "$GPU_LOCK_DIR"

# Claim EXACTLY ONE GPU non-blockingly; prefer gpu1 (gpu0 seen held), fall back.
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
  echo "EXP007_NO_FREE_GPU: could not flock a GPU under $GPU_LOCK_DIR" >&2
  exit 3
fi
export CUDA_VISIBLE_DEVICES="$CLAIMED"
export MASTER_PORT=$((29500 + CLAIMED))
echo "EXP007_CLAIMED_GPU=$CLAIMED MASTER_PORT=$MASTER_PORT PYTHONPATH_head=${PYTHONPATH%%:*} (flock held on $GPU_LOCK_DIR/gpu${CLAIMED}.lock)"

# ---- one run = the no-IDF canonical (EXP-001), everything held except TOP_T ----
run_one() {
  local tag="$1" top_t="$2" runname="$3" log="$4"
  local attempt rc
  rc=1
  for attempt in 1 2 3; do
    echo "EXP007_${tag}_ATTEMPT${attempt}_START_EPOCH=$(date +%s) TS=$(date -Is)"
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
    bash examples/qasper2/scripts/core/train_continual_am_sparse.sh >"$log" 2>&1
    rc=$?
    echo "EXP007_${tag}_ATTEMPT${attempt}_END_EPOCH=$(date +%s) TS=$(date -Is) RC=$rc"
    if [ $rc -eq 0 ]; then break; fi
    if grep -qE "beta_clamp_abs|Extra inputs are not permitted" "$log"; then
      echo "EXP007_${tag}_ATTEMPT${attempt}_BETACLAMP_RACE -> concurrent-edit race, retry in 10s"
      sleep 10
      continue
    else
      echo "EXP007_${tag}_ATTEMPT${attempt}_NONBETA_FAILURE -> not a race crash, stop retrying"
      break
    fi
  done
  return $rc
}

# ---- Run 1: TOP_T=32 ----
run_one TOP32 32 exp007_top32 "$RESDIR/run_top32.log"
RC32=$?

# ---- Run 2: TOP_T=128 ---- (runs regardless of run1 outcome; sequential, same GPU)
run_one TOP128 128 exp007_top128 "$RESDIR/run_top128.log"
RC128=$?

echo "EXP007_RC32=$RC32 EXP007_RC128=$RC128"
echo "EXP007_ALL_DONE_TS=$(date -Is)"
exit $(( RC32 != 0 || RC128 != 0 ))
