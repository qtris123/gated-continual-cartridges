#!/bin/bash
#SBATCH --job-name=eval-all
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err
set -e

if [ -n "$SLURM_SUBMIT_DIR" ]; then
  REPO_DIR="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

OUTPUTS="$REPO_DIR/outputs"
EVAL_DIR="$REPO_DIR/data/financebench/eval"

source "$REPO_DIR/.venv/bin/activate"
if [ -f "$REPO_DIR/.env" ]; then source "$REPO_DIR/.env"; fi

# ---------------------------------------------------------------------------
# eval_cache CACHE_PT  MODEL  MODEL_SHORT  EXPERIMENT  TOKS  DOC_PREFIX  LABEL_SUFFIX
#   Runs MCQ + Yes-No + Original for one cache on one document
# ---------------------------------------------------------------------------
eval_cache() {
  local CACHE="$1"
  local MODEL="$2"
  local MODEL_SHORT="$3"
  local EXPERIMENT="$4"
  local TOKS="$5"
  local DOC_PREFIX="$6"      # e.g. amd_2021_120, amd_2022_120, pepsi_2021_120
  local LABEL_SUFFIX="$7"    # e.g. initial, continual

  local OUTPUT_DIR="$REPO_DIR/outputs/eval_${MODEL_SHORT}_${EXPERIMENT}_toks${TOKS}"

  # Find parquet files for this doc
  local MCQ=$(ls "$EVAL_DIR"/${DOC_PREFIX}_*mcq.parquet 2>/dev/null | head -1)
  local YESNO=$(ls "$EVAL_DIR"/${DOC_PREFIX}_*yes-no.parquet 2>/dev/null | head -1)
  local ORIG=$(ls "$EVAL_DIR"/${DOC_PREFIX}_*original.parquet 2>/dev/null | head -1)

  local EVAL_FILES=()
  [ -f "$MCQ" ]   && EVAL_FILES+=("$MCQ")
  [ -f "$YESNO" ] && EVAL_FILES+=("$YESNO")
  [ -f "$ORIG" ]  && EVAL_FILES+=("$ORIG")

  if [ ${#EVAL_FILES[@]} -eq 0 ]; then
    echo "  SKIP: No parquet files found for $DOC_PREFIX"
    return
  fi

  echo ""
  echo "=========================================="
  echo "  $MODEL_SHORT | $EXPERIMENT | toks=$TOKS | $LABEL_SUFFIX | $DOC_PREFIX"
  echo "=========================================="

  python3 "$REPO_DIR/experiments/evaluate/cartridge.py" \
    --cache "$CACHE" \
    --eval-data "${EVAL_FILES[@]}" \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --max-new-tokens 2048 \
    --eval-mode generate-score \
    --label "${DOC_PREFIX}_${LABEL_SUFFIX}"
}

# ---------------------------------------------------------------------------
# run_config MODEL MODEL_SHORT EXPERIMENT TOKS INITIAL_CACHE CONTINUAL_CACHE
#   Runs eval for both caches on both doc sources
# ---------------------------------------------------------------------------
run_config() {
  local MODEL="$1"
  local MODEL_SHORT="$2"
  local EXPERIMENT="$3"
  local TOKS="$4"
  local INITIAL_CACHE="$5"
  local CONTINUAL_CACHE="$6"

  # AMD 2021 (initial doc) → initial cache
  eval_cache "$INITIAL_CACHE" "$MODEL" "$MODEL_SHORT" "$EXPERIMENT" "$TOKS" "amd_2021_120" "initial"

  # Second doc (continual doc) → continual cache
  if [ "$EXPERIMENT" = "amd2021_amd2022" ]; then
    eval_cache "$CONTINUAL_CACHE" "$MODEL" "$MODEL_SHORT" "$EXPERIMENT" "$TOKS" "amd_2022_120" "continual"
  elif [ "$EXPERIMENT" = "amd2021_pepsi2021" ]; then
    eval_cache "$CONTINUAL_CACHE" "$MODEL" "$MODEL_SHORT" "$EXPERIMENT" "$TOKS" "pepsi_2021_120" "continual"
  fi
}

# ===========================================================================
# Llama 3.2-3B-Instruct
# ===========================================================================
LLAMA="meta-llama/Llama-3.2-3B-Instruct"

run_config "$LLAMA" "llama3.2" "amd2021_amd2022" 512 \
  "$OUTPUTS/2026-03-24-09-31-09-train_initial/79909b2b-2a9c-407b-8b36-0db0f25f6f3f/cache_last.pt" \
  "$OUTPUTS/2026-03-24-10-08-43-train_continual/f8f2f4f4-36fd-4d60-82f6-5a5e5bf843fd/cache_last.pt"

run_config "$LLAMA" "llama3.2" "amd2021_amd2022" 1024 \
  "$OUTPUTS/2026-03-24-00-44-14-train_initial/aeb02668-e686-48c8-95a4-d517be380254/cache_last.pt" \
  "$OUTPUTS/2026-03-24-01-09-04-train_continual/b7e2c843-6949-4f91-9960-fd6a78fc4085/cache_last.pt"

run_config "$LLAMA" "llama3.2" "amd2021_amd2022" 2048 \
  "$OUTPUTS/2026-03-24-02-12-40-train_initial/958a3914-b06c-4ad6-9ccd-ab749a98fb78/cache_last.pt" \
  "$OUTPUTS/2026-03-24-03-29-47-train_continual/1150d4c9-c3f8-4783-97eb-c9d5b7abafca/cache_last.pt"

run_config "$LLAMA" "llama3.2" "amd2021_pepsi2021" 512 \
  "$OUTPUTS/2026-03-24-02-12-18-train_initial/859804cb-7529-4500-9a4d-a0de831180d8/cache_last.pt" \
  "$OUTPUTS/2026-03-24-03-31-33-train_continual/c4064a62-4130-4c05-a249-08160ef24e00/cache_last.pt"

run_config "$LLAMA" "llama3.2" "amd2021_pepsi2021" 1024 \
  "$OUTPUTS/2026-03-24-00-51-52-train_initial/bf6965fb-7360-49e0-b63d-3a144927500b/cache_last.pt" \
  "$OUTPUTS/2026-03-24-01-10-06-train_continual/1586e9f3-3b55-438b-9682-c16a9d9c158e/cache_last.pt"

run_config "$LLAMA" "llama3.2" "amd2021_pepsi2021" 2048 \
  "$OUTPUTS/2026-03-24-02-26-39-train_initial/89363b00-cdf7-4ba9-9e02-336358b18f99/cache_last.pt" \
  "$OUTPUTS/2026-03-24-03-44-52-train_continual/3f9bcfa8-df6c-490f-91a9-fc7815eb40d8/cache_last.pt"

# ===========================================================================
# Qwen3-4B
# ===========================================================================
QWEN="Qwen/Qwen3-4B"

run_config "$QWEN" "qwen3" "amd2021_amd2022" 512 \
  "$OUTPUTS/2026-03-28-00-59-25-initial/11859665-b964-4ce5-afac-cc96123fd443/cache_last.pt" \
  "$OUTPUTS/2026-03-28-09-07-20-continual/14fff93a-0515-4420-95d1-ecdcb5c82c1a/cache_last.pt"

run_config "$QWEN" "qwen3" "amd2021_amd2022" 1024 \
  "$OUTPUTS/2026-03-28-01-50-39-initial/d66546bb-e862-4771-b0ae-8ab1e54888d1/cache_last.pt" \
  "$OUTPUTS/2026-03-28-09-57-34-continual/5ab02387-999a-4d0a-8df6-dc38050637f7/cache_last.pt"

run_config "$QWEN" "qwen3" "amd2021_amd2022" 2048 \
  "$OUTPUTS/2026-03-28-12-48-24-initial/7bd35e62-3a4a-47ed-a41c-2ef0db6b044f/cache_last.pt" \
  "$OUTPUTS/2026-03-28-19-07-49-continual/0828cdb8-8c62-4b81-8fae-e3d28d928d1e/cache_last.pt"

run_config "$QWEN" "qwen3" "amd2021_pepsi2021" 512 \
  "$OUTPUTS/2026-03-27-21-09-53-initial/d8e1ba3f-2f9c-4ad8-9a2b-c10f9b7ee31f/cache_last.pt" \
  "$OUTPUTS/2026-03-27-22-24-53-continual/c178fe72-8cf5-4aa0-997a-9f11e92f9b1f/cache_last.pt"

run_config "$QWEN" "qwen3" "amd2021_pepsi2021" 1024 \
  "$OUTPUTS/2026-03-28-02-43-31-initial/ff81044f-e0ae-41a6-9910-1f2211ed5b1c/cache_last.pt" \
  "$OUTPUTS/2026-03-28-10-49-54-continual/a74c4611-409b-418c-a492-f56ae11926da/cache_last.pt"

run_config "$QWEN" "qwen3" "amd2021_pepsi2021" 2048 \
  "$OUTPUTS/2026-03-28-04-30-24-initial/b61c21b6-d73e-46b4-9a57-4c15fa2f2161/cache_last.pt" \
  "$OUTPUTS/2026-03-28-11-47-07-continual/7649d917-29b1-4cca-b5fb-d21a370c2425/cache_last.pt"

echo ""
echo "=========================================="
echo "  ALL EVALUATIONS COMPLETE"
echo "=========================================="
