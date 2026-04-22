#!/bin/bash
# Re-evaluate the TRAINED cartridge only (AMD2022-pair 'initial' carts) on the
# 3 AMD-2021 evals. The existing untrained (step-0) baseline JSONs are kept.
set -euo pipefail
source /home/phudishp/continual-cartridges/.venv/bin/activate
export CARTRIDGES_DIR=/home/phudishp/continual-cartridges
export CARTRIDGES_OUTPUT_DIR=/home/phudishp/continual-cartridges/outputs

REPO=/home/phudishp/continual-cartridges
TEXT=$REPO/data/financebench/texts/AMD_2021_10K.txt
ORIG=$REPO/data/financebench/eval/amd_2021_120_original.parquet
YESNO=$REPO/data/financebench/eval/amd_2021_120_batch-yes-no.parquet
MCQ=$REPO/data/financebench/eval/amd_2021_120_batch-mcq.parquet

rerun_trained () {
  local MODEL_KEY="$1"
  local MODEL_SHORT="$2"
  local TOKS="$3"
  local TRAINED="$4"

  local UNTRAINED=$REPO/outputs/amd_2021_untrained_${MODEL_SHORT}_toks${TOKS}/cache_last.pt
  local BASE=$REPO/outputs/amd_2021_baselines_${MODEL_SHORT}_toks${TOKS}

  echo ""
  echo "################################################################"
  echo "# $MODEL_SHORT toks=$TOKS  (AMD2022-pair initial)"
  echo "# $TRAINED"
  echo "################################################################"

  if [ ! -f "$UNTRAINED" ]; then
    echo "--- untrained cache missing, build it ---"
    python $REPO/experiments/swapping/build_init_cache.py \
      --model "$MODEL_KEY" \
      --text "$TEXT" --max-tokens "$TOKS" --num-frozen-tokens 1 \
      --output "$UNTRAINED"
  fi

  # Each invocation overwrites only trained.json (--skip-baseline-b keeps
  # the existing untrained.json).
  echo "--- openended (full gen) — trained only ---"
  python $REPO/experiments/swapping/per_layer_swap.py \
    --model "$MODEL_KEY" --mode openended \
    --cache-a "$TRAINED" --cache-b "$UNTRAINED" \
    --cache-a-label trained --cache-b-label untrained \
    --skip-baseline-b \
    --eval amd_2021=$ORIG --layers "" --directions A_into_B \
    --batch-size 16 --max-new-tokens 256 \
    --output-dir "$BASE/openended"

  echo "--- yesno (1-token) — trained only ---"
  python $REPO/experiments/swapping/per_layer_swap.py \
    --model "$MODEL_KEY" --mode yesno \
    --cache-a "$TRAINED" --cache-b "$UNTRAINED" \
    --cache-a-label trained --cache-b-label untrained \
    --skip-baseline-b \
    --eval amd_2021=$YESNO --layers "" --directions A_into_B \
    --batch-size 32 \
    --output-dir "$BASE/yesno"

  echo "--- mcq (1-token) — trained only ---"
  python $REPO/experiments/swapping/per_layer_swap.py \
    --model "$MODEL_KEY" --mode mcq \
    --cache-a "$TRAINED" --cache-b "$UNTRAINED" \
    --cache-a-label trained --cache-b-label untrained \
    --skip-baseline-b \
    --eval amd_2021=$MCQ --layers "" --directions A_into_B \
    --batch-size 32 \
    --output-dir "$BASE/mcq"
}

OUT=$REPO/outputs

# AMD2021 -> AMD2022 pair 'initial' carts (AMD-2021 trained, different seeds
# from the Pepsi-pair initial carts I used previously).
rerun_trained qwen3-4b-instruct     qwen3 512  "$OUT/2026-03-28-00-59-25-initial/11859665-b964-4ce5-afac-cc96123fd443/cache_last.pt"
rerun_trained qwen3-4b-instruct     qwen3 1024 "$OUT/2026-03-28-01-50-39-initial/d66546bb-e862-4771-b0ae-8ab1e54888d1/cache_last.pt"
rerun_trained qwen3-4b-instruct     qwen3 2048 "$OUT/2026-03-28-12-48-24-initial/7bd35e62-3a4a-47ed-a41c-2ef0db6b044f/cache_last.pt"
rerun_trained llama-3.2-3b-instruct llama 512  "$OUT/2026-03-24-09-31-09-train_initial/79909b2b-2a9c-407b-8b36-0db0f25f6f3f/cache_last.pt"
rerun_trained llama-3.2-3b-instruct llama 1024 "$OUT/2026-03-24-00-44-14-train_initial/aeb02668-e686-48c8-95a4-d517be380254/cache_last.pt"
rerun_trained llama-3.2-3b-instruct llama 2048 "$OUT/2026-03-24-02-12-40-train_initial/958a3914-b06c-4ad6-9ccd-ab749a98fb78/cache_last.pt"

echo ""
echo "################################################################"
echo "# ALL 6 CONFIGS DONE — trained.json updated to AMD2022-pair"
echo "################################################################"
