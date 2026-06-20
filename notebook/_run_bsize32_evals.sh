#!/usr/bin/env bash
# Run QA (forgetting) + MT (learning) forgetting evals for the bsize=32
# value-only per_layer cartridges that haven't been evaluated yet.
#
# Outputs go into outputs/qasper_forgetting_eval_bsize32/<label>__<phase>_eval/eval.log
# matching the existing layout for parsing.
set -euo pipefail

REPO=/localhome/local-triv/gated-continual-cartridges
RESULTS=$REPO/outputs/qasper_forgetting_eval_bsize32
mkdir -p "$RESULTS"

source "$REPO/.venv/bin/activate"

QA_EVAL=$REPO/examples/qasper2/qasper_eval_QA.parquet
MT_EVAL=$REPO/examples/qasper2/qasper_eval_MT.parquet
MODEL_NAME=meta-llama/Llama-3.2-3B-Instruct

declare -A CKPTS=(
  [qasper-value-only-per-layer-top-64-bsize32]=$REPO/outputs/2026-06-19-03-39-48-continual_sparse-qasper-per-layer-top-64-value-only-all-reduce/55c2200c-15a2-4c64-a32d-f944ce530d5a/cache-step564.pt
  [qasper-value-only-per-layer-top-128-bsize32]=$REPO/outputs/2026-06-19-03-39-52-continual_sparse-qasper-per-layer-top-128-value-only-all-reduce/ae8f5388-a145-40cb-97d0-f2ea5f755325/cache-step564.pt
  [qasper-value-only-per-layer-top-256-bsize32]=$REPO/outputs/2026-06-19-04-51-43-continual_sparse-qasper-per-layer-top-256-value-only-all-reduce/1dcf0b37-01ce-4b3b-b902-0c2bb629587e/cache-step564.pt
  [qasper-value-only-per-layer-top-512-bsize32]=$REPO/outputs/2026-06-19-04-52-33-continual_sparse-qasper-per-layer-top-512-value-only-all-reduce/19ca37fb-ca54-4629-89ec-45dd1ae79e49/cache-step564.pt
  [qasper-value-only-per-head-top-64-bsize32]=$REPO/outputs/2026-06-19-06-05-26-continual_sparse-qasper-per-head-top-64-value-only-all-reduce/3a40ff70-7a9b-4435-ac1e-863d95d6608a/cache-step564.pt
  [qasper-value-only-per-head-top-128-bsize32]=$REPO/outputs/2026-06-19-06-05-30-continual_sparse-qasper-per-head-top-128-value-only-all-reduce/0392ba49-4e43-48a2-8257-d22c3c9f99c1/cache-step564.pt
  [qasper-value-only-per-head-top-256-bsize32]=$REPO/outputs/2026-06-19-07-18-07-continual_sparse-qasper-per-head-top-256-value-only-all-reduce/b2c90725-ff10-4bfb-8147-566bf7225f9d/cache-step564.pt
  [qasper-value-only-per-head-top-512-bsize32]=$REPO/outputs/2026-06-19-07-18-27-continual_sparse-qasper-per-head-top-512-value-only-all-reduce/c9505df5-8829-4594-be36-7134615c8a65/cache-step564.pt
)

GPU_IDX=0
PIDS=()
for label in "${!CKPTS[@]}"; do
  CKPT=${CKPTS[$label]}
  for phase in qa mt; do
    if [ "$phase" = "qa" ]; then DATA=$QA_EVAL; else DATA=$MT_EVAL; fi
    OUT=$RESULTS/${label}__${phase}_eval
    mkdir -p "$OUT"
    if [ -f "$OUT/eval.log" ] && grep -q 'perplexity' "$OUT/eval.log" 2>/dev/null; then
      echo "  skip   $label/$phase  (already evaluated)"
      continue
    fi
    GPU=$(( GPU_IDX % 4 ))
    GPU_IDX=$(( GPU_IDX + 1 ))
    echo "  dispatch $label/$phase  →  GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU \
    CHECKPOINT_PATH="$CKPT" \
    EVAL_DATA_PATH="$DATA" \
    MODEL_NAME="$MODEL_NAME" \
    RUN_NAME="${label}__${phase}_eval" \
    BATCH_SIZE=8 \
    WANDB_GROUP="qasper - [bsize32 follow-up]" \
    python3 "$REPO/examples/qasper2/train/eval_forgetting.py" \
      >"$OUT/eval.log" 2>&1 &
    PIDS+=($!)
  done
done

echo "Launched ${#PIDS[@]} eval jobs"
for p in "${PIDS[@]}"; do wait "$p" || echo "job $p failed"; done
echo "All bsize=32 evals complete."

printf '\n%-60s %-12s %-12s\n' "Run" "QA ppl" "MT ppl"
printf '%-60s %-12s %-12s\n' "---" "------" "------"
for label in "${!CKPTS[@]}"; do
  QA=$(grep -oP 'perplexity[ =:]+\K[0-9.]+' "$RESULTS/${label}__qa_eval/eval.log" 2>/dev/null | tail -1 || echo "N/A")
  MT=$(grep -oP 'perplexity[ =:]+\K[0-9.]+' "$RESULTS/${label}__mt_eval/eval.log" 2>/dev/null | tail -1 || echo "N/A")
  printf '%-60s %-12s %-12s\n' "$label" "$QA" "$MT"
done
