#!/usr/bin/env bash
# TechQA 5-phase self-study synthesis via vLLM.
#
# Each phase is ~99 IBM technotes (~150k tokens total). Prompts sample
# DOCS_PER_PROMPT notes (default 1) so AM can later key on <source>.
# Dataset: nvidia/TechQA-RAG-Eval (HF cache). Frozen splits: techqa/phases.json
#
# Usage:
#   PHASES=1 NUM_SAMPLES=64 MAX_NUM_BATCHES=2 DP_SIZE=1 bash examples/techqa/synthesize/self_study.sh
#   SKIP_VLLM=1 bash examples/techqa/synthesize/self_study.sh

set -euo pipefail

export CARTRIDGES_DIR="${CARTRIDGES_DIR:-/localhome/local-triv/trivo-explore-research-work}"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$CARTRIDGES_DIR/outputs}"
export PYTHONPATH="$CARTRIDGES_DIR:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# TrainDataset targets="logits" reads stored teacher logprobs and true vocab ids.
# Neither is emitted unless the client asks and the server is started with
# --return-tokens-as-token-ids, so both sides are pinned here.
export SYNTH_NUM_TOP_LOGPROBS="${SYNTH_NUM_TOP_LOGPROBS:-20}"
export SYNTH_RETURN_TOKEN_IDS="${SYNTH_RETURN_TOKEN_IDS:-1}"
export SYNTH_MAX_COMPLETION_TOKENS_B="${SYNTH_MAX_COMPLETION_TOKENS_B:-2048}"

cd "$CARTRIDGES_DIR"
AM_PY="${AM_PY:-$CARTRIDGES_DIR/.venv/bin/python}"

MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
PHASES="${PHASES:-1 2 3 4 5}"
NUM_SAMPLES="${NUM_SAMPLES:-8192}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_NUM_BATCHES="${MAX_NUM_BATCHES:-8}"
PROB_THINKING="${PROB_THINKING:-0.2}"
DOCS_PER_PROMPT="${DOCS_PER_PROMPT:-1}"
PORT="${PORT:-8000}"
DP_SIZE="${DP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_UTIL="${GPU_UTIL:-0.85}"
SKIP_VLLM="${SKIP_VLLM:-0}"
KEEP_VLLM="${KEEP_VLLM:-0}"
SERVER_MODULE="${SERVER_MODULE:-examples.shared.synth.vllm_nan_safe_server}"
# Resume each phase into its own published run dir, so a top-up fills that
# phase's gaps in place instead of every phase sharing one RESUME_DIR.
AUTO_RESUME="${AUTO_RESUME:-1}"
# The longest technote prompt is ~44.5k tokens; overflowing max_model_len 400s the
# whole batch. Reserve the completion cap plus slack for the chat wrapper.
export TECHQA_MAX_PROMPT_TOKENS="${TECHQA_MAX_PROMPT_TOKENS:-$((MAX_MODEL_LEN - SYNTH_MAX_COMPLETION_TOKENS_B - 2048))}"

export CARTRIDGES_VLLM_URL="http://127.0.0.1:${PORT}/v1"

LOGDIR="${LOGDIR:-/localhome/local-triv/outputs_synth_logs}"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
SERVER_LOG="$LOGDIR/vllm_techqa_${STAMP}.log"

echo "=========================================="
echo "TechQA self-study | phases=$PHASES | model=$MODEL"
echo "off-policy: base model + technotes (docs_per_prompt=$DOCS_PER_PROMPT max_prompt_tokens=$TECHQA_MAX_PROMPT_TOKENS)"
echo "num_samples=$NUM_SAMPLES batch_size=$BATCH_SIZE max_num_batches=$MAX_NUM_BATCHES"
echo "dp_size=$DP_SIZE port=$PORT skip_vllm=$SKIP_VLLM keep_vllm=$KEEP_VLLM"
echo "server log: $SERVER_LOG"
echo "=========================================="

[ -x "$AM_PY" ] || { echo "FATAL: python not found at $AM_PY"; exit 1; }
"$AM_PY" -c "import vllm; print('vllm', vllm.__version__)" || { echo "FATAL: vllm missing"; exit 1; }

SERVER_PID=""
cleanup() {
  if [ "$KEEP_VLLM" = "1" ]; then
    echo "KEEP_VLLM=1 — leaving vllm running on :$PORT"
    echo "finished at: $(date)"
    return
  fi
  if [ -n "${SERVER_PID:-}" ]; then
    echo "stopping vllm (pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  echo "finished at: $(date)"
}
trap cleanup EXIT

if [ "$SKIP_VLLM" = "1" ]; then
  curl -so /dev/null "http://127.0.0.1:${PORT}/health" \
    || { echo "FATAL: SKIP_VLLM=1 but nothing answers on :$PORT"; exit 1; }
  echo "=== reusing vllm on :$PORT ==="
else
  if curl -so /dev/null "http://127.0.0.1:${PORT}/health" 2>/dev/null; then
    echo "FATAL: something already answers on :$PORT (kill it, or SKIP_VLLM=1 if it is this model)"
    exit 1
  fi
  echo "=== starting vllm (log -> $SERVER_LOG) ==="
  "$AM_PY" -m "$SERVER_MODULE" \
    --model "$MODEL" \
    --served-model-name "$MODEL" \
    --port "$PORT" \
    --data-parallel-size "$DP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --return-tokens-as-token-ids \
    --disable-log-requests >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  echo "waiting for readiness (pid $SERVER_PID)..."
  max_wait=1800 waited=0
  until curl -so /dev/null "http://127.0.0.1:${PORT}/health" 2>/dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "FATAL: vllm exited during startup. Tail of $SERVER_LOG:"; tail -40 "$SERVER_LOG"; exit 1
    fi
    if [ "$waited" -ge "$max_wait" ]; then
      echo "FATAL: vllm not ready after ${max_wait}s. Tail of $SERVER_LOG:"; tail -40 "$SERVER_LOG"; exit 1
    fi
    sleep 5; waited=$((waited + 5))
  done
  echo "=== vllm ready after ${waited}s ==="
fi

OVERALL_RC=0
for phase in $PHASES; do
  RUN_NAME="techqa_p${phase}_self_study_n${NUM_SAMPLES}"
  SYNTH_LOG="$LOGDIR/synth_techqa_p${phase}_${STAMP}.log"
  PHASE_DIR="$CARTRIDGES_DIR/data/techqa/synth/p0${phase}/self_study-n${NUM_SAMPLES}"
  PHASE_RESUME="${RESUME_DIR:-}"
  if [ -z "$PHASE_RESUME" ] && [ "$AUTO_RESUME" = "1" ] && [ -d "$PHASE_DIR" ]; then
    PHASE_RESUME="$PHASE_DIR"
  fi
  [ -n "$PHASE_RESUME" ] && echo "resume dir: $PHASE_RESUME"
  echo "=== synthesizing techqa phase=$phase (log -> $SYNTH_LOG) ==="
  set +e
  "$AM_PY" "$CARTRIDGES_DIR/examples/shared/synth/self_study_vllm.py" \
    --dataset techqa \
    --phase "$phase" \
    --model "$MODEL" \
    --base-url "$CARTRIDGES_VLLM_URL" \
    --num-samples "$NUM_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --max-num-batches "$MAX_NUM_BATCHES" \
    --prob-thinking "$PROB_THINKING" \
    --docs-per-prompt "$DOCS_PER_PROMPT" \
    --run-name "$RUN_NAME" \
    ${PHASE_RESUME:+--resume-dir "$PHASE_RESUME"} >"$SYNTH_LOG" 2>&1
  SYNTH_RC=$?
  set -e
  echo "phase $phase exit code: $SYNTH_RC"
  tail -20 "$SYNTH_LOG"
  if [ "$SYNTH_RC" -ne 0 ]; then
    OVERALL_RC=$SYNTH_RC
    echo "FATAL: phase $phase failed; remaining phases skipped"
    break
  fi
done

echo
echo "=== recent dataset.parquet under $CARTRIDGES_OUTPUT_DIR ==="
find "$CARTRIDGES_OUTPUT_DIR" -name 'dataset.parquet' -newermt '-24 hours' -printf '%T@ %p\n' 2>/dev/null \
  | sort -rn | head -10 | cut -d' ' -f2-

exit $OVERALL_RC
