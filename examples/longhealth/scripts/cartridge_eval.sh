#!/usr/bin/env bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --job-name=cartridge_eval
#SBATCH --output=cartridge_eval.out
#SBATCH --error=cartridge_eval.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# =============================================================================
# Longhealth cartridge evaluation
#
# Each EXPERIMENTS entry is pipe-delimited:
#   MODEL | CARTRIDGE [CARTRIDGE ...] | DATASET[:type] [DATASET ...] | LABEL
#
# - One cartridge and one dataset per line → one (model, cartridge, dataset) run.
# - Several space-separated cartridges or datasets → Cartesian product in
#   cartridge_eval.py (same as before).
#
# For --backend local, use absolute or resolvable paths to .pt caches (e.g. from
# examples/longhealth/scripts/init_kvcache.sh → examples/longhealth/outputs/init_caches/).
#
# BACKEND=tokasaurus: starts Tokasaurus once per MODEL (HF id), uses Hub cartridge ids.
# BACKEND=local: loads the HF model on GPU; no Tokasaurus.
# =============================================================================

export CARTRIDGES_DIR="${CARTRIDGES_DIR:-/home/vo43/cartridges}"
DATA_DIR="${DATA_DIR:-${CARTRIDGES_DIR}/examples/longhealth}"
HF_NS="${HF_NS:-qtris123}"

# Match init_kvcache.sh default (REPO_ROOT, not CARTRIDGES_DIR) so BUILD_INIT_CACHES=1 and EXPERIMENTS agree.
LONGHEALTH_CACHE_DIR="${LONGHEALTH_CACHE_DIR:-${REPO_ROOT}/examples/longhealth/outputs/init_caches}"

CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-${CARTRIDGES_DIR}/outputs}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-cross-eval_longhealth}"
RUN_ROOT="${CARTRIDGES_OUTPUT_DIR}/${EXPERIMENT_NAME}_${SLURM_JOB_ID:-$$}"

BACKEND="${BACKEND:-local}"
DP_SIZE="${DP_SIZE:-2}"
HF_CARTRIDGE_FILENAME="${HF_CARTRIDGE_FILENAME:-cache_last.pt}"
COT="${COT:-1}"
#NUM_EVAL_QUESTIONS="${NUM_EVAL_QUESTIONS:-20}"
#MAX_ANSWER_SCAN_TOKENS="${MAX_ANSWER_SCAN_TOKENS:-256}"
DEBUG="${DEBUG:-0}"
TOKA_PORT="${TOKA_PORT:-10210}"

# Set to 1 once to build init caches under LONGHEALTH_CACHE_DIR (explicit commands in init_kvcache.sh).
BUILD_INIT_CACHES="${BUILD_INIT_CACHES:-1}"

# =============================================================================
# Optional: build step-0 caches before experiments (no loops here; commands live in init_kvcache.sh)
# =============================================================================
if [[ "$BUILD_INIT_CACHES" == "1" ]]; then
  echo "========== BUILD_INIT_CACHES=1: running init_kvcache.sh =========="
  export LONGHEALTH_CACHE_DIR
  bash "${SCRIPT_DIR}/init_kvcache.sh"
  echo "========== Init caches done =========="
fi

# =============================================================================
# Experiment definitions — edit manually (no generated loops)
# Format: MODEL | CARTRIDGE_1 ... | DATASET_1 ... | LABEL
# =============================================================================
EXPERIMENTS=(
  # --- Example: local backend, one cartridge .pt, one eval file ---
  # Qwen 512
  "Qwen/Qwen3-4B-Instruct-2507 | ${LONGHEALTH_CACHE_DIR}/qwen3_512_init_cache_last.pt | ${DATA_DIR}/longhealth_patient1_10-mcq.csv:mcq | local_qwen512_mcq_p1"
  "Qwen/Qwen3-4B-Instruct-2507 | ${LONGHEALTH_CACHE_DIR}/qwen3_512_init_cache_last.pt | ${DATA_DIR}/longhealth_patient1_10-yes-no.csv:yes_no | local_qwen512_yes-no_p1"
  # Qwen 1024
  "Qwen/Qwen3-4B-Instruct-2507 | ${LONGHEALTH_CACHE_DIR}/qwen3_1024_init_cache_last.pt | ${DATA_DIR}/longhealth_patient1_10-mcq.csv:mcq | local_qwen1024_mcq_p1"
  "Qwen/Qwen3-4B-Instruct-2507 | ${LONGHEALTH_CACHE_DIR}/qwen3_1024_init_cache_last.pt | ${DATA_DIR}/longhealth_patient1_10-yes-no.csv:yes_no | local_qwen1024_yes-no_p1"
  # Qwen 2048
  "Qwen/Qwen3-4B-Instruct-2507 | ${LONGHEALTH_CACHE_DIR}/qwen3_2048_init_cache_last.pt | ${DATA_DIR}/longhealth_patient1_10-mcq.csv:mcq | local_qwen2048_mcq_p1"
  "Qwen/Qwen3-4B-Instruct-2507 | ${LONGHEALTH_CACHE_DIR}/qwen3_2048_init_cache_last.pt | ${DATA_DIR}/longhealth_patient1_10-yes-no.csv:yes_no | local_qwen2048_yes-no_p1"
  # Llama 512 (HF id uses meta-llama, not meta_llama)
  "meta-llama/Llama-3.2-3B-Instruct | ${LONGHEALTH_CACHE_DIR}/llama_512_init_cache_last.pt | ${DATA_DIR}/longhealth_patient1_10-mcq.csv:mcq | local_llama512_mcq_p1"
  "meta-llama/Llama-3.2-3B-Instruct | ${LONGHEALTH_CACHE_DIR}/llama_512_init_cache_last.pt | ${DATA_DIR}/longhealth_patient1_10-yes-no.csv:yes_no | local_llama512_yes-no_p1"
  # Llama 1024
  "meta-llama/Llama-3.2-3B-Instruct | ${LONGHEALTH_CACHE_DIR}/llama_1024_init_cache_last.pt | ${DATA_DIR}/longhealth_patient1_10-mcq.csv:mcq | local_llama1024_mcq_p1"
  "meta-llama/Llama-3.2-3B-Instruct | ${LONGHEALTH_CACHE_DIR}/llama_1024_init_cache_last.pt | ${DATA_DIR}/longhealth_patient1_10-yes-no.csv:yes_no | local_llama1024_yes-no_p1"
  # Llama 2048
  "meta-llama/Llama-3.2-3B-Instruct | ${LONGHEALTH_CACHE_DIR}/llama_2048_init_cache_last.pt | ${DATA_DIR}/longhealth_patient1_10-mcq.csv:mcq | local_llama2048_mcq_p1"
  "meta-llama/Llama-3.2-3B-Instruct | ${LONGHEALTH_CACHE_DIR}/llama_2048_init_cache_last.pt | ${DATA_DIR}/longhealth_patient1_10-yes-no.csv:yes_no | local_llama2048_yes-no_p1"
)

mkdir -p "${RUN_ROOT}"

echo "=========================================="
echo "Longhealth cartridge eval (BACKEND=${BACKEND})"
echo "=========================================="
echo "JobID=${SLURM_JOB_ID:-local}"
echo "Cluster=${SLURM_CLUSTER_NAME:-local}"
echo "Node/Server=$(hostname)"
echo "Started at: $(date)"
echo "RUN_ROOT=${RUN_ROOT}"
echo "LONGHEALTH_CACHE_DIR=${LONGHEALTH_CACHE_DIR}"
echo "Total experiments: ${#EXPERIMENTS[@]}"
echo ""

if command -v module >/dev/null 2>&1; then
  module load gcc/11.4.1 2>/dev/null || true
  module load cuda/12.1.0 2>/dev/null || true
fi

echo "CUDA_HOME=${CUDA_HOME:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L
else
  echo "(nvidia-smi not found)"
fi

GPU_LOG="${RUN_ROOT}/gpu_usage.log"
echo "=== GPU monitor started: $(date) on $(hostname) ===" >"$GPU_LOG"
(
  while true; do
    date "+[%F %T]"
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
      --format=csv,noheader,nounits 2>/dev/null || echo "(nvidia-smi query failed)"
    sleep 10
  done
) >>"$GPU_LOG" 2>&1 &
GPU_MON_PID=$!

cleanup() {
  echo ""
  echo "Cleaning up..."
  if [[ "$BACKEND" == "tokasaurus" ]] && [[ -n "${TOKA_PID:-}" ]]; then
    echo "Stopping toka server (PID=${TOKA_PID}) and children..."
    kill -- -"$TOKA_PID" 2>/dev/null || true
    pkill -KILL -P "$TOKA_PID" 2>/dev/null || true
    kill -KILL "$TOKA_PID" 2>/dev/null || true
    wait "$TOKA_PID" 2>/dev/null || true
  fi
  if [[ -n "${GPU_MON_PID:-}" ]]; then
    kill "$GPU_MON_PID" 2>/dev/null || true
    wait "$GPU_MON_PID" 2>/dev/null || true
  fi
  echo "Job finished at: $(date)"
}
trap cleanup EXIT

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cartridges
echo "Python: $(which python3)"
echo ""

COMMON_ARGS=()
[[ "$COT" == "1" ]] && COMMON_ARGS+=(--cot)
[[ -n "${HF_USERNAME:-}" ]] && COMMON_ARGS+=(--hf-username "$HF_USERNAME")
[[ "$DEBUG" == "1" ]] && COMMON_ARGS+=(--debug)

CARTRIDGE_EVAL_PY="${CARTRIDGE_EVAL_PY:-${CARTRIDGES_DIR}/examples/longhealth/experiments/cartridge_eval.py}"
if [[ ! -f "$CARTRIDGE_EVAL_PY" ]]; then
  CARTRIDGE_EVAL_PY="${REPO_ROOT}/examples/longhealth/experiments/cartridge_eval.py"
fi

start_toka_server() {
  local model="$1"
  echo ""
  echo "========== Starting toka server for ${model} =========="
  setsid tksrs model="${model}" kv_cache_num_tokens='(2048 * 2048)' \
        max_topk_logprobs=20 dp_size="${DP_SIZE}" port="${TOKA_PORT}" &
  TOKA_PID=$!
  URL="http://$(hostname):${TOKA_PORT}"
  echo "  PID=${TOKA_PID}, URL=${URL}"

  echo "  Waiting for server to be ready..."
  local max_wait=300 waited=0
  while ! curl -s "${URL}/health" >/dev/null 2>&1; do
    if ! kill -0 "$TOKA_PID" 2>/dev/null; then
      echo "ERROR: toka server died unexpectedly" >&2
      exit 1
    fi
    if [[ "$waited" -ge "$max_wait" ]]; then
      echo "ERROR: toka server not ready within ${max_wait}s" >&2
      exit 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
  echo "  Server ready (took ~${waited}s)"
}

stop_toka_server() {
  if [[ -n "${TOKA_PID:-}" ]]; then
    echo "  Stopping toka server (PID=${TOKA_PID}) and all child workers..."
    kill -- -"$TOKA_PID" 2>/dev/null || true
    pkill -KILL -P "$TOKA_PID" 2>/dev/null || true
    kill -KILL "$TOKA_PID" 2>/dev/null || true
    wait "$TOKA_PID" 2>/dev/null || true
    TOKA_PID=""

    echo "  Waiting for GPU memory to free..."
    local gpu_wait=0
    while [[ "$gpu_wait" -lt 60 ]]; do
      local mem_used
      mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
                 | awk '{s+=$1} END {print int(s)}')
      if [[ "${mem_used:-99999}" -lt 1000 ]]; then
        echo "  GPU memory freed (~${mem_used} MiB in use)"
        break
      fi
      sleep 2
      gpu_wait=$((gpu_wait + 2))
    done
    if [[ "$gpu_wait" -ge 60 ]]; then
      echo "  WARNING: GPU memory still in use after 60s, proceeding anyway"
    fi
  fi
}

TOTAL_EXPERIMENTS=${#EXPERIMENTS[@]}
EXPERIMENT_NUM=0
CURRENT_MODEL=""

for ENTRY in "${EXPERIMENTS[@]}"; do
  [[ -z "${ENTRY// }" ]] && continue
  [[ "$ENTRY" =~ ^[[:space:]]*# ]] && continue

  IFS='|' read -r EXP_MODEL EXP_CARTRIDGES EXP_DATASETS EXP_LABEL <<< "$ENTRY"
  EXP_MODEL=$(echo "$EXP_MODEL" | xargs)
  EXP_CARTRIDGES=$(echo "$EXP_CARTRIDGES" | xargs)
  EXP_DATASETS=$(echo "$EXP_DATASETS" | xargs)
  EXP_LABEL=$(echo "$EXP_LABEL" | xargs)

  if [[ "$BACKEND" == "tokasaurus" ]]; then
    if [[ "$EXP_MODEL" != "$CURRENT_MODEL" ]]; then
      stop_toka_server
      start_toka_server "$EXP_MODEL"
      CURRENT_MODEL="$EXP_MODEL"
    fi
  fi

  EXPERIMENT_NUM=$((EXPERIMENT_NUM + 1))
  read -ra CARTRIDGE_ARR <<< "$EXP_CARTRIDGES"
  read -ra DATASET_ARR <<< "$EXP_DATASETS"

  EXP_DIR="${RUN_ROOT}/${EXP_LABEL}"
  mkdir -p "$EXP_DIR"

  echo ""
  echo "══════════════════════════════════════════"
  echo "  Experiment ${EXPERIMENT_NUM}/${TOTAL_EXPERIMENTS}: ${EXP_LABEL}"
  echo "  Backend:    ${BACKEND}"
  echo "  Model:      ${EXP_MODEL}"
  echo "  Cartridges: ${CARTRIDGE_ARR[*]}"
  echo "  Datasets:   ${DATASET_ARR[*]}"
  echo "  Output:     ${EXP_DIR}"
  echo "══════════════════════════════════════════"

  if [[ "$BACKEND" == "tokasaurus" ]]; then
    python3 "$CARTRIDGE_EVAL_PY" \
      --backend tokasaurus \
      --url "$URL" \
      --cartridges "${CARTRIDGE_ARR[@]}" \
      --datasets "${DATASET_ARR[@]}" \
      --model "$EXP_MODEL" \
      --output-dir "$EXP_DIR" \
      "${COMMON_ARGS[@]}" \
      #--max-answer-scan-tokens "$MAX_ANSWER_SCAN_TOKENS" \
      #--num-eval-questions "$NUM_EVAL_QUESTIONS" \
      --hf-cartridge-filename "$HF_CARTRIDGE_FILENAME" \
      --top-logprobs 20 \
      --batch-size 5
  else
    python3 "$CARTRIDGE_EVAL_PY" \
      --backend local \
      --cartridges "${CARTRIDGE_ARR[@]}" \
      --datasets "${DATASET_ARR[@]}" \
      --model "$EXP_MODEL" \
      --output-dir "$EXP_DIR" \
      "${COMMON_ARGS[@]}" \
      #--max-answer-scan-tokens "$MAX_ANSWER_SCAN_TOKENS" \
      #--num-eval-questions "$NUM_EVAL_QUESTIONS" \
      --hf-cartridge-filename "$HF_CARTRIDGE_FILENAME"
  fi

  echo "  ✓ Experiment ${EXPERIMENT_NUM} complete"
done

if [[ "$BACKEND" == "tokasaurus" ]]; then
  stop_toka_server
fi

echo ""
echo "=========================================="
echo "All ${TOTAL_EXPERIMENTS} experiment line(s) complete!"
echo "Results in: ${RUN_ROOT}"
echo "Finished at: $(date)"
echo "=========================================="
