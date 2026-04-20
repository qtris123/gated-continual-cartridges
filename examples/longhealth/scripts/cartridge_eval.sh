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

# =============================================================================
# Cross-Evaluation: Longhealth A (patients 1-10) vs B (patients 11-20)
#
# Each experiment is defined explicitly as a pipe-delimited entry:
#   MODEL | CARTRIDGES (space-separated) | DATASETS (space-separated) | LABEL
#
# The script groups experiments by model so the toka server is started once
# per model, then all experiments for that model run before switching.
# =============================================================================

export CARTRIDGES_DIR=/home/vo43/cartridges
DATA_DIR="/scratch/scholar/vo43"
HF_NS="qtris123"

CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-${CARTRIDGES_DIR}/outputs}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-cross-eval_longhealth-AvB}"
RUN_ROOT="${CARTRIDGES_OUTPUT_DIR}/${EXPERIMENT_NAME}_${SLURM_JOB_ID:-$$}"

DP_SIZE="${DP_SIZE:-2}"
HF_CARTRIDGE_FILENAME="${HF_CARTRIDGE_FILENAME:-cache_last.pt}"
COT="${COT:-1}"
NUM_EVAL_QUESTIONS="${NUM_EVAL_QUESTIONS:-5}"
DEBUG="${DEBUG:-0}"
TOKA_PORT="${TOKA_PORT:-10210}"

# =============================================================================
# Experiment definitions — edit these directly
# Format: MODEL | CARTRIDGE_1 CARTRIDGE_2 ... | DATASET_1 DATASET_2 ... | LABEL
# =============================================================================
EXPERIMENTS=(
  # --- Qwen, 512 ---
  "Qwen/Qwen3-4B-Instruct-2507 | ${HF_NS}/qwen2507_longhealth-p1-10_8192_512_no-cartridge_10-epochs ${HF_NS}/qwen2507_longhealth-p11-20_8192_10-epochs_with-cartridge_p1-10_8192_512 | ${DATA_DIR}/longhealth_patient1_10-mcq.csv:mcq ${DATA_DIR}/longhealth_patient11_20-mcq.csv:mcq | qwen2507_512_mcq"
  "Qwen/Qwen3-4B-Instruct-2507 | ${HF_NS}/qwen2507_longhealth-p1-10_8192_512_no-cartridge_10-epochs ${HF_NS}/qwen2507_longhealth-p11-20_8192_10-epochs_with-cartridge_p1-10_8192_512 | ${DATA_DIR}/longhealth_patient1_10-yes-no.csv:yes_no ${DATA_DIR}/longhealth_patient11_20-yes-no.csv:yes_no | qwen2507_512_yes-no"
  # --- Qwen, 1024 ---
  "Qwen/Qwen3-4B-Instruct-2507 | ${HF_NS}/qwen2507_longhealth-p1-10_8192_1024_no-cartridge_10-epochs ${HF_NS}/qwen2507_longhealth-p11-20_8192_10-epochs_with-cartridge_p1-10_8192_1024 | ${DATA_DIR}/longhealth_patient1_10-mcq.csv:mcq ${DATA_DIR}/longhealth_patient11_20-mcq.csv:mcq | qwen2507_1024_mcq"
  "Qwen/Qwen3-4B-Instruct-2507 | ${HF_NS}/qwen2507_longhealth-p1-10_8192_1024_no-cartridge_10-epochs ${HF_NS}/qwen2507_longhealth-p11-20_8192_10-epochs_with-cartridge_p1-10_8192_1024 | ${DATA_DIR}/longhealth_patient1_10-yes-no.csv:yes_no ${DATA_DIR}/longhealth_patient11_20-yes-no.csv:yes_no | qwen2507_1024_yes-no"
  # --- Qwen, 2048 ---
  "Qwen/Qwen3-4B-Instruct-2507 | ${HF_NS}/qwen2507_longhealth-p1-10_8192_2048_no-cartridge_10-epochs ${HF_NS}/qwen2507_longhealth-p11-20_8192_10-epochs_with-cartridge_p1-10_8192_2048 | ${DATA_DIR}/longhealth_patient1_10-mcq.csv:mcq ${DATA_DIR}/longhealth_patient11_20-mcq.csv:mcq | qwen2507_2048_mcq"
  "Qwen/Qwen3-4B-Instruct-2507 | ${HF_NS}/qwen2507_longhealth-p1-10_8192_2048_no-cartridge_10-epochs ${HF_NS}/qwen2507_longhealth-p11-20_8192_10-epochs_with-cartridge_p1-10_8192_2048 | ${DATA_DIR}/longhealth_patient1_10-yes-no.csv:yes_no ${DATA_DIR}/longhealth_patient11_20-yes-no.csv:yes_no | qwen2507_2048_yes-no"

  # --- Llama, 512 ---
  "meta-llama/Llama-3.2-3B-Instruct | ${HF_NS}/llama_longhealth-p1-10_8192_512_no-cartridge_10-epochs ${HF_NS}/llama_longhealth-p11-20_8192_10-epochs_with-cartridge_p1-10_8192_512 | ${DATA_DIR}/longhealth_patient1_10-mcq.csv:mcq ${DATA_DIR}/longhealth_patient11_20-mcq.csv:mcq | llama_512_mcq"
  "meta-llama/Llama-3.2-3B-Instruct | ${HF_NS}/llama_longhealth-p1-10_8192_512_no-cartridge_10-epochs ${HF_NS}/llama_longhealth-p11-20_8192_10-epochs_with-cartridge_p1-10_8192_512 | ${DATA_DIR}/longhealth_patient1_10-yes-no.csv:yes_no ${DATA_DIR}/longhealth_patient11_20-yes-no.csv:yes_no | llama_512_yes-no"
  # --- Llama, 1024 ---
  "meta-llama/Llama-3.2-3B-Instruct | ${HF_NS}/llama_longhealth-p1-10_8192_1024_no-cartridge_10-epochs ${HF_NS}/llama_longhealth-p11-20_8192_10-epochs_with-cartridge_p1-10_8192_1024 | ${DATA_DIR}/longhealth_patient1_10-mcq.csv:mcq ${DATA_DIR}/longhealth_patient11_20-mcq.csv:mcq | llama_1024_mcq"
  "meta-llama/Llama-3.2-3B-Instruct | ${HF_NS}/llama_longhealth-p1-10_8192_1024_no-cartridge_10-epochs ${HF_NS}/llama_longhealth-p11-20_8192_10-epochs_with-cartridge_p1-10_8192_1024 | ${DATA_DIR}/longhealth_patient1_10-yes-no.csv:yes_no ${DATA_DIR}/longhealth_patient11_20-yes-no.csv:yes_no | llama_1024_yes-no"
  # --- Llama, 2048 ---
  "meta-llama/Llama-3.2-3B-Instruct | ${HF_NS}/llama_longhealth-p1-10_8192_2048_no-cartridge_10-epochs ${HF_NS}/llama_longhealth-p11-20_8192_10-epochs_with-cartridge_p1-10_8192_2048 | ${DATA_DIR}/longhealth_patient1_10-mcq.csv:mcq ${DATA_DIR}/longhealth_patient11_20-mcq.csv:mcq | llama_2048_mcq"
  "meta-llama/Llama-3.2-3B-Instruct | ${HF_NS}/llama_longhealth-p1-10_8192_2048_no-cartridge_10-epochs ${HF_NS}/llama_longhealth-p11-20_8192_10-epochs_with-cartridge_p1-10_8192_2048 | ${DATA_DIR}/longhealth_patient1_10-yes-no.csv:yes_no ${DATA_DIR}/longhealth_patient11_20-yes-no.csv:yes_no | llama_2048_yes-no"
)

mkdir -p "${RUN_ROOT}"

echo "=========================================="
echo "Cross-Evaluation: Longhealth A vs B"
echo "=========================================="
echo "JobID=${SLURM_JOB_ID:-local}"
echo "Cluster=${SLURM_CLUSTER_NAME:-local}"
echo "Node/Server=$(hostname)"
echo "Started at: $(date)"
echo "RUN_ROOT=${RUN_ROOT}"
echo "Total experiments: ${#EXPERIMENTS[@]}"
echo ""

# --- Load modules ---
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

# --- GPU monitoring ---
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
  if [ -n "${TOKA_PID:-}" ]; then
    echo "Stopping toka server (PID=${TOKA_PID}) and children..."
    kill -- -"$TOKA_PID" 2>/dev/null || true
    pkill -KILL -P "$TOKA_PID" 2>/dev/null || true
    kill -KILL "$TOKA_PID" 2>/dev/null || true
    wait "$TOKA_PID" 2>/dev/null || true
  fi
  if [ -n "${GPU_MON_PID:-}" ]; then
    kill "$GPU_MON_PID" 2>/dev/null || true
    wait "$GPU_MON_PID" 2>/dev/null || true
  fi
  echo "Job finished at: $(date)"
}
trap cleanup EXIT

# --- Activate conda ---
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cartridges
echo "Python: $(which python3)"
echo ""

# --- Build optional args ---
COMMON_ARGS=()
[[ "$COT" == "1" ]] && COMMON_ARGS+=(--cot)
[[ -n "${HF_USERNAME:-}" ]] && COMMON_ARGS+=(--hf-username "$HF_USERNAME")
[[ "$DEBUG" == "1" ]] && COMMON_ARGS+=(--debug)

# =============================================================================
# Helper: start toka server for a given model, wait until healthy
# =============================================================================
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
    if [ "$waited" -ge "$max_wait" ]; then
      echo "ERROR: toka server not ready within ${max_wait}s" >&2
      exit 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
  echo "  Server ready (took ~${waited}s)"
}

# =============================================================================
# Helper: stop toka server
# =============================================================================
stop_toka_server() {
  if [ -n "${TOKA_PID:-}" ]; then
    echo "  Stopping toka server (PID=${TOKA_PID}) and all child workers..."
    # Kill entire process group (parent + GPU worker children)
    kill -- -"$TOKA_PID" 2>/dev/null || true
    # Fallback: kill any remaining children directly
    pkill -KILL -P "$TOKA_PID" 2>/dev/null || true
    kill -KILL "$TOKA_PID" 2>/dev/null || true
    wait "$TOKA_PID" 2>/dev/null || true
    TOKA_PID=""

    # Wait for GPU memory to actually be released
    echo "  Waiting for GPU memory to free..."
    local gpu_wait=0
    while [ "$gpu_wait" -lt 60 ]; do
      local mem_used
      mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
                 | awk '{s+=$1} END {print int(s)}')
      if [ "${mem_used:-99999}" -lt 1000 ]; then
        echo "  GPU memory freed (~${mem_used} MiB in use)"
        break
      fi
      sleep 2
      gpu_wait=$((gpu_wait + 2))
    done
    if [ "$gpu_wait" -ge 60 ]; then
      echo "  WARNING: GPU memory still in use after 60s, proceeding anyway"
    fi
  fi
}

# =============================================================================
# Main experiment loop
# =============================================================================
TOTAL_EXPERIMENTS=${#EXPERIMENTS[@]}
EXPERIMENT_NUM=0
CURRENT_MODEL=""

for ENTRY in "${EXPERIMENTS[@]}"; do
  # Parse pipe-delimited fields
  IFS='|' read -r EXP_MODEL EXP_CARTRIDGES EXP_DATASETS EXP_LABEL <<< "$ENTRY"

  # Trim whitespace
  EXP_MODEL=$(echo "$EXP_MODEL" | xargs)
  EXP_CARTRIDGES=$(echo "$EXP_CARTRIDGES" | xargs)
  EXP_DATASETS=$(echo "$EXP_DATASETS" | xargs)
  EXP_LABEL=$(echo "$EXP_LABEL" | xargs)

  # Restart server if model changed
  if [[ "$EXP_MODEL" != "$CURRENT_MODEL" ]]; then
    stop_toka_server
    start_toka_server "$EXP_MODEL"
    CURRENT_MODEL="$EXP_MODEL"
  fi

  EXPERIMENT_NUM=$((EXPERIMENT_NUM + 1))

  # Split into arrays
  read -ra CARTRIDGE_ARR <<< "$EXP_CARTRIDGES"
  read -ra DATASET_ARR <<< "$EXP_DATASETS"

  EXP_DIR="${RUN_ROOT}/${EXP_LABEL}"
  mkdir -p "$EXP_DIR"

  echo ""
  echo "══════════════════════════════════════════"
  echo "  Experiment ${EXPERIMENT_NUM}/${TOTAL_EXPERIMENTS}: ${EXP_LABEL}"
  echo "  Model:      ${EXP_MODEL}"
  echo "  Cartridges: ${CARTRIDGE_ARR[*]}"
  echo "  Datasets:   ${DATASET_ARR[*]}"
  echo "  Output:     ${EXP_DIR}"
  echo "══════════════════════════════════════════"

  python3 "${CARTRIDGES_DIR}/examples/longhealth/experiments/cartridge_eval.py" \
    --backend tokasaurus \
    --url "$URL" \
    --cartridges "${CARTRIDGE_ARR[@]}" \
    --datasets "${DATASET_ARR[@]}" \
    --model "$EXP_MODEL" \
    --output-dir "$EXP_DIR" \
    "${COMMON_ARGS[@]}" \
    --max-answer-scan-tokens "$MAX_ANSWER_SCAN_TOKENS" \
    --num-eval-questions "$NUM_EVAL_QUESTIONS" \
    --top-logprobs 20 \
    --batch-size 5

  echo "  ✓ Experiment ${EXPERIMENT_NUM} complete"
done

stop_toka_server

echo ""
echo "=========================================="
echo "All ${TOTAL_EXPERIMENTS} experiments complete!"
echo "Results in: ${RUN_ROOT}"
echo "Finished at: $(date)"
echo "=========================================="
