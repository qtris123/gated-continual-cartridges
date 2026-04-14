#!/usr/bin/env bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G 
#SBATCH --time=45:00
#SBATCH --job-name=longhealth_train_continual
#SBATCH --output=longhealth_train_continual.out
#SBATCH --error=longhealth_train_continual.err

set -e 

# Configuration — adjust these as needed
export TORCH_CUDA_ARCH_LIST="8.0"
export CARTRIDGES_DIR=/home/vo43/cartridges
export CARTRIDGES_OUTPUT_DIR=/home/vo43/cartridges/outputs
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# if [ -n "$SLURM_SUBMIT_DIR" ]; then
#   CARTRIDGES_DIR="$SLURM_SUBMIT_DIR"
#   CARTRIDGES_OUTPUT_DIR="$SLURM_SUBMIT_DIR/outputs"
# else
#   SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
#   CARTRIDGES_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
#   CARTRIDGES_OUTPUT_DIR="$CARTRIDGES_DIR/outputs"
# fi

NUM_GPUS="${NUM_GPUS:-2}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}" # "meta-llama/Llama-3.2-3B-Instruct"
PHASE1_CACHE_PATH="${PHASE1_CACHE_PATH:-/home/vo43/cartridges/outputs/2026-03-30-09-59-53-arxiv_train/af3b8e1d-2081-4089-8542-ce8fde1b6d77/cache_last.pt}"
SYNTH_DATA_PATH="${SYNTH_DATA_PATH:-/scratch/scholar/vo43/qwen_longhealth-p11-20_8192.parquet}"
NUM_TOKENS="${NUM_TOKENS:-2048}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-2e-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MASTER_PORT="${MASTER_PORT:-29507}"
PATIENT_IDS="${PATIENT_IDS:-1-10}" # "11-20"
EVAL_EVERY_N_STEPS="${EVAL_EVERY_N_STEPS:-15}" # original is 128

echo "=========================================="
echo "LongHealth Synthesis with Tokasaurus Server"
echo "=========================================="
echo "JobID=$SLURM_JOB_ID"
echo "Partition=$SLURM_JOB_PARTITION"
echo "NodeList=$SLURM_JOB_NODELIST"
echo "Started at: $(date)"
echo ""

# Load compatible GCC for CUDA (GCC 14 causes compilation issues)
echo "Loading GCC 11.4..."
module load gcc/11.4.1
echo "GCC version: $(gcc --version | head -1)"
echo ""

# Load CUDA module for nvcc compiler (needed for flashinfer)
echo "Loading CUDA module..."
module load cuda/12.1.0
echo "CUDA version: $(nvcc --version | grep release)"
echo ""


echo "=== GPU configuration ($(hostname)) ==="
echo "CUDA_HOME=${CUDA_HOME:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L
  echo "--- nvidia-smi (full snapshot) ---"
  nvidia-smi
else
  echo "(nvidia-smi not found — no NVIDIA driver in PATH)"
fi

# GPU monitoring (separate file)
GPU_LOG="train_continual_gpu_usage.log"  
mkdir -p "$(dirname "$GPU_LOG")"
echo "=== GPU monitor started: $(date) on $(hostname) ===" >"$GPU_LOG"
(
  while true; do
    echo "index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw"
    date "+[%F %T]"
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
      --format=csv,noheader,nounits 2>/dev/null || echo "(nvidia-smi query failed)"
    echo "-----"
    sleep 5
  done
) >>"$GPU_LOG" 2>&1 &
GPU_MON_PID=$!

cleanup() {
  echo ""
  echo "Cleaning up..."
  if [ -n "${GPU_MON_PID:-}" ]; then
    kill "$GPU_MON_PID" 2>/dev/null || true
    wait "$GPU_MON_PID" 2>/dev/null || true
    echo "=== GPU monitor stopped: $(date) ===" >>"${GPU_LOG:-/dev/null}"
  fi
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  echo "Job finished at: $(date)"
}
trap cleanup EXIT


if [ -z "$PHASE1_CACHE_PATH" ]; then
  echo "Error: PHASE1_CACHE_PATH is required"
  exit 1
fi

if [ -z "$SYNTH_DATA_PATH" ]; then
  echo "Error: SYNTH_DATA_PATH is required"
  exit 1
fi

# source "$REPO_DIR/.venv/bin/activate"
# if [ -f "$REPO_DIR/.env" ]; then
#   source "$REPO_DIR/.env"
# fi

# Activate conda environment
echo "Activating cartridges conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cartridges
echo "Python: $(which python3)"
echo ""

echo "=== Training continual cartridge ==="
echo "Model:    $MODEL_NAME"
echo "GPUs:     $NUM_GPUS"
echo "Phase1 cache: $PHASE1_CACHE_PATH"
echo "Data:     $SYNTH_DATA_PATH"
echo "Tokens:   $NUM_TOKENS"
echo "Epochs:   $EPOCHS"
echo "=================================="


CARTRIDGES_DIR="$CARTRIDGES_DIR" \
PHASE1_CACHE_PATH="$PHASE1_CACHE_PATH" \
SYNTH_DATA_PATH="$SYNTH_DATA_PATH" \
NUM_TOKENS="$NUM_TOKENS" \
EPOCHS="$EPOCHS" \
LR="$LR" \
GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
MODEL_NAME="$MODEL_NAME" \
CARTRIDGES_OUTPUT_DIR="$CARTRIDGES_OUTPUT_DIR" \
PATIENT_IDS="$PATIENT_IDS" \
EVAL_EVERY_N_STEPS="$EVAL_EVERY_N_STEPS" \
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
  "$CARTRIDGES_DIR/examples/longhealth/train/continual.py"

echo "=== Done ==="