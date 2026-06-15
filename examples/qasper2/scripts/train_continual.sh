#!/usr/bin/env bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH -p scholar-j
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G 
#SBATCH --time=3:00:00
#SBATCH --job-name=qasper_train_continual
#SBATCH --output=qasper_train_continual.out
#SBATCH --error=qasper_train_continual.err

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
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}" # Qwen/Qwen3-4B-Instruct-2507}" #
PHASE1_CACHE_PATH="${PHASE1_CACHE_PATH:-/home/vo43/cartridges/outputs_transfer/2026-06-14-02-43-19-initial/1de8f704-82ed-4c3e-8e30-9f8b77dda4f0/cache_last.pt}" 
SYNTH_DATA_PATH="${SYNTH_DATA_PATH:-/scratch/scholar/vo43/qasper-MT_8192_off-policy.parquet}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-/home/vo43/cartridges/examples/qasper2/qasper_eval_MT.parquet}"
NUM_TOKENS="${NUM_TOKENS:-1024}" # check phase 1 cache size for setting this
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-2}" # adam is 2e-2, sgd is 2
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}" #8 32
MASTER_PORT="${MASTER_PORT:-29507}"
EVAL_EVERY_N_STEPS="${EVAL_EVERY_N_STEPS:-15}" #45- bs 32 is 15 - original is 128
RUN_NAME="${RUN_NAME:-qasper_phase2_freeze_value-only_adam_top-64_per-head_lr2e-2}" # !!!
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-256}" # 768 -bs 32 is 256
TOP_T="${TOP_T:-64}"
MOMENTUM_MASKING="${MOMENTUM_MASKING:-freeze}" # soft | hard | freeze | decouple
FREEZE_KEYS="${FREEZE_KEYS:-0}"              # 1=freeze keys (default), 0=update keys
GRANULARITY="${GRANULARITY:-per_layer}"         # global | per_layer | per_head
IDF_TOP_K="${IDF_TOP_K:-128}"               # top-k positions per bg batch counted toward df
IDF_SMOOTHING="${IDF_SMOOTHING:-1.0}"       # Laplace smoothing for IDF denominator
echo "=========================================="
echo "Qasper Synthesis with Tokasaurus Server"
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
EVAL_DATA_PATH="$EVAL_DATA_PATH" \
NUM_TOKENS="$NUM_TOKENS" \
EPOCHS="$EPOCHS" \
LR="$LR" \
GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
MODEL_NAME="$MODEL_NAME" \
CARTRIDGES_OUTPUT_DIR="$CARTRIDGES_OUTPUT_DIR" \
EVAL_EVERY_N_STEPS="$EVAL_EVERY_N_STEPS" \
SAVE_EVERY_N_STEPS="$SAVE_EVERY_N_STEPS" \
TOP_T="$TOP_T" \
MOMENTUM_MASKING="$MOMENTUM_MASKING" \
FREEZE_KEYS="$FREEZE_KEYS" \
GRANULARITY="$GRANULARITY" \
IDF_TOP_K="$IDF_TOP_K" \
IDF_SMOOTHING="$IDF_SMOOTHING" \
RUN_NAME="$RUN_NAME" \
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
  "$CARTRIDGES_DIR/examples/qasper2/train/continual.py"

echo "=== Training Done ==="

# Generate visualization plots from the latest output folder
echo "=== Generating Visualization Plots ==="

# Find the most recent output folder (format: YYYY-MM-DD-HH-MM-SS-{name}/uuid/)
LATEST_OUTPUT=$(find "$CARTRIDGES_OUTPUT_DIR" -maxdepth 2 -type d -name "*-*-*-*-*-*-*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

if [ -n "$LATEST_OUTPUT" ]; then
    # Find the UUID subdirectory (most recent)
    LATEST_RUN=$(find "$LATEST_OUTPUT" -maxdepth 1 -type d ! -path "$LATEST_OUTPUT" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [ -n "$LATEST_RUN" ] && [ -d "$LATEST_RUN" ]; then
        echo "Found output folder: $LATEST_RUN"
        python3 "$CARTRIDGES_DIR/examples/qasper2/viz/generate_plots.py" \
            "$LATEST_RUN" \
            --n_slots "$NUM_TOKENS" \
            --bin_size 16 \
            --top_k 50
    else
        echo "Warning: Could not find run subdirectory in $LATEST_OUTPUT"
    fi
else
    echo "Warning: Could not find output folder in $CARTRIDGES_OUTPUT_DIR"
fi

echo "=== All Done ==="