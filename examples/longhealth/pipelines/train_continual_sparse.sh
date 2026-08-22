# Phase 2: Continual longhealth cartridge WITH TF-IDF sparse cache finetuning.
# Uses train/continual_sparse.py (SparseCacheFinetuningConfig enabled).
#
# Prerequisites:
#   - Run train_initial_sparse.sh first; set PHASE1_CACHE_PATH to its cache_last.pt.
#   - Optionally set BG_STATS_PATH to the same checkpoint (or a dedicated bg_stats.pt)
#     to enable IDF-weighted slot selection. Defaults to PHASE1_CACHE_PATH if unset.

set -e

# Configuration — adjust these as needed
export TORCH_CUDA_ARCH_LIST="8.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CARTRIDGES_DIR="${CARTRIDGES_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$CARTRIDGES_DIR/outputs}"
export PATH=${CUDA_HOME:+$CUDA_HOME/bin:}$PATH
export LD_LIBRARY_PATH=${CUDA_HOME:+$CUDA_HOME/lib64:}$LD_LIBRARY_PATH

# If CUDA_VISIBLE_DEVICES is set, default NUM_GPUS to its cardinality (so this
# script DTRT in a 2x2 split). Otherwise default to 4. Explicit NUM_GPUS wins.
if [ -z "${NUM_GPUS:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_GPUS=$(awk -F',' '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
  else
    NUM_GPUS=4
  fi
fi
# Default to gloo: NCCL deadlocks during DDP._sync_module_states on some
# PCIe topologies (e.g. RTX 5880 Ada without NVLink). gloo falls back to
# CPU/host network — slower than NCCL on hosts where NCCL works, but
# completes reliably on this one. Override with
#   DISTRIBUTED_BACKEND=nccl NCCL_P2P_DISABLE=1 ./train_continual_sparse.sh
# if you want to try NCCL on a host where it works. See the qasper2/scripts
# copy of this file (and notes/2026-06-20-ddp-gloo-flock-pool.md) for the
# longer note.
DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-gloo}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}"
# TODO: set to the cache_last.pt produced by train_initial_sparse.sh
PHASE1_CACHE_PATH="${PHASE1_CACHE_PATH:-/localhome/local-triv/gated-continual-cartridges/outputs/2026-06-17-07-18-43-initial_sparse_per-head_all-reduce/ab7d2637-8640-4edc-b8dd-a0b6c6eb6ade/cache-step266.pt}"
# TODO: set to Phase 2 training parquet (patients 11-20) when available
SYNTH_DATA_PATH="${SYNTH_DATA_PATH:-/localhome/local-triv/gated-continual-cartridges/data/longhealth/train/longhealth_p11-20_8192_no-cartridge.parquet}"
# BG_STATS_PATH defaults to PHASE1_CACHE_PATH inside continual_sparse.py if unset
BG_STATS_PATH="${BG_STATS_PATH:-/localhome/local-triv/gated-continual-cartridges/outputs/2026-06-17-07-18-43-initial_sparse_per-head_all-reduce/ab7d2637-8640-4edc-b8dd-a0b6c6eb6ade/bg_stats.pt}"
NUM_TOKENS="${NUM_TOKENS:-1024}"        # must match Phase 1 cache size
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-2}"                        # adam: 2e-2
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"   # 4 GPU: 16/GPU (~40% of 46GB); 2 GPU: 32/GPU (~80% of 46GB)
MASTER_PORT="${MASTER_PORT:-29507}"
PATIENT_IDS="${PATIENT_IDS:-11-20}"    
EVAL_EVERY_N_STEPS="${EVAL_EVERY_N_STEPS:-8}"   # halved from 15 to match doubled batch size
# Optional: set to a parquet to log perplexity in W&B alongside MCQ accuracy.
EVAL_DATA_PATH="${EVAL_DATA_PATH:-/localhome/local-triv/gated-continual-cartridges/data/longhealth/eval/patients_11_to_20.parquet}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-256}"
TOP_T="${TOP_T:-64}" # 128 256 512
MOMENTUM_MASKING="${MOMENTUM_MASKING:-freeze}"  # soft | hard | freeze | decouple
FREEZE_KEYS="${FREEZE_KEYS:-0}"                 # 1=freeze keys, 0=update keys
GRANULARITY="${GRANULARITY:-per_head}"          # must match Phase 1 GRANULARITY
IDF_TOP_K="${IDF_TOP_K:-128}"                   # top-k positions per bg batch for df
IDF_SMOOTHING="${IDF_SMOOTHING:-1.0}"           # Laplace smoothing for IDF denominator
RUN_NAME="${RUN_NAME:-longhealth_phase2_sparse_${MOMENTUM_MASKING}_key-value_adam_top-${TOP_T}_${GRANULARITY}_lr${LR}_all-reduce}"

echo "=========================================="
echo "LongHealth Phase 2 — TF-IDF Sparse Continual Cartridge"
echo "=========================================="
echo "Host=$(hostname)"
echo "Started at: $(date)"
echo ""

# Load modules only when running under a module system (e.g. SLURM cluster)
if command -v module >/dev/null 2>&1; then
  echo "Loading GCC 11.4..."
  module load gcc/11.4.1 2>/dev/null || true
  echo "GCC version: $(gcc --version | head -1)"
  echo "Loading CUDA module..."
  module load cuda/12.1.0 2>/dev/null || true
  echo "CUDA version: $(nvcc --version | grep release 2>/dev/null || echo 'nvcc not found')"
  echo ""
fi

echo "=== GPU configuration ($(hostname)) ==="
echo "CUDA_HOME=${CUDA_HOME:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "NUM_GPUS=$NUM_GPUS"
echo "DISTRIBUTED_BACKEND=$DISTRIBUTED_BACKEND"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L
  echo "--- nvidia-smi (full snapshot) ---"
  nvidia-smi
else
  echo "(nvidia-smi not found — no NVIDIA driver in PATH)"
fi

# GPU monitor: include the visible-device list and master port in the filename
# so concurrent runs (e.g. a 2x2 split) don't clobber each other's logs.
_GPU_LOG_TAG="${CUDA_VISIBLE_DEVICES:-all}"
_GPU_LOG_TAG="${_GPU_LOG_TAG//,/_}"
GPU_LOG="${GPU_LOG:-train_continual_sparse_gpu_usage_${_GPU_LOG_TAG}_${MASTER_PORT}.log}"
mkdir -p "$(dirname "$GPU_LOG")"
echo "=== GPU monitor started: $(date) on $(hostname) (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}) ===" >"$GPU_LOG"
# Limit nvidia-smi to this run's devices when possible. --id accepts the same
# global indices that CUDA_VISIBLE_DEVICES uses.
NVSMI_ID_OPTS=()
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  NVSMI_ID_OPTS=(--id="$CUDA_VISIBLE_DEVICES")
fi
(
  while true; do
    echo "index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw"
    date "+[%F %T]"
    nvidia-smi "${NVSMI_ID_OPTS[@]}" --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
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
  echo "Job finished at: $(date)"
}
trap cleanup EXIT

if [ -z "$PHASE1_CACHE_PATH" ] || [ "$PHASE1_CACHE_PATH" = "PLACEHOLDER_set_PHASE1_CACHE_PATH" ]; then
  echo "Error: PHASE1_CACHE_PATH is required — set it to the cache_last.pt from Phase 1"
  exit 1
fi

if [ -z "$SYNTH_DATA_PATH" ]; then
  echo "Error: SYNTH_DATA_PATH is required"
  exit 1
fi

# Activate the repo virtual environment (pydrantic + cartridges live in .venv)
echo "Activating .venv..."
source "$CARTRIDGES_DIR/.venv/bin/activate"
echo "Python: $(which python3)"
echo ""

echo "=== Training continual (sparse TF-IDF) cartridge ==="
echo "Model:           $MODEL_NAME"
echo "GPUs:            $NUM_GPUS  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
echo "Backend:         $DISTRIBUTED_BACKEND"
echo "Master port:     $MASTER_PORT"
echo "Phase1 cache:    $PHASE1_CACHE_PATH"
echo "BG stats:        ${BG_STATS_PATH:-<same as PHASE1_CACHE_PATH>}"
echo "Train data:      $SYNTH_DATA_PATH"
echo "Tokens:          $NUM_TOKENS"
echo "Epochs:          $EPOCHS"
echo "TOP_T:           $TOP_T"
echo "Momentum:        $MOMENTUM_MASKING"
echo "Freeze keys:     $FREEZE_KEYS"
echo "Granularity:     $GRANULARITY"
echo "IDF top-k:       $IDF_TOP_K"
echo "IDF smoothing:   $IDF_SMOOTHING"
echo "Run name:        $RUN_NAME"
echo "====================================================="


CARTRIDGES_DIR="$CARTRIDGES_DIR" \
PHASE1_CACHE_PATH="$PHASE1_CACHE_PATH" \
SYNTH_DATA_PATH="$SYNTH_DATA_PATH" \
EVAL_DATA_PATH="$EVAL_DATA_PATH" \
BG_STATS_PATH="$BG_STATS_PATH" \
NUM_TOKENS="$NUM_TOKENS" \
EPOCHS="$EPOCHS" \
LR="$LR" \
GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
MODEL_NAME="$MODEL_NAME" \
CARTRIDGES_OUTPUT_DIR="$CARTRIDGES_OUTPUT_DIR" \
PATIENT_IDS="$PATIENT_IDS" \
EVAL_EVERY_N_STEPS="$EVAL_EVERY_N_STEPS" \
SAVE_EVERY_N_STEPS="$SAVE_EVERY_N_STEPS" \
TOP_T="$TOP_T" \
MOMENTUM_MASKING="$MOMENTUM_MASKING" \
FREEZE_KEYS="$FREEZE_KEYS" \
GRANULARITY="$GRANULARITY" \
IDF_TOP_K="$IDF_TOP_K" \
IDF_SMOOTHING="$IDF_SMOOTHING" \
RUN_NAME="$RUN_NAME" \
DISTRIBUTED_BACKEND="$DISTRIBUTED_BACKEND" \
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
  "$CARTRIDGES_DIR/examples/shared/train/continual_sparse_generation.py"

echo "=== Training Done ==="
echo ""
echo "Next step: run eval_forgetting.sh with the checkpoint paths from this run"
echo "  to measure Phase 1 retention (patients 1-10) and Phase 2 acquisition (patients 11-20)."
