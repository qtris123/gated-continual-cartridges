#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Configuration ──────────────────────────────────────────────────────
CARTRIDGES_DIR="${CARTRIDGES_DIR:-$HOME/continual-cartridges}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}"
MODEL_NAME_2="${MODEL_NAME_2:-Qwen/Qwen3-4B}"
NUM_GPUS="${NUM_GPUS:-1}"
DP_SIZE="${DP_SIZE:-$NUM_GPUS}"
PORT="${PORT:-8000}"
TOKA_DIR="${TOKA_DIR:-$HOME/tokasaurus}"
COMPANY="${COMPANY:-AMD}"
YEAR_Y="${YEAR_Y:-2021}"
YEAR_Y1="${YEAR_Y1:-$((YEAR_Y + 1))}"
COMPANY_2="${COMPANY_2:-PEPSICO}"
YEAR_2="${YEAR_2:-2021}"
NUM_SAMPLES="${NUM_SAMPLES:-8192}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NUM_BATCHES="${MAX_NUM_BATCHES:-256}"
NUM_TOKENS="${NUM_TOKENS:-1024}"
HF_REPO_ID="${HF_REPO_ID:-Phudish/amd-2021-llama-3.2-3b-cartridge-without-year-experiment-1}"
HF_REPO_ID_2="${HF_REPO_ID_2:-Phudish/amd-2021-qwen3-4b-cartridge-without-year-experiment-1}"
HF_REPO_ID_TAGGED="${HF_REPO_ID_TAGGED:-}"    # Phase 2 tagged cartridge — model 1
HF_REPO_ID_2_TAGGED="${HF_REPO_ID_2_TAGGED:-}" # Phase 2 tagged cartridge — model 2
LR="${LR:-2e-2}"
EPOCHS="${EPOCHS:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
TOP_K_LOGITS="${TOP_K_LOGITS:-20}"
PACKED_SEQ_LENGTH="${PACKED_SEQ_LENGTH:-2048}"
LOSS_EVAL_EVERY_N_STEPS="${LOSS_EVAL_EVERY_N_STEPS:-1}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-256}"
DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-gloo}"
PROVENANCE_TAG="${PROVENANCE_TAG:-}"   # empty = no tag; set e.g. "AMD 2021" for A→B runs

TEXT_PATH="${TEXT_PATH:-$SCRIPT_DIR/data/texts/${COMPANY}_${YEAR_Y}_10K.txt}"

# ── Validate required vars ─────────────────────────────────────────────
if [ -z "$HF_REPO_ID" ] || [ -z "$HF_REPO_ID_2" ] || \
   [ -z "$HF_REPO_ID_TAGGED" ] || [ -z "$HF_REPO_ID_2_TAGGED" ] || \
   [ -z "$PROVENANCE_TAG" ]; then
  echo "Error: HF_REPO_ID, HF_REPO_ID_2, HF_REPO_ID_TAGGED, HF_REPO_ID_2_TAGGED, and PROVENANCE_TAG are required."
  exit 1
fi

# ── Cleanup: kill the server process group when the script exits ───────
SERVER_PID=""
stop_server() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo ""
    echo "=== Stopping Tokasaurus server process group (PGID $SERVER_PID) ==="
    kill -9 -- -"$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    echo "=== Tokasaurus server stopped ==="
  fi
}
trap stop_server EXIT

# ── Install Tokasaurus if needed ───────────────────────────────────────
if [ ! -d "$TOKA_DIR/.venv" ]; then
  echo "=== Tokasaurus not found, installing... ==="
  if [ ! -d "$TOKA_DIR" ]; then
    git clone https://github.com/ScalingIntelligence/tokasaurus "$TOKA_DIR"
  fi
  cd "$TOKA_DIR"
  git checkout --track origin/sabri/batch 2>/dev/null || git checkout geoff/cartridges
  uv venv
  uv sync
  echo "=== Tokasaurus installed ==="
fi

# ── Set up continual-cartridges ────────────────────────────────────────
if [ ! -d "$CARTRIDGES_DIR/.git" ]; then
  echo "=== continual-cartridges repo not found, cloning... ==="
  git clone https://github.com/faridlazuarda/continual-cartridges.git "$CARTRIDGES_DIR"
  cd "$CARTRIDGES_DIR"
  git checkout --track origin/stress_test 2>/dev/null || git checkout experiment_1
  echo "=== continual-cartridges cloned ==="
fi

if [ ! -d "$CARTRIDGES_DIR/.venv" ]; then
  echo "=== Setting up cartridges venv... ==="
  cd "$CARTRIDGES_DIR"
  uv venv
  uv sync
  echo "=== cartridges venv created ==="
fi

cd "$CARTRIDGES_DIR"
source "$CARTRIDGES_DIR/.venv/bin/activate"

# ── Export environment variables ────────────────────────────────────────
export CARTRIDGES_DIR="$CARTRIDGES_DIR"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$CARTRIDGES_DIR/outputs}"
export CARTRIDGES_WANDB_PROJECT="${CARTRIDGES_WANDB_PROJECT:-cartridges}"
export CARTRIDGES_WANDB_ENTITY="${CARTRIDGES_WANDB_ENTITY:-phudishp-scb-datax}"
export CARTRIDGES_TOKASAURUS_URL="${CARTRIDGES_TOKASAURUS_URL:-http://localhost:$PORT}"

echo ""
echo "=== Experiment 1: Cartridge-Conditioned Self-Study Data Generation ==="
echo "Company:   $COMPANY"
echo "Year Y:    $YEAR_Y"
echo "Year Y+1:  $YEAR_Y1"
echo "Model 1:   $MODEL_NAME  ->  $HF_REPO_ID"
echo "Model 2:   $MODEL_NAME_2  ->  $HF_REPO_ID_2"
echo "Output:    $CARTRIDGES_OUTPUT_DIR"
echo "======================================================================"

# ── Helper: start Tokasaurus server ───────────────────────────────────
start_server() {
  local model="$1"
  echo "=== Starting Tokasaurus server: $model ==="
  setsid bash -c "
    source '$TOKA_DIR/.venv/bin/activate'
    exec tksrs \
      model=$model \
      kv_cache_num_tokens='(256 * 1024)' \
      max_topk_logprobs=20 \
      dp_size=$DP_SIZE \
      port=$PORT
  " &
  SERVER_PID=$!
  local MAX_WAIT=300 WAITED=0
  until curl -so /dev/null -w '' "http://localhost:$PORT/" 2>/dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Error: Tokasaurus server exited unexpectedly"
      exit 1
    fi
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
      echo "Error: Tokasaurus server did not start within ${MAX_WAIT}s"
      exit 1
    fi
    sleep 2
    WAITED=$((WAITED + 2))
  done
  echo "=== Tokasaurus server ready (waited ${WAITED}s) ==="
}

# ── Per-model pipeline ─────────────────────────────────────────────────
run_for_model() {
  local MODEL="$1"
  local HF_REPO="$2"

  echo ""
  echo "======================================================================"
  echo "=== Running pipeline for model: $MODEL ==="
  echo "=== HF repo: $HF_REPO ==="
  echo "======================================================================"

  # Step 1: Synthesize self-study data on year Y (no cartridge)
  echo ""
  echo "=== Step 1: Synthesize self-study data for $COMPANY $YEAR_Y (no cartridge) ==="
  start_server "$MODEL"
  python experiments/experiments_1/synthesize_self_study_data.py \
    --company "$COMPANY" \
    --year "$YEAR_Y" \
    --model "$MODEL" \
    --num_samples "$NUM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --max_num_batches "$MAX_NUM_BATCHES"

  STEP1_PARQUET=$(find "$CARTRIDGES_OUTPUT_DIR" \
    -path "*synthesize_${COMPANY,,}_${YEAR_Y}*" -name "dataset.parquet" | sort | tail -1)
  if [ -z "$STEP1_PARQUET" ]; then
    echo "Error: Step 1 parquet not found in $CARTRIDGES_OUTPUT_DIR"
    exit 1
  fi
  echo "=== Step 1 complete: $STEP1_PARQUET ==="

  # Stop server to free GPU memory before training
  echo ""
  echo "=== Stopping Tokasaurus server to free GPU memory for training ==="
  stop_server

  # Step 2: Train cartridge on year Y
  echo ""
  echo "=== Step 2: Train cartridge on $COMPANY $YEAR_Y ==="
  MODEL_NAME="$MODEL" \
  SYNTH_DATA_PATH_PHASE1="$STEP1_PARQUET" \
  COMPANY="$COMPANY" \
  YEAR_Y="$YEAR_Y" \
  TEXT_PATH="$TEXT_PATH" \
  NUM_TOKENS="$NUM_TOKENS" \
  LR="$LR" \
  EPOCHS="$EPOCHS" \
  GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
  TOP_K_LOGITS="$TOP_K_LOGITS" \
  PACKED_SEQ_LENGTH="$PACKED_SEQ_LENGTH" \
  LOSS_EVAL_EVERY_N_STEPS="$LOSS_EVAL_EVERY_N_STEPS" \
  SAVE_EVERY_N_STEPS="$SAVE_EVERY_N_STEPS" \
  DISTRIBUTED_BACKEND="$DISTRIBUTED_BACKEND" \
  PROVENANCE_TAG="$PROVENANCE_TAG" \
  torchrun --standalone --nproc_per_node="$NUM_GPUS" \
    experiments/experiments_1/train_initial.py

  CARTRIDGE_PT=$(find "$CARTRIDGES_OUTPUT_DIR" -path "*train_initial*" \
    -name "cache_last.pt" | sort | tail -1)
  if [ -z "$CARTRIDGE_PT" ]; then
    echo "Error: Cartridge .pt file not found in $CARTRIDGES_OUTPUT_DIR"
    exit 1
  fi
  echo "=== Step 2 complete: $CARTRIDGE_PT ==="

  # Step 3: Upload year-Y cartridge to HuggingFace
  echo ""
  echo "=== Step 3: Upload cartridge to HuggingFace ($HF_REPO) ==="
  python experiments/experiments_1/upload_cartridge_to_hf.py \
    --cartridge-path "$CARTRIDGE_PT" \
    --hf-repo-id "$HF_REPO" \
    --model-name "$MODEL"
  echo "=== Step 3 complete: uploaded to $HF_REPO ==="

  # Steps 4a + 4b: Synthesize on year Y+1 (baseline and with cartridge)
  echo ""
  echo "=== Restarting Tokasaurus server for synthesis steps ==="
  start_server "$MODEL"

  echo ""
  echo "=== Step 4a: Synthesize self-study data for $COMPANY $YEAR_Y1 (baseline, no cartridge) ==="
  python experiments/experiments_1/synthesize_self_study_data.py \
    --company "$COMPANY" \
    --year "$YEAR_Y1" \
    --model "$MODEL" \
    --num_samples "$NUM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --max_num_batches "$MAX_NUM_BATCHES"
  echo "=== Step 4a complete ==="

  echo ""
  echo "=== Step 4b: Synthesize self-study data for $COMPANY $YEAR_Y1 (with cartridge $HF_REPO) ==="
  python experiments/experiments_1/synthesize_self_study_data_with_cartridge.py \
    --company "$COMPANY" \
    --year "$YEAR_Y1" \
    --model "$MODEL" \
    --num_samples "$NUM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --max_num_batches "$MAX_NUM_BATCHES" \
    --cartridge-hf-id "$HF_REPO"
  echo "=== Step 4b complete ==="

  stop_server

  echo ""
  echo "=== Model $MODEL complete. Cartridge: $HF_REPO ==="
}

# ── Phase 2: A→B pipeline (re-train with PROVENANCE_TAG, then synth B) ──
run_for_model_b() {
  local MODEL="$1"
  local HF_REPO_TAGGED="$2"

  echo ""
  echo "======================================================================"
  echo "=== Phase 2 (A→B): Re-train with tag, then synth $COMPANY_2 $YEAR_2 ==="
  echo "=== Tagged HF repo: $HF_REPO_TAGGED ==="
  echo "======================================================================"

  # Locate Phase 1 parquet (produced by run_for_model Step 1)
  STEP1_PARQUET=$(find "$CARTRIDGES_OUTPUT_DIR" \
    -path "*synthesize_${COMPANY,,}_${YEAR_Y}*" -name "dataset.parquet" | sort | tail -1)
  if [ -z "$STEP1_PARQUET" ]; then
    echo "Error: Phase 1 parquet not found; run Phase 1 (run_for_model) first."
    exit 1
  fi
  echo "=== Using Phase 1 parquet: $STEP1_PARQUET ==="

  # Step B1: Re-train cartridge with PROVENANCE_TAG stamped
  echo ""
  echo "=== Step B1: Train tagged cartridge (PROVENANCE_TAG='$PROVENANCE_TAG') ==="
  MODEL_NAME="$MODEL" \
  SYNTH_DATA_PATH_PHASE1="$STEP1_PARQUET" \
  COMPANY="$COMPANY" \
  YEAR_Y="$YEAR_Y" \
  TEXT_PATH="$TEXT_PATH" \
  NUM_TOKENS="$NUM_TOKENS" \
  LR="$LR" \
  EPOCHS="$EPOCHS" \
  GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
  TOP_K_LOGITS="$TOP_K_LOGITS" \
  PACKED_SEQ_LENGTH="$PACKED_SEQ_LENGTH" \
  LOSS_EVAL_EVERY_N_STEPS="$LOSS_EVAL_EVERY_N_STEPS" \
  SAVE_EVERY_N_STEPS="$SAVE_EVERY_N_STEPS" \
  DISTRIBUTED_BACKEND="$DISTRIBUTED_BACKEND" \
  PROVENANCE_TAG="$PROVENANCE_TAG" \
  torchrun --standalone --nproc_per_node="$NUM_GPUS" \
    experiments/experiments_1/train_initial.py

  CARTRIDGE_PT=$(find "$CARTRIDGES_OUTPUT_DIR" -path "*train_initial*" \
    -name "cache_last.pt" | sort | tail -1)
  if [ -z "$CARTRIDGE_PT" ]; then
    echo "Error: Tagged cartridge .pt not found"
    exit 1
  fi
  echo "=== Step B1 complete: $CARTRIDGE_PT ==="

  # Step B2: Upload tagged cartridge to HF
  echo ""
  echo "=== Step B2: Upload tagged cartridge to HuggingFace ($HF_REPO_TAGGED) ==="
  python experiments/experiments_1/upload_cartridge_to_hf.py \
    --cartridge-path "$CARTRIDGE_PT" \
    --hf-repo-id "$HF_REPO_TAGGED" \
    --model-name "$MODEL"
  echo "=== Step B2 complete: uploaded to $HF_REPO_TAGGED ==="

  # Step B3: Synth self-study for Company B using tagged cartridge
  echo ""
  echo "=== Step B3: Synthesize self-study for $COMPANY_2 $YEAR_2 (tagged cartridge) ==="
  start_server "$MODEL"
  python experiments/experiments_1/synthesize_self_study_data_with_cartridge.py \
    --company "$COMPANY_2" \
    --year "$YEAR_2" \
    --model "$MODEL" \
    --num_samples "$NUM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --max_num_batches "$MAX_NUM_BATCHES" \
    --cartridge-hf-id "$HF_REPO_TAGGED"
  echo "=== Step B3 complete ==="

  stop_server

  echo ""
  echo "=== Phase 2 for $MODEL complete. Tagged cartridge: $HF_REPO_TAGGED ==="
}

# ── Main execution ─────────────────────────────────────────────────────

# ── Phase 1: A + δA (no tag) ───────────────────────────────────────────
run_for_model "$MODEL_NAME"   "$HF_REPO_ID"
run_for_model "$MODEL_NAME_2" "$HF_REPO_ID_2"

# ── Phase 2: A→B (re-train with PROVENANCE_TAG, synth B) ───────────────
run_for_model_b "$MODEL_NAME"   "$HF_REPO_ID_TAGGED"
run_for_model_b "$MODEL_NAME_2" "$HF_REPO_ID_2_TAGGED"

echo ""
echo "=== Experiment 1 complete ==="