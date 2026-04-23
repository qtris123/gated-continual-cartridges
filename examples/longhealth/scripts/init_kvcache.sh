#!/usr/bin/env bash
# Build step-0 trainable KV caches (KVFromText) into examples/longhealth/outputs/init_caches.
#
# Edit the variables and the python lines below for your model registry keys, text corpus,
# and output names. Each invocation is explicit (no shell loops), matching
# experiments_longhealth/scripts/init_kvcache.sh style but with outputs under longhealth/outputs/.
#
# Usage (from repo root, after conda/env and CARTRIDGES_* are set):
#   bash examples/longhealth/scripts/init_kvcache.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# cartridges/__init__.py requires these in the environment for child Python processes.
export CARTRIDGES_DIR="${CARTRIDGES_DIR:-${REPO_ROOT}}"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-${CARTRIDGES_DIR}/outputs}"

# Repo ships corpus at experiments_longhealth/longhealth_context.txt (and
# examples/longhealth/train/longhealth_context.txt). Override with INIT_KVCACHE_TEXT.
INIT_TEXT="${INIT_KVCACHE_TEXT:-${REPO_ROOT}/experiments_longhealth/longhealth_context.txt}"
if [[ ! -f "$INIT_TEXT" ]]; then
  echo "ERROR: init text not found: $INIT_TEXT" >&2
  echo "Set INIT_KVCACHE_TEXT to your corpus path." >&2
  exit 1
fi

LONGHEALTH_CACHE_DIR="${LONGHEALTH_CACHE_DIR:-${REPO_ROOT}/examples/longhealth/outputs/init_caches}"
mkdir -p "$LONGHEALTH_CACHE_DIR"

BUILD_PY="${REPO_ROOT}/experiments_longhealth/swapping/build_init_cache.py"
if [[ ! -f "$BUILD_PY" ]]; then
  echo "ERROR: build_init_cache.py not found at $BUILD_PY" >&2
  exit 1
fi

echo "REPO_ROOT=$REPO_ROOT"
echo "LONGHEALTH_CACHE_DIR=$LONGHEALTH_CACHE_DIR"
echo "INIT_TEXT=$INIT_TEXT"
echo ""

# So `from experiments.intervention import ...` resolves (repo root on PYTHONPATH).
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# --- Qwen3 (registry key must match experiments.intervention MODEL_REGISTRY) ---
python3 "$BUILD_PY" --model qwen3-4b-instruct --text "$INIT_TEXT" --max-tokens 512 \
  --output "${LONGHEALTH_CACHE_DIR}/qwen3_512_init_cache_last.pt"
python3 "$BUILD_PY" --model qwen3-4b-instruct --text "$INIT_TEXT" --max-tokens 1024 \
  --output "${LONGHEALTH_CACHE_DIR}/qwen3_1024_init_cache_last.pt"
python3 "$BUILD_PY" --model qwen3-4b-instruct --text "$INIT_TEXT" --max-tokens 2048 \
  --output "${LONGHEALTH_CACHE_DIR}/qwen3_2048_init_cache_last.pt"

# --- Llama 3.2 3B ---
python3 "$BUILD_PY" --model llama-3.2-3b-instruct --text "$INIT_TEXT" --max-tokens 512 \
  --output "${LONGHEALTH_CACHE_DIR}/llama_512_init_cache_last.pt"
python3 "$BUILD_PY" --model llama-3.2-3b-instruct --text "$INIT_TEXT" --max-tokens 1024 \
  --output "${LONGHEALTH_CACHE_DIR}/llama_1024_init_cache_last.pt"
python3 "$BUILD_PY" --model llama-3.2-3b-instruct --text "$INIT_TEXT" --max-tokens 2048 \
  --output "${LONGHEALTH_CACHE_DIR}/llama_2048_init_cache_last.pt"

echo "Done. Caches under: $LONGHEALTH_CACHE_DIR"
