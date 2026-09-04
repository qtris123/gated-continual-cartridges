#!/usr/bin/env bash
# Shared configuration for the HuggingFace artifact packaging/upload scripts.
# Source this from the other scripts in this directory.
set -euo pipefail

# ---- HuggingFace repos -------------------------------------------------------
HF_USER="${HF_USER:-qtris123}"
REPO_DATA="${REPO_DATA:-${HF_USER}/gated-continual-cartridges-data}"
REPO_CACHES="${REPO_CACHES:-${HF_USER}/gated-continual-cartridges-caches}"
REPO_RESULTS="${REPO_RESULTS:-${HF_USER}/gated-continual-cartridges-results}"

# ---- Local layout ------------------------------------------------------------
HF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HF_DIR/../.." && pwd)"
OUT="${OUT:-$REPO_ROOT/outputs}"
DATA="${DATA:-$REPO_ROOT/data}"
STAGE="${HF_STAGE:-$REPO_ROOT/.hf_staging}"

# Prefer the project venv's tools if present.
HF_BIN="${HF_BIN:-$REPO_ROOT/.venv/bin/hf}"
PY="${PY:-$REPO_ROOT/.venv/bin/python}"
command -v "$HF_BIN" >/dev/null 2>&1 || HF_BIN="hf"
command -v "$PY"     >/dev/null 2>&1 || PY="python3"

# ---- Packaging knobs ---------------------------------------------------------
ZSTD_LEVEL="${ZSTD_LEVEL:-3}"          # zstd compression level for shards
ZSTD_THREADS="${ZSTD_THREADS:-0}"      # 0 = all cores
# Datasets whose compacted 5-phase caches live under outputs/<ds>_5phase_runs/
RUN_CACHE_DATASETS=(finqa quality techqa)
# qasper caches already live under outputs/caches/qasper/<stage>/
QASPER_CACHE_STAGES=(p01 p02 p03 p04)

log()  { printf '\033[1;34m[hf]\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33m[hf][warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[hf][err]\033[0m %s\n' "$*" >&2; exit 1; }

require_token() {
  if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    die "HF_TOKEN is not set. export HF_TOKEN=hf_... before running."
  fi
  export HF_TOKEN="${HF_TOKEN:-$HUGGING_FACE_HUB_TOKEN}"
}
