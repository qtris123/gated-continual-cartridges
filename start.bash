#!/usr/bin/env bash
# Bootstrap materials for Qasper ICL / cartridge / sparse-grad reproduction.
#
# Downloads (or verifies) everything needed before running the pipelines in:
#   notes/2026-07-27-repro-qwen3-8b.md
#   notes/2026-07-27-repro-qwen3-30b-a3b.md
#   research_loop/RUNBOOK.md
#
# Usage:
#   bash start.bash              # print links + download missing artifacts
#   bash start.bash --dry-run    # print links / status only, no downloads
#   bash start.bash --links-only # same as --dry-run
#
# Requires: HF_TOKEN (for gated models / private artifacts), python with huggingface_hub
# Optional: SKIP_MODEL_WARMUP=1 to skip pulling the base model weights

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$SCRIPT_DIR"
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run|--links-only) DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
  esac
done

export CARTRIDGES_DIR="${CARTRIDGES_DIR:-$REPO}"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$REPO/outputs}"
export PYTHONPATH="$CARTRIDGES_DIR${PYTHONPATH:+:$PYTHONPATH}"

PY="${PY:-}"
if [[ -z "$PY" ]]; then
  if [[ -x "$REPO/.venv/bin/python" ]]; then
    PY="$REPO/.venv/bin/python"
  else
    PY="$(command -v python3 || command -v python)"
  fi
fi

# ---------------------------------------------------------------------------
# Artifact map (HF repo ids use the historical typo "qapser")
# ---------------------------------------------------------------------------
QA_TRAIN_HF="qtris123/qwen_qapser-QA-task_8192_no-cartridge"
MT_TRAIN_HF="qtris123/qwen_qapser-MT-task_8192_no-cartridge"
QA_TRAIN_URL="https://huggingface.co/datasets/${QA_TRAIN_HF}"
MT_TRAIN_URL="https://huggingface.co/datasets/${MT_TRAIN_HF}"

PHASE1_CART_HF="qtris123/qwen_qasper-QA-task_8192_512_no-cartridge_10-epochs"
PHASE2_CART_HF="qtris123/qwen_qasper-MT-task_8192_10-epochs_with-cartridge_qasper-QA-task_8192_512"
PHASE1_CART_URL="https://huggingface.co/${PHASE1_CART_HF}"
PHASE2_CART_URL="https://huggingface.co/${PHASE2_CART_HF}"

MODEL_4B="Qwen/Qwen3-4B-Instruct-2507"
MODEL_8B="Qwen/Qwen3-8B"
MODEL_30B="Qwen/Qwen3-30B-A3B-Instruct-2507"

QA_TRAIN_DST="$REPO/data/qasper/train/qwen_qasper_QA_task_8192.parquet"
MT_TRAIN_DST="$REPO/data/qasper/train/qwen_qasper_MT_task_8192.parquet"
PHASE1_DST="$REPO/outputs/phase1_selfdistill_qwen512/cache_last.pt"
PHASE2_DST="$REPO/outputs/refcart_phase2_qwen512/cache_last.pt"

QA_EVAL="$REPO/data/qasper/eval/qasper_eval_QA.parquet"
MT_EVAL="$REPO/data/qasper/eval/qasper_eval_MT.parquet"
INIT_TEXT="$REPO/examples/qasper2/train/qwen_qasper_init_512.txt"
INIT_TEXT_ALT="$REPO/data/qasper/init_text/qasper_init_512.txt"

ok()   { printf '  [ok]   %s\n' "$*"; }
miss() { printf '  [miss] %s\n' "$*"; }
info() { printf '  [..]   %s\n' "$*"; }

echo "=============================================="
echo " Cartridges Qasper — material bootstrap"
echo "=============================================="
echo "REPO=$REPO"
echo "PY=$PY"
echo "CARTRIDGES_DIR=$CARTRIDGES_DIR"
echo "CARTRIDGES_OUTPUT_DIR=$CARTRIDGES_OUTPUT_DIR"
echo "DRY_RUN=$DRY_RUN"
echo ""

echo "== Links (bookmark these) =="
cat <<EOF
  Train QA (self-study, 8192):  $QA_TRAIN_URL
  Train MT (self-study, 8192):  $MT_TRAIN_URL
  Phase-1 cartridge (4B):       $PHASE1_CART_URL
  Phase-2 cartridge (4B ref):   $PHASE2_CART_URL
  Model 4B:  https://huggingface.co/$MODEL_4B
  Model 8B:  https://huggingface.co/$MODEL_8B
  Model 30B: https://huggingface.co/$MODEL_30B
  Tokasaurus (synth regen):     https://github.com/ScalingIntelligence/tokasaurus
  Paper / upstream README:      $REPO/README.md
EOF
echo ""

# ---------------------------------------------------------------------------
# 1. Env / tokens
# ---------------------------------------------------------------------------
echo "== 1. Environment =="
if [[ -z "${HF_TOKEN:-}" ]]; then
  miss "HF_TOKEN unset — set it before downloading gated models / HF datasets"
else
  ok "HF_TOKEN is set"
fi
if [[ ! -x "$PY" ]] && ! command -v "$PY" >/dev/null 2>&1; then
  miss "Python not found at PY=$PY — create .venv and: uv pip install -e .  (see README)"
  exit 1
fi
ok "python: $($PY -c 'import sys; print(sys.executable, sys.version.split()[0])')"

if ! "$PY" -c 'import huggingface_hub' 2>/dev/null; then
  if (( DRY_RUN )); then
    miss "huggingface_hub not importable (needed for downloads)"
  else
    info "installing huggingface_hub into current env"
    "$PY" -m pip install -q huggingface_hub
  fi
fi

# Dual-package foot-gun (F1)
CART_LOC="$("$PY" -c 'import cartridges,os; print(os.path.dirname(cartridges.__file__))' 2>/dev/null || true)"
if [[ -n "$CART_LOC" ]]; then
  if [[ "$CART_LOC" == "$REPO/cartridges"* ]]; then
    ok "import cartridges -> $CART_LOC"
  else
    miss "import cartridges -> $CART_LOC  (expected under $REPO/cartridges)"
    info "export PYTHONPATH=\"$CARTRIDGES_DIR:\$PYTHONPATH\" before training"
  fi
else
  miss "cannot import cartridges — install with: cd $REPO && uv pip install -e ."
fi
echo ""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
download_dataset_parquet() {
  local repo_id="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  if [[ -f "$dest" ]]; then
    ok "already present: $dest ($(du -h "$dest" | cut -f1))"
    return 0
  fi
  if (( DRY_RUN )); then
    miss "would download $repo_id → $dest"
    return 0
  fi
  info "downloading dataset $repo_id → $dest"
  "$PY" - "$repo_id" "$dest" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
repo_id, dest = sys.argv[1], Path(sys.argv[2])
src = hf_hub_download(repo_id=repo_id, filename="dataset.parquet", repo_type="dataset")
dest.parent.mkdir(parents=True, exist_ok=True)
# Copy (not symlink) so gitignored data/ stays self-contained if the hub cache moves.
import shutil
shutil.copy2(src, dest)
print(f"wrote {dest} ({dest.stat().st_size} bytes) from {src}")
PY
}

download_model_file() {
  local repo_id="$1" filename="$2" dest="$3"
  mkdir -p "$(dirname "$dest")"
  if [[ -f "$dest" ]] && [[ ! -L "$dest" || -e "$dest" ]]; then
    # resolve dangling symlink
    if [[ -L "$dest" ]] && [[ ! -e "$dest" ]]; then
      miss "dangling symlink: $dest — re-downloading"
      rm -f "$dest"
    else
      ok "already present: $dest ($(du -h "$dest" | cut -f1))"
      return 0
    fi
  fi
  if (( DRY_RUN )); then
    miss "would download $repo_id/$filename → $dest"
    return 0
  fi
  info "downloading $repo_id/$filename → $dest"
  "$PY" - "$repo_id" "$filename" "$dest" <<'PY'
import sys, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
repo_id, filename, dest = sys.argv[1], sys.argv[2], Path(sys.argv[3])
src = hf_hub_download(repo_id=repo_id, filename=filename)
dest.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(src, dest)
print(f"wrote {dest} ({dest.stat().st_size} bytes)")
PY
}

# ---------------------------------------------------------------------------
# 2. In-repo materials (no download)
# ---------------------------------------------------------------------------
echo "== 2. In-repo materials (git-tracked) =="
for f in "$QA_EVAL" "$MT_EVAL" "$INIT_TEXT" "$INIT_TEXT_ALT"; do
  if [[ -f "$f" ]]; then ok "$f"
  else miss "$f"
  fi
done
echo ""

# ---------------------------------------------------------------------------
# 3. Train synth parquets
# ---------------------------------------------------------------------------
echo "== 3. Train self-study parquets =="
echo "  QA: $QA_TRAIN_URL"
echo "  MT: $MT_TRAIN_URL"
download_dataset_parquet "$QA_TRAIN_HF" "$QA_TRAIN_DST"
download_dataset_parquet "$MT_TRAIN_HF" "$MT_TRAIN_DST"
echo ""

# ---------------------------------------------------------------------------
# 4. Reference cartridges (4B study)
# ---------------------------------------------------------------------------
echo "== 4. Reference cartridges (Qwen3-4B study) =="
echo "  Phase-1: $PHASE1_CART_URL"
echo "  Phase-2: $PHASE2_CART_URL"
download_model_file "$PHASE1_CART_HF" "cache_last.pt" "$PHASE1_DST"
# Phase-2 optional for 8B/30B repro (those train Phase-1 fresh), but handy for 4B parity checks
download_model_file "$PHASE2_CART_HF" "cache_last.pt" "$PHASE2_DST"
echo ""

# ---------------------------------------------------------------------------
# 5. Optional model warm-up (pulls weights into HF cache)
# ---------------------------------------------------------------------------
echo "== 5. Base models =="
echo "  Default study model: $MODEL_4B"
echo "  8B repro:            $MODEL_8B   (notes/2026-07-27-repro-qwen3-8b.md)"
echo "  30B MoE repro:       $MODEL_30B  (notes/2026-07-27-repro-qwen3-30b-a3b.md)"
WARM_MODEL="${WARM_MODEL:-$MODEL_4B}"
if [[ "${SKIP_MODEL_WARMUP:-0}" == "1" ]]; then
  info "SKIP_MODEL_WARMUP=1 — not pulling $WARM_MODEL"
elif (( DRY_RUN )); then
  info "would warm HF cache for $WARM_MODEL (set WARM_MODEL=... or SKIP_MODEL_WARMUP=1)"
else
  info "warming HF cache for $WARM_MODEL (tokenizer only; full weights load on first train)"
  "$PY" -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$WARM_MODEL'); print('tokenizer ok')"
fi
echo ""

# ---------------------------------------------------------------------------
# 6. Summary / next steps
# ---------------------------------------------------------------------------
echo "== 6. Next steps =="
cat <<EOF
  Export once per shell:
    cd $REPO
    export CARTRIDGES_DIR=\$PWD
    export CARTRIDGES_OUTPUT_DIR=\$PWD/outputs
    export PYTHONPATH="\$CARTRIDGES_DIR:\$PYTHONPATH"
    PY=\$PWD/.venv/bin/python

  Sanity (4B / dense Qwen3):
    see notes/2026-07-27-repro-qwen3-8b.md §3 (swap MODEL=)

  Train paths expected by scripts:
    SYNTH QA → $QA_TRAIN_DST
    SYNTH MT → $MT_TRAIN_DST
    Phase-1  → $PHASE1_DST

  To regenerate synth instead of downloading:
    bash examples/qasper2/scripts/synthesize_self_study.sh
EOF

echo ""
echo "Done."
