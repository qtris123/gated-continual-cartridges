#!/usr/bin/env bash
#
# prepare_artifacts.sh — download and set up the gated-continual-cartridges
# artifacts (data / compacted caches / results) from HuggingFace so a fresh
# checkout matches the machine the experiments were run on.
#
#   scripts/prepare_artifacts.sh [options]
#
# Options:
#   --which data|caches|results|all   what to fetch            (default: all)
#   --dest  <dir>                      repo root to populate    (default: repo root)
#   --user  <hf-user>                  HF namespace             (default: qtris123)
#   --token <hf_token>                 HF token (else $HF_TOKEN)
#   --keep-archives                    keep downloaded snapshots + shards
#   -h|--help
#
# Datasets (override with $REPO_DATA / $REPO_CACHES / $REPO_RESULTS):
#   <user>/gated-continual-cartridges-data
#   <user>/gated-continual-cartridges-caches
#   <user>/gated-continual-cartridges-results
set -euo pipefail

# ---- defaults ----------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$(cd "$SCRIPT_DIR/.." && pwd)"
WHICH="all"
HF_USER="${HF_USER:-qtris123}"
KEEP_ARCHIVES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --which) WHICH="$2"; shift 2 ;;
    --dest)  DEST="$(cd "$2" && pwd)"; shift 2 ;;
    --user)  HF_USER="$2"; shift 2 ;;
    --token) export HF_TOKEN="$2"; shift 2 ;;
    --keep-archives) KEEP_ARCHIVES=1; shift ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

REPO_DATA="${REPO_DATA:-${HF_USER}/gated-continual-cartridges-data}"
REPO_CACHES="${REPO_CACHES:-${HF_USER}/gated-continual-cartridges-caches}"
REPO_RESULTS="${REPO_RESULTS:-${HF_USER}/gated-continual-cartridges-results}"

HF_BIN="${HF_BIN:-$DEST/.venv/bin/hf}"; command -v "$HF_BIN" >/dev/null 2>&1 || HF_BIN="hf"
PY="${PY:-$DEST/.venv/bin/python}";     command -v "$PY"     >/dev/null 2>&1 || PY="python3"
DL="$DEST/.hf_download"

log()  { printf '\033[1;34m[prepare]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[prepare][warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[prepare][err]\033[0m %s\n' "$*"; exit 1; }

command -v "$HF_BIN" >/dev/null 2>&1 || die "hf CLI not found (pip install huggingface_hub)."
command -v zstd      >/dev/null 2>&1 || die "zstd not found (apt install zstd)."
[[ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]] || warn "no HF_TOKEN set; only works if the datasets are public."

snapshot() {  # snapshot <repo> <subdir>  -> echoes local snapshot dir
  local repo="$1" sub="$2"
  local out="$DL/$sub"
  mkdir -p "$out"
  log "downloading $repo" >&2
  "$HF_BIN" download "$repo" --repo-type dataset --local-dir "$out" >&2
  echo "$out"
}

verify_manifest() {  # verify_manifest <root> <manifest.tsv>
  local root="$1" man="$2"
  [[ -f "$man" ]] || { warn "no manifest at $man; skipping verify"; return 0; }
  log "verifying $(basename "$man")"
  local bad=0
  tail -n +2 "$man" | while IFS=$'\t' read -r path bytes; do
    [[ -z "$path" ]] && continue
    if [[ ! -f "$root/$path" ]]; then echo "MISSING $path"; bad=1; continue; fi
    local got; got="$(stat -c%s "$root/$path")"
    [[ "$got" == "$bytes" ]] || { echo "SIZE   $path ($got != $bytes)"; bad=1; }
  done
  [[ "$bad" == 0 ]] && log "manifest OK" || warn "manifest reported mismatches (see above)"
}

# ---- data --------------------------------------------------------------------
prepare_data() {
  local snap; snap="$(snapshot "$REPO_DATA" data-repo)"
  log "installing data/ -> $DEST/data"
  mkdir -p "$DEST/data"
  cp -a "$snap/data/." "$DEST/data/"
  verify_manifest "$snap" "$snap/MANIFEST.tsv"
  # extract synth checkpoint shards back into data/<ds>/synth/.../checkpoints/
  if [[ -d "$snap/checkpoint-shards" ]]; then
    if [[ -f "$snap/SHARDS.sha256" ]]; then
      log "checking checkpoint shard sha256"
      ( cd "$snap/checkpoint-shards" && sha256sum -c "$snap/SHARDS.sha256" ) || die "checkpoint shard checksum failed"
    fi
    for shard in "$snap"/checkpoint-shards/*.tar.zst; do
      [[ -e "$shard" ]] || continue
      log "extracting $(basename "$shard")"
      zstd -dc "$shard" | tar -C "$DEST" -xf -   # paths are data/<ds>/.../checkpoints/...
    done
  fi
  # rehydrate symlinks (train/ aliases, qasper eval topic aliases, ...)
  local sl="$DEST/data/SYMLINKS.tsv"
  if [[ -f "$sl" ]]; then
    log "rehydrating symlinks from SYMLINKS.tsv"
    while IFS=$'\t' read -r link target; do
      [[ -z "$link" || -z "$target" ]] && continue
      local abs="$DEST/$link"
      mkdir -p "$(dirname "$abs")"
      ln -sfn "$target" "$abs"
    done < "$sl"
  fi
  log "data ready under $DEST/data"
}

# ---- caches ------------------------------------------------------------------
prepare_caches() {
  local snap; snap="$(snapshot "$REPO_CACHES" caches)"
  if [[ -f "$snap/SHARDS.sha256" ]]; then
    log "checking shard sha256"
    ( cd "$snap/shards" && sha256sum -c "$snap/SHARDS.sha256" ) || die "shard checksum failed"
  fi
  mkdir -p "$DEST/outputs/caches"
  for shard in "$snap"/shards/*.tar.zst; do
    [[ -e "$shard" ]] || continue
    local base; base="$(basename "$shard")"
    case "$base" in
      caches-qasper-*)        log "extract $base -> outputs/caches/"; zstd -dc "$shard" | tar -C "$DEST/outputs/caches" -xf - ;;
      caches-*-5phase-runs*)  log "extract $base -> outputs/";        zstd -dc "$shard" | tar -C "$DEST/outputs"        -xf - ;;
      *)                      warn "unrecognized shard $base; extracting into outputs/"; zstd -dc "$shard" | tar -C "$DEST/outputs" -xf - ;;
    esac
  done
  # normalize absolute symlinks (any machine's /.../outputs/caches/... -> relative)
  log "normalizing symlinks"
  find "$DEST/outputs/caches" -type l | while read -r l; do
    local tgt; tgt="$(readlink "$l")"
    [[ "$tgt" == /* ]] || continue
    case "$tgt" in
      */outputs/caches/*)
        local suffix="${tgt##*/outputs/caches/}"
        local rel; rel="$($PY -c "import os,sys;print(os.path.relpath(sys.argv[1],sys.argv[2]))" \
          "$DEST/outputs/caches/$suffix" "$(dirname "$l")")"
        ln -sfn "$rel" "$l" ;;
    esac
  done
  # rewrite index.json to the local absolute root
  if [[ -f "$snap/index.json" ]]; then
    log "writing outputs/caches/index.json (local root)"
    "$PY" - "$snap/index.json" "$DEST/outputs/caches/index.json" "$DEST/outputs/caches" <<'PY'
import json, os, sys
src, dst, root = sys.argv[1], sys.argv[2], sys.argv[3]
idx = json.load(open(src))
idx["root"] = root
for c in idx.get("caches", []):
    cp = c.get("cache_path", "")
    if cp and not os.path.isabs(cp):
        c["cache_path"] = os.path.join(root, cp)
json.dump(idx, open(dst, "w"), indent=2)
print("index.json ->", dst)
PY
  fi
  log "caches ready under $DEST/outputs/caches (+ *_5phase_runs)"
}

# ---- results -----------------------------------------------------------------
prepare_results() {
  local snap; snap="$(snapshot "$REPO_RESULTS" results-repo)"
  verify_manifest "$snap" "$snap/MANIFEST.tsv"
  mkdir -p "$DEST/outputs"
  for d in evaluations experiments recipes eval_plans figures; do
    [[ -d "$snap/$d" ]] && { log "installing $d"; mkdir -p "$DEST/outputs/$d"; cp -a "$snap/$d/." "$DEST/outputs/$d/"; }
  done
  if [[ -d "$snap/state" ]]; then
    for ds in "$snap"/state/*/; do
      [[ -d "$ds" ]] || continue
      local name; name="$(basename "$ds")"
      log "installing state -> outputs/${name}_5phase_state"
      mkdir -p "$DEST/outputs/${name}_5phase_state"
      cp -a "$ds/." "$DEST/outputs/${name}_5phase_state/"
    done
  fi
  log "results ready under $DEST/outputs"
}

case "$WHICH" in
  data)    prepare_data ;;
  caches)  prepare_caches ;;
  results) prepare_results ;;
  all)     prepare_data; prepare_results; prepare_caches ;;
  *) die "unknown --which: $WHICH (use data|caches|results|all)" ;;
esac

[[ "$KEEP_ARCHIVES" == 1 ]] || { log "cleaning $DL"; rm -rf "$DL"; }
log "done."
