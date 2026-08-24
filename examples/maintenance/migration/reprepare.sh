#!/usr/bin/env bash
#
# reprepare.sh - restore data/ and outputs/caches/ from Hugging Face.
#
# This is the ONE-RUN restore for a fresh machine. It does nothing else:
# no code changes, no environment setup, no regeneration. It only
# re-materializes the two large trees that are kept off git:
#
#   data/            <- qtris123/gated-continual-cartridges-data
#   outputs/caches/  <- qtris123/gated-continual-cartridges-caches
#
# Usage (from anywhere inside a fresh clone of this repo):
#   examples/maintenance/migration/reprepare.sh
#
# Requirements: hf CLI logged in (`hf auth login`), tar, zstd, sha256sum.
#
set -euo pipefail

DATA_REPO="qtris123/gated-continual-cartridges-data"
CACHES_REPO="qtris123/gated-continual-cartridges-caches"
# Absolute path the caches were built on; rewritten to REPO_ROOT on restore.
OLD_ROOT="/localhome/local-triv/trivo-explore-research-work"

# Repo root = three levels up from this script (examples/maintenance/migration/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DL="$REPO_ROOT/.migration-download"

log() { echo "[reprepare $(date '+%F %T')] $*"; }

command -v hf >/dev/null || { echo "hf CLI not found. pip install huggingface_hub[cli]"; exit 1; }
hf auth whoami >/dev/null 2>&1 || { echo "Not logged in. Run: hf auth login"; exit 1; }

cd "$REPO_ROOT"
mkdir -p "$DL"

# --- 1. data/ ---------------------------------------------------------------
log "downloading data snapshot"
hf download "$DATA_REPO" --repo-type dataset --local-dir "$DL/data-repo"
log "extracting data/"
tar --extract --zstd --file "$DL/data-repo/data.tar.zst" -C "$REPO_ROOT"
if [ -f "$DL/data-repo/data.MANIFEST.txt" ]; then
  ( cd "$DL/data-repo" && sha256sum -c data.MANIFEST.txt ) && log "data checksum OK" || log "WARN data checksum mismatch"
fi

# --- 2. outputs/caches/ -----------------------------------------------------
log "downloading caches shards"
hf download "$CACHES_REPO" --repo-type dataset --local-dir "$DL/caches-repo"
mkdir -p "$REPO_ROOT/outputs/caches"
log "extracting cache shards"
for t in "$DL/caches-repo"/caches-*.tar; do
  [ -e "$t" ] || continue
  log "  extract $(basename "$t")"
  tar --extract --sparse --file "$t" -C "$REPO_ROOT/outputs/caches"
done
# meta files (index.json, README.md) live in caches-meta/
if [ -d "$DL/caches-repo/caches-meta" ]; then
  cp -f "$DL/caches-repo/caches-meta/"* "$REPO_ROOT/outputs/caches/" 2>/dev/null || true
fi
if [ -f "$DL/caches-repo/caches.MANIFEST.txt" ]; then
  ( cd "$DL/caches-repo" && sha256sum -c caches.MANIFEST.txt ) && log "caches checksum OK" || log "WARN caches checksum mismatch"
fi

# --- 3. rewrite absolute paths to this machine ------------------------------
if [ "$OLD_ROOT" != "$REPO_ROOT" ]; then
  log "rewriting absolute paths $OLD_ROOT -> $REPO_ROOT"
  # index.json + every source.json under caches
  find "$REPO_ROOT/outputs/caches" -name 'index.json' -o -name 'source.json' | while read -r f; do
    sed -i "s#$OLD_ROOT#$REPO_ROOT#g" "$f"
  done
  # absolute symlinks that point back under the old root
  find "$REPO_ROOT/outputs/caches" -type l -lname "$OLD_ROOT/*" | while read -r link; do
    target="$(readlink "$link")"
    newtarget="${target/$OLD_ROOT/$REPO_ROOT}"
    ln -sfn "$newtarget" "$link"
  done
else
  log "paths already rooted at $REPO_ROOT, no rewrite needed"
fi

# --- 4. verify --------------------------------------------------------------
log "verifying"
ncaches=$(grep -c '"cache_path"' "$REPO_ROOT/outputs/caches/index.json" 2>/dev/null || echo 0)
log "  index.json lists $ncaches caches"
broken=$(find "$REPO_ROOT/outputs/caches" -xtype l 2>/dev/null | wc -l)
log "  broken symlinks in caches: $broken"
broken_data=$(find "$REPO_ROOT/data" -xtype l 2>/dev/null | wc -l)
log "  broken symlinks in data:   $broken_data"

log "cleaning download cache ($DL)"
rm -rf "$DL"
log "DONE. data/ and outputs/caches/ restored."
