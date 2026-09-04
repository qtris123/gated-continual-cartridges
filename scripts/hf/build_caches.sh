#!/usr/bin/env bash
# Build the staging tree for the -caches dataset.
#
# Produces $STAGE/caches/ containing:
#   shards/caches-qasper-p0X.tar.zst        (qasper compacted caches, per stage)
#   shards/caches-<ds>-5phase-runs.tar.zst  (finqa/quality/techqa 5-phase run trees)
#   index.json                              (machine-independent cache index)
#   SHARDS.sha256                           (integrity for every shard)
#   MANIFEST.tsv                            (shard -> uncompressed size / file count)
#   README.md                               (dataset card)
#
# Absolute symlinks inside the caches are NOT modified here; they are packed as
# stored and normalized to relative on extraction by prepare_artifacts.sh.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

DST="$STAGE/caches"
SHARDS="$DST/shards"
mkdir -p "$SHARDS"

pack() {   # pack <tar_root> <relative_path> <shard_name>
  local root="$1" rel="$2" name="$3"
  local out="$SHARDS/$name"
  if [[ -f "$out" ]]; then
    log "shard exists, skipping: $name"
    return 0
  fi
  [[ -e "$root/$rel" ]] || { warn "missing, skipping: $root/$rel"; return 0; }
  log "packing $rel -> $name"
  # --hard-dereference keeps things simple; symlinks are stored as symlinks.
  tar -C "$root" -cf - "$rel" \
    | zstd -q "-${ZSTD_LEVEL}" "-T${ZSTD_THREADS}" -o "$out.tmp"
  mv "$out.tmp" "$out"
}

log "=== qasper compacted caches (outputs/caches/qasper) ==="
for st in "${QASPER_CACHE_STAGES[@]}"; do
  [[ -d "$OUT/caches/qasper/$st" ]] && pack "$OUT/caches" "qasper/$st" "caches-qasper-$st.tar.zst"
done

log "=== 5-phase run caches (finqa/quality/techqa) ==="
for ds in "${RUN_CACHE_DATASETS[@]}"; do
  [[ -d "$OUT/${ds}_5phase_runs" ]] && pack "$OUT" "${ds}_5phase_runs" "caches-${ds}-5phase-runs.tar.zst"
done

log "=== machine-independent index.json ==="
if [[ -f "$OUT/caches/index.json" ]]; then
  "$PY" "$HF_DIR/relativize_index.py" "$OUT/caches/index.json" "$DST/index.json"
else
  warn "no outputs/caches/index.json; skipping index"
fi

log "=== manifest + checksums ==="
: > "$DST/MANIFEST.tsv"
printf 'shard\tbytes\n' >> "$DST/MANIFEST.tsv"
for f in "$SHARDS"/*.tar.zst; do
  [[ -e "$f" ]] || continue
  printf '%s\t%s\n' "$(basename "$f")" "$(stat -c%s "$f")" >> "$DST/MANIFEST.tsv"
done
( cd "$SHARDS" && sha256sum ./*.tar.zst > "$DST/SHARDS.sha256" ) 2>/dev/null || warn "no shards to checksum"

log "caches staging built at: $DST"
du -sh "$DST" 2>/dev/null || true
