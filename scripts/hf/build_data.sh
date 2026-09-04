#!/usr/bin/env bash
# Build the staging tree for the -data dataset.
#
# Produces $STAGE/data-repo/ containing:
#   data/<dataset>/{phases,synth,eval,raw,init_text,...}   (browsable, real files)
#   data/SYMLINKS.tsv     (link<TAB>target; train/ + eval aliases rehydrated by prepare)
#   checkpoint-shards/synth-checkpoints-<ds>.tar.zst  (self-study batch_*.parquet)
#   MANIFEST.tsv          (path<TAB>bytes for every browsable regular file)
#   SHARDS.sha256         (integrity for the checkpoint shards)
#   README.md             (dataset card; data/README.md is preserved in-tree)
#
# Symlinks (train/ aliases into synth artifacts, qasper eval topic aliases) are
# stripped from the upload and recorded in SYMLINKS.tsv to avoid duplicating
# multi-GB parquet content; prepare_artifacts.sh recreates them.
#
# The synth checkpoints/ dirs (thousands of tiny batch_*.parquet generation
# batches; the merged result is artifact/dataset.parquet) are packed into
# per-dataset tar shards so the browsable tree stays small and uploads are fast;
# prepare_artifacts.sh extracts them back in place.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

DST="$STAGE/data-repo"
rm -rf "$DST"
mkdir -p "$DST"

log "hardlink-copying data/ -> staging (no extra disk for regular files)"
cp -al "$DATA" "$DST/data"

log "pruning caches/junk"
find "$DST/data" -type d -name '.ipynb_checkpoints' -prune -exec rm -rf {} + 2>/dev/null || true
find "$DST/data" -type d -name '__pycache__'        -prune -exec rm -rf {} + 2>/dev/null || true

log "packing synth checkpoints/ into per-dataset shards (many tiny files -> tarballs)"
mkdir -p "$DST/checkpoint-shards"
for ds in "$DST"/data/*/; do
  ds="$(basename "$ds")"
  [[ -d "$DST/data/$ds" ]] || continue
  mapfile -t ckpts < <(cd "$DST" && find "data/$ds" -type d -name checkpoints | sort)
  [[ ${#ckpts[@]} -gt 0 ]] || continue
  log "  $ds: ${#ckpts[@]} checkpoint dir(s)"
  tar -C "$DST" -cf - "${ckpts[@]}" \
    | zstd -q "-${ZSTD_LEVEL}" "-T${ZSTD_THREADS}" -o "$DST/checkpoint-shards/synth-checkpoints-$ds.tar.zst"
  for c in "${ckpts[@]}"; do rm -rf "$DST/$c"; done
done
if compgen -G "$DST/checkpoint-shards/*.tar.zst" > /dev/null; then
  ( cd "$DST/checkpoint-shards" && sha256sum ./*.tar.zst > "$DST/SHARDS.sha256" )
fi

log "recording + stripping symlinks (rehydrated by prepare_artifacts.sh)"
: > "$DST/data/SYMLINKS.tsv"
# Store link + target both relative to the data/ root so they are portable.
find "$DST/data" -type l | sort | while read -r link; do
  rel_link="${link#"$DST"/}"                        # e.g. data/qasper/train/foo.parquet
  tgt="$(readlink "$link")"
  printf '%s\t%s\n' "$rel_link" "$tgt" >> "$DST/data/SYMLINKS.tsv"
  rm "$link"
done
log "recorded $(($(wc -l < "$DST/data/SYMLINKS.tsv"))) symlinks"

log "building MANIFEST.tsv"
{
  printf 'path\tbytes\n'
  ( cd "$DST" && find data -type f ! -name SYMLINKS.tsv -printf '%p\t%s\n' | sort )
} > "$DST/MANIFEST.tsv"

log "data staging built at: $DST"
du -sh "$DST" 2>/dev/null || true
