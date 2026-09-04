#!/usr/bin/env bash
# Build the staging tree for the -results dataset.
#
# Produces $STAGE/results-repo/ containing:
#   evaluations/            (teacher-forced-logppl, accuracy, generations per ds/method)
#   experiments/            (soft_locality: plots/tables/FINDINGS.md; techqa_slots_per_doc; ...)
#   state/<dataset>/<method>/p<k>.json   (5-phase continual perplexity metrics)
#   recipes/                (compaction recipe yamls)
#   eval_plans/             (accuracy eval plans)
#   figures/                (all *.png flattened, dataset-prefixed, for quick browsing)
#   MANIFEST.tsv, README.md
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

DST="$STAGE/results-repo"
rm -rf "$DST"
mkdir -p "$DST/figures"

copy_dir() {  # copy_dir <src> <dst-subpath>
  local src="$1" sub="$2"
  [[ -e "$src" ]] || { warn "missing, skipping: $src"; return 0; }
  log "copying $sub"
  mkdir -p "$DST/$sub"
  cp -aL "$src/." "$DST/$sub/" 2>/dev/null || cp -a "$src/." "$DST/$sub/"
}

copy_dir "$OUT/evaluations" "evaluations"
copy_dir "$OUT/experiments" "experiments"
copy_dir "$OUT/recipes"     "recipes"
copy_dir "$OUT/eval_plans"  "eval_plans"

log "collecting 5-phase state (perplexity metrics)"
for ds in "${RUN_CACHE_DATASETS[@]}" qasper; do
  src="$OUT/${ds}_5phase_state"
  [[ -L "$src" ]] && src="$(readlink -f "$src")"
  [[ -d "$src" ]] || continue
  mkdir -p "$DST/state/$ds"
  cp -aL "$src/." "$DST/state/$ds/" 2>/dev/null || true
done

log "flattening figures/*.png for quick browsing"
find "$DST" -type f -name '*.png' | while read -r png; do
  rel="${png#"$DST"/}"
  flat="$(echo "$rel" | tr '/' '_')"
  cp -a "$png" "$DST/figures/$flat" 2>/dev/null || true
done

log "building MANIFEST.tsv"
{
  printf 'path\tbytes\n'
  ( cd "$DST" && find . -type f -printf '%p\t%s\n' | sed 's#^\./##' | sort )
} > "$DST/MANIFEST.tsv"

log "results staging built at: $DST"
du -sh "$DST" 2>/dev/null || true
