#!/usr/bin/env bash
# Delete + recreate the HuggingFace datasets and upload the staged trees.
#
# Usage:
#   scripts/hf/upload_all.sh [data|caches|results|all]   (default: all)
#
# Requires: HF_TOKEN in the environment. Staging trees must already be built by
# build_data.sh / build_caches.sh / build_results.sh.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
require_token

WHICH="${1:-all}"

recreate() {  # recreate <repo_id>
  local repo="$1"
  log "delete + create dataset repo: $repo"
  "$PY" - "$repo" <<'PY'
import sys
from huggingface_hub import HfApi
api = HfApi()
repo = sys.argv[1]
api.delete_repo(repo_id=repo, repo_type="dataset", missing_ok=True)
api.create_repo(repo_id=repo, repo_type="dataset", private=False, exist_ok=True)
print("recreated", repo)
PY
}

upload() {  # upload <repo_id> <local_dir>
  local repo="$1" dir="$2"
  [[ -d "$dir" ]] || die "staging dir missing: $dir (run the matching build_*.sh first)"
  log "uploading $dir -> $repo"
  "$HF_BIN" upload-large-folder "$repo" "$dir" --repo-type dataset --num-workers 8
}

do_data()    { recreate "$REPO_DATA";    upload "$REPO_DATA"    "$STAGE/data-repo"; }
do_caches()  { recreate "$REPO_CACHES";  upload "$REPO_CACHES"  "$STAGE/caches"; }
do_results() { recreate "$REPO_RESULTS"; upload "$REPO_RESULTS" "$STAGE/results-repo"; }

case "$WHICH" in
  data)    do_data ;;
  caches)  do_caches ;;
  results) do_results ;;
  all)     do_data; do_results; do_caches ;;   # caches (biggest) last
  *) die "unknown target: $WHICH (use data|caches|results|all)" ;;
esac
log "upload(s) complete."
