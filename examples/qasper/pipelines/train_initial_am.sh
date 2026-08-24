#!/usr/bin/env bash
# Deprecated. KVFromText + self-match refine is not a Phase-1 procedure.
# Use classic AM compaction instead.
set -euo pipefail

echo "train_initial_am.sh is deprecated." >&2
echo "Phase 1 AM is classic compaction, not KVFromText + refine." >&2
echo "Run: examples/qasper/pipelines/train_initial_am_compaction.sh" >&2
exit 2
