#!/usr/bin/env bash
# Standalone QA/MT/SA evals on Phase-1 and the 24 Phase-2 default-dw cartridges.
# Does not write caches. Mapping is cache_last.pt next to that run's config.yaml.
#
# Usage (from repo root):
#   bash examples/qasper/sweeps/eval_p1_p2_sa_refs.sh
#   SMOKE=1 bash examples/qasper/sweeps/eval_p1_p2_sa_refs.sh

set -euo pipefail
cd "$(dirname "$0")/../../../.."
export CARTRIDGES_DIR=$PWD
export CARTRIDGES_OUTPUT_DIR=$PWD/outputs
export PYTHONPATH="$CARTRIDGES_DIR:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

PY=${PY:-.venv/bin/python}
DISCOVER=examples/shared/am/sweeps/eval_p1_p2_sa_refs.py

LOGDIR=$PWD/outputs/sweep_eval_p1_p2_sa_refs_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"
JOBS_JSON=$LOGDIR/jobs.json
QUEUE=$LOGDIR/queue.txt
: > "$QUEUE"

"$PY" "$DISCOVER" --write-jobs "$JOBS_JSON" | tee "$LOGDIR/manifest.txt"

"$PY" - "$JOBS_JSON" "$QUEUE" <<'PY'
import json, sys
from pathlib import Path
jobs = json.load(open(sys.argv[1]))
# P1 first (one cheap-ish baseline), then 24 P2.
Path(sys.argv[2]).write_text("\n".join(j["tag"] for j in jobs) + "\n")
print(f"queue {len(jobs)} tags")
PY

if [ "${SMOKE:-0}" = "1" ]; then
  echo "P1_phase1_selfdistill_qwen512" > "$QUEUE"
  echo "SMOKE: P1 only"
fi

{
  echo "logdir: $LOGDIR"
  echo "jobs:   $JOBS_JSON"
  echo "cells:  $(wc -l < "$QUEUE")"
  echo "evals:  QA + MT + SA via eval_cartridge.py (no write)"
} | tee -a "$LOGDIR/manifest.txt"

pop() {
  (
    flock 200
    line=$(head -1 "$QUEUE" 2>/dev/null || true)
    [ -n "$line" ] && sed -i '1d' "$QUEUE"
    printf '%s' "$line"
  ) 200>"$QUEUE.lock"
}

worker() {
  local gpu=$1 tag rc
  while :; do
    tag=$(pop)
    [ -z "$tag" ] && break
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      "$PY" "$DISCOVER" --run-tag "$tag" --jobs "$JOBS_JSON" --out-dir "$LOGDIR"
    ) > "$LOGDIR/${tag}.log" 2>&1
    rc=$?
    echo "$(date +%H:%M:%S) gpu$gpu $tag rc=$rc" >> "$LOGDIR/progress.txt"
  done
}

echo "queued: $(wc -l < "$QUEUE")"
worker 0 &
worker 1 &
worker 2 &
worker 3 &
wait
echo "ALL DONE $(date +%H:%M:%S)" >> "$LOGDIR/progress.txt"

"$PY" - "$LOGDIR" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
rows = []
for p in sorted(d.glob("*.json")):
    if p.name == "jobs.json":
        continue
    s = json.load(open(p))
    m = s.get("mapping") or {}
    em = s.get("eval_metrics") or {}
    def loss(k):
        v = em.get(k) or {}
        return v.get("loss")
    rows.append({
        "tag": m.get("tag") or p.stem,
        "stage": m.get("stage"),
        "obj": m.get("obj"),
        "key": m.get("key"),
        "beta": m.get("beta"),
        "idf": m.get("idf"),
        "cache": m.get("cache_rel") or s.get("cache_path"),
        "qa": loss("qa_forgetting"),
        "mt": loss("mt_acquisition"),
        "sa": loss("sa_acquisition"),
    })
out = d / "summary.json"
json.dump(rows, open(out, "w"), indent=2)
print(f"summary {len(rows)} rows -> {out}")
for r in rows:
    loc = f"({r['obj']}, {r['key']}, {r['beta']}, {r['idf']})" if r["stage"] == "P2" else "P1"
    print(f"  {r['tag']:40s} {loc:42s} QA={r['qa']} MT={r['mt']} SA={r['sa']}")
PY
echo "logs: $LOGDIR"
