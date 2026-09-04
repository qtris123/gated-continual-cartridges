"""Per-technique loss trajectory across the restored QASPER AM stages.

Read-only. Phase-2 summaries name their evals `qa_forgetting`/`mt_acquisition`
while phase-3+ summaries use bare task names, so both schemas are normalised to
the task key before tabulating.

Reference points from notes/experiments/qasper_am_experiments.md, measured on the
untouched self-distilled Phase-1 cartridge: QA 2.2388, MT 3.7825.
"""

import glob
import json
import os
import re
import statistics
from collections import defaultdict

TASKS = ["qa", "mt", "sa", "asr", "kg"]
PHASE1_REF = {"qa": 2.2388, "mt": 3.7825}
# Which task each stage is *acquiring*; earlier tasks are retention measurements.
STAGE_NEW_TASK = {"p02": "mt", "p03": "sa", "p04": "asr", "p05": "kg"}


def norm_task(name):
    base = re.sub(r"_(forgetting|acquisition|retention)$", "", name.strip().lower())
    return base if base in TASKS else None


def collect():
    out = defaultdict(dict)  # technique -> stage -> {task: loss}
    for f in sorted(glob.glob("outputs/caches/qasper/p0*/*/run/phase2_summary.json")):
        parts = f.split(os.sep)
        stage, technique = parts[3], parts[4]
        try:
            summ = json.load(open(f))
        except Exception:
            continue
        metrics = {}
        for raw, payload in (summ.get("eval_metrics") or {}).items():
            t = norm_task(raw)
            if t and isinstance(payload, dict) and payload.get("loss") is not None:
                metrics[t] = payload["loss"]
        out[technique][stage] = metrics
    return out


def main():
    data = collect()
    stages = ["p02", "p03", "p04", "p05"]

    print("Reference (untouched self-distilled Phase 1): QA %.4f  MT %.4f\n"
          % (PHASE1_REF["qa"], PHASE1_REF["mt"]))

    print("=== QA loss (the Phase-1 task being retained) across stages ===")
    print("technique".ljust(24) + "".join(s.rjust(10) for s in stages) + "   p02->p04 drift")
    drifts = []
    for tech in sorted(data):
        row = tech.ljust(24)
        vals = []
        for s in stages:
            v = data[tech].get(s, {}).get("qa")
            vals.append(v)
            row += (f"{v:10.4f}" if v is not None else "       n/a")
        if vals[0] is not None and vals[2] is not None:
            d = vals[2] - vals[0]
            drifts.append(d)
            row += f"   {d:+.4f}"
        print(row)
    if drifts:
        print(f"\nmean QA drift p02->p04: {statistics.mean(drifts):+.4f} "
              f"(n={len(drifts)}, worse in {sum(d > 0 for d in drifts)}/{len(drifts)})")

    print("\n=== stage means: newly acquired task vs retained QA ===")
    print("stage  n   new-task loss    QA loss   QA vs phase1-ref")
    for s in stages:
        qa = [m["qa"] for t in data for st, m in data[t].items() if st == s and "qa" in m]
        new = STAGE_NEW_TASK[s]
        nv = [m[new] for t in data for st, m in data[t].items() if st == s and new in m]
        if not qa and not nv:
            print(f"{s}    -   (no restored summaries)")
            continue
        nv_s = f"{statistics.mean(nv):.4f}" if nv else "n/a"
        qa_s = f"{statistics.mean(qa):.4f}" if qa else "n/a"
        delta = (f"{statistics.mean(qa) - PHASE1_REF['qa']:+.4f}" if qa else "n/a")
        print(f"{s}  {len(qa):3d}   {new}={nv_s:9s}  {qa_s:9s}  {delta}")

    print("\n=== all tasks, stage means (lower is better) ===")
    print("stage " + "".join(t.upper().rjust(9) for t in TASKS))
    for s in stages:
        row = s.ljust(6)
        for t in TASKS:
            vs = [m[t] for tech in data for st, m in data[tech].items()
                  if st == s and t in m]
            row += (f"{statistics.mean(vs):9.4f}" if vs else "      n/a")
        print(row)


if __name__ == "__main__":
    main()
