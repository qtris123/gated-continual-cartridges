#!/usr/bin/env python
"""Add the mandated `diagnostic` + `VERIFY-OPTIMA` tags to every wandb run this job made.

`eval_forgetting.py` hard-codes tags=["eval","forgetting",RUN_NAME] and exposes no tag knob,
and this worker is under a no-source-edit rule, so the tags go on post-hoc through the public
API -- the same route DIAG-KEYCURVE and DIAG-CONTROLCURVE used.
"""
from __future__ import annotations

import json
import sys

import wandb

RES = "/localhome/local-triv/gated-continual-cartridges_explore/research_loop/results/VERIFY-OPTIMA"
ENTITY_PROJECT = "vqtri-purdue-university/SEACrowd"
EXTRA_TAGS = ["diagnostic", "VERIFY-OPTIMA"]

runs = json.load(open(f"{RES}/wandb_runs.json"))["urls"]
api = wandb.Api()
out = {}
for label, url in runs.items():
    if not url or url == "MISSING":
        out[label] = "MISSING_URL"
        continue
    rid = url.rstrip("/").split("/")[-1]
    try:
        r = api.run(f"{ENTITY_PROJECT}/{rid}")
        before = list(r.tags)
        for t in EXTRA_TAGS:
            if t not in r.tags:
                r.tags = list(r.tags) + [t]
        r.update()
        out[label] = {"id": rid, "name": r.name, "group": r.group,
                      "tags_before": before, "tags_after": list(r.tags)}
        print(f"[tag] {label} {rid} name={r.name} group={r.group} tags={list(r.tags)}", flush=True)
    except Exception as e:  # noqa: BLE001
        out[label] = f"ERROR {e}"
        print(f"[tag] {label} {rid} ERROR {e}", file=sys.stderr, flush=True)

json.dump(out, open(f"{RES}/wandb_runs_tagged.json", "w"), indent=1)
n_ok = sum(1 for v in out.values() if isinstance(v, dict))
print(f"[done] tagged {n_ok}/{len(out)} runs")
