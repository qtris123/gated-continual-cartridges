#!/usr/bin/env python
"""Add the mandated `diagnostic` + `DIAG-CONTROLCURVE` tags to every wandb run this
job created.

`examples/qasper2/train/eval_forgetting.py` hard-codes `tags=["eval","forgetting",RUN_NAME]`
(L115) and exposes no tag knob, and this worker is under a no-source-edit rule, so the
tags are applied post-hoc through the public API -- exactly what DIAG-KEYCURVE did for
its 47 runs.
"""
from __future__ import annotations

import json
import sys

import wandb

RESDIR = "/localhome/local-triv/gated-continual-cartridges_explore/research_loop/results/DIAG-CONTROLCURVE"
ENTITY_PROJECT = "vqtri-purdue-university/SEACrowd"
EXTRA_TAGS = ["diagnostic", "DIAG-CONTROLCURVE"]

runs = json.load(open(f"{RESDIR}/wandb_runs.json"))
api = wandb.Api()
out = {}
for label, url in runs.items():
    if not url:
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
    except Exception as e:
        out[label] = f"ERROR {e}"
        print(f"[tag] {label} {rid} ERROR {e}", file=sys.stderr, flush=True)

json.dump(out, open(f"{RESDIR}/wandb_runs_tagged.json", "w"), indent=1)
n_ok = sum(1 for v in out.values() if isinstance(v, dict))
print(f"[done] tagged {n_ok}/{len(out)} runs")
