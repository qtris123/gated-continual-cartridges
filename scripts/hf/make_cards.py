#!/usr/bin/env python3
"""Generate README.md dataset cards for the three staging trees.

Usage: make_cards.py <stage_root>
Writes README.md into <stage_root>/{data-repo,caches,results-repo} when present.
"""
import json
import os
import sys

USER = os.environ.get("HF_USER", "qtris123")
DATA_REPO = f"{USER}/gated-continual-cartridges-data"
CACHES_REPO = f"{USER}/gated-continual-cartridges-caches"
RESULTS_REPO = f"{USER}/gated-continual-cartridges-results"


def _du(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if total < 1024 or unit == "TB":
            return f"{total:.1f} {unit}"
        total /= 1024


def _write(path, text):
    with open(path, "w") as fh:
        fh.write(text)
    print("wrote", path)


def card_data(root):
    ds = os.path.join(root, "data")
    datasets = sorted(
        d for d in os.listdir(ds)
        if os.path.isdir(os.path.join(ds, d))
    ) if os.path.isdir(ds) else []
    _write(os.path.join(root, "README.md"), f"""---
license: mit
pretty_name: Gated Continual Cartridges - Data
tags: [continual-learning, kv-cache, self-study, synthetic-data]
---

# Gated Continual Cartridges — Data ({_du(root)})

Inputs for the gated-continual-cartridges experiments: per-benchmark **phase
corpora**, **evaluation sets**, and **self-study synthetic training data**.

## Layout

```
data/<dataset>/phases/      phase<k>.txt, phase<k>_eval.parquet, manifest.json
data/<dataset>/synth/p0<k>/<technique>/  config.yaml, source.json, artifact/dataset.parquet
data/<dataset>/eval/        benchmark eval sets (+ reference baselines)
data/<dataset>/{{raw,init_text}}/  raw dumps
data/CORRUPTION_AUDIT.json  known dirty synth phases (vLLM NaN-logprob incident)
data/SYMLINKS.tsv           train/ + eval aliases, recreated by prepare_artifacts.sh
```

Datasets included: {", ".join(datasets) or "(none)"}.

## Setup

```bash
scripts/prepare_artifacts.sh --which data --dest /path/to/gated-continual-cartridges
```

This downloads the tree, verifies `MANIFEST.tsv`, and rehydrates the symlinks in
`SYMLINKS.tsv` (so `data/<ds>/train/*` point back at the synth artifacts).

> **Note:** some `dataset_clean.parquet` files referenced by `CORRUPTION_AUDIT.json`
> are regenerated locally rather than shipped; see the audit for clean-row counts.
""")


def card_caches(root):
    shards = []
    sd = os.path.join(root, "shards")
    if os.path.isdir(sd):
        shards = sorted(os.listdir(sd))
    n_caches = "?"
    idx = os.path.join(root, "index.json")
    if os.path.isfile(idx):
        try:
            n_caches = json.load(open(idx)).get("n_caches", "?")
        except Exception:
            pass
    shard_lines = "\n".join(f"- `shards/{s}`" for s in shards) or "- (none)"
    _write(os.path.join(root, "README.md"), f"""---
license: mit
pretty_name: Gated Continual Cartridges - Caches
tags: [continual-learning, kv-cache, compaction]
---

# Gated Continual Cartridges — Compacted KV Caches ({_du(root)})

Compacted key/value caches ("cartridges") produced by the compaction pipeline,
shipped as per-stage / per-benchmark **zstd tar shards** to keep the file count
small and downloads resumable.

## Contents

{shard_lines}

- `index.json` — machine-independent cache index ({n_caches} qasper caches),
  paths relative to the caches root.
- `SHARDS.sha256`, `MANIFEST.tsv` — integrity + sizes.

`caches-qasper-p0X.tar.zst` extract to `outputs/caches/qasper/<stage>/<technique>/`.
`caches-<ds>-5phase-runs.tar.zst` extract to `outputs/<ds>_5phase_runs/`; the final
compacted cache for each (method, phase) is the `cache-step<N>.pt` referenced by
`results/state/<ds>/<method>/p<k>.json` in the `-results` dataset.

## Setup

```bash
scripts/prepare_artifacts.sh --which caches --dest /path/to/gated-continual-cartridges
```

Extraction normalizes any absolute symlinks to relative and rewrites
`outputs/caches/index.json` `root` to the local absolute path.
""")


def card_results(root):
    _write(os.path.join(root, "README.md"), f"""---
license: mit
pretty_name: Gated Continual Cartridges - Results
tags: [continual-learning, evaluation, figures]
---

# Gated Continual Cartridges — Results, Plots & Tables ({_du(root)})

Everything needed to read the story without re-running anything: evaluation
outputs, the 5-phase continual-compaction metrics, sweep experiments, and all
figures/tables.

## Layout

```
evaluations/<dataset>/<method>/{{teacher-forced-logppl-v1,accuracy-*,generations-*}}/
state/<dataset>/<method>/p<k>.json   per-phase loss/perplexity (5-stage continual)
experiments/soft_locality/           FINDINGS.md, report_figs/*.png, tables/*.csv|json
experiments/techqa_slots_per_doc/    slots-per-doc ablation
recipes/*.yaml                       compaction recipes (soft_locality, top-t, rope, fullkv)
eval_plans/*.json                    accuracy eval plans
figures/                             all *.png flattened for quick browsing
```

## Setup

```bash
scripts/prepare_artifacts.sh --which results --dest /path/to/gated-continual-cartridges
```
""")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    if os.path.isdir(os.path.join(root, "data-repo")):
        card_data(os.path.join(root, "data-repo"))
    if os.path.isdir(os.path.join(root, "caches")):
        card_caches(os.path.join(root, "caches"))
    if os.path.isdir(os.path.join(root, "results-repo")):
        card_results(os.path.join(root, "results-repo"))


if __name__ == "__main__":
    main()
