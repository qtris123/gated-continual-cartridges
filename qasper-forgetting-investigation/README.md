# qasper-forgetting-investigation

Ad-hoc analysis of the qasper Phase 1 / Phase 2 sparse cartridge eval
results. This folder lives inside the `gated-continual-cartridges` repo
but is intentionally separate from the cartridge codebase — it only
reads from the repo's `outputs/` directory and writes its own artifacts
under `plots/` here.

## Layout

- `plot_forgetting_vs_learning.py` — sweeps over the bsize-32 value-only
  results (top-k ∈ {64, 128, 256, 512}, per-layer and per-head) and plots
  forgetting (Phase 1 QA perplexity) vs. learning (Phase 2 MT perplexity),
  with the Phase-1-only baselines as dashed horizontal lines.
- `plots/` — generated figures + CSV summary.

## Run

```bash
cd /localhome/local-triv/gated-continual-cartridges
source .venv/bin/activate
python qasper-forgetting-investigation/plot_forgetting_vs_learning.py
```

The script resolves the cartridges repo as its parent directory, so it
works from wherever the repo is checked out as long as this folder stays
at the repo root.
