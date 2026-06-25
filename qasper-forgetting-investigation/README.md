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
- `plot_granularity_x_sparsity.py` — granularity × sparsity sweeps for Llama
  and Qwen (top-k ∈ {32, 64, 128, 256}, granularity ∈ {global, per-head,
  per-layer}). One figure per model with two subplots (forgetting /
  acquisition), three granularity lines per subplot, and one shared dashed
  baseline line per subplot (averaged across granularities since the
  baseline cartridge is shared in spirit; per-gran values are kept in the
  CSV). Models are configured at the top of the file:
    - Llama: `outputs/qasper_forgetting_eval_llama_granularity-x-sparsity_3273481/`
    - Qwen:  `outputs/qasper_forgetting_eval_qwen_granularity-x-sparsity_3315625_3344506/`
      (a symlink-merged view of the per-head+global batch `_3315625` and the
      per-layer batch `_3344506`).
- `plots/` — generated figures + CSV summary.

## Run

```bash
cd /localhome/local-triv/gated-continual-cartridges
source .venv/bin/activate    # or: conda activate cartridges
python qasper-forgetting-investigation/plot_forgetting_vs_learning.py
python qasper-forgetting-investigation/plot_granularity_x_sparsity.py
```

The script resolves the cartridges repo as its parent directory, so it
works from wherever the repo is checked out as long as this folder stays
at the repo root.
