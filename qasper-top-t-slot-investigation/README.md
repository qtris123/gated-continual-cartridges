# qasper-top-t-slot-investigation

Two complementary analyses of the Phase-2 sparse cartridge runs (Llama-3.2-3B,
Qasper):

1. **`analyze_slot_selection.py`** — *what* the top-t mask looks like.
   Binary (sub_component × slot) heatmaps at mid-training plus 3-batch
   small multiples, and a step-to-step Jaccard stability number per run.

2. **`analyze_coverage_vs_forgetting.py`** — *how much of the cache ever
   gets written* and how that relates to Phase-1 forgetting (QA perplexity).

Both scripts only read from `outputs/<run>/<uuid>/sparse_slot_log.pt`
and (for the coverage script) from
`qasper-forgetting-investigation/plots/llama_granularity_x_sparsity_summary.csv`.

## Files produced

```
qasper-top-t-slot-investigation/
├── analyze_slot_selection.py
├── analyze_coverage_vs_forgetting.py
├── stability_table.csv / .md            # median consecutive-step Jaccard
├── run_metadata.json                    # per-run metadata
├── coverage_vs_forgetting.csv           # coverage + forgetting ppl per (gran, t)
└── figures/
    ├── main_snapshot_grid.png           # 3 gran × 4 t mid-training snapshot
    ├── per_head_full_grid.png           # full 28×8 per_head at mid-training
    ├── coverage_growth.png              # union fraction vs snapshot index
    ├── coverage_bars.png                # final cumulative coverage bars
    ├── coverage_vs_ppl.png              # scatter: coverage → forgetting ppl
    └── appendix/
        ├── global_3batch.png            # 25/50/75% snapshots
        ├── per_layer_3batch.png
        └── per_head_3batch.png
```

## Run

```bash
cd /localhome/local-triv/gated-continual-cartridges
conda activate cartridges
python qasper-top-t-slot-investigation/analyze_slot_selection.py
python qasper-top-t-slot-investigation/analyze_coverage_vs_forgetting.py
```

## Headline finding

The fraction of the cache that ever lands in the top-t during Phase-2
training (cumulative cache coverage) predicts Phase-1 QA forgetting:

> Pearson r = 0.910, Spearman ρ = 0.937 across all 12 (granularity, t).

Finer granularity → more churn step-to-step → larger union of updated
(layer, head, slot) triples → more parameters drift from the Phase-1
solution → higher Phase-1 QA perplexity.
