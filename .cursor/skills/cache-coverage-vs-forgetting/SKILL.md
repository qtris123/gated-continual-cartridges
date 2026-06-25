---
name: cache-coverage-vs-forgetting
description: >-
  Diagnostic for explaining Phase-1 retention vs Phase-2 acquisition trade-offs in
  sparse-cartridge finetuning, framed through cumulative cache coverage. Use when
  interpreting granularity × sparsity sweeps, deciding which (granularity, top_t)
  point to ship, designing follow-up experiments on the forgetting axis, or
  analysing logs from continual_sparse runs (sparse_slot_log.pt / bg_stats.pt /
  tfidf_ranking_log.pt). Companion to sparse-finetuning-design (architecture facts)
  and sparse-finetuning-hypotheses (working theories).
disable-model-invocation: true
---

# Cache Coverage vs Forgetting

This skill captures the *coverage* lens for sparse cartridge finetuning: under
freeze-style masking, a cache parameter only drifts if it ever lands in the
top-t. The fraction of `(layer, head, slot)` triples that *ever* drift during
Phase 2 is the single strongest predictor of Phase-1 forgetting we have
measured.

If you only remember one thing: **finer granularity and larger t both inflate
the same quantity — the cumulative footprint of updated positions — and that
footprint predicts Phase-1 QA perplexity with Pearson r ≈ 0.91 across our
3 × 4 Llama-3.2-3B Qasper sweep.**

---

## Headline empirical result (Llama-3.2-3B, Qasper)

Across the 12-run granularity × sparsity sweep (3 granularities × 4 budgets):

| granularity | t=32 | t=64 | t=128 | t=256 |
|---|---|---|---|---|
| **coverage**, global / per_layer / per_head | 0.09 / 0.14 / 0.17 | 0.17 / 0.25 / 0.28 | 0.32 / 0.41 / 0.44 | 0.58 / 0.66 / 0.66 |
| **QA ppl**, global / per_layer / per_head | 13.46 / 15.26 / 16.26 | 14.30 / 16.33 / 17.56 | 16.33 / 16.39 / 18.01 | 19.41 / 18.89 / 19.58 |

* Pearson r = **0.910**, Spearman ρ = **0.937** between cumulative coverage
  and forgetting ppl.
* Phase-1 baseline (no Phase-2 training): 6.12 ppl. Anything above that is
  forgetting.
* Phase-2 baseline (no gating): 33.92 ppl. All gated runs land in 6.2–7.0
  (acquisition is essentially insensitive to the gate).

Raw numbers live in `qasper-top-t-slot-investigation/coverage_vs_forgetting.csv`.

---

## Why coverage is the right object

Phase-2 sparse runs use `momentum_masking="freeze"`: non-top-t positions have
both their parameter values *and* their optimizer momentum restored after
every step. So a cache parameter `(l, h, n)` drifts **iff** it lands in the
top-t at least once during training. Forgetting is therefore determined by:

```
Footprint = union over all 540 optimizer steps of {(l,h,n) in top-t}
coverage  = |Footprint| / (L * H * N)
```

This is the *cumulative* footprint, not the per-step one. Two axes inflate it:

1. **Larger t**: the per-step floor `t / N` grows directly. At t=256 the
   per-step floor alone is 0.50 (=256/511 trainable tokens).
2. **Finer granularity**: replaces one shared ranking with `L` then `L · H`
   independent rankings. Independent rankings disagree across batches → top-t
   *rotates* across steps → union over time blows up faster than the per-step
   floor.

For Llama-3.2-3B-Instruct: `L=28, H=8` (GQA), `N=511` trainable cache tokens
(set by `num_tokens=512` with one frozen prefix).

---

## Two complementary measurements

Both come from `sparse_slot_log.pt`, which `cartridges/train.py` writes every
30 optimizer steps. Each entry stores `{step, granularity, top_positions,
access_scores}`. With 500-step runs and step-30 logging this gives 19
snapshots per run.

### Measurement A — intra-stage-2 stability (step-to-step Jaccard)

For each pair of consecutive logged snapshots, compute per-component Jaccard
and take the median across pairs. This is a churn diagnostic.

Llama-3.2-3B Qasper:

| granularity | t=32 | t=64 | t=128 | t=256 |
|---|---|---|---|---|
| global    | 1.00 | 0.97 | 0.98 | 0.96 |
| per_layer | 0.85 | 0.85 | 0.86 | 0.89 |
| per_head  | 0.74 | 0.75 | 0.79 | 0.86 |

Coarser granularity → higher Jaccard → less rotation. This is the
*upstream* cause of the coverage gap.

### Measurement B — cumulative coverage (union over snapshots)

For each logged mask, expand to a `(L, H, N)` boolean according to granularity
(global broadcasts to all `(l, h)`; per_layer broadcasts to all `h`; per_head
is already a 3-D mask), accumulate a logical OR, and divide by `L · H · N`.

The 19-snapshot union is a **lower bound** on the true 540-step union; the
gap is larger for high-churn runs (per_head), so reported coverage understates
the per_head disadvantage. The conclusion only strengthens.

Both measurements are implemented in
`qasper-top-t-slot-investigation/analyze_coverage_vs_forgetting.py`.

---

## Three observations from the slot heatmaps

The mid-training (~step 270) selection heatmaps in
`qasper-top-t-slot-investigation/figures/main_snapshot_grid.png` reveal three
patterns that the coverage number alone can't show:

1. **Stable across batches.** Within a run, snapshots at ~25 %, 50 %, 75 %
   of training look qualitatively identical (see `figures/appendix/*.png`).
   The snapshot at any single step is representative.
2. **Cluster + neighbour structure.** The set of selected slots forms a few
   tall vertical bands. Increasing t doesn't move the bands — it widens
   them, sucking in the slots immediately surrounding each cluster. So
   *primary* clusters are reused across budgets; only the *secondary*
   neighbour slots are budget-dependent.
3. **Early-prefix slots are never selected.** Slots ~0–50 stay white across
   every granularity / budget / batch. These are positions near the frozen
   prefix; the data never attends to them. They contribute trivially to the
   union and are essentially dead weight in the cache.

Observation (2) is the visual cause of the t-axis forgetting curve: larger
t = more neighbour slots in the footprint = more cache parameters drift.

---

## Common misframings to correct

### "Less step-to-step overlap means less forgetting (less overwriting)"

**Wrong sign.** Lower step-to-step Jaccard means the top-t set *rotates*
across batches, which makes the *union* over training **larger**, not smaller.
More positions get touched → more parameters drift → more forgetting.

The correct intuition is the opposite: high step-to-step Jaccard (low churn)
is *protective*, because the same small set is hit every step, so only that
set drifts. Global runs have Jaccard ≈ 1.0 and therefore the smallest
footprint.

### "Overlap with Phase-1 important slots is what matters"

This is closer to the truth but is **not what the consecutive-snapshot
Jaccard measures**. The consecutive-snapshot Jaccard is intra-stage-2.
What predicts forgetting via the coverage chain is:

```
intra-stage-2 churn  →  cumulative coverage  →  forgetting
```

If you want to test the "overlap with Phase-1 important slots" hypothesis
directly, the footprint × `bg_stats` top-k intersection is the right object
(see open questions below).

### "TF-IDF picks Phase-1-safe positions, so granularity shouldn't matter"

TF-IDF *aspires* to pick Phase-1-safe positions, but with finer granularity
each `(l, h)` ranking is computed from a smaller statistical sample (its own
attention slice), so the rankings disagree across heads / layers / batches.
The disagreement, not a worse-per-component choice, is what drives the
coverage gap.

---

## When to use which granularity

Decision tree, based on what the sweep shows:

* **Primary metric is Phase-1 retention.** Use `global` with the smallest
  t the new task tolerates. At t=32, global beats per_head by ~3 ppl QA
  (13.5 vs 16.3) at indistinguishable MT loss.
* **Primary metric is per-head expressivity** (e.g. you suspect different
  heads need genuinely different slots, the way `notebook/granularity_overlap_analysis.py`
  suggests for non-stage-2 settings). Use `per_head`, but increase t to at
  least 128 so the per-step floor dominates the rotation-induced inflation.
* **You don't know yet.** Use `per_layer` at t = 64–128. It sits in the
  middle of the coverage curve and recovers most of per_head's expressivity
  with much less footprint inflation.

The "convergence" at t=256 (all granularities land in 18.9–19.6 ppl) is a
**saturation artefact**: coverage exceeds 0.58 in every config so the
mechanism is no longer differentiating. Avoid drawing conclusions about
granularity from t=256 alone.

---

## How to reproduce / extend

### Existing scripts

| File | Purpose |
|---|---|
| `qasper-top-t-slot-investigation/analyze_slot_selection.py` | Mid-training snapshot heatmaps + step-to-step Jaccard table. |
| `qasper-top-t-slot-investigation/analyze_coverage_vs_forgetting.py` | Cumulative coverage growth curves, bar chart, scatter vs forgetting ppl. |
| `qasper-forgetting-investigation/plot_granularity_x_sparsity.py` | The 2 × 4 forgetting/acquisition curves that motivate the analysis. |
| `notebook/granularity_overlap_analysis.py` | Cross-granularity Jaccard (Borda-consensus framing) for `bg_stats` and `tfidf_ranking_log`. |

The two scripts above are keyed by `(granularity, top_t)`; drop new run dirs
into the `RUNS` dict in `analyze_slot_selection.py` to extend the sweep
(e.g. add Qwen, or add a new t).

### How to read a new model's logs

1. Confirm the run wrote `sparse_slot_log.pt` (only continual phases do,
   initial phases just collect `bg_stats.pt`).
2. Run both scripts. The CSVs are the deliverable; the figures are
   supporting evidence.
3. Headline numbers to report: Pearson r and Spearman ρ between coverage
   and forgetting ppl, plus the per-(granularity, t) coverage table.

### Caveats when applying to other models

* The coverage = footprint argument relies on `momentum_masking="freeze"`.
  Under `soft` or `decouple`, masked positions still drift (momentum decay
  or natural decay respectively), so the footprint is no longer the right
  object — every position drifts to some extent. The right diagnostic
  becomes *integrated drift*, not binary coverage.
* GQA models like Llama-3.2-3B have fewer KV heads than Q heads. Per_head
  granularity is over KV heads (H=8 here, not 24). Don't confuse the two
  when computing footprints.
* Llama-3.2-3B uses 28 layers, Qwen3-4B uses a different `L` — the
  coverage script infers shapes from `access_scores` so it adapts
  automatically, but the absolute coverage numbers are not comparable
  across models.

---

## Open questions (good targets for follow-ups)

1. **At t=256, per_head still forgets ~0.7 ppl more than per_layer despite
   identical coverage (0.66 vs 0.66).** Coverage explains 91 % of variance,
   not 100 %. The residual is a "quality-of-choice" effect: per_head writes
   may intersect Phase-1-important slots more often *per parameter touched*.
   Diagnostic: intersect each run's footprint with the top-K positions of
   `bg_stats.pt` and correlate intersection size with the residual ppl.
2. **Coverage-matched intervention.** Force global's mask to rotate randomly
   between equally-sized t-subsets so its cumulative coverage matches
   per_head's 17 % at t=32. Prediction from the coverage model: QA ppl
   should rise from 13.5 to ≈ 15.5–16.5, matching per_head.
3. **Logging granularity.** We log every 30 optimizer steps; the union is
   therefore a lower bound on the true footprint. Logging every step is
   cheap and would tighten the bound, especially for per_head where churn
   is highest. If you change the logging interval, update step targets in
   `analyze_slot_selection.py` (`mid_step` lookup uses step value, not log
   index, so it's robust).
4. **The Qwen3-4B sweep is in `qasper-forgetting-investigation` but the
   slot-log analysis has not been run for it.** Should be a 30-line addition
   to `RUNS` in `analyze_slot_selection.py`. Predict the same coverage →
   forgetting relationship; falsifies the theory if it doesn't replicate.
5. **Initial-phase `bg_stats.pt` ranking.** The three `initial_sparse-*`
   runs only collect statistics — they have no slot log. Use their
   `bg_stats.pt` to compute a "static IDF top-k" baseline footprint and
   show it as a 0/1 panel alongside the dynamic TF-IDF panels in the
   main figure. Same visual language; lets the reader contrast the two
   selection rules.
6. **What about non-Qasper datasets?** This sweep is purely on Qasper.
   The mechanism (freeze-masking + cumulative union) is dataset-agnostic,
   but cluster structure may differ (e.g. medical QA may attend to
   different slot ranges than Qasper).

---

## Cross-references

* `sparse-finetuning-design` — TF-IDF mechanics, the masking modes, why
  freeze is Pareto-dominant.
* `sparse-finetuning-hypotheses` — momentum-masking ablations and the
  decay-to-zero argument for why freeze ≫ decouple.
* `cartridges-experiment-investigation` — general protocol for digging
  through an `outputs/<launch>/<uuid>/` directory.
* `cartridges/sparse_cache_finetuning.py` — implementation of the
  three granularities (`_rank_global`, `_rank_per_layer`, `_rank_per_head`)
  and freeze-masking (`save_non_top_t_state` / `restore_non_top_t_state`).
* `cartridges/train.py` lines ~570–710 — where `sparse_slot_log` and
  `tfidf_ranking_log` are populated and saved.
