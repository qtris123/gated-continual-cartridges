# QASPER forgetting/learning vs. sparsity — value-only vs. key-value

- **Date:** 2026-06-20
- **Status:** done
- **Related:**
  [`2026-06-18-sparse-continual-sweep.md`](./2026-06-18-sparse-continual-sweep.md),
  [`2026-06-20-ddp-gloo-flock-pool.md`](./2026-06-20-ddp-gloo-flock-pool.md)

## Goal

Visualise the stability–plasticity trade-off for Phase 2 sparse continual
finetuning on QASPER: how does the active-slot budget (top-k ∈ {64, 128, 256,
512}) move Phase 1 *forgetting* (QA perplexity) vs. Phase 2 *learning* (MT
perplexity), per granularity (`per-layer`, `per-head`) and per cache-update
scope (value-only vs. key-value), with a Phase-1-only baseline anchor.

## Setup

- **Branch / commit:** `c3df9fea` on `tri_work_placeholder` (working tree
  has unrelated edits; nothing in this entry was committed).
- **Code entry point:** existing eval pipeline only — no new code in the
  cartridges repo. Investigation script lives **outside** the repo at
  `/localhome/local-triv/qasper-forgetting-investigation/plot_forgetting_vs_learning.py`.
- **Source eval logs (cartridges `outputs/`):**
  - `outputs/qasper_forgetting_eval_value-only_bsize32/`
    (16 runs: `qasper-value-only-{per-layer,per-head}-top-{64,128,256,512}-bsize32__{qa,mt}_eval`)
  - `outputs/qasper_forgetting_eval_key-value_bsize32_987953/`
    (16 runs: `qasper-key-value-{per-layer,per-head}-top-{64,128,256,512}__{qa,mt}_eval`)
  - `outputs/qasper_forgetting_eval_782782/baseline_{per-layer,per-head}__{qa,mt}_eval/`
    (Phase-1-only cartridge — no Phase 2 sparse finetuning)
- **Eval driver:** `examples/qasper2/train/eval_forgetting.py` (QA = Phase 1 →
  forgetting; MT = Phase 2 → learning).
- **Metric scraped:** final wandb summary line
  `eval_qasper_perplexity/perplexity` in each `eval.log`.
- **Data / hardware / wandb:** inherited from the sweeps in
  [`2026-06-20-ddp-gloo-flock-pool.md`](./2026-06-20-ddp-gloo-flock-pool.md);
  this entry is plotting-only, no new training.

## What I tried

- Built a single parameterised plotting script with one `Sweep` config per
  variant (value-only, key-value) and ran it per
  `(sweep, granularity) → 1 PNG`.
- Added the Phase-1-only baselines as dashed horizontal lines on every plot
  (verified per-layer / per-head baselines are *identical* — they are, since
  the baseline has no Phase-2 granularity dependence).
- Iterated on legend placement: inside-right (overlapped data) → outside-right
  (shrank the axes) → **below the axes, 2 columns**, with `bbox_inches="tight"`
  + `bbox_extra_artists=(legend,)` so the saved PNG grows downward to fit the
  legend instead of shrinking the plot area.

## Key findings / insights

- **Value-only shows a clean stability–plasticity trade-off**, key-value does
  not. Touching keys is the dominant driver of Phase 1 disruption — once any
  key update is allowed, the QA perplexity floor jumps to ~16–19 and stays
  ~flat across top-k, while value-only's QA perplexity scales smoothly with
  the slot budget (≈6.6 → 13.5).
- **Baselines anchor the picture:** Phase-1-only QA = **5.63**, MT = **33.33**
  (identical across granularities, as expected). Phase 2 training delivers a
  ~5× MT improvement at every sparsity level; the only knob is how much of
  Phase 1 you trade away to get the last fractions of MT.
- **Sweet spot ≈ top-64 value-only:** lands within ~1 ppl of the QA baseline
  on Phase 1 while still pulling MT into the ~6.9 range.
- **per-head vs. per-layer is a wash** in both sweeps (within ~0.1–1 ppl).
  Granularity is not the lever that matters here; cache-update scope is.

Numbers (final perplexity; lower is better; baselines QA = 5.63, MT = 33.33):

| sweep / granularity | top-64 (QA/MT) | top-128 | top-256 | top-512 |
| --- | --- | --- | --- | --- |
| value-only, per-layer | 6.71 / 7.01 | 7.56 / 6.72 | 9.90 / 6.14 | 13.46 / 5.98 |
| value-only, per-head  | 6.61 / 6.91 | 7.74 / 6.55 | 10.57 / 5.99 | 13.45 / 5.92 |
| key-value, per-layer  | 16.31 / 6.51 | 16.58 / 6.39 | 16.72 / 5.85 | 17.18 / 5.69 |
| key-value, per-head   | 17.97 / 6.44 | 18.66 / 6.12 | 18.62 / 5.73 | 17.38 / 5.62 |

Full CSV: `/localhome/local-triv/qasper-forgetting-investigation/plots/forgetting_vs_learning_summary.txt`.

## Gotchas / surprises

- **Eval-output folders get renamed mid-investigation.** The value-only
  folder was `qasper_forgetting_eval_bsize32` initially and was renamed to
  `qasper_forgetting_eval_value-only_bsize32` between iterations. The script
  hard-codes absolute paths; any rename surfaces as a `FileNotFoundError`
  pointing at a non-existent `eval.log`. Worth using a glob / wildcard prefix
  match if this happens again.
- **Run-name template differs across sweeps.** Value-only carries a
  `-bsize32` suffix in the per-run dirname; key-value does not (`bsize32`
  only appears in the parent folder). `Sweep.run_dir_template` handles this
  per-variant.
- **Don't trust matplotlib's auto y-limits with `axhline`.** The MT baseline
  at 33.33 was silently clipped off the first version of the plots because
  the data range was ~6–13; explicitly setting `ax.set_ylim(...)` to include
  the baselines was required. (User flagged "I don't see any baseline lines"
  before I caught it.)
- **Legend kept eating the plot.** Putting legend below the axes with
  `bbox_inches="tight"` + `bbox_extra_artists` is the only variant that kept
  the axes box the same physical size as the legend-less version.
- **Don't put investigation code in the cartridges repo.** Cosmetic-only
  scripts and one-off CSVs were initially dropped into
  `examples/qasper2/viz/` and `outputs/.../*.png`; user pushed back ("don't
  contaminate the cartridge's code"). Moved to a standalone folder outside
  the repo and removed all the in-repo artifacts.

## Artifacts

- **Investigation folder (outside the repo):**
  `/localhome/local-triv/qasper-forgetting-investigation/`
  - `plot_forgetting_vs_learning.py` — single parameterised script, two
    `Sweep` entries (`value_only_bsize32`, `key_value_bsize32`).
  - `README.md` — usage note + `CARTRIDGES_REPO` path constant.
- **Plots:**
  - `plots/forgetting_vs_learning_value_only_bsize32_{per_layer,per_head}.png`
  - `plots/forgetting_vs_learning_key_value_bsize32_{per_layer,per_head}.png`
- **CSV summary:** `plots/forgetting_vs_learning_summary.txt`
- **Source eval logs:** see `Setup` above.

## Open questions / next steps

- [ ] **Confirm what "key-value" actually updates** in the Phase 2 sparse
      finetuner — is the flat-and-high forgetting really from key updates, or
      from a different optimiser scope choice that came along for the ride?
      Check `cartridges/sparse_cache_finetuning.py` against the eval configs
      under `outputs/qasper_forgetting_eval_key-value_*`.
- [ ] **Run an intermediate scope** (e.g. value + key-norm-only, or
      gate-only) at top-64 to see if any key-touching variant preserves the
      smooth value-only trade-off, or whether *any* key update collapses
      Phase 1 retention.
- [ ] **Try smaller top-k** (32, 16) for the value-only sweep — top-64 is
      already near the QA baseline; the curve may bottom out below 64.
- [ ] **Overlay value-only vs. key-value on a single plot per granularity**
      (4 curves: {value,kv} × {forgetting, learning}). Will make the
      cache-update-scope effect visually obvious instead of requiring the
      reader to flip between files.
- [ ] **Repeat with a different bsize / lr** to check whether the key-value
      forgetting floor is fundamental or a hyperparameter artifact of the
      bsize-32 sweep.
