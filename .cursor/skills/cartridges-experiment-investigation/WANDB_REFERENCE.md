# WANDB reference for the gated-continual-cartridges project

Wandb entity/project: **`vqtri-purdue-university/SEACrowd`**
(Verify by `wandb.Api().runs("vqtri-purdue-university/SEACrowd")` or by checking
any `wandb-summary.json` under `examples/<dataset>/scripts/wandb/run-*/files/`.)

Local wandb dirs (offline mirror of run files including `output.log`,
`config.yaml`, `wandb-summary.json`, and saved `cache-step{N}.pt` /
`bg_stats.pt`):

```
gated-continual-cartridges/examples/<dataset>/scripts/wandb/run-<YYYYMMDD_HHMMSS-XXXX>/files/
```

---

## Group naming convention

Every Phase-1+Phase-2 sweep produces TWO groups: a `train` group and an `eval`
group.

- Training group: `<dataset> - granularity tf-idf` — Phase-1 + 8 Phase-2 runs
  (per_layer × {64,128,256,512}, per_head × {64,128,256,512}).
- Forgetting/learning eval group: `<dataset> - [granularity x forgetting/learning]` —
  18 short eval runs (1 baseline + 8 sweep × 2 eval splits).

Older / pre-all-reduce experiments live under `ablation - granularity tf-idf`
and `ablation - [granularity x forgetting/learning]` (covers QASPER's first
ablation pass, no all-reduce on background stats).

Other useful groups:
- `ablation-[key/value  x momentum]` — momentum_masking and freeze_keys ablations.
- `ablation - [momentum x forgetting/learning]` — companion eval group.
- `qasper - granularity tf-idf` — QASPER all-reduce sweep (the working reference).

---

## Run-name regexes

### Training runs

```
phase 1:
  <dataset>_phase1_<gran>_all_reduced
  e.g. longhealth_phase1_per_head_all_reduced
  e.g. qasper_phase1_per-head_all-reduce

phase 2 (sparse):
  <dataset>_phase2_sparse_<momentum>_<scope>_<optimizer>_top-<T>_<gran>_lr<lr>_all-reduce
  e.g. longhealth_phase2_sparse_freeze_key-value_adam_top-128_per_head_lr2e-2_all-reduce
       (regex: top-(\d+) → 128, gran token after → per_head)
```

### Eval runs (post-hoc forgetting/learning)

```
<label>__<eval_split>_<variant>
  e.g. baseline__p1-10_eval                 (LongHealth)
  e.g. per-head-top-128__p11-20_eval        (LongHealth)
  e.g. baseline_qa_eval_key-value           (QASPER, note single underscore)
  e.g. per_layer_top128__mt_eval_value-only (QASPER)
```

LongHealth uses double-underscore separators; QASPER mixes single and double.
Allow both:

```python
re.match(r"(?P<label>.+?)_*(?P<eset>(p1-10|p11-20|qa_eval|mt_eval))_(?P<variant>(eval|key-value|value-only))$", name)
```

---

## Metric key conventions

Eval metric is dataset-name-templated. Same pattern across all examples:

| dataset | loss key | ppl key |
|---|---|---|
| LongHealth | `eval_longhealth_perplexity/loss` | `eval_longhealth_perplexity/perplexity` |
| QASPER | `eval_qasper_perplexity/loss` | `eval_qasper_perplexity/perplexity` |

The eval-run *name* encodes which dataset/split was used. The metric key only
encodes the dataset family. So you must parse the run name to know whether a
ppl number is on Phase-1 data (forgetting) or Phase-2 data (learning).

Other useful keys (training runs):
- `train/optimizer_step` — true X-axis (vs `_step`, which counts wandb log calls).
- `train/loss`, `train/perplexity` — per-step training loss.
- `generate_<dataset>_accuracy/score` — MCQ accuracy when the eval has it.
- `optimizer/lr_group0` — current LR (useful for spotting LR-floor tail).
- `num_trainable_tokens` — confirms cache size (may be `NUM_TOKENS - 1` due to
  `num_frozen_tokens=1`).

---

## Saved checkpoint conventions

In `outputs/<timestamp>-<runtag>/<uuid>/`:

- `cache-step{N}.pt` — periodic checkpoint per `save_every_n_steps`. Multiple
  exist; the LATEST step number is **typically not** the best ppl checkpoint.
  Eval the second-to-last (or earlier mid-training) checkpoint when checking
  for U-shape overshoot.
- `cache_last.pt` — alias for the last `cache-step{N}.pt`.
- `bg_stats.pt` — background access ranks for IDF (granularity-shaped).
- `sparse_slot_log.pt` — list of dicts logging top-T positions every ~30 steps.
- `tfidf_ranking_log.pt` — TF-IDF stats per logged step.
- `config.yaml` — full pydrantic config (peak LR, scheduler, top_t, granularity,
  paths to Phase-1 cache and bg_stats).

---

## Reference baseline gaps (for "is the dataset producing a continual-learning
challenge?" check)

| dataset (ablation) | P1-task ppl | P2-task ppl | gap | verdict |
|---|---|---|---|---|
| QASPER (key-value) | 5.66 | 31.24 | 25.59 | strong CL signal |
| LongHealth (per-head/per-layer all-reduce) | 2.375 | 2.491 | 0.116 | no CL signal |

If your dataset's baseline P2-task ppl is within ~5-10% of P1-task ppl, the
two phases are too similar to produce visible forgetting/learning. Redesign
the task split before chasing method-level differences.

**Mechanism reference**: see [DATASET_TRAIN_EVAL_AUDIT.md](DATASET_TRAIN_EVAL_AUDIT.md)
for the per-dataset audit of *why* these gaps look the way they do. Summary:
QASPER eval is near-OOD relative to its synth (no identity cue, open-vocab
answers, format wrapper unseen in synth). LongHealth eval is near-IID once the
verbatim patient identity tuple is hit, and it's a 5-way MCQ with fuzzy
`SequenceMatcher` scoring — so the tiny baseline gap is a *metric-floor*
phenomenon, not a property of the cartridge or recipe. Most cross-dataset
deltas in this repo are dataset-design artifacts; that file lists the
eval-only ablations that disambiguate this from method-level effects.
