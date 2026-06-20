---
name: cartridges-experiment-investigation
description: Investigate unexpected results from cartridge sparse continual-learning runs (LongHealth, QASPER, MTOB, etc.) in the gated-continual-cartridges repo. Use when the user shares an eval table, wandb screenshot, or training-loss plot and asks "why is this happening?", "is it the pipeline, the data, or the method?", or asks to debug forgetting/learning trade-offs in Phase-2 sparse runs. Companion to `sparse-finetuning-design` (architecture facts) and `sparse-finetuning-hypotheses` (open questions).
disable-model-invocation: true
---

# Cartridges Experiment Investigation

A workflow for diagnosing cartridge continual-learning experimental surprises **without burning new GPU compute**. The default mode is *log mining + data probes + reference comparisons*, with new training proposed only as a follow-up.

## When to use this skill

Trigger on questions like:
- "These eval-ppl numbers look weird, what's wrong — pipeline, data, or method?"
- "Why does Phase 2 *improve* the Phase-1 perplexity instead of degrading it?"
- "Per-head beats per-layer here but the opposite on the other dataset — is the granularity ranking real?"
- "Should I re-run the sweep, or is something else wrong upstream?"

If the user asks you to **make code changes** to the sparse-finetuning algorithm itself, prefer `sparse-finetuning-design`. If they're brainstorming **what to ablate next**, prefer `sparse-finetuning-hypotheses`. This skill is the diagnostic harness that produces evidence to feed into those.

## Top-level workflow (the loop)

Copy this checklist into your scratch notes for the session:

```
- [ ] Phase 0: clarify scope (AskQuestion: dataset, metrics, GPU budget, missing baseline)
- [ ] Phase 1: build investigation workspace + research log skeleton (single source of truth)
- [ ] Phase 2: enumerate hypotheses, rank by (explanatory_power × cost-to-test)
- [ ] Phase 3: dispatch cheap parallel probes (sub-agents, read-only)
- [ ] Phase 4: pull per-step trajectories from wandb (not output.log — see Pitfalls)
- [ ] Phase 5: compare vs a working reference dataset (e.g., QASPER) to separate
       method-issue from dataset-issue
- [ ] Phase 6: synthesize findings into the research log; update Headline
- [ ] Phase 7: produce concrete next-step recommendations (eval-only first, training only if needed)
```

**Hard rule**: *update the research log after every probe.* Future-you (or another agent) must be able to resume from `notes/00_research_workflow.md` alone.

---

## Phase 0 — Clarify (skip only if explicitly told to)

Use `AskQuestion` to pin these down before launching probes:

1. **Primary concern**: which observation drives priority? (no-forgetting / granularity ranking / non-monotonic top-T / baseline-too-close / all-of-the-above).
2. **Metric focus**: perplexity, MCQ accuracy, or both. Perplexity on synth data is a *weak* forgetting probe; MCQ is more discriminative.
3. **Missing baselines**: does a dense (no-sparsity) Phase-2 baseline exist, or do you only have Phase-1-only as the "baseline"?
4. **Compute budget**: analysis-only / few short runs / free rein. Default to analysis-only.
5. **Cache size intent**: is `NUM_TOKENS` deliberate or a default the user is open to changing?

---

## Phase 1 — Investigation workspace

Create a workspace **outside the repo** (do not modify shipped scripts/code):

```
/localhome/local-triv/sparse_continual_investigation/   # or similar, outside the repo
├── notes/
│   ├── 00_research_workflow.md   # SoT — see RESEARCH_LOG_TEMPLATE.md
│   ├── 01_trajectories.md
│   ├── 02_data_probe.md
│   ├── 03_sparse_dynamics.md
│   └── 04_<reference-vs-target>.md
├── scripts/
│   └── (copies of skill scripts adapted to current question)
├── results/
│   ├── trajectories.csv
│   ├── wandb_*_summaries.json
│   ├── wandb_*_histories.json
│   └── *.png
└── data_probes/
    └── (raw token stats, leakage tables, etc.)
```

Seed `00_research_workflow.md` from `RESEARCH_LOG_TEMPLATE.md`. Fill in Headline, Hypotheses, and Constraints **before** running probes.

---

## Phase 2 — Hypothesis menu (start here, prune as evidence arrives)

Default ranked list for any cartridge continual-learning surprise. Prune liberally based on the user's specific concern.

| H | Hypothesis | Cheap probe |
|---|---|---|
| H1 | Phase-1 is **undertrained** → continued Phase-2 training keeps improving regardless of new content. | Mine Phase-1 wandb history; plot eval-ppl vs step. |
| H1' | Phase-1 is **overtrained** → final ckpt is past the eval-ppl minimum; Phase-2 acts as un-overfit regularizer. | Same as H1; look for U-shape. |
| H2 | Eval set is too **small / structural** to detect forgetting. | Tokenize eval parquet, compute structural-vs-unique token ratio. |
| H3 | Two phases are **too similar** (no real continual-learning challenge). | Compare baseline gap (P2-task ppl − P1-task ppl) to a working reference (QASPER). See `DATASET_TRAIN_EVAL_AUDIT.md` for the mechanism. |
| H4 | Cache init **pre-encodes** target patients/docs (information leak). | Tokenize cache init source, identify which docs land in first NUM_TOKENS tokens. |
| H5 | Eval set too **small** for the deltas being measured (variance dominates). | Bootstrap CI from per-batch losses if available; or count tokens in eval. |
| H6 | `packed_seq_length` truncation **drops most content**. | Token-count distribution per row; % truncated. |
| H7 | Granularity ranking reflects **parameter-budget asymmetry** rather than method. | Read `apply_gradient_mask_to_cache` and verify identical budget at matched top-T. |
| H8 | Backward transfer via **shared structural format** (template tokens, system prompts, identity tuples). | Decompose loss into template vs unique tokens. See `DATASET_TRAIN_EVAL_AUDIT.md` for per-dataset audit of which identity cues / templates leak across train↔eval. |
| H9 | LR-scheduler horizon mismatch → last steps run at LR floor and overshoot. | Read `lr_scheduler` config; compare `max_steps` vs actual `train/optimizer_step`. |
| H10 | Saved checkpoint is **last** rather than **best** → reported numbers are post-optimum. | Check `save_every_n_steps`, `keep_last_n_saved`; eval `cache-step{N-mid}.pt`. |

For **every Phase-2 result that looks too good or too bad**, check H1', H9, and H10 before believing the method-level interpretation.

---

## Phase 3 — Dispatch parallel probes (sub-agents)

Use `Task` with `subagent_type=explore` for read-only investigation. Three template prompts work well in parallel:

1. **Trajectory probe** — mine wandb histories per training run; build trajectories CSV; identify U-shape minimums; flag overshoot.
2. **Data probe** — tokenize train+eval parquets; check cross-phase leakage; characterize structural-vs-unique token ratio; inspect cache-init source.
3. **Sparse-dynamics probe** — load `sparse_slot_log.pt` / `tfidf_ranking_log.pt` / `bg_stats.pt`; check slot churn, IDF avoidance, masking integrity (cache delta on never-selected positions).

Keep each probe **<15 minutes of agent time** and have it return numbers *and* a markdown summary the parent agent can paste into the research log. See `RESEARCH_LOG_TEMPLATE.md` for the prompt patterns I used.

---

## Phase 4 — Pull per-step trajectories (wandb is source of truth)

Do NOT rely on `output.log` from local wandb dirs for trajectories. The tqdm output is full of CR-overwrites + ANSI codes + Unicode block chars; regex parsing is brittle. Use the wandb API:

```bash
python /localhome/local-triv/gated-continual-cartridges/.cursor/skills/cartridges-experiment-investigation/scripts/fetch_wandb_data.py \
  --groups "longhealth - granularity tf-idf" "longhealth - [granularity x forgetting/learning]" \
  --out-dir /localhome/local-triv/sparse_continual_investigation/results/
```

The script writes `wandb_<tag>_summaries.json` and `wandb_<tag>_histories.json` per group. See `scripts/fetch_wandb_data.py` for usage details.

The eval *metric key* in cartridge runs is dataset-name-templated:
- LongHealth: `eval_longhealth_perplexity/{loss,perplexity}`
- QASPER: `eval_qasper_perplexity/{loss,perplexity}`
- The eval-run name encodes which dataset/split (e.g., `..__p1-10_eval`, `..__qa_eval_value-only`).

---

## Phase 5 — Reference comparison (the closer)

The single most informative move when the user is confused about a result on dataset X: **pull the same recipe's results on a known-working dataset Y** (typically QASPER for this repo) and put the **baseline cross-phase gaps side-by-side**.

A working continual-learning benchmark in this repo has a baseline gap of order **multiple ppl** (QASPER LongHealth: 25.6). If your dataset's gap is sub-1 ppl, the perplexity metric *cannot* show forgetting because there's nothing to forget — the Phase-1 cartridge already generalises.

Use `scripts/compare_recipes.py` to render the bar-chart comparison and an interpretation table. Save to `notes/04_<reference-vs-target>.md` and `results/<reference>_vs_<target>.png`.

**Before crediting a method-level interpretation to a cross-dataset delta**, read [DATASET_TRAIN_EVAL_AUDIT.md](DATASET_TRAIN_EVAL_AUDIT.md). It documents the per-dataset train↔eval distribution overlap (identity cues, answer-space constraint, scoring tolerance, format wrappers) — most cross-dataset gaps in this repo are explained by *dataset design*, not by the recipe being compared. The audit also lists eval-only ablations (e.g. LongHealth `include_diagnosis=False`) that disambiguate "method works" from "eval was a retrieval task all along".

---

## Phase 6 — Synthesis

Update the **Headline** at the top of `00_research_workflow.md`. The headline should:

1. State the dominant cause in one sentence.
2. Quantify it with a single number from a credible probe.
3. Distinguish it from secondary effects (which still get sub-bullets).

Also update the **Hypothesis verdict table** (CONFIRMED / REFUTED / PARTIAL / INCONCLUSIVE) with one-line evidence per row.

---

## Phase 7 — Recommendations

Always produce **eval-only recommendations first** (cheap, no training):
- Eval `cache-step{mid}.pt` checkpoints already on disk to test H10.
- Add MCQ-accuracy collection to the post-hoc forgetting eval (often missing).

Only then propose training changes (scheduler horizon, save-best-not-last, cache-size sweep, dataset redesign for genuine cross-phase difference). Frame these as **prioritized next experiments** with predicted information gain.

---

## Repo cheat sheet (essential paths and conventions)

See [WANDB_REFERENCE.md](WANDB_REFERENCE.md) for the full list of groups, run-name regexes, and metric keys.

```
gated-continual-cartridges/
├── cartridges/sparse_cache_finetuning.py     # core algorithm (read this first)
├── cartridges/train.py                       # CosWithWarmup, eval cadence, do_step
├── examples/<dataset>/scripts/train_initial_sparse.sh
├── examples/<dataset>/scripts/train_continual_sparse.sh
├── examples/<dataset>/scripts/eval_forgetting.sh   # post-hoc P1+P2 eval table
├── examples/<dataset>/train/{initial,continual}_sparse.py
├── data/<dataset>/{train,eval}/*.parquet
└── outputs/<timestamp>-<runtag>/<uuid>/
    ├── cache-step{N}.pt        # mid-training checkpoint(s) — keep these around
    ├── cache_last.pt
    ├── bg_stats.pt             # IDF source for Phase-2
    ├── sparse_slot_log.pt
    ├── tfidf_ranking_log.pt
    └── config.yaml
```

---

## Pitfalls (high-value, easy to miss)

1. **`output.log` regex parsing is fragile.** tqdm uses CR-overwrites + ANSI codes; the `loss=`, `ppl=`, `optimizer_step=` fields live *inside* the `[...]` time bracket, not after it. Use wandb API instead.
2. **`FREEZE_KEYS=0` means keys ARE updated**, not frozen. The shell flag's name is the same as the boolean it sets, so `os.environ.get("FREEZE_KEYS", "0") not in ("0", "false", "False")` is `False` when `FREEZE_KEYS=0`. Read the resolved value from `config.yaml` to be sure.
3. **`packed_seq_length` is set per-config and differs train vs eval.** Train uses `truncate`, eval often uses `pad`. Check both.
4. **`eval_longhealth_perplexity/num_elements: 0`** in summaries is a logging artifact, not a bug — `num_elements` is the per-batch sum of `assistant_tokens`, which is `nan` because `targets="tokens"` (full-sequence loss). Don't waste time chasing it.
5. **`max_steps` in `CosWithWarmup` is decoupled from total training steps.** `EPOCHS * steps_per_epoch` may exceed `max_steps`, so the tail of training runs at LR floor (0.1× peak by default). Always compare `max_steps` to actual final `train/optimizer_step`.
6. **Saved checkpoint = last, not best.** `keep_last_n_saved=1` keeps only the final cache. Mid-training checkpoints (e.g., `cache-step256.pt`) often exist alongside; eval those before believing the "saved" numbers.
7. **`longhealth_context.txt` is incomplete** — only patients 01-14, ~156k tokens. With `NUM_TOKENS=1024`, cache init covers a sliver of patient_01 only. Don't assume the cache "knows" all patients pre-training.
8. **Phase-1 and Phase-2 cache initialization paths differ.** per-layer P2 runs use the per-layer Phase-1 ckpt; per-head P2 uses the per-head one. Verify in `config.yaml: kv_cache_initializer.path` and `background_indices_path` before claiming a comparison is apples-to-apples.

---

## Reusable artifacts in this skill

- [RESEARCH_LOG_TEMPLATE.md](RESEARCH_LOG_TEMPLATE.md) — seed for `notes/00_research_workflow.md`
- [WANDB_REFERENCE.md](WANDB_REFERENCE.md) — groups, run-name regexes, metric keys, naming conventions
- [DATASET_TRAIN_EVAL_AUDIT.md](DATASET_TRAIN_EVAL_AUDIT.md) — per-dataset train↔eval distribution overlap (QASPER, LongHealth). Read before interpreting any cross-dataset comparison or claiming a method-level effect.
- [scripts/fetch_wandb_data.py](scripts/fetch_wandb_data.py) — pull histories + summaries by wandb group name
- [scripts/build_trajectories_csv.py](scripts/build_trajectories_csv.py) — wandb histories → tidy per-step CSV
- [scripts/compare_recipes.py](scripts/compare_recipes.py) — side-by-side bar chart + delta table for two datasets

Adapt and copy into the investigation workspace; keep originals here clean.

---

## Output template (final delivery to the user)

End the session with:

1. **A single image** — the most decisive plot (usually the reference-vs-target bar chart, or the U-shape trajectory).
2. **A 3-row TL;DR** — dominant cause + single number + one secondary effect.
3. **A pointer to the workspace** — `notes/00_research_workflow.md` and the file map.
4. **Two follow-up experiments**, with predicted outcomes — one eval-only (cheap), one training-level.
5. **One open question** to push the user on (e.g., "Was `FREEZE_KEYS=0` intentional?").
