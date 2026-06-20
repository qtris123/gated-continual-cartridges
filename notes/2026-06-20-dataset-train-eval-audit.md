# Dataset train/eval distribution audit — QASPER vs LongHealth

- **Date:** 2026-06-20
- **Status:** done
- **Related:** `.cursor/skills/cartridges-experiment-investigation/DATASET_TRAIN_EVAL_AUDIT.md` (durable artifact), `.cursor/skills/cartridges-experiment-investigation/SKILL.md`, `.cursor/skills/cartridges-experiment-investigation/WANDB_REFERENCE.md`

## Goal

Two questions, treated as one investigation:

1. How is the QASPER eval dataset actually constructed end-to-end? (entry point: `cartridges/data/qasper/evals.py`)
2. Given that synth and eval both come from the same QASPER papers, how similar are their distributions — and how does that compare to LongHealth, where eval accuracy is consistently better?

Working hypothesis going in: LongHealth's apparent superiority over QASPER on eval is **not** a method-level effect; it's a dataset-design artifact (train↔eval surface alignment + closed-vocab MCQ + tolerant scoring).

## Setup

- **Branch / commit:** `c3df9fe` on `tri_work_placeholder`
- **Code entry points read:**
  - QASPER: `cartridges/data/qasper/{resources,rewrite,evals}.py`, `cartridges/benchmark/datasets.py::_load_qasper`, `examples/qasper2/convert_hf_to_qasper_eval_mt_parquet.py`, `examples/qasper2/synthesize/self_study.py`
  - LongHealth: `cartridges/data/longhealth/{utils,resources,evals}.py`, `examples/longhealth/synthesize/self_study.py`, `examples/longhealth/train/initial_sparse.py`
  - Shared: `cartridges/synthesizers/self_study.py`, `cartridges/data/resources.py` (`SEED_PROMPT_REGISTRY`)
- **Data inspected:** `data/longhealth/eval/patients_01_to_10.parquet` (200 rows, schema `messages | system_prompt | metadata | type`) via ad-hoc `pandas` read.
- **No training run was launched.** Analysis-only.

## What I tried

1. Traced the QASPER eval pipeline from `evals.py` backward to `rewrite.py` (HF push) and forward to (a) the perplexity parquet builder `convert_hf_to_qasper_eval_mt_parquet.py`, and (b) the generation-eval `_load_qasper`. Confirmed all three paths consume the same `qtris123/...-{MT,SA}-task` (and `sabrieyuboglu/...-QA`) HF datasets that `rewrite.py` produces.
2. Traced the QASPER synth pipeline (`QASPERResource.sample_prompt` → `SelfStudySynthesizer`) and compared its prompt template, seed prompts, and answer-author distribution to the eval pipeline above.
3. Repeated the same trace for LongHealth: `LongHealthMultipleChoiceGenerateDataset.wrap_question` + the perplexity parquet rows + `LongHealthResource.sample_prompt`. Inspected one parquet row directly to confirm format (user prompt embeds full `(patient_id, name, birthday, diagnosis)` tuple; assistant gold = `<answer>\nVincristine\n</answer>`).
4. Built a side-by-side audit table (identity overlap / answer-space constraint / scoring tolerance / format wrappers / writer-style overlap) and reconciled it with the existing `WANDB_REFERENCE.md` "Reference baseline gaps" numbers (QASPER 25.59 ppl, LongHealth 0.116 ppl).
5. Wrote up the audit as a durable reference file in the investigation skill and cross-linked it from `SKILL.md` (H3, H8, Phase 5, Reusable artifacts) and `WANDB_REFERENCE.md`. No repo source code changed.

## Key findings / insights

- **The QASPER eval set is built in 3 stages**:
  1. Pick 16 arXiv paper IDs per topic via `TOPIC_TO_IDS` in `cartridges/data/qasper/resources.py`.
  2. For every non-`unanswerable` QA pair in those papers, GPT-4.1 (`temperature=0.0`) rewrites the question to be self-contained (conditioned on title only) and re-derives a succinct closed-book answer from the original `answer_data` (`rewrite.py:32-57`). Result pushed to HF as `{sabrieyuboglu/qasper-rewrite-gpt-4.1, qtris123/...-MT-task, qtris123/...-SA-task}`, split name `"question"`.
  3. Each row becomes a 2-message convo with user prompt `Please write a succinct answer ... <answer>{...}</answer>` and assistant gold `<answer>\n{rewritten}\n</answer>`. The `examples/qasper2/qasper_eval_{QA,MT}.parquet` files are baked from this HF dataset with empty `system_prompt`. Perplexity is measured on the assistant tokens; closed-book at eval time (no paper text in prompt).

- **QASPER train ↔ eval is near-OOD on everything except the underlying paper text.**
  - Different writer (Qwen3-4B at temp 0.6/0.0 vs GPT-4.1 at temp 0.0).
  - Different answer length distribution (Qwen open-book long answers vs GPT-4.1 succinct).
  - `<answer>...</answer>` wrapper is required at eval but **never appears in synth** — a non-trivial fraction of any QASPER ppl improvement is plausibly format adaptation, not content recall.
  - No identity cue in the eval user prompt — empty `system_prompt`, no paper title/abstract embedded, so the cartridge has no token-level retrieval key to hit.
  - Section coverage is mismatched: synth samples sections uniformly, but human QASPER questions were written against the abstract → "headline" content only.

- **LongHealth train ↔ eval is near-IID on the dimensions that matter most.**
  - `(patient_id, name, birthday, diagnosis)` appears **verbatim** in both `LongHealthResource.SYSTEM_PROMPT_TEMPLATE` (synth context) and `LongHealthMultipleChoiceGenerateDataset.wrap_question` user prompt (eval). The cartridge only needs to remember "which note belongs to this patient key", not the content from scratch.
  - Eval is **5-way MCQ with fuzzy `SequenceMatcher` scoring** against the 5 provided options (`evals.py:117`). The model only has to push P(correct option string) > P(other 4); paraphrase / format drift are forgiven.
  - Synth seed-prompt mix (`structuring / summarization / question / use_case / creative`) naturally produces atomic-fact Q/As over notes — exactly the granularity at which the human-written MCQs operate.

- **The 25.59 vs 0.116 ppl baseline-gap difference in `WANDB_REFERENCE.md` is dataset-design, not method.** QASPER's gap reflects a genuinely hard cross-phase task with multi-ppl headroom; LongHealth's sub-1-ppl gap is a metric-floor phenomenon because Phase 1 already generalises to Phase 2 patients via the shared identity-tuple + closed-vocab MCQ structure. **Any cross-dataset comparison of recipes that doesn't account for this will systematically over-credit LongHealth-friendly methods.**

- **Disambiguating ablations identified (cheap, eval-only)** — recorded in the durable artifact so future investigations don't re-derive them:
  - LongHealth: rerun `LongHealthMultipleChoiceGenerateDataset` with `include_diagnosis=False`. If accuracy collapses, the eval was retrieval-by-identity all along.
  - LongHealth: paraphrase / shuffle the 5 option strings before scoring. Tests whether the closed-vocab cheat was doing the work.
  - QASPER: synthesize a small training mix with only `seed_prompts=["question"]` and force `<answer>...</answer>`-wrapped bot-B outputs. Measures how much of any QASPER ppl improvement is format-adaptation vs. content-recall.

- **Two minor pipeline observations worth recording:**
  - `cartridges/data/qasper/evals.py::QasperEvalDataset` imports `CartridgePerplexityDataset` from `cartridges.datasets`, which no longer exists there (the current parent is `LossEvalDataset`), and `__getitem__` reassigns `self.data` on every call. The active pipeline bypasses this and goes through the `qasper_eval_*.parquet` → `LossEvalDataset` path instead.
  - `cartridges/data/longhealth/utils.py` defines `load_longhealth_dataset` **twice** (lines 39-57 and 62-79); the second definition shadows the first. Harmless because they're identical, but worth a future cleanup.

## Gotchas / surprises

- The synth bot answers QASPER/LongHealth questions **open-book** (paper / notes are in its system prompt at generation time), but those answers are replayed **closed-book** at training time. So the synth labels are systematically more confident/specific than what a closed-book Qwen would produce — fine for distillation, but it means "train ppl − eval ppl" is not a meaningful generalisation gap; it's mostly the open-book/closed-book authoring gap.
- LongHealth's `score()` extracts the answer between `<answer>...</answer>` tags and **falls back to `question.answer_a` when the tags are missing** (`evals.py:140-143`). So a model that produces malformed output scores the same as one that confidently picks option A — not random. This biases accuracy upward whenever option A happens to be correct in the eval set.
- `LongHealthMultipleChoiceGenerateDataset.Config` defaults to `include_diagnosis=True` and `cot=True`. Both of these inflate accuracy substantially. If you see a LongHealth accuracy quoted without these flags spelled out, be suspicious.
- `examples/qasper2/synthesize/self_study.py:87` hard-codes the QASPER topic split's HF dataset to `qtris123/qtris123qasper-rewrite-gpt-4.1-SA-task`. The matching MT/QA datasets exist on HF but you have to switch the path manually — there's no `topic`-keyed selector in `evals.py` (it's a single string in the Config, not a dict).
- `data/longhealth/eval/patients_*.parquet` rows use the same prompt template as `LongHealthMultipleChoiceGenerateDataset` but **without** the CoT instruction wrapper, since they're for perplexity (teacher-forced on `<answer>\n{option}\n</answer>`). The MCQ generation eval *does* include the CoT prompt at inference. So the same eval-set examples test slightly different things in the two paths.
- QASPER's eval has no system prompt at all. LongHealth's eval also has empty `system_prompt`, but the identity tuple lives in the **user** message preamble — so both pipelines look "system-prompt-empty" superficially, but only LongHealth provides a retrieval key in-prompt.

## Artifacts

- **New durable reference (commit pending):** `.cursor/skills/cartridges-experiment-investigation/DATASET_TRAIN_EVAL_AUDIT.md` — full per-dataset audit, cross-dataset comparison table, disambiguating-ablation list, and quick-lookup table mapping every pipeline component to file + key symbol.
- **Updated:** `.cursor/skills/cartridges-experiment-investigation/SKILL.md` — H3 row, H8 row, Phase 5 section, and Reusable Artifacts list now point at the audit.
- **Updated:** `.cursor/skills/cartridges-experiment-investigation/WANDB_REFERENCE.md` — the Reference baseline gaps table now has a "Mechanism reference" paragraph explaining the QASPER 25.59 vs LongHealth 0.116 gap.
- **No repo source code changed.** No training runs, no new outputs, no wandb runs.

## Open questions / next steps

- [ ] Run the LongHealth `include_diagnosis=False` ablation on the current Phase-1+Phase-2 forgetting eval. If accuracy on the `per-head-top-512` checkpoint collapses, the strong LongHealth eval numbers in the 2026-06-18 sweep are mostly identity-tuple retrieval, not knowledge preservation — and the per-head > per-layer ranking on LongHealth would need to be re-checked on QASPER before being believed.
- [ ] Apply the disambiguating ablations to the **current sweep results** (`2026-06-18-sparse-continual-sweep.md`) before publishing any cross-dataset claim about granularity / top_t.
- [ ] Synthesize a small QASPER "format-only" training mix (`seed_prompts=["question"]` + forced `<answer>...</answer>` wrapping) and measure how much of an existing QASPER eval-ppl improvement it accounts for in isolation. If >50%, the recipe-level interpretation of QASPER deltas needs revising.
- [ ] Clean up `cartridges/data/qasper/evals.py` (broken `CartridgePerplexityDataset` import + per-call `self.data` reassignment) and the duplicate `load_longhealth_dataset` definition in `cartridges/data/longhealth/utils.py`. Both are dormant — not blocking anything — but they'll confuse the next reader.
- [ ] Consider extending the audit file with the third dataset (MTOB) once that pipeline is added, using the same identity-cue / answer-space / scoring-tolerance / format-wrapper schema.
