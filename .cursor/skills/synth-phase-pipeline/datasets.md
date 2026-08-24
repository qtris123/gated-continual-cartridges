# Per-dataset splits, synth sampling, and AM gaps

Token tables, eval samples, and LPT proofs live in `cartridges/data/PHASES.md`.
This file is the **why** and the **code contract**. Do not copy PHASES.md here.

## Discovery map

| concern | look here |
|---|---|
| Phase assignment (committed tables) | `cartridges/data/{qasper,quality,finqa,techqa}/resources.py`, `longhealth/phases.py` |
| Synth sampling | same modules, `sample_prompt` vs `to_string` |
| Unified synth CLI | `examples/shared/synth/self_study_vllm.py` |
| Drivers | `examples/{qasper,longhealth,quality,finqa,techqa}/synthesize/self_study.sh` |
| Parquet checks | `examples/shared/synth/validate_self_study.py` |
| Resource tests (no vLLM) | `cartridges/tests/resources/test_phase_synth.py` |
| AM grouping + teacher | `cartridges/am/components/queries.py` |
| Prompt-byte pins | `cartridges/tests/test_am_reference_data.py` |
| Materialized concat corpora | `data/<ds>/phases/phasek.txt` (built from each resource's `to_string()`) |

`PHASES.md` "Still outstanding" is stale: QASPER ASR/KG evals and much of the
synth pipeline exist. Treat the code + this skill as current; treat PHASES.md
as the **split rationale**.

## Shared design choices

**Unit of forgetting.** A phase is a set of entities (patients, papers, tickers,
stories, technotes). Continual AM writes **one document at a time**. Mixing two
entities in one synth prompt trains the generator on comparisons that eval never
asks, and makes `document_key` grouping lie.

**`sample_prompt` vs `to_string`.** Synth uses `sample_prompt` (one unit, often
a subset). Concatenated `to_string()` is for ICL/export and token accounting
only. Never feed `phasek.txt` to the synthesizer.

**LPT** (QuALITY, FinQA, TechQA): sort items largest-first, place each into the
currently smallest phase; ticker/filename/id as tie-break; equal load → lower
phase number. `build_phases()` must reproduce the committed table.

**Not LPT:** QASPER is topic panels of 16; LongHealth is patients 01–20 in ID
order, four per phase.

**Seeds.** Production synth uses `structuring summarization question use_case
creative` (`self_study_vllm.py`). Resource defaults of `["generic"]` are
overridden by that CLI. Registry: `cartridges/data/resources.py`.

**vLLM recipe** (match QASPER ASR): 8192 rows, batch 32, 8 parallel batches,
DP=4, `max_model_len=32768`, `prob_thinking=0.2`, port 8000, `WANDB_MODE=offline`.
Tokasaurus is unusable here (flashinfer vs CUDA 13.3). Off-policy: base model +
document as system prompt; AM does not need stored logprobs.
`return_tokens_as_token_ids` is currently `False` in the CLI.

**Empty-batch trap.** vLLM 500 / `APIConnectionError` can yield `exit 0` with
far fewer than 8192 rows. Trust parquet row counts.

**Checker heuristic.** `FAIL: user turns not diverse enough` is unique user
*prefixes* (seed questions repeat). Empty fields and missing ID tags are the
real fails.

## QASPER

- **Table:** `TOPIC_TO_IDS`. Phases `1=QA 2=MT 3=SA 4=ASR 5=KG` (`QASPER_PHASE_TO_TOPIC`).
- **Why KG not summarization:** embedding centroids; summarization overlapped
  QA. ASR excludes speech-*translation* (that's MT). MT still contains
  `1810.03459` (speech recognition) — left in place to avoid regenerating MT eval.
- **Synth:** random paper, random subset of sections. Context includes
  `---Paper Title: ...---` dividers.
- **AM teacher:** `full_paper_prompt` reconstructs **all** sections from HF,
  **without** those dividers (~5% tokens). Bytes are RoPE (`doc_rope_offset`).
  Hashes pinned in `test_am_reference_data.py`.
- **ID:** `<title>` — works with `document_key`.
- **Eval:** rewritten by `qasper/rewrite.py` (names the paper); not raw HF QAs.
  Paths: `data/qasper/phases/` and `data/qasper/eval/`.

## QuALITY

- **Table:** `PHASE_TO_ARTICLE_IDS` (35 stories ≥ 6k tokens, 7 per phase, LPT).
- **Rejected:** split on `writer_id` (that's the **question crowdworker**, not
  the author). Author-split is unbalanceable. Length filter drops Slate.
- **Synth:** **full** story in `<story><title>…`. Grouping by title matches the
  teacher if you still rebuild from `QuALITYResource` (byte-stable).
- **AM:** `full_quality_prompt` needs `dataset=quality` **and** `quality_phase`.
  Default teacher is QASPER — wrong dataset → `KeyError`.
- **Eval:** 4-way MCQ; gold is **1-based** `gold_label`. Optional `difficult`.
  Files: `data/quality/phases/phase{1–5}_eval.parquet`, plus
  `phase{1–5}_eval_anchored.parquet` with the story named in each question.

## FinQA

- **Table:** `PHASE_TO_COMPANIES`, FY2017 only, 13–15 tickers/phase, LPT on
  whole-company token totals (`FinQAPage.text` including `<page><source>`).
- **Rejected:** one company across years (cross-year figure blur; leftover
  year-less questions). Five largest FY2012 companies were too small (~9–13k).
- **Synth:** pick **one ticker**, then 1..`pages_per_prompt` (default 4) pages.
  Do not mix companies in one prompt.
- **AM:** not wired. Teacher must be **all pages of that ticker**, not the
  sampled pages in the row. `document_key` only reads `<title>`; FinQA IDs live
  in `<source>TICKER/YEAR/page_N.pdf`. Checker extracts ticker from `<source>`.
- **Eval:** numeric + executable `program`. Only exactly-checkable stream.

## LongHealth

- **Table:** `PHASE_TO_PATIENT_IDS` — patients 01–20 in order, 4/phase. Cleanest
  forgetting curve (80 questions every phase). Heterogeneous diagnoses **inside**
  a phase by design.
- **Synth:** one patient; default 1 note (`min_notes_per_prompt` /
  `max_notes_per_prompt`). Optional char chunking. Prompt has name/id/birthday
  in prose, **no `<title>`**. `document_key` will fall back to the entire system
  prompt (one fake doc per unique note sample) unless you add a tag or metadata
  before AM.
- **AM:** not wired. Teacher should be the **full patient record**, not one note.
- **Eval:** 5-way MCQ; gold is option **text** (`"Vincristine"`), not a letter.

## TechQA

- **Table:** `PHASE_TO_FILENAMES` frozen in `phases.json` (~99 notes/phase, LPT).
- **Rejected:** one filename per phase (notes are tiny). Product split
  (WebSphere vs Tivoli) is unbalanceable. `is_impossible` rows (`answer='-'`,
  empty contexts) are dropped from phases; they need a separate abstention eval.
- **Synth:** resource default `docs_per_prompt=4`, but `self_study_vllm.py`
  forces **1** unless `--docs-per-prompt` is set. Notes over
  `TECHQA_MAX_PROMPT_TOKENS` (default 60k) are dropped at resource init.
- **AM:** not wired. ID is `<source>` filename. Mixing notes (`docs_per_prompt>1`)
  makes one system prompt many documents — keep 1 for AM-compatible grouping.
- **Eval:** free-form support tickets; judge or log-ppl.

## `document_key` / teacher contract

```
document_key(convo) = <title> inner text OR entire system_prompt
full_document_prompt(title, dataset=, qasper_topic=, quality_phase=)
  dataset qasper → full_paper_prompt
  dataset quality → full_quality_prompt (phase required)
  else → ValueError
```

When adding FinQA / LongHealth / TechQA AM:

1. Extract a **stable entity id** (`<source>` ticker, patient_id, filename) —
   not the whole prompt.
2. Look up the **complete** unit from the catalog, even if synth was a subset.
3. Pin prompt bytes (hash or token count) if numbers must stay comparable.
4. Pass `TeacherTarget.dataset` (and phase/topic) so QASPER lookup is not used
   by default.

## Eval format adapters (do not unify)

| dataset | gold |
|---|---|
| LongHealth | option text |
| QuALITY | 1-based index |
| FinQA | number + program |
| QASPER / TechQA | prose |

ICL concat `phasek.txt` overflows: see PHASES.md "Consequences for the
in-context baseline". Cartridges exist because cumulative ICL does not fit.

## Adding a sixth dataset (worked constraints)

A good stream: many disjoint entities, token totals even across 5 phases (~40k+
so a 1024-slot cartridge is actually compressing), questions that stay on one
entity, and a stable ID in every synth prompt.

A bad stream: same entity across phases with a hidden confounder (year,
crowdworker, product line that is 15× another), or synth that concatenates the
whole phase.

Checklist is in [SKILL.md](SKILL.md).
