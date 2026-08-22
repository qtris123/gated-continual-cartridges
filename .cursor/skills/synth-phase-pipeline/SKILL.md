---
name: synth-phase-pipeline
description: >-
  Documents the five-stream self-study synthesis pipeline (QASPER, LongHealth,
  FinQA, QuALITY, TechQA): phase construction, document units vs concatenated
  corpora, vLLM synth knobs, parquet layout, AM teacher lookup, and how to add
  a new dataset or phase. Use when synthesizing train parquets, designing
  phases, wiring Resources, debugging document_key/teacher KeyErrors, or
  incorporating a new benchmark into continual-cartridge experiments.
---

# Five-stream synthesis and phase pipeline

Canonical writeup of splits: `cartridges/data/PHASES.md`. This skill is the
**operational contract** between phase tables, synth rows, and AM writes.

Repos: code and synth live in `trivo-explore-research-work`. Concatenated
`data/phases/<ds>/phasek.txt` + `phasek_eval.parquet` also exist on
`gated-continual-cartridges` (`feat/baseline-implementation`).

## Hard rules

1. **Never synthesize from concatenated `phasek.txt`.** That file is ICL/export.
   Synth samples one **coherent document unit** (paper, patient, company pages,
   story, technote). Mixing units in one prompt is forbidden except TechQA's
   explicit `docs_per_prompt` (default **1**).
2. **A phase is a set of documents, not a blob.** AM groups synth rows by
   `document_key()` (usually `<title>` / `<source>`). If the tag is missing,
   the key falls back to the **entire system prompt** and explodes into fake
   documents.
3. **Synth context can be a subset; the teacher must be the full unit.**
   QASPER samples random sections; AM prefills the **full paper** via
   `full_paper_prompt`. FinQA synth samples 1–4 pages of **one ticker**; AM
   still needs **all pages of that company** (catalog not wired yet).
4. **Prompt bytes are the RoPE operating point.** Changing wrappers, dividers,
   or whitespace changes `T_doc` / `doc_rope_offset` and invalidates recorded
   AM numbers. QASPER hashes: `test_am_reference_data.py`.
5. **vLLM, not Tokasaurus** on this host (flashinfer vs CUDA 13.3). Off-policy:
   base model + document in the system prompt. AM does not need stored logprobs.

## Architecture

```
phase table in resources.py  ──sample_prompt──►  SelfStudySynthesizer
        │                                            │
        │ to_string()  (ICL / phasek.txt only)       ▼
        └────────────────────────────────     parquet rows
                                              system_prompt + user/asst
                                                     │
                              document_key + full_document_prompt
                                                     ▼
                                              AM P1 compact / P2+ writes
```

Entry point: `examples/shared/synth/self_study_vllm.py`
`--dataset {qasper,longhealth,finqa,quality,techqa} --phase 1..5`

QASPER phase map: `1=QA 2=MT 3=SA 4=ASR 5=KG`.

Default synth knobs (match QASPER ASR): **8192** rows, batch **32**, **8**
in-flight batches, vLLM **DP=4**, `max_model_len=32768`, `prob_thinking=0.2`,
seeds `structuring summarization question use_case creative`.

Drivers: `examples/{qasper,longhealth,quality,finqa,techqa}/synthesize/self_study.sh`.
Chain with `KEEP_VLLM=1` then `SKIP_VLLM=1`. Checker:
`examples/shared/synth/validate_self_study.py`.

Outputs: `data/<dataset>/synth/p0N/self_study-n8192/` (`config.yaml`, artifact parquet).
AM trains on `data/<dataset>/train/qwen_<dataset>_pN_task_8192.parquet` (symlink to that artifact).
QASPER train names use topics: `qwen_qasper_ASR_task_8192.parquet`.
`outputs/synth/<dataset>` is a view of `data/<dataset>/synth`.
`data/phases/` stays the ICL/eval export tree.

## Per-dataset units

| dataset | phase table | synth unit | ID tag | AM teacher today |
|---|---|---|---|---|
| QASPER | `TOPIC_TO_IDS` | random **sections of one paper** | `<title>` | full paper (`full_paper_prompt`) |
| QuALITY | `PHASE_TO_ARTICLE_IDS` | **full story** | `<title>` | full story (`full_quality_prompt`, needs `quality_phase`) |
| LongHealth | `PHASE_TO_PATIENT_IDS` | one **patient** (default 1 note) | **no `<title>` today** — `document_key` falls back to the whole prompt | **not wired** (need full record) |
| FinQA | `PHASE_TO_COMPANIES` | 1–4 **pages of one ticker** | `<source>TICKER/...` (AM `document_key` ignores this) | **not wired** (need all pages of ticker) |
| TechQA | `phases.json` | **1** technote (`self_study_vllm.py` overrides resource default 4) | `<source>` filename | **not wired** |

Eval formats (scorer adapters required): LongHealth = option **text**; QuALITY =
**1-based** `gold_label`; FinQA = numeric + program; QASPER/TechQA = prose.

Phase construction details and LPT: [datasets.md](datasets.md).

## Warnings from production runs

- Concat `phasek.txt` is **not** the teacher document and **not** the synth corpus.
- QuALITY `writer_id` is the **question crowdworker**, not the story author. Do
  not split on it. Length ≥ 6k tokens + LPT.
- FinQA is **FY2017, vary company**, not one company across years.
- TechQA phases are **sets of filenames** (notes are tiny). Drop `is_impossible`.
- QASPER eval questions are **rewritten** (`qasper/rewrite.py`), not raw HF QAs.
- `document_key` only extracts `<title>`. FinQA/TechQA grouping for AM needs
  `<source>` (ticker or filename). LongHealth needs a patient tag or metadata.
- `TeacherTarget` / `full_document_prompt` support **qasper** and **quality**
  only. Pointing QASPER teacher at QuALITY titles used to `KeyError`; QuALITY
  now needs `AM_DATASET=quality` + `AM_QUALITY_PHASE`.
- Synth job `exit 0` with **short row counts** happened on vLLM
  `InternalServerError` / `APIConnectionError` (empty checkpoint batches).
  Check `dataset.parquet` row count, not the shell status.
- `check_synth_parquet` `FAIL: user turns not diverse enough` is a prefix
  heuristic (seed questions repeat). Empty fields / missing tags are the real
  fails.
- Do not compact concatenated `phase1.txt` as one blob. P1 AM **prefills each
  document, concatenates KV, then ridge-compacts to t**.

## Adding a dataset or phase

Copy this checklist:

```
- [ ] Choose a document unit that is coherent for forgetting (one entity).
- [ ] Write resources.py: committed phase table + build_phases() (LPT if sizes skew).
- [ ] Put a stable ID in every synth system prompt (<title> or <source>).
- [ ] sample_prompt: one unit (or documented docs_per_prompt); never mix entities.
- [ ] to_string(): full phase concat for export/ICL only.
- [ ] Wire examples/shared/synth/self_study_vllm.py DATASETS + build_resource_config.
- [ ] Driver sh (vLLM flags, KEEP_VLLM/SKIP_VLLM) + check_synth_parquet extractor.
- [ ] Tests in cartridges/tests/resources/test_phase_synth.py (no live vLLM).
- [ ] Eval parquet: LossEvalDataset schema (messages=[user, assistant], metadata id).
- [ ] AM: full_document_prompt + TeacherTarget.dataset + document_key for the ID tag.
- [ ] If synth is a subset of the unit, teacher looks up the complete unit from
      the catalog, not from the row's system_prompt.
- [ ] Pin teacher prompt bytes (hash or token count) if AM numbers must stay comparable.
- [ ] Export phasek.txt via the same to_string() used for token counts.
```

Do **not** invent a split that confounds forgetting (same story, different
question writers; same company, different years with unnamed years).

## Launch (existing)

```bash
cd trivo-explore-research-work
export CARTRIDGES_DIR=$PWD CARTRIDGES_OUTPUT_DIR=$PWD/outputs PYTHONPATH=$PWD WANDB_MODE=offline
bash examples/quality/synthesize/self_study.sh          # or finqa / techqa
# reuse server:
SKIP_VLLM=1 bash examples/finqa/synthesize/self_study.sh
PHASES="2 3 4 5" bash examples/techqa/synthesize/self_study.sh
```

Python:

```bash
python examples/shared/synth/self_study_vllm.py \
  --dataset quality --phase 1 --num-samples 8192 --batch-size 32 --max-num-batches 8
```

## Additional resources

- Per-dataset why/rejected alternatives/AM gaps: [datasets.md](datasets.md)
- Token tables and LPT worked examples: `cartridges/data/PHASES.md` (split rationale; “Still outstanding” is stale)
