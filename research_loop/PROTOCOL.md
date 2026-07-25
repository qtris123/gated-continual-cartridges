# The Qasper 2-stage continual-training protocol

There is no single prose spec in the repo; this file consolidates the authoritative definition
from the script docstrings (`examples/qasper2/train/{initial,continual,baseline_continual,
continual_am_sparse}.py`), the data/eval wiring, and `notes/2026-06-20-qasper2-longhealth-port.md`.
**Human: please confirm the two ⚠️ points below.**

## Shared setup (fixed)
- Frozen LM: **Qwen3-4B** (`FlexQwen3ForCausalLM`, `Qwen/Qwen3-4B-Instruct-2507`).
- Cartridge: fixed-size trainable KV cache, **512 slots** (`NUM_TOKENS=512`), `num_frozen_tokens=1`.
- Optimizer (gradient variants): adam, `LR=2e-2`, `EPOCHS=10`, `GLOBAL_BATCH_SIZE=32`.
- Eval = mean cross-entropy ("Eval loss") on held-out splits; wandb key `qasper_perplexity`
  (use `.../loss`, NOT the broken `.../perplexity` field).

## Stage 1 — Phase 1: build the base cartridge by self-distillation
- Init the 512-slot cache from text (`KVFromText`, `data/qasper/init_text/qasper_init_512.txt`).
- **Dense gradient context-distillation** on the QA-task self-study data
  `data/qasper/train/qwen_qasper_QA_task_8192.parquet` (`train/initial.py`, or `initial_sparse.py`
  which additionally collects `bg_stats.pt` for TF-IDF/IDF).
- Output: `cache_last.pt` = **the self-distilled cartridge**. This is the human-provided HF artifact
  `qtris123/qwen_qasper-QA-task_8192_512_no-cartridge_10-epochs`.  ⚠️ its bundled config.yaml is
  MT-labelled — verify (RUNBOOK §8b) before use.
- Reference eval: `qasper_eval_QA.parquet` → the QA retention floor.
- "no-cartridge" in the name = the QA training data was synthesized WITHOUT a cartridge in context
  (teacher = full-context ICL), i.e. standard self-study.

## Stage 2 — Phase 2: continually add the new task WITHOUT forgetting
- **Load the SAME Phase-1 `cache_last.pt`** and keep updating the SAME fixed-size cache on the
  Phase-2 task data `data/qasper/train/qwen_qasper_MT_task_8192.parquet`. The cache never grows.
- Three method families, all starting from the identical Phase-1 cache — this is the comparison:
  | family | script | nature |
  |---|---|---|
  | **self-distillation baseline (TARGET)** | `baseline_continual.py` | dense gradient, all slots update |
  | TF-IDF sparse | `continual_sparse.py` | gradient, only TF-IDF-selected slots update |
  | **AM-sparse (METHOD UNDER STUDY)** | `continual_am_sparse.py` | backprop-free closed-form value solve on selected slots |
- Eval on **BOTH** splits after Stage 2:
  - `qasper_eval_QA.parquet` → **forgetting / retention** (did Phase-1 knowledge survive?)
  - `qasper_eval_MT.parquet` → **acquisition** (did the new task get learned?)

## What "match self-distillation" means (the loop's target)
After Stage 2, the AM-sparse cartridge's **(QA-loss, MT-loss)** pair should be ≤ the dense
`baseline_continual.py` cartridge's pair (within +0.15 on each axis) — same Phase-1 start, same
eval, only the Phase-2 update rule differs. Ideal region: QA near the Phase-1 floor (little
forgetting) AND MT pulled down from ~33 toward the baseline's acquisition level.

## Task domains = Qasper sub-tasks (confirmed by human)
The stages are different **Qasper task domains**, each a self-study corpus + eval split:
- Phase 1 = **QA** (Question Answering) — builds the base cartridge.
- Phase 2 = **MT** (Machine Translation) — the new domain written in continually.
- Phase 3 = **SA** (Sentiment Analysis) — a possible further continual step.
This autoresearch is scoped to **QA→MT (2-stage)**; QA→MT→SA is a future longer-chain extension
(see NORTH_STAR.md for scope). QA-eval after Stage 2 = forgetting; MT-eval = acquisition.

## Cache provenance
Human believes the HF cache is **mislabelled** (name QA vs config MT). Being re-verified by a
dual-split eval (EXP-000-verify): a real QA Phase-1 cache → low QA (~5-8) / high MT (~30-40).
