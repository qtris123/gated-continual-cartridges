# Dataset train/eval distribution audit

Reference for **why a given dataset does or does not produce a meaningful
continual-learning signal**. Read this **before** interpreting any cross-dataset
comparison (QASPER vs LongHealth vs MTOB ...) — large eval-gap differences are
usually dataset-design artifacts, not method-level effects.

Companion to `WANDB_REFERENCE.md` (which lists the *symptom*: baseline P1-task
vs P2-task ppl gaps). This file documents the *mechanism*.

---

## TL;DR — what determines train/eval alignment

For each dataset, three things drive whether perplexity/accuracy improvements
generalise from synth-trained cartridge → held-out eval:

1. **Identity overlap** — is there a token-level *key* (paper title, patient id,
   diagnosis) that appears identically in synth system-prompt and eval user-prompt?
   If yes, eval becomes a retrieval task on a known key. If no, the cartridge has
   to recall content without a hook.
2. **Answer-space constraint** — is the eval open-ended generation (perplexity
   over arbitrary string) or closed-vocab (MCQ over k provided options + fuzzy
   scoring)? Closed-vocab + fuzzy scoring inflates apparent accuracy substantially.
3. **Format and writer-style overlap** — synth `(question, answer)` pairs are
   produced by Qwen3-4B at temp ~0.6 / 0.0 via `SelfStudySynthesizer`. Eval
   answers may be written by GPT-4.1 (QASPER) or be human-curated benchmark
   strings (LongHealth). The closer the eval format to what Qwen-4B naturally
   emits during synth, the lower the surface-distribution gap.

---

## QASPER

### Source corpus
- `allenai/qasper` (train split), filtered to 16 hand-picked arXiv IDs per topic.
- Topics: `QA`, `MT`, `SA` — see `TOPIC_TO_IDS` in
  `cartridges/data/qasper/resources.py:11`.

### Eval pipeline (`cartridges/data/qasper/`)
- **Question rewriting** (`rewrite.py`):
  - Filter to topic's paper IDs, drop `unanswerable=True`.
  - GPT-4.1 `temperature=0.0` rewrites each human QASPER question to be
    self-contained / closed-book, conditioned **only on the paper title**
    (`QUESTION_PROMPT`, lines 45–57).
  - GPT-4.1 also rewrites the gold answer to a succinct closed-book answer using
    the original `answer_data` (`ANSWER_PROMPT`, lines 32–43).
  - Pushed to HF as `sabrieyuboglu/qasper-rewrite-gpt-4.1` (QA),
    `qtris123/...-MT-task`, `qtris123/...-SA-task`. Split name: `"question"`.
- **Eval dataset** (`evals.py::QasperEvalDataset` + `convert_hf_to_qasper_eval_mt_parquet.py`):
  - Each row becomes a 2-message conversation:
    - user: `Please write a succinct answer ... <question>...</question> ... <answer>{...}</answer>`
    - assistant: `<answer>\n{rewritten answer}\n</answer>`
  - `system_prompt=""` — no paper text, no identity cue in the eval prompt.
  - Perplexity measured over the assistant tokens. Open-ended; no MCQ.
  - Baked to `examples/qasper2/qasper_eval_{QA,MT}.parquet`.
- **Generation eval** (`cartridges/benchmark/datasets.py::_load_qasper`):
  - Same HF dataset, same template, scored by exact / chrF against gold string.

### Synth pipeline (`examples/qasper2/synthesize/self_study.py`)
- `QASPERResource.sample_prompt` picks **one paper** from `TOPIC_TO_IDS[topic]`,
  picks a **random subset of sections** (`num_sections_per_paper = random.randint(1, len(paper.sections))`),
  wraps in `SYSTEM_PROMPT_TEMPLATE`:
  ```
  Below is (part of) a scientific paper. Please read it and be prepared to answer questions.
  <paper>{paper_template_with_sections}</paper>
  ```
- 5 seed-prompt types mixed uniformly: `structuring / summarization / question /
  use_case / creative` (see `examples/qasper2/synthesize/self_study.py:87` and
  `cartridges/data/resources.py` for prompt strings).
- `SelfStudySynthesizer`: Qwen3-4B, bot A (`user`) temp=0.6, bot B (`assistant`)
  temp=0.0, `prob_thinking=0.2`. Bot B sees the paper text in system prompt at
  generation time → answers are **open-book Qwen**, then replayed without the
  paper at training time.

### Distribution gap summary

| Axis | Synth (train) | Eval | Aligned? |
|---|---|---|---|
| Source corpus | Same 16 papers | Same 16 papers | ✓ |
| Question author | Qwen3-4B (temp 0.6) | Human → GPT-4.1 rewrite | ✗ |
| Answer author | Qwen3-4B (open-book) | GPT-4.1 (succinct, gold-conditioned) | ✗ |
| Answer length | Long, paraphrased, sometimes CoT | Sentence-fragment, factual | ✗ |
| `<answer>...</answer>` wrapper | **Never seen in synth** | Required at eval | ✗ |
| Identity cue in eval prompt | n/a | None (just rewritten question) | ✗ |
| Synth seed prompts vs eval style | 5 styles, only `question` matches QA | Single QA template | partial |
| Section coverage | Uniform over all paper sections | Abstract-anchored (human annotators wrote QAs from abstract) | partial |

**Verdict**: train and eval share *the underlying paper text* but very little
else. QASPER eval is **near-OOD** relative to synth. Improvements in
`eval_qasper_perplexity/loss` reflect a combination of (a) actual content recall
and (b) format/style adaptation (`<answer>` wrapping, GPT-4.1-style succinct
phrasing). A format-mismatch-only improvement on the `<answer>` wrapper would
look like a real ppl drop with no content gain.

---

## LongHealth

### Source corpus
- `benchmark_v5.json` from `kbressem/LongHealth` GitHub
  (`cartridges/data/longhealth/utils.py:37`).
- 20 patients total; standard split:
  - Phase 1 / "p1-10": `patient_01 ... patient_10`
  - Phase 2 / "p11-20": `patient_11 ... patient_20`
- Each `LongHealthPatient` has `(name, birthday, diagnosis, texts: Dict[note_id, str], questions: List[LongHealthQuestion])`.
- Questions are **human-written, 5-way MCQ**, pre-existing in the benchmark JSON.

### Eval pipeline (`cartridges/data/longhealth/evals.py`)
- **MCQ generation eval** (`LongHealthMultipleChoiceGenerateDataset`,
  `evals.py:13`):
  - Each question wrapped via `wrap_question`:
    ```
    Please answer the question below about the following patient:
    ID patient_01, Name: Anna Sample, Birthday: 1970-01-01 00:00:00, Diagnosis: DLBCL

    <question>...</question>
    <options>a\nb\nc\nd\ne</options>
    {cot_prompt}
    <answer>{YOUR_ANSWER}</answer>
    ```
  - `cot=True` by default → CoT prompt asks for `<thinking>...</thinking>` then
    `<answer>...</answer>`.
  - `include_diagnosis=True` by default → diagnosis is in the prompt.
  - Scored by `score()` (`evals.py:117`): regex-extract `<answer>` body, run
    `SequenceMatcher` against the 5 options, **closest-match wins** — tolerant
    of paraphrase and partial answers.
  - Reports `generate_<dataset>_accuracy/score`.
- **Perplexity eval parquet** (`data/longhealth/eval/patients_{01_to_10,11_to_20}.parquet`):
  - 200 rows each. Same prompt structure as MCQ eval.
  - Assistant gold = exactly one option string wrapped in `<answer>...</answer>`
    (e.g. `<answer>\nVincristine\n</answer>`).
  - `system_prompt=""`. Identity tuple lives in the user message preamble.

### Synth pipeline (`examples/longhealth/synthesize/self_study.py`)
- `LongHealthResource.sample_prompt` (`resources.py:54`): pick a patient,
  pick `randint(min, max)` notes (default `[1, 5]`), wrap in
  `SYSTEM_PROMPT_TEMPLATE`:
  ```
  Below is a section of {name}'s medical record (ID: {patient_id}).
  They were born on {birthday} and have the following diagnosis: {diagnosis}.
  The patients medical record consists of {num_notes} notes.
  {notes}
  ```
- Default `_DEFAULT_PATIENTS = patient_01..patient_10` (matches Phase-1 eval split).
- Same 5 seed prompts as QASPER. Same Qwen3-4B + `SelfStudySynthesizer`.

### Distribution gap summary

| Axis | Synth (train) | Eval | Aligned? |
|---|---|---|---|
| Source corpus | Same patient set | Same patient set | ✓ |
| **Identity tuple `(patient_id, name, birthday, diagnosis)`** | In synth system-prompt verbatim | In eval user-prompt verbatim | **✓ — identical strings** |
| Question author | Qwen3-4B | Human (LongHealth authors) | ✗ |
| Answer author | Qwen3-4B (open-book) | Human-curated correct option | ✗ |
| Answer space | Open vocab | **Closed: 1 of 5 provided options** | trivially small |
| Scoring tolerance | n/a | `SequenceMatcher`-best-of-5 | very tolerant |
| Note ↔ question alignment | Synth samples 1–5 notes per call; questions target single facts within single notes | MCQ targets facts in specific note(s) | ✓ |

**Verdict**: train and eval are **strongly aligned** on three axes that QASPER
lacks:
1. Verbatim identity-tuple match → eval becomes "lookup by patient key".
2. Closed answer space + fuzzy scoring → format/paraphrase mismatch can't hurt.
3. Atomic-fact granularity matches note granularity → seed prompts naturally
   produce the kind of `(Q, atomic-fact-A)` pairs MCQs reward.

---

## Cross-dataset comparison (the actual question to ask before interpreting)

| Property | QASPER | LongHealth |
|---|---|---|
| Eval task | open-ended short-answer (ppl over free text) | **5-way MCQ** + ppl over 1-of-5 string |
| Eval answer length | sentence-fragment | 1–4 tokens (drug/lab/date) |
| Identity cue in eval prompt | None | Patient tuple verbatim |
| Format wrapper in synth | None | None |
| Format wrapper in eval | `<answer>...</answer>` (unseen in synth) | `<answer>...</answer>` (unseen in synth) |
| Scoring tolerance | exact ppl | fuzzy MCQ best-of-5 |
| Synth ↔ eval surface gap | **near-OOD** | **near-IID once retrieval key is hit** |
| Baseline P1-task vs P2-task ppl gap (recorded in `WANDB_REFERENCE.md`) | 25.59 | 0.116 |
| Forgetting/learning detectable? | **Yes** (multiple-ppl headroom) | **No** (sub-1-ppl gap → metric floor) |

### Practical implications when you see "method X works better on LongHealth"

Be suspicious. The dominant alternative explanation is that LongHealth's eval
is **intrinsically a much easier and much more train-aligned task** than
QASPER's eval:

- ~5× narrower hypothesis space at every answer slot (1 of 5 vs open vocab).
- A retrieval key (`patient_id`+`diagnosis`) is provided at eval time.
- Fuzzy scoring forgives format/paraphrase drift.
- Note-level facts match the synth-sampling granularity.

So a method that "preserves LongHealth eval better" may just be preserving the
identity-tuple → note mapping, not preserving knowledge in any deep sense.
Conversely, the same method "looking worse on QASPER" may be the QASPER eval
*finally* being sensitive enough to detect a real effect (multi-ppl headroom).

### Probes that disambiguate

Before crediting (or blaming) a sparse-finetuning recipe for a cross-dataset
delta, run one or more of these eval-only diagnostics:

1. **LongHealth identity-ablation eval**: rerun `LongHealthMultipleChoiceGenerateDataset`
   with `include_diagnosis=False`. If accuracy collapses, the eval was a
   retrieval task, not a knowledge task — and the method-level interpretation
   was overstated.
2. **LongHealth options-ablation**: run with shuffled / paraphrased option
   strings to test sensitivity to the closed-vocab cheat. If accuracy is
   unchanged, the cartridge is matching content (good). If it drops, MCQ
   closedness was doing the work.
3. **QASPER format probe**: synthesize a separate training mix using only
   `seed_prompts=["question"]` and force the bot-B response template to
   `<answer>\n...\n</answer>`. If this alone closes a meaningful fraction of
   the QASPER ppl gap, the apparent improvement from a sparse recipe was
   partly format adaptation.
4. **Baseline gap check** (cheap and immediate): from
   `WANDB_REFERENCE.md::Reference baseline gaps`. If `P2-task ppl - P1-task ppl
   < ~1 ppl`, the metric cannot show forgetting. Stop interpreting before
   redesigning the eval.

---

## Quick lookup: where to find each pipeline component

| Concern | File | Key symbol(s) |
|---|---|---|
| QASPER paper IDs per topic | `cartridges/data/qasper/resources.py` | `TOPIC_TO_IDS` |
| QASPER eval-question rewriting | `cartridges/data/qasper/rewrite.py` | `QUESTION_PROMPT`, `ANSWER_PROMPT`, `RewrittenQasperQuestion` |
| QASPER eval-perplexity dataset | `cartridges/data/qasper/evals.py` | `QasperEvalDataset`, `PROMPT` |
| QASPER eval parquet builder | `examples/qasper2/convert_hf_to_qasper_eval_mt_parquet.py` | `build_messages`, `hf_split_to_dataframe` |
| QASPER generation eval | `cartridges/benchmark/datasets.py` | `_load_qasper`, `_QASPER_PROMPT` |
| QASPER synth resource | `cartridges/data/qasper/resources.py` | `QASPERResource`, `SYSTEM_PROMPT_TEMPLATE` |
| QASPER synth driver | `examples/qasper2/synthesize/self_study.py` | seed prompts list (line 87) |
| LongHealth source loader | `cartridges/data/longhealth/utils.py` | `load_longhealth_dataset`, `LongHealthQuestion`, `LongHealthPatient` |
| LongHealth eval (MCQ) | `cartridges/data/longhealth/evals.py` | `LongHealthMultipleChoiceGenerateDataset`, `wrap_question`, `score` |
| LongHealth eval parquets | `data/longhealth/eval/patients_{01_to_10,11_to_20}.parquet` | 200 rows each |
| LongHealth synth resource | `cartridges/data/longhealth/resources.py` | `LongHealthResource`, `SYSTEM_PROMPT_TEMPLATE` |
| LongHealth synth driver | `examples/longhealth/synthesize/self_study.py` | `_DEFAULT_PATIENTS`, `_DEFAULT_SEED_PROMPTS` |
| Shared self-study synthesizer | `cartridges/synthesizers/self_study.py` | `SelfStudySynthesizer`, `SYSTEM_PROMPT_TEMPLATE` |
| Shared seed prompts | `cartridges/data/resources.py` | `SEED_PROMPT_REGISTRY`, `question_seed_prompt`, etc. |
