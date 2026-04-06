from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from cartridges.utils import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkItem:
    prompt: str
    ground_truth: str
    metadata: dict[str, Any] = field(default_factory=dict)


LoaderFn = Callable[..., list[BenchmarkItem]]
DATASET_REGISTRY: dict[str, LoaderFn] = {}


def register_dataset(name: str):
    def decorator(fn: LoaderFn) -> LoaderFn:
        DATASET_REGISTRY[name] = fn
        return fn
    return decorator


def load_dataset_items(
    dataset_name: str,
    *,
    dataset_path: Optional[str] = None,
    subset: Optional[str] = None,
    split: str = "test",
    num_few_shot: int = 0,
    prompt_template: Optional[str] = None,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> list[BenchmarkItem]:
    """Unified entry point: resolve a dataset name to a list of BenchmarkItems."""

    if dataset_name in DATASET_REGISTRY:
        items = DATASET_REGISTRY[dataset_name](
            dataset_path=dataset_path,
            subset=subset,
            split=split,
            num_few_shot=num_few_shot,
            prompt_template=prompt_template,
            seed=seed,
        )
    elif dataset_name == "local":
        items = _load_local(dataset_path, prompt_template=prompt_template)
    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_REGISTRY.keys()) + ['local']}"
        )

    if max_samples is not None and len(items) > max_samples:
        rng = random.Random(seed)
        items = rng.sample(items, max_samples)

    logger.info(f"Loaded {len(items)} benchmark items from '{dataset_name}'")
    return items


# ---------------------------------------------------------------------------
# Built-in loaders
# ---------------------------------------------------------------------------

@register_dataset("mmlu")
def _load_mmlu(
    *,
    dataset_path: Optional[str] = None,
    subset: Optional[str] = None,
    split: str = "test",
    num_few_shot: int = 5,
    prompt_template: Optional[str] = None,
    seed: int = 42,
) -> list[BenchmarkItem]:
    from datasets import load_dataset

    hf_name = dataset_path or "cais/mmlu"
    subsets = [subset] if subset else _mmlu_all_subjects(hf_name)

    items: list[BenchmarkItem] = []
    for subj in subsets:
        ds_test = load_dataset(hf_name, subj, split=split)

        few_shot_examples: list[dict] = []
        if num_few_shot > 0:
            try:
                ds_dev = load_dataset(hf_name, subj, split="dev")
                few_shot_examples = list(ds_dev.select(range(min(num_few_shot, len(ds_dev)))))
            except Exception:
                try:
                    ds_val = load_dataset(hf_name, subj, split="validation")
                    few_shot_examples = list(ds_val.select(range(min(num_few_shot, len(ds_val)))))
                except Exception:
                    logger.warning(f"No dev/validation split for {subj}, running zero-shot")

        for row in ds_test:
            prompt = _format_mmlu_prompt(
                row, subj, few_shot_examples, template=prompt_template
            )
            answer_idx = row["answer"]
            answer_letter = "ABCD"[answer_idx] if isinstance(answer_idx, int) else str(answer_idx)

            items.append(BenchmarkItem(
                prompt=prompt,
                ground_truth=answer_letter,
                metadata={"subject": subj, "answer_index": answer_idx},
            ))
    return items


_QASPER_PROMPT = """\
Please write a succinct answer to the following question.
You do not need to restate the paper name or answer in complete sentences.

<question>
{question}
</question>

Provide your answer in the following format (output nothing else):

<answer>
{{your answer here}}
</answer>"""


@register_dataset("qasper")
def _load_qasper(
    *,
    dataset_path: Optional[str] = None,
    subset: Optional[str] = None,
    split: str = "question",
    num_few_shot: int = 0,
    prompt_template: Optional[str] = None,
    seed: int = 42,
) -> list[BenchmarkItem]:
    """Load QASPER using the rewritten dataset from cartridges.data.qasper.evals.

    Uses the same HF dataset (rewritten questions via GPT-4.1) and prompt template
    as QasperEvalDataset.  Falls back to allenai/qasper raw format when the
    dataset does not contain the rewritten schema.
    """
    from datasets import load_dataset

    hf_name = dataset_path or "qtris123/qtris123qasper-rewrite-gpt-4.1-MT-task"
    hf_split = "question"
    template = prompt_template or _QASPER_PROMPT

   
    ds = load_dataset(hf_name, split=hf_split)

    items: list[BenchmarkItem] = []
    for row in ds:
        question_text = row["question"]
        answer_text = row["answer"]
        prompt = template.format(question=question_text)
        items.append(BenchmarkItem(
            prompt=prompt,
            ground_truth=answer_text,
            metadata={
                "paper_id": row.get("paper_id", ""),
                "title": row.get("title", ""),
                "abstract": row.get("abstract", ""),
            },
        ))
    return items


def _load_qasper_raw(
    *,
    dataset_path: str = "qtris123/qtris123qasper-rewrite-gpt-4.1-MT-task",
    split: str = "question",
    prompt_template: str,
) -> list[BenchmarkItem]:
    """Fallback loader for the raw allenai/qasper format.

    Note: allenai/qasper is a script-based dataset and requires
    ``datasets < 3.0``.  With newer versions of the library this
    fallback will fail gracefully.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset(dataset_path, split=split)
    except RuntimeError as e:
        if "scripts are no longer supported" in str(e):
            raise RuntimeError(
                f"Cannot load '{dataset_path}' because it uses a loading script "
                f"which is unsupported in datasets >= 3.0. Either use the rewritten "
                f"QASPER dataset (qtris123/qtris123qasper-rewrite-gpt-4.1-MT-task) "
                f"or downgrade: pip install 'datasets<3'."
            ) from e
        raise

    items: list[BenchmarkItem] = []
    for paper in ds:
        paper_id = paper.get("id", "")
        title = paper.get("title", "")
        qas = paper.get("qas", {})
        questions = qas.get("question", [])
        answers_list = qas.get("answers", [])

        for q_idx, question_text in enumerate(questions):
            if q_idx >= len(answers_list):
                continue
            answer_entries = answers_list[q_idx].get("answer", [])
            gt_texts = [
                a.get("free_form_answer", "") or a.get("extractive_spans", [""])[0]
                for a in answer_entries
                if (a.get("free_form_answer") or a.get("extractive_spans"))
                   and a.get("unanswerable", "no") != "yes"
            ]
            if not gt_texts:
                continue

            prompt = prompt_template.format(question=question_text)
            items.append(BenchmarkItem(
                prompt=prompt,
                ground_truth=gt_texts[0],
                metadata={
                    "paper_id": paper_id,
                    "title": title,
                    "all_ground_truths": gt_texts,
                },
            ))
    return items


@register_dataset("longhealth")
def _load_longhealth(
    *,
    dataset_path: Optional[str] = None,
    subset: Optional[str] = None,
    split: str = "test",
    num_few_shot: int = 0,
    prompt_template: Optional[str] = None,
    seed: int = 42,
) -> list[BenchmarkItem]:
    """Load LongHealth using the same logic as LongHealthMultipleChoiceGenerateDataset.

    ``subset`` is interpreted as a comma-separated list of patient IDs
    (e.g. "patient_01,patient_02").  If omitted, all patients are loaded.
    """
    from cartridges.data.longhealth.utils import load_longhealth_dataset

    patient_ids = subset.split(",") if subset else None
    patients = load_longhealth_dataset(patient_ids)

    cot_prompt = (
        "You should first think step by step. Then give your final answer "
        "exactly as it appears in the options. Your output should be in the "
        "following format: \n<thinking> {YOUR_THOUGHT_PROCESS} </thinking> "
    )

    items: list[BenchmarkItem] = []
    for patient in patients:
        patient_info = (
            f"ID {patient.patient_id}, Name: {patient.name}, "
            f"Birthday: {patient.birthday}, Diagnosis: {patient.diagnosis}"
        )
        for question in patient.questions:
            options_text = (
                f"{question.answer_a}\n"
                f"{question.answer_b}\n"
                f"{question.answer_c}\n"
                f"{question.answer_d}\n"
                f"{question.answer_e}"
            )

            if prompt_template is not None:
                formatted = prompt_template.format(
                    patient_info=patient_info,
                    question=question.question,
                    options=options_text,
                )
            else:
                formatted = (
                    "Please answer the question below about the following patient: "
                    f"{patient_info}"
                    f"\n\n<question>\n{question.question}\n</question>"
                    f"\n\n<options>\n{options_text}\n</options>\n{cot_prompt}"
                    f"\n\n<answer>\n{{YOUR_ANSWER}}\n</answer>"
                )

            items.append(BenchmarkItem(
                prompt=formatted,
                ground_truth=question.correct,
                metadata={
                    "question_id": question.question_id,
                    "patient_id": patient.patient_id,
                    "options": [
                        question.answer_a,
                        question.answer_b,
                        question.answer_c,
                        question.answer_d,
                        question.answer_e,
                    ],
                },
            ))

    rng = random.Random(seed)
    rng.shuffle(items)
    return items


@register_dataset("hellaswag")
def _load_hellaswag(
    *,
    dataset_path: Optional[str] = None,
    subset: Optional[str] = None,
    split: str = "validation",
    num_few_shot: int = 0,
    prompt_template: Optional[str] = None,
    seed: int = 42,
) -> list[BenchmarkItem]:
    from datasets import load_dataset

    hf_name = dataset_path or "Rowan/hellaswag"
    ds = load_dataset(hf_name, split=split)

    template = prompt_template or (
        "Pick the most plausible continuation of the following text.\n\n"
        "{context}\n\n"
        "Options:\n{options}\n\n"
        "Answer with only the letter (A, B, C, or D)."
    )

    items: list[BenchmarkItem] = []
    for row in ds:
        ctx = row.get("ctx", row.get("ctx_a", "") + " " + row.get("ctx_b", ""))
        endings = row["endings"]
        options_str = "\n".join(
            f"{chr(65 + i)}. {e}" for i, e in enumerate(endings)
        )
        prompt = template.format(context=ctx.strip(), options=options_str)
        label = int(row["label"])
        answer_letter = chr(65 + label)

        items.append(BenchmarkItem(
            prompt=prompt,
            ground_truth=answer_letter,
            metadata={"label_index": label, "activity_label": row.get("activity_label", "")},
        ))
    return items


@register_dataset("truthfulqa")
def _load_truthfulqa(
    *,
    dataset_path: Optional[str] = None,
    subset: Optional[str] = None,
    split: str = "validation",
    num_few_shot: int = 0,
    prompt_template: Optional[str] = None,
    seed: int = 42,
) -> list[BenchmarkItem]:
    from datasets import load_dataset

    hf_name = dataset_path or "truthfulqa/truthful_qa"
    cfg = subset or "multiple_choice"
    ds = load_dataset(hf_name, cfg, split=split)

    template = prompt_template or "Answer the following question truthfully.\n\n{question}"

    items: list[BenchmarkItem] = []
    for row in ds:
        question = row["question"]
        prompt = template.format(question=question)
        best_answer = row.get("best_answer", row.get("correct_answers", [""])[0] if "correct_answers" in row else "")
        items.append(BenchmarkItem(
            prompt=prompt,
            ground_truth=str(best_answer),
            metadata={"category": row.get("category", "")},
        ))
    return items


# ---------------------------------------------------------------------------
# Local file loader
# ---------------------------------------------------------------------------

def _load_local(
    path: Optional[str],
    *,
    prompt_template: Optional[str] = None,
) -> list[BenchmarkItem]:
    if path is None:
        raise ValueError("dataset_path is required when dataset_name='local'")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Local dataset not found: {path}")

    if p.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(p)
    elif p.suffix == ".json" or p.suffix == ".jsonl":
        import pandas as pd
        df = pd.read_json(p, lines=p.suffix == ".jsonl")
    elif p.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(p)
    else:
        raise ValueError(f"Unsupported local file format: {p.suffix}. Use csv, json, jsonl, or parquet.")

    q_col = _find_column(df, ["question", "prompt", "input", "text"])
    a_col = _find_column(df, ["answer", "ground_truth", "label", "target", "output"])

    if q_col is None:
        raise ValueError(f"Cannot find question column. Available: {list(df.columns)}")
    if a_col is None:
        raise ValueError(f"Cannot find answer column. Available: {list(df.columns)}")

    template = prompt_template or "{question}"

    items: list[BenchmarkItem] = []
    meta_cols = [c for c in df.columns if c not in (q_col, a_col)]
    for _, row in df.iterrows():
        prompt = template.format(question=str(row[q_col]))
        items.append(BenchmarkItem(
            prompt=prompt,
            ground_truth=str(row[a_col]),
            metadata={c: row[c] for c in meta_cols},
        ))
    return items


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_column(df, candidates: list[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    return None


def _format_mmlu_prompt(
    row: dict,
    subject: str,
    few_shot_examples: list[dict],
    *,
    template: Optional[str] = None,
) -> str:
    choices = row["choices"]
    letters = "ABCD"

    if template is not None:
        options_str = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
        return template.format(
            question=row["question"],
            options=options_str,
            subject=subject.replace("_", " "),
        )

    subject_display = subject.replace("_", " ")
    parts: list[str] = [
        f"The following are multiple choice questions (with answers) about {subject_display}.\n"
    ]

    for ex in few_shot_examples:
        ex_choices = ex["choices"]
        q_text = ex["question"]
        options = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(ex_choices))
        answer_idx = ex["answer"]
        a_letter = letters[answer_idx] if isinstance(answer_idx, int) else str(answer_idx)
        parts.append(f"{q_text}\n{options}\nAnswer: {a_letter}\n")

    options_str = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    parts.append(f"{row['question']}\n{options_str}\nAnswer:")

    return "\n".join(parts)


def _mmlu_all_subjects(hf_name: str) -> list[str]:
    from datasets import get_dataset_config_names
    configs = get_dataset_config_names(hf_name)
    return [c for c in configs if c not in ("all", "auxiliary_train")]
