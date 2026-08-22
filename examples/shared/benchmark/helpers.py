import csv
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------

def _strip_header(fieldnames: Optional[List[str]]) -> List[str]:
    return [h.strip() for h in (fieldnames or [])]


def _normalize_csv_row(row: dict) -> dict:
    return {(k.strip() if k else k): v for k, v in row.items()}


def _detect_csv_kind(fieldnames: List[str]) -> str:
    s = set(fieldnames)
    if "yes_no_question" in s:
        return "yes_no"
    if "mcq_question" in s and "option_a" in s:
        return "mcq"
    raise ValueError(
        f"Unrecognized CSV schema (expected Qasper yes/no or MCQ columns). "
        f"Got columns: {sorted(s)}"
    )


def _load_yes_no_csv(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = _strip_header(reader.fieldnames)
        if _detect_csv_kind(fieldnames) != "yes_no":
            raise ValueError(f"Expected yes/no CSV at {path}")
        for idx, raw in enumerate(reader):
            row = _normalize_csv_row(raw)
            qid = str(row.get("id", idx)).strip()
            yn = (row.get("yes_no_question") or "").strip()
            ref = (row.get("answer") or "").strip()
            out.append({
                "id": qid,
                "question": yn,
                "correct": ref,
                "question_type": "yes_no",
                "original_question": (row.get("original_question") or "").strip(),
                "original_answer": (row.get("original_answer") or "").strip(),
                "yes_no_question": yn,
            })
    return out


def _load_mcq_csv(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = _strip_header(reader.fieldnames)
        if _detect_csv_kind(fieldnames) != "mcq":
            raise ValueError(f"Expected MCQ CSV at {path}")
        for idx, raw in enumerate(reader):
            row = _normalize_csv_row(raw)
            qid = str(row.get("id", idx)).strip()
            mq = (row.get("mcq_question") or "").strip()
            oa = (row.get("option_a") or "").strip()
            ob = (row.get("option_b") or "").strip()
            oc = (row.get("option_c") or "").strip()
            od = (row.get("option_d") or "").strip()
            letter = (row.get("answer") or "").strip().upper()[:1]
            if letter not in ("A", "B", "C", "D"):
                raise ValueError(
                    f"Row {idx}: invalid MCQ answer {row.get('answer')!r}, expected A-D"
                )
            opt_map = {"A": oa, "B": ob, "C": oc, "D": od}
            out.append({
                "id": qid,
                "mcq_question": mq,
                "correct": opt_map[letter],
                "correct_letter": letter,
                "question_type": "mcq",
                "option_a": oa,
                "option_b": ob,
                "option_c": oc,
                "option_d": od,
                "original_question": (row.get("original_question") or "").strip(),
                "original_answer": (row.get("original_answer") or "").strip(),
            })
    return out


def load_eval_records(eval_path: Path, dataset_type: str) -> List[dict]:
    """Load evaluation records from a CSV file.

    ``dataset_type`` is ``"mcq"`` or ``"yes_no"``.  If ``"auto"``, the type is
    inferred from the CSV header.
    """
    eval_path = eval_path.expanduser()
    if not eval_path.exists():
        raise FileNotFoundError(str(eval_path))

    if dataset_type == "auto":
        with eval_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = _strip_header(reader.fieldnames)
        dataset_type = _detect_csv_kind(fieldnames)

    if dataset_type == "yes_no":
        return _load_yes_no_csv(eval_path)
    if dataset_type == "mcq":
        return _load_mcq_csv(eval_path)
    raise ValueError(f"Unknown dataset_type={dataset_type!r}; expected 'mcq' or 'yes_no'")



# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def infer_question_type(record: dict) -> str:
    if record.get("mcq_question") and str(record["mcq_question"]).strip():
        return "mcq"
    if record.get("yes_no_question") and str(record["yes_no_question"]).strip():
        return "yes_no"
    raise ValueError(f"Cannot infer question type from record: {record}")

def option_letters_and_texts(record: dict) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for L in ("a", "b", "c", "d"):
        t = record.get(f"option_{L}")
        if t:
            out.append((L.upper(), t))
    return out

def reference_answer_letter(record: dict) -> Optional[str]:
    explicit = str(record.get("correct_letter") or "").strip().upper()
    if explicit in {"A", "B", "C", "D"}:
        return explicit
    correct = str(record.get("correct", "")).strip().lower()
    for letter, text in option_letters_and_texts(record):
        if str(text).strip().lower() == correct:
            return letter.upper()
    return None
