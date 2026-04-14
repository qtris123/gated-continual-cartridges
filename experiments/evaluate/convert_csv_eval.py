"""Convert CSV eval files (MCQ and Yes-No) into JSON and Parquet.

Reads CSVs from data/financebench/eval_csv_file/ and produces:
  - Per-CSV: JSON + Parquet for MCQ and Yes-No question types
  - Per-document: JSON + Parquet for deduplicated original open-ended Q&A

Output:
  JSON    → data/financebench/eval_data_json/{stem}.json
  Parquet → data/financebench/eval/{stem}.parquet

Usage:
    python experiments/evaluate/convert_csv_eval.py \
        --csv-dir data/financebench/eval_csv_file \
        --json-dir data/financebench/eval_data_json \
        --parquet-dir data/financebench/eval
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from cartridges.structs import Conversation, write_conversations, read_conversations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_question_type(df: pd.DataFrame) -> str:
    """Detect whether a CSV is MCQ or Yes-No based on columns."""
    if "mcq_question" in df.columns:
        return "mcq"
    elif "yes_no_question" in df.columns:
        return "yes_no"
    else:
        raise ValueError(f"Cannot detect question type. Columns: {list(df.columns)}")


def extract_doc_source(filename: str) -> str:
    """Extract document source from filename, e.g. 'amd_2021' from 'amd_2021_120_batch-mcq.csv'."""
    match = re.match(r"^([a-z]+_\d{4})", filename)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot extract doc_source from filename: {filename}")


def format_mcq_question(row: pd.Series) -> str:
    """Format MCQ question with options."""
    q = row["mcq_question"]
    options = f"\nA) {row['option_a']}\nB) {row['option_b']}\nC) {row['option_c']}\nD) {row['option_d']}"
    return q + options


def make_conversation(
    question: str,
    answer: str,
    metadata: dict,
) -> Conversation:
    return Conversation(
        system_prompt="",
        messages=[
            Conversation.Message(role="user", content=question, token_ids=None, top_logprobs=None),
            Conversation.Message(role="assistant", content=answer, token_ids=None, top_logprobs=None),
        ],
        metadata=metadata,
        type="continual_eval",
    )


# ---------------------------------------------------------------------------
# MCQ / Yes-No conversion
# ---------------------------------------------------------------------------

def convert_mcq_csv(df: pd.DataFrame, doc_source: str, filename: str):
    """Convert MCQ CSV to JSON dict and list of Conversations."""
    questions_by_category = defaultdict(list)
    conversations = []

    for _, row in df.iterrows():
        qid = row["id"]
        category = row["category"]
        options = {
            "A": row["option_a"],
            "B": row["option_b"],
            "C": row["option_c"],
            "D": row["option_d"],
        }

        # JSON entry
        questions_by_category[category].append({
            "id": qid,
            "question": row["mcq_question"],
            "golden_answer": row["answer"],
            "original_question": row["original_question"],
            "original_answer": row["original_answer"],
            "options": options,
        })

        # Conversation for parquet
        metadata = {
            "category": category,
            "question_id": qid,
            "question_type": "mcq",
            "doc_source": doc_source,
            "original_question": row["original_question"],
            "original_answer": row["original_answer"],
            "options": options,
        }
        conversations.append(make_conversation(
            question=format_mcq_question(row),
            answer=str(row["answer"]),
            metadata=metadata,
        ))

    json_data = {
        "description": f"MCQ eval questions from {filename}",
        "doc_source": doc_source,
        "question_type": "mcq",
        "questions": dict(questions_by_category),
    }
    return json_data, conversations


def convert_yesno_csv(df: pd.DataFrame, doc_source: str, filename: str):
    """Convert Yes-No CSV to JSON dict and list of Conversations."""
    questions_by_category = defaultdict(list)
    conversations = []

    for _, row in df.iterrows():
        qid = row["id"]
        category = row["category"]

        # JSON entry
        questions_by_category[category].append({
            "id": qid,
            "question": row["yes_no_question"],
            "golden_answer": row["answer"],
            "original_question": row["original_question"],
            "original_answer": row["original_answer"],
        })

        # Conversation for parquet
        metadata = {
            "category": category,
            "question_id": qid,
            "question_type": "yes_no",
            "doc_source": doc_source,
            "original_question": row["original_question"],
            "original_answer": row["original_answer"],
        }
        conversations.append(make_conversation(
            question=row["yes_no_question"],
            answer=str(row["answer"]),
            metadata=metadata,
        ))

    json_data = {
        "description": f"Yes-No eval questions from {filename}",
        "doc_source": doc_source,
        "question_type": "yes_no",
        "questions": dict(questions_by_category),
    }
    return json_data, conversations


# ---------------------------------------------------------------------------
# Original (open-ended) conversion
# ---------------------------------------------------------------------------

def convert_original_from_mcq(df: pd.DataFrame, doc_source: str):
    """Extract deduplicated original Q&A from an MCQ CSV."""
    seen = set()
    questions_by_category = defaultdict(list)
    conversations = []

    for _, row in df.iterrows():
        qid = row["id"]
        if qid in seen:
            continue
        seen.add(qid)

        category = row["category"]

        questions_by_category[category].append({
            "id": qid,
            "question": row["original_question"],
            "golden_answer": row["original_answer"],
        })

        metadata = {
            "category": category,
            "question_id": qid,
            "question_type": "original",
            "doc_source": doc_source,
        }
        conversations.append(make_conversation(
            question=row["original_question"],
            answer=str(row["original_answer"]),
            metadata=metadata,
        ))

    json_data = {
        "description": f"Original open-ended eval questions for {doc_source}",
        "doc_source": doc_source,
        "question_type": "original",
        "questions": dict(questions_by_category),
    }
    return json_data, conversations


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_parquet(conversations: list[Conversation], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_conversations(conversations, str(path))


def verify_parquet(path: Path, expected_count: int):
    loaded = read_conversations(str(path))
    assert len(loaded) == expected_count, f"Count mismatch in {path}: {len(loaded)} != {expected_count}"
    for convo in loaded:
        assert convo.type == "continual_eval"
        assert "question_type" in convo.metadata
        assert "doc_source" in convo.metadata
    return len(loaded)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert CSV eval files to JSON and Parquet")
    parser.add_argument(
        "--csv-dir",
        default="data/financebench/eval_csv_file",
        help="Directory containing CSV files",
    )
    parser.add_argument(
        "--json-dir",
        default="data/financebench/eval_data_json",
        help="Output directory for JSON files",
    )
    parser.add_argument(
        "--parquet-dir",
        default="data/financebench/eval",
        help="Output directory for Parquet files",
    )
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    json_dir = Path(args.json_dir)
    parquet_dir = Path(args.parquet_dir)

    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        return

    print(f"Found {len(csv_files)} CSV files in {csv_dir}\n")

    # Track MCQ CSVs per document for original Q&A extraction
    mcq_by_doc: dict[str, tuple[pd.DataFrame, str]] = {}

    # ---- Step 1: Convert each CSV to MCQ/Yes-No JSON + Parquet ----
    results = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        qtype = detect_question_type(df)
        doc_source = extract_doc_source(csv_path.name)
        stem = csv_path.stem

        print(f"Processing: {csv_path.name} ({qtype}, {len(df)} rows, doc={doc_source})")

        if qtype == "mcq":
            json_data, conversations = convert_mcq_csv(df, doc_source, csv_path.name)
            mcq_by_doc[doc_source] = (df, stem)
        else:
            json_data, conversations = convert_yesno_csv(df, doc_source, csv_path.name)

        json_path = json_dir / f"{stem}.json"
        parquet_path = parquet_dir / f"{stem}.parquet"

        write_json(json_data, json_path)
        write_parquet(conversations, parquet_path)
        verified = verify_parquet(parquet_path, len(conversations))

        results.append((stem, qtype, len(df), verified, json_path, parquet_path))
        print(f"  → {json_path}")
        print(f"  → {parquet_path} (verified: {verified} conversations)\n")

    # ---- Step 2: Create deduplicated original Q&A per document ----
    print("=" * 60)
    print("Creating deduplicated original Q&A files\n")

    for doc_source, (df, mcq_stem) in sorted(mcq_by_doc.items()):
        # Derive original stem from MCQ stem: replace type suffix with "original"
        # e.g. amd_2021_120_batch-mcq -> amd_2021_120_original
        original_stem = re.sub(r"_(batch|adapted)-mcq$", "_original", mcq_stem)

        print(f"Processing original Q&A for {doc_source} (from {mcq_stem})")

        json_data, conversations = convert_original_from_mcq(df, doc_source)

        json_path = json_dir / f"{original_stem}.json"
        parquet_path = parquet_dir / f"{original_stem}.parquet"

        write_json(json_data, json_path)
        write_parquet(conversations, parquet_path)
        verified = verify_parquet(parquet_path, len(conversations))

        results.append((original_stem, "original", len(conversations), verified, json_path, parquet_path))
        print(f"  → {json_path}")
        print(f"  → {parquet_path} (verified: {verified} conversations)\n")

    # ---- Summary ----
    print("=" * 60)
    print(f"Summary: {len(results)} file pairs created\n")
    print(f"{'Stem':<45} {'Type':<10} {'Rows':<6} {'OK':<4}")
    print("-" * 70)
    for stem, qtype, rows, verified, _, _ in results:
        print(f"{stem:<45} {qtype:<10} {rows:<6} {'✓' if rows == verified else '✗'}")

    print(f"\nDone. {len(results)} JSON + {len(results)} Parquet files written.")


if __name__ == "__main__":
    main()
