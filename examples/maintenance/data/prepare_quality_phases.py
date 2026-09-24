#!/usr/bin/env python3
"""Materialize QuALITY 5-phase eval parquets and corpora.

Prompt format matches the LongHealth no-CoT eval style:
  - Question stem
  - Lettered options (a)-(d)
  - Explicit instruction: "Output only the letter of the correct option"
  - reference_answer = full option text (mc_options scorer resolves letter → text)

Phases are defined by PHASE_TO_ARTICLE_IDS in cartridges/data/quality/resources.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent.parent.parent

import sys
sys.path.insert(0, str(ROOT))

from cartridges.data.quality.resources import (
    PHASE_TO_ARTICLE_IDS,
    QUALITY_DATASET,
    QUALITY_SPLITS,
    load_articles,
)

OPTION_LETTERS = ["a", "b", "c", "d"]


def load_questions_by_article() -> dict[str, list[dict]]:
    """Load all QuALITY questions from HF, keyed by article_id."""
    from datasets import load_dataset

    questions_by_article: dict[str, list[dict]] = {}
    for split in QUALITY_SPLITS:
        for row in load_dataset(QUALITY_DATASET, split=split):
            aid = str(row["article_id"])
            questions_by_article.setdefault(aid, []).append(row)
    return questions_by_article


def build_phase_evals() -> None:
    phases_dir = ROOT / "data/quality/phases"
    phases_dir.mkdir(parents=True, exist_ok=True)

    print("Loading QuALITY articles and questions from HF ...")
    articles = load_articles()
    questions_by_article = load_questions_by_article()
    print(f"  loaded {sum(len(v) for v in questions_by_article.values())} questions"
          f" across {len(questions_by_article)} articles")

    manifest: dict[str, dict] = {}

    for phase, article_ids in PHASE_TO_ARTICLE_IDS.items():
        phase_rows = []
        for aid in article_ids:
            article = articles.get(aid)
            story_info = (
                f"Title: {article.title}, Author: {article.author}"
                if article is not None
                else f"Article {aid}"
            )
            article_questions = questions_by_article.get(aid, [])
            for q in article_questions:
                options: list[str] = list(q["options"])  # 4 options
                raw_label = q.get("gold_label")
                if raw_label is None or raw_label == "":
                    raw_label = q.get("writer_label", 1)
                # QuALITY gold_label is 1-indexed (1 to 4)
                correct_idx: int = int(raw_label) - 1
                if not (0 <= correct_idx < len(options)):
                    correct_idx = 0
                correct_text: str = options[correct_idx]
                question_text: str = q["question"]
                question_id: str = str(q.get("question_unique_id") or q.get("question_id"))
                difficult: int = int(q.get("difficult", 0))
                question_type: str = str(q.get("question_type", "original"))

                # Build lettered options block, matching LongHealth style
                options_block = "\n".join(
                    f"({OPTION_LETTERS[i]}) {opt}"
                    for i, opt in enumerate(options)
                )
                user_content = (
                    f"Please answer the question below about the following story: {story_info}"
                    f"\n\n<question>\n{question_text}\n</question>"
                    f"\n\n<options>\n{options_block}\n</options>"
                    f"\nOutput only the letter of the correct option (e.g. (a), (b), (c), or (d)) along with the content of the option."
                )

                meta = {
                    "article_id": aid,
                    "question_id": question_id,
                    "category": "quality_mcq",
                    "options": options,
                    "correct": correct_text,
                    "difficult": difficult,
                    "question_type": question_type,
                    "doc_source": f"quality_phase{phase}",
                }

                phase_rows.append(
                    {
                        "messages": [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": correct_text},
                        ],
                        "system_prompt": "",
                        "metadata": meta,
                        "type": "quality_mcq",
                    }
                )

        eval_out = phases_dir / f"phase{phase}_eval.parquet"

        # Build schema matching generation_accuracy_matrix expectations
        schema = pa.schema([
            ("messages", pa.list_(pa.struct([
                ("role", pa.string()),
                ("content", pa.string()),
            ]))),
            ("system_prompt", pa.string()),
            ("metadata", pa.struct([
                ("article_id", pa.string()),
                ("question_id", pa.string()),
                ("category", pa.string()),
                ("options", pa.list_(pa.string())),
                ("correct", pa.string()),
                ("difficult", pa.int64()),
                ("question_type", pa.string()),
                ("doc_source", pa.string()),
            ])),
            ("type", pa.string()),
        ])

        table_out = pa.Table.from_pylist(phase_rows, schema=schema)
        pq.write_table(table_out, eval_out)
        print(f"  wrote {eval_out.name}: {len(phase_rows)} questions"
              f" (articles: {article_ids})")

        manifest[str(phase)] = {
            "eval": str(eval_out.relative_to(ROOT)),
            "questions": len(phase_rows),
            "articles": article_ids,
        }

    manifest_path = phases_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  wrote {manifest_path.relative_to(ROOT)}")


def main():
    print("=== Building 5-phase QuALITY evals ===")
    build_phase_evals()
    print("=== QuALITY phase eval preparation complete ===")


if __name__ == "__main__":
    main()
