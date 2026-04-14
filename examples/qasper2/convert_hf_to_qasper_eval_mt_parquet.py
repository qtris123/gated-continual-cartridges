#!/usr/bin/env python3
"""
Convert Hugging Face QASPER-style rows into the same parquet layout as
`examples/qasper2/qasper_eval_MT.parquet`: columns `messages`, `system_prompt`,
`metadata`, `type`.

Example:

  python examples/qasper2/convert_hf_to_qasper_eval_mt_parquet.py \\
    --output outputs/qasper_rewrite_mt.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

# Matches the user-turn template in `qasper_eval_MT.parquet`.
USER_MESSAGE_TEMPLATE = """Please write a succinct answer to the following question.
You do not need to restate the paper name or answer in complete sentences.

<question>
{question}
</question>

Provide your answer in the following format (output nothing else):

<answer>
{{your answer here}}
</answer>"""


def build_messages(question: str, answer: str) -> np.ndarray:
    user_content = USER_MESSAGE_TEMPLATE.format(question=question)
    answer = answer if isinstance(answer, str) else str(answer)
    assistant_content = f"<answer>\n{answer}\n</answer>"
    msgs = [
        {
            "content": user_content,
            "role": "user",
            "token_ids": None,
            "top_logprobs": None,
        },
        {
            "content": assistant_content,
            "role": "assistant",
            "token_ids": None,
            "top_logprobs": None,
        },
    ]
    return np.array(msgs, dtype=object)


def hf_split_to_dataframe(split) -> pd.DataFrame:
    rows = []
    for i in range(len(split)):
        r = split[i]
        rows.append(
            {
                "messages": build_messages(r["question"], r["answer"]),
                "system_prompt": "",
                "metadata": {
                    "abstract": r["abstract"],
                    "paper_id": r["paper_id"],
                    "title": r["title"],
                },
                "type": None,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HF QASPER rewrite dataset → qasper_eval_MT-style parquet"
    )
    parser.add_argument(
        "--hf-repo",
        default="qtris123/qtris123qasper-rewrite-gpt-4.1-SA-task",
        help="Hugging Face dataset id",
    )
    parser.add_argument(
        "--split",
        default="question",
        help="Dataset split name (this repo uses split 'question')",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .parquet path",
    )
    args = parser.parse_args()

    ds = load_dataset(args.hf_repo)
    if args.split not in ds:
        raise SystemExit(
            f"Split {args.split!r} not found. Available: {list(ds.keys())}"
        )
    split = ds[args.split]
    required = {"paper_id", "title", "abstract", "question", "answer"}
    missing = required - set(split.column_names)
    if missing:
        raise SystemExit(f"Dataset missing columns: {sorted(missing)}")

    df = hf_split_to_dataframe(split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
