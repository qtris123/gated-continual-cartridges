"""Convert plasticity/forgetting eval JSON into parquet files.

Supports two experiment types:
  A → delta_A  (temporal update):  amd_eval_questions.json
  A → B        (domain shift):     amd_pepsi_eval_questions.json

JSON structure:
  - "doc_a": name of the initial document (phase 1 training source)
  - "doc_b": name of the new document (phase 2 training source)
  - "phase_1": categories of questions about doc_a only (forgetting eval)
  - "phase_2": categories of questions about doc_a + doc_b (plasticity eval)

The cartridge IS the document context — no system prompt is embedded in the eval.
The model answers questions using only the KV cache from the cartridge.

Workflow:
    python experiments/evaluate/convert_eval.py \\
        --json data/eval_data/amd_eval_questions.json \\
        --phase both

Output (default):
    <json-parent-parent>/eval/<json-stem>_phase1.parquet
    <json-parent-parent>/eval/<json-stem>_phase2.parquet
"""

import argparse
import json
import sys
from pathlib import Path

from cartridges.structs import Conversation, write_conversations, read_conversations


def make_conversation(
    question: str,
    answer: str,
    category: str,
    question_id: str,
    difficulty: str,
    reasoning_type: str,
    phase: int,
) -> Conversation:
    return Conversation(
        system_prompt="",
        messages=[
            Conversation.Message(
                role="user",
                content=question,
                token_ids=None,
                top_logprobs=None,
            ),
            Conversation.Message(
                role="assistant",
                content=answer,
                token_ids=None,
                top_logprobs=None,
            ),
        ],
        metadata={
            "category": category,
            "question_id": question_id,
            "difficulty": difficulty,
            "reasoning_type": reasoning_type,
            "phase": phase,
        },
        type="continual_eval",
    )


def build_conversations(
    phase_data: dict,
    phase: int,
) -> tuple[list[Conversation], dict[str, int]]:
    """Build conversations from a phase dict (categories → list of questions)."""
    convos: list[Conversation] = []
    category_counts: dict[str, int] = {}

    for category, questions in phase_data.items():
        if not isinstance(questions, list):
            continue
        for q in questions:
            convos.append(make_conversation(
                question=q["question"],
                answer=q["golden_answer"],
                category=category,
                question_id=q["id"],
                difficulty=q.get("difficulty", "medium"),
                reasoning_type=q.get("reasoning_type", "factual"),
                phase=phase,
            ))
        category_counts[category] = len(questions)

    return convos, category_counts


def convert_phase(
    phase: int,
    data: dict,
    json_path: Path,
    output: str | None,
) -> Path | None:
    """Convert one phase from the JSON to parquet. Returns output path or None if skipped."""
    phase_key = f"phase_{phase}"
    if phase_key not in data:
        print(f"  [phase {phase}] Key '{phase_key}' not found in JSON — skipping.")
        return None

    print(f"\n=== Phase {phase} ({'forgetting' if phase == 1 else 'plasticity'}) ===")

    convos, category_counts = build_conversations(data[phase_key], phase)
    total = sum(category_counts.values())
    print(f"  {total} questions across {len(category_counts)} categories:")
    for cat, count in category_counts.items():
        print(f"    {cat}: {count}")

    if output:
        out_path = Path(output).resolve()
    else:
        out_path = json_path.parent.parent / "eval" / f"{json_path.stem}_phase{phase}.parquet"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_conversations(convos, str(out_path))
    print(f"  → wrote {len(convos)} conversations to {out_path}")

    # Verification
    loaded = read_conversations(str(out_path))
    assert len(loaded) == total, f"Count mismatch: {len(loaded)} != {total}"
    for convo in loaded:
        assert convo.type == "continual_eval"
        assert convo.metadata["phase"] == phase
        assert convo.system_prompt == "", "system_prompt should be empty"
    print(f"  verification passed.")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Convert plasticity/forgetting eval JSON to parquet")
    parser.add_argument(
        "--json",
        required=True,
        help="Path to the eval questions JSON file",
    )
    parser.add_argument(
        "--phase",
        choices=["1", "2", "both"],
        default="both",
        help="Which phase to convert: 1 (forgetting), 2 (plasticity), or both (default)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output parquet path (only valid when --phase is 1 or 2). "
            "Defaults to <json-parent-parent>/eval/<json-stem>_phase{N}.parquet."
        ),
    )
    args = parser.parse_args()

    if args.output and args.phase == "both":
        print("ERROR: --output cannot be used with --phase both (two files would be produced).")
        sys.exit(1)

    json_path = Path(args.json).resolve()

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    phases = [1, 2] if args.phase == "both" else [int(args.phase)]
    for phase in phases:
        convert_phase(phase, data, json_path, args.output if args.phase != "both" else None)

    print("\nDone.")


if __name__ == "__main__":
    main()
