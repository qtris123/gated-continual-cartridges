"""Convert plasticity eval JSON into eval_phase2.parquet.

Supports two experiment types:
  A → delta_A  (temporal update):  amd_eval_questions.json
  A → B        (domain shift):     amd_pepsi_eval_questions.json

Both JSON files must have a top-level "phase_2" key containing named
category arrays.  doc_a / doc_b at root select which documents to load;
defaults to AMD_2021_10K / AMD_2022_10K when absent.

Workflow:
    python experiments/experiments_2/convert_plasticity_eval.py \\
        --json experiments/experiments_2/data/eval_data/amd_eval_questions.json

Output:
    <json-parent-parent>/eval/<json-stem>_phase2.parquet
"""

import argparse
import json
import sys
from pathlib import Path

from cartridges.structs import Conversation, write_conversations, read_conversations


def load_document(name: str, doc_paths: dict) -> str:
    path = doc_paths[name]
    text = path.read_text(encoding="utf-8")
    return f'<document name="{name}">\n{text}\n</document>'


def make_conversation(
    question: str,
    answer: str,
    system_prompt: str,
    category: str,
    question_id: str,
    difficulty: str,
    reasoning_type: str,
) -> Conversation:
    return Conversation(
        system_prompt=system_prompt,
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
            "phase": 2,
        },
        type="continual_eval",
    )


def main():
    parser = argparse.ArgumentParser(description="Convert plasticity eval JSON to parquet")
    parser.add_argument(
        "--json",
        required=True,
        help="Path to the eval questions JSON file",
    )
    parser.add_argument(
        "--texts-dir",
        default=None,
        help=(
            "Directory containing document .txt files. "
            "Defaults to <json-parent-parent>/texts/ (sibling of eval_data/)."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output parquet path. Defaults to <json-parent-parent>/eval/<json-stem>_phase2.parquet "
            "(e.g. amd_eval_questions.json → .../eval/amd_eval_questions_phase2.parquet)"
        ),
    )
    args = parser.parse_args()

    json_path = Path(args.json).resolve()

    texts_dir = Path(args.texts_dir) if args.texts_dir else json_path.parent.parent / "texts"

    doc_paths = {
        "AMD_2021_10K":     texts_dir / "AMD_2021_10K.txt",
        "AMD_2022_10K":     texts_dir / "AMD_2022_10K.txt",
        "PEPSICO_2021_10K": texts_dir / "PEPSICO_2021_10K.txt",
    }

    # ------------------------------------------------------------------
    # Load JSON
    # ------------------------------------------------------------------
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if "phase_2" not in data:
        print(
            f"ERROR: JSON file {json_path.name} is missing a top-level 'phase_2' key.\n"
            "Please restructure the file so all eval categories live under 'phase_2'."
        )
        sys.exit(1)

    doc_a = data.get("doc_a", "AMD_2021_10K")
    doc_b = data.get("doc_b", "AMD_2022_10K")

    # ------------------------------------------------------------------
    # Check document availability
    # ------------------------------------------------------------------
    for name in (doc_a, doc_b):
        if name not in doc_paths:
            print(f"ERROR: Unknown document '{name}'. Add it to doc_paths.")
            sys.exit(1)
        if not doc_paths[name].exists():
            print(
                f"ERROR: {name}.txt not found at {doc_paths[name]}\n"
                "Please place the text file there before running this script."
            )
            sys.exit(1)

    # ------------------------------------------------------------------
    # Load documents
    # ------------------------------------------------------------------
    system_prompt = f"{load_document(doc_a, doc_paths)}\n\n{load_document(doc_b, doc_paths)}"
    print(f"doc_a: {doc_a}")
    print(f"doc_b: {doc_b}")
    print(f"System prompt length: {len(system_prompt):,} chars")

    # ------------------------------------------------------------------
    # Build phase 2 conversations (all categories, dynamically)
    # ------------------------------------------------------------------
    phase2_convos: list[Conversation] = []
    category_counts = {}

    for category, questions in data["phase_2"].items():
        if not isinstance(questions, list):
            # Skip non-list entries (e.g. a nested "description" string)
            continue

        for q in questions:
            phase2_convos.append(make_conversation(
                question=q["question"],
                answer=q["golden_answer"],
                system_prompt=system_prompt,
                category=category,
                question_id=q["id"],
                difficulty=q.get("difficulty", "medium"),
                reasoning_type=q.get("reasoning_type", "factual"),
            ))

        category_counts[category] = len(questions)

    total = sum(category_counts.values())
    print(f"\nLoaded {total} questions across {len(category_counts)} categories:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count}")

    # ------------------------------------------------------------------
    # Write parquet
    # ------------------------------------------------------------------
    if args.output:
        phase2_path = Path(args.output).resolve()
    else:
        phase2_path = json_path.parent.parent / "eval" / f"{json_path.stem}_phase2.parquet"

    phase2_path.parent.mkdir(parents=True, exist_ok=True)
    write_conversations(phase2_convos, str(phase2_path))
    print(f"\nWrote {len(phase2_convos)} conversations to {phase2_path}")

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    print("\n--- Verification ---")
    p2 = read_conversations(str(phase2_path))
    print(f"Phase 2: {len(p2)} conversations (expected {total})")
    assert len(p2) == total, f"Count mismatch: {len(p2)} != {total}"

    actual_cats: dict[str, int] = {}
    for convo in p2:
        assert convo.type == "continual_eval"
        assert convo.metadata["phase"] == 2
        cat = convo.metadata["category"]
        actual_cats[cat] = actual_cats.get(cat, 0) + 1

    assert actual_cats == category_counts, f"Category mismatch: {actual_cats} != {category_counts}"
    print(f"Categories: {actual_cats}")

    for convo in p2:
        assert len(convo.messages) == 2
        assert convo.messages[0].role == "user"
        assert convo.messages[1].role == "assistant"
        assert len(convo.messages[0].content) > 0
        assert len(convo.messages[1].content) > 0

    assert doc_a in p2[0].system_prompt
    assert doc_b in p2[0].system_prompt

    print("\nAll verification checks passed!")


if __name__ == "__main__":
    main()
