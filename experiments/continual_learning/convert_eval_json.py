"""
Convert cartridge_eval_questions.json into eval parquet files for the
CARTRIDGE continual-learning experiment.

Workflow:
    1. Place AMD_2021_10K.txt and AMD_2022_10K.txt in data/texts/
    2. Run:  python experiments/continual_learning/convert_eval_json.py

Output:
    data/eval/eval_phase1.parquet  – 35 conversations (stable + update, 2021 answers)
    data/eval/eval_phase2.parquet  – 50 conversations (all categories, 2022/cross-temporal answers)
"""

import argparse
import json
import sys
from pathlib import Path

from cartridges.structs import Conversation, write_conversations, read_conversations

SCRIPT_DIR = Path(__file__).resolve().parent
TEXT_DIR = SCRIPT_DIR / "data" / "texts"

DEFAULT_JSON = "cartridge_eval_questions.json"

parser = argparse.ArgumentParser(description="Convert eval JSON to parquet files")
parser.add_argument(
    "--json",
    default=str(SCRIPT_DIR / "data" / DEFAULT_JSON),
    help="Path to the eval questions JSON file (default: data/cartridge_eval_questions.json)",
)
args = parser.parse_args()

JSON_PATH = Path(args.json).resolve()

# Derive output directory from JSON filename
json_filename = JSON_PATH.name
if json_filename == DEFAULT_JSON:
    EVAL_DIR = SCRIPT_DIR / "data" / "eval"
else:
    tag = json_filename.replace("cartridge_eval_questions_", "").replace(".json", "")
    EVAL_DIR = SCRIPT_DIR / "data" / f"eval_{tag}"

DOC_PATHS = {
    "AMD_2021_10K": TEXT_DIR / "AMD_2021_10K.txt",
    "AMD_2022_10K": TEXT_DIR / "AMD_2022_10K.txt",
}


def load_document(name: str) -> str:
    """Load a single document and wrap it in labeled XML tags."""
    path = DOC_PATHS[name]
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
    phase: int,
) -> Conversation:
    """Create a single Conversation object for an eval question."""
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
            "phase": phase,
        },
        type="continual_eval",
    )


def main():
    # ------------------------------------------------------------------
    # Check document availability
    # ------------------------------------------------------------------
    missing = [name for name, p in DOC_PATHS.items() if not p.exists()]
    if "AMD_2021_10K" in missing:
        print(
            f"ERROR: AMD_2021_10K.txt not found at {DOC_PATHS['AMD_2021_10K']}\n"
            "Please place the 2021 10-K text file there before running this script."
        )
        sys.exit(1)
    if "AMD_2022_10K" in missing:
        print(
            f"ERROR: AMD_2022_10K.txt not found at {DOC_PATHS['AMD_2022_10K']}\n"
            "Please place the 2022 10-K text file there before running this script."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load documents
    # ------------------------------------------------------------------
    doc_2021 = load_document("AMD_2021_10K")
    doc_2022 = load_document("AMD_2022_10K")

    system_prompt_phase1 = doc_2021
    system_prompt_phase2 = f"{doc_2021}\n\n{doc_2022}"

    print(f"Phase 1 system prompt length: {len(system_prompt_phase1):,} chars")
    print(f"Phase 2 system prompt length: {len(system_prompt_phase2):,} chars")

    # ------------------------------------------------------------------
    # Load eval questions
    # ------------------------------------------------------------------
    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    stable_qs = data["stable"]        # 15
    update_qs = data["update"]         # 20
    cross_qs = data["cross_temporal"]  # 15

    print(f"\nLoaded questions: {len(stable_qs)} stable, {len(update_qs)} update, {len(cross_qs)} cross_temporal")

    # ------------------------------------------------------------------
    # Phase 1: 35 conversations (stable + update, answer_2021)
    # System prompt: 2021 doc only
    # ------------------------------------------------------------------
    phase1_convos: list[Conversation] = []

    for q in stable_qs:
        phase1_convos.append(make_conversation(
            question=q["question"],
            answer=q["answer_2021"],
            system_prompt=system_prompt_phase1,
            category="stable",
            question_id=q["id"],
            difficulty=q["difficulty"],
            reasoning_type=q["reasoning_type"],
            phase=1,
        ))

    for q in update_qs:
        phase1_convos.append(make_conversation(
            question=q.get("question_2021", q.get("question")),
            answer=q["answer_2021"],
            system_prompt=system_prompt_phase1,
            category="update",
            question_id=q["id"],
            difficulty=q["difficulty"],
            reasoning_type=q["reasoning_type"],
            phase=1,
        ))

    # ------------------------------------------------------------------
    # Phase 2: 50 conversations (all categories, 2022/cross-temporal answers)
    # System prompt: both docs
    # ------------------------------------------------------------------
    phase2_convos: list[Conversation] = []

    for q in stable_qs:
        phase2_convos.append(make_conversation(
            question=q["question"],
            answer=q["answer_2022"],
            system_prompt=system_prompt_phase2,
            category="stable",
            question_id=q["id"],
            difficulty=q["difficulty"],
            reasoning_type=q["reasoning_type"],
            phase=2,
        ))

    for q in update_qs:
        phase2_convos.append(make_conversation(
            question=q.get("question_2022", q.get("question")),
            answer=q["answer_2022"],
            system_prompt=system_prompt_phase2,
            category="update",
            question_id=q["id"],
            difficulty=q["difficulty"],
            reasoning_type=q["reasoning_type"],
            phase=2,
        ))

    for q in cross_qs:
        phase2_convos.append(make_conversation(
            question=q["question"],
            answer=q["answer"],
            system_prompt=system_prompt_phase2,
            category="cross_temporal",
            question_id=q["id"],
            difficulty=q["difficulty"],
            reasoning_type=q["reasoning_type"],
            phase=2,
        ))

    # ------------------------------------------------------------------
    # Write parquet files
    # ------------------------------------------------------------------
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    phase1_path = EVAL_DIR / "eval_phase1.parquet"
    phase2_path = EVAL_DIR / "eval_phase2.parquet"

    write_conversations(phase1_convos, str(phase1_path))
    print(f"\nWrote {len(phase1_convos)} conversations to {phase1_path}")

    write_conversations(phase2_convos, str(phase2_path))
    print(f"Wrote {len(phase2_convos)} conversations to {phase2_path}")

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    print("\n--- Verification ---")

    p1 = read_conversations(str(phase1_path))
    p2 = read_conversations(str(phase2_path))
    print(f"Phase 1: {len(p1)} conversations (expected 35)")
    print(f"Phase 2: {len(p2)} conversations (expected 50)")

    assert len(p1) == 35, f"Phase 1 count mismatch: {len(p1)}"
    assert len(p2) == 50, f"Phase 2 count mismatch: {len(p2)}"

    # Check metadata
    for convo in p1:
        assert convo.type == "continual_eval"
        assert convo.metadata["phase"] == 1
        assert convo.metadata["category"] in ("stable", "update")
        assert convo.metadata["question_id"]
        assert convo.metadata["difficulty"] in ("easy", "medium", "hard")

    for convo in p2:
        assert convo.type == "continual_eval"
        assert convo.metadata["phase"] == 2
        assert convo.metadata["category"] in ("stable", "update", "cross_temporal")

    # Check message structure
    for convo in p1 + p2:
        assert len(convo.messages) == 2
        assert convo.messages[0].role == "user"
        assert convo.messages[1].role == "assistant"
        assert len(convo.messages[0].content) > 0
        assert len(convo.messages[1].content) > 0

    # Check system prompts contain document text
    assert "AMD_2021_10K" in p1[0].system_prompt
    assert "AMD_2021_10K" in p2[0].system_prompt
    assert "AMD_2022_10K" in p2[0].system_prompt
    assert "AMD_2022_10K" not in p1[0].system_prompt  # Phase 1 only has 2021

    # Category counts
    p2_cats = {}
    for c in p2:
        cat = c.metadata["category"]
        p2_cats[cat] = p2_cats.get(cat, 0) + 1
    print(f"Phase 2 categories: {p2_cats}")
    assert p2_cats == {"stable": 15, "update": 20, "cross_temporal": 15}

    print("\nAll verification checks passed!")


if __name__ == "__main__":
    main()
