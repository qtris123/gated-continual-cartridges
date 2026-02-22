"""
Convert eval JSON files into parquet files for the sequential cross-document
continual learning experiment.

Workflow:
    1. Place AMD_2022_10K.txt in data/texts/ (symlink from multi_documents)
    2. Place PEPSICO_2022_10K.txt in data/texts/
    3. Run:  python experiments/sequential_documents/convert_eval_json.py

Output:
    data/eval/eval_amd.parquet    – 35 conversations (amd_stable + amd_update, 2022 answers)
    data/eval/eval_pepsi.parquet  – 35 conversations (pepsi_factual)
"""

import json
import sys
from pathlib import Path

from cartridges.structs import Conversation, write_conversations, read_conversations

SCRIPT_DIR = Path(__file__).resolve().parent
TEXT_DIR = SCRIPT_DIR / "data" / "texts"
EVAL_DIR = SCRIPT_DIR / "data" / "eval"

AMD_JSON_PATH = SCRIPT_DIR / "eval_questions_amd.json"
PEPSI_JSON_PATH = SCRIPT_DIR / "pepsi_eval_questions.json"

DOC_PATHS = {
    "AMD_2022_10K": TEXT_DIR / "AMD_2022_10K.txt",
    "PEPSICO_2022_10K": TEXT_DIR / "PEPSICO_2022_10K.txt",
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
        },
        type="sequential_eval",
    )


def main():
    # ------------------------------------------------------------------
    # Check document availability
    # ------------------------------------------------------------------
    missing = [name for name, p in DOC_PATHS.items() if not p.exists()]
    for name in missing:
        print(
            f"ERROR: {name}.txt not found at {DOC_PATHS[name]}\n"
            "Please place the text file there before running this script."
        )
    if missing:
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load documents
    # ------------------------------------------------------------------
    doc_amd = load_document("AMD_2022_10K")
    doc_pepsi = load_document("PEPSICO_2022_10K")

    print(f"AMD 2022 system prompt length: {len(doc_amd):,} chars")
    print(f"PepsiCo 2022 system prompt length: {len(doc_pepsi):,} chars")

    # ------------------------------------------------------------------
    # Load eval questions
    # ------------------------------------------------------------------
    with open(AMD_JSON_PATH, encoding="utf-8") as f:
        amd_qs = json.load(f)

    with open(PEPSI_JSON_PATH, encoding="utf-8") as f:
        pepsi_qs = json.load(f)

    print(f"\nLoaded questions: {len(amd_qs)} AMD, {len(pepsi_qs)} PepsiCo")

    # ------------------------------------------------------------------
    # AMD eval: 35 conversations (system prompt = AMD 2022 doc)
    # ------------------------------------------------------------------
    amd_convos: list[Conversation] = []
    for q in amd_qs:
        amd_convos.append(make_conversation(
            question=q["question"],
            answer=q["reference_answer"],
            system_prompt=doc_amd,
            category=q["category"],
            question_id=q["question_id"],
        ))

    # ------------------------------------------------------------------
    # PepsiCo eval: 35 conversations (system prompt = PepsiCo 2022 doc)
    # ------------------------------------------------------------------
    pepsi_convos: list[Conversation] = []
    for q in pepsi_qs:
        pepsi_convos.append(make_conversation(
            question=q["question"],
            answer=q["reference_answer"],
            system_prompt=doc_pepsi,
            category=q["category"],
            question_id=q["question_id"],
        ))

    # ------------------------------------------------------------------
    # Write parquet files
    # ------------------------------------------------------------------
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    amd_path = EVAL_DIR / "eval_amd.parquet"
    pepsi_path = EVAL_DIR / "eval_pepsi.parquet"

    write_conversations(amd_convos, str(amd_path))
    print(f"\nWrote {len(amd_convos)} conversations to {amd_path}")

    write_conversations(pepsi_convos, str(pepsi_path))
    print(f"Wrote {len(pepsi_convos)} conversations to {pepsi_path}")

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    print("\n--- Verification ---")

    amd_read = read_conversations(str(amd_path))
    pepsi_read = read_conversations(str(pepsi_path))
    print(f"AMD eval:   {len(amd_read)} conversations (expected 35)")
    print(f"PepsiCo eval: {len(pepsi_read)} conversations (expected 35)")

    assert len(amd_read) == 35, f"AMD count mismatch: {len(amd_read)}"
    assert len(pepsi_read) == 35, f"PepsiCo count mismatch: {len(pepsi_read)}"

    # Check metadata
    for convo in amd_read:
        assert convo.type == "sequential_eval"
        assert convo.metadata["category"] in ("amd_stable", "amd_update")
        assert convo.metadata["question_id"]

    for convo in pepsi_read:
        assert convo.type == "sequential_eval"
        assert convo.metadata["category"] == "pepsi_factual"
        assert convo.metadata["question_id"]

    # Check message structure
    for convo in amd_read + pepsi_read:
        assert len(convo.messages) == 2
        assert convo.messages[0].role == "user"
        assert convo.messages[1].role == "assistant"
        assert len(convo.messages[0].content) > 0
        assert len(convo.messages[1].content) > 0

    # Check system prompts contain document text
    assert "AMD_2022_10K" in amd_read[0].system_prompt
    assert "PEPSICO_2022_10K" in pepsi_read[0].system_prompt

    # Category counts
    amd_cats = {}
    for c in amd_read:
        cat = c.metadata["category"]
        amd_cats[cat] = amd_cats.get(cat, 0) + 1
    print(f"AMD categories: {amd_cats}")
    assert amd_cats == {"amd_stable": 15, "amd_update": 20}

    pepsi_cats = {}
    for c in pepsi_read:
        cat = c.metadata["category"]
        pepsi_cats[cat] = pepsi_cats.get(cat, 0) + 1
    print(f"PepsiCo categories: {pepsi_cats}")
    assert pepsi_cats == {"pepsi_factual": 35}

    print("\nAll verification checks passed!")


if __name__ == "__main__":
    main()
