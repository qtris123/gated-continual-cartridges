"""
Parse Claude UI responses into a .pkl of Conversation objects.

Workflow:
    1. Paste prompts into Claude UI (with both 10-K docs uploaded).
    2. Save each response as experiments/multi_documents/data/responses/{template}.txt
       e.g. factual.txt, knowledge.txt, disjoint.txt, ...
    3. Run this script:
           python experiments/multi_documents/parse_responses_to_pkl.py

Expected format inside each .txt file:
    1. Q: <question>
       A: <answer>

    2. Q: <question>
       A: <answer>
    ...

Output: a .pkl file in data/ containing Conversation objects compatible with LossEvalDataset.
"""

import os
import re
from glob import glob
from pathlib import Path

import dotenv

from cartridges.structs import Conversation, write_conversations

dotenv.load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
TEXT_DIR = SCRIPT_DIR / "data" / "texts"
RESPONSES_DIR = SCRIPT_DIR / "data" / "responses"
OUTPUT_DIR = Path(os.environ.get("CARTRIDGES_OUTPUT_DIR", str(SCRIPT_DIR / "data")))

DOC_PATHS = {
    "AMD_2022_10K": TEXT_DIR / "AMD_2022_10K.txt",
    "PEPSICO_2022_10K": TEXT_DIR / "PEPSICO_2022_10K.txt",
}


def load_documents() -> str:
    """Load both document texts and wrap them in labeled XML tags."""
    parts: list[str] = []
    for name, path in DOC_PATHS.items():
        text = path.read_text(encoding="utf-8")
        parts.append(f'<document name="{name}">\n{text}\n</document>')
    return "\n\n".join(parts)


def parse_qa_pairs(text: str) -> list[tuple[str, str]]:
    """Parse numbered Q&A pairs from a Claude response.

    Handles the expected format:
        1. Q: <question>
           A: <answer>

        2. Q: <question>
           A: <answer (possibly multi-line)>
    """
    # Split into blocks per numbered item.
    # Each block starts with a number followed by . or ) and then Q:
    blocks = re.split(r"\n\s*\d+[\.\)]\s*", "\n" + text)

    pairs: list[tuple[str, str]] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Try to find Q: ... A: ... pattern
        match = re.match(
            r"Q:\s*(.+?)\s*\n\s*A:\s*(.+)",
            block,
            re.DOTALL,
        )
        if match:
            question = match.group(1).strip()
            answer = match.group(2).strip()
            pairs.append((question, answer))

    return pairs


def main():
    response_files = sorted(glob(str(RESPONSES_DIR / "*.txt")))
    if not response_files:
        print(f"No .txt files found in {RESPONSES_DIR}")
        print("Save Claude UI responses there first (e.g. factual.txt, knowledge.txt, ...)")
        return

    print("Loading documents...")
    system_prompt = load_documents()
    print(f"  Combined document length: {len(system_prompt):,} characters")

    conversations: list[Conversation] = []

    for fpath in response_files:
        fname = Path(fpath).stem  # e.g. "factual"
        template_name = f"genconvo_{fname}"
        raw = Path(fpath).read_text(encoding="utf-8")
        pairs = parse_qa_pairs(raw)
        print(f"  [{template_name}] parsed {len(pairs)} Q&A pairs from {Path(fpath).name}")

        for question, answer in pairs:
            convo = Conversation(
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
                metadata={"seed_type": template_name},
                type="genconvo_eval",
            )
            conversations.append(convo)

    if not conversations:
        print("\nNo Q&A pairs parsed. Check the format of your response files.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"generate_eval_questions_claude_n{len(conversations)}.pkl"
    write_conversations(conversations, str(out_path))
    print(f"\nSaved {len(conversations)} conversations to {out_path}")


if __name__ == "__main__":
    main()
