"""Adapt existing eval questions from one document version to another using an LLM.

Given eval questions generated for one document (e.g., AMD 2021 10-K), this script
uses an LLM to adapt them for a new version of the document (e.g., AMD 2022 10-K).
The LLM determines which questions need updated answers, updated question text,
or should be dropped entirely.

Usage:
    python experiments/synthesize/adapt_eval_questions.py \
        --questions-json data/eval_data_json/amd_2021_120_batch.json \
        --text-path data/financebench/texts/AMD_2022_10K.txt \
        --output data/eval_data_json/amd_2022_120_adapted.json
"""

import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://dtx-rd-aoi.cognitiveservices.azure.com/openai/v1/")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_from_response(text: str):
    """Extract JSON from an LLM response, handling markdown code fences."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


def _build_adapt_prompt(category: str, questions: list[dict]) -> str:
    """Build the user prompt for adapting questions in one category."""
    questions_json = json.dumps(questions, indent=2, ensure_ascii=False)
    return (
        f"Below are {len(questions)} eval questions in the \"{category}\" category. "
        "These questions were originally written for a PREVIOUS version of the document. "
        "You are now given a NEW version of the document (provided in the system message).\n\n"
        "For each question, determine:\n"
        "1. If the question and answer are still exactly correct for the new document → keep unchanged\n"
        "2. If the question needs updates (e.g., year references, updated numbers, changed facts) "
        "→ update both the question and golden_answer to match the new document\n"
        "3. If the question is no longer answerable from the new document (e.g., references "
        "something that doesn't exist in the new version) → mark as dropped\n\n"
        "IMPORTANT RULES:\n"
        "- For questions that reference specific years (e.g., '2021'), update them to reference "
        "the corresponding year in the new document\n"
        "- For financial figures, employee counts, dates, page numbers, etc., verify against "
        "the new document and update if different\n"
        "- For questions about document structure (page numbers, section ordering), verify "
        "against the new document\n"
        "- Keep the same question style and difficulty level\n"
        "- Preserve the original question ID\n\n"
        f"Questions to adapt:\n{questions_json}\n\n"
        "Respond ONLY with a JSON array where each element has:\n"
        '- "id": original question ID\n'
        '- "changed": true or false\n'
        '- "dropped": true or false (use true only if the question cannot be answered from the new document)\n'
        '- "question": the question text (updated if changed, original if unchanged)\n'
        '- "golden_answer": the answer (updated if changed, original if unchanged)\n'
        '- "change_reason": brief explanation if changed or dropped (empty string if unchanged)\n\n'
        "Do NOT include any text outside the JSON array."
    )


def _call_llm(client: OpenAI, model: str, system_content: str, user_content: str) -> str:
    """Make a single chat completion call and return the response text."""
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
    )
    return completion.choices[0].message.content


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def adapt_category(
    client: OpenAI,
    model: str,
    document: str,
    category: str,
    questions: list[dict],
) -> tuple[list[dict], dict]:
    """Adapt questions in one category. Returns (adapted_questions, stats)."""
    print(f"\n  [{category}] Adapting {len(questions)} questions...")

    system_content = f"You are given the following document:\n\n{document}"
    user_prompt = _build_adapt_prompt(category, questions)
    text = _call_llm(client, model, system_content, user_prompt)

    results = _parse_json_from_response(text)

    adapted = []
    stats = {"unchanged": 0, "changed": 0, "dropped": 0}

    for item in results:
        if item.get("dropped", False):
            stats["dropped"] += 1
            reason = item.get("change_reason", "")
            print(f"    DROPPED {item['id']}: {reason}")
            adapted.append({
                "id": item["id"],
                "question": item["question"],
                "golden_answer": "This question cannot be answered from the document.",
            })
            continue

        if item.get("changed", False):
            stats["changed"] += 1
            reason = item.get("change_reason", "")
            print(f"    CHANGED {item['id']}: {reason}")
        else:
            stats["unchanged"] += 1

        adapted.append({
            "id": item["id"],
            "question": item["question"],
            "golden_answer": item["golden_answer"],
        })

    print(f"    Summary: {stats['unchanged']} unchanged, {stats['changed']} changed, {stats['dropped']} dropped")
    return adapted, stats


def run(args):
    # Read existing questions
    print(f"Reading questions: {args.questions_json}")
    with open(args.questions_json, encoding="utf-8") as f:
        data = json.load(f)

    source_key = f"phase_{args.source_phase}"
    if source_key not in data:
        print(f"ERROR: '{source_key}' not found in {args.questions_json}")
        return
    phase_data = data[source_key]

    # Read new document
    print(f"Reading new document: {args.text_path}")
    document = Path(args.text_path).read_text(encoding="utf-8")
    print(f"  Document length: {len(document)} chars")

    # Create client
    client = OpenAI(
        base_url=ENDPOINT,
        api_key=API_KEY,
    )
    model = args.model or DEPLOYMENT_NAME

    # Adapt each category
    print(f"\nModel: {model}")
    output_phase = {}
    total_stats = {"unchanged": 0, "changed": 0, "dropped": 0}

    for category, questions in phase_data.items():
        if not isinstance(questions, list):
            continue
        adapted, stats = adapt_category(client, model, document, category, questions)
        output_phase[category] = adapted
        for k in total_stats:
            total_stats[k] += stats[k]

    # Build output
    phase_key = f"phase_{args.phase}"
    source_name = Path(args.questions_json).stem
    doc_name = Path(args.text_path).stem
    output = {
        "description": (
            f"Adapted eval questions from {source_name} to {doc_name}. "
            f"Model: {model}. "
            f"{total_stats['unchanged']} unchanged, {total_stats['changed']} changed, "
            f"{total_stats['dropped']} dropped."
        ),
        phase_key: output_phase,
    }

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    total_kept = total_stats["unchanged"] + total_stats["changed"]
    print(f"\n{'='*60}")
    print(f"Total: {total_kept} questions kept ({total_stats['unchanged']} unchanged, "
          f"{total_stats['changed']} changed), {total_stats['dropped']} dropped")
    print(f"Saved to {output_path}")
    print(f"Convert to parquet: python experiments/evaluate/convert_eval.py --json {output_path} --phase {args.phase}")


def main():
    parser = argparse.ArgumentParser(
        description="Adapt eval questions from one document version to another using an LLM",
    )
    parser.add_argument("--questions-json", required=True, help="Path to existing eval questions JSON")
    parser.add_argument("--text-path", required=True, help="Path to the new document text file")
    parser.add_argument("--model", default=None, help="Model/deployment name (default: AZURE_OPENAI_DEPLOYMENT env or gpt-5.4)")
    parser.add_argument("--phase", type=int, default=2, help="Phase number for output JSON (default: 2)")
    parser.add_argument("--source-phase", type=int, default=1, help="Phase key to read from input JSON (default: 1)")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
