"""Generate eval questions for a document using an LLM.

Supports two generation modes:
  - batch:      All 5 prompt templates in a single LLM call
  - sequential: One LLM call per prompt template

Output is a JSON file compatible with experiments/evaluate/convert_eval.py.

Usage:
    # Batch mode (all prompts in one call)
    python experiments/synthesize/generate_eval_questions.py \
        --text-path data/texts/AMD_2021_10K.txt \
        --model gpt-4o \
        --mode batch \
        --output data/financebench/eval_data_json/my_eval.json

    # Sequential mode (one call per prompt)
    python experiments/synthesize/generate_eval_questions.py \
        --text-path data/texts/AMD_2021_10K.txt \
        --model gpt-4o \
        --mode sequential \
        --output data/financebench/eval_data_json/my_eval.json
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
# Prompt templates
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES = {
    "factual": (
        "Please generate a question to test someone's ability to remember factual details from the document. "
        "The answer should be a few tokens long and be a factual detail from the document, such as a number, "
        "entity, date, title, or name.\n"
        "This question should not be common knowledge: instead, it should be something that is only answerable "
        "via information in the document."
    ),
    "knowledge": (
        "Please generate a question that requires combining information mentioned both inside and outside the document.\n"
        "This question should require using a fact from the document and also a fact that you are confident about, "
        "but is not mentioned in the document. For instance:\n"
        "- What are the founding dates of the companies that got acquired this year? "
        "(names of acquired companies are in the document, founding dates are not)\n"
        "- What is the name of the CEO's spouse? "
        "(CEO name is in the document, spouse name is not)\n"
        "The answer should be a fact that is a few tokens long such as a number, entity, date, title, or name."
    ),
    "synthesize": (
        "Please generate a question that requires synthesizing and aggregating information in the document.\n"
        "For instance, you could ask someone to summarize a page of the document, list all the key competitors "
        "mentioned in the document, or summarize the company's business model."
    ),
    "structure": (
        "Please generate a question that requires understanding the structure of the document.\n"
        "This question should be more about the structure of the document, rather than the precise statement details. "
        "For instance, you could ask someone to list the titles of all the sections in the document, describe the "
        "document structure, report the total number of pages, ask which section amongst two sections comes first, "
        "or report the section with the largest number of tables."
    ),
    "reasoning": (
        "Please generate a question that requires mathematical reasoning over the values in the document.\n"
        "This question should require going beyond the facts directly mentioned in the document, such as asking "
        "to compute the percentage increase in revenue between two years, find the largest expense category, or "
        "calculate difference in profit between two years.\n"
        "The answer should be a number."
    ),
    "multi_hop": (
        "Please generate a multi-hop question that tests someone's ability to use factual information mentioned "
        "in at least two very different sub-sections of the document.\n"
        "This question shouldn't be a standard question about this kind of document. Instead, it should ask "
        "about two particularly disconnected ideas, like comparing information about the amount of owned space "
        "for the company headquarters with the amount of dollars of estimated liability or comparing the revenue "
        "number with the number of employees.\n"
        "This question should also test one's ability to do retrieval: do not give away part of the answer in "
        "the question. Ensure that for one to get the correct answer to the question, they need to understand "
        "the document.\n"
        "The answer should be a short: for example, a number, entity, date, title, or name."
    ),
}

ID_PREFIXES = {
    "factual": "F",
    "knowledge": "K",
    "synthesize": "SY",
    "structure": "ST",
    "reasoning": "R",
    "multi_hop": "M",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_sequential_prompt(template: str, num_questions: int) -> str:
    return (
        f"{template}\n\n"
        f"Generate exactly {num_questions} question-answer pairs.\n"
        "Respond ONLY with a JSON array where each element has keys \"question\" and \"golden_answer\". "
        "Example:\n"
        '[{"question": "What is ...?", "golden_answer": "42"}]\n'
        "Do NOT include any text outside the JSON array."
    )


def _build_batch_prompt(num_questions: int, categories: list[str]) -> str:
    parts = []
    for name in categories:
        parts.append(f"### {name}\n{PROMPT_TEMPLATES[name]}")
    all_templates = "\n\n".join(parts)
    cat_list = ", ".join(categories)

    return (
        f"Below are {len(categories)} prompt categories. For EACH category, generate exactly {num_questions} "
        "question-answer pairs about the document.\n\n"
        f"{all_templates}\n\n"
        "Respond ONLY with a JSON object where each key is the category name "
        f"({cat_list}) and each value is an array of objects "
        'with keys "question" and "golden_answer".\n'
        "Example:\n"
        '{"' + categories[0] + '": [{"question": "...", "golden_answer": "..."}], ...}\n'
        "Do NOT include any text outside the JSON object."
    )


def _parse_json_from_response(text: str):
    """Extract JSON from an LLM response, handling markdown code fences."""
    text = text.strip()
    # Strip markdown code fences if present
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


def _format_questions(raw_list: list, category: str) -> list[dict]:
    prefix = ID_PREFIXES[category]
    formatted = []
    for i, item in enumerate(raw_list, 1):
        formatted.append({
            "id": f"{prefix}{i:02d}",
            "question": item["question"],
            "golden_answer": item["golden_answer"],
        })
    return formatted


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

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


def generate_sequential(
    client: OpenAI, model: str, document: str,
    num_questions: int, questions_per_call: int, categories: list[str],
) -> dict:
    """Generate questions via multiple LLM calls.

    For each category, splits num_questions into chunks of questions_per_call
    and makes one LLM call per chunk. This gives the LLM fresh context each
    time, which can produce more diverse questions.
    """
    system_content = f"You are given the following document:\n\n{document}"
    results = {}
    for category in categories:
        template = PROMPT_TEMPLATES[category]
        all_raw = []
        remaining = num_questions
        call_num = 0
        while remaining > 0:
            n = min(questions_per_call, remaining)
            call_num += 1
            print(f"  [{category}] call {call_num}: generating {n} questions...")
            user_prompt = _build_sequential_prompt(template, n)
            text = _call_llm(client, model, system_content, user_prompt)
            raw = _parse_json_from_response(text)
            all_raw.extend(raw)
            remaining -= n

        results[category] = _format_questions(all_raw, category)
        print(f"  [{category}] total: {len(results[category])} questions")

    return results


def generate_batch(
    client: OpenAI, model: str, document: str,
    num_questions: int, categories: list[str],
) -> dict:
    """Generate all questions in a single LLM call."""
    print(f"  Generating {num_questions} questions per category in one call...")
    system_content = f"You are given the following document:\n\n{document}"
    user_prompt = _build_batch_prompt(num_questions, categories)
    text = _call_llm(client, model, system_content, user_prompt)

    raw = _parse_json_from_response(text)
    results = {}
    for category in categories:
        if category in raw:
            results[category] = _format_questions(raw[category], category)
            print(f"    {category}: {len(results[category])} questions")
        else:
            print(f"    WARNING: {category} missing from response")
            results[category] = []

    return results


def run(args):
    # Read document
    print(f"Reading document: {args.text_path}")
    document = Path(args.text_path).read_text(encoding="utf-8")
    print(f"  Document length: {len(document)} chars")

    # Create client
    client = OpenAI(
        base_url=ENDPOINT,
        api_key=API_KEY,
    )
    model = args.model or DEPLOYMENT_NAME

    # Resolve categories
    categories = args.categories if args.categories else list(PROMPT_TEMPLATES.keys())

    # Generate
    print(f"\nMode: {args.mode}, model: {model}, categories: {categories}")
    if args.mode == "sequential":
        questions = generate_sequential(
            client, model, document, args.num_questions, args.questions_per_call, categories,
        )
    else:
        questions = generate_batch(client, model, document, args.num_questions, categories)

    # Build output
    phase_key = f"phase_{args.phase}"
    output = {
        "description": (
            f"Auto-generated eval questions from {Path(args.text_path).name}. "
            f"Mode: {args.mode}, model: {model}, "
            f"{args.num_questions} questions per category."
        ),
        phase_key: questions,
    }

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    total = sum(len(v) for v in questions.values())
    print(f"\nSaved {total} questions to {output_path}")
    print(f"Convert to parquet with: python experiments/evaluate/convert_eval.py --json {output_path} --phase {args.phase}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate eval questions from a document using an LLM",
    )
    parser.add_argument("--text-path", required=True, help="Path to the document text file")
    parser.add_argument("--model", default=None, help="Model/deployment name (default: AZURE_OPENAI_DEPLOYMENT env or gpt-5.4)")
    parser.add_argument(
        "--mode", choices=["batch", "sequential"], default="batch",
        help="batch = all prompts in one call, sequential = one call per prompt (default: batch)",
    )
    parser.add_argument("--num-questions", type=int, default=5, help="Total questions per category (default: 5)")
    parser.add_argument("--questions-per-call", type=int, default=5, help="Questions per LLM call in sequential mode (default: 5)")
    parser.add_argument(
        "--categories", nargs="+", choices=list(PROMPT_TEMPLATES.keys()), default=None,
        help="Which categories to generate (default: all 5)",
    )
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2], help="Phase number for output JSON (default: 1)")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
