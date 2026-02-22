"""
Generate multi-document eval Q&A following the GenConvo approach (Appendix D.1).

Instead of chunking documents and generating questions from random chunks, this
script passes **both** full documents as context and asks the model to generate
16 unique questions per prompt template.  Each question is then answered in a
separate call with the same full-document context.

Output: a parquet file of Conversation objects compatible with LossEvalDataset.

Usage:
    python experiments/multi_documents/generate_eval_questions.py
"""

import asyncio
import os
import re
from pathlib import Path

import dotenv
from openai import AsyncOpenAI

from cartridges.structs import Conversation, write_conversations

dotenv.load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
TEXT_DIR = SCRIPT_DIR / "data" / "texts"
OUTPUT_DIR = Path(os.environ.get("CARTRIDGES_OUTPUT_DIR", str(SCRIPT_DIR / "data")))

DOC_PATHS = {
    "AMD_2022_10K": TEXT_DIR / "AMD_2022_10K.txt",
    "PEPSICO_2022_10K": TEXT_DIR / "PEPSICO_2022_10K.txt",
}

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = 1.0
NUM_QUESTIONS_PER_TEMPLATE = 1

GENCONVO_TEMPLATES: dict[str, str] = {
    "genconvo_factual": (
        """
        Please generate {n} unique questions to test someone’s ability to remember factual details from the document. The
        answer should be a few tokens long and be a factual detail from the statement, such as a number, entity,
        date, title, or name.
        This question should not be common knowledge: instead, it should be something that is only answerable
        via information in the document.
        """
    ),
    "genconvo_knowledge": (
        """
        Please generate {n} unique questions that requires combining information mentioned both inside and outside the
        document.
        This question should require using a fact from the document and also a fact that you are confident about,
        but is not mentioned in the document. For instance: - What are the founding dates of the companies
        that got acquired this year? This is a good question because the names of the acquired companies are
        mentioned in the document and the founding dates are not mentioned. - What is the name of the CEO’s
        spouse? This is a good question because the name of the CEO is mentioned in the document and the
        spouse’s name is not mentioned.
        The answer should be a fact that is a few tokens long such as a number, entity, date, title, or name.
        """
    ),
    "genconvo_disjoint": (
        """
        Please generate {n} unique multi-hop questions that tests someone's ability to use factual information mentioned
        in at least two very different sub-sections of the document.
        This question shouldn't be a standard question about this kind of document. Instead, it should ask
        about two particularly disconnected ideas, like comparing information about the amount of owned space
        for the company headquarters with the amount of dollars of estimated liability or comparing the revenue
        number with the number of employees.
        This question should also test one's ability to do retrieval: do not give away part of the answer in
        the question. Ensure that for one to get the correct answer to the question, they need to understand
        the document.
        The answer should be a short: for example, a number, entity, date, title, or name.
        """
    ),
    "genconvo_synthesize": (
        """
        Please generate {n} unique questions that requires synthesizing and aggregating information in the document.
        For instance, you could ask someone to summarize a page of the document, list all the key competitors
        mentioned in the document, or summarize the company's business model.
        """
    ),
    "genconvo_structure": (
        """
        Please generate {n} unique questions that requires understanding the structure of the document.
        This question should be more about the structure of the document, rather than the precise statement
        details. For instance, you could ask someone to list the titles of all the sections in the document,
        describe the document structure, report the total number of pages, ask which section amongst two sections
        comes first, or report the section with the largest number of tables.
        """
    ),
    "genconvo_creative": (
        """
        Please generate {n} unique questions about the document to test someone's ability to comprehend the content of the
        document. This question specifically should be focused on their ability to generalize the information
        about the document to a strange question of sorts.
        This question shouldn't be a standard question about this kind of document, it should ask to do something
        abnormal and creative, like writing a poem about a financial document.
        """
    ),
    "genconvo_counting": (
        """
        Please generate {n} unique questions that requires counting how frequently different events occur in the document.
        This question should be about statistical properties of the document, rather than the statement details.
        For instance, you could ask someone to count the number of times the word "million" is mentioned or
        count the length of the shortest section title.
        The answer should be a number.
        """
    ),
    "genconvo_reasoning": (
        """
        Please generate {n} unique questions that requires mathematical reasoning over the values in the document.
        This question should require going beyond the facts directly mentioned in the statement, such as asking
        to compute the percentage increase in revenue between two years, find the largest expense category, or
        calculate difference in profit between two years.
        The answer should be a number.
        """
    ),
}



def load_documents() -> str:
    """Load both document texts and wrap them in labeled XML tags."""
    parts: list[str] = []
    for name, path in DOC_PATHS.items():
        text = path.read_text(encoding="utf-8")
        parts.append(f'<document name="{name}">\n{text}\n</document>')
    return "\n\n".join(parts)


def parse_questions(response_text: str) -> list[str]:
    """Parse numbered questions from model output.

    Handles formats like:
        1. What is ...
        2) What is ...
        1: What is ...
    as well as plain lines.
    """
    lines = response_text.strip().splitlines()
    questions: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip leading number + punctuation (e.g. "1. ", "2) ", "3: ")
        cleaned = re.sub(r"^\d+[\.\)\:]\s*", "", line)
        if cleaned:
            questions.append(cleaned)
    return questions


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------

async def generate_questions_for_template(
    client: AsyncOpenAI,
    system_prompt: str,
    template_name: str,
    template_text: str,
) -> list[tuple[str, str]]:
    """Generate questions for one template and return (template_name, question) pairs."""
    user_msg = template_text.format(n=NUM_QUESTIONS_PER_TEMPLATE)

    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=TEMPERATURE,
    )
    raw = response.choices[0].message.content or ""
    questions = parse_questions(raw)
    print(f"  [{template_name}] generated {len(questions)} questions")
    return [(template_name, q) for q in questions]


async def generate_answer(
    client: AsyncOpenAI,
    system_prompt: str,
    question: str,
) -> str:
    """Generate an answer for a single question with the full document context."""
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content or ""


async def main():
    # --- Setup ---
    print("Loading documents...")
    system_prompt = load_documents()
    print(f"  Combined document length: {len(system_prompt):,} characters")

    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_API_BASE_URL"),
    )

    # --- Step 1: Generate questions (8 templates in parallel) ---
    print("\nGenerating questions across all templates...")
    question_tasks = [
        generate_questions_for_template(client, system_prompt, name, tmpl)
        for name, tmpl in GENCONVO_TEMPLATES.items()
    ]
    results = await asyncio.gather(*question_tasks)

    # Flatten into list of (template_name, question)
    all_questions: list[tuple[str, str]] = []
    for batch in results:
        all_questions.extend(batch)
    print(f"\nTotal questions generated: {len(all_questions)}")

    # --- Step 2: Generate answers (all in parallel, with concurrency limit) ---
    print("\nGenerating answers...")
    semaphore = asyncio.Semaphore(16)

    async def answer_with_limit(question: str) -> str:
        async with semaphore:
            return await generate_answer(client, system_prompt, question)

    answer_tasks = [answer_with_limit(q) for _, q in all_questions]
    answers = await asyncio.gather(*answer_tasks)
    print(f"  Generated {len(answers)} answers")

    # --- Step 3: Build Conversation objects ---
    conversations: list[Conversation] = []
    for (template_name, question), answer in zip(all_questions, answers):
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

    # --- Step 4: Save to parquet ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"generate_eval_questions_{MODEL}_n{len(conversations)}.pkl"
    write_conversations(conversations, str(out_path))
    print(f"\nSaved {len(conversations)} conversations to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
