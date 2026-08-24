"""Anchor FinQA / QuALITY phase-eval questions to their source document.

Both datasets were built as single-document QA: each question implicitly
refers to "the" document in front of the annotator, so nothing in the text
says which company's filing or which story it means. In the multi-document
phase panels that makes many questions ambiguous in principle (FinQA phase 1
alone holds 13 companies), which confounds the accuracy R-matrix: degradation
mixes attention interference with growing referential ambiguity. QASPER had
the identical problem and was fixed by a GPT-4.1 rewrite naming each paper
(cartridges/data/qasper/rewrite.py); this applies the same treatment.

Only the question stem is rewritten — answers, MCQ options, metadata and the
Conversation layout are preserved bit-for-bit, and outputs are NEW files so
existing results stay comparable:

    data/<ds>/phases/phase<k>_eval_anchored.parquet   rewritten eval set
    data/<ds>/phases/rewrite_raw/phase<k>_questions.parquet  provenance
                                                     (old/new question pairs)

Anchors:
    finqa    company ticker + fiscal year from metadata; the model is asked
             to use the company's common name plus ticker
    quality  story title, looked up from tasksource/QuALITY by article_id

Everything is written locally before/without any network upload, and the
model is pinned to the same snapshot used for QASPER so all rewritten
datasets are produced the same way.

Usage:
    set -a; source .env; set +a; unset OPENAI_API_BASE_URL
    python cartridges/data/anchor_rewrite.py --dataset finqa --limit 3   # smoke
    python cartridges/data/anchor_rewrite.py --dataset finqa
    python cartridges/data/anchor_rewrite.py --dataset quality
"""

import argparse
import asyncio
import os
from pathlib import Path

import pandas as pd

from cartridges.structs import Conversation, read_conversations, write_conversations

MODEL = "gpt-4.1-2025-04-14"
PHASES = [1, 2, 3, 4, 5]
# The org's gpt-4.1 limit is 30k tokens/min (~130 of these calls); a large
# burst just trades throughput for 429s, so stay modest and back off long.
MAX_CONCURRENT = 8

FINQA_PROMPT = """\
Rewrite the following financial question so it explicitly states which company's \
SEC filing it is asking about. The question refers to the fiscal year {year} filing \
of the company with stock ticker {company}. Refer to the company by its common name \
followed by the ticker, e.g. "Air Products (APD)", and mention the fiscal year.
Do not change what the question asks, its numbers, or its units. Output only the \
rewritten question.

<question>
{question}
</question>
"""

QUALITY_PROMPT = """\
Rewrite the following reading-comprehension question so it explicitly names the \
story it is about: "{title}". Do not change what the question asks or its wording \
beyond adding the story identification. Output only the rewritten question.

<question>
{question}
</question>
"""


def quality_titles() -> dict[str, str]:
    from datasets import load_dataset

    from cartridges.data.quality.resources import QUALITY_SPLITS

    titles: dict[str, str] = {}
    for split in QUALITY_SPLITS:
        for row in load_dataset("tasksource/QuALITY", split=split):
            titles[str(row["article_id"])] = row["title"]
    return titles


def split_quality_content(content: str) -> tuple[str, str]:
    """The exporter writes '{stem}\n\n(a) ...' — split stem from options."""
    idx = content.find("\n\n(a) ")
    if idx < 0:
        raise ValueError(f"no options block in: {content[:80]!r}")
    return content[:idx], content[idx:]


async def rewrite_all(prompts: list[str]) -> list[str]:
    import openai

    client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async def one(prompt: str) -> str:
        async with sem:
            for attempt in range(8):
                try:
                    resp = await client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=256,
                        temperature=0.0,
                    )
                    return resp.choices[0].message.content.strip()
                except openai.RateLimitError:
                    if attempt == 7:
                        raise
                    # 429s here mean the whole run is at the TPM ceiling —
                    # wait long enough for the window to roll over.
                    await asyncio.sleep(min(60, 4 * 2 ** attempt))
                except Exception:
                    if attempt == 7:
                        raise
                    await asyncio.sleep(2 * (attempt + 1))

    return await asyncio.gather(*(one(p) for p in prompts))


def process_phase(dataset: str, phase: int, limit: int | None,
                  titles: dict[str, str] | None) -> None:
    phase_dir = Path("data") / dataset / "phases"
    src = phase_dir / f"phase{phase}_eval.parquet"
    convos = read_conversations(str(src))
    if limit:
        convos = convos[:limit]

    prompts, stems, tails = [], [], []
    for c in convos:
        content = c.messages[0].content
        if dataset == "finqa":
            stem, tail = content, ""
            prompts.append(FINQA_PROMPT.format(
                question=stem,
                company=c.metadata.get("company"),
                year=c.metadata.get("year"),
            ))
        else:
            stem, tail = split_quality_content(content)
            title = titles[str(c.metadata["article_id"])]
            prompts.append(QUALITY_PROMPT.format(question=stem, title=title))
        stems.append(stem)
        tails.append(tail)

    print(f"  phase {phase}: rewriting {len(prompts)} questions...")
    new_stems = asyncio.run(rewrite_all(prompts))

    # provenance FIRST — a later failure must never lose the API spend
    raw_dir = phase_dir / "rewrite_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {
            "question_id": c.metadata.get("question_id"),
            "old_question": old,
            "new_question": new,
            "model": MODEL,
        }
        for c, old, new in zip(convos, stems, new_stems)
    ]).to_parquet(raw_dir / f"phase{phase}_questions.parquet", index=False)

    out = [
        Conversation(
            messages=[
                Conversation.Message(content=new + tail, role="user", token_ids=None),
                c.messages[1],
            ],
            system_prompt=c.system_prompt,
            metadata={**c.metadata, "anchored": True},
            type=c.type,
        )
        for c, new, tail in zip(convos, new_stems, tails)
    ]
    suffix = "_anchored" if not limit else f"_anchored_smoke{limit}"
    dst = phase_dir / f"phase{phase}_eval{suffix}.parquet"
    write_conversations(out, str(dst))
    print(f"  phase {phase}: wrote {len(out)} → {dst}")
    for old, new in list(zip(stems, new_stems))[:2]:
        print(f"    OLD: {old[:90]}")
        print(f"    NEW: {new[:110]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, choices=["finqa", "quality"])
    ap.add_argument("--phases", type=int, nargs="+", default=PHASES)
    ap.add_argument("--limit", type=int, default=None,
                    help="Rewrite only the first N questions per phase (smoke test)")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set — run: set -a; source .env; set +a")

    titles = quality_titles() if args.dataset == "quality" else None
    for phase in args.phases:
        process_phase(args.dataset, phase, args.limit, titles)


if __name__ == "__main__":
    main()
