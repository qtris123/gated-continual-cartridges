
from dataclasses import dataclass, asdict
import random
from datasets import load_dataset
import pydrantic

import asyncio
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import openai

from cartridges.data.qasper.resources import TOPIC_TO_IDS

# Where the rewritten topics live, alongside every other dataset under data/.
#   data/qasper/eval/qasper_eval_<TOPIC>.parquet  eval sets, one per topic
#   data/qasper/raw/<TOPIC>_raw.parquet           question/answer provenance
DEFAULT_OUTPUT_DIR = "data/qasper"

# Matches the user turn in the already-generated qasper_eval_{QA,MT,SA}.parquet,
# copied from examples/qasper/convert_hf_to_qasper_eval_mt_parquet.py so ASR and
# KG come out byte-compatible with the three that already exist.
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
    answer = answer if isinstance(answer, str) else str(answer)
    return np.array(
        [
            {
                "content": USER_MESSAGE_TEMPLATE.format(question=question),
                "role": "user",
                "token_ids": None,
                "top_logprobs": None,
            },
            {
                "content": f"<answer>\n{answer}\n</answer>",
                "role": "assistant",
                "token_ids": None,
                "top_logprobs": None,
            },
        ],
        dtype=object,
    )


# Set your OpenAI API key here or via environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY")

@dataclass
class RewrittenQasperQuestion:
    paper_id: str
    title: str
    abstract: str
    
    question: str
    answer: str

    old_answer: str
    old_question: str
    

ANSWER_PROMPT = """\
Can you please write a succinct answer to the following question based on the information provided?
You do not need to restate the paper name or answer in complete sentences. 

<question>
{question}
</question>

<answer-details>
{answer_details}
</answer-details>
"""

QUESTION_PROMPT = """\
Can you please rewrite the following question to be specific about which paper it is asking about?
The question should be answerable in a closed-book setting. It should require knowledge of the paper.
Do not output anything but the rewritten question.

<title>
{title}
</title>

<question>
{question}
</question>
"""

# --- OPENAI MIGRATION: Use openai>=1.0.0 API ---

async def async_openai_completion(prompt: str, model: str, max_tokens: int = 256) -> str:
    # Use the OpenAI async API for chat completion (openai>=1.0.0)
    client = openai.AsyncOpenAI(
        api_key=openai.api_key,
        base_url=os.environ.get("OPENAI_API_BASE_URL")
    )
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()

async def rewrite_questions(requests: List[Dict[str, Any]], model: str) -> List[str]:
    # Rewrite all questions asynchronously
    tasks = []
    for req in requests:
        prompt = QUESTION_PROMPT.format(
            title=req["title"],
            question=req["question"]
        )
        tasks.append(async_openai_completion(prompt, model=model))
    return await asyncio.gather(*tasks)

async def rewrite_answers(requests: List[Dict[str, Any]], new_questions: List[str], model: str) -> List[str]:
    # Rewrite all answers asynchronously, using the new questions
    tasks = []
    for req, new_q in zip(requests, new_questions):
        prompt = ANSWER_PROMPT.format(
            question=new_q,
            answer_details=req["answer_data"]
        )
        tasks.append(async_openai_completion(prompt, model=model))
    return await asyncio.gather(*tasks)


class RewriteQasperConfig(pydrantic.RunConfig):
    topic: str
    model: str = "gpt-4.1-2025-04-14"
    limit: Optional[int] = None

    # Results are written to disk before any network call, so a failed upload
    # cannot destroy the API spend (a 401 on repos/create used to do exactly
    # that). Pushing is opt-in and off by default: repo_id is derived from the
    # model name, so every topic would target the same repo, and
    # DatasetDict.push_to_hub deletes splits absent from the dict it pushes —
    # uploading KG would drop ASR.
    # NB: not ``output_dir`` — pydrantic.RunConfig already defines that and uses
    # it as the root for its own per-run config dumps, so reusing the name makes
    # every run scatter a <timestamp>-rewrite/ directory in here.
    save_dir: str = DEFAULT_OUTPUT_DIR
    push: bool = False

    def run(self):
        if self.topic not in TOPIC_TO_IDS:
            raise ValueError(
                f"topic must be one of {sorted(TOPIC_TO_IDS)}, got {self.topic!r}"
            )
        ids = TOPIC_TO_IDS[self.topic]
        dataset = load_dataset("allenai/qasper", split="train", trust_remote_code=True)
        df = dataset.to_pandas()
        df = df[df["id"].isin(ids)]
        papers = df.to_dict(orient="records")

        if self.limit is not None:
            papers = papers[:self.limit]

        requests = []
        for paper in papers:
            qas = paper["qas"]
            for question, answer in zip(qas["question"], qas["answers"]):
                answer = answer["answer"][0]
                if answer["unanswerable"]:
                    continue
                requests.append(
                    {
                        "paper_id": paper["id"],
                        "question": question,
                        "title": paper["title"],
                        "abstract": paper["abstract"],
                        "answer_data": answer,
                        "old_answer": answer,
                    }
                )

        # Run async rewriting
        async def process():
            print(f"Rewriting {len(requests)} questions...")
            new_questions = await rewrite_questions(requests, self.model)
            print("Questions rewritten. Rewriting answers...")
            new_answers = await rewrite_answers(requests, new_questions, self.model)
            print("Answers rewritten. Saving...")

            rewritten: List[RewrittenQasperQuestion] = []
            for req, new_q, new_a in zip(requests, new_questions, new_answers):
                rewritten.append(
                    RewrittenQasperQuestion(
                        paper_id=req["paper_id"],
                        title=req["title"],
                        abstract=req["abstract"],
                        question=new_q,
                        answer=new_a,
                        
                        old_question=req["question"],
                        old_answer=req["old_answer"],
                    )
                )
            return rewritten
        rewritten = asyncio.run(process())
       
        for question in random.sample(rewritten, min(5, len(rewritten))):
            print(f"Question: {question.question}")
            print(f"Old question: {question.old_question}")
            print(f"Answer: {question.answer}")
            print("-"*100)

        data = [asdict(r) for r in rewritten]

        # --- save locally, before anything that can fail over the network ---
        out_root = Path(self.save_dir)
        raw_path = out_root / "raw" / f"{self.topic}_raw.parquet"
        eval_path = out_root / "eval" / f"qasper_eval_{self.topic}.parquet"
        for path in (raw_path, eval_path):
            path.parent.mkdir(parents=True, exist_ok=True)

        # Provenance: the rewritten pair alongside the originals it came from.
        pd.DataFrame(data).to_parquet(raw_path, index=False)

        # Eval set, in the same layout as qasper_eval_{QA,MT,SA}.parquet.
        pd.DataFrame(
            [
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
                for r in data
            ]
        ).to_parquet(eval_path, index=False)

        print(f"Wrote {len(data)} rows to {eval_path}")
        print(f"Wrote provenance to {raw_path}")

        if self.push:
            from datasets import Dataset, DatasetDict

            repo_id = f"Phudish/qasper-rewrite-{self.model}"
            DatasetDict({self.topic: Dataset.from_list(data)}).push_to_hub(
                repo_id, private=False
            )
            print(f"Pushed to {repo_id}")


if __name__ == "__main__":
    # Override on the CLI, e.g.
    #   python cartridges/data/qasper/rewrite.py topic=ASR
    # QA/MT/SA were produced with gpt-4.1, which resolves to this snapshot; keep
    # it pinned so ASR and KG are generated the same way as the other three.
    config = RewriteQasperConfig(
        topic="ASR",
        model="gpt-4.1-2025-04-14",
    )
    pydrantic.main([config])