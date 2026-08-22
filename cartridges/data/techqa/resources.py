"""TechQA technotes grouped into five continual-learning phases.

TechQA is IBM technical support: user questions answered from IBM Technotes.
This uses ``nvidia/TechQA-RAG-Eval``, where each row is a question with a list
of ``contexts``, and each context is ``{filename, text}`` naming one technote.
496 distinct technotes back 610 answerable questions; the remaining 300 rows are
``is_impossible`` with an empty context list, so they belong to no phase and are
excluded here (they are still useful as an unanswerable/hallucination probe).

Phase construction follows the plan doc — disjoint sets of filenames — with one
adjustment forced by the data. Technotes are small (median 737 tokens, p75
1,222), so one filename per phase would be a few hundred tokens and defeat the
point of compaction; a phase is a *set* of filenames instead.

Assignment is longest-processing-time bin packing: documents sorted by token
count descending (filename as tie-break), each placed into the currently
smallest phase. That matters because the size distribution is heavy-tailed — one
technote is 44,383 tokens and three more clear 20k. Simple round-robin over the
sorted list leaves a 26.3% spread between the largest and smallest phase; LPT
brings it to 0.05% while keeping every document, including the outliers.

A product-based split was considered and rejected: WebSphere alone accounts for
145 documents and 293k tokens, Tivoli 28 documents and 125k, and everything else
falls off sharply (MQ 21/20k, DB2 18/12k). No balanced five-way split exists
along that axis.

Phase sizes (Qwen3-4B tokenizer):

    phase 1   99 docs  149,727 tokens  116 questions
    phase 2   98 docs  149,662 tokens  123 questions
    phase 3   99 docs  149,662 tokens  129 questions
    phase 4  100 docs  149,662 tokens  123 questions
    phase 5  100 docs  149,659 tokens  119 questions
    total    496 docs  748,372 tokens  610 questions

These are the largest phases of the five datasets; a single phase exceeds
Llama-3.2-3B's 131k context window, so the in-context baseline needs the
summarization branch here even for one phase.

The filename-to-phase assignment is frozen in ``phases.json`` next to this
module. Regenerate it with ``build_phases()`` if the packing inputs change.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import random

from datasets import load_dataset

from cartridges.data.resources import Resource, sample_seed_prompts, SEED_TYPES

TECHQA_DATASET = "nvidia/TechQA-RAG-Eval"
TECHQA_SPLIT = "train"
NUM_PHASES = 5

_PHASES_PATH = os.path.join(os.path.dirname(__file__), "phases.json")


def _load_phase_table() -> Dict[int, List[str]]:
    with open(_PHASES_PATH) as f:
        return {int(k): v for k, v in json.load(f).items()}


# phase -> list of technote filenames
PHASE_TO_FILENAMES: Dict[int, List[str]] = _load_phase_table()

SYSTEM_PROMPT_TEMPLATE = """\
Below are IBM technical support documents. Please read them and be prepared to
answer support questions.
<documents>
{documents}
</documents>
"""

DOCUMENT_TEMPLATE = """\
<document>
<source>{filename}</source>
{text}
</document>
"""


@dataclass
class Technote:
    filename: str
    text: str

    @property
    def to_string(self) -> str:
        return DOCUMENT_TEMPLATE.format(filename=self.filename, text=self.text)


@dataclass
class TechQAQuestion:
    question_id: str
    question: str
    answer: str
    filenames: List[str]


def load_corpus() -> tuple[Dict[str, Technote], List[TechQAQuestion]]:
    """Load every technote and every answerable question.

    Technotes repeat across the questions that cite them, so documents are
    collapsed on filename. ``is_impossible`` rows carry no contexts and are
    dropped.
    """
    documents: Dict[str, Technote] = {}
    questions: List[TechQAQuestion] = []
    for row in load_dataset(TECHQA_DATASET, split=TECHQA_SPLIT):
        filenames = []
        for context in row["contexts"]:
            filenames.append(context["filename"])
            if context["filename"] not in documents:
                documents[context["filename"]] = Technote(
                    filename=context["filename"], text=context["text"]
                )
        if row["is_impossible"] or not filenames:
            continue
        questions.append(
            TechQAQuestion(
                question_id=row["id"],
                question=row["question"],
                answer=row["answer"],
                filenames=filenames,
            )
        )
    return documents, questions


def questions_for_phase(phase: int) -> List[TechQAQuestion]:
    """Questions whose every cited technote lives in ``phase``.

    Questions cite at most one document in this release, so this is just a
    filter, but the all() guard keeps it correct if that ever changes.
    """
    wanted = set(PHASE_TO_FILENAMES[phase])
    _, questions = load_corpus()
    return [q for q in questions if q.filenames and all(f in wanted for f in q.filenames)]


def build_phases(tokenizer, num_phases: int = NUM_PHASES) -> Dict[int, List[str]]:
    """Recompute the phase assignment by LPT bin packing.

    Deterministic: documents are ordered by token count descending with the
    filename as tie-break, and ties between equally loaded phases go to the
    lower-numbered phase.
    """
    documents, _ = load_corpus()
    lengths = {
        name: len(tokenizer(doc.text, add_special_tokens=False)["input_ids"])
        for name, doc in documents.items()
    }
    order = sorted(documents, key=lambda name: (-lengths[name], name))

    phases: Dict[int, List[str]] = {p: [] for p in range(1, num_phases + 1)}
    load: Dict[int, int] = {p: 0 for p in range(1, num_phases + 1)}
    for name in order:
        target = min(load, key=lambda p: (load[p], p))
        phases[target].append(name)
        load[target] += lengths[name]
    return {p: sorted(names) for p, names in phases.items()}


def approx_token_count(text: str) -> int:
    """Conservative token estimate without a tokenizer (≈3 chars/token)."""
    return max(1, (len(text or "") + 2) // 3)


def documents_within_token_budget(
    documents: List[Technote], max_tokens: int
) -> List[Technote]:
    """Drop notes that cannot fit in the vLLM context window."""
    return [d for d in documents if approx_token_count(d.text) <= max_tokens]


class TechQAResource(Resource):
    """One phase of TechQA technotes, for self-study synthesis."""

    class Config(Resource.Config):
        phase: int = 1
        seed_prompts: List[SEED_TYPES] = ["generic"]
        # Documents average ~1.5k tokens; a handful per prompt gives the
        # generator enough context without blowing past the chunk budget.
        docs_per_prompt: int = 4
        # Leave headroom for chat wrapper + generation vs vLLM --max-model-len.
        max_prompt_tokens: int = 60000

    def __init__(self, config: Config):
        self.config = config
        if config.phase not in PHASE_TO_FILENAMES:
            raise ValueError(
                f"phase must be one of {sorted(PHASE_TO_FILENAMES)}, got {config.phase}"
            )
        documents, _ = load_corpus()
        missing = [f for f in PHASE_TO_FILENAMES[config.phase] if f not in documents]
        if missing:
            raise ValueError(f"technotes missing from the dataset: {missing[:5]}")
        phase_docs = [documents[f] for f in PHASE_TO_FILENAMES[config.phase]]
        kept = documents_within_token_budget(phase_docs, config.max_prompt_tokens)
        dropped = len(phase_docs) - len(kept)
        if dropped:
            print(
                f"TechQA phase {config.phase}: dropped {dropped} notes over "
                f"{config.max_prompt_tokens} tokens "
                f"({len(kept)} remaining)"
            )
        if not kept:
            raise ValueError(
                f"no TechQA notes in phase {config.phase} fit in "
                f"{config.max_prompt_tokens} tokens"
            )
        self.documents = kept

    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        num = random.randint(1, min(self.config.docs_per_prompt, len(self.documents)))
        documents = random.sample(self.documents, num)
        ctx = SYSTEM_PROMPT_TEMPLATE.format(
            documents="\n".join(d.to_string for d in documents)
        )
        seed_prompts = sample_seed_prompts(self.config.seed_prompts, batch_size)
        return ctx, seed_prompts

    def to_string(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(
            documents="\n".join(d.to_string for d in self.documents)
        )
