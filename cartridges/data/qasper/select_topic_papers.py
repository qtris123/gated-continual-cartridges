#!/usr/bin/env python3
"""Pick QASPER papers for a topic by keyword-matching title and abstract.

``TOPIC_TO_IDS`` in ``resources.py`` holds a hand-curated panel of papers per
topic (QA / MT / SA). This script builds the same kind of panel for topics that
were not curated by hand — summarization and speech recognition — so the five
QASPER phases are comparable in size and composition.

Matching is deliberately title-weighted: a paper whose *title* names the task is
almost certainly about that task, whereas an abstract mention is often
incidental ("we also report ROUGE", "extractive question answering"). Exclude
patterns kill the systematic false-positive families (hate speech and
part-of-speech for ASR, extractive QA for summarization).

Papers are ranked by topical fit, then by how many answerable questions QASPER
provides for them, since those questions are what ``rewrite.py`` turns into the
evaluation set.

Usage:

  # inspect the ranking for one topic
  python cartridges/data/qasper/select_topic_papers.py --topic SUMM --show 40

  # emit the block to paste into resources.py
  python cartridges/data/qasper/select_topic_papers.py --topic SUMM --emit-python

  # stats for the already-curated topics, for comparison
  python cartridges/data/qasper/select_topic_papers.py --stats
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import List

from datasets import load_dataset

from cartridges.data.qasper.resources import TOPIC_TO_IDS

QASPER_DATASET = "allenai/qasper"
QASPER_REVISION = "refs/convert/parquet"
QASPER_SPLIT = "train"


@dataclass
class TopicQuery:
    """Keyword spec for one topic.

    Attributes:
        strong: patterns that name the task outright. A title hit is decisive.
        weak: supporting vocabulary — evidence, but not on its own enough.
        exclude: patterns that mark a systematic false positive; any hit in the
            title or abstract disqualifies the paper.
    """

    strong: List[str]
    weak: List[str]
    exclude: List[str]


TOPIC_QUERIES = {
    # Text summarization. "extractive"/"abstractive" alone are weak because
    # extractive question answering and abstractive dialogue generation use the
    # same words; the exclusions drop the reading-comprehension papers that
    # merely evaluate with ROUGE.
    "SUMM": TopicQuery(
        strong=[r"summariz", r"summaris", r"\bsummar(y|ies)\b"],
        weak=[r"abstractive", r"extractive", r"\bROUGE\b", r"headline generation",
              r"\bTL;DR\b", r"salien(ce|t) sentence", r"content selection"],
        # Sentiment and translation are their own phases; a paper that summarizes
        # *in service of* sentiment analysis belongs to neither cleanly, so drop it.
        exclude=[r"extractive question answering", r"extractive (QA|reading)",
                 r"sentiment analysis", r"sentiment classification",
                 r"machine translation"],
    ),
    # Automatic speech recognition and closely-adjacent spoken-language tasks.
    # "speech" alone is a trap: hate speech, part-of-speech and speech acts are
    # all far more common in QASPER than acoustics. Speech *translation* is
    # excluded too — it is half an MT paper, and MT is its own phase, so letting
    # it in would blur the boundary between two phases of the same experiment.
    "ASR": TopicQuery(
        strong=[r"speech recognition", r"\bASR\b", r"speech-to-text",
                r"speech to text", r"acoustic model", r"end-to-end speech",
                r"spoken language"],
        weak=[r"\bphonem", r"\bphonetic", r"word error rate", r"\bWER\b",
              r"\bspeech corpus\b", r"\baudio\b", r"\bspeech\b", r"\butterance"],
        exclude=[r"hate speech", r"offensive speech", r"abusive speech",
                 r"part[- ]of[- ]speech", r"\bPOS tag", r"speech act",
                 r"figurative speech", r"speech emotion", r"parts of speech",
                 r"speech translation", r"speech-to-speech", r"automatic dubbing"],
    ),
    # Knowledge graphs / knowledge bases: link prediction, embedding models,
    # entity linking, KB completion. The nearest existing phase is QA, via KBQA,
    # so knowledge-base question answering is excluded outright.
    "KG": TopicQuery(
        strong=[r"knowledge graph", r"knowledge base", r"entity linking",
                r"link prediction", r"\bontolog", r"\bKB completion\b",
                r"knowledge base completion"],
        weak=[r"\btriple", r"\bentit(y|ies)\b", r"relation extraction",
              r"\bembedding", r"\bTransE\b", r"graph attention", r"\bWikidata\b",
              r"\bFreebase\b", r"\bDBpedia\b"],
        exclude=[r"knowledge.base question answering", r"\bKBQA\b",
                 r"question answering over", r"dialogue state tracking"],
    ),
    # Dialogue systems: state tracking, task-oriented agents, response
    # generation. Highest paper count in QASPER but the softest boundaries —
    # dialogue papers routinely also do QA, generation or summarization.
    "DIALOG": TopicQuery(
        strong=[r"\bdialog(ue)?\b", r"conversational agent", r"\bchatbot\b",
                r"conversational (system|model|response)"],
        weak=[r"\butterance", r"response generation", r"\bturn-level\b",
              r"state tracking", r"\bintent\b", r"\bslot\b"],
        exclude=[r"summariz", r"summaris", r"speech recognition",
                 r"question answering"],
    ),
}


def _hits(patterns: List[str], text: str) -> List[str]:
    return [p for p in patterns if re.search(p, text, re.IGNORECASE)]


def _word_count(row) -> int:
    n = len(row["abstract"].split())
    for paragraphs in row["full_text"]["paragraphs"]:
        for paragraph in paragraphs:
            n += len(paragraph.split())
    return n


def _answerable_qas(row) -> int:
    """Questions QASPER marks answerable — the ones rewrite.py keeps."""
    n = 0
    for answer in row["qas"]["answers"]:
        entries = answer["answer"]
        if len(entries) and not entries[0]["unanswerable"]:
            n += 1
    return n


@dataclass
class Candidate:
    paper_id: str
    title: str
    words: int
    qas: int
    title_strong: List[str]
    title_weak: List[str]
    abstract_strong: List[str]
    abstract_weak: List[str]
    taken_by: str | None

    @property
    def score(self) -> int:
        # A title naming the task dominates everything else; abstract evidence
        # only separates papers that already agree on the title signal.
        return (
            100 * len(self.title_strong)
            + 10 * len(self.title_weak)
            + 3 * len(self.abstract_strong)
            + len(self.abstract_weak)
        )

    @property
    def eligible(self) -> bool:
        return (
            self.taken_by is None
            and bool(self.title_strong)
            and bool(self.abstract_strong or self.abstract_weak)
        )


def load_papers():
    dataset = load_dataset(QASPER_DATASET, split=QASPER_SPLIT, revision=QASPER_REVISION)
    return dataset.to_pandas().to_dict(orient="records")


def rank(topic: str, papers) -> List[Candidate]:
    query = TOPIC_QUERIES[topic]
    owner = {pid: name for name, ids in TOPIC_TO_IDS.items() for pid in ids}

    candidates = []
    for row in papers:
        title, abstract = row["title"], row["abstract"]
        blob = f"{title}\n{abstract}"
        if _hits(query.exclude, blob):
            continue

        candidate = Candidate(
            paper_id=row["id"],
            title=title,
            words=_word_count(row),
            qas=_answerable_qas(row),
            title_strong=_hits(query.strong, title),
            title_weak=_hits(query.weak, title),
            abstract_strong=_hits(query.strong, abstract),
            abstract_weak=_hits(query.weak, abstract),
            taken_by=owner.get(row["id"]),
        )
        if candidate.score:
            candidates.append(candidate)

    candidates.sort(key=lambda c: (-c.score, -c.qas, -c.words, c.paper_id))
    return candidates


def select(candidates: List[Candidate], n: int) -> List[Candidate]:
    """Choose the panel from the eligible candidates.

    Eligibility already requires the title to name the task, so the remaining
    score differences are mostly noise (a paper does not become more about ASR
    for saying "end-to-end speech" as well as "speech recognition"). What does
    differ materially is how many answerable questions QASPER ships for the
    paper, and those questions are the evaluation set — so order by that first
    and keep the fit score only as a tie-break.
    """
    eligible = [c for c in candidates if c.eligible]
    eligible.sort(key=lambda c: (-c.qas, -c.score, -c.words, c.paper_id))
    return eligible[:n]


def print_ranking(candidates: List[Candidate], selected: List[Candidate], show: int) -> None:
    chosen = {c.paper_id for c in selected}
    print(f"{'':1}{'paper_id':<12} {'score':>5} {'qas':>4} {'words':>6}  title")
    print("-" * 100)
    for candidate in candidates[:show]:
        if candidate.paper_id in chosen:
            mark = "+"  # in the emitted panel
        elif candidate.taken_by:
            mark = "="  # already curated under another topic
        elif candidate.eligible:
            mark = " "  # eligible but cut
        else:
            mark = "~"  # matched, but not on title — needs a human look
        print(
            f"{mark}{candidate.paper_id:<12} {candidate.score:>5} {candidate.qas:>4} "
            f"{candidate.words:>6}  {candidate.title[:66]}"
        )
    print("\n  + selected   (blank) eligible but cut   = already in TOPIC_TO_IDS"
          "   ~ abstract-only match")


def topic_stats(papers, ids: List[str], label: str) -> None:
    by_id = {row["id"]: row for row in papers}
    rows = [by_id[pid] for pid in ids if pid in by_id]
    if not rows:
        print(f"{label:<8} no papers found in the {QASPER_SPLIT} split")
        return
    words = [_word_count(r) for r in rows]
    qas = [_answerable_qas(r) for r in rows]
    print(
        f"{label:<8} papers={len(rows):>3}  words: total={sum(words):>7,} "
        f"mean={sum(words) // len(words):>6,}  answerable_qas: total={sum(qas):>4} "
        f"mean={sum(qas) / len(qas):>4.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--topic", choices=sorted(TOPIC_QUERIES), help="topic to rank")
    parser.add_argument("--n", type=int, default=16,
                        help="panel size to emit (default matches the curated topics)")
    parser.add_argument("--show", type=int, default=30, help="rows to print")
    parser.add_argument("--emit-python", action="store_true",
                        help="print the TOPIC_TO_IDS entry for the top --n papers")
    parser.add_argument("--stats", action="store_true",
                        help="print size stats for every curated topic")
    args = parser.parse_args()

    papers = load_papers()
    print(f"loaded {len(papers)} QASPER {QASPER_SPLIT} papers\n")

    if args.stats:
        for label, ids in TOPIC_TO_IDS.items():
            topic_stats(papers, ids, label)
        return

    if not args.topic:
        parser.error("pass --topic or --stats")

    candidates = rank(args.topic, papers)
    eligible = [c for c in candidates if c.eligible]
    selected = select(candidates, args.n)
    print(f"topic={args.topic}  matched={len(candidates)}  eligible={len(eligible)}\n")
    print_ranking(candidates, selected, args.show)

    if len(selected) < args.n:
        print(f"\n!! only {len(selected)} eligible papers, asked for {args.n}")

    print()
    topic_stats(papers, [c.paper_id for c in selected], f"{args.topic}[{len(selected)}]")

    if args.emit_python:
        print(f'\n    "{args.topic}": [')
        for candidate in selected:
            print(f"        '{candidate.paper_id}',  # {candidate.title[:64]}")
        print("    ],")


if __name__ == "__main__":
    main()
