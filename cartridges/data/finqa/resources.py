"""FinQA filings grouped into five continual-learning phases.

FinQA pairs numerical-reasoning questions with pages of corporate filings. Each
example carries a ``filename`` of the form ``COMPANY/YEAR/page_N.pdf`` plus the
page itself split into ``pre_text``, a ``table``, and ``post_text``. A document
is one *page*, roughly 1k tokens — far smaller than the 10-Ks used in the
financebench experiments — so a phase has to bundle many of them.

Phase construction holds the **fiscal year fixed and varies the company**, the
alternative raised in review ("we had some empirical trouble in the past with
this, lots of hallucination; why not same year different companies?"). The plan
doc originally proposed the opposite — one company across five years — and that
is defensible on the data: 163 of 172 Lockheed questions (94.8%) name their year
explicitly, and only two question texts recur across years once years are
masked. It was dropped anyway because the residual risk is not worth carrying:
nine Lockheed questions name no year at all, and a cartridge trained on five
near-identical filings is exactly the setting where cross-year figures blur.

Varying the company makes FinQA a knowledge-*accumulation* stream, the same
shape as LongHealth (different patients) and QuALITY (different stories), and
removes year ambiguity entirely.

FY2017 is used because it has the largest inventory of any year — 71 companies,
241 pages, 737 questions — which is what lets each phase reach ~46k tokens. An
earlier draft took only the five *largest* companies of FY2012, one per phase,
and that left phases at 9-13k tokens: four to fifteen times smaller than the
other datasets, and only ~10x compression against a 1024-slot cartridge. Packing
whole companies instead brings FinQA in line with LongHealth (~46k) without
giving up the fixed-year property.

Companies are the atomic unit — every page of a company stays in one phase — and
are assigned by longest-processing-time bin packing: sorted by token count
descending with the ticker as tie-break, each placed into the currently smallest
phase. That balances the phases to within 0.96%. Packing is measured on
``FinQAPage.text``, i.e. including the per-page ``<page><source>`` wrapper, so
the balance holds for the string that is actually prefilled.

Phase sizes (Qwen3-4B tokenizer):

    phase 1  13 companies  52 pages  47,232 tokens  155 questions
    phase 2  13 companies  44 pages  46,778 tokens  116 questions
    phase 3  15 companies  46 pages  47,196 tokens  150 questions
    phase 4  15 companies  48 pages  47,114 tokens  150 questions
    phase 5  15 companies  51 pages  47,100 tokens  166 questions
    total    71 companies 241 pages 235,420 tokens  737 questions

That is the highest question count per phase of the five datasets. Question
counts are balanced only as a side effect of balancing tokens, so phase 2 is
light at 116; if that matters more than token balance, pack on question count
instead by swapping the ``lengths`` map in ``build_phases``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import random

import requests

from cartridges.data.resources import Resource, sample_seed_prompts, SEED_TYPES

FINQA_URL = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/{split}.json"
FINQA_SPLITS = ("train", "dev", "test")

# Every phase draws from this one fiscal year, so no question can be ambiguous
# about which year it refers to.
FISCAL_YEAR = "2017"
NUM_PHASES = 5

# phase -> company tickers. Produced by build_phases(); see the module docstring.
PHASE_TO_COMPANIES: Dict[int, List[str]] = {
    1: ["AAPL", "APD", "BDX", "BKR", "CE", "ETR", "GIS", "KHC", "RL", "RSG",
        "SLG", "SPGI", "TMUS"],
    2: ["AAL", "ALLE", "DISCA", "DVN", "EMR", "FTV", "GS", "HST", "HWM",
        "ILMN", "PM", "PNC", "UPS"],
    3: ["AES", "AMAT", "BKNG", "BLK", "C", "FIS", "HUM", "JKHY", "LMT", "MS",
        "STT", "TFX", "UNP", "VLO", "ZBH"],
    4: ["CAT", "ECL", "EOG", "FBHS", "HII", "INTC", "IPG", "MAA", "MAS",
        "NCLH", "ORLY", "SNPS", "TXN", "UAA", "WELL"],
    5: ["ANET", "AWK", "CDW", "CME", "EW", "EXPD", "GPN", "HFC", "JPM", "MO",
        "MRO", "NWS", "RCL", "RE", "TSCO"],
}

PAGE_TEMPLATE = """\
<page>
<source>{filename}</source>
{pre_text}

{table}

{post_text}
</page>
"""

SYSTEM_PROMPT_TEMPLATE = """\
Below are pages from the {year} financial filings of several companies. Please
read them and be prepared to answer questions that may require arithmetic over
the figures.
<filings>
{filings}
</filings>
"""


@dataclass
class FinQAPage:
    filename: str
    company: str
    year: str
    pre_text: List[str]
    post_text: List[str]
    table: List[List[str]]

    @property
    def text(self) -> str:
        table = "\n".join("\t".join(map(str, row)) for row in self.table)
        return PAGE_TEMPLATE.format(
            filename=self.filename,
            pre_text="\n".join(self.pre_text),
            table=table,
            post_text="\n".join(self.post_text),
        )


@dataclass
class FinQAQuestion:
    filename: str
    company: str
    year: str
    question: str
    answer: str
    program: Optional[str]


def _cache_dir() -> str:
    root = os.environ.get("CARTRIDGES_DIR", os.getcwd())
    path = os.path.join(root, "data", "finqa")
    os.makedirs(path, exist_ok=True)
    return path


def _load_raw() -> List[dict]:
    """Download the three FinQA splits once, then read from disk.

    ``train.json`` is ~78MB, so unlike the LongHealth loader this caches rather
    than re-fetching on every call.
    """
    rows: List[dict] = []
    for split in FINQA_SPLITS:
        path = os.path.join(_cache_dir(), f"{split}.json")
        if not os.path.exists(path):
            response = requests.get(FINQA_URL.format(split=split), timeout=120)
            response.raise_for_status()
            with open(path, "w") as f:
                f.write(response.text)
        with open(path) as f:
            rows.extend(json.load(f))
    return rows


def load_company_year(year: str = FISCAL_YEAR):
    """Group every FinQA row by company for one fiscal year.

    Returns ``{ticker: ([pages], [questions])}``. Several examples share a page
    — one per question — so pages are collapsed on filename while every question
    is kept.
    """
    pages: Dict[str, Dict[str, FinQAPage]] = {}
    questions: Dict[str, List[FinQAQuestion]] = {}
    for row in _load_raw():
        company, row_year, _ = row["filename"].split("/")
        if row_year != year:
            continue
        pages.setdefault(company, {})
        questions.setdefault(company, [])
        if row["filename"] not in pages[company]:
            pages[company][row["filename"]] = FinQAPage(
                filename=row["filename"],
                company=company,
                year=row_year,
                pre_text=row["pre_text"],
                post_text=row["post_text"],
                table=row["table"],
            )
        qa = row["qa"]
        questions[company].append(
            FinQAQuestion(
                filename=row["filename"],
                company=company,
                year=row_year,
                question=qa["question"],
                answer=qa["answer"],
                program=qa.get("program"),
            )
        )
    return {
        company: ([pages[company][f] for f in sorted(pages[company])], questions[company])
        for company in pages
    }


def load_phase(phase: int) -> tuple[List[FinQAPage], List[FinQAQuestion]]:
    """Return every page and question for the companies in ``phase``."""
    if phase not in PHASE_TO_COMPANIES:
        raise ValueError(
            f"phase must be one of {sorted(PHASE_TO_COMPANIES)}, got {phase}"
        )
    by_company = load_company_year()
    missing = [c for c in PHASE_TO_COMPANIES[phase] if c not in by_company]
    if missing:
        raise ValueError(f"no FinQA rows for {missing} in FY{FISCAL_YEAR}")

    pages: List[FinQAPage] = []
    questions: List[FinQAQuestion] = []
    for company in PHASE_TO_COMPANIES[phase]:
        company_pages, company_questions = by_company[company]
        pages.extend(company_pages)
        questions.extend(company_questions)
    return pages, questions


def build_phases(tokenizer, num_phases: int = NUM_PHASES) -> Dict[int, List[str]]:
    """Recompute ``PHASE_TO_COMPANIES`` by LPT bin packing over whole companies.

    Deterministic: companies are ordered by total token count descending with
    the ticker as tie-break, and ties between equally loaded phases go to the
    lower-numbered phase.
    """
    by_company = load_company_year()
    lengths = {
        company: sum(
            len(tokenizer(page.text, add_special_tokens=False)["input_ids"])
            for page in pages
        )
        for company, (pages, _) in by_company.items()
    }
    order = sorted(by_company, key=lambda c: (-lengths[c], c))

    phases: Dict[int, List[str]] = {p: [] for p in range(1, num_phases + 1)}
    load: Dict[int, int] = {p: 0 for p in range(1, num_phases + 1)}
    for company in order:
        target = min(load, key=lambda p: (load[p], p))
        phases[target].append(company)
        load[target] += lengths[company]
    return {p: sorted(companies) for p, companies in phases.items()}


class FinQAResource(Resource):
    """One phase of FinQA filings, for self-study synthesis."""

    class Config(Resource.Config):
        phase: int = 1
        seed_prompts: List[SEED_TYPES] = ["generic"]
        # Pages are ~1k tokens each and a single page rarely holds both a figure
        # and the context that explains it, so sample a few per prompt.
        pages_per_prompt: int = 4

    def __init__(self, config: Config):
        self.config = config
        self.year = FISCAL_YEAR
        self.companies = PHASE_TO_COMPANIES[config.phase]
        self.pages, self.questions = load_phase(config.phase)

    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        # Draw from one company at a time: mixing pages from unrelated filings
        # invites the generator to compare figures across companies, which no
        # eval question asks about.
        company = random.choice(self.companies)
        company_pages = [p for p in self.pages if p.company == company]
        num = random.randint(1, min(self.config.pages_per_prompt, len(company_pages)))
        pages = random.sample(company_pages, num)
        ctx = SYSTEM_PROMPT_TEMPLATE.format(
            year=self.year, filings="\n".join(p.text for p in pages)
        )
        seed_prompts = sample_seed_prompts(self.config.seed_prompts, batch_size)
        return ctx, seed_prompts

    def to_string(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(
            year=self.year, filings="\n".join(p.text for p in self.pages)
        )
