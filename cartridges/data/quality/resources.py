"""QuALITY articles grouped into five continual-learning phases.

QuALITY is multiple-choice reading comprehension over long narratives: 265
documents and 4,609 questions, mostly 1950s-60s pulp science fiction from
Project Gutenberg plus a tail of late-1990s Slate journalism. One row of the HF
dataset is one *question* — the full article text is repeated on every row that
belongs to it — so ``article_id`` is the document key.

Phase construction: articles of at least ``MIN_ARTICLE_TOKENS`` tokens, the 35
longest, assigned to five phases by longest-processing-time bin packing (the
same routine FinQA and TechQA use — see ``cartridges/data/PHASES.md``).

These articles are near-uniform in size, 7,442 to 8,445 tokens, a 1.13x ratio
against TechQA's 608x, so simple round-robin dealing already balanced them to
1.68%. LPT is used anyway because it is never worse and here is 48x tighter,
at 0.03%.

Why length and not authorship. The plan doc proposed grouping by ``writer_id``,
but that column is the crowdworker who wrote the *questions*, not the author of
the story (the dataset carries ``author`` separately, and every article has
exactly two question writers). Holding ``writer_id`` fixed therefore mixes pulp
science fiction and Slate film criticism inside one phase while holding only the
question phrasing constant. Grouping by ``author`` is coherent but only six
authors have five or more articles, and their totals range from 21k to 74k
tokens — a 4x spread with no room to rebalance. Filtering on length instead
matches the review comment on the doc ("docs being from the same author doesn't
matter too much ... we can filter for docs over 5k or 6k tokens only") and
balances the phases to within 2%.

Each article is self-contained fiction, so forgetting stays measurable
regardless of how the buckets are drawn: the question is whether the model can
still answer about phase-1 *stories* after training through phase 5.

Phase sizes (Qwen3-4B tokenizer, raw article text):

    phase 1   54,708 tokens   124 questions
    phase 2   54,720 tokens   128 questions
    phase 3   54,716 tokens   122 questions
    phase 4   54,701 tokens   126 questions
    phase 5   54,703 tokens   122 questions
    total    273,548 tokens   622 questions

Regenerate the lists below with ``build_phases()`` if MIN_ARTICLE_TOKENS or
ARTICLES_PER_PHASE changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import random

from datasets import load_dataset

from cartridges.data.resources import Resource, sample_seed_prompts, SEED_TYPES

QUALITY_DATASET = "tasksource/QuALITY"
QUALITY_SPLITS = ("train", "validation")

# The review comment asked for 5k-6k; 6k keeps the phases inside Gutenberg
# (Slate articles top out around 4.3k tokens) so phase length stays uniform.
MIN_ARTICLE_TOKENS = 6000
ARTICLES_PER_PHASE = 7
NUM_PHASES = 5

# article_id -> phase. Selected by build_phases(); see the module docstring.
PHASE_TO_ARTICLE_IDS: Dict[int, List[str]] = {
    1: [
        '51651',  #  8,445 tok  17q  Miller, Walter M.     Conditionally Human
        '22867',  #  8,036 tok  20q  Nourse, Alan Edward   Meeting of the Board
        '26843',  #  7,865 tok  15q  Sharkey, Jack         The Dope on Mars
        '24161',  #  7,702 tok  16q  Kuykendall, Roger     All Day September
        '50103',  #  7,658 tok  19q  Del Rey, Lester       The Dwindling Years
        '31599',  #  7,509 tok  20q  Aycock, Roger D.      To Remember Charlie By
        '62569',  #  7,493 tok  17q  Bradbury, Ray         The Monster Maker
    ],
    2: [
        '22590',  #  8,413 tok  18q  Fontenay, Charles L.  Wind
        '61434',  #  8,098 tok  18q  Laumer, Keith         Mightiest Qorn
        '51656',  #  7,799 tok  16q  Smith, Richard Rein   Pick a Crime
        '31282',  #  7,716 tok  18q  Browne, Howard        Mars Confidential
        '22524',  #  7,662 tok  20q  Samachson, Joseph     The Hunters
        '22579',  #  7,583 tok  19q  Leiber, Fritz         Bread Overhead
        '49897',  #  7,449 tok  19q  Gunn, James E.        The Gravity Business
    ],
    3: [
        '23791',  #  8,260 tok  15q  Leinster, Murray      Scrimshaw
        '22876',  #  8,110 tok  20q  Nourse, Alan Edward   The Link
        '51350',  #  8,017 tok  17q  Harmon, Jim           No Substitutions
        '61263',  #  7,671 tok  16q  Laumer, Keith         Cultural Exchange
        '51609',  #  7,623 tok  18q  Lee, Stanley R.       A Fall of Glass
        '61139',  #  7,593 tok  17q  Laumer, Keith         The Madman From Earth
        '51361',  #  7,442 tok  19q  Silverberg, Robert    Birds of a Feather
    ],
    4: [
        '51433',  #  8,239 tok  19q  Neville, Kris         Hunt the Hunter
        '51296',  #  8,141 tok  17q  Marlowe, Stephen      The Sense of Wonder
        '51344',  #  7,914 tok  14q  Neville, Kris         Voyage to Far N'jurd
        '53269',  #  7,798 tok  20q  Coombs, Charles Ira   Atom Mystery
        '22218',  #  7,599 tok  20q  Jacobi, Carl          The Street That Wasn't There
        '27492',  #  7,535 tok  18q  Stecher, L. J., Jr.   Upstarts
        '51650',  #  7,475 tok  18q  Anderson, Poul        Innocent at Large
    ],
    5: [
        '51256',  #  8,204 tok  18q  Fetler, Andrew        The Cool War
        '24275',  #  8,172 tok  15q  Nourse, Alan Edward   Letter of the Law
        '51351',  #  7,919 tok  17q  Harmon, Jim           The Spicy Sound of Success
        '51699',  #  7,781 tok  16q  Doede, William R.     The God Next Door
        '50969',  #  7,614 tok  18q  Wallace, F. L.        Big Ancestor
        '51274',  #  7,551 tok  20q  Bade, William L.      Ambition
        '51267',  #  7,462 tok  18q  Laumer, Keith         End as a Hero
    ],
}


@dataclass
class Article:
    article_id: str
    title: str
    author: str
    source: str
    year: Optional[float]
    topic: str
    text: str

    @property
    def to_string(self) -> str:
        return ARTICLE_TEMPLATE.format(
            title=self.title, author=self.author, text=self.text
        )


ARTICLE_TEMPLATE = """\
<title>{title}</title>
<author>{author}</author>
<text>
{text}
</text>
"""

SYSTEM_PROMPT_TEMPLATE = """\
Below is (part of) a story. Please read it and be prepared to answer questions.
<story>
{story}
</story>
"""


def load_articles(article_ids: Optional[List[str]] = None) -> Dict[str, Article]:
    """Load QuALITY articles keyed by ``article_id``, deduplicated across rows.

    The HF dataset stores one row per question with the article text repeated,
    so this collapses to one entry per document.
    """
    wanted = set(article_ids) if article_ids is not None else None
    articles: Dict[str, Article] = {}
    for split in QUALITY_SPLITS:
        for row in load_dataset(QUALITY_DATASET, split=split):
            aid = str(row["article_id"])
            if aid in articles or (wanted is not None and aid not in wanted):
                continue
            articles[aid] = Article(
                article_id=aid,
                title=row["title"],
                author=row["author"],
                source=row["source"],
                year=row["year"],
                topic=row["topic"],
                text=row["article"],
            )
    if wanted is not None:
        missing = wanted - set(articles)
        if missing:
            raise ValueError(f"article_ids not found in QuALITY: {sorted(missing)}")
    return articles


def build_phases(
    tokenizer,
    min_tokens: int = MIN_ARTICLE_TOKENS,
    articles_per_phase: int = ARTICLES_PER_PHASE,
    num_phases: int = NUM_PHASES,
) -> Dict[int, List[str]]:
    """Recompute ``PHASE_TO_ARTICLE_IDS`` by LPT bin packing.

    Deterministic: articles are ordered by token count descending with the
    ``article_id`` as tie-break, each placed into the currently smallest phase,
    and ties between equally loaded phases go to the lower-numbered phase.
    """
    articles = load_articles()
    lengths = {
        aid: len(tokenizer(a.text, add_special_tokens=False)["input_ids"])
        for aid, a in articles.items()
    }
    eligible = sorted(
        (aid for aid, n in lengths.items() if n >= min_tokens),
        key=lambda aid: (-lengths[aid], aid),
    )
    needed = articles_per_phase * num_phases
    if len(eligible) < needed:
        raise ValueError(
            f"only {len(eligible)} articles clear {min_tokens} tokens, need {needed}"
        )

    phases: Dict[int, List[str]] = {p: [] for p in range(1, num_phases + 1)}
    load: Dict[int, int] = {p: 0 for p in range(1, num_phases + 1)}
    for aid in eligible[:needed]:
        target = min(load, key=lambda p: (load[p], p))
        phases[target].append(aid)
        load[target] += lengths[aid]
    # Keep each phase in descending-length order so the committed table reads
    # the same way it was built.
    return {
        p: sorted(aids, key=lambda aid: (-lengths[aid], aid))
        for p, aids in phases.items()
    }


class QuALITYResource(Resource):
    """One phase of QuALITY, for self-study synthesis.

    Mirrors ``QASPERResource``: sample a random article, wrap it in the system
    prompt, and pair it with seed prompts.
    """

    class Config(Resource.Config):
        phase: int = 1
        seed_prompts: List[SEED_TYPES] = ["generic"]

    def __init__(self, config: Config):
        self.config = config
        if config.phase not in PHASE_TO_ARTICLE_IDS:
            raise ValueError(
                f"phase must be one of {sorted(PHASE_TO_ARTICLE_IDS)}, got {config.phase}"
            )
        article_ids = PHASE_TO_ARTICLE_IDS[config.phase]
        articles = load_articles(article_ids)
        # Preserve the declared order so runs are reproducible.
        self.articles = [articles[aid] for aid in article_ids]

    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        article = random.choice(self.articles)
        ctx = SYSTEM_PROMPT_TEMPLATE.format(story=article.to_string)
        seed_prompts = sample_seed_prompts(self.config.seed_prompts, batch_size)
        return ctx, seed_prompts

    def to_string(self) -> str:
        out = "Below is a collection of stories."
        for article in self.articles:
            out += "\n\n"
            out += f"<story>\n{article.to_string}\n</story>\n"
        return out
