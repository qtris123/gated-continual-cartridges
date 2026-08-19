from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import random

from datasets import load_dataset

from cartridges.data.resources import Resource, sample_seed_prompts, SEED_TYPES


TOPIC_TO_IDS = {
    "QA": [ # Question Answering
        '1908.06606', #part1
        '1704.05572', #part1
        '1905.08949', #part1
        '1808.09920', #part1
        '1603.01417', #part1
        '1808.03986', #part1
        '1907.08501',
        '1603.07044',
        '1903.00172',
        '1912.01046',
        '1909.00542',
        '1811.08048',
        '2004.02393',
        '1703.06492',
        '1607.06275',
        '1703.04617'
    ],
    "MT": [ # Machine Translation
        '1905.11901',
        '1911.03310',
        '1910.11471',
        '1903.03467',
        '1911.00069',
        '2001.01589',
        '1806.00722',
        '1909.01013',
        '1910.10408',
        '1903.00058',
        '2002.08899',
        '2002.02427',
        '1610.05243',
        '1910.11768',
        '1810.03459',
        '1906.00378',
    ],
    "SA": [ # Sentiment Analysis (every paper below is sentiment, not semantics)
        '1808.05077',
        '1912.05066',
        '2001.07209',
        '1911.12569',
        '1904.07342',
        '1910.04006',
        '1807.07961',
        '1803.07771',
        '1611.09441',
        '1704.00939',
        '1801.02243',
        '2004.03925',
        '2003.04967',
        '1904.09678',
        '1710.01492',
        '1909.00088'
    ],
    # Selected by `select_topic_papers.py --topic ASR` rather than by hand: a
    # title naming the task, then ranked by how many answerable questions QASPER
    # ships (those become the eval set via rewrite.py). Speech *translation* is
    # deliberately excluded so the phase does not overlap MT.
    #
    # Caveat inherited from the hand-curated panels: MT already contains
    # '1810.03459' (multilingual seq2seq speech recognition), so a little ASR
    # content is present in the MT phase. Left as-is because the MT eval parquet
    # was generated from that list.
    "ASR": [ # Automatic Speech Recognition
        '1911.13087',  # Kurdish (Sorani) Speech to Text: Presenting an Experimental Dataset
        '1909.06937',  # CM-Net: Collaborative Memory Network for Spoken Language Understanding
        '2002.06675',  # Speech Corpus of Ainu Folklore and End-to-end Speech Recognition
        '1804.08050',  # Multi-Head Decoder for End-to-End Speech Recognition
        '1910.08502',  # End-to-End Speech Recognition: A review for the French Language
        '2001.05284',  # Improving Spoken Language Understanding By Exploiting ASR N-best
        '1904.05862',  # wav2vec: Unsupervised Pre-training for Speech Recognition
        '2002.11268',  # A Density Ratio Approach to LM Fusion in End-to-End ASR
        '2002.02562',  # Transformer Transducer: A Streamable Speech Recognition Model
        '1908.01060',  # Multilingual Speech Recognition with Corpus Relatedness Sampling
        '1807.00868',  # Exploring End-to-End Techniques for Low-Resource Speech Recognition
        '1910.04269',  # Spoken Language Identification using ConvNets
        '1707.07048',  # Progressive Joint Modeling in Single-channel Overlapped Speech Recognition
        '1910.14443',  # Multi-scale Octave Convolutions for Robust Speech Recognition
        '1909.06522',  # Multilingual Graphemic Hybrid ASR with Massive Data Augmentation
        '1909.13695',  # Non-native Speaker Verification for Spoken Language Assessment
    ],
    # Fifth phase. Chosen over summarization on measured distinctness: embedding
    # the panels and comparing centroids, a summarization panel sits closer to
    # the other four phases (mean cosine 0.65) than they sit to each other
    # (0.51-0.67), and the QA panel above already contains a summarisation paper
    # ('1909.00542'). Knowledge graphs come in at 0.60 with no such overlap.
    # KBQA is excluded by keyword so the phase does not lean on QA.
    "KG": [ # Knowledge Graphs / Knowledge Bases
        '1910.03891',  # Learning High-order Structural and Attribute info by KG Attention
        '1911.09419',  # Learning Hierarchy-Aware KG Embeddings for Link Prediction
        '1909.08191',  # Exploring Scholarly Data by Semantic Query on KG Embedding Space
        '1902.00330',  # Joint Entity Linking with Deep Reinforcement Learning
        '1811.01399',  # Logic Attention Based Neighborhood Aggregation for Inductive KGE
        '1909.08402',  # Enriching BERT with KG Embeddings for Document Classification
        '1712.03547',  # Inducing Interpretability in Knowledge Graph Embeddings
        '1905.00563',  # Investigating Robustness and Interpretability of Link Prediction
        '1906.11180',  # Canonicalizing Knowledge Base Literals
        '1911.03681',  # BERT is Not a Knowledge Base (Yet): Factual Knowledge vs Reasoning
        '1712.02121',  # A Novel Embedding Model for KB Completion Based on CNN
        '1909.01515',  # Meta Relational Learning for Few-Shot Link Prediction in KGs
        '1606.08140',  # STransE: embedding model of entities and relationships in KBs
        '1810.05320',  # Important Attribute Identification in Knowledge Graph
        '1601.00901',  # Joint learning of ontology and semantic parser from text
        '1908.06556',  # Transfer in Deep Reinforcement Learning using Knowledge Graphs
    ],
}

SYSTEM_PROMPT_TEMPLATE = """\
Below is (part of) a scientific paper. Please read it and be prepared to answer questions. 
<paper>
{paper}
</paper>
"""

class QASPERResource(Resource):
    class Config(Resource.Config):
        topic: str = "QA"
        seed_prompts: List[SEED_TYPES] = ["generic"]


    
    def __init__(self, config: Config):
        self.config = config

        dataset = load_dataset("allenai/qasper", split="train", revision="refs/convert/parquet")
        df = dataset.to_pandas()

        paper_ids = TOPIC_TO_IDS[self.config.topic]
        df = df[df["id"].isin(paper_ids)]
        assert len(df) == len(paper_ids)

        papers = []
        for row in df.to_dict(orient="records"):
            sections = []
            for section_idx, (section_title, paragraphs) in enumerate(zip(
                row["full_text"]["section_name"], row["full_text"]["paragraphs"]
            )):
                sections.append(Section(
                    title=section_title,
                    section_number=section_idx,
                    paragraphs=paragraphs.tolist()
                ))
            paper = Paper(
                id=row["id"],
                title=row["title"],
                abstract=row["abstract"],
                sections=sections
            )
            papers.append(paper)
        self.papers = papers
    
    # choose a random paper -> choose a random number of sections in that paper -> formulate a ctx out of them -> sample batch_size number of seed prompts -> get batch_size number of prompts from that ctx.
    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        paper: Paper = random.choice(self.papers)
        num_sections_per_paper = random.randint(1, len(paper.sections))
        sections = random.sample(paper.sections, num_sections_per_paper)
        sections_str = "\n".join([section.text for section in sections]) # section.tostring -> section.text

        section_divider = f"\n---Paper Title: {paper.title}---\n"
        context = PAPER_TEMPLATE.format(
            title=paper.title,
            abstract=paper.abstract,
            sections=section_divider.join([section.text for section in sections])
        )
        ctx = SYSTEM_PROMPT_TEMPLATE.format(
            paper=context
        )

        seed_prompts = sample_seed_prompts(self.config.seed_prompts, batch_size)
        return ctx, seed_prompts

    def to_string(self) -> str:
        out = f"Below is a panel of scientific papers."
        for paper in self.papers:
            out += "\n\n"
            out += f"<paper>\n{paper.to_string}\n</paper>\n"
        return out
        

SECTION_TEMPLATE = """\
<section>
<section-title>{title}</section-title>
<section-number>{section_number}</section-number>
<paragraphs>
{paragraphs}
</paragraphs>
</section>
"""

@dataclass
class Section:
    title: str
    section_number: int
    paragraphs: List[str]

    @property
    def text(self) -> str:
        paragraph_divider = "\n\n"
        return SECTION_TEMPLATE.format(
            title=self.title,
            section_number=self.section_number,
            paragraphs=paragraph_divider.join(self.paragraphs)
        )


PAPER_TEMPLATE = """\
<title>{title}</title>
<abstract>{abstract}</abstract>
<sections>
{sections}
</sections>
"""
@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    sections: List[Section]

    @property
    def to_string(self) -> str:
        section_divider = f"\n---Paper Title: {self.title}---\n"
        return PAPER_TEMPLATE.format(
            title=self.title,
            abstract=self.abstract,
            sections=section_divider.join([section.text for section in self.sections])
        )
