"""
generate_qasper_abstracts.py

Core function:
    generate_init_text(p, paper_ids, tokenizer_name, split="train") -> str

    Loads QASPER papers by `paper_ids`, takes the first (p // n) tokens
    from each paper's full text (title + abstract + sections), and returns
    a single concatenated string suitable for KV-cache initialization.

    Output format (one <paper> block per paper):
        Below is a panel of scientific papers.

        <paper>
        <title>...</title>
        <abstract>...</abstract>
        <sections>
          <section>
            <section-title>...</section-title>
            <section-number>0</section-number>
            <paragraphs>
        ...
            </paragraphs>
          </section>
          ...
        </sections>
        </paper>

CLI usage:
    python generate_qasper_abstracts.py \
        --paper_ids 1908.06606 1704.05572 \
        --p 4096 \
        --tokenizer Qwen/Qwen3-4B \
        --output qasper_init.txt
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from datasets import load_dataset
from transformers import AutoTokenizer

from cartridges.data.qasper.resources import TOPIC_TO_IDS


DEFAULT_PATH = "/home/vo43/cartridges/examples/qasper/qasper_init_20480.txt"

def _paper_to_text(row: dict) -> str:
    """Render a single QASPER row as a self-contained <paper> XML block."""
    title    = row["title"].strip()
    abstract = row["abstract"].strip()

    section_tags = []
    for idx, (section_title, paragraphs) in enumerate(zip(
        row["full_text"]["section_name"],
        row["full_text"]["paragraphs"],
    )):
        paras = "\n\n".join(p.strip() for p in paragraphs if p.strip())
        section_tags.append(
            f"  <section>\n"
            f"    <section-title>{section_title}</section-title>\n"
            f"    <section-number>{idx}</section-number>\n"
            f"    <paragraphs>\n{paras}\n    </paragraphs>\n"
            f"  </section>"
        )

    sections_block = "\n\n".join(section_tags)
    return (
        f"<paper>\n"
        f"<title>{title}</title>\n"
        f"<abstract>{abstract}</abstract>\n"
        f"<sections>\n{sections_block}\n</sections>\n"
        f"</paper>"
    )


def generate_qasper_init_text(
    p: int,
    paper_ids: Optional[List[str]] = None,
    tokenizer_name: str = "Qwen/Qwen3-4B",
    path: str = DEFAULT_PATH,
    split: str = "train",
) -> str:
    """
    Build a KV-cache initialization string from QASPER papers.

    Parameters
    ----------
    p : int
        Total token budget across all papers.
    paper_ids : list[str] or None
        ArXiv IDs of the QASPER papers to include.
        Defaults to DEFAULT_PAPER_IDS (looked up at call time, not import time).
    tokenizer_name : str
        HuggingFace tokenizer used to count / slice tokens.
    split : str
        QASPER split to load ("train", "validation", or "test").

    Returns
    -------
    str
        Concatenated text, each paper trimmed to floor(p / n) tokens.
    """
    if paper_ids is None:
        paper_ids = TOPIC_TO_IDS["QA"]
    if not paper_ids:
        raise ValueError("paper_ids is empty — add at least one ArXiv ID to DEFAULT_PAPER_IDS.")
    n = len(paper_ids)
    CONST = 50
    tokens_per_paper = (p // n) + CONST

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print(f"Loading QASPER ({split} split)…")
    dataset = load_dataset("allenai/qasper", split=split,
                           revision="refs/convert/parquet")
    df = dataset.to_pandas()
    df = df[df["id"].isin(paper_ids)]

    # Preserve the requested order
    id_to_row = {row["id"]: row for row in df.to_dict(orient="records")}
    missing = [pid for pid in paper_ids if pid not in id_to_row]
    if missing:
        raise ValueError(f"Paper IDs not found in split '{split}': {missing}")

    chunks = []
    for pid in paper_ids:
        row = id_to_row[pid]
        full_text = _paper_to_text(row)
        token_ids = tokenizer.encode(full_text, add_special_tokens=False)
        sliced   = token_ids[:tokens_per_paper]
        chunk    = tokenizer.decode(sliced, skip_special_tokens=True)
        print(f"  [{pid}] {len(sliced)}/{len(token_ids)} tokens kept")
        chunks.append(chunk)

    preamble = "Below is a panel of scientific papers."
    completed =  preamble + "\n\n" + "\n\n".join(chunks)
    with open(path, "w", encoding="utf-8") as f:
        f.write(completed)

    return path


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate a KV-cache init text from QASPER papers."
    )
    parser.add_argument("--paper_ids", nargs="+", default=DEFAULT_PAPER_IDS,
                        help="ArXiv paper IDs to include.")
    parser.add_argument("--p", type=int, default=4096,
                        help="Total token budget (split evenly across papers).")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-4B",
                        help="HuggingFace tokenizer to use for token counting.")
    parser.add_argument("--split", default="train",
                        help="QASPER split (default: train).")
    parser.add_argument("--output", default="qasper_init.txt",
                        help="Output .txt file path.")
    args = parser.parse_args()

    text = generate_qasper_init_text(
        p=args.p,
        paper_ids=args.paper_ids,
        tokenizer_name=args.tokenizer,
        split=args.split,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text)

    n = len(args.paper_ids)
    print(f"Wrote {n} papers × {args.p // n} tokens → {args.output} ({len(text)} chars)")


if __name__ == "__main__":
    main()
