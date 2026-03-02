"""Self-study synthesis for 10-K documents using a trained cartridge.

The model runs **locally** (no Tokasaurus server required) and is optionally
conditioned on a compressed cartridge that encodes prior document knowledge.

Usage:
    # Local cartridge
    python experiments/continual_learning/synthesize_data_with_cartridge.py \
        --company AMD --year 2021 \
        --cartridge-path ./outputs/train_amd_2020/cache_last.pt

    # HuggingFace cartridge
    python experiments/continual_learning/synthesize_data_with_cartridge.py \
        --company AMD --year 2021 \
        --cartridge-hf-id username/amd-2020-cartridge
"""

import argparse
import os
import sys
from pathlib import Path

import fitz
import requests

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.local_cache import LocalCacheClient
from cartridges.data.chunkers import TokenChunker
from cartridges.data.resources import TextFileResource
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer

SCRIPT_DIR = Path(__file__).resolve().parent
PDF_DIR = SCRIPT_DIR / "data" / "pdfs"
TEXT_DIR = SCRIPT_DIR / "data" / "texts"

FINANCEBENCH_PDF_BASE = (
    "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/"
)


def download_pdf(filename: str, output_dir: Path) -> Path:
    """Download a PDF from the FinanceBench GitHub repo."""
    output_path = output_dir / filename
    if output_path.exists():
        print(f"Already downloaded: {output_path}")
        return output_path

    url = FINANCEBENCH_PDF_BASE + filename
    print(f"Downloading {url} ...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(resp.content)
    print(f"Saved to {output_path} ({len(resp.content)} bytes)")
    return output_path


def pdf_to_text(pdf_path: Path) -> str:
    """Extract text from a PDF using PyMuPDF (fitz)."""
    with fitz.open(str(pdf_path)) as doc:
        return "\n".join(page.get_text() for page in doc)


def ensure_text_file(company: str, year: int) -> Path:
    """Ensure the text file exists, downloading and converting the PDF if needed."""
    doc_name = f"{company.upper()}_{year}_10K"
    text_path = TEXT_DIR / f"{doc_name}.txt"

    if text_path.exists():
        print(f"Text file already exists: {text_path}")
        return text_path

    pdf_filename = f"{doc_name}.pdf"
    pdf_path = download_pdf(pdf_filename, PDF_DIR)

    text = pdf_to_text(pdf_path)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    text_path.write_text(text, encoding="utf-8")
    print(f"Converted {pdf_filename} -> {text_path.name} ({len(text)} chars)")
    return text_path


parser = argparse.ArgumentParser(description="Synthesize 10-K data with a trained cartridge")
parser.add_argument(
    "--company",
    type=str,
    required=True,
    help="Company name (used to find data/texts/{COMPANY}_{YEAR}_10K.txt)",
)
parser.add_argument(
    "--year",
    type=int,
    required=True,
    help="10-K filing year",
)
parser.add_argument(
    "--include-year",
    action="store_true",
    default=False,
    help="Include explicit year in number-related prompts",
)
parser.add_argument(
    "--cartridge-path",
    type=str,
    default=None,
    help="Local path to a trained cartridge .pt file",
)
parser.add_argument(
    "--cartridge-hf-id",
    type=str,
    default=None,
    help="HuggingFace repo ID containing a .pt cartridge (e.g. 'username/my-cartridge')",
)
args, remaining = parser.parse_known_args()

if args.cartridge_path is None and args.cartridge_hf_id is None:
    parser.error("At least one of --cartridge-path or --cartridge-hf-id is required.")

sys.argv = [sys.argv[0]] + remaining

ensure_text_file(args.company, args.year)
TEXT_PATH = str(TEXT_DIR / f"{args.company.upper()}_{args.year}_10K.txt")

client = LocalCacheClient.Config(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    cartridge_path=args.cartridge_path,
    cartridge_hf_id=args.cartridge_hf_id,
)

config = SynthesizeConfig(
    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        max_rounds=1,
        prob_thinking=0.2,
        tools=[],
        resources=[
            TextFileResource.Config(
                path=TEXT_PATH,
                year=args.year,
                include_year=args.include_year,
                seed_prompts=[
                    "genconvo_factual",
                    "genconvo_knowledge",
                    "genconvo_disjoint",
                    "genconvo_synthesize",
                    "genconvo_structure",
                    "genconvo_creative",
                    "genconvo_counting",
                    "genconvo_reasoning",
                ],
                chunker=TokenChunker.Config(
                    tokenizer=client.model_name,
                    min_tokens_per_chunk=None,
                    max_tokens_per_chunk=8192,
                ),
            )
        ],
    ),
    num_samples=64,
    batch_size=4,
    max_num_batches_in_parallel=4,
    name=FormatStringVariable(
        f"synthesize_{args.company.lower()}_{args.year}_{{synthesizer.client.model_name}}_n{{num_samples}}_with_cartridge"
    ),
    run_id=FormatStringVariable("{name}"),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    upload_to_wandb=False,
    save_wandb_preview=False,
    upload_to_hf=False,
    hf_repo_id="hazyresearch/{wandb_run_id}",
)

if __name__ == "__main__":
    pydrantic.main([config])
