"""Self-study synthesis for 10-K documents using a trained cartridge.

Usage:
    python experiments/synthesize/self_study_with_cartridge.py \
        --company AMD --year 2021 \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --num_samples 4 --batch_size 2 --max_num_batches 2 \
        --cartridge-hf-id username/amd-2021-cartridge
"""

import argparse
import os
import sys
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.base import CartridgeConfig
from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.data.chunkers import TokenChunker
from cartridges.data.resources import TextFileResource
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from experiments.utils.financebench import ensure_text_file

DATA_DIR = Path(os.environ["DATA_DIR"])
PDF_DIR = DATA_DIR / "pdfs"
TEXT_DIR = DATA_DIR / "texts"


parser = argparse.ArgumentParser(description="Synthesize 10-K data with cartridge")
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
    "--model",
    type=str,
    required=True,
    help="Model use to generate the self-study-data",
)
parser.add_argument(
    "--num_samples",
    type=int,
    required=True,
    help="number of samples to generate",
)
parser.add_argument(
    "--batch_size",
    type=int,
    required=True,
    help="Batch size"
)
parser.add_argument(
    "--max_num_batches",
    type=int,
    required=True,
    help="Max number of batches"
)
parser.add_argument(
    "--cartridge-hf-id",
    type=str,
    required=True,
    help="HuggingFace repo ID of the trained cartridge (e.g. username/amd-2021-cartridge)",
)
args, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining

ensure_text_file(args.company, args.year, PDF_DIR, TEXT_DIR)
TEXT_PATH = str(TEXT_DIR / f"{args.company.upper()}_{args.year}_10K.txt")

client = TokasaurusClient.Config(
    url=os.environ.get("CARTRIDGES_TOKASAURUS_URL", "http://localhost:8000"),
    model_name=args.model,
    cartridges=[
        CartridgeConfig(id=args.cartridge_hf_id, source="huggingface")
    ],
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
    num_samples=args.num_samples,
    batch_size=args.batch_size,
    max_num_batches_in_parallel=args.max_num_batches,
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
