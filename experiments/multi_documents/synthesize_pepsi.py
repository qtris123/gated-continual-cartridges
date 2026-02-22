"""Self-study synthesis for PepsiCo 2022 10-K."""

import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.data.chunkers import TokenChunker
from cartridges.data.resources import TextFileResource
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer

SCRIPT_DIR = Path(__file__).resolve().parent
TEXT_PATH = str(SCRIPT_DIR / "data" / "texts" / "PEPSICO_2022_10K.txt")

client = TokasaurusClient.Config(
    url=os.environ.get("CARTRIDGES_TOKASAURUS_URL", "http://localhost:8000"),
    model_name="meta-llama/Llama-3.2-3B-Instruct",
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
    num_samples=65536,
    batch_size=4,
    max_num_batches_in_parallel=64,
    name=FormatStringVariable(
        f"{Path(__file__).stem}_{{synthesizer.client.model_name}}_n{{num_samples}}"
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
