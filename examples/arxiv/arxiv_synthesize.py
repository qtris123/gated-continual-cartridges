import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.data.chunkers import TokenChunker
from cartridges.data.resources import TextFileResource
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.utils.wandb import WandBConfig
from cartridges.data.qasper.resources import QASPERResource
from cartridges.data.longhealth.resources import LongHealthResource
from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.clients.base import CartridgeConfig

client = TokasaurusClient.Config(
    url="http://scholar-j002:10210",
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    model_name="default",
    # cartridges=[
    #     CartridgeConfig(
    #         id="qtris123/longhealth-p1-10_8192_1024_no-cartridge",
    #         source="huggingface",
    #         force_redownload=False
    #     )
    # ],
)

config = SynthesizeConfig(

    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        max_rounds=1,
        prob_thinking=0.2,
        tools=[],
        resources=[
            
            LongHealthResource.Config(
                patient_ids=["patient_11", "patient_12", "patient_13", "patient_14", "patient_15", "patient_16", "patient_17", "patient_18", "patient_19", "patient_20"],
                max_notes_per_prompt=5,
                min_notes_per_prompt=1,
                max_chars_per_note= 2048,#None, # use the entire notes, not chunking
                seed_prompts=["structuring", "summarization", "question", "use_case", "creative"],
            )

            # QASPERResource.Config(
            #     topic="question",
            #     seed_prompts=["structuring", "summarization", "question", "use_case", "creative"],
            # )

            # TextFileResource.Config(
            #     path=os.path.join(os.environ["CARTRIDGES_DIR"], "examples/arxiv/cartridges.tex"),
            #     seed_prompts=[
            #         "structuring",
            #         "summarization",
            #         "question",
            #         "use_case",
            #         "creative",
            #     ],
            #     chunker=TokenChunker.Config(
            #         tokenizer=client.model_name,
            #         min_tokens_per_chunk=512,
            #         max_tokens_per_chunk=1024,
            #     ),
            # )
        ],
    ),

    num_samples=8192, 
    batch_size=16,  
    max_num_batches_in_parallel=16,

    #name=FormatStringVariable(f"{Path(__file__).stem}_{{synthesizer.client.model_name}}_n{{num_samples}}"),
    name=FormatStringVariable(f"{Path(__file__).stem}_qasper_n{{num_samples}}_1024_cartridge_on_QA_task"),
    run_id=FormatStringVariable("{name}"),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    
    upload_to_wandb=False,
    save_wandb_preview=False,
    
    upload_to_hf=False,
    hf_repo_id=None,
)


if __name__ == "__main__": 
    pydrantic.main([config])