import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.data.qasper.resources import QASPERResource
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.clients.tokasaurus import TokasaurusClient

client = TokasaurusClient.Config(
    url=os.environ.get("CARTRIDGES_TOKASAURUS_QWEN3_4B_URL", "http://localhost:8000"),
    model_name="Qwen/Qwen3-4b",
)

config = SynthesizeConfig(
    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        max_rounds=1,
        prob_thinking=0.2,
        tools=[],
        use_tools_a=False,
        use_tools_b=False,
        
        resources=[
            QASPERResource.Config(
                topic="question",
                seed_prompts=["structuring", "summarization","question", "use_case", "creative"],
            )
        ],
    ),

    num_samples=256,
    batch_size=4,
    max_num_batches_in_parallel=64,
    
    name=FormatStringVariable(f"{Path(__file__).stem}_{{synthesizer.client.model_name}}_n{{num_samples}}"),
    run_id=FormatStringVariable("{name}"),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "./outputs"),
    
    upload_to_wandb=False,
    save_wandb_preview=False,

    upload_to_hf=False,
    hf_repo_id=None, 
)

if __name__ == "__main__":
    pydrantic.main([config])
