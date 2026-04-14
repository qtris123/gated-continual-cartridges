import os
from pathlib import Path
import sys

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.data.qasper.resources import QASPERResource
from cartridges.utils.wandb import WandBConfig

# Add the infra/tuning directory to the path to import tune_toka
sys.path.append(os.path.join(os.environ.get("CARTRIDGES_DIR", "."), "infra/tuning"))
from tune_toka import EvaluateTokaConfig, TokaConfig

# Tokasaurus server configuration
toka_config = TokaConfig(
    model="meta-llama/Llama-3.2-3B-Instruct", #Qwen/Qwen3-4B or Qwen3-4b work the same I think
    port=10210,  # Will auto-find available port if this is in use
    
    # Performance settings
    max_topk_logprobs=20,
    use_hydragen=True,
    hydragen_min_group_size=32,
    kv_cache_num_tokens=1024 * 128,  # 8K tokens (for MIG slices - use 1024 * 128 if you get full GPUs)
     
    # Memory/compute settings
    #use_cudagraphs=False,  # Disabled to fix assertion error
    cudagraph_max_size=16,  # Reduced for faster startup
    max_tokens_per_forward=8192, 
    max_seqs_per_forward=1024,
    max_num_tokens_per_request=15000,  # Increased from default 8192 to handle long QASPER prompts + max_tokens
    
    # Parallelism - use both GPUs with tensor parallelism
    dp_size=1,  # Data parallelism
    tp_size=2,  # Tensor parallelism (split across 2 GPUs)
    pp_size=1,  # Pipeline parallelism
    
    # Logging
    log_level="INFO",
    uvicorn_log_level="info",
)

# Synthesis configuration (same as qasper_synthesize.py but wrapped)
synthesize_config = SynthesizeConfig(
    synthesizer=SelfStudySynthesizer.Config(
        client=TokasaurusClient.Config(
            # URL will be auto-updated by EvaluateTokaConfig to match the launched server
            url="http://localhost:10210",  # Placeholder
            model_name="meta-llama/Llama-3.2-3B-Instruct", #Qwen/Qwen3-4B
        ),
        max_rounds=1,
        prob_thinking=0.2,
        
        # FIX: Qwen3-4B has a known issue with temperature=0 causing endless repetitive garbage
        # HuggingFace recommends temperature=0.7-0.8 for Qwen3-4B-Instruct
        # temperature_a=0.6,  # Keep existing value (user questions)
        temperature_b=0.8,  # CHANGED from default 0.0 to 0.8 to fix repetition bug
        
        tools=[],
        use_tools_a=False,
        use_tools_b=False,
        
        resources=[
            QASPERResource.Config(
                topic="question",
                seed_prompts=["structuring", "summarization", "question", "use_case", "creative"],
            )
        ],
    ),

    # Generation settings
    num_samples= 24560, #(128,) 1024, 10240, 30720
    batch_size=16,
    max_num_batches_in_parallel=64,
    
    # Output configuration
    name=FormatStringVariable(f"{Path(__file__).stem}_{{synthesizer.client.model_name}}_n{{num_samples}}"),
    run_id=FormatStringVariable("{name}"), 
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "./outputs"),
    
    # Upload settings
    upload_to_wandb=True, 
    save_wandb_preview=True, 
    upload_to_hf=False,
    hf_repo_id=None,
)

# Wrap both configs with EvaluateTokaConfig
# This will:
# 1. Start tokasaurus server with toka_config
# 2. Wait for it to be ready (ping with timeout)
# 3. Update synthesize_config.client.url to match the server
# 4. Run synthesis
# 5. Shutdown server gracefully
config = EvaluateTokaConfig(
    synthesize=synthesize_config,
    tokasaurus=toka_config,
    
    # Optional: specify conda environment if tokasaurus is in a different env
    # conda_env="toka12",  # Uncomment if needed
    conda_env=None,
)

if __name__ == "__main__":
    pydrantic.main([config])
