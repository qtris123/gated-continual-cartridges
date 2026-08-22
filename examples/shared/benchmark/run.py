"""
Benchmark runner examples — two modes:

1. BenchmarkConfig             — connect to an already-running server
2. BenchmarkWithServerConfig   — auto-launch Tokasaurus with full GPU config,
                                  run benchmark, then shut down (like qasper_synthesize_with_server.py)

Usage:
    python run_benchmark.py                              # runs all configs
    python run_benchmark.py benchmark.dataset_name=mmlu benchmark.subset=college_physics benchmark.max_samples=50
"""

import os

import pydrantic

from cartridges.benchmark import BenchmarkConfig, BenchmarkWithServerConfig, TokaServerConfig
from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.clients.base import CartridgeConfig
from cartridges.clients.openai import OpenAIClient

# =====================================================================
# Client configs
# =====================================================================

# # Model only (no cartridge) — requires a server already running
# toka_client = TokasaurusClient.Config(
#     url="http://localhost:10210",
#     model_name="Qwen/Qwen3-4B",
# )

# Model + cartridge — cartridges are sent with every chat() request
# and automatically route to the /custom/cartridge/ endpoint on the Toka server
toka_client_with_cartridge = TokasaurusClient.Config(
    url="http://gilbreth-n014:10210",
    model_name="meta-llama/Llama-3.2-3B-Instruct", #"Qwen/Qwen3-4B",
    cartridges=[
        CartridgeConfig(
            id="qtris123/longhealth-p1-10_8192_1024_no-cartridge",
            source="huggingface",
            force_redownload=False,
        )
    ],
)

# openai_client = OpenAIClient.Config(
#     base_url="http://localhost:8000/v1",
#     model_name="Qwen/Qwen3-4B",
# )

# =====================================================================
# Tokasaurus server config (GPU params: tp, dp, kv cache, etc.)
# Only needed for BenchmarkWithServerConfig
# =====================================================================

# toka_server = TokaServerConfig(
#     model="Qwen/Qwen3-4B",
#     port=10210,

#     # Parallelism
#     dp_size=1,
#     tp_size=1,
#     pp_size=1,

#     # KV cache / memory
#     kv_cache_num_tokens=1024 * 128,
#     max_tokens_per_forward=8192,
#     max_seqs_per_forward=1024,
#     max_num_tokens_per_request=15000,

#     # Performance
#     use_hydragen=True,
#     hydragen_min_group_size=32,
#     max_topk_logprobs=20,
#     cudagraph_max_size=16,

#     log_level="INFO",
#     uvicorn_log_level="info",
# )

# =====================================================================
# Benchmark configs
# =====================================================================

configs = [

    # # # -----------------------------------------------------------------
    # # # Mode 1: BenchmarkConfig (server already running)
    # # # -----------------------------------------------------------------

    # # MMLU — 5-shot, multiple-choice scoring
    # BenchmarkConfig(
    #     name="mmlu_abstract_algebra",
    #     dataset_name="mmlu",
    #     subset="abstract_algebra",
    #     num_few_shot=5,
    #     client=toka_client_with_cartridge,
    #     scorer="multiple_choice",
    #     temperature=0.0,
    #     max_completion_tokens=16,
    #     batch_size=32,
    #     max_concurrent_batches=4,
    #     max_samples=None,
    # ),

    # # HellaSwag — 0-shot, multiple-choice
    # BenchmarkConfig(
    #     name="hellaswag_quick",
    #     dataset_name="hellaswag",
    #     num_few_shot=0,
    #     client=toka_client_with_cartridge,
    #     scorer="multiple_choice",
    #     temperature=0.0,
    #     max_completion_tokens=16,
    #     batch_size=32,
    #     max_concurrent_batches=4,
    #     max_samples=200,
    # ),

    # QASPER — uses rewritten dataset (same as cartridges.data.qasper.evals)
    BenchmarkConfig(
        name="qasper_ppl",
        dataset_name="qasper",
        num_few_shot=0,
        client=toka_client_with_cartridge,
        scorer="f1",
        temperature=0.0,
        max_completion_tokens=256,
        batch_size=16,
        max_concurrent_batches=4,
        max_samples=100,
    ),

    # # LongHealth — model + cartridge, fuzzy option matching (same as longhealth/evals.py)
    # BenchmarkConfig(
    #     name="longhealth_mc",
    #     dataset_name="longhealth",
    #     subset="patient_01,patient_02,patient_03,patient_04,patient_05,patient_06,patient_07,patient_08,patient_09,patient_10",
    #     client=toka_client_with_cartridge,
    #     scorer="longhealth_mc",
    #     temperature=0.0,
    #     max_completion_tokens=512,
    #     batch_size=16,
    #     max_concurrent_batches=4,
    # ),

    # -----------------------------------------------------------------
    # Mode 2: BenchmarkWithServerConfig (auto-launch Tokasaurus)
    # Launches server → patches client URL → runs benchmark → shuts down.
    # All GPU params (tp_size, kv_cache_num_tokens, …) come from toka_server.
    # -----------------------------------------------------------------

    # BenchmarkWithServerConfig(
    #     benchmark=BenchmarkConfig(
    #         name="mmlu_with_server",
    #         dataset_name="mmlu",
    #         subset="abstract_algebra",
    #         num_few_shot=5,
    #         client=TokasaurusClient.Config(
    #             url="http://localhost:10210",      # will be overwritten by server
    #             model_name="Qwen/Qwen3-4B", 
    #             cartridges=[
    #                 CartridgeConfig(
    #                     id="qtris123/longhealth-p1-10_8192_1024_no-cartridge",
    #                     source="huggingface",
    #                 )
    #             ],
    #         ),
    #         scorer="multiple_choice",
    #         temperature=0.0,
    #         max_completion_tokens=16,
    #         batch_size=32,
    #         max_concurrent_batches=4,
    #     ),
    #     tokasaurus=toka_server,
    #    # conda_env="aa",   # set to e.g. "toka12" if toka is in a different env
    # ),

    # # With cartridge + auto-launched server
    # BenchmarkWithServerConfig(
    #     benchmark=BenchmarkConfig(
    #         name="longhealth_mc_with_server",
    #         dataset_name="longhealth",
    #         subset="patient_01,patient_02,patient_03",
    #         client=TokasaurusClient.Config(
    #             url="http://localhost:10210",
    #             model_name="Qwen/Qwen3-4B",
    #             cartridges=[
    #                 CartridgeConfig(
    #                     id="qtris123/longhealth-p1-10_8192_1024_no-cartridge",
    #                     source="huggingface",
    #                 )
    #             ],
    #         ),
    #         scorer="longhealth_mc",
    #         temperature=0.0,
    #         max_completion_tokens=512,
    #         batch_size=4,
    #     ),
    #     tokasaurus=toka_server,
    # ),
]

if __name__ == "__main__":
    pydrantic.main(configs)
