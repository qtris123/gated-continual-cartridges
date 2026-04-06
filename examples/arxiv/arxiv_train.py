import os
from pathlib import Path
import pydrantic

from cartridges.initialization import KVFromText
from cartridges.train import CosWithWarmup
from cartridges.train import TrainConfig, LossEvalConfig, GenerationEvalConfig
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM, FlexLlamaForCausalLM
from cartridges.datasets import DataSource, GenerateEvalDataset, TrainDataset, LossEvalDataset
from cartridges.clients.base import CartridgeConfig

from cartridges.data.longhealth.evals import LongHealthMultipleChoiceGenerateDataset
from cartridges.data.mtob.evals import MTOBKalamangToEnglishGenerateDataset
from cartridges.data.qasper.generate_qasper_abstracts import generate_qasper_init_text

P_KV = 1024
# INITIAL_KV_PATH = "/home/vo43/cartridges/examples/qasper/qasper_init_mt_1024.txt"

# if os.path.exists(INITIAL_KV_PATH):
#     saved_path = INITIAL_KV_PATH
# else:
#     saved_path = generate_qasper_init_text( p=P_KV, tokenizer_name="meta-llama/llama-3.2-3B-Instruct", path=INITIAL_KV_PATH)
#     print(f"Finished generating init_text with {P_KV} tokens at {saved_path}")

config = TrainConfig(
    model=HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-4B",#"meta-llama/llama-3.2-3B-Instruct",
        model_cls=FlexQwen3ForCausalLM,
    ),
    kv_cache_initializer=KVFromText.Config( # QASPER
        text_source="/home/vo43/cartridges/examples/arxiv/longhealth_context.txt", #saved_path,
        max_tokens= P_KV # p : the number of tokens to use for constructing the initial KV cache. 
    ),
    # kv_cache_initializer=KVFromText.Config( # QASPER
    #     text_source="/home/vo43/cartridges/examples/qasper/qasper_context.txt",
    #     max_tokens= 2048 # p : the number of tokens to use for constructing the initial KV cache. 
    # ),
    # kv_cache_initializer=KVFromText.Config( # LongHealth
    #     text_source=os.path.join(os.environ["CARTRIDGES_DIR"], "examples/arxiv/longhealth_context.txt"),
    #     max_tokens=512 # p : the number of tokens to use for constructing the initial KV cache. 
    # ),
    # kv_cache_initializer=KVFromText.Config( # MTOB
    #     text_source=os.path.join(os.environ["CARTRIDGES_DIR"], "cartridges/data/mtob/_data/grammar_book_for_claude_medium.txt"),
    #     max_tokens=512  # p : the number of tokens to use for constructing the initial KV cache.
    # ),
    
    lr=2e-2, #OG: 2e-2
    epochs=1,
    global_batch_size=32, 

    dataset=TrainDataset.Config(
        data_sources=[
            #DataSource(path="/home/phudishp/cartridges/outputs/2026-02-08-06-27-11-arxiv_synthesize/arxiv_synthesize_meta-llama/Llama-3.2-3B-Instruct_n256-0/artifact/dataset.parquet", type="local"),
            DataSource(path="/scratch/scholar/vo43/longhealth_p1-10_qwen_n8192.parquet", type="local"),
        ],
        top_k_logits=20,
        packed_seq_length=2048, # figure 5 says they use 1024 instead, but their bsize is 64 and mine is 32 => so basically the same. I packed 2x more batches than theirs, and they use 2x more bsize than me.
        packing_mode="truncate",
    ),
    # lr_scheduler=CosWithWarmup.Config( # 3B-3
    #     warmup_steps=40,
    #     max_steps=400,
    # ),
    #max_train_batches=100,  # Limit training dataset to x batches per epoch. Because data.instantiate() returns dataset in batches => len(dataset) is the number of batches

    # # QASPER uses log perplexity as metric.
    # loss_eval_every_n_steps=16,
    # loss_evals=[
    #     LossEvalConfig(
    #         dataset=LossEvalDataset.Config(
    #             data_source=DataSource( #n128
    #                 path="/home/vo43/cartridges/examples/arxiv/qasper_rewrite_eval_MT_standard.parquet", #qasper_gpt41_rewrite.parquet"
    #                 #path="/home/vo43/cartridges/outputs/0_n128/7a282c8d-0eb6-4fa2-893a-6427f8e3d987/artifact/dataset.parquet",
    #                 type="local",
    #             ),
    #             packed_seq_length=2048,
    #         ),
    #         name_for_wandb="qasper_perplexity",
    #         max_eval_samples=200,
    #     )
    # ],

    # LongHealth uses Accuracy as metric.
    generate_eval_every_n_steps= 15, #128,
    generate_evals=[
        GenerationEvalConfig(
            dataset=LongHealthMultipleChoiceGenerateDataset.Config(
            patient_ids=["patient_11", "patient_12", "patient_13", "patient_14", "patient_15", "patient_16", "patient_17", "patient_18", "patient_19", "patient_20"], 
            max_questions=100, 
            include_diagnosis=True, 
            cot=True,
        ),
        name_for_wandb="longhealth_accuracy",
        generate_max_new_tokens=512,
        batch_size=16,
        temperature=0.3,
        #max_eval_samples=200,  # Optional: limit samples 
        )
    ],

    # # MTOB uses chrF as metric. Paper uses Kalamang to English
    # generate_eval_every_n_steps=128,
    # generate_evals=[
    #     GenerationEvalConfig(
    #         name_for_wandb="mtob-ke-test",
    #         dataset=MTOBKalamangToEnglishGenerateDataset.Config(use_cot=True),
    #         batch_size=16,
    #         generate_max_new_tokens=128,
    #         num_samples=1,
    #         temperature=0,
    #     ),
    # ],
    distributed_backend="gloo",

    save_every_n_steps=500,
    name="cartridges_longhealth-p1-10_off-policy_1024_n8192_Qwen3B",
)


if __name__ == "__main__":
    pydrantic.main(config)