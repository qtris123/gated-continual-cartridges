"""Phase 1: Train initial cartridge.

Usage:
    TEXT_PATH=/path/to/AMD_2021_10K.txt \
    SYNTH_DATA_PATH_PHASE1=/path/to/dataset.parquet \
    python experiments/train/initial.py

    # Or with torchrun for multi-GPU:
    TEXT_PATH=/path/to/AMD_2021_10K.txt \
    SYNTH_DATA_PATH_PHASE1=/path/to/dataset.parquet \
    torchrun --nproc_per_node=2 experiments/train/initial.py

Env vars:
    TEXT_PATH               (required) — path to source text file
    SYNTH_DATA_PATH_PHASE1  (required) — path to synthesized training data parquet
    NUM_TOKENS              — cartridge size (default: 4096)
    MODEL_NAME              — HF model name (default: meta-llama/Llama-3.2-3B-Instruct)
    LR                      — learning rate (default: 2e-2)
    EPOCHS                  — number of epochs (default: 1)
    GLOBAL_BATCH_SIZE       — global batch size (default: 32)
    TOP_K_LOGITS            — top-k logits to store (default: 20)
    PACKED_SEQ_LENGTH       — packed sequence length (default: 2048)
    SAVE_EVERY_N_STEPS      — save checkpoint every N steps (default: 256)
    DISTRIBUTED_BACKEND     — distributed training backend (default: gloo)
    COMPANY                 — company name for run naming (default: unknown)
    YEAR_Y                  — year for run naming (default: unknown)
    PROVENANCE_TAG          — tag appended to last user message per convo (default: no tag)
"""

import os

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.datasets import DataSource, TrainDataset
from cartridges.initialization import KVFromText
from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM, HFModelConfig
from cartridges.train import TrainConfig, GenerationEvalConfig
from cartridges.data.longhealth.evals import LongHealthMultipleChoiceGenerateDataset
from cartridges.utils.wandb import WandBConfig

TEXT_PATH = os.environ["TEXT_PATH"]
SYNTH_DATA_PATH = os.environ["SYNTH_DATA_PATH"]
NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "512"))
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
LR = float(os.environ.get("LR", "2e-2"))
EPOCHS = int(os.environ.get("EPOCHS", "1"))
GLOBAL_BATCH_SIZE = int(os.environ.get("GLOBAL_BATCH_SIZE", "32"))
PATIENT_IDS = os.environ.get("PATIENT_IDS", "1-10")
EVAL_EVERY_N_STEPS = int(os.environ.get("EVAL_EVERY_N_STEPS", "128"))
SAVE_EVERY_N_STEPS = int(os.environ.get("SAVE_EVERY_N_STEPS", "256"))
DISTRIBUTED_BACKEND = os.environ.get("DISTRIBUTED_BACKEND", "gloo")

_model_cls = FlexQwen3ForCausalLM if "qwen" in MODEL_NAME.lower() else FlexLlamaForCausalLM

_PATIENTS = { "1-10": [
    "patient_01",
    "patient_02",
    "patient_03",
    "patient_04",
    "patient_05",
    "patient_06",
    "patient_07",
    "patient_08",
    "patient_09",
    "patient_10",
], "11-20": [
    "patient_11",
    "patient_12",
    "patient_13",
    "patient_14",
    "patient_15",
    "patient_16",
    "patient_17",
    "patient_18",
    "patient_19",
    "patient_20",
]}

config = TrainConfig(
    model=HFModelConfig(
        pretrained_model_name_or_path=MODEL_NAME,
        model_cls=_model_cls,
    ),
    kv_cache_initializer=KVFromText.Config(
        text_source=TEXT_PATH,
        max_tokens=NUM_TOKENS,
    ),
    lr=LR,
    epochs=EPOCHS,
    global_batch_size=GLOBAL_BATCH_SIZE,
    dataset=TrainDataset.Config(
        data_sources=[
            DataSource(
                path=SYNTH_DATA_PATH,
                type="local",
            ),
        ],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),
    generate_eval_every_n_steps= EVAL_EVERY_N_STEPS,
    generate_evals=[
        GenerationEvalConfig(
            dataset=LongHealthMultipleChoiceGenerateDataset.Config(
            patient_ids=_PATIENTS[PATIENT_IDS],
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
    save_every_n_steps=SAVE_EVERY_N_STEPS,
    distributed_backend=DISTRIBUTED_BACKEND,
    wandb=WandBConfig(tags=["train", "longhealth", "phase1"]),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    name=FormatStringVariable(
        f"longhealth_phase1_{MODEL_NAME.split('/')[-1]}_{NUM_TOKENS}"
    ),
)

if __name__ == "__main__":
    pydrantic.main(config)