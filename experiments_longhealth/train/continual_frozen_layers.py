"""Phase 2: Continual update with selected layers of the KV cache frozen.

Identical to continual.py except that specific layers of the trainable KV cache
can be frozen before training begins. Frozen layers keep their phase-1 values
throughout training; only the remaining layers are updated by the optimizer.

Motivation: the layer-swap forgetting analysis shows that middle layers
(~10–17 for Llama 3.2-3B, ~15–25 for Qwen 3-4B) are causally responsible
for AMD forgetting. Freezing them during Pepsi continual training should
let the cache learn Pepsi knowledge in the other layers while preserving AMD.

Usage:
    PHASE1_CACHE_PATH=/path/to/cache_last.pt \
    SYNTH_DATA_PATH=/path/to/dataset_phase2.parquet \
    FROZEN_LAYERS="10,11,12,13,14,15,16,17" \
    FREEZE_COMPONENT="both" \
    python experiments/train/continual_frozen_layers.py

Env vars:
    PHASE1_CACHE_PATH   (required) — path to Phase 1 cartridge checkpoint
    SYNTH_DATA_PATH     (required) — path to synthesized training data parquet
    FROZEN_LAYERS       — comma-separated layer indices to freeze (default: "" = none)
    FREEZE_COMPONENT    — "values", "keys", or "both" (default: "both")
    NUM_TOKENS          — cartridge size for naming only (default: 1024)
    MODEL_NAME          — HuggingFace model ID (default: meta-llama/Llama-3.2-3B-Instruct)
    COMPANY             — company name for run naming (default: unknown)
    YEAR_Y              — year for run naming (default: unknown)
    PROVENANCE_TAG      — tag appended to last user message per convo (default: no tag)
    LR                  — learning rate (default: 2e-2)
    EPOCHS              — number of training epochs (default: 1)
    GLOBAL_BATCH_SIZE   — global batch size (default: 32)
"""

import os
from dataclasses import field

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.cache import KVCacheFactory, TrainableCache
from cartridges.datasets import DataSource, TrainDataset
from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM, HFModelConfig
from cartridges.train import TrainConfig
from cartridges.utils.wandb import WandBConfig


# ---------------------------------------------------------------------------
# Env vars
# ---------------------------------------------------------------------------

PHASE1_CACHE_PATH = os.environ["PHASE1_CACHE_PATH"]
SYNTH_DATA_PATH   = os.environ["SYNTH_DATA_PATH"]
MODEL_NAME        = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")

NUM_TOKENS        = int(os.environ.get("NUM_TOKENS", "1024"))
LR                = float(os.environ.get("LR", "2e-2"))
EPOCHS            = int(os.environ.get("EPOCHS", "1"))
GLOBAL_BATCH_SIZE = int(os.environ.get("GLOBAL_BATCH_SIZE", "32"))
PROVENANCE_TAG    = os.environ.get("PROVENANCE_TAG", "")
COMPANY           = os.environ.get("COMPANY", "unknown")
YEAR_Y            = os.environ.get("YEAR_Y",  "unknown")

# Layer freezing config
_frozen_layers_str = os.environ.get("FROZEN_LAYERS", "")
FROZEN_LAYERS     = [int(x) for x in _frozen_layers_str.split(",") if x.strip()]
FREEZE_COMPONENT  = os.environ.get("FREEZE_COMPONENT", "both")  # "values" | "keys" | "both"

assert FREEZE_COMPONENT in ("values", "keys", "both"), (
    f"FREEZE_COMPONENT must be 'values', 'keys', or 'both', got: {FREEZE_COMPONENT!r}"
)

# ---------------------------------------------------------------------------
# KV cache initializer with per-layer freezing
# ---------------------------------------------------------------------------

class KVFromLocalFrozenLayers(KVCacheFactory):
    """Load a KV cache checkpoint and freeze specified layers before training."""

    class Config(KVCacheFactory.Config):
        path: str
        frozen_layers: list = field(default_factory=list)
        freeze_component: str = "both"

    def initialize_kv_cache(self, tokenizer=None, model=None, attn_config=None):
        cache = TrainableCache.from_pretrained(self.config.path, device="cuda")
        n_total = cache.config.n_layers

        for layer_idx in self.config.frozen_layers:
            if layer_idx < 0 or layer_idx >= n_total:
                raise ValueError(
                    f"FROZEN_LAYERS contains out-of-range index {layer_idx} "
                    f"(model has {n_total} layers, valid range: 0–{n_total - 1})"
                )
            if self.config.freeze_component in ("both", "values"):
                cache.trainable_values[layer_idx].requires_grad = False
            if self.config.freeze_component in ("both", "keys"):
                cache.trainable_keys[layer_idx].requires_grad = False

        n_frozen = len(self.config.frozen_layers)
        if n_frozen:
            print(
                f"[frozen_layers] Froze {n_frozen}/{n_total} layers "
                f"(component={self.config.freeze_component!r}): {self.config.frozen_layers}"
            )
        else:
            print("[frozen_layers] No layers frozen — behaves identically to continual.py")

        return cache


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

_model_cls = FlexQwen3ForCausalLM if "qwen" in MODEL_NAME.lower() else FlexLlamaForCausalLM
_n_frozen  = len(FROZEN_LAYERS)

config = TrainConfig(
    model=HFModelConfig(
        pretrained_model_name_or_path=MODEL_NAME,
        model_cls=_model_cls,
    ),
    kv_cache_initializer=KVFromLocalFrozenLayers.Config(
        path=PHASE1_CACHE_PATH,
        frozen_layers=FROZEN_LAYERS,
        freeze_component=FREEZE_COMPONENT,
    ),
    lr=LR,
    epochs=EPOCHS,
    global_batch_size=GLOBAL_BATCH_SIZE,
    dataset=TrainDataset.Config(
        data_sources=[
            DataSource(
                path=SYNTH_DATA_PATH,
                type="local",
                source_tag=PROVENANCE_TAG if PROVENANCE_TAG else None,
            ),
        ],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),
    save_every_n_steps=256,
    distributed_backend="gloo",
    wandb=WandBConfig(tags=["train", "continual_learning", "phase2", "frozen_layers"]),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    name=FormatStringVariable(
        f"{COMPANY}_{YEAR_Y}_continual_frozen{_n_frozen}layers_{FREEZE_COMPONENT}_lr{{lr}}_toks{NUM_TOKENS}"
    ),
)

if __name__ == "__main__":
    pydrantic.main(config)
