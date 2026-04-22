"""Model registry + loader for cartridge intervention experiments."""
from __future__ import annotations

import torch
from transformers import AutoTokenizer

from cartridges.models import (
    FlexLlamaForCausalLM,
    FlexQwen3ForCausalLM,
    HFModelConfig,
)

MODEL_REGISTRY = {
    "qwen3-4b-instruct": ("Qwen/Qwen3-4B-Instruct-2507", FlexQwen3ForCausalLM),
    "llama-3.2-3b-instruct": ("meta-llama/Llama-3.2-3B-Instruct", FlexLlamaForCausalLM),
}
DEFAULT_MODEL_KEY = "qwen3-4b-instruct"


def load_model(model_key: str, device: str = "cuda"):
    """Load (model, tokenizer, num_hidden_layers) for the given model key."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model key: {model_key!r}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    name, model_cls = MODEL_REGISTRY[model_key]
    print(f"Loading model: {name}")
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = HFModelConfig(
        pretrained_model_name_or_path=name,
        model_cls=model_cls,
    ).instantiate()
    model = model.to(device).to(torch.bfloat16)
    for p in model.parameters():
        p.requires_grad = False
    num_layers = getattr(model.config, "num_hidden_layers", None) or 28
    return model, tokenizer, num_layers
