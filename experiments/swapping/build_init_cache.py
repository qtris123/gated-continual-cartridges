#!/usr/bin/env python3
"""Rebuild a step-0 (context-init) cartridge via KVFromText and save it.

Mirrors the init path run at training step 0 (cartridges.train -> KVFromText),
so the saved cache has the same shape and init values as the trained
`cache_last.pt` would have had before any gradient steps were taken.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from cartridges.cache import AttnConfig
from cartridges.initialization.text import KVFromText

from experiments.intervention import DEFAULT_MODEL_KEY, MODEL_REGISTRY, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL_KEY,
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--text", required=True,
                        help="Path to the text file used as the KV init corpus.")
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--num-frozen-tokens", type=int, default=1)
    parser.add_argument("--output", required=True,
                        help="Destination .pt for the TrainableCache checkpoint.")
    args = parser.parse_args()

    device = "cuda"
    model, tokenizer, num_layers = load_model(args.model, device)
    print(f"  num_hidden_layers={num_layers}")

    attn_config = AttnConfig(
        n_layers=model.config.num_hidden_layers,
        n_heads=model.config.num_key_value_heads,
        head_dim=(
            model.config.head_dim
            if hasattr(model.config, "head_dim")
            else model.config.hidden_size // model.config.num_attention_heads
        ),
    )
    print(f"  attn_config: n_layers={attn_config.n_layers} "
          f"n_heads={attn_config.n_heads} head_dim={attn_config.head_dim}")

    cfg = KVFromText.Config(
        max_tokens=args.max_tokens,
        text_source=args.text,
        num_frozen_tokens=args.num_frozen_tokens,
        system_prompt_template="{text}",
    )
    factory = KVFromText(config=cfg)
    cache = factory.initialize_kv_cache(
        tokenizer=tokenizer, model=model, attn_config=attn_config,
    )
    print(f"  built cache: frozen={cache._num_frozen_tokens} "
          f"trainable={cache._num_trainable_tokens} "
          f"V[0].shape={tuple(cache.trainable_values[0].shape)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache.save(str(out))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
