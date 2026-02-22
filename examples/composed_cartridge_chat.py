#!/usr/bin/env python3
"""
Interactive CLI for chatting with locally-loaded (and optionally composed) cartridges.

Usage:
    # Single local cartridge
    python examples/composed_cartridge_chat.py --cache path/to/cache_last.pt

    # Composed cartridges (two caches concatenated)
    python examples/composed_cartridge_chat.py --cache path/to/amd/cache_last.pt --cache2 path/to/pepsi/cache_last.pt

    # Override model name
    python examples/composed_cartridge_chat.py --cache path/to/cache_last.pt --model meta-llama/Llama-3.2-3B-Instruct
"""

import argparse
import torch
from transformers import AutoTokenizer

from cartridges.cache import TrainableCache
from cartridges.models import FlexLlamaForCausalLM, HFModelConfig
from examples.cartridge_chat import ChatSession

DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"


def compose_caches(
    cache_a: TrainableCache,
    cache_b: TrainableCache,
) -> TrainableCache:
    """Compose two trained cartridges by concatenating their KV caches.

    Both caches must have the same AttnConfig (n_layers, n_heads, head_dim).
    The composed cache concatenates all tokens (frozen + trainable) from both
    caches along the sequence dimension (dim=2).
    """
    assert cache_a.config == cache_b.config, (
        f"AttnConfig mismatch: {cache_a.config} vs {cache_b.config}"
    )

    n_layers = cache_a.config.n_layers

    def _get_full_kv(cache: TrainableCache, layer_idx: int):
        parts_k, parts_v = [], []
        if cache._num_frozen_tokens > 0:
            parts_k.append(cache.frozen_keys[layer_idx].data)
            parts_v.append(cache.frozen_values[layer_idx].data)
        if cache._num_trainable_tokens > 0:
            parts_k.append(cache.trainable_keys[layer_idx].data)
            parts_v.append(cache.trainable_values[layer_idx].data)
        return torch.cat(parts_k, dim=2), torch.cat(parts_v, dim=2)

    composed_keys = []
    composed_values = []
    for layer_idx in range(n_layers):
        keys_a, values_a = _get_full_kv(cache_a, layer_idx)
        keys_b, values_b = _get_full_kv(cache_b, layer_idx)
        composed_keys.append(torch.cat([keys_a, keys_b], dim=2))
        composed_values.append(torch.cat([values_a, values_b], dim=2))

    total_tokens = composed_keys[0].shape[2]
    return TrainableCache(
        config=cache_a.config,
        init_keys=composed_keys,
        init_values=composed_values,
        num_frozen_tokens=total_tokens,
    )


def load_cache(path: str, device: str) -> TrainableCache:
    """Load a cartridge checkpoint and move to device."""
    print(f"  Loading cache: {path}")
    cache = TrainableCache.from_pretrained(path, device=device)
    cache = cache.to(device)
    n_total = cache._num_frozen_tokens + cache._num_trainable_tokens
    print(f"    {cache._num_frozen_tokens} frozen + {cache._num_trainable_tokens} trainable = {n_total} tokens")
    return cache


def main():
    parser = argparse.ArgumentParser(
        description="Chat with locally-loaded (and optionally composed) cartridges"
    )
    parser.add_argument(
        "--cache", type=str, required=True,
        help="Path to a local cache checkpoint (e.g. cache_last.pt)",
    )
    parser.add_argument(
        "--cache2", type=str, default=None,
        help="Optional second cache to compose with the first",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"HuggingFace model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--enable_thinking", action="store_true",
        help="Enable thinking mode in chat template",
    )
    args = parser.parse_args()

    device = "cuda"

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = HFModelConfig(
        pretrained_model_name_or_path=args.model,
        model_cls=FlexLlamaForCausalLM,
    ).instantiate()
    model = model.to(device).to(torch.bfloat16)
    for param in model.parameters():
        param.requires_grad = False

    # Load cache(s)
    cache = load_cache(args.cache, device)
    cache = cache.to(torch.bfloat16)

    if args.cache2:
        cache2 = load_cache(args.cache2, device)
        cache2 = cache2.to(torch.bfloat16)
        print("Composing cartridges...")
        cache = compose_caches(cache, cache2)
        cache = cache.to(device).to(torch.bfloat16)

    print("Model and cache loaded successfully!\n")

    # Initialize chat session (reuse from cartridge_chat)
    chat = ChatSession(model, tokenizer, cache)

    import readline
    readline.set_startup_hook(None)

    mode = "composed" if args.cache2 else "single"
    print(f"=== Chat with {mode.title()} Cartridge ===")
    print("Commands:")
    print("  /undo  - Undo the last message exchange")
    print("  /clear - Clear the entire conversation")
    print("  /quit  - Exit the chat")
    print("  /help  - Show this help message")
    print("Arrow keys: up/down for command history, left/right for line editing")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input == "/quit":
                print("Goodbye!")
                break
            elif user_input == "/help":
                print("\nCommands:")
                print("  /undo  - Undo the last message exchange")
                print("  /clear - Clear the entire conversation")
                print("  /quit  - Exit the chat")
                print("  /help  - Show this help message")
                continue
            elif user_input == "/undo":
                if chat.undo_last_message():
                    print("Last message exchange undone.")
                else:
                    print("No messages to undo.")
                continue
            elif user_input == "/clear":
                chat.clear_conversation()
                print("Conversation cleared.")
                continue

            print("Assistant: ", end="", flush=True)
            response = chat.generate_response(user_input, enable_thinking=args.enable_thinking)
            print(response)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError generating response: {e}")
            print("Type /help for available commands.")


if __name__ == "__main__":
    main()
