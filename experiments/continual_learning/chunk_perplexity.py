"""Chunk perplexity probing — "What to Update?"

Given an old document (e.g. AMD 2021) encoded into a cartridge and an updated
document (e.g. AMD 2022), identify which parts of the new document the cartridge
already knows vs. which parts contain genuinely new information.

Load the model with the 2021 cartridge, feed the 2022 text chunk-by-chunk, and
measure per-chunk perplexity.  Low perplexity = the cartridge already has this
knowledge.  High perplexity = new information that needs updating.

Usage:
    python experiments/continual_learning/chunk_perplexity.py \
        --cache outputs/cartridge_amd_2021_10k_512.pt \
        --document data/texts/AMD_2022_10K.txt

    # With baseline comparison:
    python experiments/continual_learning/chunk_perplexity.py \
        --cache outputs/cartridge_amd_2021_10k_512.pt \
        --document data/texts/AMD_2022_10K.txt \
        --baseline
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.models import FlexLlamaForCausalLM, HFModelConfig

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def load_cache(path: str, device: str) -> TrainableCache:
    """Load a cartridge checkpoint."""
    print(f"  Loading cache: {path}")
    cache = TrainableCache.from_pretrained(path, device=device)
    cache = cache.to(device)
    n_total = cache._num_frozen_tokens + cache._num_trainable_tokens
    print(f"    {cache._num_frozen_tokens} frozen + {cache._num_trainable_tokens} trainable = {n_total} tokens")
    return cache


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_chunk_perplexity(
    model: torch.nn.Module,
    cart: TrainableCache,
    chunk_ids: torch.Tensor,
    device: str,
) -> dict:
    """Single forward pass over a chunk, returns loss and perplexity.

    Uses mode="train" with use_cache=True so that chunk tokens are NOT
    appended to the cartridge cache (skip_append=True in train mode).
    The cartridge state is preserved unchanged.

    Args:
        model: The language model.
        cart: Cartridge cache (or empty cache for baseline).
        chunk_ids: 1-D tensor of token IDs for this chunk.
        device: Device string.

    Returns:
        Dict with avg_loss, perplexity, and per_token_loss.
    """
    chunk_ids = chunk_ids.to(device)
    seq_ids = torch.zeros_like(chunk_ids)
    position_ids = torch.arange(len(chunk_ids), device=device)

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(
            input_ids=chunk_ids,
            seq_ids=seq_ids,
            position_ids=position_ids,
            past_key_values=cart,
            use_cache=True,
            mode="train",
        )

    logits = outputs.logits[0]  # [chunk_size, vocab_size]
    shift_logits = logits[:-1]  # predict next token
    shift_labels = chunk_ids[1:]

    per_token_loss = F.cross_entropy(
        shift_logits.float(), shift_labels, reduction="none",
    )

    avg_loss = per_token_loss.mean().item()
    perplexity = torch.exp(per_token_loss.mean()).item()

    return {
        "avg_loss": round(avg_loss, 4),
        "perplexity": round(perplexity, 4),
        "per_token_loss": [round(x, 4) for x in per_token_loss.tolist()],
    }


def run_chunk_perplexity(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    cart: TrainableCache,
    chunks: list[torch.Tensor],
    device: str,
    baseline_cart: TrainableCache | None = None,
) -> list[dict]:
    """Iterate over chunks, compute perplexity with cartridge (and optionally baseline).

    Args:
        model: The language model.
        tokenizer: Tokenizer for text preview decoding.
        cart: Loaded cartridge cache.
        chunks: List of 1-D token ID tensors.
        device: Device string.
        baseline_cart: If not None, also compute perplexity without cartridge.

    Returns:
        List of result dicts, one per chunk.
    """
    results = []
    start_token = 0

    for chunk_idx, chunk_ids in enumerate(tqdm(chunks, desc="Chunks (cartridge)")):
        num_tokens = len(chunk_ids)
        text_preview = tokenizer.decode(chunk_ids.tolist(), skip_special_tokens=True)

        # Cartridge run
        cart_result = compute_chunk_perplexity(model, cart, chunk_ids, device)

        entry = {
            "chunk_idx": chunk_idx,
            "start_token": start_token,
            "end_token": start_token + num_tokens,
            "num_tokens": num_tokens,
            "text_preview": text_preview,
            "avg_loss": cart_result["avg_loss"],
            "perplexity": cart_result["perplexity"],
        }

        results.append(entry)
        start_token += num_tokens

    # Baseline pass (if requested)
    if baseline_cart is not None:
        print("\nRunning baseline (no cartridge)...")
        for chunk_idx, chunk_ids in enumerate(tqdm(chunks, desc="Chunks (baseline)")):
            baseline_result = compute_chunk_perplexity(
                model, baseline_cart, chunk_ids, device,
            )
            results[chunk_idx]["baseline_avg_loss"] = baseline_result["avg_loss"]
            results[chunk_idx]["baseline_perplexity"] = baseline_result["perplexity"]

            cart_ppl = results[chunk_idx]["perplexity"]
            base_ppl = baseline_result["perplexity"]
            reduction = round(1.0 - (cart_ppl / base_ppl), 4) if base_ppl > 0 else 0.0
            results[chunk_idx]["perplexity_reduction"] = reduction

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Chunk perplexity probing — identify what a cartridge already knows"
    )
    parser.add_argument("--cache", type=str, required=True,
                        help="Path to cartridge checkpoint (.pt)")
    parser.add_argument("--document", type=str, required=True,
                        help="Path to text file (e.g. data/texts/AMD_2022_10K.txt)")
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="Tokens per chunk (default: 512)")
    parser.add_argument("--baseline", action="store_true",
                        help="Also run without cartridge for comparison")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str,
                        default=os.environ.get("CARTRIDGES_OUTPUT_DIR", "./outputs"))
    args = parser.parse_args()

    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = HFModelConfig(
        pretrained_model_name_or_path=MODEL_NAME,
        model_cls=FlexLlamaForCausalLM,
    ).instantiate()
    model = model.to(device).to(torch.bfloat16)
    for param in model.parameters():
        param.requires_grad = False

    # Load cartridge
    cart = load_cache(args.cache, device)

    # Load and tokenize document
    print(f"Loading document: {args.document}")
    with open(args.document, "r") as f:
        text = f.read()
    all_token_ids = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(all_token_ids)
    print(f"  Total tokens: {total_tokens}")

    # Split into chunks
    chunks = []
    for i in range(0, total_tokens, args.chunk_size):
        chunk = torch.tensor(all_token_ids[i : i + args.chunk_size], dtype=torch.long)
        chunks.append(chunk)
    print(f"  {len(chunks)} chunks of size {args.chunk_size}")

    # Build empty cache for baseline if requested
    baseline_cart = None
    if args.baseline:
        baseline_cart = TrainableCache(config=AttnConfig(
            n_layers=model.config.num_hidden_layers,
            n_heads=model.config.num_key_value_heads,
            head_dim=(
                model.config.head_dim
                if hasattr(model.config, "head_dim")
                else model.config.hidden_size // model.config.num_attention_heads
            ),
        ))
        baseline_cart = baseline_cart.to(device)

    # Run
    print("\n" + "=" * 60)
    print("Chunk perplexity probing")
    print("=" * 60)
    results = run_chunk_perplexity(
        model, tokenizer, cart, chunks, device,
        baseline_cart=baseline_cart,
    )

    # Build metadata
    metadata = {
        "cache_path": args.cache,
        "document_path": args.document,
        "chunk_size": args.chunk_size,
        "num_chunks": len(chunks),
        "total_document_tokens": total_tokens,
        "num_frozen_tokens": cart._num_frozen_tokens,
        "num_trainable_tokens": cart._num_trainable_tokens,
        "num_total_tokens": cart._num_frozen_tokens + cart._num_trainable_tokens,
        "n_layers": cart.config.n_layers,
        "baseline": args.baseline,
    }

    # Save
    out_path = output_dir / f"chunk_perplexity_{cart._num_frozen_tokens + cart._num_trainable_tokens}.json"
    with open(out_path, "w") as f:
        json.dump({"metadata": metadata, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} chunk results to {out_path}")


if __name__ == "__main__":
    main()
