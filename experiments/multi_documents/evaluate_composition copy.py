"""Evaluate composed cartridges on multi-document questions.

This script implements the cartridge composition experiment (Figure 7 of the
Cartridges paper). It loads independently trained cartridges for AMD and Pepsi,
composes them by concatenating their KV caches, and measures log-perplexity on
multi-document evaluation questions.

Conditions evaluated per cartridge size:
  1. Composed (AMD + Pepsi) cartridge
  2. AMD-only cartridge on multi-doc questions
  3. Pepsi-only cartridge on multi-doc questions
  4. No cartridge baseline

Usage:
    # Evaluate a single pair of cartridge checkpoints:
    python experiments/multi_documents/evaluate_composition.py \
        --amd-cache path/to/amd/cache_last.pt \
        --pepsi-cache path/to/pepsi/cache_last.pt \
        --eval-data path/to/eval/artifact/dataset.parquet

    # Evaluate all 4 sizes at once (looks for checkpoints in CARTRIDGES_OUTPUT_DIR):
    python experiments/multi_documents/evaluate_composition.py \
        --sweep \
        --eval-data path/to/eval/artifact/dataset.parquet
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.datasets import DataSource, LossEvalDataset
from cartridges.models import FlexLlamaForCausalLM, HFModelConfig
from cartridges.train import CacheAndModel

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
CARTRIDGE_SIZES = [1024]
AMD_TEXT_PATH = str(SCRIPT_DIR / "data" / "texts" / "AMD_2022_10K.txt")
PEPSI_TEXT_PATH = str(SCRIPT_DIR / "data" / "texts" / "PEPSICO_2022_10K.txt")


def compose_caches(
    cache_a: TrainableCache,
    cache_b: TrainableCache,
) -> TrainableCache:
    """Compose two trained cartridges by concatenating their KV caches.

    Both caches must have the same AttnConfig (n_layers, n_heads, head_dim).
    The composed cache concatenates all tokens (frozen + trainable) from both
    caches along the sequence dimension (dim=2). All tokens get
    CARTRIDGE_SEQ_ID = -1, making them visible to all query tokens via
    FlexAttention block masks.
    """
    assert cache_a.config == cache_b.config, (
        f"AttnConfig mismatch: {cache_a.config} vs {cache_b.config}"
    )

    n_layers = cache_a.config.n_layers

    def _get_full_kv(cache: TrainableCache, layer_idx: int):
        """Get the full key/value for a layer (frozen + trainable)."""
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
    # All tokens are frozen in the composed cache (no training needed)
    return TrainableCache(
        config=cache_a.config,
        init_keys=composed_keys,
        init_values=composed_values,
        num_frozen_tokens=total_tokens,
    )


def evaluate_loss(
    model: torch.nn.Module,
    cache: Optional[TrainableCache],
    eval_dataset: LossEvalDataset,
    device: str = "cuda",
) -> dict:
    """Compute mean cross-entropy loss on an evaluation dataset.

    Returns dict with 'loss', 'log_perplexity', 'num_tokens', 'num_batches'.
    """
    dataloader = DataLoader(
        eval_dataset,
        batch_size=1,
        collate_fn=lambda x: x[0],
        num_workers=0,
    )

    if cache is not None:
        wrapped = CacheAndModel(cache, model)
    else:
        empty_cache = TrainableCache(config=AttnConfig(
            n_layers=model.config.num_hidden_layers,
            n_heads=model.config.num_key_value_heads,
            head_dim=(
                model.config.head_dim
                if hasattr(model.config, "head_dim")
                else model.config.hidden_size // model.config.num_attention_heads
            ),
        ))
        empty_cache = empty_cache.to(device)
        wrapped = CacheAndModel(empty_cache, model)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = wrapped(
                    input_ids=batch.input_ids.to(device),
                    seq_ids=batch.element_ids.to(device),
                    position_ids=batch.position_ids.to(device),
                )

                topk_pred_logprobs = F.log_softmax(outputs.logits, dim=-1)[
                    0,
                    batch.topk_token_idxs.to(device) - 1,
                    batch.topk_token_ids.to(device),
                ]

                ce_by_token = (
                    -batch.topk_logprobs.to(device).exp() * topk_pred_logprobs
                )

                total_loss += ce_by_token.sum().item()
                total_tokens += ce_by_token.shape[0]

            if cache is not None:
                cache.clear()
            else:
                empty_cache.clear()

    mean_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": mean_loss,
        "log_perplexity": mean_loss,
        "num_tokens": total_tokens,
        "num_batches": len(dataloader),
    }


def load_cache(path: str, device: str) -> TrainableCache:
    """Load a cartridge checkpoint and move to device."""
    print(f"  Loading cache: {path}")
    cache = TrainableCache.from_pretrained(path, device=device)
    cache = cache.to(device)
    n_total = cache._num_frozen_tokens + cache._num_trainable_tokens
    print(f"    {cache._num_frozen_tokens} frozen + {cache._num_trainable_tokens} trainable = {n_total} tokens")
    return cache


def evaluate_single_size(
    model: torch.nn.Module,
    eval_dataset: LossEvalDataset,
    amd_cache_path: str,
    pepsi_cache_path: str,
    device: str,
    tokenizer=None,
    eval_data_path: str = None,
) -> dict:
    """Evaluate all conditions for a single cartridge size.

    Returns a dict mapping condition name -> result dict.
    """
    cache_amd = load_cache(amd_cache_path, device)
    cache_pepsi = load_cache(pepsi_cache_path, device)

    print("  Composing cartridges (AMD + Pepsi)...")
    cache_composed = compose_caches(cache_amd, cache_pepsi)
    cache_composed = cache_composed.to(device)

    conditions = [
        ("composed", cache_composed),
        ("amd_only", cache_amd),
        ("pepsi_only", cache_pepsi),
        #("no_cartridge", None),
    ]

    results = {}
    for name, cache in conditions:
        print(f"  Evaluating: {name}")
        result = evaluate_loss(model, cache, eval_dataset, device=device)
        results[name] = result
        print(f"    Loss: {result['loss']:.4f}  Log-PPL: {result['log_perplexity']:.4f}")

    #ICL baseline: feed both raw documents as system prompt
    if tokenizer is not None and eval_data_path is not None:
        print("  Evaluating: icl_128k")
        ICL_SEQ_LENGTH = 32000
        QUESTION_RESERVE = 512  # tokens reserved for eval questions

        with open(AMD_TEXT_PATH, "r") as f:
            amd_text = f.read()
        with open(PEPSI_TEXT_PATH, "r") as f:
            pepsi_text = f.read()

        # Truncate each document to fit within context budget
        per_doc_budget = (ICL_SEQ_LENGTH - QUESTION_RESERVE) // 2
        amd_tokens = tokenizer.encode(amd_text)[:per_doc_budget]
        pepsi_tokens = tokenizer.encode(pepsi_text)[:per_doc_budget]
        amd_text_trunc = tokenizer.decode(amd_tokens, skip_special_tokens=True)
        pepsi_text_trunc = tokenizer.decode(pepsi_tokens, skip_special_tokens=True)
        combined_text = amd_text_trunc + "\n\n" + pepsi_text_trunc

        print(f"    ICL context: {len(amd_tokens)} AMD + {len(pepsi_tokens)} Pepsi tokens, {QUESTION_RESERVE} reserved for questions")

        icl_eval_dataset = LossEvalDataset.Config(
            data_source=DataSource(path=eval_data_path, type="local"),
            packed_seq_length=ICL_SEQ_LENGTH,
            system_prompt=combined_text,
        ).instantiate(tokenizer=tokenizer, seed=42)

        result = evaluate_loss(model, None, icl_eval_dataset, device=device)
        results["icl_128k"] = result
        print(f"    Loss: {result['loss']:.4f}  Log-PPL: {result['log_perplexity']:.4f}")

    return results


def find_cache_path(output_dir: str, doc_name: str, num_tokens: int) -> str:
    """Find the cache checkpoint path for a given doc/size in CARTRIDGES_OUTPUT_DIR.

    Looks for the pattern: {doc}_train_lr0.02_toks{num_tokens}/cache_last.pt
    """
    run_name = f"{doc_name}_train_lr0.02_toks{num_tokens}"
    path = os.path.join(output_dir, run_name, "cache_last.pt")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate composed cartridges on multi-document questions"
    )
    parser.add_argument(
        "--amd-cache", type=str, default=None,
        help="Path to AMD cartridge checkpoint (e.g. cache_last.pt)",
    )
    parser.add_argument(
        "--pepsi-cache", type=str, default=None,
        help="Path to Pepsi cartridge checkpoint (e.g. cache_last.pt)",
    )
    parser.add_argument(
        "--eval-data", type=str, required=True,
        help="Path to multi-doc eval questions parquet file",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Sweep over all 4 cartridge sizes {512,1024,2048,4096}",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    output_dir = os.environ.get("CARTRIDGES_OUTPUT_DIR", ".")

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = HFModelConfig(
        pretrained_model_name_or_path=MODEL_NAME,
        model_cls=FlexLlamaForCausalLM,
    ).instantiate()
    model = model.to(device).to(torch.bfloat16)
    for param in model.parameters():
        param.requires_grad = False

    # Load eval dataset
    eval_dataset = LossEvalDataset.Config(
        data_source=DataSource(path=args.eval_data, type="local"),
        packed_seq_length=2048,
    ).instantiate(tokenizer=tokenizer, seed=42)
    print(f"Eval dataset: {len(eval_dataset)} batches")

    all_results = {}

    if args.sweep:
        # Evaluate all 4 cartridge sizes
        for num_tokens in CARTRIDGE_SIZES:
            print(f"\n{'='*60}")
            print(f"Cartridge size: {num_tokens}")
            print(f"{'='*60}")

            amd_path = find_cache_path(output_dir, "amd", num_tokens)
            pepsi_path = find_cache_path(output_dir, "pepsi", num_tokens)

            if not os.path.exists(amd_path):
                print(f"  WARNING: AMD cache not found at {amd_path}, skipping")
                continue
            if not os.path.exists(pepsi_path):
                print(f"  WARNING: Pepsi cache not found at {pepsi_path}, skipping")
                continue

            results = evaluate_single_size(
                model, eval_dataset, amd_path, pepsi_path, device,
                tokenizer=tokenizer, eval_data_path=args.eval_data,
            )
            all_results[num_tokens] = results
    else:
        # Single pair of caches
        if args.amd_cache is None or args.pepsi_cache is None:
            parser.error("--amd-cache and --pepsi-cache are required when not using --sweep")

        results = evaluate_single_size(
            model, eval_dataset, args.amd_cache, args.pepsi_cache, device,
            tokenizer=tokenizer, eval_data_path=args.eval_data,
        )
        all_results["single"] = results

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Size':<10} {'Condition':<20} {'Loss':>10} {'Log-PPL':>12} {'Tokens':>10}")
    print(f"{'-'*80}")
    for size, results in all_results.items():
        for condition, result in results.items():
            print(
                f"{str(size):<10} {condition:<20} {result['loss']:>10.4f} "
                f"{result['log_perplexity']:>12.4f} {result['num_tokens']:>10}"
            )
    print(f"{'='*80}")

    # Save results
    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "composition_results.json"

    # Convert int keys to strings for JSON
    serializable = {str(k): v for k, v in all_results.items()}
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
