"""Evaluate cartridges on plasticity (phase 2) and/or forgetting (phase 1) eval sets.

Supports three evaluation modes:

  1. Single cartridge on a single eval set (original behaviour):
       --cache path/to/cache_last.pt --eval-data path/to/eval.parquet

  2. Continual evaluation (plasticity + forgetting) with two eval sets:
       --cache path/to/continual_cache.pt \\
       --phase1-eval path/to/phase1.parquet \\
       --phase2-eval path/to/phase2.parquet

  3. Full comparison — initial vs continual on both eval sets:
       --initial-cache  path/to/initial_cache.pt \\
       --cache          path/to/continual_cache.pt \\
       --phase1-eval    path/to/phase1.parquet \\
       --phase2-eval    path/to/phase2.parquet

Run combinations:
  initial  × phase1  → baseline retention
  continual × phase1 → forgetting check  (did continual training hurt phase-1 knowledge?)
  continual × phase2 → plasticity check  (did continual training add phase-2 knowledge?)
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.datasets import DataSource, GenerateEvalDataset, LossEvalDataset
from cartridges.generation import flex_generate
from cartridges.models import FlexLlamaForCausalLM, FlexQwen3ForCausalLM, HFModelConfig
from cartridges.train import CacheAndModel

MODEL_NAME = os.environ.get("MODEL_NAME")


def load_cache(path: str, device: str) -> TrainableCache:
    print(f"  Loading cache: {path}")
    cache = TrainableCache.from_pretrained(path, device=device)
    cache = cache.to(device)
    n_total = cache.num_cartridge_tokens()
    print(f"    {cache._num_frozen_tokens} frozen + {cache._num_trainable_tokens} trainable = {n_total} tokens")
    return cache


def evaluate_loss(
    model: torch.nn.Module,
    cache: TrainableCache,
    eval_dataset: LossEvalDataset,
    device: str = "cuda",
) -> dict:
    dataloader = DataLoader(
        eval_dataset,
        batch_size=1,
        collate_fn=lambda x: x[0],
        num_workers=0,
    )

    wrapped = CacheAndModel(cache, model)
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

            cache.clear()

    mean_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": mean_loss,
        "log_perplexity": mean_loss,
        "num_tokens": total_tokens,
        "num_batches": len(dataloader),
    }


def generate_for_dataset(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    cache: TrainableCache,
    eval_dataset: GenerateEvalDataset,
    device: str,
    max_new_tokens: int = 256,
) -> list[dict]:
    results = []

    for idx in tqdm(range(len(eval_dataset)), desc="Generating"):
        element = eval_dataset[idx]

        input_ids = element.input_ids.flatten().to(device)
        seq_ids = torch.zeros_like(input_ids)
        position_ids = torch.arange(len(input_ids), device=device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated_tokens = flex_generate(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                seq_ids=seq_ids,
                position_ids=position_ids,
                cache=cache,
                max_new_tokens=max_new_tokens,
            )

        gen_token_ids = generated_tokens.get(0, [])
        generated_text = tokenizer.decode(gen_token_ids, skip_special_tokens=True)

        if isinstance(element.prompt, list):
            question_text = element.prompt[-1].get("content", "") if element.prompt else ""
        else:
            question_text = element.prompt

        results.append({
            "question_id": element.convo_id,
            "question": question_text,
            "generated_answer": generated_text,
            "reference_answer": element.answer,
            **element.metadata,
        })

    return results


def run_eval(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    cache: TrainableCache,
    eval_path: str,
    device: str,
    max_new_tokens: int,
    output_dir: Path,
    label: str,
    model_name: str,
) -> dict:
    """Run loss + generation eval for one (cache, eval_set) pair. Returns perplexity result dict."""
    print(f"\n--- {label} ---")
    cartridge_size = cache.num_cartridge_tokens()
    model_short = model_name.split("/")[-1]
    eval_stem = Path(eval_path).stem  # e.g. amd_eval_questions_phase2

    file_tag = f"{model_short}_{cartridge_size}tok_{eval_stem}_{label}"
    file_tag = file_tag.replace(" ", "_").replace("/", "-")

    # Loss eval
    loss_dataset = LossEvalDataset.Config(
        data_source=DataSource(path=eval_path, type="local"),
    ).instantiate(tokenizer=tokenizer, seed=42)
    print(f"  loss eval samples: {len(loss_dataset)}")
    ppl_result = evaluate_loss(model, cache, loss_dataset, device=device)
    ppl_result["cache"] = str(cache)
    ppl_result["eval_set"] = eval_path
    ppl_result["label"] = label
    ppl_result["model"] = model_name
    ppl_result["cartridge_tokens"] = cartridge_size
    print(f"  log-perplexity: {ppl_result['log_perplexity']:.4f} ({ppl_result['num_tokens']} tokens)")

    # Generation eval
    gen_dataset = GenerateEvalDataset.Config(
        data_source=DataSource(path=eval_path, type="local"),
    ).instantiate(tokenizer=tokenizer, seed=42)
    print(f"  generation questions: {len(gen_dataset)}")
    generations = generate_for_dataset(
        model=model,
        tokenizer=tokenizer,
        cache=cache,
        eval_dataset=gen_dataset,
        device=device,
        max_new_tokens=max_new_tokens,
    )

    ppl_path = output_dir / f"perplexity_{file_tag}.json"
    gen_path = output_dir / f"generations_{file_tag}.json"
    with open(ppl_path, "w") as f:
        json.dump(ppl_result, f, indent=2)
    with open(gen_path, "w") as f:
        json.dump(generations, f, indent=2)
    print(f"  saved → {ppl_path.name}, {gen_path.name}")

    return ppl_result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate cartridges on plasticity and/or forgetting eval sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cache", type=str, default=None,
        help="Path to continual (phase 2) cartridge checkpoint",
    )
    parser.add_argument(
        "--initial-cache", type=str, default=None,
        help="Path to initial (phase 1) cartridge checkpoint",
    )
    parser.add_argument(
        "--eval-data", type=str,
        default=os.environ.get("EVAL_DATA_PATH"),
        help="Alias for --phase2-eval (backward compat)",
    )
    parser.add_argument(
        "--phase1-eval", type=str, default=None,
        help="Path to phase-1 eval parquet (forgetting detection, doc_a only)",
    )
    parser.add_argument(
        "--phase2-eval", type=str, default=None,
        help="Path to phase-2 eval parquet (plasticity, doc_a + doc_b)",
    )
    parser.add_argument("--model", type=str, default=MODEL_NAME, required=MODEL_NAME is None, help="HuggingFace model ID (or set MODEL_NAME env var)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-dir", type=str,
        default=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
        help="Directory to save evaluation results",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument(
        "--check-forgetting", action="store_true",
        help=(
            "Also run continual × phase1 (forgetting check). "
            "Useful for A→B experiments (e.g. AMD→Pepsi) where domains are disjoint. "
            "Not needed for A→delta_A (e.g. AMD2021→AMD2022) where phase1 answers change."
        ),
    )
    args = parser.parse_args()

    # --phase2-eval / --eval-data aliasing
    phase2_eval = args.phase2_eval or args.eval_data
    phase1_eval = args.phase1_eval

    if not args.cache and not args.initial_cache:
        parser.error("At least one of --cache or --initial-cache is required.")
    if not phase1_eval and not phase2_eval:
        parser.error("At least one of --phase1-eval / --phase2-eval / --eval-data is required.")

    device = args.device
    model_name = args.model
    model_cls = FlexQwen3ForCausalLM if "qwen" in model_name.lower() else FlexLlamaForCausalLM

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = HFModelConfig(
        pretrained_model_name_or_path=model_name,
        model_cls=model_cls,
    ).instantiate()
    model = model.to(device).to(torch.bfloat16)
    for param in model.parameters():
        param.requires_grad = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    # Build list of (cache_path, cache_label, eval_path, eval_label) to run
    runs: list[tuple[str, str, str, str]] = []

    if args.initial_cache and phase1_eval:
        runs.append((args.initial_cache, "initial", phase1_eval, "phase1_baseline"))
    if args.cache and phase1_eval and args.check_forgetting:
        runs.append((args.cache, "continual", phase1_eval, "phase1_forgetting"))
    if args.cache and phase2_eval:
        runs.append((args.cache, "continual", phase2_eval, "phase2_plasticity"))
    # Fallback: single cache + single eval (original behaviour)
    if not runs:
        cache_path = args.cache or args.initial_cache
        eval_path = phase2_eval or phase1_eval
        runs.append((cache_path, "cache", eval_path, "eval"))

    loaded_caches: dict[str, TrainableCache] = {}
    for cache_path, cache_label, eval_path, eval_label in runs:
        if cache_path not in loaded_caches:
            loaded_caches[cache_path] = load_cache(cache_path, device)
        cache = loaded_caches[cache_path]
        label = f"{cache_label}_x_{eval_label}"
        result = run_eval(
            model=model,
            tokenizer=tokenizer,
            cache=cache,
            eval_path=eval_path,
            device=device,
            max_new_tokens=args.max_new_tokens,
            output_dir=output_dir,
            label=label,
            model_name=model_name,
        )
        all_results.append(result)

    # Summary table
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"{'Label':<40} {'log-ppl':>8}  {'tokens':>8}")
        print("-" * 60)
        for r in all_results:
            print(f"  {r['label']:<38} {r['log_perplexity']:>8.4f}  {r['num_tokens']:>8}")
        print("=" * 60)

        model_short = model_name.split("/")[-1]
        cartridge_size = all_results[0].get("cartridge_tokens", "unknown")
        eval_stems = "_".join(dict.fromkeys(
            Path(r["eval_set"]).stem for r in all_results
        ))
        summary_path = output_dir / f"summary_{model_short}_{cartridge_size}tok_{eval_stems}.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
