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

from cartridges.benchmark.scorers import multiple_choice, yes_no_match
from cartridges.cache import TrainableCache
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


def _get_choice_token_ids(tokenizer: AutoTokenizer, question_type: str) -> dict[str, int]:
    """Get token IDs for choice labels based on question type.

    Encodes with a leading space because the model predicts ' B' (not 'B')
    after 'Answer:' and ' Yes'/' No' as the first generated token.
    """
    if question_type == "mcq":
        labels = [" A", " B", " C", " D"]
    elif question_type == "yes_no":
        labels = [" Yes", " No"]
    else:
        return {}
    return {label.strip(): tokenizer.encode(label, add_special_tokens=False)[0] for label in labels}


def _find_answer_token_position(gen_token_ids: list[int], tokenizer: AutoTokenizer) -> int | None:
    """Find the position of 'Answer:' in generated tokens.

    Returns the index of the last token of 'Answer:' so that logits at that
    position predict the choice token (A/B/C/D or Yes/No).
    """
    # Encode "Answer:" — may be 1 or more tokens depending on tokenizer
    answer_tokens = tokenizer.encode("Answer:", add_special_tokens=False)
    answer_len = len(answer_tokens)
    # Search from the end (the final "Answer:" is the one that matters)
    for i in range(len(gen_token_ids) - answer_len, -1, -1):
        if gen_token_ids[i:i + answer_len] == answer_tokens:
            # Return position of last token of "Answer:"
            # Logits at this position predicts it so low then 
            # the next token (the choice)
            return i + answer_len - 1
    return None



def generate_for_dataset(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    cache: TrainableCache,
    eval_dataset: GenerateEvalDataset,
    device: str,
    max_new_tokens: int = 256,
    eval_mode: str = "generate",
    max_samples: int | None = None,
    batch_size: int = 1,
) -> list[dict]:
    # Pre-compute choice token IDs per question type (cached across questions)
    choice_tokens_cache: dict[str, dict[str, int]] = {}

    results = []
    n = min(len(eval_dataset), max_samples) if max_samples else len(eval_dataset)
    print(f"  batch_size={batch_size}, total={n}, num_batches={-(-n // batch_size)}")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        print(f"  GPU memory before generation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # Determine question type and choice token IDs for logprob collection
    first_question_type = eval_dataset[0].metadata.get("question_type", "original") if n > 0 else "original"
    collect_token_ids = None
    if eval_mode == "generate-score" and first_question_type in ("mcq", "yes_no"):
        if first_question_type not in choice_tokens_cache:
            choice_tokens_cache[first_question_type] = _get_choice_token_ids(tokenizer, first_question_type)
        # Map label -> token_id, we need the raw token IDs for collection
        choice_map = choice_tokens_cache[first_question_type]  # e.g. {"A": 362, "B": 426, ...}
        collect_token_ids = list(choice_map.values())

    for batch_start in tqdm(range(0, n, batch_size), desc="Generating"):
        batch_end = min(batch_start + batch_size, n)
        batch_elements = [eval_dataset[idx] for idx in range(batch_start, batch_end)]

        # --- Batched generation: pack all questions with different seq_ids ---
        all_input_ids = []
        all_seq_ids = []
        all_position_ids = []

        for i, element in enumerate(batch_elements):
            input_ids = element.input_ids.flatten().to(device)
            all_input_ids.append(input_ids)
            all_seq_ids.append(torch.full_like(input_ids, i))
            all_position_ids.append(torch.arange(len(input_ids), device=device))

        batched_input_ids = torch.cat(all_input_ids)
        batched_seq_ids = torch.cat(all_seq_ids)
        batched_position_ids = torch.cat(all_position_ids)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            gen_result = flex_generate(
                model=model,
                tokenizer=tokenizer,
                input_ids=batched_input_ids,
                seq_ids=batched_seq_ids,
                position_ids=batched_position_ids,
                cache=cache,
                max_new_tokens=max_new_tokens,
                collect_token_ids=collect_token_ids,
            )

        if collect_token_ids is not None:
            generated_tokens, collected_logprobs = gen_result
        else:
            generated_tokens = gen_result
            collected_logprobs = None

        # --- Post-generation processing per sample ---
        for i, element in enumerate(batch_elements):
            question_type = element.metadata.get("question_type", "original")
            gen_token_ids = generated_tokens.get(i, [])
            generated_text = tokenizer.decode(gen_token_ids, skip_special_tokens=True)

            choice_logprobs = None
            predicted_from_logits = None

            # MCQ / Yes-No: look up logprobs at "Answer:" position from collected logits
            if eval_mode == "generate-score" and question_type in ("mcq", "yes_no") and collected_logprobs is not None:
                answer_pos = _find_answer_token_position(gen_token_ids, tokenizer)
                # answer_pos is the index of the last token of "Answer:" in gen_token_ids.
                # collected_logprobs[k] predicts gen_token_ids[k], so we need
                # answer_pos + 1 to get logprobs predicting the choice token.
                choice_pos = answer_pos + 1 if answer_pos is not None else None
                if choice_pos is not None and choice_pos < len(collected_logprobs.get(i, [])):
                    step_logprobs = collected_logprobs[i][choice_pos]
                    # Convert from {token_id: logprob} to {label: logprob}
                    choice_logprobs = {
                        label: step_logprobs[tid]
                        for label, tid in choice_map.items()
                    }
                    predicted_from_logits = max(choice_logprobs, key=choice_logprobs.get)

            if isinstance(element.prompt, list):
                question_text = element.prompt[-1].get("content", "") if element.prompt else ""
            else:
                question_text = element.prompt

            result = {
                "question_id": element.convo_id,
                "question": question_text,
                "generated_answer": generated_text,
                "reference_answer": element.answer,
                **element.metadata,
            }

            # --- Scoring (MCQ / Yes-No only, generate-score mode) ---
            if eval_mode == "generate-score" and question_type == "mcq":
                result["score"] = multiple_choice(generated_text, element.answer)
                result["correct"] = result["score"] == 1.0
                result["scorer"] = "multiple_choice"
            elif eval_mode == "generate-score" and question_type == "yes_no":
                result["score"] = yes_no_match(generated_text, element.answer)
                result["correct"] = result["score"] == 1.0
                result["scorer"] = "yes_no"

            if choice_logprobs is not None:
                result["choice_logprobs"] = choice_logprobs
                result["predicted_choice_from_logits"] = predicted_from_logits

            results.append(result)

    if device == "cuda":
        print(f"  GPU peak memory during generation: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    # --- Print accuracy summary for scored questions ---
    if eval_mode == "generate-score":
        scored = [r for r in results if "score" in r]
        if scored:
            total_score = sum(r["score"] for r in scored)
            print(f"\n  Accuracy: {total_score}/{len(scored)} = {total_score/len(scored):.2%}")

            # Breakdown by category
            from collections import defaultdict
            by_cat = defaultdict(list)
            for r in scored:
                by_cat[r.get("category", "unknown")].append(r["score"])
            for cat, scores in sorted(by_cat.items()):
                acc = sum(scores) / len(scores)
                print(f"    {cat}: {sum(scores):.0f}/{len(scores)} = {acc:.2%}")

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
    eval_mode: str = "generate",
    max_samples: int | None = None,
    batch_size: int = 1,
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
        eval_mode=eval_mode,
        max_samples=max_samples,
        batch_size=batch_size,
    )

    # Compute accuracy breakdown and add to ppl_result
    if eval_mode == "generate-score":
        from collections import defaultdict
        scored = [r for r in generations if "score" in r]
        if scored:
            total_correct = sum(r["score"] for r in scored)
            ppl_result["accuracy"] = total_correct / len(scored)
            ppl_result["num_correct"] = int(total_correct)
            ppl_result["num_scored"] = len(scored)

            # Breakdown by category
            by_cat = defaultdict(list)
            for r in scored:
                by_cat[r.get("category", "unknown")].append(r["score"])
            ppl_result["accuracy_by_category"] = {
                cat: {
                    "accuracy": sum(scores) / len(scores),
                    "num_correct": int(sum(scores)),
                    "num_total": len(scores),
                }
                for cat, scores in sorted(by_cat.items())
            }

            # Breakdown by question_type (useful for mixed eval sets)
            by_type = defaultdict(list)
            for r in scored:
                by_type[r.get("question_type", "unknown")].append(r["score"])
            ppl_result["accuracy_by_question_type"] = {
                qtype: {
                    "accuracy": sum(scores) / len(scores),
                    "num_correct": int(sum(scores)),
                    "num_total": len(scores),
                }
                for qtype, scores in sorted(by_type.items())
            }

    # Remove verbose cache repr, keep just the token counts
    ppl_result.pop("cache", None)
    ppl_result["cache_frozen_tokens"] = cache._num_frozen_tokens
    ppl_result["cache_trainable_tokens"] = cache._num_trainable_tokens

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
        description="Evaluate a cartridge on an eval set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Simple eval
  python cartridge.py --cache path/to/cache.pt --eval-data path/to/eval.parquet --model meta-llama/Llama-3.2-3B-Instruct

  # Multiple eval sets at once
  python cartridge.py --cache path/to/cache.pt --eval-data a.parquet b.parquet --model meta-llama/Llama-3.2-3B-Instruct

  # Custom label for output filenames
  python cartridge.py --cache path/to/cache.pt --eval-data eval.parquet --model ... --label initial_amd2021
""",
    )
    parser.add_argument(
        "--cache", type=str, required=True,
        help="Path to cartridge checkpoint (.pt file)",
    )
    parser.add_argument(
        "--eval-data", type=str, nargs="+", required=True,
        help="Path(s) to eval parquet file(s). Can pass multiple.",
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_NAME,
        required=MODEL_NAME is None,
        help="HuggingFace model ID (or set MODEL_NAME env var)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-dir", type=str,
        default=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
        help="Directory to save evaluation results",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of generation samples (for quick testing)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of questions to generate in parallel (default: 8)")
    parser.add_argument(
        "--eval-mode", type=str, default="generate-score",
        choices=["generate", "generate-score"],
        help=(
            "'generate': text generation only. "
            "'generate-score': generation + scoring + first-token logits (default)."
        ),
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="Optional label for output filenames. Defaults to the eval parquet stem.",
    )
    args = parser.parse_args()

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

    cache = load_cache(args.cache, device)

    all_results: list[dict] = []
    for eval_path in args.eval_data:
        label = args.label or Path(eval_path).stem
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
            eval_mode=args.eval_mode,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
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
            acc_str = f"  acc={r['accuracy']:.2%}" if "accuracy" in r else ""
            print(f"  {r['label']:<38} {r['log_perplexity']:>8.4f}  {r['num_tokens']:>8}{acc_str}")
        print("=" * 60)


if __name__ == "__main__":
    main()
