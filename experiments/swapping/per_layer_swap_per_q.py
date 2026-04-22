#!/usr/bin/env python3
"""Per-question per-layer A→B V-swap sweep."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.intervention import (
    DEFAULT_MODEL_KEY,
    MODEL_REGISTRY,
    build_datasets,
    load_cache,
    load_model,
    parse_eval_args,
    per_layer_swap_trace_per_q,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL_KEY,
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--cache-a", required=True)
    parser.add_argument("--cache-b", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    device = "cuda"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, num_layers = load_model(args.model, device)
    print(f"  num_hidden_layers={num_layers}")

    eval_files = parse_eval_args([args.eval])
    datasets = build_datasets(eval_files, tokenizer, "openended")
    (eval_name, ds), = datasets.items()
    print(f"  eval={eval_name}  size={len(ds)}")
    question_ids = [str(ds[i].convo_id) for i in range(len(ds))]

    base_cache = load_cache(args.cache_b, device)
    donor_cache = load_cache(args.cache_a, device)

    t0 = time.time()
    base_per_q, donor_per_q, grid = per_layer_swap_trace_per_q(
        model, base_cache, donor_cache, ds, device, batch_size=args.batch_size,
    )
    dt = time.time() - t0

    gap_per_q = base_per_q - donor_per_q
    R_per_q = (base_per_q.unsqueeze(1) - grid) / gap_per_q.unsqueeze(1).clamp(min=1e-6)
    print(f"\nDone in {dt:.1f}s")
    print(f"questions with gap > 0.1: {(gap_per_q > 0.1).sum().item()} / {len(ds)}")

    torch.save(
        {
            "grid_log_ppl": grid,
            "R_per_q": R_per_q,
            "base_per_q": base_per_q,
            "donor_per_q": donor_per_q,
            "gap_per_q": gap_per_q,
        },
        out_dir / "per_layer_swap_per_q.pt",
    )
    meta = {
        "model": args.model,
        "cache_a_donor": args.cache_a,
        "cache_b_base": args.cache_b,
        "eval_name": eval_name,
        "eval_path": eval_files[eval_name],
        "n_questions": len(ds),
        "num_layers": int(num_layers),
        "question_ids": question_ids,
        "elapsed_sec": dt,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {out_dir}/per_layer_swap_per_q.pt and meta.json")


if __name__ == "__main__":
    main()
