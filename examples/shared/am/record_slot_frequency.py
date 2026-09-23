#!/usr/bin/env python3
"""Record, aggregate, and save slot frequency and geometry artifacts for an evaluated sweep arm.

Collects per-stage `slots_written.pt` artifacts produced across stages p02-p05,
computes cross-stage overlap (Jaccard similarity), cumulative cache coverage,
and frequency distributions, and saves consolidated artifacts under:
    outputs/evaluations/{dataset}/{tag}/slot_frequency/
        ├── slot_frequency.pt       # Consolidated per-stage & cumulative count tensors
        ├── summary.json            # High-level metrics, coverage, and overlap stats
        ├── jaccard_overlap.csv     # 4x4 cross-stage Jaccard matrix (p02..p05)
        └── per_layer_stats.csv     # Per-layer breakdown of slots touched & write counts

Usage:
    python -m examples.shared.am.record_slot_frequency --dataset qasper --tag llama3_2_3b_budget512_topt128
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Optional

import torch

from examples.shared.paths import ROOT

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("slot_frequency")

STAGE_KEYS = ["p02", "p03", "p04", "p05"]


def _get_state_dir(dataset: str, tag: str) -> Path:
    candidates = [
        ROOT / f"outputs/{dataset}_5phase_state" / tag,
        ROOT / "outputs" / f"{dataset}_state" / tag,
    ]
    if dataset == "qasper":
        candidates.append(ROOT / "outputs/qasper_asr_kg_state" / tag)
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def record_slot_frequency(
    dataset: str,
    tag: str,
    output_dir: Optional[Path] = None,
) -> Optional[dict]:
    state_dir = _get_state_dir(dataset, tag)
    if not state_dir.exists():
        logger.warning(f"  [slot_frequency] State directory not found: {state_dir}")
        return None

    if output_dir is None:
        output_dir = ROOT / "outputs/evaluations" / dataset / tag / "slot_frequency"
    output_dir = Path(output_dir)

    stage_records = []
    for sid in STAGE_KEYS:
        marker = state_dir / f"{sid}.json"
        if not marker.exists():
            logger.warning(f"  [slot_frequency] Missing stage marker: {marker}")
            return None
        data = json.loads(marker.read_text())
        cache_path = data.get("cache_path")
        if not cache_path:
            logger.warning(f"  [slot_frequency] No cache_path in marker: {marker}")
            return None
        cache_p = Path(cache_path)
        sidecar = cache_p.parent / "slots_written.pt"
        if not sidecar.exists():
            logger.warning(f"  [slot_frequency] Missing slots_written.pt beside {cache_p}")
            return None
        stage_records.append((sid, sidecar))

    # Load per-stage frequency tensors
    per_stage_counts = []
    cumulative_tensor = None
    usage_lambda = 0.0
    usage_decay = 1.0
    n_layers = 0
    n_slots = 0

    for sid, sidecar_path in stage_records:
        obj = torch.load(sidecar_path, map_location="cpu")
        if isinstance(obj, dict):
            st = obj["stage"].float()
            cum = obj["cumulative"].float()
            usage_lambda = float(obj.get("usage_penalty_lambda", usage_lambda))
            usage_decay = float(obj.get("usage_decay", usage_decay))
        else:
            st = obj.float()
            cum = st
        per_stage_counts.append(st)
        cumulative_tensor = cum
        n_layers, n_slots = st.shape

    stacked_stages = torch.stack(per_stage_counts, dim=0)  # (4, n_layers, n_slots)
    masks = [s > 0 for s in per_stage_counts]              # 4 boolean (n_layers, n_slots)

    # 1. Coverage
    stage_cov = [float(m.float().mean().item()) for m in masks]
    union_mask = torch.zeros_like(masks[0])
    for m in masks:
        union_mask = union_mask | m
    union_cov = float(union_mask.float().mean().item())
    total_distinct_slots = int((union_mask.any(dim=0)).sum().item())

    # 2. Cross-stage Jaccard overlap
    ns = len(masks)
    J = [[0.0] * ns for _ in range(ns)]
    for i in range(ns):
        for j in range(ns):
            inter = (masks[i] & masks[j]).float().sum(dim=1)
            uni = (masks[i] | masks[j]).float().sum(dim=1).clamp(min=1)
            J[i][j] = float((inter / uni).mean().item())
    consecutive_j = sum(J[k][k + 1] for k in range(ns - 1)) / max(ns - 1, 1)

    # 3. Frequency statistics
    max_writes = float(cumulative_tensor.max().item())
    mean_writes_on_active = (
        float(cumulative_tensor[cumulative_tensor > 0].mean().item())
        if (cumulative_tensor > 0).any()
        else 0.0
    )

    summary = {
        "dataset": dataset,
        "tag": tag,
        "n_stages": ns,
        "stages": STAGE_KEYS,
        "n_layers": n_layers,
        "n_slots": n_slots,
        "usage_penalty_lambda": usage_lambda,
        "usage_decay": usage_decay,
        "cumulative_union_coverage_ratio": round(union_cov, 4),
        "cumulative_union_coverage_pct": round(union_cov * 100, 2),
        "total_distinct_slots_written": total_distinct_slots,
        "per_stage_coverage_ratio": [round(c, 4) for c in stage_cov],
        "per_stage_coverage_pct": [round(c * 100, 2) for c in stage_cov],
        "jaccard_overlap_matrix": [[round(val, 4) for val in row] for row in J],
        "consecutive_jaccard_mean": round(consecutive_j, 4),
        "max_slot_write_count": int(max_writes),
        "mean_writes_per_active_slot": round(mean_writes_on_active, 2),
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save 1: slot_frequency.pt (PyTorch tensor archive)
    pt_archive = {
        "per_stage_counts": stacked_stages,       # (4, n_layers, n_slots)
        "cumulative_counts": cumulative_tensor,   # (n_layers, n_slots)
        "stages": STAGE_KEYS,
        "summary": summary,
    }
    torch.save(pt_archive, output_dir / "slot_frequency.pt")

    # Save 2: summary.json
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Save 3: jaccard_overlap.csv
    with open(output_dir / "jaccard_overlap.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage"] + STAGE_KEYS)
        for idx, sid in enumerate(STAGE_KEYS):
            writer.writerow([sid] + [f"{val:.4f}" for val in J[idx]])

    # Save 4: per_layer_stats.csv
    with open(output_dir / "per_layer_stats.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["layer", "distinct_slots_written", "coverage_pct", "total_writes", "max_slot_writes"]
        writer.writerow(header)
        for layer_idx in range(n_layers):
            layer_cum = cumulative_tensor[layer_idx]
            distinct = int((layer_cum > 0).sum().item())
            cov_pct = (distinct / n_slots) * 100.0 if n_slots > 0 else 0.0
            tot = float(layer_cum.sum().item())
            mx = float(layer_cum.max().item())
            writer.writerow([layer_idx, distinct, f"{cov_pct:.1f}", int(tot), int(mx)])

    logger.info(
        f"  [slot_frequency] Recorded slot frequency outputs to: {output_dir}\n"
        f"                   Coverage: {summary['cumulative_union_coverage_pct']}% | "
        f"Consecutive Jaccard: {summary['consecutive_jaccard_mean']:.3f} | "
        f"Max Writes/Slot: {summary['max_slot_write_count']}"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Record and save slot frequency summary.")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. qasper, quality, techqa, finqa)")
    parser.add_argument("--tag", required=True, help="Arm lineage tag (e.g. llama3_2_3b_budget512_topt128)")
    parser.add_argument("--output-dir", default=None, help="Destination directory for slot frequency outputs")
    args = parser.parse_args()

    record_slot_frequency(args.dataset, args.tag, Path(args.output_dir) if args.output_dir else None)


if __name__ == "__main__":
    main()
