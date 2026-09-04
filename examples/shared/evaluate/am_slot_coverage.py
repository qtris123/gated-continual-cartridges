#!/usr/bin/env python3
"""Report which cartridge slots a continual-AM lineage actually rewrites.

Each stage applies one closed-form write per document, and each write is confined
to the `top_t` slots the slot selector picked for that document. If the selector
is largely document-independent, the per-document writes land on the same slots
and overwrite each other, so the stage's effective capacity is far below
`n_documents * top_t`. This reads the `positions_per_layer` recorded in the
`am_doc_*.pt` artifacts and reports, per stage, how many distinct slots per layer
were touched and how many times each was rewritten.

Usage:
    python examples/shared/evaluate/am_slot_coverage.py --dataset quality --tag <tag>
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch

from examples.shared.am.run_chain import RUN_DIRS


def stage_counts(runs_dir, dataset: str, tag: str, phase: int) -> tuple[torch.Tensor, int]:
    """Return an (n_layers, n_slots) rewrite-count tensor for one stage."""
    pattern = str(runs_dir / f"*{dataset.upper()}_P{phase}_{tag}" / "*" / "am_doc_*.pt")
    files = sorted(glob.glob(pattern))
    counts = None
    for path in files:
        record = torch.load(path, map_location="cpu", weights_only=False)
        mask = record["ranking_info"].mask
        if counts is None:
            counts = torch.zeros(len(mask.positions_per_layer), mask.n_tokens)
        for layer_idx, positions in mask.positions_per_layer.items():
            counts[layer_idx, positions.long()] += 1
    if counts is None:
        raise FileNotFoundError(f"No am_doc_*.pt under {pattern}")
    return counts, len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="qasper", choices=tuple(RUN_DIRS))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--phases", default="2,3,4,5")
    parser.add_argument("--runs-dir", default=None)
    args = parser.parse_args()
    phases = [int(p) for p in args.phases.split(",") if p.strip()]
    runs_dir = Path(args.runs_dir) if args.runs_dir else RUN_DIRS[args.dataset][0]

    header = (
        f"{'stage':<7}{'docs':>5}{'writes/layer':>14}{'unique/layer':>14}"
        f"{'coverage':>10}{'mean rewrites':>15}{'max':>5}"
    )
    print(header)
    cumulative = None
    for phase in phases:
        counts, n_docs = stage_counts(runs_dir, args.dataset, args.tag, phase)
        cumulative = counts.clone() if cumulative is None else cumulative + counts
        touched = counts[counts > 0]
        unique = (counts > 0).sum(1).float().mean().item()
        n_slots = counts.size(1)
        print(
            f"{'p%02d' % phase:<7}{n_docs:>5}{counts.sum(1).mean().item():>14.0f}"
            f"{unique:>14.1f}{unique / n_slots:>9.1%}"
            f"{touched.mean().item():>15.2f}{touched.max().item():>5.0f}"
        )

    unique = (cumulative > 0).sum(1).float().mean().item()
    n_slots = cumulative.size(1)
    print(
        f"\ncumulative: {unique:.1f}/{n_slots} slots per layer touched "
        f"({unique / n_slots:.1%}); mean rewrites on touched slots "
        f"{cumulative[cumulative > 0].mean().item():.2f}"
    )


if __name__ == "__main__":
    main()
