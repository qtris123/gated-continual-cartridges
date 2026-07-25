#!/usr/bin/env python3
"""
Merge ICL and cartridge eval summaries into a unified comparison table.

Usage::

    python compare_benchmark_results.py \\
        --icl-summaries path/to/icl_eval_summary.json ... \\
        --cartridge-summaries path/to/cartridge_eval_summary.json ... \\
        --output benchmark_comparison.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_summaries(paths: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"Warning: summary not found: {p}")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            rows.extend(data)
        else:
            rows.append(data)
    return rows


def _dataset_short_name(dataset_path: str) -> str:
    name = Path(dataset_path).stem
    if "QA" in name:
        task = "QA"
    elif "MT" in name:
        task = "MT"
    else:
        task = name
    if "mcq" in name.lower():
        fmt = "mcq"
    elif "yes" in name.lower():
        fmt = "yes_no"
    else:
        fmt = "unknown"
    return f"{task}_{fmt}"


def _classify_row(row: Dict[str, Any]) -> str:
    """Assign a method label for comparison."""
    if "condition" in row:
        condition = row["condition"]
        cart = row.get("cartridge", "none")
        if "+cartridge" in condition:
            return f"icl_{row.get('context_topic', '?')}_plus_cartridge"
        return f"icl_{row.get('context_topic', '?')}_raw"
    if "cartridge" in row:
        cart = row["cartridge"]
        if "QA-task" in cart and "no-cartridge" in cart:
            return "cartridge_p1"
        if "MT-task" in cart:
            return "cartridge_p2"
        return f"cartridge_{Path(cart).name}"
    return "unknown"


def build_comparison_table(
    icl_rows: List[Dict[str, Any]],
    cart_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    table: List[Dict[str, Any]] = []

    for row in icl_rows + cart_rows:
        ds = _dataset_short_name(row.get("dataset", ""))
        method = _classify_row(row)
        acc = row.get("accuracy")
        ppl = row.get("avg_gt_answer_perplexity")

        table.append({
            "method": method,
            "dataset": ds,
            "dataset_path": row.get("dataset"),
            "context_topic": row.get("context_topic"),
            "cartridge": row.get("cartridge"),
            "accuracy": acc,
            "avg_gt_answer_perplexity": ppl,
            "num_answered": row.get("num_answered"),
            "num_correct": row.get("num_correct"),
            "num_total": row.get("num_total"),
        })

    table.sort(key=lambda r: (r["dataset"], r["method"]))
    return table


def print_table(table: List[Dict[str, Any]]) -> None:
    if not table:
        print("(empty comparison table)")
        return

    datasets = sorted({r["dataset"] for r in table})
    methods = sorted({r["method"] for r in table})

    header = f"{'method':<35}" + "".join(f"{ds:>14}" for ds in datasets)
    print(header)
    print("-" * len(header))

    for method in methods:
        line = f"{method:<35}"
        for ds in datasets:
            match = [r for r in table if r["method"] == method and r["dataset"] == ds]
            if match and match[0]["accuracy"] is not None:
                line += f"{match[0]['accuracy']:>14.4f}"
            else:
                line += f"{'N/A':>14}"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge ICL and cartridge eval summaries.")
    parser.add_argument("--icl-summaries", nargs="*", default=[], help="icl_eval_summary.json paths")
    parser.add_argument("--cartridge-summaries", nargs="*", default=[], help="cartridge_eval_summary.json paths")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    args = parser.parse_args()

    icl_rows = _load_summaries(args.icl_summaries)
    cart_rows = _load_summaries(args.cartridge_summaries)
    table = build_comparison_table(icl_rows, cart_rows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2), encoding="utf-8")

    print(f"\nWrote {len(table)} rows to {out}\n")
    print_table(table)


if __name__ == "__main__":
    main()
