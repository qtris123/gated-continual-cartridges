"""wandb training-history JSON → tidy per-step trajectories.csv.

Usage:
  python build_trajectories_csv.py \\
      --histories /path/to/results/wandb_<group>_histories.json \\
      --dataset longhealth \\
      --out /path/to/results/trajectories.csv

The dataset name selects which `eval_<dataset>_perplexity/{loss,perplexity}`
key family to read. MCQ accuracy is read from `generate_<dataset>_accuracy/score`
when present.

Run-name parsing extracts (phase, granularity, top_t) for grouping in plots.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import re
from pathlib import Path


def parse_run_name(name: str):
    """Return (phase, granularity, top_t) for a cartridge sparse training run."""
    if "phase1" in name:
        gran = "per_head" if "per_head" in name or "per-head" in name else "per_layer"
        return "1", gran, ""
    gran = "per_head" if "per_head" in name or "per-head" in name else "per_layer"
    m = re.search(r"top-?(\d+)", name)
    return "2", gran, (m.group(1) if m else "")


def maybe_float(x):
    if x in (None, "None"):
        return ""
    try:
        v = float(x)
        if math.isnan(v):
            return ""
        return v
    except (TypeError, ValueError):
        return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--histories", required=True,
                   help="wandb_<group>_histories.json from fetch_wandb_data.py")
    p.add_argument("--dataset", required=True, help="e.g. longhealth, qasper")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    hist = json.loads(Path(args.histories).read_text())
    eval_loss_key = f"eval_{args.dataset}_perplexity/loss"
    eval_ppl_key  = f"eval_{args.dataset}_perplexity/perplexity"
    mcq_key       = f"generate_{args.dataset}_accuracy/score"

    out_rows = []
    for run, rows in hist.items():
        if not isinstance(rows, list):
            continue  # ERROR string
        phase, gran, top_t = parse_run_name(run)
        for r in rows:
            step = r.get("train/optimizer_step")
            if step is None:
                continue
            try:
                step = int(step)
            except (TypeError, ValueError):
                continue
            out_rows.append({
                "run": run,
                "phase": phase,
                "granularity": gran,
                "top_t": top_t,
                "step": step,
                "train_loss": maybe_float(r.get("train/loss")),
                "train_ppl":  maybe_float(r.get("train/perplexity")),
                "eval_loss":  maybe_float(r.get(eval_loss_key)),
                "eval_ppl":   maybe_float(r.get(eval_ppl_key)),
                "mcq_acc":    maybe_float(r.get(mcq_key)),
                "lr":         maybe_float(r.get("optimizer/lr_group0")),
            })
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else
                           ["run","phase","granularity","top_t","step","train_loss","train_ppl","eval_loss","eval_ppl","mcq_acc","lr"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {out_path}  ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
