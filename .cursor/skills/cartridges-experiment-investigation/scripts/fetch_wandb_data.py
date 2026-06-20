"""Pull histories + summaries for a list of wandb groups.

Usage:
  python fetch_wandb_data.py \\
    --entity vqtri-purdue-university \\
    --project SEACrowd \\
    --groups "longhealth - granularity tf-idf" "longhealth - [granularity x forgetting/learning]" \\
    --out-dir /path/to/results/

Outputs (per group):
  wandb_<slug>_summaries.json   final scalar summary metrics for each run
  wandb_<slug>_histories.json   per-step history (only when keys match)

Run from any CWD; only requires WANDB_API_KEY in env (or `wandb login`).

The history-key set is dataset-agnostic — we ask for both LongHealth and
QASPER metric keys, plus the universal training keys, and rely on wandb to
silently drop keys not present in a given run.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import wandb

DEFAULT_HISTORY_KEYS = [
    "_step",
    "train/optimizer_step",
    "train/loss",
    "train/perplexity",
    "optimizer/lr_group0",
    # eval metric families across datasets
    "eval_longhealth_perplexity/loss",
    "eval_longhealth_perplexity/perplexity",
    "eval_qasper_perplexity/loss",
    "eval_qasper_perplexity/perplexity",
    # MCQ / generation accuracy (LongHealth)
    "generate_longhealth_accuracy/score",
    # cache shape sanity
    "num_trainable_tokens",
]


def slugify(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\[\]]", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")[:60]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", default="vqtri-purdue-university")
    p.add_argument("--project", default="SEACrowd")
    p.add_argument("--groups", nargs="+", required=True,
                   help="wandb group names; quote spaces and brackets")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--history-keys", nargs="*", default=DEFAULT_HISTORY_KEYS)
    p.add_argument("--samples", type=int, default=2000,
                   help="max samples per run history")
    p.add_argument("--no-history", action="store_true",
                   help="skip per-step history (eval-only groups have 1 row anyway)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()

    for grp in args.groups:
        slug = slugify(grp)
        print(f"=== {grp!r}  →  slug={slug}")
        runs = list(api.runs(f"{args.entity}/{args.project}", filters={"group": grp}))
        print(f"   n_runs = {len(runs)}")
        summaries, histories = {}, {}
        for r in runs:
            s = dict(r.summary)
            summaries[r.name] = {k: v for k, v in s.items()
                                 if isinstance(v, (int, float, str)) or v is None}
            if args.no_history:
                continue
            try:
                h = r.history(keys=args.history_keys, samples=args.samples)
                histories[r.name] = h.to_dict(orient="records") if len(h) else []
            except Exception as e:
                histories[r.name] = f"ERROR: {e}"
        (out_dir / f"wandb_{slug}_summaries.json").write_text(
            json.dumps(summaries, default=str, indent=1))
        if histories:
            (out_dir / f"wandb_{slug}_histories.json").write_text(
                json.dumps(histories, default=str, indent=1))
        print(f"   wrote summaries ({len(summaries)} runs) "
              f"+ histories ({len(histories)} runs) to {out_dir}/")


if __name__ == "__main__":
    main()
