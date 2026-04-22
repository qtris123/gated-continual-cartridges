#!/usr/bin/env python3
"""Per-layer V-swap plots.

Two views from the same summary.json:
  --view curve  (default): log-perplexity curve over per-layer swaps.
  --view groups          : bars for group swaps (top3/top5/top10/…), with
                           --metric {raw,R}.

Single panel or MxN grid:

    # Single summary.json -> 1x1 figure
    plot_per_layer_swap.py --summary outputs/<tag>/summary.json --output out.png

    # MxN grid (curve)
    plot_per_layer_swap.py --outputs-root outputs --models qwen3 llama \\
        --tokens 512 1024 2048 \\
        --tag-template "layer_swap_{model}_toks{toks}_pepsi_A_into_B_on_amd" \\
        --output grid.png

    # MxN grid (groups, R metric)
    plot_per_layer_swap.py --outputs-root outputs --models qwen3 \\
        --tokens 512 1024 2048 --view groups --metric R \\
        --tag-template "layer_swap_{model}_toks{toks}_groups" \\
        --output grid_R.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from _lib import (
    bar_panel_layer_swap_groups_R,
    bar_panel_layer_swap_groups_raw,
    curve_panel_layer_swap,
    get_axes,
    load_layer_swap_summary,
)

DEFAULT_DISPLAY = {"qwen3": "Qwen", "llama": "Llama"}
DEFAULT_GROUPS = ["top3", "top5", "top10"]


def _draw_curve(ax, summary_path: Path, eval_name: str, direction: str,
                panel_title: str, baseline_a_label: str, baseline_b_label: str,
                donor_label_override: str | None,
                base_label_override: str | None,
                swap_label_override: str | None):
    curve = load_layer_swap_summary(
        summary_path, eval_name, direction,
        baseline_a=baseline_a_label, baseline_b=baseline_b_label,
    )
    kw = {}
    if donor_label_override is not None:
        kw["donor_label"] = donor_label_override
    if base_label_override is not None:
        kw["base_label"] = base_label_override
    if swap_label_override is not None:
        kw["swap_label"] = swap_label_override
    curve_panel_layer_swap(ax, curve, panel_title, **kw)


def _draw_groups(ax, summary_path: Path, eval_name: str, direction: str,
                 panel_title: str, baseline_a_label: str, baseline_b_label: str,
                 groups: list[str], metric: str,
                 base_label: str, donor_label: str):
    summary = json.loads(Path(summary_path).read_text())
    if metric == "raw":
        bar_panel_layer_swap_groups_raw(
            ax, summary, eval_name, direction, groups, panel_title,
            baseline_a_label, baseline_b_label, base_label, donor_label,
        )
    else:
        bar_panel_layer_swap_groups_R(
            ax, summary, eval_name, direction, groups, panel_title,
            baseline_a_label, baseline_b_label,
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    # --- single-config mode ---
    ap.add_argument("--summary", default=None,
                    help="Single summary.json path. Enables single-config mode.")
    ap.add_argument("--panel-title", default=None,
                    help="Title for single-config mode.")

    # --- grid mode ---
    ap.add_argument("--outputs-root", default="outputs")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--tokens", nargs="+", type=int, default=[512, 1024, 2048])
    ap.add_argument("--tag-template", default=None,
                    help="Must include {model} and {toks}, e.g. "
                         "'layer_swap_{model}_toks{toks}_pepsi_A_into_B_on_amd'.")
    ap.add_argument("--model-display", nargs="+", default=None)
    ap.add_argument("--panel-title-template",
                    default="{model_disp} | toks={toks}")
    ap.add_argument("--suptitle", default=None)

    # --- view selection ---
    ap.add_argument("--view", default="curve", choices=["curve", "groups"],
                    help="'curve' = per-layer log-ppl line. "
                         "'groups' = bars across layer groups (top3/5/10/...).")
    ap.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS,
                    help="Group keys (only used when --view groups).")
    ap.add_argument("--metric", default="raw", choices=["raw", "R"],
                    help="Metric for --view groups: raw token_nll or "
                         "restoration R. Ignored for --view curve.")

    # --- shared ---
    ap.add_argument("--eval-name", default="amd_2021")
    ap.add_argument("--direction", default="A_into_B",
                    choices=["A_into_B", "B_into_A"])
    ap.add_argument("--baseline-a-label", default="baseline_A",
                    help="Key used for cache-A's baseline in summary.json "
                         "(match per_layer_swap.py --cache-a-label).")
    ap.add_argument("--baseline-b-label", default="baseline_B")
    ap.add_argument("--donor-label", default=None,
                    help="Legend label for the donor. Overrides default.")
    ap.add_argument("--base-label", default=None,
                    help="Legend label for the base's baseline. Overrides default.")
    ap.add_argument("--swap-label", default=None,
                    help="Legend label for the per-layer swap line (curve view).")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    # Groups-view label defaults
    if args.direction == "A_into_B":
        default_base, default_donor = args.baseline_b_label, args.baseline_a_label
    else:
        default_base, default_donor = args.baseline_a_label, args.baseline_b_label
    group_base_label = args.base_label or f"{default_base} (base)"
    group_donor_label = args.donor_label or f"{default_donor} (target)"

    def draw(ax, summary_path, panel_title):
        if args.view == "curve":
            _draw_curve(ax, summary_path, args.eval_name, args.direction,
                        panel_title, args.baseline_a_label, args.baseline_b_label,
                        args.donor_label, args.base_label, args.swap_label)
        else:
            _draw_groups(ax, summary_path, args.eval_name, args.direction,
                         panel_title, args.baseline_a_label, args.baseline_b_label,
                         args.groups, args.metric,
                         group_base_label, group_donor_label)

    if args.summary is not None:
        figsize = (6, 4) if args.view == "curve" else (5.4, 4.0)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        title = args.panel_title or (
            f"{args.direction} swap" if args.view == "curve"
            else f"{args.direction} group swap ({args.metric})"
        )
        draw(ax, Path(args.summary), title)
        fig.tight_layout()
    else:
        if not args.models or not args.tag_template:
            ap.error("Grid mode requires --models and --tag-template")
        root = Path(args.outputs_root)
        display = dict(zip(args.models, args.model_display or []))
        nrows = len(args.models)
        ncols = len(args.tokens)
        sharey = (args.view == "groups" and args.metric == "R")
        figw, figh = (5.2, 3.6) if args.view == "curve" else (5.4, 4.0)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(figw * ncols, figh * nrows),
                                 sharey=sharey)
        for r, model in enumerate(args.models):
            disp = display.get(model) or DEFAULT_DISPLAY.get(model, model)
            for c, toks in enumerate(args.tokens):
                ax = get_axes(axes, r, c, nrows, ncols)
                tag = args.tag_template.format(model=model, toks=toks)
                summary = root / tag / "summary.json"
                if not summary.exists():
                    ax.set_title(f"{model} toks={toks} (missing)", fontsize=10)
                    ax.axis("off")
                    continue
                title = args.panel_title_template.format(
                    model_disp=disp, model=model, toks=toks,
                )
                draw(ax, summary, title)
        if args.suptitle:
            fig.suptitle(args.suptitle, fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.93])
        else:
            fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
