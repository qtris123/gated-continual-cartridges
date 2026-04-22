#!/usr/bin/env python3
"""Per-slot V-swap R(p) bar plot (single config or MxN grid).

    # Single .pt
    plot_per_slot_swap.py --pt outputs/<tag>/per_slot_swap.pt --output out.png

    # MxN grid
    plot_per_slot_swap.py --outputs-root outputs --models qwen3 llama \\
        --tokens 512 1024 2048 \\
        --tag-template "per_slot_swap_{model}_toks{toks}" \\
        --output grid.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from _lib import (
    bar_panel_per_slot_R,
    get_axes,
    line_panel_per_slot_nll,
    load_per_slot_pt,
)

DEFAULT_DISPLAY = {"qwen3": "Qwen", "llama": "Llama"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    # --- single-config mode ---
    ap.add_argument("--pt", default=None,
                    help="Single per_slot_swap.pt path. Enables single mode.")
    ap.add_argument("--title", default="Per-slot V-swap A→B on AMD-2021")

    # --- grid mode ---
    ap.add_argument("--outputs-root", default="outputs")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--tokens", nargs="+", type=int, default=[512, 1024, 2048])
    ap.add_argument("--tag-template", default="per_slot_swap_{model}_toks{toks}")
    ap.add_argument("--pt-name", default="per_slot_swap.pt",
                    help="Filename inside each tagged dir "
                         "(historical dirs may use per_token_slot_swap.pt).")
    ap.add_argument("--model-display", nargs="+", default=None)
    ap.add_argument("--suptitle",
                    default=("Per-slot V-swap A→B on AMD-2021  "
                             r"$R(p) = (\mathrm{log\_ppl}_B - \mathrm{log\_ppl}_{\mathrm{swap}(p)})"
                             r" / (\mathrm{log\_ppl}_B - \mathrm{log\_ppl}_A)$"))

    # --- shared ---
    ap.add_argument("--metric", choices=["R", "nll"], default="R",
                    help="R = restoration score per slot (bars); "
                         "nll = raw log-perplexity per slot (line).")
    ap.add_argument("--top-k-label", type=int, default=5,
                    help="Annotate this many extreme slots per panel "
                         "(highest R for --metric R, lowest swap nll for --metric nll).")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.pt is not None:
        slot = load_per_slot_pt(Path(args.pt))
        gap = slot["base_nll"] - slot["donor_nll"]
        fig, ax = plt.subplots(1, 1, figsize=(13, 4.5))
        if args.metric == "R":
            panel_title = (f"{args.title}\n"
                           f"log_ppl_B={slot['base_nll']:.3f}  "
                           f"log_ppl_A={slot['donor_nll']:.3f}  "
                           f"gap={gap:.3f}  max R={slot['R'].max():+.3f}")
            bar_panel_per_slot_R(ax, slot, panel_title,
                                 top_k_label=args.top_k_label,
                                 show_full_restoration_line=True,
                                 annotate_prefix="p=")
        else:
            import torch
            lp = slot["log_ppl_per_slot"]
            lp_min = float(lp.min() if isinstance(lp, torch.Tensor) else min(lp))
            panel_title = (f"{args.title}\n"
                           f"untrained={slot['base_nll']:.3f}  "
                           f"trained={slot['donor_nll']:.3f}  "
                           f"min swap={lp_min:.3f}")
            line_panel_per_slot_nll(ax, slot, panel_title,
                                    top_k_label=args.top_k_label,
                                    annotate_prefix="p=")
        ax.legend()
        fig.tight_layout()
    else:
        if not args.models:
            ap.error("Grid mode requires --models")
        root = Path(args.outputs_root)
        display = dict(zip(args.models, args.model_display or []))
        nrows = len(args.models)
        ncols = len(args.tokens)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.6 * nrows))
        for r, model in enumerate(args.models):
            disp = display.get(model) or DEFAULT_DISPLAY.get(model, model)
            for c, toks in enumerate(args.tokens):
                ax = get_axes(axes, r, c, nrows, ncols)
                pt = root / args.tag_template.format(model=model, toks=toks) / args.pt_name
                if not pt.exists():
                    ax.set_title(f"{model} toks={toks} (missing)")
                    ax.axis("off")
                    continue
                slot = load_per_slot_pt(pt)
                if args.metric == "R":
                    gap = slot["base_nll"] - slot["donor_nll"]
                    title = (f"{disp} | toks={toks}  "
                             f"(gap={gap:.2f}, max R={slot['R'].max():+.3f})")
                    bar_panel_per_slot_R(ax, slot, title,
                                         top_k_label=args.top_k_label,
                                         show_full_restoration_line=False,
                                         annotate_prefix="")
                else:
                    import torch
                    lp = slot["log_ppl_per_slot"]
                    lp_min = float(lp.min() if isinstance(lp, torch.Tensor) else min(lp))
                    title = (f"{disp} | toks={toks}  "
                             f"untrained={slot['base_nll']:.2f}  "
                             f"trained={slot['donor_nll']:.2f}  "
                             f"min swap={lp_min:.2f}")
                    line_panel_per_slot_nll(ax, slot, title,
                                            top_k_label=args.top_k_label,
                                            annotate_prefix="p=")
                    if r == 0 and c == ncols - 1:
                        ax.legend(loc="upper right", fontsize=7)
                if r < nrows - 1:
                    ax.set_xlabel("")
                if c > 0:
                    ax.set_ylabel("")
        suptitle = args.suptitle
        if args.metric == "nll" and suptitle == ap.get_default("suptitle"):
            suptitle = ("Per-token-slot V-swap trained \u2192 untrained on AMD-2021 "
                        "(raw log-perplexity per swapped slot)")
        fig.suptitle(suptitle, fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
