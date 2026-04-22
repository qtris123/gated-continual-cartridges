#!/usr/bin/env python3
"""Trained-vs-untrained baselines plot (openended log-ppl, contains, yes/no, mcq).

Single model or grid over models.

    # 1xN (one model)
    plot_baselines.py --models qwen3 --tokens 512 1024 2048 --output out.png

    # MxN (multiple models stacked)
    plot_baselines.py --models qwen3 llama --tokens 512 1024 2048 --output out.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from _lib import (
    bar_panel_trained_vs_untrained,
    get_axes,
    iter_configs,
    load_baseline_json,
)

DEFAULT_DISPLAY = {"qwen3": "Qwen3-4B", "llama": "Llama-3.2-3B"}


def _collect_rows(root: Path, model_short: str, tokens: list[int]) -> list[dict]:
    rows = []
    for t in tokens:
        d = root / f"amd_2021_baselines_{model_short}_toks{t}"
        if not d.exists():
            continue
        oe_t = load_baseline_json(d, "openended", "trained")
        oe_u = load_baseline_json(d, "openended", "untrained")
        yn_t = load_baseline_json(d, "yesno", "trained")
        yn_u = load_baseline_json(d, "yesno", "untrained")
        mc_t = load_baseline_json(d, "mcq", "trained")
        mc_u = load_baseline_json(d, "mcq", "untrained")
        rows.append({
            "toks": t,
            "nll_t": oe_t["token_nll"], "nll_u": oe_u["token_nll"],
            "cm_t": oe_t.get("contains_match"), "cm_u": oe_u.get("contains_match"),
            "yn_t": yn_t["accuracy"], "yn_u": yn_u["accuracy"],
            "mc_t": mc_t["accuracy"], "mc_u": mc_u["accuracy"],
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outputs-root", default="outputs")
    ap.add_argument("--models", nargs="+", required=True,
                    help="Model shortnames (e.g. qwen3 llama). One = 1xN; "
                         "multiple = MxN grid.")
    ap.add_argument("--tokens", nargs="+", type=int, default=[512, 1024, 2048])
    ap.add_argument("--model-display", nargs="+", default=None,
                    help="Display names matching --models (default: use table).")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    display = dict(zip(args.models, args.model_display or []))
    root = Path(args.outputs_root)
    nrows = len(args.models)
    ncols = 4  # log-ppl / contains / yes/no / mcq

    fig, axes = plt.subplots(nrows, ncols, figsize=(18.5, 4.2 * nrows))

    printed_any = False
    for r, model in enumerate(args.models):
        disp = display.get(model) or DEFAULT_DISPLAY.get(model, model)
        rows = _collect_rows(root, model, args.tokens)
        if not rows:
            for c in range(ncols):
                ax = get_axes(axes, r, c, nrows, ncols)
                ax.set_title(f"{disp} (missing)")
                ax.axis("off")
            continue

        panels = [
            ("nll_t", "nll_u",
             f"{disp}  openended log-perplexity\n(lower = better)", False, None),
            ("cm_t", "cm_u",
             f"{disp}  openended free-form acc\n(contains_match, higher = better)",
             True, None),
            ("yn_t", "yn_u",
             f"{disp}  yes/no 1-token accuracy\n(higher = better)", True, 0.5),
            ("mc_t", "mc_u",
             f"{disp}  mcq 1-token accuracy\n(higher = better)", True, 0.25),
        ]
        for c, (kt, ku, title, is_acc, chance) in enumerate(panels):
            ax = get_axes(axes, r, c, nrows, ncols)
            bar_panel_trained_vs_untrained(ax, rows, kt, ku, title, is_acc,
                                           chance_line=chance)

        # Print the markdown row (per model) once per run.
        if not printed_any:
            print("\n| model | toks | log_ppl_t | log_ppl_u | contains_t | "
                  "contains_u | yn_t | yn_u | mcq_t | mcq_u |")
            print("|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            printed_any = True
        for rr in rows:
            print(f"| {disp} | {rr['toks']} | {rr['nll_t']:.3f} | "
                  f"{rr['nll_u']:.3f} | {rr['cm_t']*100:.2f}% | "
                  f"{rr['cm_u']*100:.2f}% | {rr['yn_t']*100:.2f}% | "
                  f"{rr['yn_u']*100:.2f}% | {rr['mc_t']*100:.2f}% | "
                  f"{rr['mc_u']*100:.2f}% |")

    if nrows == 1:
        suptitle = (f"{display.get(args.models[0]) or DEFAULT_DISPLAY.get(args.models[0], args.models[0])}"
                    "  AMD-2021 baselines  (trained vs step-0 context-init)")
    else:
        suptitle = "AMD-2021 baselines: trained cartridge vs step-0 context-init"
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96 if nrows > 1 else 0.94])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
