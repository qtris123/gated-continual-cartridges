"""Side-by-side forgetting/learning comparison for two datasets that share a
recipe (e.g. QASPER reference vs LongHealth target).

Inputs are two `wandb_<group>_summaries.json` files for *eval* groups
(typically named `<dataset> - [granularity x forgetting/learning]` or the
`ablation - [granularity x forgetting/learning]` variant).

Usage:
  python compare_recipes.py \\
    --reference results/wandb_ablation_granularity_x_forgetting_learning_summaries.json \\
    --reference-dataset qasper \\
    --reference-p1-split qa_eval --reference-p2-split mt_eval \\
    --reference-variant key-value \\
    --target results/wandb_longhealth_granularity_x_forgetting_learning_summaries.json \\
    --target-dataset longhealth \\
    --target-p1-split p1-10 --target-p2-split p11-20 \\
    --md-out notes/04_qasper_vs_longhealth.md \\
    --png-out results/qasper_vs_longhealth.png

The two `<dataset>` names select which `eval_<dataset>_perplexity/perplexity`
metric key to read.

Run-name parsing handles both `__` and `_` separators between label and split.
"""
from __future__ import annotations
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def parse_eval_runs(summaries: dict, dataset: str, splits: tuple[str, str], variant: str | None):
    """Return list of {label, eval_set, ppl} dicts for a dataset's eval group."""
    p1, p2 = splits
    metric_key = f"eval_{dataset}_perplexity/perplexity"
    out = []
    pat = re.compile(
        rf"(?P<label>.+?)_+(?P<eset>({re.escape(p1)}|{re.escape(p2)}))(?:_(?P<variant>[a-z\-]+))?(?:_eval)?$"
    )
    for name, s in summaries.items():
        ppl = s.get(metric_key)
        if ppl in (None, "None"):
            continue
        m = pat.match(name)
        if not m:
            continue
        if variant and m.group("variant") and m.group("variant") != variant:
            continue
        label = m.group("label").rstrip("_")
        out.append({
            "run": name,
            "label": label,
            "eval_set": m.group("eset"),
            "ppl": float(ppl),
        })
    return out


def pivot(rows, key="label", set_key="eval_set", val_key="ppl"):
    out = defaultdict(dict)
    for r in rows:
        out[r[key]][r[set_key]] = r[val_key]
    return out


def render_md(args, ref_pivot, tgt_pivot, ref_p1, ref_p2, tgt_p1, tgt_p2):
    base_ref = ref_pivot.get("baseline", {})
    base_tgt = tgt_pivot.get("baseline", {})
    base_ref_p1 = base_ref.get(ref_p1); base_ref_p2 = base_ref.get(ref_p2)
    base_tgt_p1 = base_tgt.get(tgt_p1); base_tgt_p2 = base_tgt.get(tgt_p2)

    ref_best = min((k for k in ref_pivot if k != "baseline"
                    and ref_p2 in ref_pivot[k]),
                   key=lambda k: ref_pivot[k][ref_p2])
    tgt_best = min((k for k in tgt_pivot if k != "baseline"
                    and tgt_p2 in tgt_pivot[k]),
                   key=lambda k: tgt_pivot[k][tgt_p2])

    md = []
    md.append(f"# {args.reference_dataset} vs {args.target_dataset} — forgetting & learning")
    md.append("")
    md.append("Same training recipe; different datasets.")
    md.append("")
    md.append("## Side-by-side scale comparison")
    md.append("")
    md.append(f"| Quantity | {args.reference_dataset} | {args.target_dataset} |")
    md.append("|---|---|---|")
    md.append(f"| baseline P1-task ppl ({ref_p1} / {tgt_p1}) | {base_ref_p1:.3f} | {base_tgt_p1:.3f} |")
    md.append(f"| baseline P2-task ppl ({ref_p2} / {tgt_p2}) | {base_ref_p2:.3f} | {base_tgt_p2:.3f} |")
    md.append(f"| **baseline gap (P2−P1)** | **{(base_ref_p2-base_ref_p1):+.2f}** | **{(base_tgt_p2-base_tgt_p1):+.3f}** |")
    rb = ref_pivot[ref_best]; tb = tgt_pivot[tgt_best]
    md.append(f"| best run (by P2-task ppl) | {ref_best} → P2={rb[ref_p2]:.2f}, P1={rb[ref_p1]:.2f} | {tgt_best} → P2={tb[tgt_p2]:.3f}, P1={tb[tgt_p1]:.3f} |")
    md.append(f"| **learning** (Δ P2 vs baseline) | **{rb[ref_p2]-base_ref_p2:+.2f}** | **{tb[tgt_p2]-base_tgt_p2:+.3f}** |")
    md.append(f"| **forgetting** (Δ P1 vs baseline) | **{rb[ref_p1]-base_ref_p1:+.2f}** | **{tb[tgt_p1]-base_tgt_p1:+.3f}** |")
    md.append("")

    for tag, pivot_, p1, p2 in [
        (args.reference_dataset, ref_pivot, ref_p1, ref_p2),
        (args.target_dataset,    tgt_pivot, tgt_p1, tgt_p2),
    ]:
        base = pivot_.get("baseline", {})
        bp1 = base.get(p1); bp2 = base.get(p2)
        md.append(f"## {tag} — full sweep")
        md.append("")
        md.append(f"| Run | {p1} ppl | {p2} ppl | Δ{p1} vs baseline | Δ{p2} vs baseline |")
        md.append("|---|---|---|---|---|")
        for k in sorted(pivot_.keys()):
            d = pivot_[k]
            v1 = d.get(p1); v2 = d.get(p2)
            if v1 is None or v2 is None: continue
            d1 = (v1 - bp1) if bp1 is not None else None
            d2 = (v2 - bp2) if bp2 is not None else None
            md.append(f"| {k} | {v1:.3f} | {v2:.3f} | {d1:+.3f} | {d2:+.3f} |")
        md.append("")
    Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md_out).write_text("\n".join(md))
    print(f"wrote {args.md_out}")


def render_png(args, ref_pivot, tgt_pivot, ref_p1, ref_p2, tgt_p1, tgt_p2):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    def labels_in_order(pivot_):
        # Put baseline first, then sort the rest by canonical sweep order.
        keys = list(pivot_.keys())
        keys.sort(key=lambda k: (
            0 if k == "baseline" else 1,
            "head" in k,                                    # layer before head
            int(re.search(r"(\d+)", k).group(1) if re.search(r"(\d+)", k) else 0),
        ))
        return keys

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (tag, pivot_, p1, p2) in zip(axes, [
        (args.reference_dataset, ref_pivot, ref_p1, ref_p2),
        (args.target_dataset,    tgt_pivot, tgt_p1, tgt_p2),
    ]):
        keys = [k for k in labels_in_order(pivot_) if p1 in pivot_[k] and p2 in pivot_[k]]
        x = list(range(len(keys)))
        w = 0.4
        ax.bar([xi - w/2 for xi in x], [pivot_[k][p1] for k in keys], width=w,
               label=f"{p1} (P1 task = forgetting)", color="#1f77b4")
        ax.bar([xi + w/2 for xi in x], [pivot_[k][p2] for k in keys], width=w,
               label=f"{p2} (P2 task = learning)", color="#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("perplexity")
        ax.set_title(tag)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle(f"{args.reference_dataset} (left) vs {args.target_dataset} (right) — same recipe",
                 fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    Path(args.png_out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.png_out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.png_out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--reference-dataset", required=True)
    p.add_argument("--reference-p1-split", required=True)
    p.add_argument("--reference-p2-split", required=True)
    p.add_argument("--reference-variant", default=None)
    p.add_argument("--target", required=True)
    p.add_argument("--target-dataset", required=True)
    p.add_argument("--target-p1-split", required=True)
    p.add_argument("--target-p2-split", required=True)
    p.add_argument("--target-variant", default=None)
    p.add_argument("--md-out", required=True)
    p.add_argument("--png-out", required=True)
    args = p.parse_args()

    ref = json.loads(Path(args.reference).read_text())
    tgt = json.loads(Path(args.target).read_text())

    ref_rows = parse_eval_runs(
        ref, args.reference_dataset,
        (args.reference_p1_split, args.reference_p2_split), args.reference_variant)
    tgt_rows = parse_eval_runs(
        tgt, args.target_dataset,
        (args.target_p1_split, args.target_p2_split), args.target_variant)

    if not ref_rows or not tgt_rows:
        raise SystemExit(
            f"empty pivot: ref={len(ref_rows)}, tgt={len(tgt_rows)} — "
            "check splits/variants and run-name regex"
        )
    ref_pivot = pivot(ref_rows)
    tgt_pivot = pivot(tgt_rows)

    render_md(args, ref_pivot, tgt_pivot,
              args.reference_p1_split, args.reference_p2_split,
              args.target_p1_split, args.target_p2_split)
    render_png(args, ref_pivot, tgt_pivot,
               args.reference_p1_split, args.reference_p2_split,
               args.target_p1_split, args.target_p2_split)


if __name__ == "__main__":
    main()
