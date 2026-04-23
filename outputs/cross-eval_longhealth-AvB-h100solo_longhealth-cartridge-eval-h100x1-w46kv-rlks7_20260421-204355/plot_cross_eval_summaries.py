"""Load cartridge_eval_summary.json per experiment and plot metrics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SUMMARY_NAME = "cartridge_eval_summary.json"

PHASE1_MARK = "longhealth-p1-10_819"
PHASE2_MARK = "longhealth-p11-20_8192_10-epochs_with-cartridge_p1-10_8192"


def phase_from_cartridge(cartridge: str) -> str:
    if PHASE2_MARK in cartridge:
        return "phase 2"
    if PHASE1_MARK in cartridge:
        return "phase 1"
    return "other"


def dataset_cohort_tag(dataset_path: str) -> str:
    stem = Path(dataset_path).stem
    if "patient11_20" in stem:
        return "p11–20 qs"
    if "patient1_10" in stem:
        return "p1–10 qs"
    return stem.replace("longhealth_", "")


def bar_label(cartridge: str, dataset_path: str) -> str:
    return f"{phase_from_cartridge(cartridge)}\n({dataset_cohort_tag(dataset_path)})"


def main() -> None:
    exp_dirs = sorted(p for p in ROOT.iterdir() if p.is_dir() and (p / SUMMARY_NAME).is_file())
    if not exp_dirs:
        raise SystemExit(f"No subfolders with {SUMMARY_NAME} under {ROOT}")

    rows = []
    for exp_dir in exp_dirs:
        data = json.loads((exp_dir / SUMMARY_NAME).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            data = [data]

        labels = [bar_label(r["cartridge"], r["dataset"]) for r in data]
        x = list(range(len(data)))

        for i, r in enumerate(data):
            rows.append(
                {
                    "experiment": exp_dir.name,
                    "eval_index": i,
                    "cartridge": r["cartridge"],
                    "num_total": r["num_total"],
                    "num_answered": r["num_answered"],
                    "num_unanswered": r["num_unanswered"],
                    "num_correct": r["num_correct"],
                    "num_gt_missing_from_topk": r["num_gt_missing_from_topk"],
                    "accuracy": r["accuracy"],
                }
            )

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        fig.suptitle(exp_dir.name)

        ax_acc, ax_un, ax_miss, ax_ans = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

        ax_acc.bar(x, [r["accuracy"] for r in data], color="steelblue")
        ax_acc.set_title("accuracy")
        ax_acc.set_ylim(0, 1.05)
        ax_acc.set_xticks(x)
        ax_acc.set_xticklabels(labels, rotation=35, ha="right")

        ax_un.bar(x, [r["num_unanswered"] for r in data], color="coral")
        ax_un.set_title("num_unanswered")
        ax_un.set_xticks(x)
        ax_un.set_xticklabels(labels, rotation=35, ha="right")

        ax_miss.bar(x, [r["num_gt_missing_from_topk"] for r in data], color="seagreen")
        ax_miss.set_title("num_gt_missing_from_topk")
        ax_miss.set_xticks(x)
        ax_miss.set_xticklabels(labels, rotation=35, ha="right")

        ax_ans.bar(x, [r["num_answered"] for r in data], color="mediumpurple")
        ax_ans.set_title("num_answered")
        ax_ans.set_xticks(x)
        ax_ans.set_xticklabels(labels, rotation=35, ha="right")

        fig.tight_layout()
        out = exp_dir / "summary_metrics_plots.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")

        # Stacked outcomes: total bar height = num_total; correct / wrong / unanswered
        correct = [int(r["num_correct"]) for r in data]
        wrong = [int(r["num_answered"] - r["num_correct"]) for r in data]
        unans = [int(r["num_unanswered"]) for r in data]
        totals = [int(r["num_total"]) for r in data]
        accs = [float(r["accuracy"]) for r in data]

        fig2, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x, correct, label="correct", color="#2ca02c")
        ax.bar(x, wrong, bottom=correct, label="inaccurate (answered)", color="#d62728")
        bottom_wrong = [c + w for c, w in zip(correct, wrong)]
        ax.bar(x, unans, bottom=bottom_wrong, label="unanswered", color="#7f7f7f")

        ymax = max(totals) if totals else 1
        ax.set_ylim(0, ymax * 1.12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("count (stacked = # questions)")
        ax.legend(loc="upper right")
        ax.set_title(f"{exp_dir.name}: outcomes (bar height = num_total)")

        for xi, t, a in zip(x, totals, accs):
            ax.text(xi, t + ymax * 0.02, f"{a:.1%}", ha="center", va="bottom", fontsize=10)

        out2 = exp_dir / "summary_outcome_stacked.png"
        fig2.tight_layout()
        fig2.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Wrote {out2}")

    print("\n--- collected metrics ---")
    for r in rows:
        print(
            f"{r['experiment']:<22}  [{r['eval_index']}]  "
            f"total={r['num_total']:<4}  correct={r['num_correct']:<4}  "
            f"answered={r['num_answered']:<4}  unanswered={r['num_unanswered']:<4}  "
            f"gt_missing_topk={r['num_gt_missing_from_topk']:<4}  acc={r['accuracy']:.4f}  "
            f"{r['cartridge']}"
        )


if __name__ == "__main__":
    main()
