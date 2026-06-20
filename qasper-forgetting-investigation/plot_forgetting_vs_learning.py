"""Plot the forgetting/learning trade-off vs. sparsity for the qasper sparse
cartridges.

This script is an ad-hoc investigation that lives inside the
gated-continual-cartridges repo (under `qasper-forgetting-investigation/`).
It only reads eval logs from the repo's `outputs/` directory and writes
plots into ./plots/ alongside this file.

For each registered sweep, two figures are produced -- one for per-layer and
one for per-head -- showing how cartridge perplexity on Phase 1 QA data
(forgetting) and Phase 2 MT data (learning) changes as the per-step active
slot budget grows from 64 -> 128 -> 256 -> 512. Two horizontal baseline
lines are added from the Phase-1-only (no Phase 2 finetuning) eval; these
are identical across per-layer and per-head because Phase 1 doesn't depend
on the Phase 2 granularity.

Numbers are scraped from the wandb summary lines in each run's eval.log.

Run:
    python plot_forgetting_vs_learning.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
CARTRIDGES_REPO = HERE.parent
BASELINE_DIR = CARTRIDGES_REPO / "outputs" / "qasper_forgetting_eval_782782"
OUT_DIR = HERE / "plots"

SPARSITY_LEVELS = [64, 128, 256, 512]
GRANULARITIES = ["per-layer", "per-head"]

_PPL_RE = re.compile(r"eval_qasper_perplexity/perplexity\s+([0-9.]+)")


@dataclass(frozen=True)
class Sweep:
    """One forgetting-eval sweep over (granularity, top-k).

    `run_dir_template` is formatted with ``gran`` and ``k`` and must produce
    the base run-name; ``__qa_eval`` / ``__mt_eval`` are appended.
    """

    slug: str              # short id, used in output filenames and titles
    label: str             # human-readable name for plot titles
    eval_dir: Path
    run_dir_template: str


SWEEPS: list[Sweep] = [
    Sweep(
        slug="value_only_bsize32",
        label="value-only (bsize 32)",
        eval_dir=CARTRIDGES_REPO
        / "outputs"
        / "qasper_forgetting_eval_value-only_bsize32",
        run_dir_template="qasper-value-only-{gran}-top-{k}-bsize32",
    ),
    Sweep(
        slug="key_value_bsize32",
        label="key-value (bsize 32)",
        eval_dir=CARTRIDGES_REPO
        / "outputs"
        / "qasper_forgetting_eval_key-value_bsize32_987953",
        run_dir_template="qasper-key-value-{gran}-top-{k}",
    ),
]


def read_perplexity(run_dir: Path) -> float:
    log = run_dir / "eval.log"
    text = log.read_text()
    matches = _PPL_RE.findall(text)
    if not matches:
        raise RuntimeError(f"no perplexity in {log}")
    # last match is the final wandb summary line
    return float(matches[-1])


def collect(sweep: Sweep) -> dict[str, dict[str, list[float] | float]]:
    # qa_eval == perplexity on Phase 1 (QA) data -> forgetting
    # mt_eval == perplexity on Phase 2 (MT) data -> learning
    out: dict[str, dict[str, list[float] | float]] = {}
    for gran in GRANULARITIES:
        forget, learn = [], []
        for k in SPARSITY_LEVELS:
            base = sweep.run_dir_template.format(gran=gran, k=k)
            forget.append(read_perplexity(sweep.eval_dir / f"{base}__qa_eval"))
            learn.append(read_perplexity(sweep.eval_dir / f"{base}__mt_eval"))
        baseline_forget = read_perplexity(BASELINE_DIR / f"baseline_{gran}__qa_eval")
        baseline_learn = read_perplexity(BASELINE_DIR / f"baseline_{gran}__mt_eval")
        out[gran] = {
            "forgetting": forget,
            "learning": learn,
            "baseline_forgetting": baseline_forget,
            "baseline_learning": baseline_learn,
        }
    return out


def plot_one(
    sweep: Sweep,
    gran: str,
    data: dict[str, list[float] | float],
    path: Path,
) -> None:
    # Keep the axes the same size as the original (no-legend) plot;
    # the external legend is added to the saved figure via bbox_inches="tight".
    fig, ax = plt.subplots(figsize=(6.8, 4.6))

    ax.plot(
        SPARSITY_LEVELS,
        data["forgetting"],
        marker="o",
        color="#d62728",
        label="Forgetting (Phase 1 QA perplexity)",
    )
    ax.plot(
        SPARSITY_LEVELS,
        data["learning"],
        marker="s",
        color="#1f77b4",
        label="Learning (Phase 2 MT perplexity)",
    )

    # Baselines: Phase-1-only cartridge (no Phase 2 sparse finetuning).
    baseline_forget = float(data["baseline_forgetting"])
    baseline_learn = float(data["baseline_learning"])
    ax.axhline(
        baseline_forget,
        linestyle="--",
        color="#d62728",
        alpha=0.6,
        linewidth=1.3,
        label=f"Baseline QA (no Phase 2) = {baseline_forget:.2f}",
    )
    ax.axhline(
        baseline_learn,
        linestyle="--",
        color="#1f77b4",
        alpha=0.6,
        linewidth=1.3,
        label=f"Baseline MT (no Phase 2) = {baseline_learn:.2f}",
    )

    for k, yf, yl in zip(SPARSITY_LEVELS, data["forgetting"], data["learning"]):
        ax.annotate(
            f"{yf:.2f}", (k, yf), textcoords="offset points",
            xytext=(0, 6), ha="center", fontsize=8, color="#d62728",
        )
        ax.annotate(
            f"{yl:.2f}", (k, yl), textcoords="offset points",
            xytext=(0, -12), ha="center", fontsize=8, color="#1f77b4",
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(SPARSITY_LEVELS)
    ax.set_xticklabels([str(s) for s in SPARSITY_LEVELS])
    ax.set_xlabel("Active slot budget (top-k)")
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title(
        f"QASPER {sweep.label}, {gran} sparse cartridge\n"
        f"Forgetting vs. learning as sparsity grows"
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    # Legend below the plot, in two columns (4 entries -> 2x2 grid).
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        borderaxespad=0.0,
        frameon=False,
        fontsize=9,
        ncol=2,
        handlelength=2.4,
        columnspacing=2.0,
    )

    # Make sure both baselines + all data points are visible.
    ymin = min(
        min(data["forgetting"]),
        min(data["learning"]),
        baseline_forget,
    ) - 0.5
    ymax = max(
        max(data["forgetting"]),
        max(data["learning"]),
        baseline_learn,
    ) + 2.0
    ax.set_ylim(ymin, ymax)

    # Layout uses the original figsize so the plot area matches the
    # no-legend version. The external legend is added back via
    # bbox_inches="tight" + bbox_extra_artists when saving, which
    # extends the saved image rather than shrinking the axes.
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
        bbox_extra_artists=(legend,),
    )
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_lines = ["sweep,granularity,top_k,forgetting_ppl,learning_ppl"]

    for sweep in SWEEPS:
        data = collect(sweep)
        for gran in GRANULARITIES:
            fname = (
                f"forgetting_vs_learning_"
                f"{sweep.slug}_{gran.replace('-', '_')}.png"
            )
            plot_one(sweep, gran, data[gran], OUT_DIR / fname)

            for k, f, l in zip(
                SPARSITY_LEVELS,
                data[gran]["forgetting"],
                data[gran]["learning"],
            ):
                summary_lines.append(f"{sweep.slug},{gran},{k},{f:.5f},{l:.5f}")
            summary_lines.append(
                f"{sweep.slug},{gran},baseline,"
                f"{float(data[gran]['baseline_forgetting']):.5f},"
                f"{float(data[gran]['baseline_learning']):.5f}"
            )

    summary = OUT_DIR / "forgetting_vs_learning_summary.txt"
    summary.write_text("\n".join(summary_lines) + "\n")
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
