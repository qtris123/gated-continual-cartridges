"""Plot per-model forgetting/acquisition vs. sparsity for global / per-head / per-layer.

Mirrors the reference figure style (one figure, two subplots, three granularity
lines per subplot, single shared baseline-cartridge dashed line per subplot).

Reads eval logs from each model's
``outputs/qasper_forgetting_eval_<model>_granularity-x-sparsity_*`` directory
and writes a figure to ``./plots/<model>_granularity_x_sparsity.png`` next to
this file. One model = one figure.

Numbers are scraped from the wandb summary lines in each run's eval.log.

Run:
    python plot_granularity_x_sparsity.py
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
CARTRIDGES_REPO = HERE.parent
OUT_DIR = HERE / "plots"

# Active slot budget grid for these sweeps (note: 32..256, not 64..512).
SPARSITY_LEVELS = [32, 64, 128, 256]

# Order matters: this is the legend order.
GRANULARITIES = ["per-head", "per-layer", "global"]

# Per-granularity styling for the line plots.
GRAN_STYLE = {
    "per-head":  dict(color="#1f77b4", marker="o", label="per-head"),
    "per-layer": dict(color="#d6604d", marker="s", label="per-layer"),
    "global":    dict(color="#2ca02c", marker="^", label="global"),
}

_PPL_RE = re.compile(r"eval_qasper_perplexity/perplexity\s+([0-9.]+)")


@dataclass(frozen=True)
class Model:
    """One model's forgetting/acquisition sweep configuration."""

    slug: str            # short id, used in output filenames
    title_name: str      # human-readable model name for the figure title
    eval_dir: Path       # outputs/<...> directory holding the per-run subdirs
    run_prefix: str      # prefix shared by every run dir name in eval_dir
                         # (e.g. "qasper-llama" → "qasper-llama-key-value-…")


MODELS: list[Model] = [
    Model(
        slug="llama",
        title_name="Llama",
        eval_dir=CARTRIDGES_REPO
        / "outputs"
        / "qasper_forgetting_eval_llama_granularity-x-sparsity_3273481",
        run_prefix="qasper-llama",
    ),
    Model(
        slug="qwen",
        title_name="Qwen",
        # Symlink-based merge of the per-head+global (3315625) and per-layer
        # (3344506) Slurm batches; see the dir for individual run pointers.
        eval_dir=CARTRIDGES_REPO
        / "outputs"
        / "qasper_forgetting_eval_qwen_granularity-x-sparsity_3315625_3344506",
        run_prefix="qasper-qwen",
    ),
]


def read_perplexity(run_dir: Path) -> float:
    log = run_dir / "eval.log"
    text = log.read_text()
    matches = _PPL_RE.findall(text)
    if not matches:
        raise RuntimeError(f"no perplexity in {log}")
    # Final wandb summary line is always the last match.
    return float(matches[-1])


def collect(model: Model) -> dict:
    """Return forgetting/acquisition curves + averaged baseline for one model.

    The Phase-1-only baseline cartridge does not depend on Phase 2 granularity
    in spirit — it's just "the cartridge before any Phase 2 finetuning". The
    reference figure uses one shared baseline line, so we average the three
    granularity baselines into a single number for each subplot. Per-gran
    baselines are still emitted to the CSV summary.
    """
    out: dict = {"per_gran": {}}
    base_qa: list[float] = []
    base_mt: list[float] = []
    for gran in GRANULARITIES:
        forget, learn = [], []
        for k in SPARSITY_LEVELS:
            forget.append(
                read_perplexity(
                    model.eval_dir
                    / f"{model.run_prefix}-key-value-{gran}-top-{k}__qa_eval"
                )
            )
            learn.append(
                read_perplexity(
                    model.eval_dir
                    / f"{model.run_prefix}-key-value-{gran}-top-{k}__mt_eval"
                )
            )
        out["per_gran"][gran] = {"forgetting": forget, "acquisition": learn}

        base_qa.append(
            read_perplexity(
                model.eval_dir / f"{model.run_prefix}-baseline-{gran}__qa_eval"
            )
        )
        base_mt.append(
            read_perplexity(
                model.eval_dir / f"{model.run_prefix}-baseline-{gran}__mt_eval"
            )
        )

    out["baseline_forgetting"] = sum(base_qa) / len(base_qa)
    out["baseline_acquisition"] = sum(base_mt) / len(base_mt)
    out["baseline_forgetting_per_gran"] = dict(zip(GRANULARITIES, base_qa))
    out["baseline_acquisition_per_gran"] = dict(zip(GRANULARITIES, base_mt))
    return out


def _annotate_points(ax, xs, ys, color, *, dy: int) -> None:
    """Label every data point.

    Endpoints are offset outward (left of x[0], right of x[-1]) so the text
    stays inside the axes; interior points are centered above/below the
    marker. The per-line vertical offset `dy` is set by the caller and lets
    the three granularity lines stagger their labels so values don't pile on
    top of each other where the lines bunch close together (e.g. the
    acquisition subplot's ~1 ppl band).
    """
    va = "bottom" if dy >= 0 else "top"
    n = len(xs)
    for i, (x, y) in enumerate(zip(xs, ys)):
        if i == 0:
            xytext, ha = (-6, dy), "right"
        elif i == n - 1:
            xytext, ha = (6, dy), "left"
        else:
            xytext, ha = (0, dy), "center"
        ax.annotate(
            f"{y:.2f}", (x, y),
            textcoords="offset points", xytext=xytext,
            ha=ha, va=va, fontsize=8, color=color,
        )


def _setup_axis(ax, title, subtitle, ylabel) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(SPARSITY_LEVELS)
    ax.set_xticklabels([str(s) for s in SPARSITY_LEVELS])
    ax.set_xlabel("Active slot budget (top-t)")
    # Y-axis is linear; the data being plotted is already ln(ppl), i.e.
    # cross-entropy loss / NLL, so a linear axis on the log-transformed
    # values gives evenly-spaced ticks (e.g. 2.0, 2.5, 3.0, 3.5).
    ax.set_ylabel(ylabel)
    # Title is bumped up to leave room for a non-overlapping italic subtitle.
    ax.set_title(title, fontsize=12, weight="bold", y=1.10)
    ax.text(
        0.5, 1.02, subtitle,
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=9, style="italic", color="#555555",
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.4)


def plot(model: Model, data: dict, path: Path) -> None:
    fig, (ax_f, ax_a) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # Plot ln(ppl) (= NLL / cross-entropy loss) on a linear y-axis. Same
    # qualitative shape as a log-y plot of ppl, but axis labels and annotation
    # values are the actual log-perplexity numbers (~2.0–3.7).
    log = math.log

    base_f = log(data["baseline_forgetting"])
    base_a = log(data["baseline_acquisition"])

    # Stagger labels per granularity so values don't pile on top of each
    # other where the 3 lines bunch close together.
    LABEL_DY = {"per-head": 10, "per-layer": -14, "global": 22}

    # -------- Forgetting (Phase 1 ln(QA ppl)) --------
    for gran in GRANULARITIES:
        ys = [log(y) for y in data["per_gran"][gran]["forgetting"]]
        style = GRAN_STYLE[gran]
        ax_f.plot(SPARSITY_LEVELS, ys, lw=2, ms=8, **style)
        _annotate_points(
            ax_f, SPARSITY_LEVELS, ys, style["color"], dy=LABEL_DY[gran]
        )

    ax_f.axhline(base_f, linestyle="--", color="grey", alpha=0.7, linewidth=1.3)
    ax_f.annotate(
        f"baseline\n{base_f:.2f}",
        xy=(SPARSITY_LEVELS[-1], base_f), xytext=(-2, 4),
        textcoords="offset points",
        ha="right", va="bottom", fontsize=8, color="grey",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="grey", lw=0.6),
    )

    _setup_axis(
        ax_f,
        "Forgetting — Phase 1 (Question-Answering papers)",
        "lower (closer to the baseline cartridge) = less forgetting ",
        "ln(perplexity)",
    )

    # -------- Acquisition (Phase 2 ln(MT ppl)) --------
    log_acq_per_gran = {
        gran: [log(v) for v in data["per_gran"][gran]["acquisition"]]
        for gran in GRANULARITIES
    }
    for gran in GRANULARITIES:
        ys = log_acq_per_gran[gran]
        style = GRAN_STYLE[gran]
        ax_a.plot(SPARSITY_LEVELS, ys, lw=2, ms=8, **style)
        _annotate_points(
            ax_a, SPARSITY_LEVELS, ys, style["color"], dy=LABEL_DY[gran]
        )

    ax_a.axhline(base_a, linestyle="--", color="grey", alpha=0.7, linewidth=1.3)
    ax_a.annotate(
        f"baseline\n{base_a:.2f}",
        xy=(SPARSITY_LEVELS[-1], base_a), xytext=(-2, -4),
        textcoords="offset points",
        ha="right", va="top", fontsize=8, color="grey",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="grey", lw=0.6),
    )

    _setup_axis(
        ax_a,
        "Acquisition — Phase 2 (Machine-Translation papers)",
        "lower (further below baseline) = more new-task learning",
        "ln(perplexity)",
    )
    # Stretch acquisition y-range to include the (much higher) baseline.
    all_log_acq = [v for gran in GRANULARITIES for v in log_acq_per_gran[gran]]
    ax_a.set_ylim(min(all_log_acq) - 0.10, base_a + 0.15)

    # -------- Single shared legend below the figure --------
    handles = [
        plt.Line2D(
            [], [], color=GRAN_STYLE[g]["color"], marker=GRAN_STYLE[g]["marker"],
            lw=2, ms=8, label=GRAN_STYLE[g]["label"],
        )
        for g in GRANULARITIES
    ] + [
        plt.Line2D(
            [], [], color="grey", linestyle="--", lw=1.3,
            label="baseline cartridge (no Phase 2 finetuning)",
        ),
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center", bbox_to_anchor=(0.5, -0.02),
        ncol=4, frameon=False, fontsize=10,
        handlelength=2.4, columnspacing=2.4,
    )

    fig.suptitle(
        f"Sparse Qasper cartridges ({model.title_name}) — "
        "forgetting vs. acquisition across sparsity",
        fontsize=13, weight="bold", y=1.02,
    )
    fig.text(
        0.5, 0.965,
        "key-value sparse finetuning · global vs. per-head vs. per-layer granularity"
        " · y-axis = ln(perplexity)",
        ha="center", va="bottom",
        fontsize=10, style="italic", color="#555555",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(
        path,
        dpi=160, bbox_inches="tight",
        bbox_extra_artists=(legend,),
    )
    plt.close(fig)
    print(f"wrote {path}")


def write_summary(model: Model, data: dict, path: Path) -> None:
    header = (
        "model,granularity,top_k,"
        "forgetting_ppl,acquisition_ppl,"
        "forgetting_log_ppl,acquisition_log_ppl"
    )
    lines = [header]
    for gran in GRANULARITIES:
        for k, f, a in zip(
            SPARSITY_LEVELS,
            data["per_gran"][gran]["forgetting"],
            data["per_gran"][gran]["acquisition"],
        ):
            lines.append(
                f"{model.slug},{gran},{k},"
                f"{f:.5f},{a:.5f},"
                f"{math.log(f):.5f},{math.log(a):.5f}"
            )
        bf = data["baseline_forgetting_per_gran"][gran]
        ba = data["baseline_acquisition_per_gran"][gran]
        lines.append(
            f"{model.slug},{gran},baseline,"
            f"{bf:.5f},{ba:.5f},"
            f"{math.log(bf):.5f},{math.log(ba):.5f}"
        )
    avg_f = data["baseline_forgetting"]
    avg_a = data["baseline_acquisition"]
    lines.append(
        f"{model.slug},shared,baseline_avg,"
        f"{avg_f:.5f},{avg_a:.5f},"
        f"{math.log(avg_f):.5f},{math.log(avg_a):.5f}"
    )
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        if not model.eval_dir.exists():
            print(f"skipping {model.slug}: missing {model.eval_dir}")
            continue
        data = collect(model)
        plot(model, data, OUT_DIR / f"{model.slug}_granularity_x_sparsity.png")
        write_summary(
            model, data, OUT_DIR / f"{model.slug}_granularity_x_sparsity_summary.csv"
        )


if __name__ == "__main__":
    main()
