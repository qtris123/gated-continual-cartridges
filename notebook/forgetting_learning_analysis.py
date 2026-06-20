"""
Forgetting / learning analysis (Part 2) — full bsize=32 qasper grid.

Pulls per-checkpoint perplexity from the post-hoc forgetting evaluations and
produces the four Part-2 figures requested by the user:

  2.1  Per-head vs Per-layer at the *same* sparse-FT setting
       (bsize=32, value-only, top_t=128 — bar chart of forgetting & learning ppl)
  2.2  Per-head vs Per-layer across sparsity in {64, 128, 256, 512}
       (bsize=32, value-only — line plot)
  2.3  Forgetting vs learning Pareto scatter for bsize=32
       (axes = QA Δppl & MT ppl; marker shape = top_t;
        hollow = value-only, filled = key-value;
        colour = per-head vs per-layer; star = baseline)
  2.4  Batch-size effect on value-only across {per_layer, per_head}
       (line plot of forgetting & learning vs t for bsize=32 vs bsize=64)

Conventions:
    QA eval = phase-1 task = forgetting (lower is better)
    MT eval = phase-2 task = learning   (lower is better)

Sources (all bsize=32 runs; configs verified in this conversation):
    Old CSV      = sparse_continual_dynamics_investigation/results/forgetting_eval.csv
                   - the qasper "key-value-per-{head,layer}-top-{64,128,256,512}" rows
                     point at the .../qasper-per-{head,layer}-top-X-key-value_all-reduce
                     run dirs whose configs say global_batch_size=32
                   - the "value-only-per-{head,layer}-top-{64,128,256,512}" rows in the
                     CSV point at the .../2026-06-18-...-bsize-64 run dirs (i.e. bsize=64).
                     We use those only for the bsize-effect ablation (Fig 2.4).
    New evals    = outputs/qasper_forgetting_eval_bsize32/* — produced by
                   notebook/_run_bsize32_evals.sh on the 2026-06-19 bsize=32 ckpts.

Outputs land in notebook/figures/.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EXISTING_CSV = Path(
    "/localhome/local-triv/sparse_continual_dynamics_investigation/results/forgetting_eval.csv"
)
BSIZE32_DIR = REPO / "outputs" / "qasper_forgetting_eval_bsize32"
FIG_DIR = REPO / "notebook" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@dataclass
class EvalRow:
    scope: str           # "key-value" | "value-only" | "baseline"
    granularity: str     # "per_layer" | "per_head"
    top_t: int | None    # None for baseline
    batch_size: int      # 32 or 64
    qa_ppl: float        # forgetting (phase-1 = QA)
    mt_ppl: float        # learning (phase-2 = MT)


def _bsize_for_legacy_label(label: str, scope: str) -> int:
    """Infer batch size for a row from sparse_continual_dynamics_investigation/forgetting_eval.csv.

    The qasper rows in that CSV came from a sweep that used:
      - all key-value runs   → bsize=32 (run dirs verified)
      - all value-only runs  → bsize=64 (the '2026-06-18-...-bsize-64' run dirs;
                                          the eval.log paths still reference the
                                          original dir name before the rename,
                                          but the configs are bsize=64)
      - both baseline runs   → bsize=32 (Stage-1 cartridges)

    The new bsize=32 value-only runs (Jun 19) live in a separate eval directory
    (qasper_forgetting_eval_bsize32/) and are loaded by load_bsize32_evals(),
    so the legacy CSV value-only rows are unambiguously bsize=64.
    """
    if "baseline" in label:
        return 32
    if scope == "key-value":
        return 32
    return 64  # legacy value-only rows were the bsize=64 sweep


def load_existing_qasper_csv(path: Path) -> list[EvalRow]:
    df = pd.read_csv(path)
    df = df[df["dataset"] == "qasper"].copy()
    rows: list[EvalRow] = []
    grouped = df.groupby(["label", "scope", "granularity", "top_t"], dropna=False)
    for (label, scope, gran, top_t), group in grouped:
        try:
            qa_row = group[group["eval_split"] == "phase1_data"].iloc[0]
            mt_row = group[group["eval_split"] == "phase2_data"].iloc[0]
        except IndexError:
            continue
        is_baseline = "baseline" in str(label)
        bsize = _bsize_for_legacy_label(str(label), str(scope))
        rows.append(
            EvalRow(
                scope="baseline" if is_baseline else scope,
                granularity=gran,
                top_t=None if is_baseline else int(top_t),
                batch_size=bsize,
                qa_ppl=float(qa_row["ppl"]),
                mt_ppl=float(mt_row["ppl"]),
            )
        )
    return rows


_PPL_RE = re.compile(r"perplexity[ =:]+([0-9.]+)")
_LABEL_RE = re.compile(
    r"qasper-(value-only|key-value)-per-(head|layer)-top-(\d+)-bsize(\d+)"
)


def parse_eval_log(path: Path) -> float | None:
    if not path.exists():
        return None
    matches = _PPL_RE.findall(path.read_text())
    return float(matches[-1]) if matches else None


def load_bsize32_evals(directory: Path) -> list[EvalRow]:
    if not directory.exists():
        return []
    rows: list[EvalRow] = []
    label_set = set()
    for sub in directory.iterdir():
        if not sub.is_dir():
            continue
        for suffix in ("__qa_eval", "__mt_eval"):
            if sub.name.endswith(suffix):
                label_set.add(sub.name[: -len(suffix)])
    for label in sorted(label_set):
        m = _LABEL_RE.match(label)
        if not m:
            continue
        scope, gran, top_t_s, bs_s = m.groups()
        gran = "per_layer" if gran == "layer" else "per_head"
        qa = parse_eval_log(directory / f"{label}__qa_eval" / "eval.log")
        mt = parse_eval_log(directory / f"{label}__mt_eval" / "eval.log")
        if qa is None or mt is None:
            continue
        rows.append(
            EvalRow(
                scope=scope,
                granularity=gran,
                top_t=int(top_t_s),
                batch_size=int(bs_s),
                qa_ppl=qa,
                mt_ppl=mt,
            )
        )
    return rows


def to_dataframe(rows: list[EvalRow]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scope": r.scope,
                "granularity": r.granularity,
                "top_t": r.top_t,
                "batch_size": r.batch_size,
                "qa_ppl": r.qa_ppl,
                "mt_ppl": r.mt_ppl,
            }
            for r in rows
        ]
    )


def summarise(df: pd.DataFrame) -> str:
    pieces = []
    for (scope, gran, bs), grp in df.groupby(["scope", "granularity", "batch_size"]):
        for _, row in grp.sort_values("top_t").iterrows():
            t = "BL" if pd.isna(row.top_t) else int(row.top_t)
            pieces.append(
                f"  {scope:<10} {gran:<9} bs={bs:<2} t={t:>4}  "
                f"QA(forget)={row.qa_ppl:>7.3f}  MT(learn)={row.mt_ppl:>7.3f}"
            )
    return "\n".join(pieces)


# ---------------------------------------------------------------------------
# Style maps
# ---------------------------------------------------------------------------
SHAPE_BY_TOPT = {64: "o", 128: "s", 256: "D", 512: "^"}
COLOR_BY_GRAN = {"per_layer": "#1f4e8e", "per_head": "#9c1c1c"}
BSIZE_COLORS = {32: "#d35400", 64: "#1f4e8e"}


def _baseline(df: pd.DataFrame) -> pd.Series:
    """Return one canonical baseline row (per_layer)."""
    return df[(df["scope"] == "baseline") & (df["granularity"] == "per_layer")].iloc[0]


# ---------------------------------------------------------------------------
# 2.1 — per-head vs per-layer at the same setting (bsize=32, value-only, t=128)
# ---------------------------------------------------------------------------
def plot_2_1(df: pd.DataFrame, save_path: Path):
    target_bs = 32
    sub = df[
        (df["scope"] == "value-only")
        & (df["top_t"] == 128)
        & (df["batch_size"] == target_bs)
    ].sort_values("granularity")
    base = _baseline(df)

    grans = ["per_layer", "per_head"]
    qa_values = [float(sub[sub["granularity"] == g]["qa_ppl"].iloc[0]) for g in grans]
    mt_values = [float(sub[sub["granularity"] == g]["mt_ppl"].iloc[0]) for g in grans]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    width = 0.5
    xpos = np.arange(len(grans))

    ax = axes[0]
    ax.bar(xpos, qa_values, width,
           color=[COLOR_BY_GRAN[g] for g in grans], edgecolor="black")
    ax.axhline(base.qa_ppl, color="grey", lw=1.2, ls="--",
               label=f"baseline = {base.qa_ppl:.2f}")
    for x, v in zip(xpos, qa_values):
        ax.text(x, v + 0.05, f"{v:.2f}\nΔ={v - base.qa_ppl:+.2f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_xticks(xpos)
    ax.set_xticklabels(grans)
    ax.set_ylabel("QA perplexity (forgetting — lower is better)")
    ax.set_title(f"(A) Forgetting (QA eval, value-only, t=128, bsize={target_bs})")
    y_lo = min(qa_values + [base.qa_ppl]) - 0.5
    y_hi = max(qa_values + [base.qa_ppl]) + 0.7
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.bar(xpos, mt_values, width,
           color=[COLOR_BY_GRAN[g] for g in grans], edgecolor="black")
    for x, v in zip(xpos, mt_values):
        ax.text(x, v + 0.04, f"{v:.2f}\nΔ={v - base.mt_ppl:+.2f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_xticks(xpos)
    ax.set_xticklabels(grans)
    ax.set_ylabel("MT perplexity (learning — lower is better)")
    ax.set_title(f"(B) Learning (MT eval, value-only, t=128, bsize={target_bs})")
    ax.set_ylim(min(mt_values) - 0.5, max(mt_values) + 0.6)
    ax.grid(True, axis="y", alpha=0.3)
    ax.text(
        0.98, 0.98,
        f"baseline (no FT) = {base.mt_ppl:.2f}\n(off-chart, axes zoomed)",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, color="grey",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey"),
    )

    fig.suptitle(
        "Task 2.1 — per-head vs per-layer at the same sparse-FT setting "
        f"(qasper, value-only, t=128, bsize={target_bs})",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


# ---------------------------------------------------------------------------
# 2.2 — sparsity sweep: per-head vs per-layer (bsize=32, value-only)
# ---------------------------------------------------------------------------
def plot_2_2(df: pd.DataFrame, save_path: Path):
    target_bs = 32
    sub = df[
        (df["scope"] == "value-only") & (df["batch_size"] == target_bs)
    ].sort_values(["granularity", "top_t"])
    base = _baseline(df)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax_qa, ax_mt = axes

    for gran, color in COLOR_BY_GRAN.items():
        rows = sub[sub["granularity"] == gran]
        ts = rows["top_t"].astype(int).tolist()
        ax_qa.plot(ts, rows["qa_ppl"].tolist(), "o-", color=color, lw=2, ms=9, label=gran)
        ax_mt.plot(ts, rows["mt_ppl"].tolist(), "o-", color=color, lw=2, ms=9, label=gran)

    ax_qa.axhline(base.qa_ppl, color="grey", ls="--",
                  label=f"baseline = {base.qa_ppl:.2f}")

    for ax, title, ylabel in [
        (ax_qa, "(A) Forgetting (QA eval)",
         "QA perplexity — lower = less forgetting"),
        (ax_mt, "(B) Learning (MT eval)",
         "MT perplexity — lower = better learning"),
    ]:
        ax.set_xscale("log", base=2)
        ax.set_xticks([64, 128, 256, 512])
        ax.set_xticklabels(["64", "128", "256", "512"])
        ax.set_xlabel("Sparsity top_t (cache positions trained per layer/head)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")

    ax_mt.set_ylim(min(sub["mt_ppl"]) * 0.95, max(sub["mt_ppl"]) * 1.05)
    ax_mt.text(
        0.98, 0.98,
        f"baseline (no FT) = {base.mt_ppl:.2f}\n(off-chart, axes zoomed)",
        transform=ax_mt.transAxes, ha="right", va="top",
        fontsize=9, color="grey",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey"),
    )
    fig.suptitle(
        f"Task 2.2 — per-head vs per-layer across sparsity (qasper, value-only, "
        f"bsize={target_bs})",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


# ---------------------------------------------------------------------------
# 2.3 — Forgetting vs learning Pareto plot (bsize=32 only)
# ---------------------------------------------------------------------------
def plot_2_3(df: pd.DataFrame, save_path: Path):
    target_bs = 32
    base = _baseline(df)
    sub = df[
        (df["scope"].isin(["key-value", "value-only"]))
        & (df["batch_size"] == target_bs)
    ].copy()
    sub["delta_qa"] = sub["qa_ppl"] - base.qa_ppl

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    for _, row in sub.iterrows():
        marker = SHAPE_BY_TOPT[int(row.top_t)]
        color = COLOR_BY_GRAN[row.granularity]
        if row.scope == "key-value":
            mfc, mec, edge_w = color, "black", 0.6
        else:
            mfc, mec, edge_w = "white", color, 1.6
        ax.scatter(
            row["delta_qa"], row["mt_ppl"],
            marker=marker, s=170,
            facecolors=mfc, edgecolors=mec, linewidths=edge_w,
            zorder=3,
        )

    ax.scatter(
        0, base.mt_ppl, marker="*", s=320,
        facecolors="gold", edgecolors="black", linewidths=1.2,
        zorder=4,
    )
    ax.axvline(0, color="grey", lw=0.7, ls=":")
    ax.axhline(base.mt_ppl, color="grey", lw=0.7, ls=":")

    ax.set_xlabel("Forgetting:  Δ QA perplexity (vs baseline) — lower is better")
    ax.set_ylabel("Learning:  MT perplexity — lower is better")
    ax.set_title(
        f"Task 2.3 — Forgetting vs Learning Pareto\n"
        f"(qasper, bsize={target_bs}, all stage-2 sparse-FT runs)"
    )
    ax.grid(True, alpha=0.3)

    granularity_handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=11,
                   markerfacecolor=COLOR_BY_GRAN["per_layer"],
                   markeredgecolor="black", label="per_layer"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=11,
                   markerfacecolor=COLOR_BY_GRAN["per_head"],
                   markeredgecolor="black", label="per_head"),
    ]
    scope_handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=11,
                   markerfacecolor="grey", markeredgecolor="black",
                   label="filled = key-value"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=11,
                   markerfacecolor="white", markeredgecolor="grey",
                   markeredgewidth=1.6, label="hollow = value-only"),
    ]
    sparsity_handles = [
        plt.Line2D([], [], marker=SHAPE_BY_TOPT[t], linestyle="",
                   markersize=11, markerfacecolor="grey",
                   markeredgecolor="black", label=f"t = {t}")
        for t in [64, 128, 256, 512]
    ]
    baseline_handle = [
        plt.Line2D([], [], marker="*", linestyle="", markersize=15,
                   markerfacecolor="gold", markeredgecolor="black",
                   label=f"baseline (no FT)\nQA={base.qa_ppl:.2f}, MT={base.mt_ppl:.2f}"),
    ]
    ax.legend(
        handles=granularity_handles + scope_handles + sparsity_handles + baseline_handle,
        loc="best", fontsize=9, ncol=2, handletextpad=0.4,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


# ---------------------------------------------------------------------------
# 2.4 — bsize effect on value-only (per_layer + per_head)
# ---------------------------------------------------------------------------
def plot_2_4(df: pd.DataFrame, save_path: Path):
    sub = df[df["scope"] == "value-only"].sort_values(
        ["granularity", "batch_size", "top_t"]
    )
    base = _baseline(df)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax_qa, ax_mt = axes

    style_map = {
        ("per_layer", 32): dict(color="#d35400", ls="-",  marker="o", label="per_layer / bsize=32"),
        ("per_layer", 64): dict(color="#1f4e8e", ls="-",  marker="o", label="per_layer / bsize=64"),
        ("per_head",  32): dict(color="#d35400", ls="--", marker="s", label="per_head  / bsize=32"),
        ("per_head",  64): dict(color="#1f4e8e", ls="--", marker="s", label="per_head  / bsize=64"),
    }
    for (gran, bs), style in style_map.items():
        rows = sub[(sub["granularity"] == gran) & (sub["batch_size"] == bs)]
        if rows.empty:
            continue
        ts = rows["top_t"].astype(int).tolist()
        ax_qa.plot(ts, rows["qa_ppl"].tolist(), lw=2, ms=9, **style)
        ax_mt.plot(ts, rows["mt_ppl"].tolist(), lw=2, ms=9, **{k: v for k, v in style.items() if k != "label"}, label=style["label"])

    ax_qa.axhline(base.qa_ppl, color="grey", ls=":",
                  label=f"baseline = {base.qa_ppl:.2f}")

    for ax, title, ylabel in [
        (ax_qa, "(A) Forgetting (QA eval)",
         "QA perplexity — lower = less forgetting"),
        (ax_mt, "(B) Learning (MT eval)",
         "MT perplexity — lower = better learning"),
    ]:
        ax.set_xscale("log", base=2)
        ax.set_xticks([64, 128, 256, 512])
        ax.set_xticklabels(["64", "128", "256", "512"])
        ax.set_xlabel("Sparsity top_t")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    ax_mt.set_ylim(min(sub["mt_ppl"]) * 0.95, max(sub["mt_ppl"]) * 1.05)
    ax_mt.text(
        0.98, 0.98,
        f"baseline (no FT) = {base.mt_ppl:.2f}\n(off-chart, axes zoomed)",
        transform=ax_mt.transAxes, ha="right", va="top",
        fontsize=9, color="grey",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey"),
    )

    fig.suptitle(
        "Task 2.4 — batch-size effect on value-only sparse FT (qasper)\n"
        "Smaller bsize ⇒ fewer documents per batch ⇒ TF-IDF mass concentrates "
        "on fewer slots ⇒ stronger update on a few positions",
        fontsize=11, y=1.06,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading existing forgetting_eval.csv (qasper rows)…")
    rows = load_existing_qasper_csv(EXISTING_CSV)
    print(f"  {len(rows)} rows loaded from CSV")

    print(f"\nLoading bsize=32 evals from {BSIZE32_DIR}…")
    rows32 = load_bsize32_evals(BSIZE32_DIR)
    print(f"  {len(rows32)} rows loaded from new evals")

    df = to_dataframe(rows + rows32)
    print(f"\nFull dataset ({len(df)} rows):")
    print(summarise(df))

    csv_dump = FIG_DIR / "part2_perplexity_summary.csv"
    df.sort_values(["scope", "granularity", "batch_size", "top_t"]).to_csv(
        csv_dump, index=False
    )
    print(f"\n  CSV summary → {csv_dump}")

    plot_2_1(df, FIG_DIR / "part2_1_per-head_vs_per-layer_t128.png")
    plot_2_2(df, FIG_DIR / "part2_2_sparsity_sweep.png")
    plot_2_3(df, FIG_DIR / "part2_3_forgetting_vs_learning_pareto.png")
    plot_2_4(df, FIG_DIR / "part2_4_bsize_effect.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
