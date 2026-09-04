#!/usr/bin/env python3
"""Manifest-driven reporting for continual-AM selection sweeps.

Reads artifacts under outputs/ (per-arm teacher-forced logppl + generation
accuracy matrices) and produces, for a sweep declared in a manifest:

  --mode frontier : printed acquisition / final / forgetting tables (loss + acc).
  --mode tables   : tidy CSVs (results_long / results_by_stage / results_arm).

Arms come from the manifest (examples/shared/am/manifests/*.yaml) via
`examples.shared.am.sweep.resolve`, so there are NO hardcoded arm lists: adding
an arm or a new selection stream is a manifest edit. `top_t` / `lambda` labels are
derived from each arm's axis value (+ the stream's `fixed:` block).

Usage:
    python -m examples.shared.analysis.continual_sweep_report \\
        --manifest examples/shared/am/manifests/soft_locality.yaml \\
        --dataset quality --mode frontier --stream top_t
    python -m examples.shared.analysis.continual_sweep_report \\
        --manifest examples/shared/am/manifests/soft_locality.yaml \\
        --dataset quality --mode tables --out-dir outputs/experiments/soft_locality/tables
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from examples.shared.am.sweep import load_manifest, resolve
from examples.shared.paths import ROOT

LOSS_PROTO = "teacher-forced-logppl-v1"
ACC_PROTO = "accuracy-freeform-mc-options-primeAnswer-v1"
DEFAULT_TOP_T = 32


def _matrix(dataset: str, tag: str, proto: str):
    p = ROOT / f"outputs/evaluations/{dataset}/{tag}/{proto}/matrix.json"
    return json.load(open(p)) if p.exists() else None


def arm_knobs(manifest: dict, arm) -> tuple[int, float]:
    """(top_t, lambda) for an arm from its axis value + the stream's fixed block."""
    stream = manifest["streams"][arm.stream]
    fixed = stream.get("fixed", {}) or {}
    field = arm.axis.split(".")[-1]
    top_t = arm.value if field == "top_t" else fixed.get("slots.top_t", DEFAULT_TOP_T)
    lam = (arm.value if field == "usage_penalty_lambda"
           else fixed.get("slots.usage_penalty_lambda", 0.0))
    return int(top_t), float(lam)


# ---- loss / accuracy frontier rows -----------------------------------------
def loss_frontier(dataset: str, tag: str):
    m = _matrix(dataset, tag, LOSS_PROTO)
    if m is None:
        return None
    tasks, cells = m["task_order"], m["cells"]
    stages = sorted(cells)
    cm = m.get("continual_metrics", {})
    diag = [cells[f"p{i+1:02d}"][t]["loss"]
            for i, t in enumerate(tasks)
            if cells.get(f"p{i+1:02d}", {}).get(t)]
    return {
        "acq": sum(diag) / len(diag) if diag else float("nan"),
        "final_avg": cm.get("average_loss_on_seen_tasks", {}).get(stages[-1], float("nan")),
        "mean_forget": cm.get("mean_forgetting", float("nan")),
    }


def acc_frontier(dataset: str, tag: str):
    m = _matrix(dataset, tag, ACC_PROTO)
    if m is None:
        return None
    acc = m["accuracy"]
    n = len(acc)
    diag = [acc[i][i] for i in range(n) if acc[i][i] is not None]
    final = [x for x in acc[-1] if x is not None]
    forget = [acc[k][k] - acc[-1][k] for k in range(n - 1)
              if acc[k][k] is not None and acc[-1][k] is not None]
    return {
        "diag": sum(diag) / len(diag) if diag else float("nan"),
        "final": sum(final) / len(final) if final else float("nan"),
        "forget": sum(forget) / len(forget) if forget else float("nan"),
        "complete": m.get("num_complete_cells"),
        "expected": m.get("num_expected_cells"),
    }


def frontier_report(manifest, dataset, stream):
    print(f"\n===== FRONTIER :: {dataset} :: stream={stream or 'all'} =====")
    print(f"{'arm':<12}{'top_t':>6}{'lam':>6}  |{'loss_acq':>9}{'loss_fin':>9}"
          f"{'loss_fgt':>9}  |{'acc_acq':>8}{'acc_fin':>8}{'acc_fgt':>8}  cells")
    for arm in resolve(manifest, stream):
        tt, lam = arm_knobs(manifest, arm)
        L, A = loss_frontier(dataset, arm.tag), acc_frontier(dataset, arm.tag)
        ls = (f"{L['acq']:>9.3f}{L['final_avg']:>9.3f}{L['mean_forget']:>9.3f}"
              if L else f"{'(pending)':>27}")
        if A:
            cells = "" if A["complete"] == A["expected"] else f" {A['complete']}/{A['expected']}"
            as_ = f"{A['diag']:>8.3f}{A['final']:>8.3f}{A['forget']:>8.3f}{cells}"
        else:
            as_ = f"{'(pending)':>24}"
        print(f"{arm.label:<12}{tt:>6}{lam:>6.1f}  |{ls}  |{as_}")
    print("  loss lower=better; acc higher=better; ~1-3%/cell acc noise (trust gaps >0.03)")


# ---- tidy CSV export -------------------------------------------------------
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _long_rows(manifest, datasets):
    seen = set()
    rows = []
    for arm in resolve(manifest):  # all streams
        if arm.tag in seen:
            continue
        seen.add(arm.tag)
        tt, lam = arm_knobs(manifest, arm)
        for dataset in datasets:
            L = _matrix(dataset, arm.tag, LOSS_PROTO)
            A = _matrix(dataset, arm.tag, ACC_PROTO)
            for stage in range(1, 6):
                for ev in range(1, 6):
                    logppl = None
                    if L is not None:
                        c = L["cells"].get(f"p{stage:02d}", {}).get(f"p{ev}")
                        logppl = c["loss"] if c else None
                    acc = None
                    if A is not None:
                        try:
                            acc = A["accuracy"][stage - 1][ev - 1]
                        except (IndexError, TypeError):
                            acc = None
                    rows.append({
                        "dataset": dataset, "tag": arm.tag, "top_t": tt, "lambda": lam,
                        "stage": stage, "eval": ev, "seen": int(ev <= stage),
                        "is_diag": int(ev == stage), "logppl": logppl, "accuracy": acc,
                    })
    return rows


def _by_stage(long_rows):
    groups = {}
    for r in long_rows:
        groups.setdefault((r["dataset"], r["tag"], r["top_t"], r["lambda"], r["stage"]), []).append(r)
    out = []
    for (ds, tag, tt, lam, stage), rs in sorted(groups.items()):
        seen = [r for r in rs if r["seen"]]
        diag = next((r for r in rs if r["is_diag"]), None)
        out.append({
            "dataset": ds, "tag": tag, "top_t": tt, "lambda": lam, "stage": stage,
            "logppl_all": _mean([r["logppl"] for r in rs]),
            "logppl_seen": _mean([r["logppl"] for r in seen]),
            "acc_all": _mean([r["accuracy"] for r in rs]),
            "acc_seen": _mean([r["accuracy"] for r in seen]),
            "logppl_diag": diag["logppl"] if diag else None,
            "acc_diag": diag["accuracy"] if diag else None,
        })
    return out


def _arm_summary(stage_rows):
    groups = {}
    for r in stage_rows:
        groups.setdefault((r["dataset"], r["tag"], r["top_t"], r["lambda"]), {})[r["stage"]] = r
    out = []
    for (ds, tag, tt, lam), by in sorted(groups.items()):
        final = by.get(5, {})
        out.append({
            "dataset": ds, "tag": tag, "top_t": tt, "lambda": lam,
            "final_logppl_seen": final.get("logppl_seen"),
            "final_acc_all": final.get("acc_all"),
            "avg_acc_5stg": _mean([by[s].get("acc_all") for s in by]),
            "acc_acq_diag_mean": _mean([by[s].get("acc_diag") for s in by]),
        })
    return out


def _write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("wrote", path, f"({len(rows)} rows)")


def tables_report(manifest, datasets, out_dir: Path):
    long_rows = _long_rows(manifest, datasets)
    stage_rows = _by_stage(long_rows)
    arm_rows = _arm_summary(stage_rows)
    _write_csv(out_dir / "results_long.csv", long_rows,
               ["dataset", "tag", "top_t", "lambda", "stage", "eval", "seen",
                "is_diag", "logppl", "accuracy"])
    _write_csv(out_dir / "results_by_stage.csv", stage_rows,
               ["dataset", "tag", "top_t", "lambda", "stage", "logppl_all",
                "logppl_seen", "acc_all", "acc_seen", "logppl_diag", "acc_diag"])
    _write_csv(out_dir / "results_arm.csv", arm_rows,
               ["dataset", "tag", "top_t", "lambda", "final_logppl_seen",
                "final_acc_all", "avg_acc_5stg", "acc_acq_diag_mean"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--mode", choices=["frontier", "tables"], default="frontier")
    ap.add_argument("--dataset", default=None, help="frontier: one dataset")
    ap.add_argument("--stream", default=None, help="frontier: restrict to one stream")
    ap.add_argument("--out-dir", default=None, help="tables: CSV output dir")
    args = ap.parse_args()
    manifest = load_manifest(args.manifest)
    if args.mode == "frontier":
        ds = args.dataset or manifest["datasets"][0]
        frontier_report(manifest, ds, args.stream)
    else:
        datasets = [args.dataset] if args.dataset else manifest["datasets"]
        out = Path(args.out_dir) if args.out_dir else ROOT / "outputs/experiments/soft_locality/tables"
        tables_report(manifest, datasets, out)


if __name__ == "__main__":
    main()
