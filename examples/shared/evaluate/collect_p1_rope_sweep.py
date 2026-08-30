#!/usr/bin/env python3
"""Tabulate the Phase-1 positional-encoding sweep from its per-arm summaries.

Reads every ``summary.json`` under a sweep root (one subdirectory per arm, as laid
out by ``examples/qasper/pipelines/sweep_initial_am_rope.sh``) and prints the
positional configuration next to the eval losses, so the rotary base can be read
against the metric it is supposed to move.

Usage:
    python examples/shared/evaluate/collect_p1_rope_sweep.py [SWEEP_ROOT]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DEFAULT_ROOT = Path(
    os.environ.get("CARTRIDGES_DIR", ".")
) / "outputs" / "p1_rope_sweep"


def load_arms(root: Path) -> list[dict]:
    arms = []
    for summary_path in sorted(root.glob("*/*/*/summary.json")):
        arm = summary_path.relative_to(root).parts[0]
        summary = json.loads(summary_path.read_text())
        stats = summary.get("compaction_stats") or {}
        evals = summary.get("eval_metrics") or {}
        arms.append(
            {
                "arm": arm,
                "beta": summary.get("enable_beta"),
                "rebake": summary.get("rebake_key_positions"),
                "theta": summary.get("rope_theta"),
                "global_pos": summary.get("global_teacher_positions"),
                "theta_ok": stats.get("rope_theta_matches_model"),
                "t_teacher": stats.get("T_teacher"),
                "n_docs": stats.get("n_documents"),
                "mse": stats.get("recon_mse_mean"),
                "key_absmax": stats.get("key_absmax"),
                "value_absmax": stats.get("value_absmax"),
                "losses": {k: v.get("loss") for k, v in evals.items()},
                "ppls": {k: v.get("perplexity") for k, v in evals.items()},
                "note": stats.get("rope_note"),
                "path": str(summary_path.parent),
            }
        )
    return arms


def fmt(value, spec: str = ".4f") -> str:
    return "—" if value is None else format(value, spec)


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    arms = load_arms(root)
    if not arms:
        print(f"No summary.json found under {root}")
        return 1

    eval_names = sorted({name for a in arms for name in a["losses"]})

    header = (
        f"{'arm':<28} {'beta':>5} {'rebake':>7} {'theta':>10} {'gpos':>5} "
        f"{'theta_ok':>9} {'recon_mse':>11} {'|v|max':>9}"
        + "".join(f"{n + ' loss':>11}" for n in eval_names)
        + "".join(f"{n + ' ppl':>11}" for n in eval_names)
    )
    print(header)
    print("-" * len(header))
    for a in arms:
        row = (
            f"{a['arm']:<28} {str(a['beta']):>5} {str(a['rebake']):>7} "
            f"{fmt(a['theta'], '.0f'):>10} {str(a['global_pos']):>5} "
            f"{str(a['theta_ok']):>9} {fmt(a['mse'], '.6f'):>11} "
            f"{fmt(a['value_absmax'], '.1f'):>9}"
            + "".join(f"{fmt(a['losses'].get(n)):>11}" for n in eval_names)
            + "".join(f"{fmt(a['ppls'].get(n), '.3f'):>11}" for n in eval_names)
        )
        print(row)

    print("\nteacher corpus:")
    for a in arms:
        print(f"  {a['arm']:<28} {a['n_docs']} docs / {a['t_teacher']} tokens")

    print("\nrope_note per arm:")
    for a in arms:
        print(f"  {a['arm']}:\n    {a['note']}")

    print("\ncache paths:")
    for a in arms:
        print(f"  {a['arm']:<28} {a['path']}/cache_last.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
