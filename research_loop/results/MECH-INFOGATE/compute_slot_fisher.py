#!/usr/bin/env python
"""MECH-008 — materialise the cached diagonal-Fisher slot scores the ranker loads.

`SLOT_SELECTION=fisher` scores each cartridge slot by the diagonal empirical
Fisher of the **QA** loss w.r.t. that slot's value vector:

    L_e          = (1/t_e) * sum_{scored tokens of example e} ce_by_token
    fisher[l, j] = (1/E) * sum_e sum_h sum_c ( dL_e / dv[l, h, j, c] )^2

That is a *diagnostic* backward pass: no optimizer is ever constructed, `.grad`
is zeroed after each read and the cache is never modified, so `gradient_steps`
stays 0. It is nonetheless a real cost -- **98.8 s of backward over 78 QA + 69 MT
examples on one GH200** (DIAG-IMPORTANCE `diagnostics.fisher_cost.total_s`) --
which is exactly why it is paid ONCE and cached to disk rather than recomputed
per document inside the training loop.

Two modes:

  --from-diag (default)
      Copy `score_fisher` out of `state/diagnostics/DIAG-IMPORTANCE.npz`. That
      array was measured by `research_loop/results/DIAG-IMPORTANCE/
      measure_slot_importance.py::collect_fisher` on **this same untouched
      Phase-1 cartridge**, so it is the already-paid cache, not an approximation.
      No GPU.

  --recompute
      Re-run the measurement from scratch. This just shells out to
      DIAG-IMPORTANCE's own entry point (`launch_diag_importance.sh`) rather than
      forking its code, so there is exactly one implementation of the Fisher.

Usage:
    python compute_slot_fisher.py --out research_loop/state/diagnostics/slot_fisher_qa_phase1.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

REPO = os.environ.get("CARTRIDGES_DIR") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
DIAG_NPZ = os.path.join(REPO, "research_loop/state/diagnostics/DIAG-IMPORTANCE.npz")
DEFAULT_OUT = os.path.join(
    REPO, "research_loop/state/diagnostics/slot_fisher_qa_phase1.npz"
)
PHASE1 = os.path.join(REPO, "outputs/phase1_selfdistill_qwen512/cache_last.pt")


def _sha256(path: str, nbytes: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(nbytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--diag-npz", default=DIAG_NPZ)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    if args.recompute:
        print(
            "Re-running the Fisher measurement means re-running DIAG-IMPORTANCE:\n"
            f"    bash {REPO}/research_loop/results/DIAG-IMPORTANCE/"
            "launch_diag_importance.sh\n"
            "then re-run this script without --recompute. Not forked here so that "
            "there is exactly one implementation of the Fisher.",
            file=sys.stderr,
        )
        return 2

    if not os.path.exists(args.diag_npz):
        print(f"missing {args.diag_npz}", file=sys.stderr)
        return 1

    with np.load(args.diag_npz) as z:
        fisher = np.asarray(z["score_fisher"], dtype=np.float32)  # (36, 511) QA
        fisher_mt = np.asarray(z["score_fisher_mt"], dtype=np.float32)

    assert np.isfinite(fisher).all(), "non-finite Fisher"
    assert (fisher >= 0).all(), "negative Fisher (it is a sum of squares)"

    meta = {
        "metric": "diagonal empirical Fisher of the QA loss w.r.t. slot values",
        "formula": "fisher[l,j] = (1/E) sum_e sum_h sum_c (dL_e/dv[l,h,j,c])^2",
        "split": "data/qasper/eval/qasper_eval_QA.parquet (E = 78 examples)",
        "measured_on_cartridge": PHASE1,
        "cartridge_sha256": _sha256(PHASE1) if os.path.exists(PHASE1) else None,
        "source": "DIAG-IMPORTANCE::collect_fisher (measure_slot_importance.py)",
        "source_npz": args.diag_npz,
        "cost_s_backward_total": 98.8,
        "cost_note": (
            "98.8 s of diagnostic backward over 78 QA + 69 MT examples on one "
            "GH200; gradient_steps = 0 (no optimizer constructed, cache never "
            "modified). Paid once, cached here."
        ),
        "shape": list(fisher.shape),
        "direction": "ASCENDING — lowest Fisher = safest to overwrite",
        "leakage_caveat": (
            "Scored on the QA EVAL split, matching DIAG-IMPORTANCE. The selector "
            "therefore sees the retention eval's own gradients; the QA number for "
            "SLOT_SELECTION=fisher is an optimistic bound on retention, not a "
            "clean held-out measurement. It is used because the question this arm "
            "answers is about the MT axis (does QA-safety selection cost "
            "acquisition bandwidth?), where no leakage applies."
        ),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        score_fisher=fisher,
        score_fisher_mt=fisher_mt,
        meta=np.array(json.dumps(meta)),
    )
    print(json.dumps(meta, indent=2))
    print(f"wrote {args.out}  shape={fisher.shape}  sha256={_sha256(args.out)[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
