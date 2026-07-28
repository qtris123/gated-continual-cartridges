"""DIAG-OBJ helper: pull per-document / per-layer solve MSE out of an AM phase-2 run dir.

Usage:  PYTHONPATH=/tmp/DIAG-OBJ_frozen python collect_mse.py <run_dir> [<run_dir> ...]
Prints one JSON blob per run dir to stdout.  Read-only; touches no source.
"""

import json
import sys
from pathlib import Path

import torch


def summarize(run_dir: str) -> dict:
    rd = Path(run_dir)
    out = {"run_dir": str(rd)}

    summary_path = rd / "phase2_summary.json"
    if summary_path.exists():
        out["phase2_summary"] = json.loads(summary_path.read_text())

    agg_path = rd / "per_document_am_stats.pt"
    per_doc = []
    if agg_path.exists():
        agg = torch.load(agg_path, weights_only=False, map_location="cpu")
        out["n_documents"] = agg.get("n_documents")
        out["wall_clock_s"] = agg.get("wall_clock_s")
        for rec in agg.get("per_document", []):
            per_doc.append(
                {
                    "doc_index": rec.get("doc_index"),
                    "slug": rec.get("slug"),
                    "mean_mse": rec.get("mean_mse"),
                    "n_queries": rec.get("n_queries"),
                    "total_s": (rec.get("timing_s") or {}).get("total_s"),
                }
            )
    out["per_document"] = per_doc
    mses = [d["mean_mse"] for d in per_doc if d["mean_mse"] is not None]
    if mses:
        out["mse_first"] = mses[0]
        out["mse_last"] = mses[-1]
        out["mse_mean"] = sum(mses) / len(mses)
        out["mse_min"] = min(mses)
        out["mse_max"] = max(mses)

    # per-layer MSE from the last document's am_doc_*.pt
    doc_files = sorted(rd.glob("am_doc_*.pt"))
    if doc_files:
        blob = torch.load(doc_files[-1], weights_only=False, map_location="cpu")
        st = blob.get("am_stats")
        mpl = getattr(st, "mse_per_layer", None)
        if mpl:
            mpl = {int(k): float(v) for k, v in mpl.items()}
            out["last_doc_file"] = doc_files[-1].name
            out["last_doc_mse_per_layer"] = mpl
            vals = list(mpl.values())
            out["last_doc_mse_per_layer_stats"] = {
                "n_layers": len(vals),
                "mean": sum(vals) / len(vals),
                "min": min(vals),
                "max": max(vals),
                "argmin_layer": min(mpl, key=mpl.get),
                "argmax_layer": max(mpl, key=mpl.get),
            }
        mask = getattr(st, "mask", None)
        if mask is not None:
            out["last_doc_mask_top_t"] = getattr(mask, "top_t", None)
            ppl = getattr(mask, "positions_per_layer", None)
            if ppl:
                sizes = {int(k): int(v.numel()) for k, v in ppl.items()}
                out["last_doc_selected_per_layer_min"] = min(sizes.values())
                out["last_doc_selected_per_layer_max"] = max(sizes.values())
                out["last_doc_n_tokens"] = getattr(mask, "n_tokens", None)
        # first doc too
        blob0 = torch.load(doc_files[0], weights_only=False, map_location="cpu")
        st0 = blob0.get("am_stats")
        mpl0 = getattr(st0, "mse_per_layer", None)
        if mpl0:
            vals0 = [float(v) for v in mpl0.values()]
            out["first_doc_mse_per_layer_stats"] = {
                "n_layers": len(vals0),
                "mean": sum(vals0) / len(vals0),
                "min": min(vals0),
                "max": max(vals0),
            }
    return out


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(json.dumps(summarize(d), indent=2))
