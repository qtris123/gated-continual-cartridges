"""DIAG-CONTENT: per-arm solve-side diagnostics from the run dirs' AM artefacts.

Reads each arm's `am_doc_*.pt` (AMUpdateStats: `mse_per_layer`, `extra` carrying
`ref_mass_on_S_per_layer` + `v_selected_absmax_after_per_layer`), `phase2_summary.json`
(eval metrics, |v|max, reg config) and `per_document_am_stats.pt` (timings), and writes
one JSON with the per-arm blocks.

Adapted from research_loop/results/DIAG-ROPE/collect_stats.py (same fields, 3 arms).

Env: RUNDIR_A / RUNDIR_B / RUNDIR_C, OUT_JSON
"""

from __future__ import annotations

import json
import os
from glob import glob

import torch

OUT_JSON = os.environ["OUT_JSON"]
ARMS = {
    "armA_canonical_MT": os.environ.get("RUNDIR_A", ""),
    "armB_QAcorpus": os.environ.get("RUNDIR_B", ""),
    "armC_MT_reversed": os.environ.get("RUNDIR_C", ""),
}


def collect(run_dir: str) -> dict:
    out: dict = {"run_dir": run_dir}
    summary_path = os.path.join(run_dir, "phase2_summary.json")
    if os.path.exists(summary_path):
        out["phase2_summary"] = json.load(open(summary_path))

    docs = sorted(glob(os.path.join(run_dir, "am_doc_*.pt")))
    per_doc = []
    layer_mse_sum: dict[int, float] = {}
    layer_mass_sum: dict[int, float] = {}
    layer_vmax: dict[int, float] = {}
    n = 0
    for p in docs:
        blob = torch.load(p, map_location="cpu", weights_only=False)
        st = blob["am_stats"]
        extra = dict(getattr(st, "extra", None) or {})
        mass = extra.get("ref_mass_on_S_per_layer", {}) or {}
        vmax = extra.get("v_selected_absmax_after_per_layer", {}) or {}
        mass_vals = [float(v) for v in mass.values()]
        per_doc.append({
            "slug": blob["doc_record"]["slug"],
            "n_conversations": blob["doc_record"].get("n_conversations"),
            "mean_mse": float(st.mean_mse),
            "n_queries": int(st.n_queries),
            "mass_on_S_mean_over_layers": (sum(mass_vals) / len(mass_vals)) if mass_vals else None,
            "v_absmax_over_layers": max([float(v) for v in vmax.values()], default=None),
            "total_s": blob["doc_record"].get("timing_s", {}).get("total_s"),
            "prefill_s": blob["doc_record"].get("timing_s", {}).get("prefill_s"),
        })
        for k, v in st.mse_per_layer.items():
            layer_mse_sum[int(k)] = layer_mse_sum.get(int(k), 0.0) + float(v)
        for k, v in mass.items():
            layer_mass_sum[int(k)] = layer_mass_sum.get(int(k), 0.0) + float(v)
        for k, v in vmax.items():
            layer_vmax[int(k)] = max(layer_vmax.get(int(k), 0.0), float(v))
        n += 1

    out["n_documents"] = n
    out["per_document"] = per_doc
    out["mean_mse_over_documents"] = (sum(d["mean_mse"] for d in per_doc) / n) if n else None
    out["mse_per_layer_mean_over_docs"] = (
        {str(k): layer_mse_sum[k] / n for k in sorted(layer_mse_sum)} if n else {}
    )
    out["mass_on_S_per_layer_mean_over_docs"] = (
        {str(k): layer_mass_sum[k] / n for k in sorted(layer_mass_sum)} if n else {}
    )
    out["v_selected_absmax_per_layer_max_over_docs"] = {
        str(k): layer_vmax[k] for k in sorted(layer_vmax)
    }
    if out["mass_on_S_per_layer_mean_over_docs"]:
        vals = list(out["mass_on_S_per_layer_mean_over_docs"].values())
        out["mass_on_S_mean_over_layers"] = sum(vals) / len(vals)
        out["mass_on_S_min_max_over_layers"] = [min(vals), max(vals)]

    agg_path = os.path.join(run_dir, "per_document_am_stats.pt")
    if os.path.exists(agg_path):
        agg = torch.load(agg_path, map_location="cpu", weights_only=False)
        out["wall_clock_s"] = agg.get("wall_clock_s")
        out["prefill_s_total"] = sum(
            d.get("timing_s", {}).get("prefill_s", 0.0) for d in agg.get("per_document", [])
        )
        out["doc_total_s_sum"] = sum(
            d.get("timing_s", {}).get("total_s", 0.0) for d in agg.get("per_document", [])
        )
    out["config_yaml_exists"] = os.path.exists(os.path.join(run_dir, "config.yaml"))
    return out


result: dict = {"arms": {}}
for arm, run_dir in ARMS.items():
    if not run_dir or not os.path.isdir(run_dir):
        result["arms"][arm] = {"error": f"missing run dir: {run_dir!r}"}
        continue
    result["arms"][arm] = collect(run_dir)

os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(result, f, indent=1)
print("WROTE", OUT_JSON)
for arm, d in result["arms"].items():
    if "error" in d:
        print(arm, d["error"])
        continue
    ps = d.get("phase2_summary") or {}
    ev = ps.get("eval_metrics", {})
    print(
        arm,
        "n_docs=%s" % d["n_documents"],
        "mean_mse=%.8f" % (d["mean_mse_over_documents"] or float("nan")),
        "mass_on_S=%.5f" % (d.get("mass_on_S_mean_over_layers") or float("nan")),
        "vmax=%s" % (ps.get("value_norms", {}) or {}).get("global_max_abs"),
        "wall_s=%.1f" % (d.get("wall_clock_s") or float("nan")),
        "QA=%s" % (ev.get("qa_forgetting", {}) or {}).get("loss"),
        "MT=%s" % (ev.get("mt_acquisition", {}) or {}).get("loss"),
    )
