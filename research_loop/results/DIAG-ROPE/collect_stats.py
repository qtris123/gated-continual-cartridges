"""DIAG-ROPE: collect per-doc / per-layer solve diagnostics for both arms.

Reads the run dirs' `am_doc_*.pt` (AMUpdateStats with `mse_per_layer` and `extra`,
which carries `ref_mass_on_S_per_layer` + `v_selected_absmax_after_per_layer`) plus
`phase2_summary.json`, and writes one JSON with per-layer arrays for both arms and a
`baseline_reproduced` block for arm A.
"""

from __future__ import annotations

import json
import os
from glob import glob

import torch

BASELINE = {"qa_forgetting": 2.1766157150268555, "mt_acquisition": 2.5483615398406982}
OUT_JSON = os.environ["OUT_JSON"]
ARMS = {"theta1e4": os.environ.get("RUNDIR_A", ""), "theta5e6": os.environ.get("RUNDIR_B", "")}


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
        per_doc.append({
            "slug": blob["doc_record"]["slug"],
            "mean_mse": float(st.mean_mse),
            "n_queries": int(st.n_queries),
            "mse_per_layer": {int(k): float(v) for k, v in st.mse_per_layer.items()},
            "ref_mass_on_S_per_layer": {int(k): float(v) for k, v in mass.items()},
            "v_selected_absmax_after_per_layer": {int(k): float(v) for k, v in vmax.items()},
            "rope_theta_in_extra": extra.get("rope_theta"),
            "max_queries_per_head": extra.get("max_queries_per_head"),
            "n_queries_available_min": extra.get("n_queries_available_min"),
            "n_queries_available_max": extra.get("n_queries_available_max"),
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
    out["mean_mse_over_documents"] = (
        sum(d["mean_mse"] for d in per_doc) / n if n else None
    )
    out["mse_per_layer_mean_over_docs"] = {
        str(k): layer_mse_sum[k] / n for k in sorted(layer_mse_sum)
    } if n else {}
    out["mass_on_S_per_layer_mean_over_docs"] = {
        str(k): layer_mass_sum[k] / n for k in sorted(layer_mass_sum)
    } if n else {}
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
    return out


result = {"arms": {}}
for arm, run_dir in ARMS.items():
    if not run_dir or not os.path.isdir(run_dir):
        result["arms"][arm] = {"error": f"missing run dir: {run_dir!r}"}
        continue
    result["arms"][arm] = collect(run_dir)

a = result["arms"].get("theta1e4", {})
ev = (a.get("phase2_summary") or {}).get("eval_metrics", {})
result["baseline_reproduced"] = {
    "reference": BASELINE,
    "measured": {k: ev.get(k, {}).get("loss") for k in BASELINE},
    "delta": {
        k: (ev[k]["loss"] - BASELINE[k]) if k in ev else None for k in BASELINE
    },
    "bit_identical": all(
        k in ev and ev[k]["loss"] == BASELINE[k] for k in BASELINE
    ),
}

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(result, f, indent=1)
print("WROTE", OUT_JSON)
print("baseline_reproduced:", json.dumps(result["baseline_reproduced"]))
for arm, d in result["arms"].items():
    if "error" in d:
        print(arm, d["error"])
        continue
    print(
        arm,
        "mean_mse_over_docs=%.8f" % (d["mean_mse_over_documents"] or float("nan")),
        "mass_on_S_mean=%.5f" % (d.get("mass_on_S_mean_over_layers") or float("nan")),
        "vmax=%s" % ((d.get("phase2_summary") or {}).get("value_norms", {}).get("global_max_abs")),
        "wall_s=%.1f" % (d.get("wall_clock_s") or float("nan")),
    )
