"""MECH-BETA: collect per-doc / per-layer solve + beta diagnostics for all arms.

Adapted from `results/DIAG-ROPE/collect_stats.py`. Reads each arm's run dir
(`am_doc_*.pt` -> AMUpdateStats with `mse_per_layer` and `extra`, which now also
carries `beta` / `beta_per_layer`) plus `phase2_summary.json`, and writes one JSON
with per-layer arrays for every arm and a `baseline_reproduced` block for the
control.

env:  ARMS="label:run_dir,label:run_dir,..."   OUT_JSON=...
"""

from __future__ import annotations

import json
import os
from glob import glob

import torch

BASELINE = {"qa_forgetting": 2.1766157150268555, "mt_acquisition": 2.5483615398406982}
OUT_JSON = os.environ["OUT_JSON"]
ARMS_ENV = os.environ["ARMS"]
CONTROL = os.environ.get("CONTROL_ARM", "control")

ARMS = {}
for spec in ARMS_ENV.split(","):
    if not spec.strip():
        continue
    lbl, d = spec.split(":", 1)
    ARMS[lbl] = d


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
    layer_beta_min: dict[int, float] = {}
    layer_beta_max: dict[int, float] = {}
    layer_beta_med_sum: dict[int, float] = {}
    n = 0
    for p in docs:
        blob = torch.load(p, map_location="cpu", weights_only=False)
        st = blob["am_stats"]
        extra = dict(getattr(st, "extra", None) or {})
        mass = extra.get("ref_mass_on_S_per_layer", {}) or {}
        vmax = extra.get("v_selected_absmax_after_per_layer", {}) or {}
        beta = extra.get("beta")
        beta_pl = extra.get("beta_per_layer", {}) or {}
        per_doc.append({
            "slug": blob["doc_record"]["slug"],
            "mean_mse": float(st.mean_mse),
            "n_queries": int(st.n_queries),
            "mse_per_layer": {int(k): float(v) for k, v in st.mse_per_layer.items()},
            "ref_mass_on_S_per_layer": {int(k): float(v) for k, v in mass.items()},
            "v_selected_absmax_after_per_layer": {int(k): float(v) for k, v in vmax.items()},
            "rope_theta_in_extra": extra.get("rope_theta"),
            "max_queries_per_head": extra.get("max_queries_per_head"),
            "beta": beta,
            "beta_per_layer": {int(k): v for k, v in beta_pl.items()},
        })
        for k, v in st.mse_per_layer.items():
            layer_mse_sum[int(k)] = layer_mse_sum.get(int(k), 0.0) + float(v)
        for k, v in mass.items():
            layer_mass_sum[int(k)] = layer_mass_sum.get(int(k), 0.0) + float(v)
        for k, v in vmax.items():
            layer_vmax[int(k)] = max(layer_vmax.get(int(k), 0.0), float(v))
        for k, v in beta_pl.items():
            k = int(k)
            layer_beta_min[k] = min(layer_beta_min.get(k, 1e30), float(v["min"]))
            layer_beta_max[k] = max(layer_beta_max.get(k, -1e30), float(v["max"]))
            layer_beta_med_sum[k] = layer_beta_med_sum.get(k, 0.0) + float(v["median"])
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
    if layer_beta_min:
        out["beta_per_layer_over_docs"] = {
            str(k): {
                "min": layer_beta_min[k],
                "median_mean": layer_beta_med_sum[k] / n,
                "max": layer_beta_max[k],
            }
            for k in sorted(layer_beta_min)
        }
        meds = [v["median_mean"] for v in out["beta_per_layer_over_docs"].values()]
        out["beta_summary"] = {
            "min_over_layers": min(layer_beta_min.values()),
            "max_over_layers": max(layer_beta_max.values()),
            "median_mean_over_layers": sum(meds) / len(meds),
            "last_doc_beta": per_doc[-1]["beta"] if per_doc else None,
            "first_doc_beta": per_doc[0]["beta"] if per_doc else None,
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

a = result["arms"].get(CONTROL, {})
ev = (a.get("phase2_summary") or {}).get("eval_metrics", {})
result["baseline_reproduced"] = {
    "control_arm": CONTROL,
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
        "ref_mass_on_S_mean=%.5f" % (d.get("mass_on_S_mean_over_layers") or float("nan")),
        "vmax=%s" % ((d.get("phase2_summary") or {}).get("value_norms", {}).get("global_max_abs")),
        "wall_s=%.1f" % (d.get("wall_clock_s") or float("nan")),
        "beta=%s" % json.dumps(d.get("beta_summary", {}).get("last_doc_beta")),
    )
