"""MECH-SEQUENTIAL: per-doc / per-layer solve + on-policy diagnostics per arm.

Adapted from `results/MECH-KEYS/collect_stats.py`, plus the LIT-006 on-policy
block: `extra["onpolicy"]` / `extra["onpolicy_events"]` carry, per group boundary,
the **query drift** (cosine similarity and relative L2 between the stale queries
that layer would have been solved against and the on-policy ones) and the cost of
each re-extraction.

The control baseline here is **MECH-KEYS' control arm** (theta = 5e6,
KEY_MODE=freeze, top_t=32).

env:  ARMS="label:run_dir,label:run_dir,..."   OUT_JSON=...   CONTROL_ARM=control
"""

from __future__ import annotations

import json
import os
from glob import glob

import torch

# MECH-KEYS control arm in-run losses (theta=5e6, KEY_MODE=freeze, top_t=32)
# == DIAG-ROPE arm B. Standalone eval of the same checkpoint: QA 2.159724712371826
# / MT 2.529625177383423, which is the "QA 2.15972 / MT 2.52963" gate in the brief.
BASELINE = {"qa_forgetting": 2.15320086479187, "mt_acquisition": 2.5296061038970947}
BASELINE_STANDALONE = {
    "qa_forgetting": 2.159724712371826,
    "mt_acquisition": 2.529625177383423,
}
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
    layer_kw: dict[int, dict[str, float]] = {}
    onpolicy_docs: list[dict] = []
    onpolicy_by_layer: dict[int, dict[str, float]] = {}
    n = 0
    for p in docs:
        blob = torch.load(p, map_location="cpu", weights_only=False)
        st = blob["am_stats"]
        extra = dict(getattr(st, "extra", None) or {})
        mass = extra.get("ref_mass_on_S_per_layer", {}) or {}
        vmax = extra.get("v_selected_absmax_after_per_layer", {}) or {}
        kw = extra.get("key_rewrite")
        kw_pl = extra.get("key_rewrite_per_layer", {}) or {}
        op = extra.get("onpolicy")
        op_events = extra.get("onpolicy_events") or []
        if op is not None:
            onpolicy_docs.append(op)
            for e in op_events:
                d = onpolicy_by_layer.setdefault(
                    int(e["layer"]),
                    {"cos_sum": 0.0, "cos_min": 1.0, "rel_l2_sum": 0.0,
                     "refresh_s_sum": 0.0, "n": 0},
                )
                d["cos_sum"] += float(e["query_cos_mean"])
                d["cos_min"] = min(d["cos_min"], float(e["query_cos_min"]))
                d["rel_l2_sum"] += float(e["query_rel_l2"])
                d["refresh_s_sum"] += float(e["refresh_s"])
                d["n"] += 1
        per_doc.append({
            "onpolicy": op,
            "onpolicy_events": op_events,
            "slug": blob["doc_record"]["slug"],
            "mean_mse": float(st.mean_mse),
            "n_queries": int(st.n_queries),
            "mse_per_layer": {int(k): float(v) for k, v in st.mse_per_layer.items()},
            "ref_mass_on_S_per_layer": {int(k): float(v) for k, v in mass.items()},
            "v_selected_absmax_after_per_layer": {int(k): float(v) for k, v in vmax.items()},
            "rope_theta_in_extra": extra.get("rope_theta"),
            "max_queries_per_head": extra.get("max_queries_per_head"),
            "key_rewrite": kw,
            "key_rewrite_per_layer": {int(k): v for k, v in kw_pl.items()},
        })
        for k, v in st.mse_per_layer.items():
            layer_mse_sum[int(k)] = layer_mse_sum.get(int(k), 0.0) + float(v)
        for k, v in mass.items():
            layer_mass_sum[int(k)] = layer_mass_sum.get(int(k), 0.0) + float(v)
        for k, v in vmax.items():
            layer_vmax[int(k)] = max(layer_vmax.get(int(k), 0.0), float(v))
        for k, v in kw_pl.items():
            k = int(k)
            acc = layer_kw.setdefault(
                k, {"n_selected": 0.0, "n_from_doc": 0.0, "n_changed": 0.0,
                    "n_repositioned": 0.0, "n_heads": 0.0}
            )
            for f in acc:
                acc[f] += float(v.get(f, 0))
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
    if layer_kw:
        # per layer, summed over the 8 KV heads and all 16 documents
        out["key_rewrite_per_layer_total"] = {
            str(k): {f: int(v) for f, v in layer_kw[k].items()} for k in sorted(layer_kw)
        }
        out["key_rewrite_summary"] = {
            "mode": per_doc[-1]["key_rewrite"]["mode"],
            "reposition": per_doc[-1]["key_rewrite"]["reposition"],
            "rope_delta_last_doc": per_doc[-1]["key_rewrite"].get("rope_delta"),
            "n_selected_total": sum(d["key_rewrite"]["n_selected_total"] for d in per_doc),
            "n_from_doc_total": sum(d["key_rewrite"]["n_from_doc_total"] for d in per_doc),
            "n_from_cartridge_total": sum(
                d["key_rewrite"]["n_from_cartridge_total"] for d in per_doc
            ),
            "n_changed_total": sum(d["key_rewrite"]["n_changed_total"] for d in per_doc),
            "n_repositioned_total": sum(
                d["key_rewrite"]["n_repositioned_total"] for d in per_doc
            ),
            "doc_fraction_of_support": (
                sum(d["key_rewrite"]["n_from_doc_total"] for d in per_doc)
                / max(sum(d["key_rewrite"]["n_selected_total"] for d in per_doc), 1)
            ),
            "per_document_n_from_doc": [
                d["key_rewrite"]["n_from_doc_total"] for d in per_doc
            ],
            "keys_rewritten_per_layer_per_doc_mean": {
                str(k): layer_kw[k]["n_from_doc"] / max(n, 1) for k in sorted(layer_kw)
            },
        }
    if onpolicy_docs:
        # LIT-006 / MECH-006. `query_cos` is the direct measurement of the
        # cross-layer activation shift: 1.0 would mean the write did not move the
        # queries at all and on-policy re-extraction is vacuous.
        out["onpolicy_summary"] = {
            "group_size": onpolicy_docs[-1]["group_size"],
            "doc_kv_refreshed": onpolicy_docs[-1]["doc_kv_refreshed"],
            "n_refreshes_per_doc": onpolicy_docs[-1]["n_refreshes"],
            "refresh_layers": onpolicy_docs[-1]["refresh_layers"],
            "n_docs_with_onpolicy": len(onpolicy_docs),
            "refresh_s_total_all_docs": sum(
                d["refresh_s_total"] for d in onpolicy_docs
            ),
            "query_cos_mean_over_docs": sum(
                d["query_cos_mean"] for d in onpolicy_docs
            ) / len(onpolicy_docs),
            "query_cos_min_over_docs": min(d["query_cos_min"] for d in onpolicy_docs),
            "query_rel_l2_mean_over_docs": sum(
                d["query_rel_l2_mean"] for d in onpolicy_docs
            ) / len(onpolicy_docs),
            "query_rel_l2_max_over_docs": max(
                d["query_rel_l2_max"] for d in onpolicy_docs
            ),
            "per_document_query_cos_mean": [
                d["query_cos_mean"] for d in onpolicy_docs
            ],
            "per_document_refresh_s": [d["refresh_s_total"] for d in onpolicy_docs],
        }
        out["onpolicy_drift_per_layer"] = {
            str(l): {
                "query_cos_mean": v["cos_sum"] / max(v["n"], 1),
                "query_cos_min": v["cos_min"],
                "query_rel_l2_mean": v["rel_l2_sum"] / max(v["n"], 1),
                "refresh_s_mean": v["refresh_s_sum"] / max(v["n"], 1),
                "n_docs": v["n"],
            }
            for l, v in sorted(onpolicy_by_layer.items())
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
        # solve_s == the closed-form write time == total per-doc time minus prefill
        # minus reference-query collection, as recorded by `continual.py`.
        out["timing_per_document"] = [
            d.get("timing_s", {}) for d in agg.get("per_document", [])
        ]
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
    "reference_standalone": BASELINE_STANDALONE,
    "reference_source": (
        "MECH-KEYS control == DIAG-ROPE arm B (theta=5e6, KEY_MODE=freeze, "
        "top_t=32), in-run; standalone eval of the same checkpoint in "
        "reference_standalone"
    ),
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
        "keyrw=%s" % json.dumps(d.get("key_rewrite_summary", {}).get("doc_fraction_of_support")),
    )
