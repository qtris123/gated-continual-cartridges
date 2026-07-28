"""MECH-QUERIES-B (B-CASCADE) diagnostic collector — CPU only, reads artifacts.

Extends MECH-QUERIES' collect_diag.py with the |dV| measurement the task requires:
with DELTA_WEIGHT scaled to 2.56 the trust region may pull the solution back to
V_old, i.e. the write could become near-trivial. A "no change" result must not be
misread as "no effect of queries", so we measure how far the cache actually moved
from the Phase-1 cache it started from.

Per arm, per layer:
  - value_absmax                (final cache, the ORACLE-WRITE "984" number)
  - ref_mass_on_S               (solve-time, value_solve.py; standing requirement)
  - v_selected_absmax_after     (solved rows only)
  - dV_absmax / dV_rel_fro / n_rows_changed   (final cache vs the Phase-1 cache)
plus per-document mean_mse / timings and the reference-query supply ceiling.

Usage:  RUNS="lbl:<run_dir>,..." PHASE1=<cache.pt> OUT_JSON=<path> python collect_diag_b.py
"""

from __future__ import annotations

import json
import os
from glob import glob

import torch


RUNS = os.environ["RUNS"]
OUT_JSON = os.environ["OUT_JSON"]
PHASE1 = os.environ["PHASE1"]


def load_values(cache_path: str):
    blob = torch.load(cache_path, map_location="cpu", weights_only=False)
    tv = blob["trainable_values"] if isinstance(blob, dict) else blob.trainable_values
    return [v.detach().float() for v in tv]


def value_delta_stats(cache_path: str, base_vals) -> dict:
    vals = load_values(cache_path)
    per_layer = {}
    tot_num, tot_den, tot_rows = 0.0, 0.0, 0
    for l, (v, v0) in enumerate(zip(vals, base_vals)):
        d = (v - v0)
        # (n_heads, T, d) or (T, d); collapse everything but the slot axis
        flat_d = d.reshape(-1, d.shape[-2], d.shape[-1])
        row_norm = flat_d.norm(dim=-1)            # (heads, T)
        changed = int((row_norm > 0).any(dim=0).sum().item())
        num = float(d.norm().item())
        den = float(v0.norm().item())
        per_layer[l] = {
            "value_absmax": float(v.abs().max().item()),
            "dV_absmax": float(d.abs().max().item()),
            "dV_fro": num,
            "V_old_fro": den,
            "dV_rel_fro": num / den if den > 0 else 0.0,
            "n_slots_changed": changed,
        }
        tot_num += num ** 2
        tot_den += den ** 2
        tot_rows += changed
    return {
        "per_layer": per_layer,
        "value_global_max_abs": max(p["value_absmax"] for p in per_layer.values()),
        "value_absmax_argmax_layer": max(per_layer, key=lambda l: per_layer[l]["value_absmax"]),
        "dV_global_max_abs": max(p["dV_absmax"] for p in per_layer.values()),
        "dV_rel_fro_global": (tot_num ** 0.5) / (tot_den ** 0.5) if tot_den > 0 else 0.0,
        "dV_rel_fro_mean_over_layers": sum(p["dV_rel_fro"] for p in per_layer.values())
        / len(per_layer),
        "n_slots_changed_total": tot_rows,
    }


def collect_run(run_dir: str, base_vals) -> dict:
    docs = sorted(glob(os.path.join(run_dir, "am_doc_*.pt")))
    per_doc = []
    mass_acc: dict[int, list[float]] = {}
    vsel_acc: dict[int, list[float]] = {}
    avail_min, avail_max, used_min, used_max = [], [], [], []
    for f in docs:
        p = torch.load(f, map_location="cpu", weights_only=False)
        rec = p["doc_record"]
        st = p["am_stats"]
        e = dict(getattr(st, "extra", None) or {})
        for l, m in (e.get("ref_mass_on_S_per_layer") or {}).items():
            mass_acc.setdefault(int(l), []).append(float(m))
        for l, m in (e.get("v_selected_absmax_after_per_layer") or {}).items():
            vsel_acc.setdefault(int(l), []).append(float(m))
        if "n_queries_available_min" in e:
            avail_min.append(e["n_queries_available_min"])
            avail_max.append(e["n_queries_available_max"])
            used_min.append(e["n_queries_used_min"])
            used_max.append(e["n_queries_used_max"])
        per_doc.append(
            {
                "doc_index": rec.get("doc_index"),
                "slug": rec.get("slug"),
                "mean_mse": rec.get("mean_mse"),
                "n_queries": rec.get("n_queries"),
                "prefill_s": rec.get("timing_s", {}).get("prefill_s"),
                "total_s": rec.get("timing_s", {}).get("total_s"),
                "max_queries_per_head": e.get("max_queries_per_head"),
            }
        )

    n_layers = max(mass_acc) + 1 if mass_acc else 0
    out = {
        "run_dir": run_dir,
        "n_documents": len(docs),
        "per_document": per_doc,
        "mean_mse_last_doc": per_doc[-1]["mean_mse"] if per_doc else None,
        "mean_mse_over_docs": (
            sum(d["mean_mse"] for d in per_doc) / len(per_doc) if per_doc else None
        ),
        "n_queries_available_min": min(avail_min) if avail_min else None,
        "n_queries_available_max": max(avail_max) if avail_max else None,
        "n_queries_used_min": min(used_min) if used_min else None,
        "n_queries_used_max": max(used_max) if used_max else None,
        "ref_mass_on_S_per_layer_mean_over_docs": {
            l: sum(v) / len(v) for l, v in sorted(mass_acc.items())
        },
        "v_selected_absmax_after_per_layer_max_over_docs": {
            l: max(v) for l, v in sorted(vsel_acc.items())
        },
    }
    if mass_acc:
        vals = [out["ref_mass_on_S_per_layer_mean_over_docs"][l] for l in range(n_layers)]
        out["ref_mass_on_S_mean_over_layers"] = sum(vals) / len(vals)
        out["ref_mass_on_S_min"] = min(vals)
        out["ref_mass_on_S_max"] = max(vals)

    ck = os.path.join(run_dir, "cache_last.pt")
    if os.path.exists(ck):
        ds = value_delta_stats(ck, base_vals)
        out["value_absmax_per_layer_final_cache"] = {
            l: ds["per_layer"][l]["value_absmax"] for l in sorted(ds["per_layer"])
        }
        out["dV_vs_phase1_per_layer"] = ds["per_layer"]
        for k in (
            "value_global_max_abs",
            "value_absmax_argmax_layer",
            "dV_global_max_abs",
            "dV_rel_fro_global",
            "dV_rel_fro_mean_over_layers",
            "n_slots_changed_total",
        ):
            out[k] = ds[k]

    summ = os.path.join(run_dir, "phase2_summary.json")
    if os.path.exists(summ):
        out["phase2_summary"] = json.load(open(summ))

    agg_path = os.path.join(run_dir, "per_document_am_stats.pt")
    if os.path.exists(agg_path):
        agg = torch.load(agg_path, map_location="cpu", weights_only=False)
        out["wall_clock_s"] = agg.get("wall_clock_s")
    return out


def main():
    base_vals = load_values(PHASE1)
    print(f"[collect] phase1={PHASE1} n_layers={len(base_vals)}", flush=True)
    results = {"phase1_cache": PHASE1}
    for spec in RUNS.split(","):
        label, run_dir = spec.split(":", 1)
        print(f"[collect] {label} {run_dir}", flush=True)
        results[label] = collect_run(run_dir, base_vals)
        r = results[label]
        print(
            f"  n_docs={r['n_documents']} vmax={r.get('value_global_max_abs')} "
            f"dVmax={r.get('dV_global_max_abs')} dV_rel={r.get('dV_rel_fro_global')} "
            f"slots_changed={r.get('n_slots_changed_total')} "
            f"massS={r.get('ref_mass_on_S_mean_over_layers')} "
            f"avail=[{r['n_queries_available_min']},{r['n_queries_available_max']}] "
            f"used=[{r['n_queries_used_min']},{r['n_queries_used_max']}] "
            f"mse_last={r['mean_mse_last_doc']} wall={r.get('wall_clock_s')}",
            flush=True,
        )
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(results, open(OUT_JSON, "w"), indent=2)
    print(f"[done] wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
