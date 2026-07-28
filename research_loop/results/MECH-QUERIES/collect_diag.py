"""MECH-QUERIES (B-CASCADE) diagnostic collector — CPU only, reads artifacts.

For each n-arm run dir it collects, per layer:
  - max |v| over the whole trainable value tensor of the FINAL cache (the number
    ORACLE-WRITE reported as 984 for the solved write / 63 for the teacher)
  - ref_mass_on_S (attention mass the reference queries put on the written slots,
    computed inside the solve at value_solve.py; standing WORKERS.md requirement)
  - v_selected_absmax_after (the solved rows only)
plus the per-document mean_mse / timings and the reference-query supply ceiling.

Usage:  RUNS="64:<run_dir>,256:<run_dir>,..." OUT_JSON=<path> python collect_diag.py
"""

from __future__ import annotations

import json
import os
from glob import glob

import torch


RUNS = os.environ["RUNS"]
OUT_JSON = os.environ["OUT_JSON"]


def per_layer_value_absmax(cache_path: str) -> dict:
    blob = torch.load(cache_path, map_location="cpu", weights_only=False)
    # TrainableCache.save payload: find the trainable value tensors.
    if isinstance(blob, dict) and "state_dict" in blob:
        sd = blob["state_dict"]
    elif isinstance(blob, dict):
        sd = blob
    else:
        sd = blob.state_dict()
    out = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if "trainable_values" in k or ("values" in k and "frozen" not in k):
            # key looks like "trainable_values.<layer>" or similar
            tail = k.rsplit(".", 1)[-1]
            try:
                layer = int(tail)
            except ValueError:
                continue
            out[layer] = float(v.detach().float().abs().max().item())
    return out


def collect_run(run_dir: str) -> dict:
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
        pl = per_layer_value_absmax(ck)
        if pl:
            out["value_absmax_per_layer_final_cache"] = {
                l: pl[l] for l in sorted(pl)
            }
            out["value_global_max_abs"] = max(pl.values())
            out["value_absmax_argmax_layer"] = max(pl, key=pl.get)

    summ = os.path.join(run_dir, "phase2_summary.json")
    if os.path.exists(summ):
        out["phase2_summary"] = json.load(open(summ))

    agg_path = os.path.join(run_dir, "per_document_am_stats.pt")
    if os.path.exists(agg_path):
        agg = torch.load(agg_path, map_location="cpu", weights_only=False)
        out["wall_clock_s"] = agg.get("wall_clock_s")
    return out


def main():
    results = {}
    for spec in RUNS.split(","):
        label, run_dir = spec.split(":", 1)
        print(f"[collect] n={label} {run_dir}", flush=True)
        results[label] = collect_run(run_dir)
        r = results[label]
        print(
            f"  n_docs={r['n_documents']} vmax={r.get('value_global_max_abs')} "
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
