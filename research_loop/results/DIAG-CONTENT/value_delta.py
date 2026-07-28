"""DIAG-CONTENT: how much did each arm's write actually displace the Phase-1 values?

Directly addresses the confound the brief names: "the Phase-1 cartridge was BUILT from
the QA corpus, so writing it back may be closer to a no-op than a random perturbation".
If arm B's write displaces a comparable fraction of the Phase-1 value tensor as the
canonical arm A, the write is not a no-op and the confound is weakened.

CPU-only tensor comparison of the final caches against the Phase-1 cache.
Convention copied from research_loop/results/MECH-QUERIES-B/collect_diag_b.py.

Env: CACHES="label:path,..."  PHASE1=<cache.pt>  OUT_JSON=<path>
"""

from __future__ import annotations

import json
import os

import torch

CACHES = os.environ["CACHES"]
PHASE1 = os.environ["PHASE1"]
OUT_JSON = os.environ["OUT_JSON"]


def load_values(path: str):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    tv = blob["trainable_values"] if isinstance(blob, dict) else blob.trainable_values
    return [v.detach().float() for v in tv]


base = load_values(PHASE1)

out = {"phase1": PHASE1, "arms": {}}
for spec in CACHES.split(","):
    label, path = spec.split(":", 1)
    if not path or not os.path.exists(path):
        out["arms"][label] = {"error": f"missing cache: {path!r}"}
        continue
    vals = load_values(path)
    per_layer = {}
    tot_num = tot_den = 0.0
    tot_rows = 0
    for l, (v, v0) in enumerate(zip(vals, base)):
        d = v - v0
        flat_d = d.reshape(-1, d.shape[-2], d.shape[-1])
        row_norm = flat_d.norm(dim=-1)
        changed = int((row_norm > 0).any(dim=0).sum().item())
        num = float(d.norm().item())
        den = float(v0.norm().item())
        per_layer[l] = {
            "value_absmax": float(v.abs().max().item()),
            "dV_absmax": float(d.abs().max().item()),
            "dV_rel_fro": num / den if den > 0 else 0.0,
            "n_slots_changed": changed,
        }
        tot_num += num ** 2
        tot_den += den ** 2
        tot_rows += changed
    out["arms"][label] = {
        "cache_path": path,
        "value_global_max_abs": max(p["value_absmax"] for p in per_layer.values()),
        "dV_global_max_abs": max(p["dV_absmax"] for p in per_layer.values()),
        "dV_rel_fro_global": (tot_num ** 0.5) / (tot_den ** 0.5) if tot_den > 0 else 0.0,
        "dV_rel_fro_mean_over_layers": sum(p["dV_rel_fro"] for p in per_layer.values()) / len(per_layer),
        "n_slots_changed_total": tot_rows,
        "n_layers": len(per_layer),
        "per_layer": per_layer,
    }
    print(
        label,
        "dV_rel_fro_global=%.4f" % out["arms"][label]["dV_rel_fro_global"],
        "n_slots_changed=%d" % tot_rows,
        "|v|max=%.1f" % out["arms"][label]["value_global_max_abs"],
        "|dV|max=%.1f" % out["arms"][label]["dV_global_max_abs"],
    )

os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
json.dump(out, open(OUT_JSON, "w"), indent=1)
print("WROTE", OUT_JSON)
