"""ORACLE-WRITE-512: collect per-document + per-layer diagnostics from the two run dirs.

Read-only. Writes research_loop/state/diagnostics/ORACLE-WRITE-512.json.
Env: ORAC_DIR, CTRL_DIR, ROUTE_JSON, OUT_JSON, SNAP_PATH, IMPORT_PATH, SNAP_HEAD, SNAP_MANIFEST
"""

from __future__ import annotations

import json
import os
from glob import glob

import torch


def per_doc(run_dir: str) -> list[dict]:
    out = []
    for f in sorted(glob(os.path.join(run_dir, "am_doc_*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        rec = d["doc_record"]
        extra = rec.get("extra") or {}
        mask = d["am_stats"].mask
        n_slots = {
            int(l): int(p.flatten().numel()) for l, p in mask.positions_per_layer.items()
        }
        out.append(
            {
                "doc_index": rec["doc_index"],
                "slug": rec["slug"],
                "n_conversations": rec["n_conversations"],
                "n_ref_batches": rec["n_ref_batches"],
                "mean_mse": rec["mean_mse"],
                "n_queries": rec["n_queries"],
                "timing_s": rec["timing_s"],
                "n_written_min": extra.get("oracle_n_written_min"),
                "n_written_max": extra.get("oracle_n_written_max"),
                "oracle_write": extra.get("oracle_write", False),
                "n_selected_slots_per_layer_min": min(n_slots.values()),
                "n_selected_slots_per_layer_max": max(n_slots.values()),
                "ref_mass_on_S_per_layer": extra.get("oracle_ref_mass_on_S_per_layer"),
                "teacher_doc_mass_per_layer": extra.get("oracle_teacher_doc_mass_per_layer"),
            }
        )
    return out


def slot_union(run_dir: str) -> dict[int, int]:
    from collections import defaultdict

    per_layer = defaultdict(set)
    for f in sorted(glob(os.path.join(run_dir, "am_doc_*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        mask = d["am_stats"].mask
        for l, p in mask.positions_per_layer.items():
            per_layer[int(l)].update(p.flatten().tolist())
    return {l: len(v) for l, v in sorted(per_layer.items())}


def summary(run_dir: str) -> dict:
    p = os.path.join(run_dir, "phase2_summary.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def main():
    orac, ctrl = os.environ["ORAC_DIR"], os.environ["CTRL_DIR"]
    route = {}
    rp = os.environ.get("ROUTE_JSON")
    if rp and os.path.exists(rp):
        route = json.load(open(rp))

    doc_o, doc_c = per_doc(orac), per_doc(ctrl)
    out = {
        "id": "ORACLE-WRITE-512",
        "board_entry": "B-ROUTE",
        "role": "measure",
        "question": (
            "Is value-only writing capped by BANDWIDTH or by something deeper? Read the "
            "write-ceiling oracle at full support (top_t=512 -> all 511 trainable slots, "
            "mass_on_S ~0.57-0.59) instead of the tf-idf top-32 union (mass_on_S ~0.08-0.09)."
        ),
        "provenance": {
            "snapshot_path": os.environ.get("SNAP_PATH"),
            "snapshot_head": os.environ.get("SNAP_HEAD"),
            "snapshot_manifest_sha256_before": os.environ.get("SNAP_MANIFEST_BEFORE"),
            "snapshot_manifest_sha256_after": os.environ.get("SNAP_MANIFEST_AFTER"),
            "verified_import_path": os.environ.get("IMPORT_PATH"),
            "oracle_run_dir": orac,
            "control_run_dir": ctrl,
        },
        "config": {
            "TOP_T": 512,
            "effective_slots_per_layer": doc_o[0]["n_selected_slots_per_layer_max"] if doc_o else None,
            "GRANULARITY": "per_layer",
            "SLOT_SELECTION": "tfidf",
            "USE_IDF": 0,
            "KEY_MODE": "freeze",
            "TARGET_MODE": "cartridge_plus_doc",
            "RIDGE_LAMBDA": 1e-4,
            "RIDGE_SCALE": "spectral",
            "DELTA_WEIGHT": 1e-2,
            "AM_EXECUTION_MODE": "per_document",
            "AM_ORACLE_WRITE_ASSIGN": "mass_ranked",
            "MAX_REF_EXAMPLES_PER_DOC": 32,
            "max_queries_per_head": 64,
            "MODEL_NAME": "Qwen/Qwen3-4B-Instruct-2507",
            "NUM_TOKENS": 512,
            "PHASE1_CACHE_PATH": "outputs/phase1_selfdistill_qwen512/cache_last.pt",
            "SYNTH_DATA_PATH": "data/qasper/train/qwen_qasper_MT_task_8192.parquet",
            "gradient_steps": 0,
        },
        "wandb_run_url": os.environ.get("WANDB_URL_ORACLE"),
        "wandb_run_url_control": os.environ.get("WANDB_URL_CONTROL"),
        "per_document": {"oracle": doc_o, "control": doc_c},
        "slot_union_size_per_layer": {
            "oracle": slot_union(orac),
            "control": slot_union(ctrl),
        },
        "phase2_summary": {"oracle": summary(orac), "control": summary(ctrl)},
        "eval_time_route_mass": route,
    }

    # compact roll-ups
    nw = [d["n_written_max"] for d in doc_o if d["n_written_max"] is not None]
    out["rollup"] = {
        "n_written_per_doc": {d["slug"]: d["n_written_max"] for d in doc_o},
        "n_written_min_over_docs": min(nw) if nw else None,
        "n_written_max_over_docs": max(nw) if nw else None,
        "n_written_equals_support_for_all_docs": all(
            d["n_written_min"] == d["n_written_max"] == d["n_selected_slots_per_layer_max"]
            for d in doc_o
        )
        if doc_o
        else None,
        "mean_mse_per_doc_oracle": [d["mean_mse"] for d in doc_o],
        "mean_mse_per_doc_control": [d["mean_mse"] for d in doc_c],
        "mean_mse_mean_over_docs_oracle": (
            sum(d["mean_mse"] for d in doc_o) / len(doc_o) if doc_o else None
        ),
        "mean_mse_mean_over_docs_control": (
            sum(d["mean_mse"] for d in doc_c) / len(doc_c) if doc_c else None
        ),
    }
    if route:
        rs = {k: v["summary"] for k, v in route.get("runs", {}).items()}
        out["rollup"]["route_mass_summary"] = rs
        out["per_layer_arrays"] = {
            k: {
                "mass_on_S": [v["per_layer"][str(l)]["mass_on_S"] for l in range(36)],
                "mass_on_cart": [v["per_layer"][str(l)]["mass_on_cart"] for l in range(36)],
                "share_of_cart": [v["per_layer"][str(l)]["share_of_cart"] for l in range(36)],
                "mass_on_S_scored": [
                    v["per_layer"][str(l)]["mass_on_S_scored"] for l in range(36)
                ],
                "n_slots_per_layer": [v["n_slots_per_layer"][str(l)] for l in range(36)],
            }
            for k, v in route.get("runs", {}).items()
        }
    out["measurements"] = out["rollup"]

    op = os.environ["OUT_JSON"]
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["rollup"], indent=2)[:4000])
    print(f"[done] wrote {op}")


if __name__ == "__main__":
    main()
