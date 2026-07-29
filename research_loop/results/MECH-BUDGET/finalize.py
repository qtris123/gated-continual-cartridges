"""MECH-BUDGET (W4 TEST, B-GATE): build the result bundle.

Reads what `launch_mech_budget.sh` produced:
  curve.tsv          arm/top_t/k/split -> Eval loss + ckpt + sha + wandb url
  rundirs.tsv        arm/top_t/run_dir/rc/phase2_e2e_s/wandb url
  route_mass.json    eval-time mass_on_S / mass_on_cart / MT-QA ratio per arm
  <run_dir>/am_doc_*.pt, phase2_summary.json, per_document_am_stats.pt

Writes:
  research_loop/results/MECH-BUDGET/result.json          (WORKERS.md contract)
  research_loop/state/diagnostics/MECH-BUDGET.json       (per-arm k-curves, cost, raw)

Nothing here touches cartridges/ or examples/.
"""

from __future__ import annotations

import json
import os
import subprocess
from glob import glob

import torch

REPO = "/localhome/local-triv/gated-continual-cartridges_explore"
RESDIR = os.path.join(REPO, "research_loop/results/MECH-BUDGET")
DIAGDIR = os.path.join(REPO, "research_loop/state/diagnostics")

# DIAG-KEYCURVE reference for the TOP_T=32 gate (standalone eval of the same config)
GATE_REF = {
    (12, "QA"): 1.9560121297836304,
    (12, "MT"): 2.2720184326171875,
    (16, "QA"): 2.034916400909424,
    (16, "MT"): 2.330503463745117,
}
# DIAG-NOISE measured paired resolution (95%)
RES = {"MT": 0.0494, "QA": 0.0539}
RES_COMPOSITE = {"MT": 0.1055, "QA": 0.197}  # conservative composite incl. DIAG-PERDOC term

ARM_ORDER = ["A_t32", "B_t64", "C_t128"]
ARM_TOPT = {"A_t32": 32, "B_t64": 64, "C_t128": 128}


def read_tsv(path):
    if not os.path.exists(path):
        return []
    lines = open(path).read().splitlines()
    if not lines:
        return []
    hdr = lines[0].split("\t")
    return [dict(zip(hdr, l.split("\t"))) for l in lines[1:] if l.strip()]


def collect_run(run_dir: str) -> dict:
    """Per-arm internals: mean_mse, ref mass_on_S per layer, |v|max, timings."""
    out: dict = {"run_dir": run_dir}
    sp = os.path.join(run_dir, "phase2_summary.json")
    if os.path.exists(sp):
        out["phase2_summary"] = json.load(open(sp))

    docs = sorted(glob(os.path.join(run_dir, "am_doc_*.pt")))
    per_doc, layer_mass, layer_mse, layer_vmax = [], {}, {}, {}
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
            "timing": dict(getattr(st, "timing", None) or {}),
            "max_queries_per_head": extra.get("max_queries_per_head"),
            "n_queries_available_min": extra.get("n_queries_available_min"),
            "n_queries_available_max": extra.get("n_queries_available_max"),
            "n_queries_used_min": extra.get("n_queries_used_min"),
            "n_queries_used_max": extra.get("n_queries_used_max"),
            "rope_theta_in_extra": extra.get("rope_theta"),
            "key_rewrite": extra.get("key_rewrite"),
        })
        for k, v in st.mse_per_layer.items():
            layer_mse[int(k)] = layer_mse.get(int(k), 0.0) + float(v)
        for k, v in mass.items():
            layer_mass[int(k)] = layer_mass.get(int(k), 0.0) + float(v)
        for k, v in vmax.items():
            layer_vmax[int(k)] = max(layer_vmax.get(int(k), 0.0), float(v))
        n += 1

    out["n_documents"] = n
    out["per_document"] = per_doc
    out["mean_mse_over_documents"] = (sum(d["mean_mse"] for d in per_doc) / n) if n else None
    out["mse_per_layer_mean_over_docs"] = {str(k): layer_mse[k] / n for k in sorted(layer_mse)} if n else {}
    out["ref_mass_on_S_per_layer_mean_over_docs"] = (
        {str(k): layer_mass[k] / n for k in sorted(layer_mass)} if n else {}
    )
    out["v_selected_absmax_per_layer_max_over_docs"] = {
        str(k): layer_vmax[k] for k in sorted(layer_vmax)
    }
    if out["ref_mass_on_S_per_layer_mean_over_docs"]:
        vals = list(out["ref_mass_on_S_per_layer_mean_over_docs"].values())
        out["ref_mass_on_S_mean_over_layers"] = sum(vals) / len(vals)
        out["ref_mass_on_S_min_max_over_layers"] = [min(vals), max(vals)]

    agg_path = os.path.join(run_dir, "per_document_am_stats.pt")
    if os.path.exists(agg_path):
        agg = torch.load(agg_path, map_location="cpu", weights_only=False)
        out["wall_clock_s"] = agg.get("wall_clock_s")
        pd_ = agg.get("per_document", [])
        out["prefill_s_total"] = sum(d.get("timing_s", {}).get("prefill_s", 0.0) for d in pd_)
        out["doc_total_s_sum"] = sum(d.get("timing_s", {}).get("total_s", 0.0) for d in pd_)
        out["timing_per_document"] = [d.get("timing_s", {}) for d in pd_]
    # closed-form solve time as recorded by finetune.py's own timer, summed over docs
    out["am_solve_s_sum"] = sum(d["timing"].get("solve_s", 0.0) for d in per_doc) or None
    out["am_rank_s_sum"] = sum(d["timing"].get("rank_s", 0.0) for d in per_doc) or None
    out["am_collect_s_sum"] = sum(d["timing"].get("collect_s", 0.0) for d in per_doc) or None
    return out


def main():
    curve = read_tsv(os.path.join(RESDIR, "curve.tsv"))
    rundirs = read_tsv(os.path.join(RESDIR, "rundirs.tsv"))

    rd = {r["arm"]: r for r in rundirs}
    losses: dict = {}
    ckpt_meta: dict = {}
    eval_urls: dict = {}
    for r in curve:
        arm, k, sp = r["arm"], int(r["k"]), r["split"]
        try:
            losses.setdefault(arm, {}).setdefault(sp, {})[k] = float(r["loss"])
        except ValueError:
            losses.setdefault(arm, {}).setdefault(sp, {})[k] = None
        ckpt_meta.setdefault(arm, {})[(k, sp)] = {"ckpt": r["ckpt"], "sha256_16": r.get("ckpt_sha256")}
        eval_urls.setdefault(arm, {})[f"k{k}_{sp}"] = r.get("wandb_url")

    route = {}
    rp = os.path.join(RESDIR, "route_mass.json")
    if os.path.exists(rp):
        route = json.load(open(rp))

    arms_out: dict = {}
    for arm in ARM_ORDER:
        if arm not in rd:
            continue
        run_dir = rd[arm]["run_dir"]
        info = collect_run(run_dir) if run_dir and os.path.isdir(run_dir) else {"error": "no run dir"}
        ev = (info.get("phase2_summary") or {}).get("eval_metrics", {})
        lbl = f"t{ARM_TOPT[arm]}"
        arms_out[arm] = {
            "top_t": ARM_TOPT[arm],
            "slot_selection": "tfidf",
            "run_dir": run_dir,
            "rc": int(rd[arm]["rc"]),
            "wandb_run_url": rd[arm].get("wandb_url"),
            "wandb_run_id": (rd[arm].get("wandb_url") or "").rsplit("/", 1)[-1] or None,
            "k_curve": {
                "QA": losses.get(arm, {}).get("QA", {}),
                "MT": losses.get(arm, {}).get("MT", {}),
            },
            "checkpoints": {f"k{k}_{sp}": v for (k, sp), v in ckpt_meta.get(arm, {}).items()},
            "eval_wandb_urls": eval_urls.get(arm, {}),
            "in_run_eval_k16": {
                "qa_forgetting": ev.get("qa_forgetting", {}).get("loss"),
                "mt_acquisition": ev.get("mt_acquisition", {}).get("loss"),
            },
            "cost": {
                "phase2_e2e_s": int(rd[arm]["phase2_e2e_s"]),
                "solve_s": info.get("doc_total_s_sum"),
                "am_solve_s_sum": info.get("am_solve_s_sum"),
                "am_rank_s_sum": info.get("am_rank_s_sum"),
                "am_collect_s_sum": info.get("am_collect_s_sum"),
                "prefill_s_total": info.get("prefill_s_total"),
                "wall_clock_s": info.get("wall_clock_s"),
                "gradient_steps": 0,
            },
            "am_mean_mse": info.get("mean_mse_over_documents"),
            "am_mean_mse_last_doc": (info.get("phase2_summary") or {}).get("mean_mse_last_doc"),
            "v_absmax_global": ((info.get("phase2_summary") or {}).get("value_norms") or {}).get("global_max_abs"),
            "ref_mass_on_S_mean_over_layers": info.get("ref_mass_on_S_mean_over_layers"),
            "ref_mass_on_S_min_max_over_layers": info.get("ref_mass_on_S_min_max_over_layers"),
            "n_queries_per_head_cap": (info.get("per_document") or [{}])[0].get("max_queries_per_head"),
            "n_queries_available_min": (info.get("per_document") or [{}])[0].get("n_queries_available_min"),
            "n_queries_available_max": (info.get("per_document") or [{}])[0].get("n_queries_available_max"),
            "_raw": info,
        }
        # eval-time routing mass
        rr = (route.get("ratios") or {}).get(lbl)
        if rr:
            arms_out[arm]["eval_mass_on_S"] = {
                "MT": rr["mass_on_S_MT"],
                "QA": rr["mass_on_S_QA"],
                "mt_over_qa_ratio": rr["mt_over_qa_mass_on_S"],
                "mass_on_cart_MT": rr["mass_on_cart_MT"],
                "mass_on_cart_QA": rr["mass_on_cart_QA"],
            }
        for sp in ("QA", "MT"):
            run = (route.get("runs") or {}).get(f"{lbl}|{sp}")
            if run:
                arms_out[arm].setdefault("eval_mass_per_layer", {})[sp] = run["per_layer"]
                arms_out[arm].setdefault("eval_mass_summary", {})[sp] = run["summary"]
                arms_out[arm]["n_slots_union_per_layer"] = run["n_slots_per_layer"]

    # ---- gate ---------------------------------------------------------------
    gate_measured, gate_delta = {}, {}
    for (k, sp), want in GATE_REF.items():
        got = losses.get("A_t32", {}).get(sp, {}).get(k)
        gate_measured[f"k{k}_{sp}"] = got
        gate_delta[f"k{k}_{sp}"] = (got - want) if got is not None else None
    baseline_reproduced = {
        "arm": "A_t32",
        "reference": {f"k{k}_{sp}": v for (k, sp), v in GATE_REF.items()},
        "reference_source": "DIAG-KEYCURVE standalone eval, same config, TOP_T=32",
        "measured": gate_measured,
        "delta": gate_delta,
        "exact_to_1e-6": all(
            d is not None and abs(d) <= 1e-6 for d in gate_delta.values()
        ),
    }

    # ---- deltas vs the incumbent top_t=32, per k, per split ------------------
    deltas = {}
    for arm in ("B_t64", "C_t128"):
        if arm not in arms_out:
            continue
        d = {}
        for sp in ("QA", "MT"):
            for k in (8, 10, 12, 16):
                a = losses.get(arm, {}).get(sp, {}).get(k)
                b = losses.get("A_t32", {}).get(sp, {}).get(k)
                if a is None or b is None:
                    continue
                dv = a - b
                d[f"k{k}_{sp}"] = {
                    "delta_vs_t32": dv,
                    "clears_paired_resolution": abs(dv) > RES[sp],
                    "x_resolution": abs(dv) / RES[sp],
                    "clears_conservative_composite": abs(dv) > RES_COMPOSITE[sp],
                }
        # own-optimum (argmin over the measured k grid) comparison
        for sp in ("QA", "MT"):
            ca = {k: v for k, v in losses.get(arm, {}).get(sp, {}).items() if v is not None}
            cb = {k: v for k, v in losses.get("A_t32", {}).get(sp, {}).items() if v is not None}
            if ca and cb:
                ka, kb = min(ca, key=ca.get), min(cb, key=cb.get)
                dv = ca[ka] - cb[kb]
                d[f"own_optimum_{sp}"] = {
                    "arm_argmin_k": ka, "arm_min": ca[ka],
                    "t32_argmin_k": kb, "t32_min": cb[kb],
                    "delta_vs_t32": dv,
                    "clears_paired_resolution": abs(dv) > RES[sp],
                    "x_resolution": abs(dv) / RES[sp],
                }
        deltas[arm] = d

    # cost scaling
    cost_scaling = {}
    base = arms_out.get("A_t32", {}).get("cost", {})
    for arm in ARM_ORDER:
        if arm not in arms_out:
            continue
        c = arms_out[arm]["cost"]
        cost_scaling[arm] = {
            "top_t": ARM_TOPT[arm],
            "phase2_e2e_s": c["phase2_e2e_s"],
            "solve_s": c["solve_s"],
            "am_solve_s_sum": c["am_solve_s_sum"],
            "e2e_multiple_vs_t32": (c["phase2_e2e_s"] / base["phase2_e2e_s"]) if base.get("phase2_e2e_s") else None,
            "solve_multiple_vs_t32": (c["solve_s"] / base["solve_s"]) if base.get("solve_s") else None,
            "am_solve_multiple_vs_t32": (
                (c["am_solve_s_sum"] / base["am_solve_s_sum"]) if base.get("am_solve_s_sum") else None
            ),
        }

    targets = {
        "target_qa_le": 2.52, "target_mt_le": 2.02,
        "best_gradient_free": {"QA": 1.9560, "MT": 2.2720, "where": "tfidf@32, k=12"},
        "seed_mean_best": {"QA": 1.9468, "MT": 2.2681},
        "the_bar_dense": {"QA": 2.3721, "MT": 1.8725},
        "phase1_floor": {"QA": 2.2388, "MT": 3.7825},
    }
    for arm in ARM_ORDER:
        if arm not in arms_out:
            continue
        mt = {k: v for k, v in losses.get(arm, {}).get("MT", {}).items() if v is not None}
        qa = {k: v for k, v in losses.get(arm, {}).get("QA", {}).items() if v is not None}
        if mt and qa:
            kmt = min(mt, key=mt.get)
            arms_out[arm]["pass_check"] = {
                "best_MT": mt[kmt], "best_MT_k": kmt,
                "QA_at_best_MT_k": qa.get(kmt),
                "meets_MT_target": mt[kmt] <= targets["target_mt_le"],
                "meets_QA_target": (qa.get(kmt) is not None and qa[kmt] <= targets["target_qa_le"]),
                "MT_shortfall": mt[kmt] - targets["target_mt_le"],
            }

    head_after = subprocess.run(
        ["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    wl = open(os.path.join(RESDIR, "wrapper.log")).read() if os.path.exists(os.path.join(RESDIR, "wrapper.log")) else ""
    def _grab(key):
        for line in wl.splitlines():
            if line.startswith(key + "="):
                return line.split("=", 1)[1]
        return None

    provenance = {
        "snapshot_path": "/tmp/amsnap_budget",
        "snapshot_source": "git archive HEAD cartridges examples",
        "repo_head_at_launch": _grab("REPO_HEAD_AT_LAUNCH"),
        "repo_head_after": _grab("REPO_HEAD_AFTER") or head_after,
        "snapshot_manifest_at_launch": _grab("SNAPSHOT_MANIFEST_AT_LAUNCH"),
        "snapshot_manifest_after": _grab("SNAPSHOT_MANIFEST_AFTER"),
        "snapshot_manifest_stable": _grab("SNAPSHOT_MANIFEST_STABLE"),
        "import_probe_from_tmp": _grab("IMPORT_PROBE"),
        "gpu": _grab("MB_CLAIMED_GPU"),
        "concurrent_worker_edits_at_launch": open(
            os.path.join(RESDIR, "git_status_at_launch.txt")
        ).read().split() if os.path.exists(os.path.join(RESDIR, "git_status_at_launch.txt")) else None,
        "snapshot_has_MECH008_redundancy_selector": False,
        "worker_edited_no_source_file": True,
    }

    diag = {
        "id": "MECH-BUDGET",
        "board_entry": "B-GATE",
        "role": "test",
        "variable_under_test": "TOP_T in {32, 64, 128} with SLOT_SELECTION=tfidf (the incumbent)",
        "paired_resolution_used": RES,
        "conservative_composite": RES_COMPOSITE,
        "arms": arms_out,
        "k_curves": {a: arms_out[a]["k_curve"] for a in arms_out},
        "deltas_vs_t32": deltas,
        "cost_scaling": cost_scaling,
        "baseline_reproduced": baseline_reproduced,
        "targets": targets,
        "provenance": provenance,
        "route_mass_raw_path": rp if os.path.exists(rp) else None,
    }
    os.makedirs(DIAGDIR, exist_ok=True)
    with open(os.path.join(DIAGDIR, "MECH-BUDGET.json"), "w") as f:
        json.dump(diag, f, indent=1)
    print("WROTE", os.path.join(DIAGDIR, "MECH-BUDGET.json"))

    # ---- console summary ----------------------------------------------------
    print("\nbaseline_reproduced:", json.dumps(baseline_reproduced, indent=1))
    print("\nk-curves (loss):")
    for arm in ARM_ORDER:
        if arm not in arms_out:
            continue
        for sp in ("QA", "MT"):
            c = arms_out[arm]["k_curve"][sp]
            print(f"  {arm:7s} {sp}: " + "  ".join(
                f"k{k}={c.get(k)!s:.8s}" for k in (8, 10, 12, 16)))
    print("\ncost:", json.dumps(cost_scaling, indent=1))
    print("\ndeltas:", json.dumps(deltas, indent=1))
    for arm in ARM_ORDER:
        if arm in arms_out and "eval_mass_on_S" in arms_out[arm]:
            print(arm, "eval_mass_on_S:", json.dumps(arms_out[arm]["eval_mass_on_S"]))
        if arm in arms_out:
            print(arm, "ref_mass_on_S_mean=", arms_out[arm]["ref_mass_on_S_mean_over_layers"],
                  "mean_mse=", arms_out[arm]["am_mean_mse"],
                  "|v|max=", arms_out[arm]["v_absmax_global"])


if __name__ == "__main__":
    main()
