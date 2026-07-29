"""MECH-BUDGET-B (W4 TEST, B-GATE): build the result bundle.

Reads what `launch.sh` produced:
  curve.tsv          arm/top_t/nq/dw/k/split -> Eval loss + ckpt + sha + wandb url
  rundirs.tsv        arm/top_t/nq/dw/run_dir/rc/phase2_e2e_s/wandb url
  route_mass.json    eval-time mass_on_S / mass_on_cart / MT-QA ratio per arm
  <run_dir>/am_doc_*.pt, phase2_summary.json, per_document_am_stats.pt

Writes:
  research_loop/results/MECH-BUDGET-B/result.json        (WORKERS.md contract)
  research_loop/state/diagnostics/MECH-BUDGET-B.json     (per-arm k-curves, cost, raw)

Adapted from results/MECH-BUDGET/finalize.py. Nothing here touches cartridges/ or examples/.
"""

from __future__ import annotations

import json
import os
import subprocess
from glob import glob

import torch

REPO = "/localhome/local-triv/gated-continual-cartridges_explore"
RESDIR = os.path.join(REPO, "research_loop/results/MECH-BUDGET-B")
DIAGDIR = os.path.join(REPO, "research_loop/state/diagnostics")

# DIAG-KEYCURVE / MECH-BUDGET A_t32 reference for the gate (standalone eval, same config)
GATE_REF = {
    (12, "QA"): 1.9560121297836304,
    (12, "MT"): 2.2720184326171875,
    (16, "QA"): 2.034916400909424,
    (16, "MT"): 2.330503463745117,
}
# DIAG-NOISE measured paired resolution (95%)
RES = {"MT": 0.0494, "QA": 0.0539}
RES_COMPOSITE = {"MT": 0.1055, "QA": 0.197}

ARM_ORDER = ["G_t32q64", "A_t128q512", "C_t64q512", "B_t256q1024", "D_t128q512dw"]
BASE = "G_t32q64"

# MECH-BUDGET's arms, for the "does raising n change the t128 verdict?" contrast
MECH_BUDGET = {
    "t32_q64": {"MT": {8: 2.32890248298645, 10: 2.319589376449585,
                       12: 2.2720184326171875, 16: 2.330503463745117},
                "QA": {8: 1.9463961124420166, 10: 1.968939185142517,
                       12: 1.9560121297836304, 16: 2.034916400909424}},
    "t64_q64": {"MT": {8: 2.231785774230957, 10: 2.262895107269287,
                       12: 2.288086414337158, 16: 2.3589916229248047},
                "QA": {8: 2.036161422729492, 10: 2.0982749462127686,
                       12: 2.156097412109375, 16: 2.2381808757781982}},
    "t128_q64": {"MT": {8: 2.41554856300354, 10: 2.3583920001983643,
                        12: 2.3570852279663086, 16: 2.505143880844116},
                 "QA": {8: 2.3824715614318848, 10: 2.338186740875244,
                        12: 2.3470959663391113, 16: 2.4991507530212402}},
}


def read_tsv(path):
    if not os.path.exists(path):
        return []
    lines = open(path).read().splitlines()
    if not lines:
        return []
    hdr = lines[0].split("\t")
    return [dict(zip(hdr, l.split("\t"))) for l in lines[1:] if l.strip()]


def collect_run(run_dir: str) -> dict:
    """Per-arm internals: mean_mse, ref mass_on_S per layer, |v|max, n_queries, timings."""
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
    out["am_solve_s_sum"] = sum(d["timing"].get("solve_s", 0.0) for d in per_doc) or None
    out["am_rank_s_sum"] = sum(d["timing"].get("rank_s", 0.0) for d in per_doc) or None
    out["am_collect_s_sum"] = sum(d["timing"].get("collect_s", 0.0) for d in per_doc) or None
    return out


def cmp_block(a, b, sp):
    dv = a - b
    return {
        "delta": dv,
        "clears_paired_resolution": abs(dv) > RES[sp],
        "x_resolution": abs(dv) / RES[sp],
        "clears_conservative_composite": abs(dv) > RES_COMPOSITE[sp],
    }


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
        pd0 = (info.get("per_document") or [{}])[0]
        top_t = int(rd[arm]["top_t"])
        nq_cap = int(rd[arm]["nq"])
        nq_used_min = pd0.get("n_queries_used_min")
        nq_used_max = pd0.get("n_queries_used_max")
        # determinacy: the DATA block alone must have >= top_t rows
        det = None
        if nq_used_min is not None:
            ratio = nq_used_min / top_t
            det = {
                "top_t_unknowns_per_head": top_t,
                "n_queries_used_min": nq_used_min,
                "n_queries_used_max": nq_used_max,
                "data_rows_over_unknowns": ratio,
                "data_block_determined": nq_used_min >= top_t,
                "verdict": (
                    "over-determined %.1f:1" % ratio if ratio > 1.0
                    else ("exactly determined 1:1" if ratio == 1.0
                          else "UNDER-determined 1:%.1f" % (1.0 / ratio))
                ),
            }
        arms_out[arm] = {
            "top_t": top_t,
            "max_queries_per_head": nq_cap,
            "delta_weight": float(rd[arm]["dw"]),
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
            "solve_determinacy": det,
            "n_queries_available_min": pd0.get("n_queries_available_min"),
            "n_queries_available_max": pd0.get("n_queries_available_max"),
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
            "_raw": info,
        }
        rr = (route.get("ratios") or {}).get(arm)
        if rr:
            arms_out[arm]["eval_mass_on_S"] = {
                "MT": rr["mass_on_S_MT"],
                "QA": rr["mass_on_S_QA"],
                "mt_over_qa_ratio": rr["mt_over_qa_mass_on_S"],
                "total_cartridge_mass_MT": rr["mass_on_cart_MT"],
                "total_cartridge_mass_QA": rr["mass_on_cart_QA"],
            }
        for sp in ("QA", "MT"):
            run = (route.get("runs") or {}).get(f"{arm}|{sp}")
            if run:
                arms_out[arm].setdefault("eval_mass_per_layer", {})[sp] = run["per_layer"]
                arms_out[arm].setdefault("eval_mass_summary", {})[sp] = run["summary"]
                arms_out[arm]["n_slots_union_per_layer"] = run["n_slots_per_layer"]

    # ---- gate ---------------------------------------------------------------
    gate_measured, gate_delta = {}, {}
    for (k, sp), want in GATE_REF.items():
        got = losses.get(BASE, {}).get(sp, {}).get(k)
        gate_measured[f"k{k}_{sp}"] = got
        gate_delta[f"k{k}_{sp}"] = (got - want) if got is not None else None
    baseline_reproduced = {
        "arm": BASE,
        "reference": {f"k{k}_{sp}": v for (k, sp), v in GATE_REF.items()},
        "reference_source": "DIAG-KEYCURVE / MECH-BUDGET A_t32 standalone eval, same config, TOP_T=32, MAX_QUERIES_PER_HEAD=64",
        "measured": gate_measured,
        "delta": gate_delta,
        "exact_to_1e-6": all(d is not None and abs(d) <= 1e-6 for d in gate_delta.values()),
    }

    # ---- deltas vs the incumbent (this session's own gate arm) --------------
    deltas = {}
    for arm in ARM_ORDER:
        if arm == BASE or arm not in arms_out:
            continue
        d = {}
        for sp in ("QA", "MT"):
            for k in (8, 10, 12, 16):
                a = losses.get(arm, {}).get(sp, {}).get(k)
                b = losses.get(BASE, {}).get(sp, {}).get(k)
                if a is None or b is None:
                    continue
                d[f"k{k}_{sp}"] = cmp_block(a, b, sp)
            ca = {k: v for k, v in losses.get(arm, {}).get(sp, {}).items() if v is not None}
            cb = {k: v for k, v in losses.get(BASE, {}).get(sp, {}).items() if v is not None}
            if ca and cb:
                ka, kb = min(ca, key=ca.get), min(cb, key=cb.get)
                blk = cmp_block(ca[ka], cb[kb], sp)
                blk.update({"arm_argmin_k": ka, "arm_min": ca[ka],
                            "base_argmin_k": kb, "base_min": cb[kb]})
                d[f"own_optimum_{sp}"] = blk
        deltas[arm] = d

    # ---- THE decisive contrasts --------------------------------------------
    contrasts = {}

    def pair(name, arm_a, arm_b, question):
        if arm_a not in losses or arm_b not in losses:
            return
        blk = {"question": question, "arm_a": arm_a, "arm_b": arm_b, "per_k": {}}
        for sp in ("QA", "MT"):
            for k in (8, 10, 12, 16):
                a = losses[arm_a].get(sp, {}).get(k)
                b = losses[arm_b].get(sp, {}).get(k)
                if a is None or b is None:
                    continue
                blk["per_k"][f"k{k}_{sp}"] = cmp_block(a, b, sp)
            ca = {k: v for k, v in losses[arm_a].get(sp, {}).items() if v is not None}
            cb = {k: v for k, v in losses[arm_b].get(sp, {}).items() if v is not None}
            if ca and cb:
                ka, kb = min(ca, key=ca.get), min(cb, key=cb.get)
                x = cmp_block(ca[ka], cb[kb], sp)
                x.update({"a_argmin_k": ka, "a_min": ca[ka], "b_argmin_k": kb, "b_min": cb[kb]})
                blk[f"own_optimum_{sp}"] = x
        contrasts[name] = blk

    pair("support_at_matched_queries__t128q512_vs_t64q512", "A_t128q512", "C_t64q512",
         "same n=512, same w=1e-2, both determined: does MORE SUPPORT help?")
    pair("queries_alone__t64q512_vs_t32q64", "C_t64q512", "G_t32q64",
         "the control the brief asked for: does raising n (and doubling t) move the incumbent?")
    pair("t256_vs_t128__both_determined", "B_t256q1024", "A_t128q512",
         "does pushing support further at held ratio keep helping/hurting?")
    pair("delta_weight_sensitivity__t128q512_dw0.08_vs_dw0.01", "D_t128q512dw", "A_t128q512",
         "is the t128/q512 result an artefact of the trust region's 1/n dilution?")

    # vs MECH-BUDGET's underdetermined t128 (same top_t, n 64 -> 512)
    if "A_t128q512" in losses:
        blk = {"question": "MECH-BUDGET t128 was solved with n=64 (under-determined). "
                           "Same top_t, n raised to 512: what changed?",
               "per_k": {}}
        for sp in ("QA", "MT"):
            for k in (8, 10, 12, 16):
                a = losses["A_t128q512"].get(sp, {}).get(k)
                b = MECH_BUDGET["t128_q64"][sp].get(k)
                if a is None:
                    continue
                blk["per_k"][f"k{k}_{sp}"] = cmp_block(a, b, sp)
            ca = {k: v for k, v in losses["A_t128q512"].get(sp, {}).items() if v is not None}
            cb = MECH_BUDGET["t128_q64"][sp]
            ka, kb = min(ca, key=ca.get), min(cb, key=cb.get)
            x = cmp_block(ca[ka], cb[kb], sp)
            x.update({"a_argmin_k": ka, "a_min": ca[ka], "b_argmin_k": kb, "b_min": cb[kb]})
            blk[f"own_optimum_{sp}"] = x
        contrasts["determinacy__t128q512_vs_MECHBUDGET_t128q64"] = blk

    if "C_t64q512" in losses:
        blk = {"question": "MECH-BUDGET t64 was already exactly determined at n=64. "
                           "Same top_t, n raised to 512: does more n alone move it?",
               "per_k": {}}
        for sp in ("QA", "MT"):
            for k in (8, 10, 12, 16):
                a = losses["C_t64q512"].get(sp, {}).get(k)
                b = MECH_BUDGET["t64_q64"][sp].get(k)
                if a is None:
                    continue
                blk["per_k"][f"k{k}_{sp}"] = cmp_block(a, b, sp)
            ca = {k: v for k, v in losses["C_t64q512"].get(sp, {}).items() if v is not None}
            cb = MECH_BUDGET["t64_q64"][sp]
            ka, kb = min(ca, key=ca.get), min(cb, key=cb.get)
            x = cmp_block(ca[ka], cb[kb], sp)
            x.update({"a_argmin_k": ka, "a_min": ca[ka], "b_argmin_k": kb, "b_min": cb[kb]})
            blk[f"own_optimum_{sp}"] = x
        contrasts["queries_alone__t64q512_vs_MECHBUDGET_t64q64"] = blk

    # cost scaling
    cost_scaling = {}
    base_cost = arms_out.get(BASE, {}).get("cost", {})
    for arm in ARM_ORDER:
        if arm not in arms_out:
            continue
        c = arms_out[arm]["cost"]
        cost_scaling[arm] = {
            "top_t": arms_out[arm]["top_t"],
            "max_queries_per_head": arms_out[arm]["max_queries_per_head"],
            "phase2_e2e_s": c["phase2_e2e_s"],
            "solve_s": c["solve_s"],
            "am_solve_s_sum": c["am_solve_s_sum"],
            "wall_clock_s": c["wall_clock_s"],
            "solve_multiple_vs_base": (c["solve_s"] / base_cost["solve_s"]) if base_cost.get("solve_s") else None,
            "e2e_multiple_vs_base": (c["phase2_e2e_s"] / base_cost["phase2_e2e_s"]) if base_cost.get("phase2_e2e_s") else None,
        }

    targets = {
        "target_qa_le": 2.52, "target_mt_le": 2.02,
        "incumbent_best": {"QA": 1.9560, "MT": 2.2720, "where": "tfidf@32 q64, k=12"},
        "mech_budget_t64_best_MT": 2.2318,
        "mech_budget_t128_underdetermined_best_MT": 2.3571,
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
                "beats_mech_budget_t64_best": mt[kmt] < targets["mech_budget_t64_best_MT"],
            }

    head_after = subprocess.run(
        ["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    wlp = os.path.join(RESDIR, "wrapper.log")
    wl = open(wlp).read() if os.path.exists(wlp) else ""

    def _grab(key):
        for line in wl.splitlines():
            if line.startswith(key + "="):
                return line.split("=", 1)[1]
        return None

    gsl = os.path.join(RESDIR, "git_status_at_launch.txt")
    provenance = {
        "snapshot_path": "/tmp/amsnap_budgetb",
        "snapshot_source": "git archive HEAD cartridges examples",
        "repo_head_at_launch": _grab("REPO_HEAD_AT_LAUNCH"),
        "repo_head_after": _grab("REPO_HEAD_AFTER") or head_after,
        "snapshot_manifest_at_launch": _grab("SNAPSHOT_MANIFEST_AT_LAUNCH"),
        "snapshot_manifest_after": _grab("SNAPSHOT_MANIFEST_AFTER"),
        "snapshot_manifest_stable": _grab("SNAPSHOT_MANIFEST_STABLE"),
        "snapshot_manifest_equals_MECH_BUDGET": (
            _grab("SNAPSHOT_MANIFEST_AT_LAUNCH")
            == "b71eaf15e00306be667183893a745c19460b9261739d179433ddc919c8abe1c9"
        ),
        "import_probe_from_tmp": _grab("IMPORT_PROBE"),
        "gpu": _grab("MBB_CLAIMED_GPU"),
        "concurrent_worker_edits_at_launch": (
            open(gsl).read().split() if os.path.exists(gsl) else None
        ),
        "snapshot_has_MECH008_redundancy_selector": False,
        "worker_edited_no_source_file": True,
    }

    diag = {
        "id": "MECH-BUDGET-B",
        "board_entry": "B-GATE",
        "role": "test",
        "variable_under_test": (
            "TOP_T x MAX_QUERIES_PER_HEAD, held so the DATA block alone determines the solve "
            "(t128/q512, t256/q1024, t64/q512 control), vs the incumbent t32/q64"
        ),
        "delta_weight_policy": (
            "The four primary arms hold DELTA_WEIGHT=1e-2 (the base config). D_t128q512dw repeats "
            "A_t128q512 with DELTA_WEIGHT=8e-2 = 1e-2*n/64, MECH-QUERIES-B's w-proportional-to-n line, "
            "which holds the trust region's RELATIVE pull at its n=64 value instead of letting it decay "
            "like 1/n."
        ),
        "paired_resolution_used": RES,
        "conservative_composite": RES_COMPOSITE,
        "arms": arms_out,
        "k_curves": {a: arms_out[a]["k_curve"] for a in arms_out},
        "deltas_vs_incumbent_t32q64": deltas,
        "contrasts": contrasts,
        "mech_budget_reference_curves": MECH_BUDGET,
        "cost_scaling": cost_scaling,
        "baseline_reproduced": baseline_reproduced,
        "targets": targets,
        "provenance": provenance,
        "route_mass_raw_path": rp if os.path.exists(rp) else None,
    }
    os.makedirs(DIAGDIR, exist_ok=True)
    with open(os.path.join(DIAGDIR, "MECH-BUDGET-B.json"), "w") as f:
        json.dump(diag, f, indent=1)
    print("WROTE", os.path.join(DIAGDIR, "MECH-BUDGET-B.json"))

    print("\nbaseline_reproduced:", json.dumps(baseline_reproduced, indent=1))
    print("\nk-curves (loss):")
    for arm in ARM_ORDER:
        if arm not in arms_out:
            continue
        for sp in ("QA", "MT"):
            c = arms_out[arm]["k_curve"][sp]
            print(f"  {arm:14s} {sp}: " + "  ".join(
                f"k{k}={c.get(k)!s:.8s}" for k in (8, 10, 12, 16)))
    print("\ndeterminacy:")
    for arm in ARM_ORDER:
        if arm in arms_out:
            print(" ", arm, json.dumps(arms_out[arm].get("solve_determinacy")))
    print("\ncost:", json.dumps(cost_scaling, indent=1))
    for arm in ARM_ORDER:
        if arm in arms_out:
            print(arm, "eval_mass_on_S:", json.dumps(arms_out[arm].get("eval_mass_on_S")),
                  "ref_mass=", arms_out[arm]["ref_mass_on_S_mean_over_layers"],
                  "mse=", arms_out[arm]["am_mean_mse"],
                  "|v|max=", arms_out[arm]["v_absmax_global"])
    print("\ncontrasts:", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_k"}
                                      for k, v in contrasts.items()}, indent=1))

    # ---- result.json (WORKERS.md contract) ----------------------------------
    obs_path = os.path.join(RESDIR, "observations.txt")
    observations = open(obs_path).read().strip() if os.path.exists(obs_path) else "PENDING"

    head_arm = "A_t128q512" if "A_t128q512" in arms_out else BASE
    ha = arms_out.get(head_arm, {})
    hpc = ha.get("pass_check", {}) or {}

    def slim(arm):
        a = arms_out[arm]
        return {
            "top_t": a["top_t"],
            "max_queries_per_head": a["max_queries_per_head"],
            "delta_weight": a["delta_weight"],
            "n_queries_used_min": (a.get("solve_determinacy") or {}).get("n_queries_used_min"),
            "n_queries_used_max": (a.get("solve_determinacy") or {}).get("n_queries_used_max"),
            "n_queries_available_max": a.get("n_queries_available_max"),
            "solve_determinacy": (a.get("solve_determinacy") or {}).get("verdict"),
            "data_block_determined": (a.get("solve_determinacy") or {}).get("data_block_determined"),
            "k_curve_QA": {f"k{k}": v for k, v in sorted(a["k_curve"]["QA"].items())},
            "k_curve_MT": {f"k{k}": v for k, v in sorted(a["k_curve"]["MT"].items())},
            "pass_check": a.get("pass_check"),
            "am_mean_mse_over_docs": a["am_mean_mse"],
            "am_mean_mse_last_doc": a["am_mean_mse_last_doc"],
            "v_absmax_global": a["v_absmax_global"],
            "ref_mass_on_S_mean_over_layers": a["ref_mass_on_S_mean_over_layers"],
            "eval_mass_on_S": a.get("eval_mass_on_S"),
            "n_slots_union_per_layer_min_max": (
                [min(int(x) for x in a["n_slots_union_per_layer"].values()),
                 max(int(x) for x in a["n_slots_union_per_layer"].values())]
                if a.get("n_slots_union_per_layer") else None
            ),
            "cost": a["cost"],
            "wandb_run_url": a["wandb_run_url"],
            "wandb_run_id": a["wandb_run_id"],
            "run_dir": a["run_dir"],
            "rc": a["rc"],
        }

    result = {
        "id": "MECH-BUDGET-B",
        "role": "test",
        "board_entry": "B-GATE",
        "question": (
            "MECH-BUDGET swept TOP_T with MAX_QUERIES_PER_HEAD pinned at 64, so top_t=128 solved "
            "128 unknowns per head against a rank-64 data block and its verdict measured "
            "underdetermination rather than support. With enough reference queries that the data "
            "block alone determines the solve (n >> t), does LARGE SUPPORT convert the project's QA "
            "slack into MT acquisition?"
        ),
        "variable_under_test": (
            "TOP_T and MAX_QUERIES_PER_HEAD moved together so n >> t: t128/q512, t256/q1024, plus the "
            "t64/q512 control that separates 'more queries' from 'more support', against the incumbent "
            "t32/q64. DELTA_WEIGHT held at the base 1e-2 in all four; one extra sensitivity arm repeats "
            "t128/q512 at DELTA_WEIGHT=8e-2 = 1e-2*n/64 (MECH-QUERIES-B's w-proportional-to-n line). "
            "Everything else at the project's best point: KEY_MODE=highest_attention, "
            "AM_KEY_REPOSITION=1, AM_ROPE_THETA=5e6, ENABLE_BETA=0, GRANULARITY=per_layer, "
            "SLOT_SELECTION=tfidf, USE_IDF=0, RIDGE_LAMBDA=1e-4, RIDGE_SCALE=spectral."
        ),
        "baseline_ref": (
            "G_t32q64 (this session's own re-run of the incumbent) gated against DIAG-KEYCURVE / "
            "MECH-BUDGET A_t32: k=12 QA 1.9560/MT 2.2720, k=16 QA 2.0349/MT 2.3305. Also compared "
            "arm-for-arm against MECH-BUDGET's t64 and t128 curves at n=64."
        ),
        "status": "done",
        "numbers": {
            "qa_forgetting_loss": hpc.get("QA_at_best_MT_k"),
            "mt_acquisition_loss": hpc.get("best_MT"),
            "solve_s": (ha.get("cost") or {}).get("solve_s"),
            "phase2_e2e_s": (ha.get("cost") or {}).get("phase2_e2e_s"),
            "gradient_steps": 0,
            "_note": (
                f"headline row = {head_arm} at its own MT optimum (k={hpc.get('best_MT_k')}); "
                "every arm is in diagnostics.arms. solve_s is the cost figure to quote; "
                "phase2_e2e_s is contaminated by cold-CUDA JIT (MECH-BUDGET showed this)."
            ),
        },
        "diagnostics": {
            "paired_resolution_yardstick": {"MT": RES["MT"], "QA": RES["QA"],
                                            "source": "DIAG-NOISE measured paired 95% interval"},
            "conservative_composite": RES_COMPOSITE,
            "delta_weight_policy": diag["delta_weight_policy"],
            "arms": {a: slim(a) for a in ARM_ORDER if a in arms_out},
            "contrasts": {k: {kk: vv for kk, vv in v.items() if kk != "per_k"}
                          for k, v in contrasts.items()},
            "deltas_vs_incumbent_t32q64": deltas,
            "cost_scaling": cost_scaling,
            "baseline_reproduced": baseline_reproduced,
            "targets": targets,
            "provenance": provenance,
        },
        "wandb_run_url": ha.get("wandb_run_url"),
        "wandb_run_id": ha.get("wandb_run_id"),
        "wandb_run_urls": {
            "train": {a: arms_out[a]["wandb_run_url"] for a in ARM_ORDER if a in arms_out},
            "evals": {a: arms_out[a]["eval_wandb_urls"] for a in ARM_ORDER if a in arms_out},
        },
        "artifacts": (
            [arms_out[a]["run_dir"] for a in ARM_ORDER if a in arms_out]
            + [
                "research_loop/results/MECH-BUDGET-B/launch.sh",
                "research_loop/results/MECH-BUDGET-B/finalize.py",
                "research_loop/results/MECH-BUDGET-B/wrapper.log",
                "research_loop/results/MECH-BUDGET-B/curve.tsv",
                "research_loop/results/MECH-BUDGET-B/rundirs.tsv",
                "research_loop/results/MECH-BUDGET-B/route_mass.json",
                "research_loop/results/MECH-BUDGET-B/determinacy_check.log",
                "research_loop/state/diagnostics/MECH-BUDGET-B.json",
            ]
        ),
        "command": (
            "bash research_loop/results/MECH-BUDGET-B/launch.sh  (env-only; NO source file edited. "
            "Imports pinned to the frozen snapshot /tmp/amsnap_budgetb = `git archive HEAD cartridges "
            "examples`; import probe run from /tmp printed /tmp/amsnap_budgetb/cartridges; snapshot "
            "manifest verified before and after. Driver, eval and the routing-mass probe all ran from "
            "the snapshot. One GPU claimed via flock.)"
        ),
        "observations": observations,
    }
    with open(os.path.join(RESDIR, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    print("WROTE", os.path.join(RESDIR, "result.json"),
          "(observations:", "REAL)" if observations != "PENDING" else "PENDING)")


if __name__ == "__main__":
    main()
