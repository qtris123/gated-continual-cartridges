#!/usr/bin/env python
"""VERIFY-OPTIMA — build the diagnostics bundle from curve.tsv.

Reads this job's own evals (curve.tsv), splices in the numbers this job did NOT
re-measure (MECH-SEED's k=12/k=16 at all three offsets; DIAG-CONTROLCURVE's full
offset-0 control curve; DIAG-KEYCURVE's full offset-0 keys curve), and computes:
  * the correctness gates,
  * the own-optimum comparison (keys k=12 vs control k=10) per offset,
  * the common-k comparison at k=10,
  * the adversarial pairing,
  * argmin stability of the control's MT curve across seeds.

Nothing here re-runs anything; it is arithmetic over measured losses.
"""
from __future__ import annotations

import json
import os
from statistics import mean

RES = "/localhome/local-triv/gated-continual-cartridges_explore/research_loop/results/VERIFY-OPTIMA"
STATE = "/localhome/local-triv/gated-continual-cartridges_explore/research_loop/state/diagnostics"

MT_PAIRED_RES = 0.049414709208868444   # DIAG-NOISE, 95% paired MT
QA_PAIRED_RES = 0.05389150679130719    # DIAG-NOISE, 95% paired QA
MT_COMPOSITE = 0.10553301160022083     # DIAG-NOISE conservative composite (incl. reference-draw term)
QA_COMPOSITE = 0.1967026909783353

# ---- numbers measured by earlier jobs (spliced, not re-measured here) --------
MECH_SEED = {
    ("keys", 0, 12): {"QA": 1.9560121297836304, "MT": 2.2720184326171875},
    ("keys", 0, 16): {"QA": 2.034916400909424, "MT": 2.330503463745117},
    ("keys", 1000, 12): {"QA": 1.9397201538085938, "MT": 2.269538164138794},
    ("keys", 1000, 16): {"QA": 1.996092677116394, "MT": 2.306238889694214},
    ("keys", 2000, 12): {"QA": 1.9446364641189575, "MT": 2.262629747390747},
    ("keys", 2000, 16): {"QA": 2.0024614334106445, "MT": 2.290109395980835},
    ("control", 0, 12): {"QA": 2.074141263961792, "MT": 2.4704575538635254},
    ("control", 0, 16): {"QA": 2.159724712371826, "MT": 2.529625177383423},
    ("control", 1000, 12): {"QA": 2.0522191524505615, "MT": 2.448606491088867},
    ("control", 1000, 16): {"QA": 2.138103485107422, "MT": 2.482379198074341},
    ("control", 2000, 12): {"QA": 2.070828914642334, "MT": 2.5163300037384033},
    ("control", 2000, 16): {"QA": 2.2329885959625244, "MT": 2.6113579273223877},
}
CONTROLCURVE_OFF0_MT = {  # DIAG-CONTROLCURVE, all 16 k, offset 0
    1: 3.1145591735839844, 2: 2.7727901935577393, 3: 2.6456825733184814, 4: 2.526649236679077,
    5: 2.536327838897705, 6: 2.5201644897460938, 7: 2.4667556285858154, 8: 2.453585624694824,
    9: 2.4436779022216797, 10: 2.4083292484283447, 11: 2.4306740760803223, 12: 2.4704575538635254,
    13: 2.486041784286499, 14: 2.4835116863250732, 15: 2.514181613922119, 16: 2.529625177383423,
}
CONTROLCURVE_OFF0_QA = {
    1: 1.9870699644088745, 2: 1.8753408193588257, 3: 1.8177090883255005, 4: 1.8310279846191406,
    5: 1.900788426399231, 6: 1.931013822555542, 7: 1.9014028310775757, 8: 1.9329710006713867,
    9: 1.9655601978302002, 10: 1.9674913883209229, 11: 1.9997590780258179, 12: 2.074141263961792,
    13: 2.079558849334717, 14: 2.0921683311462402, 15: 2.1373283863067627, 16: 2.159724712371826,
}
KEYCURVE_OFF0_MT = {  # DIAG-KEYCURVE, all 16 k, offset 0
    1: 2.965111255645752, 2: 2.6856672763824463, 3: 2.5314693450927734, 4: 2.4055702686309814,
    5: 2.4009695053100586, 6: 2.346301555633545, 7: 2.3156590461730957, 8: 2.32890248298645,
    9: 2.31606125831604, 10: 2.319589376449585, 11: 2.2814929485321045, 12: 2.2720184326171875,
    13: 2.298682928085327, 14: 2.319815158843994, 15: 2.3325002193450928, 16: 2.330503463745117,
}
KEYCURVE_OFF0_QA = {
    1: 1.913637638092041, 2: 1.8658878803253174, 3: 1.8607447147369385, 4: 1.8669999837875366,
    5: 1.9200594425201416, 6: 1.9135236740112305, 7: 1.9160072803497314, 8: 1.9463961124420166,
    9: 1.93601655960083, 10: 1.968939185142517, 11: 1.9373838901519775, 12: 1.9560121297836304,
    13: 1.989310383796692, 14: 2.015522003173828, 15: 2.0397305488586426, 16: 2.034916400909424,
}

GATES = {
    ("control", 1000, 12, "MT"): (2.448606491088867, "MECH-SEED"),
    ("control", 2000, 12, "MT"): (2.5163300037384033, "MECH-SEED"),
    ("control", 0, 10, "MT"): (2.4083292484283447, "DIAG-CONTROLCURVE"),
    ("control", 0, 10, "QA"): (1.9674913883209229, "DIAG-CONTROLCURVE"),
    ("keys", 0, 10, "MT"): (2.319589376449585, "DIAG-KEYCURVE"),
    ("keys", 0, 10, "QA"): (1.968939185142517, "DIAG-KEYCURVE"),
}


def load_curve():
    rows, urls, provenance = {}, {}, []
    path = f"{RES}/curve.tsv"
    with open(path) as fh:
        head = fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            arm, off, k, split, loss, ckpt, url, ckpt_in_log, pkg = parts[:9]
            if loss in ("MISSING", "MISSING_SNAPSHOT"):
                provenance.append({"arm": arm, "offset": int(off), "k": int(k), "split": split,
                                   "status": loss})
                continue
            rows[(arm, int(off), int(k), split)] = float(loss)
            urls[(arm, int(off), int(k), split)] = url
            provenance.append({"arm": arm, "offset": int(off), "k": int(k), "split": split,
                               "loss": float(loss), "ckpt": ckpt, "wandb_url": url,
                               "ckpt_path_found_in_eval_log": int(ckpt_in_log), "cartridges_pkg": pkg})
    return rows, urls, provenance


def main():
    new, urls, provenance = load_curve()

    # ---------------- gates ----------------
    gate_report, gates_ok = {}, True
    for key, (want, src) in GATES.items():
        got = new.get(key)
        ok = got is not None and abs(got - want) <= 1e-9
        gates_ok &= ok
        gate_report["_".join(map(str, key))] = {
            "expected": want, "observed": got, "source_of_expectation": src,
            "exact_match": bool(ok),
            "abs_diff": None if got is None else abs(got - want),
        }

    # ---------------- merged loss table ----------------
    losses = {}   # (arm, off, k, split) -> (loss, origin)
    for (arm, off, k), d in MECH_SEED.items():
        for split, v in d.items():
            losses[(arm, off, k, split)] = (v, "MECH-SEED")
    for k, v in CONTROLCURVE_OFF0_MT.items():
        losses.setdefault(("control", 0, k, "MT"), (v, "DIAG-CONTROLCURVE"))
    for k, v in CONTROLCURVE_OFF0_QA.items():
        losses.setdefault(("control", 0, k, "QA"), (v, "DIAG-CONTROLCURVE"))
    for k, v in KEYCURVE_OFF0_MT.items():
        losses.setdefault(("keys", 0, k, "MT"), (v, "DIAG-KEYCURVE"))
    for k, v in KEYCURVE_OFF0_QA.items():
        losses.setdefault(("keys", 0, k, "QA"), (v, "DIAG-KEYCURVE"))
    for key, v in new.items():
        losses[key] = (v, "VERIFY-OPTIMA")   # this session's measurement wins

    def L(arm, off, k, split):
        e = losses.get((arm, off, k, split))
        return None if e is None else e[0]

    offsets = [0, 1000, 2000]

    # ---------------- own-optimum comparison ----------------
    own = {}
    for off in offsets:
        km, kc = 12, 10
        row = {}
        for split in ("MT", "QA"):
            a, b = L("keys", off, km, split), L("control", off, kc, split)
            row[f"keys_k12_{split}"] = a
            row[f"control_k10_{split}"] = b
            row[f"d{split}"] = None if (a is None or b is None) else a - b
        own[off] = row
    d_mt = [own[o]["dMT"] for o in offsets]
    d_qa = [own[o]["dQA"] for o in offsets]

    # ---------------- common-k (k=10) comparison ----------------
    common = {}
    for off in offsets:
        row = {}
        for split in ("MT", "QA"):
            a, b = L("keys", off, 10, split), L("control", off, 10, split)
            row[f"keys_k10_{split}"] = a
            row[f"control_k10_{split}"] = b
            row[f"d{split}"] = None if (a is None or b is None) else a - b
        common[off] = row

    # ---------------- adversarial pairing ----------------
    keys_k12_mt = [L("keys", o, 12, "MT") for o in offsets]
    ctrl_k10_mt = [L("control", o, 10, "MT") for o in offsets]
    keys_k12_qa = [L("keys", o, 12, "QA") for o in offsets]
    ctrl_k10_qa = [L("control", o, 10, "QA") for o in offsets]
    adversarial = {
        "note": "keys' WORST seed at its argmin k=12 against the control's BEST seed at its argmin k=10.",
        "keys_worst_MT": max(keys_k12_mt), "keys_worst_MT_offset": offsets[keys_k12_mt.index(max(keys_k12_mt))],
        "control_best_MT": min(ctrl_k10_mt), "control_best_MT_offset": offsets[ctrl_k10_mt.index(min(ctrl_k10_mt))],
        "dMT": max(keys_k12_mt) - min(ctrl_k10_mt),
        "keys_worst_QA": max(keys_k12_qa), "control_best_QA": min(ctrl_k10_qa),
        "dQA": max(keys_k12_qa) - min(ctrl_k10_qa),
    }
    adversarial["dMT_clears_paired_resolution"] = abs(adversarial["dMT"]) > MT_PAIRED_RES
    adversarial["dMT_clears_composite_resolution"] = abs(adversarial["dMT"]) > MT_COMPOSITE
    adversarial["dQA_clears_paired_resolution"] = abs(adversarial["dQA"]) > QA_PAIRED_RES

    # ---------------- argmin stability ----------------
    argmin = {}
    for off in offsets:
        curve = {k: L("control", off, k, "MT") for k in range(1, 17)}
        curve = {k: v for k, v in curve.items() if v is not None}
        best_k = min(curve, key=curve.get)
        argmin[off] = {
            "control_MT_by_k": curve,
            "k_measured": sorted(curve),
            "k_not_measured": [k for k in range(1, 17) if k not in curve],
            "argmin_k": best_k, "argmin_MT": curve[best_k],
            "MT_at_k10": curve.get(10),
            "gap_k10_minus_argmin": None if curve.get(10) is None else curve[10] - curve[best_k],
            "second_best_k": sorted(curve, key=curve.get)[1] if len(curve) > 1 else None,
            "gap_best_to_second": (sorted(curve.values())[1] - sorted(curve.values())[0]) if len(curve) > 1 else None,
        }
    # per-seed own-argmin comparison: control at ITS per-seed argmin
    per_seed_argmin_cmp = {}
    for off in offsets:
        ck = argmin[off]["argmin_k"]
        keys_curve = {k: L("keys", off, k, "MT") for k in range(1, 17)}
        keys_curve = {k: v for k, v in keys_curve.items() if v is not None}
        kk = min(keys_curve, key=keys_curve.get)
        per_seed_argmin_cmp[off] = {
            "control_argmin_k": ck, "control_argmin_MT": argmin[off]["argmin_MT"],
            "keys_k_measured": sorted(keys_curve),
            "keys_argmin_k_over_measured": kk, "keys_argmin_MT_over_measured": keys_curve[kk],
            "dMT_each_arm_at_its_own_measured_argmin": keys_curve[kk] - argmin[off]["argmin_MT"],
        }

    # per-seed own-argmin dMT (the honest version: each arm at ITS OWN per-seed argmin)
    ps = [per_seed_argmin_cmp[o]["dMT_each_arm_at_its_own_measured_argmin"] for o in offsets]
    ctrl_min_vals = [argmin[o]["argmin_MT"] for o in offsets]
    per_seed_summary = {
        "dMT_per_offset": {str(o): v for o, v in zip(offsets, ps)},
        "mean": mean(ps), "range": max(ps) - min(ps),
        "all_same_sign": len({v < 0 for v in ps}) == 1,
        "all_clear_paired_resolution": all(abs(v) > MT_PAIRED_RES for v in ps),
        "all_clear_composite_resolution": all(abs(v) > MT_COMPOSITE for v in ps),
        "ratio_each_over_paired_resolution": {str(o): abs(v) / MT_PAIRED_RES for o, v in zip(offsets, ps)},
        "adversarial_dMT": max(keys_k12_mt) - min(ctrl_min_vals),
        "adversarial_note": "keys' worst seed at k=12 vs the control's best (seed, k) pair anywhere on its curve",
        "control_own_minimum_value_across_seeds": {
            "per_offset": {str(o): v for o, v in zip(offsets, ctrl_min_vals)},
            "mean": mean(ctrl_min_vals), "range": max(ctrl_min_vals) - min(ctrl_min_vals),
            "range_below_paired_resolution": (max(ctrl_min_vals) - min(ctrl_min_vals)) < MT_PAIRED_RES,
        },
        "argmin_k_per_offset": {str(o): argmin[o]["argmin_k"] for o in offsets},
        "argmin_moves_across_seeds": len({argmin[o]["argmin_k"] for o in offsets}) > 1,
        "gap_best_to_second_per_offset": {str(o): argmin[o]["gap_best_to_second"] for o in offsets},
        "argmin_selection_is_inside_noise": all(
            argmin[o]["gap_best_to_second"] < MT_PAIRED_RES for o in offsets),
    }

    def stats(vals):
        vals = [v for v in vals if v is not None]
        return {"per_offset": {str(o): v for o, v in zip(offsets, vals)},
                "mean": mean(vals), "min": min(vals), "max": max(vals),
                "range": max(vals) - min(vals)}

    out = {
        "id": "VERIFY-OPTIMA",
        "what_this_job_measured": (
            "the frozen-key control at k=10 (its MT argmin) and the keys arm at k=10, at seed offsets "
            "{0,1000,2000}, plus a control MT argmin-stability sweep at the two non-zero offsets. "
            "All from pre-existing per-document snapshots; nothing re-trained."
        ),
        "noise_yardstick_DIAG_NOISE": {
            "MT_paired_95pct": MT_PAIRED_RES, "QA_paired_95pct": QA_PAIRED_RES,
            "MT_composite_95pct": MT_COMPOSITE, "QA_composite_95pct": QA_COMPOSITE,
            "note": "the correct yardstick for these paired same-example comparisons is 0.0494 MT, "
                    "NOT the inherited +-0.15 rule of thumb, which is ~3x too wide.",
        },
        "correctness_gates": gate_report,
        "correctness_gates_all_passed": bool(gates_ok),
        "losses_by_arm_offset_k_split": {
            "|".join([a, str(o), str(k), s]): {"loss": v, "measured_by": src}
            for (a, o, k, s), (v, src) in sorted(losses.items())
        },
        "own_optimum_comparison_keys_k12_vs_control_k10": own,
        "own_optimum_dMT": stats(d_mt),
        "own_optimum_dQA": stats(d_qa),
        "own_optimum_dMT_vs_resolution": {
            "paired_95pct": MT_PAIRED_RES,
            "all_offsets_clear_paired": all(abs(v) > MT_PAIRED_RES for v in d_mt),
            "all_same_sign": len({v < 0 for v in d_mt}) == 1,
            "ratio_mean_effect_over_paired_resolution": abs(mean(d_mt)) / MT_PAIRED_RES,
            "all_offsets_clear_composite": all(abs(v) > MT_COMPOSITE for v in d_mt),
        },
        "own_optimum_dQA_vs_resolution": {
            "paired_95pct": QA_PAIRED_RES,
            "all_offsets_clear_paired": all(abs(v) > QA_PAIRED_RES for v in d_qa),
            "all_same_sign": len({v < 0 for v in d_qa}) == 1,
        },
        "common_k10_comparison": common,
        "common_k10_dMT": stats([common[o]["dMT"] for o in offsets]),
        "common_k10_dQA": stats([common[o]["dQA"] for o in offsets]),
        "adversarial_pairing_own_optimum": adversarial,
        "argmin_stability_control_MT": argmin,
        "per_seed_own_argmin_comparison": per_seed_argmin_cmp,
        "per_seed_own_argmin_summary": per_seed_summary,
        "matched_k_comparison_for_reference_MECH_SEED": {
            "k12_dMT": {str(o): L("keys", o, 12, "MT") - L("control", o, 12, "MT") for o in offsets},
            "k16_dMT": {str(o): L("keys", o, 16, "MT") - L("control", o, 16, "MT") for o in offsets},
        },
        "eval_provenance": provenance,
    }
    os.makedirs(STATE, exist_ok=True)
    json.dump(out, open(f"{STATE}/VERIFY-OPTIMA.json", "w"), indent=1)
    json.dump({"urls": {"|".join(map(str, k)): v for k, v in urls.items()}},
              open(f"{RES}/wandb_runs.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in (
        "correctness_gates_all_passed", "own_optimum_dMT", "own_optimum_dQA",
        "own_optimum_dMT_vs_resolution", "common_k10_dMT", "adversarial_pairing_own_optimum",
        "per_seed_own_argmin_comparison")}, indent=1))
    for off, a in argmin.items():
        print(f"[argmin] offset {off}: argmin_k={a['argmin_k']} MT={a['argmin_MT']:.6f} "
              f"k10={a['MT_at_k10']} measured={a['k_measured']}")


if __name__ == "__main__":
    main()
