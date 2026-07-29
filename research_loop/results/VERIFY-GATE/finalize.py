#!/usr/bin/env python
"""VERIFY-GATE finalize: build the comparison tables, the bundle and the diagnostics dump.

Everything here is arithmetic on `curve.tsv` (this session's evals) plus the
published MECH-INFOGATE offset-0 tfidf k-curve, which is REUSED rather than
re-run (the brief's instruction). No GPU, no model.
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics

REPO = "/localhome/local-triv/gated-continual-cartridges_explore"
RESDIR = os.path.join(REPO, "research_loop/results/VERIFY-GATE")
KS = [8, 10, 12, 16]

# DIAG-NOISE measured paired 95% resolution (NOT the inherited +-0.15 rule of thumb)
RES_MT = 0.049414709208868444
RES_QA = 0.05389150679130719
COMPOSITE_MT = 0.10553301160022083
COMPOSITE_QA = 0.1967026909783353

# ---- reused, not re-run: MECH-INFOGATE C0_control (tfidf, TOP_T=32, offset 0) ----
REUSED_T_OFF0 = {
    ("MT", 8): 2.32890248298645, ("MT", 10): 2.319589376449585,
    ("MT", 12): 2.2720184326171875, ("MT", 16): 2.330503463745117,
    ("QA", 8): 1.9463961124420166, ("QA", 10): 1.968939185142517,
    ("QA", 12): 1.9560121297836304, ("QA", 16): 2.034916400909424,
}
REUSED_T_OFF0_SRC = ("MECH-INFOGATE C0_control (identical config, SLOT_SELECTION=tfidf, "
                     "AM_SEED_OFFSET unset==0); its k=12/k=16 values are themselves "
                     "exact-to-16-digit reproductions of DIAG-KEYCURVE/MECH-005.")

# published values this bundle re-measured as reproduction gates
PUBLISHED_GATES = {
    ("redundancy", 0, "MT", 8): 2.5448672771453857,
    ("redundancy", 0, "MT", 10): 2.522254705429077,
    ("redundancy", 0, "MT", 12): 2.478079080581665,
    ("redundancy", 0, "MT", 16): 2.4386465549468994,
    ("redundancy", 0, "QA", 8): 1.930572748184204,
    ("redundancy", 0, "QA", 10): 1.9247082471847534,
    ("redundancy", 0, "QA", 12): 1.9499645233154297,
    ("redundancy", 0, "QA", 16): 1.9893748760223389,
    ("tfidf", 1000, "MT", 10): 2.273444652557373,
    ("tfidf", 1000, "MT", 12): 2.269538164138794,
    ("tfidf", 1000, "MT", 16): 2.306238889694214,
    ("tfidf", 1000, "QA", 10): 1.911603569984436,
    ("tfidf", 1000, "QA", 12): 1.9397201538085938,
    ("tfidf", 1000, "QA", 16): 1.996092677116394,
    ("tfidf", 2000, "MT", 10): 2.303954601287842,
    ("tfidf", 2000, "MT", 12): 2.262629747390747,
    ("tfidf", 2000, "MT", 16): 2.290109395980835,
    ("tfidf", 2000, "QA", 10): 1.9478639364242554,
    ("tfidf", 2000, "QA", 12): 1.9446364641189575,
    ("tfidf", 2000, "QA", 16): 2.0024614334106445,
}
GATE_SRC = {"redundancy": "MECH-INFOGATE R32", "tfidf": "MECH-SEED B1/B2 + VERIFY-OPTIMA"}


def load_curve():
    """-> loss[selector][offset][split][k], urls, ckpts"""
    loss, urls, ckpts = {}, {}, {}
    path = os.path.join(RESDIR, "curve.tsv")
    for line in open(path).read().splitlines()[1:]:
        f = line.split("\t")
        arm, sel, off, k, split, val, ckpt, url = f[0], f[1], int(f[2]), int(f[3]), f[4], f[5], f[6], f[7]
        if val == "MISSING":
            continue
        loss.setdefault(sel, {}).setdefault(off, {}).setdefault(split, {})[k] = float(val)
        urls[(arm, k, split)] = url
        ckpts[(arm, k, split)] = ckpt
    return loss, urls, ckpts


def argmin_info(curve: dict):
    """curve: {k: loss} -> (best_k, best, runner_up_gap)"""
    ks = sorted(curve)
    best_k = min(ks, key=lambda k: curve[k])
    rest = [curve[k] for k in ks if k != best_k]
    return best_k, curve[best_k], (min(rest) - curve[best_k]) if rest else None


def main() -> int:
    loss, urls, ckpts = load_curve()

    # offset-0 tfidf is reused, not re-run
    loss.setdefault("tfidf", {}).setdefault(0, {})
    for (split, k), v in REUSED_T_OFF0.items():
        loss["tfidf"][0].setdefault(split, {})[k] = v

    offsets = [0, 1000, 2000]
    out = {"id": "VERIFY-GATE", "resolution": {
        "paired_MT": RES_MT, "paired_QA": RES_QA,
        "composite_MT": COMPOSITE_MT, "composite_QA": COMPOSITE_QA,
        "source": "DIAG-NOISE measured paired 95% interval; NOT the inherited +-0.15"}}

    # ---------------- raw losses -------------------------------------------
    out["losses"] = {
        sel: {str(off): {sp: {str(k): loss[sel][off][sp][k] for k in sorted(loss[sel][off][sp])}
                         for sp in sorted(loss[sel][off])}
              for off in sorted(loss[sel])}
        for sel in sorted(loss)
    }
    out["losses_provenance"] = {
        "tfidf/0": "REUSED, not re-run: " + REUSED_T_OFF0_SRC,
        "tfidf/1000": "re-evaluated in THIS session on MECH-SEED B1's existing cache-after-doc-*.pt (no retrain)",
        "tfidf/2000": "re-evaluated in THIS session on MECH-SEED B2's existing cache-after-doc-*.pt (no retrain)",
        "redundancy/0": "RE-TRAINED and re-evaluated in this session (the gate)",
        "redundancy/1000": "trained + evaluated in this session (new)",
        "redundancy/2000": "trained + evaluated in this session (new)",
    }

    # ---------------- reproduction gates -----------------------------------
    gates = {}
    for (sel, off, split, k), want in sorted(PUBLISHED_GATES.items()):
        got = loss.get(sel, {}).get(off, {}).get(split, {}).get(k)
        gates[f"{sel}_off{off}_k{k}_{split}"] = {
            "expected": want, "observed": got,
            "exact_16_digits": got is not None and got == want,
            "within_1e-6": got is not None and abs(got - want) <= 1e-6,
            "source": GATE_SRC[sel],
        }
    out["reproduction_gates"] = gates
    out["reproduction_gates_all_exact"] = all(g["exact_16_digits"] for g in gates.values())
    out["redundancy_off0_gate_all_exact"] = all(
        g["exact_16_digits"] for n, g in gates.items() if n.startswith("redundancy_off0"))

    # ---------------- matched-k deltas -------------------------------------
    matched = {}
    for split, res in (("MT", RES_MT), ("QA", RES_QA)):
        matched[split] = {}
        for off in offsets:
            row = {}
            for k in KS:
                r = loss["redundancy"][off][split].get(k)
                t = loss["tfidf"][off][split].get(k)
                if r is None or t is None:
                    continue
                d = r - t
                row[str(k)] = {
                    "redundancy": r, "tfidf": t, "delta": d,
                    "abs_over_resolution": abs(d) / res,
                    "clears_paired": abs(d) > res,
                    "clears_composite": abs(d) > (COMPOSITE_MT if split == "MT" else COMPOSITE_QA),
                    "sign": "redundancy_worse" if d > 0 else "redundancy_better",
                }
            matched[split][str(off)] = row
    out["matched_k"] = matched

    dmt_all = [matched["MT"][str(o)][str(k)]["delta"] for o in offsets for k in KS
               if str(k) in matched["MT"][str(o)]]
    out["matched_k_MT_summary"] = {
        "n": len(dmt_all), "all_positive_redundancy_worse": all(d > 0 for d in dmt_all),
        "min": min(dmt_all), "max": max(dmt_all), "mean": statistics.fmean(dmt_all),
        "n_clearing_paired": sum(1 for d in dmt_all if abs(d) > RES_MT),
        "n_clearing_composite": sum(1 for d in dmt_all if abs(d) > COMPOSITE_MT),
        "min_abs_over_resolution": min(abs(d) for d in dmt_all) / RES_MT,
    }

    # ---------------- own per-seed argmin ----------------------------------
    own = {}
    for split, res in (("MT", RES_MT), ("QA", RES_QA)):
        own[split] = {}
        for off in offsets:
            rk, rv, rgap = argmin_info(loss["redundancy"][off][split])
            tk, tv, tgap = argmin_info(loss["tfidf"][off][split])
            d = rv - tv
            own[split][str(off)] = {
                "redundancy_argmin_k": rk, "redundancy_best": rv,
                "redundancy_gap_to_runner_up": rgap,
                "tfidf_argmin_k": tk, "tfidf_best": tv,
                "tfidf_gap_to_runner_up": tgap,
                "delta": d, "abs_over_resolution": abs(d) / res,
                "clears_paired": abs(d) > res,
                "clears_composite": abs(d) > (COMPOSITE_MT if split == "MT" else COMPOSITE_QA),
            }
        ds = [own[split][str(o)]["delta"] for o in offsets]
        own[split]["summary"] = {
            "per_offset": ds, "mean": statistics.fmean(ds),
            "range": max(ds) - min(ds), "all_same_sign": all(d > 0 for d in ds) or all(d < 0 for d in ds),
            "all_clear_paired": all(abs(d) > res for d in ds),
            "all_clear_composite": all(abs(d) > (COMPOSITE_MT if split == "MT" else COMPOSITE_QA) for d in ds),
            "min_abs_over_resolution": min(abs(d) for d in ds) / res,
        }
    out["own_per_seed_argmin"] = own

    # ---------------- argmin stability -------------------------------------
    stab = {}
    for sel in ("redundancy", "tfidf"):
        stab[sel] = {}
        for split in ("MT", "QA"):
            ks_ = {}
            for off in offsets:
                k, v, gap = argmin_info(loss[sel][off][split])
                ks_[str(off)] = {"argmin_k": k, "value": v, "gap_to_runner_up": gap,
                                 "gap_over_resolution": (gap / (RES_MT if split == "MT" else RES_QA))
                                 if gap is not None else None}
            vals = [ks_[str(o)]["value"] for o in offsets]
            argks = [ks_[str(o)]["argmin_k"] for o in offsets]
            stab[sel][split] = {
                "per_offset": ks_,
                "argmin_k_moves_across_seeds": len(set(argks)) > 1,
                "argmin_ks": argks,
                "minimum_VALUE_mean": statistics.fmean(vals),
                "minimum_VALUE_range": max(vals) - min(vals),
                "minimum_VALUE_range_below_paired_resolution":
                    (max(vals) - min(vals)) < (RES_MT if split == "MT" else RES_QA),
            }
    out["argmin_stability"] = stab

    # ---------------- across-seed spread -----------------------------------
    spread = {}
    for sel in ("redundancy", "tfidf"):
        spread[sel] = {}
        for split in ("MT", "QA"):
            per_k = {}
            for k in KS:
                vals = [loss[sel][o][split][k] for o in offsets if k in loss[sel][o][split]]
                if len(vals) == 3:
                    per_k[str(k)] = {"values": vals, "mean": statistics.fmean(vals),
                                     "range": max(vals) - min(vals),
                                     "range_over_paired_resolution":
                                         (max(vals) - min(vals)) / (RES_MT if split == "MT" else RES_QA)}
            ownvals = [argmin_info(loss[sel][o][split])[1] for o in offsets]
            spread[sel][split] = {
                "per_k": per_k,
                "at_own_per_seed_argmin": {
                    "values": ownvals, "mean": statistics.fmean(ownvals),
                    "range": max(ownvals) - min(ownvals),
                    "range_over_paired_resolution":
                        (max(ownvals) - min(ownvals)) / (RES_MT if split == "MT" else RES_QA)},
            }
    spread["_reference"] = {
        "MECH-SEED_keys_arm_k12_MT_range": 0.0093886852264404,
        "note": "MECH-SEED's finding: the keys arm's across-seed MT spread at its operating point (k=12) was 0.0094.",
    }
    out["across_seed_spread"] = spread

    # ---------------- adversarial pairing ----------------------------------
    adv = {}
    for split, res in (("MT", RES_MT), ("QA", RES_QA)):
        r_own = {o: argmin_info(loss["redundancy"][o][split]) for o in offsets}
        t_own = {o: argmin_info(loss["tfidf"][o][split]) for o in offsets}
        r_best_off = min(offsets, key=lambda o: r_own[o][1])   # redundancy's BEST seed
        t_worst_off = max(offsets, key=lambda o: t_own[o][1])  # tfidf's WORST seed
        d = r_own[r_best_off][1] - t_own[t_worst_off][1]
        # also: redundancy's best value anywhere on any curve vs tfidf's worst-anywhere minimum
        adv[split] = {
            "redundancy_best_seed": {"offset": r_best_off, "k": r_own[r_best_off][0],
                                     "value": r_own[r_best_off][1]},
            "tfidf_worst_seed": {"offset": t_worst_off, "k": t_own[t_worst_off][0],
                                 "value": t_own[t_worst_off][1]},
            "delta": d, "abs_over_resolution": abs(d) / res,
            "clears_paired": abs(d) > res,
            "clears_composite": abs(d) > (COMPOSITE_MT if split == "MT" else COMPOSITE_QA),
            "sign": "redundancy_worse" if d > 0 else "redundancy_better",
        }
        # strictest form: redundancy's lowest value at ANY (seed, k) vs tfidf's highest per-seed minimum
        r_any = min(loss["redundancy"][o][split][k] for o in offsets for k in loss["redundancy"][o][split])
        adv[split]["redundancy_lowest_at_any_seed_and_k"] = r_any
        adv[split]["delta_lowest_redundancy_vs_worst_tfidf_minimum"] = r_any - t_own[t_worst_off][1]
    out["adversarial_pairing"] = adv

    # ---------------- selection / routing mass -----------------------------
    ap = os.path.join(RESDIR, "arms_analysis.json")
    if os.path.exists(ap):
        arms = json.load(open(ap))
        out["selection_and_mass"] = arms
        pts = []
        for arm, r in arms.items():
            if "error" in r or "realised_frac_writable_MT_routing_mass" not in r:
                continue
            sel = r.get("selector") or ("tfidf" if arm.startswith("T_") else "redundancy")
            off = r.get("seed_offset")
            if off is None:
                m = re.search(r"off(\d+)", arm)
                off = int(m.group(1)) if m else 0
            best = loss.get(sel, {}).get(off, {}).get("MT")
            if not best:
                continue
            pts.append({"arm": arm, "selector": sel, "offset": off,
                        "MT_routing_mass": r["realised_frac_writable_MT_routing_mass"],
                        "best_MT": min(best.values())})
        if len(pts) >= 3:
            xs = [math.log(p["MT_routing_mass"]) for p in pts]
            ys = [p["best_MT"] for p in pts]
            mx, my = statistics.fmean(xs), statistics.fmean(ys)
            num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
            den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
            out["bandwidth_vs_acquisition_at_new_seeds"] = {
                "points": pts, "n": len(pts),
                "pearson_logMTmass_vs_bestMT": (num / den) if den else None,
                "MECH-INFOGATE_reference_pearson": -0.8766490587087726,
                "_definition": "does the realised write bandwidth still predict acquisition once the "
                               "seed is varied? MECH-INFOGATE measured -0.877 across its six selectors.",
            }

    dest = os.path.join(REPO, "research_loop/state/diagnostics/VERIFY-GATE.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: out[k] for k in
                      ("reproduction_gates_all_exact", "redundancy_off0_gate_all_exact",
                       "matched_k_MT_summary")}, indent=2))
    print("\nOWN-ARGMIN MT:", json.dumps(out["own_per_seed_argmin"]["MT"]["summary"], indent=2))
    print("\nADVERSARIAL MT:", json.dumps(out["adversarial_pairing"]["MT"], indent=2))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
