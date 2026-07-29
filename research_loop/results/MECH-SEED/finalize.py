"""MECH-SEED finalizer: build result.json + state/diagnostics/MECH-SEED.json."""
import glob
import json
import os
import re
import statistics as st

import torch

REPO = "/localhome/local-triv/gated-continual-cartridges_explore"
RES = f"{REPO}/research_loop/results/MECH-SEED"
NOISE = 0.15  # RUNBOOK §1 band midpoint used elsewhere in the loop (0.1-0.2)

# reference numbers this task is gated on / compared against
REF = {
    ("keys", 16, "QA"): 2.034916400909424,
    ("keys", 16, "MT"): 2.330503463745117,
    ("keys", 12, "QA"): 1.9560121297836304,
    ("keys", 12, "MT"): 2.2720184326171875,
    ("control", 16, "QA"): 2.159724712371826,
    ("control", 16, "MT"): 2.529625177383423,
    ("control", 12, "QA"): 2.074141263961792,
    ("control", 12, "MT"): 2.4704575538635254,
    ("phase1", 0, "QA"): 2.23880672454834,
    ("phase1", 0, "MT"): 3.7825491428375244,
}

# ---------------------------------------------------------------- curve.tsv
rows = []
with open(f"{RES}/curve.tsv") as f:
    header = f.readline()
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) < 7:
            continue
        arm, off, k, split, loss, ckpt, url = p[:7]
        rows.append(
            {
                "arm": arm,
                "offset": int(off),
                "k": int(k),
                "split": split,
                "loss": None if loss == "MISSING" else float(loss),
                "ckpt": ckpt,
                "wandb_url": None if url == "MISSING" else url,
            }
        )

L = {(r["arm"], r["offset"], r["k"], r["split"]): r["loss"] for r in rows}
URL = {(r["arm"], r["offset"], r["k"], r["split"]): r["wandb_url"] for r in rows}
CKPT = {(r["arm"], r["offset"], r["k"], r["split"]): r["ckpt"] for r in rows}

# ------------------------------------------------------ run dirs + internals
rundirs = {}
with open(f"{RES}/rundirs.tsv") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 4:
            rundirs[p[0]] = {"dir": p[1], "rc": int(p[2]), "wall_s": int(p[3])}


def internals(d):
    p = os.path.join(d, "per_document_am_stats.pt")
    if not os.path.exists(p):
        return {}
    a = torch.load(p, weights_only=False)
    docs = a.get("per_document", [])
    mse = [x["mean_mse"] for x in docs]
    mass, vmax = [], []
    for x in docs:
        e = x.get("extra") or {}
        m = e.get("ref_mass_on_S_per_layer") or {}
        if m:
            mass.append(sum(m.values()) / len(m))
        v = e.get("v_selected_absmax_after_per_layer") or {}
        if v:
            vmax.append(max(v.values()))
    return {
        "n_documents": a.get("n_documents"),
        "seed_offset_recorded": a.get("seed_offset", 0),
        "wall_clock_s": a.get("wall_clock_s"),
        "mean_mse_over_docs": sum(mse) / len(mse) if mse else None,
        "mean_mse_per_doc": mse,
        "ref_mass_on_S_mean_over_layers_and_docs": (
            sum(mass) / len(mass) if mass else None
        ),
        "ref_mass_on_S_per_doc": mass,
        "v_selected_absmax_max": max(vmax) if vmax else None,
    }


def summary(d):
    p = os.path.join(d, "phase2_summary.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def wandb_url_of(tag):
    log = f"{RES}/runs/{tag}.log"
    if not os.path.exists(log):
        return None
    m = re.findall(r"https://wandb\.ai/\S+/runs/[A-Za-z0-9]+", open(log, errors="ignore").read())
    return m[-1] if m else None


runs = {}
for tag, meta in rundirs.items():
    runs[tag] = {
        **meta,
        "wandb_run_url": wandb_url_of(tag),
        "phase2_summary": summary(meta["dir"]),
        "internals": internals(meta["dir"]),
        "config_seed_offset": None,
    }
    cfg = os.path.join(meta["dir"], "config.yaml")
    if os.path.exists(cfg):
        m = re.search(r"^\s*seed_offset:\s*(-?\d+)\s*$", open(cfg).read(), re.M)
        runs[tag]["config_seed_offset"] = int(m.group(1)) if m else None

# ------------------------------------------------------ default preservation
def read_lines(p):
    return open(p).read().splitlines() if os.path.exists(p) else []


h1, h2, h3 = (read_lines(f"{RES}/hashes_{x}.txt") for x in ("A1", "A2", "A3"))
default_pres = {
    "n_artefacts_compared": len(h1),
    "artefacts": [x.split("  ")[-1] for x in h1],
    "flag_unset_vs_git_archive_HEAD_bit_identical": bool(h1) and h1 == h2,
    "flag_zero_vs_git_archive_HEAD_bit_identical": bool(h1) and h1 == h3,
    "hashes_A1_head_snapshot": h1,
    "hashes_A2_edited_flag_unset": h2,
    "hashes_A3_edited_flag_zero": h3,
    "config_yaml_diff_A1_vs_A2": read_lines(f"{RES}/config_diff_A1_A2.txt"),
}

# ------------------------------------------------------------- drawn-id probe
probe = json.load(open(f"{RES}/seed_offset_probe.json"))
probe_small = {
    "mt_parquet": probe["mt_parquet"],
    "max_ref_examples_per_doc": probe["max_ref_examples_per_doc"],
    "n_documents": probe["n_documents"],
    "n_conversations_per_document": probe["n_conversations_per_document"],
    "pairs": [
        {
            k: v
            for k, v in p.items()
        }
        for p in probe["pairs"]
    ],
}

# ------------------------------------------------------------------ analysis
OFFSETS = sorted({r["offset"] for r in rows if r["arm"] in ("keys", "control")})
KS = [12, 16]


def cell(arm, off, k, split):
    return L.get((arm, off, k, split))


by_arm = {}
for arm in ("keys", "control"):
    by_arm[arm] = {}
    for k in KS:
        for split in ("QA", "MT"):
            vals = [
                (off, cell(arm, off, k, split))
                for off in OFFSETS
                if cell(arm, off, k, split) is not None
            ]
            if not vals:
                continue
            v = [x[1] for x in vals]
            by_arm[arm][f"k{k}_{split}"] = {
                "per_offset": {str(o): x for o, x in vals},
                "mean": st.mean(v),
                "min": min(v),
                "max": max(v),
                "range": max(v) - min(v),
                "sd": st.stdev(v) if len(v) > 1 else 0.0,
                "n_offsets": len(v),
            }

deltas = {}
for k in KS:
    for split in ("QA", "MT"):
        per_off = {}
        for off in OFFSETS:
            a, b = cell("keys", off, k, split), cell("control", off, k, split)
            if a is not None and b is not None:
                per_off[str(off)] = a - b
        if not per_off:
            continue
        v = list(per_off.values())
        deltas[f"k{k}_d{split}"] = {
            "per_offset": per_off,
            "mean": st.mean(v),
            "min": min(v),
            "max": max(v),
            "range": max(v) - min(v),
            "sd": st.stdev(v) if len(v) > 1 else 0.0,
            "n_offsets": len(v),
            "all_same_sign": all(x < 0 for x in v) or all(x > 0 for x in v),
            "all_clear_noise_band": all(abs(x) > NOISE for x in v),
        }

gate = {
    f"k{k}_{s}": {
        "expected": REF[("keys", k, s)],
        "observed": cell("keys", 0, k, s),
        "exact": cell("keys", 0, k, s) is not None
        and abs(cell("keys", 0, k, s) - REF[("keys", k, s)]) < 1e-12,
    }
    for k in KS
    for s in ("QA", "MT")
}
gate["passed"] = all(g["exact"] for g in gate.values() if isinstance(g, dict))

session_gates = {
    "control_offset0_k16_QA": {"expected": REF[("control", 16, "QA")], "observed": cell("control", 0, 16, "QA")},
    "control_offset0_k16_MT": {"expected": REF[("control", 16, "MT")], "observed": cell("control", 0, 16, "MT")},
    "control_offset0_k12_QA": {"expected": REF[("control", 12, "QA")], "observed": cell("control", 0, 12, "QA")},
    "control_offset0_k12_MT": {"expected": REF[("control", 12, "MT")], "observed": cell("control", 0, 12, "MT")},
    "phase1_floor_QA": {"expected": REF[("phase1", 0, "QA")], "observed": cell("phase1", 0, 0, "QA")},
    "phase1_floor_MT": {"expected": REF[("phase1", 0, "MT")], "observed": cell("phase1", 0, 0, "MT")},
}
for g in session_gates.values():
    g["delta"] = (
        None if g["observed"] is None else g["observed"] - g["expected"]
    )

# checkpoint-provenance: every eval log must have mentioned its own checkpoint path
ckpt_ok = []
for r in rows:
    log = glob.glob(
        f"{RES}/evals/eval_{r['arm']}_off{r['offset']}_doc{r['k']}_{r['split']}.log"
    )
    ok = False
    if log:
        ok = r["ckpt"] in open(log[0], errors="ignore").read()
    ckpt_ok.append({"arm": r["arm"], "offset": r["offset"], "k": r["k"],
                    "split": r["split"], "ckpt_path_in_log": ok})

# mission target
TARGET_QA, TARGET_MT = 2.52, 2.02
best_mt = by_arm.get("keys", {}).get("k12_MT", {})

diag = {
    "id": "MECH-SEED",
    "mechanism": "MECH-007 (AM_SEED_OFFSET)",
    "default_preservation": default_pres,
    "drawn_id_overlap_probe": probe_small,
    "offset0_gate": gate,
    "same_session_gates": session_gates,
    "losses_long": rows,
    "losses_by_arm": by_arm,
    "keys_minus_control_deltas": deltas,
    "runs": runs,
    "eval_checkpoint_provenance": ckpt_ok,
    "noise_band_used": NOISE,
    "vs_mission_target": {
        "target_qa": TARGET_QA,
        "target_mt": TARGET_MT,
        "keys_k12_MT_mean_over_offsets": best_mt.get("mean"),
        "keys_k12_MT_worst_over_offsets": best_mt.get("max"),
        "mt_short_by_at_mean": (best_mt.get("mean") - TARGET_MT) if best_mt else None,
    },
}

os.makedirs(f"{REPO}/research_loop/state/diagnostics", exist_ok=True)
with open(f"{REPO}/research_loop/state/diagnostics/MECH-SEED.json", "w") as f:
    json.dump(diag, f, indent=1)
print("wrote state/diagnostics/MECH-SEED.json")

# ------------------------------------------------------------------- printout
print("\n=== losses (arm, offset) -> k12 QA/MT, k16 QA/MT ===")
for arm in ("keys", "control"):
    for off in OFFSETS:
        vals = [cell(arm, off, k, s) for k in KS for s in ("QA", "MT")]
        if all(v is None for v in vals):
            continue
        fmt = lambda v: "   n/a  " if v is None else f"{v:.5f}"
        print(f"{arm:8s} off={off:<5d} k12 QA {fmt(vals[0])} MT {fmt(vals[1])} | "
              f"k16 QA {fmt(vals[2])} MT {fmt(vals[3])}")
print("\n=== keys - control ===")
for key, d in deltas.items():
    print(f"{key}: per-offset {[f'{o}:{v:+.4f}' for o, v in d['per_offset'].items()]} "
          f"mean {d['mean']:+.4f} range {d['range']:.4f}")
print("\n=== across-seed spread (same arm/k, different offset) ===")
for arm in ("keys", "control"):
    for key, d in by_arm[arm].items():
        print(f"{arm:8s} {key}: mean {d['mean']:.5f} range {d['range']:.4f} "
              f"(min {d['min']:.5f} max {d['max']:.5f}, n={d['n_offsets']})")
print("\nGATE:", "PASS" if gate["passed"] else "FAIL")
print("BIT-IDENTICAL flag unset:", default_pres["flag_unset_vs_git_archive_HEAD_bit_identical"],
      "| flag=0:", default_pres["flag_zero_vs_git_archive_HEAD_bit_identical"])
