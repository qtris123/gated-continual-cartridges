"""DIAG-CONTENT finalizer: assemble the per-arm bundle, dump diagnostics, log wandb.

Reads (all produced by launch_diag_content.sh):
  results/DIAG-CONTENT/curve.tsv          per-k standalone evals, arms B and C + phase1
  results/DIAG-CONTENT/solve_stats.json   solve-side stats for arms A/B/C
  state/diagnostics/DIAG-CONTENT_route_mass.json   eval-time mass_on_S / cartridge mass
  results/DIAG-CONTENT/run_arm{B,C}.log   in-run eval losses + wandb URLs

Writes state/diagnostics/DIAG-CONTENT.json, tags the per-eval wandb runs `diagnostic`,
and logs one summary wandb run under group B-OVERWRITE.

Arm A (canonical MT corpus) is REUSED, not re-run: its per-k curve comes from
DIAG-SEQUENCE and its run dir from DIAG-OVERWRITE.

No source file is touched; this only reads artefacts.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

REPO = Path("/localhome/local-triv/gated-continual-cartridges_explore")
RESDIR = REPO / "research_loop/results/DIAG-CONTENT"
DIAGDIR = REPO / "research_loop/state/diagnostics"
SNAP = Path("/tmp/amsnap_content")

# ---------------------------------------------------------------- reference --
# arm A: canonical MT-corpus write. Standalone-eval convention (same harness/script
# as this task's per-k evals), from DIAG-SEQUENCE.
ARM_A_MT = {0: 3.7825491428375244, 1: 3.7445480823516846, 4: 3.0337772369384766,
            8: 2.709873914718628, 12: 2.435236930847168, 16: 2.5524158477783203}
ARM_A_QA = {0: 2.23880672454834, 1: 2.2298452854156494, 4: 2.113368272781372,
            8: 2.0471396446228027, 12: 2.0421719551086426, 16: 2.1771795749664307}
ARM_A_INRUN = {"qa_forgetting": 2.1766157150268555, "mt_acquisition": 2.5483615398406982}
PHASE1 = {"QA": 2.23880672454834, "MT": 3.7825491428375244}
NOISE = 0.15

# ---------------------------------------------------------------- curve.tsv --
rows = []
with open(RESDIR / "curve.tsv") as f:
    f.readline()
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 6:
            continue
        arm, k, split, loss, ckpt, url = parts[:6]
        rows.append(dict(arm=arm, k=int(k), split=split,
                         loss=float(loss) if loss != "MISSING" else None,
                         ckpt=ckpt, url=url))

curves: dict[str, dict[str, dict[int, float]]] = {}
for r in rows:
    curves.setdefault(r["arm"], {}).setdefault(r["split"], {})[r["k"]] = r["loss"]

phase1_session = curves.get("phase1", {})

# ------------------------------------------------- in-run evals + wandb URLs --
def parse_run_log(path: Path) -> dict:
    if not path.exists():
        return {}
    txt = path.read_text(errors="replace")
    losses = re.findall(r"Eval loss - ([0-9.]+)", txt)
    urls = re.findall(r"https://wandb\.ai/[^\s]+/runs/([A-Za-z0-9]+)", txt)
    ndocs = re.search(r"Per-document AM: (\d+) unique documents", txt)
    return {
        "in_run_eval_losses_in_order": [float(x) for x in losses],
        "wandb_run_id": urls[0] if urls else None,
        "wandb_run_url": (f"https://wandb.ai/vqtri-purdue-university/SEACrowd/runs/{urls[0]}"
                          if urls else None),
        "n_documents_written": int(ndocs.group(1)) if ndocs else None,
    }


run_meta = {arm: parse_run_log(RESDIR / f"run_{arm}.log") for arm in ("armB", "armC")}

solve = json.load(open(RESDIR / "solve_stats.json")) if (RESDIR / "solve_stats.json").exists() else {"arms": {}}
route_path = DIAGDIR / "DIAG-CONTENT_route_mass.json"
route = json.load(open(route_path)) if route_path.exists() else {"runs": {}}


def route_summary(label: str) -> dict:
    out = {}
    for split in ("QA", "MT"):
        r = route.get("runs", {}).get(f"{label}|{split}")
        if r:
            out[split] = {
                "mean_mass_on_S": r["summary"]["mean_mass_on_S"],
                "mean_mass_on_S_scored": r["summary"]["mean_mass_on_S_scored"],
                "mean_mass_on_cart": r["summary"]["mean_mass_on_cart"],
                "n_slots_per_layer_min": min(r["n_slots_per_layer"].values()),
                "n_slots_per_layer_max": max(r["n_slots_per_layer"].values()),
            }
    return out


ARM_LABEL = {"armA_canonical_MT": "armA", "armB_QAcorpus": "armB", "armC_MT_reversed": "armC"}

arms_out: dict[str, dict] = {}
for solve_key, label in ARM_LABEL.items():
    s = solve.get("arms", {}).get(solve_key, {})
    ps = (s.get("phase2_summary") or {})
    ev = ps.get("eval_metrics", {}) or {}
    entry = {
        "run_dir": s.get("run_dir"),
        "n_documents_written": s.get("n_documents"),
        "mean_mse_over_documents": s.get("mean_mse_over_documents"),
        "mass_on_S_solve_side_mean_over_layers": s.get("mass_on_S_mean_over_layers"),
        "mass_on_S_solve_side_min_max": s.get("mass_on_S_min_max_over_layers"),
        "v_absmax_global": (ps.get("value_norms", {}) or {}).get("global_max_abs"),
        "solve_s_wall_clock": s.get("wall_clock_s"),
        "doc_total_s_sum": s.get("doc_total_s_sum"),
        "prefill_s_total": s.get("prefill_s_total"),
        "gradient_steps": 0,
        "in_run_eval": {k: (ev.get(k, {}) or {}).get("loss") for k in ("qa_forgetting", "mt_acquisition")},
        "eval_time_route_mass": route_summary(label),
        "per_document_mean_mse": [d["mean_mse"] for d in s.get("per_document", [])],
        "per_document_mass_on_S": [d.get("mass_on_S_mean_over_layers") for d in s.get("per_document", [])],
        "per_document_v_absmax": [d.get("v_absmax_over_layers") for d in s.get("per_document", [])],
    }
    if label == "armA":
        entry["standalone_eval_curve_MT"] = {str(k): v for k, v in ARM_A_MT.items()}
        entry["standalone_eval_curve_QA"] = {str(k): v for k, v in ARM_A_QA.items()}
        entry["provenance"] = "REUSED: run dir from DIAG-OVERWRITE, per-k curve from DIAG-SEQUENCE"
        entry["in_run_eval_reference"] = ARM_A_INRUN
    else:
        entry["standalone_eval_curve_MT"] = {str(k): v for k, v in sorted(curves.get(label, {}).get("MT", {}).items())}
        entry["standalone_eval_curve_QA"] = {str(k): v for k, v in sorted(curves.get(label, {}).get("QA", {}).items())}
        entry.update({k: v for k, v in run_meta.get(label, {}).items()
                      if k in ("wandb_run_id", "wandb_run_url")})
        if run_meta.get(label, {}).get("n_documents_written"):
            entry["n_documents_written_from_log"] = run_meta[label]["n_documents_written"]
    arms_out[label] = entry

# ------------------------------------------------------- the headline split --
mtA0, mtA16 = ARM_A_MT[0], ARM_A_MT[16]
total_mt_gain = mtA0 - mtA16          # 1.2301 -- the number being decomposed


def gain(mt16: float | None) -> dict:
    if mt16 is None:
        return {}
    g = mtA0 - mt16
    return {
        "mt_final": mt16,
        "mt_improvement_vs_phase1": g,
        "fraction_of_canonical_mt_improvement": g / total_mt_gain,
        "inside_noise_band": abs(g) < NOISE,
    }


headline = {
    "canonical_total_mt_improvement_k0_to_k16": total_mt_gain,
    "armA_canonical": gain(ARM_A_MT[16]),
    "armB_content_free": gain(arms_out["armB"]["standalone_eval_curve_MT"].get("16")),
    "armC_reversed_order": gain(arms_out["armC"]["standalone_eval_curve_MT"].get("16")),
    "noise_band_loss": NOISE,
}
mtB_best = arms_out["armB"]["standalone_eval_curve_MT"]
if mtB_best:
    kbest = min(mtB_best, key=lambda k: mtB_best[k])
    headline["armB_best_over_measured_k"] = {"k": int(kbest), "mt": mtB_best[kbest],
                                             "fraction_of_canonical_mt_improvement":
                                             (mtA0 - mtB_best[kbest]) / total_mt_gain}
mtC = arms_out["armC"]["standalone_eval_curve_MT"]
if mtC:
    kbest = min(mtC, key=lambda k: mtC[k])
    headline["armC_best_over_measured_k"] = {"k": int(kbest), "mt": mtC[kbest],
                                             "fraction_of_canonical_mt_improvement":
                                             (mtA0 - mtC[kbest]) / total_mt_gain}

# ---------------------------------------------------------------- provenance --
def manifest() -> str:
    cmd = ("find cartridges examples -type f -name '*.py' | LC_ALL=C sort | xargs sha256sum "
           "| awk '{print $1\"  \"$2}' | LC_ALL=C sort | sha256sum | cut -d' ' -f1")
    return subprocess.run(["bash", "-lc", cmd], cwd=SNAP, capture_output=True, text=True).stdout.strip()


wrapper = (RESDIR / "wrapper.log").read_text(errors="replace")
prov = {
    "snapshot_path": str(SNAP),
    "snapshot_manifest_at_launch": (re.search(r"SNAPSHOT_MANIFEST=(\w+)", wrapper) or [None, None])[1],
    "snapshot_manifest_after_run": (re.search(r"SNAPSHOT_MANIFEST_AFTER=(\w+)", wrapper) or [None, None])[1],
    "snapshot_manifest_now": manifest(),
    "verified_import_path": (re.search(r"CARTRIDGES_IMPORT_PATH=(\S+)", wrapper) or [None, None])[1],
    "repo_head_at_launch": (re.search(r"REPO_HEAD_AT_LAUNCH=(\w+)", wrapper) or [None, None])[1],
    "repo_head_after": (re.search(r"REPO_HEAD_AFTER=(\w+)", wrapper) or [None, None])[1],
    "gpu_claimed": (re.search(r"DIAG_CONTENT_CLAIMED_GPU=(\d+)", wrapper) or [None, None])[1],
    "source_files_edited": "none",
    "data_artefact_created": str(REPO / "data/qasper/train/DIAG-CONTENT_qwen_qasper_MT_task_8192_revorder.parquet"),
}
for arm in ("armB", "armC"):
    m = re.search(rf"DIAG_CONTENT_TRAIN_{arm}_START_EPOCH=(\d+)", wrapper)
    n = re.search(rf"DIAG_CONTENT_TRAIN_{arm}_END_EPOCH=(\d+)", wrapper)
    if m and n:
        arms_out[arm]["phase2_e2e_s"] = int(n.group(1)) - int(m.group(1))

out = {
    "id": "DIAG-CONTENT",
    "board_entry": "B-ROUTE / B-OVERWRITE",
    "question": ("is AM's Phase-2 write storing document CONTENT, or is most of the measured MT "
                 "'acquisition' a content-free distribution/format shift from perturbing the "
                 "cartridge at all?"),
    "phase1_floor": PHASE1,
    "phase1_floor_same_session": {s: phase1_session.get(s, {}).get(0) for s in ("QA", "MT")},
    "noise_band_loss": NOISE,
    "headline": headline,
    "arms": arms_out,
    "per_eval_wandb_urls": {f"{r['arm']}_k{r['k']}_{r['split']}": r["url"] for r in rows},
    "eval_corpus_overlap": {
        "MT_eval_titles_vs_MT_train_docs": 16,
        "MT_eval_titles_vs_QA_train_docs": 0,
        "QA_eval_titles_vs_QA_train_docs": 16,
        "QA_eval_titles_vs_MT_train_docs": 0,
        "note": ("the MT eval's 16 papers are exactly the 16 MT-corpus documents and share NOTHING "
                 "with the QA corpus, so arm B's MT number is content-free by construction; the QA "
                 "eval's 16 papers are exactly the 16 QA-corpus documents, so arm B's QA number is "
                 "NOT a retention measure -- it is re-writing the very papers the QA eval scores"),
    },
    "provenance": prov,
}

DIAGDIR.mkdir(parents=True, exist_ok=True)
json.dump(out, open(DIAGDIR / "DIAG-CONTENT.json", "w"), indent=1)
print("WROTE", DIAGDIR / "DIAG-CONTENT.json")
print(json.dumps(headline, indent=1))

# ------------------------------------------------------------------ wandb ----
import wandb  # noqa: E402

ent = os.environ["CARTRIDGES_WANDB_ENTITY"]
proj = os.environ["CARTRIDGES_WANDB_PROJECT"]
api = wandb.Api()
tagged = 0
ids = [r["url"].rsplit("/", 1)[-1] for r in rows if r["url"] and r["url"] != "MISSING"]
ids += [v["wandb_run_id"] for v in run_meta.values() if v.get("wandb_run_id")]
for rid in ids:
    try:
        run = api.run(f"{ent}/{proj}/{rid}")
        if "DIAG-CONTENT" not in run.tags:
            run.tags = list(run.tags) + ["diagnostic", "DIAG-CONTENT"]
            run.update()
        tagged += 1
    except Exception as e:  # noqa: BLE001
        print("TAG_FAIL", rid, e)
print(f"tagged {tagged}/{len(ids)} runs")

summary = wandb.init(
    entity=ent, project=proj, name="DIAG-CONTENT_summary",
    group="B-OVERWRITE", tags=["diagnostic", "DIAG-CONTENT", "B-OVERWRITE", "verify"],
    notes="content-free write control: QA-corpus documents, MT eval",
    config={"top_t": 32, "granularity": "per_layer", "slot_selection": "tfidf", "use_idf": 0,
            "key_mode": "freeze", "target_mode": "cartridge_plus_doc", "ridge_lambda": 1e-4,
            "ridge_scale": "spectral", "delta_weight": 1e-2, "max_queries_per_head": 64,
            "max_ref_examples_per_doc": 32, "n_documents": 16,
            "armA_synth": "qwen_qasper_MT_task_8192.parquet (reused)",
            "armB_synth": "qwen_qasper_QA_task_8192.parquet",
            "armC_synth": "DIAG-CONTENT_qwen_qasper_MT_task_8192_revorder.parquet"},
)
tbl = wandb.Table(columns=["arm", "k", "mt", "qa"])
for label, curve_mt, curve_qa in (
    ("armA", arms_out["armA"]["standalone_eval_curve_MT"], arms_out["armA"]["standalone_eval_curve_QA"]),
    ("armB", arms_out["armB"]["standalone_eval_curve_MT"], arms_out["armB"]["standalone_eval_curve_QA"]),
    ("armC", arms_out["armC"]["standalone_eval_curve_MT"], arms_out["armC"]["standalone_eval_curve_QA"]),
):
    for k in sorted(curve_mt, key=int):
        tbl.add_data(label, int(k), curve_mt.get(k), curve_qa.get(k))
summary.log({"diag/curves": tbl})
flat = {}
for label in ("armA", "armB", "armC"):
    a = arms_out[label]
    flat[f"diag/{label}/mt_k16"] = a["standalone_eval_curve_MT"].get("16")
    flat[f"diag/{label}/qa_k16"] = a["standalone_eval_curve_QA"].get("16")
    flat[f"diag/{label}/mean_mse"] = a["mean_mse_over_documents"]
    flat[f"diag/{label}/mass_on_S_solve"] = a["mass_on_S_solve_side_mean_over_layers"]
    flat[f"diag/{label}/v_absmax"] = a["v_absmax_global"]
    flat[f"diag/{label}/solve_s"] = a["solve_s_wall_clock"]
    rm = a.get("eval_time_route_mass", {})
    for split in ("QA", "MT"):
        if split in rm:
            flat[f"diag/{label}/eval_mass_on_S_{split}"] = rm[split]["mean_mass_on_S"]
            flat[f"diag/{label}/eval_cart_mass_{split}"] = rm[split]["mean_mass_on_cart"]
for key in ("armA_canonical", "armB_content_free", "armC_reversed_order"):
    if headline.get(key):
        flat[f"diag/frac_of_canonical_gain/{key}"] = headline[key]["fraction_of_canonical_mt_improvement"]
summary.summary.update({k: v for k, v in flat.items() if v is not None})
url, rid = summary.url, summary.id
summary.finish()
print("SUMMARY_WANDB_URL=", url)
print("SUMMARY_WANDB_ID=", rid)
json.dump({"summary_wandb_url": url, "summary_wandb_id": rid},
          open(RESDIR / "summary_wandb.json", "w"), indent=1)
