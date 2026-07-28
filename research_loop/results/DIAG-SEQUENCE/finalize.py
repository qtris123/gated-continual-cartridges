"""DIAG-SEQUENCE finalizer: build the acquisition curve, dump diagnostics, log to wandb.

Reads research_loop/results/DIAG-SEQUENCE/curve.tsv (written by launch_diag_sequence.sh),
joins it with the document-order / eval-coverage metadata, writes
research_loop/state/diagnostics/DIAG-SEQUENCE.json, adds the `diagnostic` tag to the
per-eval wandb runs, and logs one summary wandb run under group B-OVERWRITE.

No source file is touched; this only reads artefacts.
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path

REPO = Path("/localhome/local-triv/gated-continual-cartridges_explore")
RESDIR = REPO / "research_loop/results/DIAG-SEQUENCE"
SNAP = Path("/tmp/amsnap_seq")

# ---------------------------------------------------------------- curve.tsv --
rows = []
with open(RESDIR / "curve.tsv") as f:
    header = f.readline()
    for line in f:
        k, split, loss, ckpt, url = line.rstrip("\n").split("\t")
        rows.append(dict(k=int(k), split=split, loss=float(loss), ckpt=ckpt, url=url))

by_k = {}
for r in rows:
    by_k.setdefault(r["k"], {})[r["split"]] = r

ks = sorted(by_k)
qa = {k: by_k[k]["QA"]["loss"] for k in ks if "QA" in by_k[k]}
mt = {k: by_k[k]["MT"]["loss"] for k in ks if "MT" in by_k[k]}

# ---------------------------------------------------------- correctness gate --
GATE_QA, GATE_MT = 2.1772, 2.5524
gate = {
    "expected_qa_doc16": GATE_QA,
    "expected_mt_doc16": GATE_MT,
    "observed_qa_doc16": qa.get(16),
    "observed_mt_doc16": mt.get(16),
    "qa_matches_4dp": abs(qa.get(16, 1e9) - GATE_QA) < 5e-5,
    "mt_matches_4dp": abs(mt.get(16, 1e9) - GATE_MT) < 5e-5,
    "expected_qa_phase1": 2.23880672454834,
    "expected_mt_phase1": 3.7825491428375244,
    "observed_qa_phase1": qa.get(0),
    "observed_mt_phase1": mt.get(0),
}

# ------------------------------------------------------------- curve shape ---
mt0, mt16 = mt.get(0), mt.get(16)
total_drop = (mt0 - mt16) if (mt0 is not None and mt16 is not None) else None
curve = []
for k in ks:
    row = {"k": k, "qa": qa.get(k), "mt": mt.get(k)}
    if k > 0 and (k - 1) in mt:
        row["d_mt_from_prev"] = mt[k] - mt[k - 1]
    if total_drop:
        row["frac_of_total_mt_drop_achieved"] = (mt0 - mt[k]) / total_drop
    curve.append(row)

frac_after_doc1 = (mt0 - mt[1]) / total_drop if total_drop else None

# Two competing models for the curve shape, both anchored at k=0 and k=16:
#   (A) "documents accumulate": every written document persists, and each eval question
#       is answerable once its own document has been written -> MT(k) tracks the
#       cumulative fraction of eval questions whose document is in 1..k.
#   (B) "only the last document survives": whatever k is, the cartridge holds ~1 document,
#       so MT(k) stays near MT(0) for every k < 16.
cov = json.load(open("/tmp/diagseq_eval_coverage.json"))
cumfrac = {r["k"]: r["cum_frac_docs_1..k"] for r in cov["MT_per_k"]}
cumfrac[0] = 0.0
for row in curve:
    k = row["k"]
    if total_drop and k in cumfrac:
        row["mt_pred_accumulate"] = mt0 - cumfrac[k] * total_drop
        row["mt_pred_one_doc_only"] = mt0 - (cov["MT_per_k"][k - 1]["n_eval_questions"] / cov["MT_total"]) * total_drop if k > 0 else mt0
        row["resid_vs_accumulate"] = row["mt"] - row["mt_pred_accumulate"]

# ------------------------------------------------------------- provenance ----
def manifest():
    out = subprocess.run(
        "find cartridges examples -type f -name '*.py' | LC_ALL=C sort | xargs sha256sum "
        "| awk '{print $1\"  \"$2}' | LC_ALL=C sort | sha256sum | cut -d' ' -f1",
        shell=True, cwd=SNAP, capture_output=True, text=True,
    )
    return out.stdout.strip()


head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                      capture_output=True, text=True).stdout.strip()

wrapper = (RESDIR / "wrapper.log").read_text()
head_at_launch = next((l.split("=", 1)[1] for l in wrapper.splitlines()
                       if l.startswith("REPO_HEAD_AT_LAUNCH=")), None)
manifest_at_launch = next((l.split("=", 1)[1] for l in wrapper.splitlines()
                           if l.startswith("SNAPSHOT_MANIFEST=")), None)
gpu = next((l.split("=", 1)[1] for l in wrapper.splitlines()
            if l.startswith("SEQ_CLAIMED_GPU=")), None)

doc_order = json.load(open("/tmp/diagseq_doc_order.json"))
coverage = json.load(open("/tmp/diagseq_eval_coverage.json"))

# each eval log prints the checkpoint path it loaded -> verify
loaded_ok = {}
for r in rows:
    log = RESDIR / "evals" / f"eval_doc{r['k']}_{r['split']}.log"
    txt = log.read_text(errors="ignore")
    loaded_ok[f"doc{r['k']}_{r['split']}"] = r["ckpt"] in txt

out = {
    "id": "DIAG-SEQUENCE",
    "board_entry": "B-OVERWRITE",
    "question": (
        "Does MT acquisition improve as documents 2..16 are written, or is the final "
        "cartridge worth about one document? (Evaluates the per-document cache snapshots "
        "DIAG-OVERWRITE's canonical run already saved.)"
    ),
    "provenance": {
        "snapshot_path": str(SNAP),
        "snapshot_contents": f"git archive HEAD {{cartridges,examples}} taken at HEAD={head_at_launch}",
        "snapshot_manifest_sha256_at_launch": manifest_at_launch,
        "snapshot_manifest_sha256_after": manifest(),
        "snapshot_manifest_recipe": (
            "cd $SNAP && find cartridges examples -type f -name '*.py' | LC_ALL=C sort | "
            "xargs sha256sum | awk '{print $1\"  \"$2}' | LC_ALL=C sort | sha256sum"
        ),
        "verified_import_path": "/tmp/amsnap_seq/cartridges (printed from cwd=/tmp in wrapper.log)",
        "CARTRIDGES_DIR": str(SNAP),
        "repo_HEAD_at_launch": head_at_launch,
        "repo_HEAD_after": head,
        "gpu_claimed_via_flock": gpu,
        "source_run_dir": str(
            REPO / "outputs/2026-07-28-21-43-17-continual_am_sparse/4b0edcab-d1c5-4313-bc43-309d7ff77097"
        ),
        "source_run_bundle": "research_loop/results/DIAG-OVERWRITE/result.json",
        "snapshots_are_stock_artefacts": (
            "cache-after-doc-*.pt saved by SAVE_AFTER_EACH_DOCUMENT (default 1); no re-run, "
            "no instrumentation, no source edit"
        ),
        "instrumentation": "NONE",
        "eval_convention": (
            "standalone examples/qasper2/train/eval_forgetting.py, BATCH_SIZE=4 (default), "
            "MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507, one fresh process per (k, split); "
            "'Eval loss' mean-CE parsed from the log, then PID killed (RUNBOOK 1)"
        ),
        "each_eval_loaded_intended_snapshot": loaded_ok,
        "all_evals_loaded_intended_snapshot": all(loaded_ok.values()),
    },
    "correctness_gate": gate,
    "curve": curve,
    "qa_by_k": qa,
    "mt_by_k": mt,
    "wandb_urls": {f"doc{r['k']}_{r['split']}": r["url"] for r in rows},
    "checkpoints_evaluated": {str(r["k"]): r["ckpt"] for r in rows if r["split"] == "MT"},
    "document_write_order": doc_order,
    "eval_coverage": coverage,
    "summary_scalars": {
        "mt_phase1_k0": mt0,
        "mt_after_doc1": mt.get(1),
        "mt_after_doc16": mt16,
        "mt_total_drop_k0_to_k16": total_drop,
        "mt_drop_from_doc1_alone": (mt0 - mt[1]) if 1 in mt else None,
        "frac_of_total_mt_drop_from_doc1_alone": frac_after_doc1,
        "mt_drop_from_docs2_to_16": (mt[1] - mt16) if 1 in mt else None,
        "qa_phase1_k0": qa.get(0),
        "qa_after_doc1": qa.get(1),
        "qa_after_doc16": qa.get(16),
        "qa_min_over_k": min(qa.values()) if qa else None,
        "qa_argmin_over_k": min(qa, key=qa.get) if qa else None,
        "mean_abs_resid_vs_accumulate_model": (
            sum(abs(r["resid_vs_accumulate"]) for r in curve if "resid_vs_accumulate" in r)
            / max(1, sum(1 for r in curve if "resid_vs_accumulate" in r))
        ),
        "max_abs_resid_vs_accumulate_model": max(
            (abs(r["resid_vs_accumulate"]) for r in curve if "resid_vs_accumulate" in r),
            default=None,
        ),
    },
    "curve_models": {
        "accumulate": "MT(k) = MT(0) - cumfrac_docs_1..k * (MT(0)-MT(16)); every written document persists",
        "one_doc_only": "MT(k) = MT(0) - frac_of_eval_questions_about_doc_k * (MT(0)-MT(16)); only the last write survives",
    },
}

diagpath = REPO / "research_loop/state/diagnostics/DIAG-SEQUENCE.json"
diagpath.parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(diagpath, "w"), indent=1)
print("wrote", diagpath)

for k in ks:
    print(f"k={k:2d}  QA={qa.get(k)}  MT={mt.get(k)}")
print("GATE:", json.dumps(gate, indent=1))

# ------------------------------------------------------------------ wandb ----
import wandb

api = wandb.Api()
ent = os.environ["CARTRIDGES_WANDB_ENTITY"]
proj = os.environ["CARTRIDGES_WANDB_PROJECT"]
tagged = 0
for r in rows:
    rid = r["url"].rsplit("/", 1)[-1]
    try:
        run = api.run(f"{ent}/{proj}/{rid}")
        if "diagnostic" not in run.tags:
            run.tags = list(run.tags) + ["diagnostic", "DIAG-SEQUENCE"]
            run.update()
        tagged += 1
    except Exception as e:  # noqa: BLE001
        print("TAG_FAIL", rid, e)
print(f"tagged {tagged}/{len(rows)} per-eval runs with `diagnostic`")

summary = wandb.init(
    entity=ent, project=proj, name="DIAG-SEQUENCE_curve",
    group="B-OVERWRITE", tags=["diagnostic", "DIAG-SEQUENCE", "B-OVERWRITE"],
    notes="MT/QA vs number of documents written (per-document cache snapshots of the canonical top-32 run)",
    config={"top_t": 32, "granularity": "per_layer", "slot_selection": "tfidf", "use_idf": 0,
            "key_mode": "freeze", "ridge_lambda": 1e-4, "ridge_scale": "spectral",
            "delta_weight": 1e-2, "max_queries_per_head": 64, "n_documents": 16,
            "source_run": out["provenance"]["source_run_dir"]},
)
tbl = wandb.Table(columns=["k", "qa", "mt", "d_mt_from_prev", "frac_of_total_mt_drop"])
for row in curve:
    summary.log({"diag/k": row["k"], "diag/qa_loss": row["qa"], "diag/mt_loss": row["mt"]},
                step=row["k"])
    tbl.add_data(row["k"], row["qa"], row["mt"],
                 row.get("d_mt_from_prev"), row.get("frac_of_total_mt_drop_achieved"))
summary.log({"diag/curve": tbl})
summary.summary.update({f"diag/{k}": v for k, v in out["summary_scalars"].items() if v is not None})
summary.summary.update({
    "diag/gate_qa_doc16_matches": gate["qa_matches_4dp"],
    "diag/gate_mt_doc16_matches": gate["mt_matches_4dp"],
    "diag/mt_eval_questions_per_doc_min": coverage.get("MT_min_per_doc"),
    "diag/mt_eval_questions_per_doc_max": coverage.get("MT_max_per_doc"),
    "diag/mt_eval_n_examples": coverage.get("MT_total"),
})
url = summary.url
rid = summary.id
summary.finish()
print("SUMMARY_WANDB_URL=", url)
print("SUMMARY_WANDB_ID=", rid)
json.dump({"summary_wandb_url": url, "summary_wandb_id": rid},
          open(RESDIR / "summary_wandb.json", "w"), indent=1)
