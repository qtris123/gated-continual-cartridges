"""Log DIAG-OVERWRITE's localizing scalars onto the run that produced them.

Resumes the training run (WANDB_RUN_ID), adds the `diagnostic` tag, and logs the
Q1/Q2/Q3/Q4 summaries under a `diag/` prefix.
"""

from __future__ import annotations

import json
import os

import wandb

RUN_ID = os.environ["WANDB_RUN_ID"]
ENTITY = os.environ.get("CARTRIDGES_WANDB_ENTITY", "vqtri-purdue-university")
PROJECT = os.environ.get("CARTRIDGES_WANDB_PROJECT", "SEACrowd")
DIAG_JSON = os.environ["DIAG_JSON"]

d = json.load(open(DIAG_JSON))
q1 = d["q1_collision"]["summary"]
q2 = d["q2_survival"]
q4 = d["q4_document_disagreement"]["summary"]
q3 = d.get("q3_slot_mass", {})

payload = {
    "diag/union_size_mean": q1["union_size_mean"],
    "diag/writes_per_slot_mean": q1["writes_per_slot_mean"],
    "diag/mean_pairwise_jaccard": q1["mean_pairwise_jaccard"],
    "diag/n_slots_ge2_mean": q1["n_slots_ge2_mean"],
    "diag/n_slots_ge4_mean": q1["n_slots_ge4_mean"],
    "diag/n_slots_ge8_mean": q1["n_slots_ge8_mean"],
    "diag/n_slots_all16_mean": q1["n_slots_all16_mean"],
    "diag/frac_writes_into_slots_ge8": q1["frac_of_writes_into_slots_ge8"],
    "diag/survival_mean_docs1_15": q2["summary"]["mean_survival_combinatorial_docs_1_to_15"],
    "diag/survival_doc1": q2["summary"]["survival_doc1"],
    "diag/survival_doc8": q2["summary"]["survival_doc8"],
    "diag/median_rel_drift_docs1_15": q2["summary"]["mean_median_rel_drift_docs_1_to_15"],
    "diag/value_cosine_docs1_15": q2["summary"]["mean_value_cosine_docs_1_to_15"],
    "diag/spearman_between_docs": q4["spearman_mean"],
    "diag/overlap_with_doc_independent_top32": q4["overlap_with_doc_independent_top32_mean"],
}
for split in ("MT", "QA"):
    s = (q3.get("splits") or {}).get(split, {}).get("summary")
    if s:
        payload[f"diag/{split}_share_cart_mass_on_written_union"] = s["share_union_norm"]
        payload[f"diag/{split}_share_cart_mass_on_massranked_top32"] = s["share_mass32_norm"]
        payload[f"diag/{split}_share_cart_mass_on_massranked_top32_writable"] = s[
            "share_mass32_writable_norm"
        ]
        payload[f"diag/{split}_share_cart_mass_on_frozen_sink"] = s["share_sink_norm"]
        payload[f"diag/{split}_mass_on_cart"] = s["mass_on_cart"]

for i, v in enumerate(q2["per_doc_mean_over_layers"]["survival_combinatorial"], start=1):
    payload[f"diag/survival_curve/doc{i:02d}"] = v

run = wandb.init(
    entity=ENTITY, project=PROJECT, id=RUN_ID, resume="must",
)
tags = set(run.tags or ())
tags.update({"diagnostic", "B-OVERWRITE"})
run.tags = tuple(sorted(tags))
wandb.log(payload, step=16)
wandb.summary.update(payload)
wandb.finish()
print(json.dumps(payload, indent=2))
print(f"[done] logged {len(payload)} scalars to {ENTITY}/{PROJECT}/{RUN_ID}")
