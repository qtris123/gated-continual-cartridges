"""MECH-007 sanity: does AM_SEED_OFFSET actually change the drawn conversations?

CPU-only. Replays exactly what `cartridges/am/continual.py` does to pick each
document's reference subset (`group_conversations_by_document` ->
`limit_conversations(..., seed=doc_idx + offset)`) on the real MT parquet, and
reports the per-document overlap of the drawn row-ids between offsets.
"""
import json
import os
import sys

from cartridges.am.reference_data import (
    group_conversations_by_document,
    limit_conversations,
    load_conversations,
)

MT = os.environ.get(
    "SYNTH_DATA_PATH", "data/qasper/train/qwen_qasper_MT_task_8192.parquet"
)
MAX_REF = int(os.environ.get("MAX_REF_EXAMPLES_PER_DOC", "32"))
OFFSETS = [int(x) for x in os.environ.get("OFFSETS", "0,1000,2000").split(",")]
OUT = os.environ.get("OUT_JSON", "research_loop/results/MECH-SEED/seed_offset_probe.json")

convos = load_conversations(MT)
groups = group_conversations_by_document(convos)
print(f"documents: {len(groups)}  conversations: {len(convos)}")

# id() is not stable across draws -> identify a conversation by its index inside
# its own document group, which is exactly what limit_conversations samples over.
drawn: dict[int, dict[int, list[int]]] = {}
n_avail: dict[int, int] = {}
for off in OFFSETS:
    drawn[off] = {}
    for doc_idx, (_doc_id, doc_convos) in enumerate(groups.items()):
        n_avail[doc_idx] = len(doc_convos)
        import random

        rng = random.Random(doc_idx + off)
        if len(doc_convos) <= MAX_REF:
            idxs = list(range(len(doc_convos)))
        else:
            idxs = sorted(rng.sample(range(len(doc_convos)), MAX_REF))
        # cross-check against the library function itself (object identity)
        lim = limit_conversations(doc_convos, MAX_REF, seed=doc_idx + off)
        by_id = {id(c): i for i, c in enumerate(doc_convos)}
        lib_idxs = sorted(by_id[id(c)] for c in lim)
        assert lib_idxs == idxs, (doc_idx, off, lib_idxs[:5], idxs[:5])
        drawn[off][doc_idx] = idxs

report = {
    "mt_parquet": MT,
    "max_ref_examples_per_doc": MAX_REF,
    "offsets": OFFSETS,
    "n_documents": len(groups),
    "n_conversations_per_document": n_avail,
    "pairs": [],
    "drawn_ids": {str(o): {str(d): v for d, v in drawn[o].items()} for o in OFFSETS},
}
base = OFFSETS[0]
for off in OFFSETS[1:]:
    per_doc = []
    for doc_idx in sorted(drawn[base]):
        a, b = set(drawn[base][doc_idx]), set(drawn[off][doc_idx])
        inter = len(a & b)
        per_doc.append(
            {
                "doc_idx": doc_idx,
                "n_available": n_avail[doc_idx],
                "n_drawn": len(a),
                "n_shared": inter,
                "n_differ": len(a) - inter,
                "jaccard": inter / len(a | b) if a | b else 1.0,
            }
        )
    ident = sum(1 for r in per_doc if r["n_differ"] == 0)
    report["pairs"].append(
        {
            "a": base,
            "b": off,
            "per_document": per_doc,
            "mean_shared_of_32": sum(r["n_shared"] for r in per_doc) / len(per_doc),
            "mean_differ_of_32": sum(r["n_differ"] for r in per_doc) / len(per_doc),
            "mean_jaccard": sum(r["jaccard"] for r in per_doc) / len(per_doc),
            "n_documents_with_identical_draw": ident,
        }
    )
    print(
        f"offset {base} vs {off}: mean shared "
        f"{report['pairs'][-1]['mean_shared_of_32']:.2f}/32, mean differ "
        f"{report['pairs'][-1]['mean_differ_of_32']:.2f}/32, identical docs {ident}"
    )

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(report, f, indent=1)
print("wrote", OUT)
