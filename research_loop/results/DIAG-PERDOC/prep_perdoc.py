"""DIAG-PERDOC prep (CPU only, no source edits).

Builds, under /tmp/perdoc/:
  * synth/doc<IDX>.parquet        -- the MT synthesis rows of ONE document (all of them,
                                     so `canonical_document_prompt` merges the SAME sections
                                     as the in-sequence write).
  * synth/doc<IDX>_shuf.parquet   -- same rows, row order permuted (reference-draw control).
  * eval/mt_doc<IDX>.parquet      -- the MT eval questions whose metadata.paper_id belongs to
                                     that document.
  * eval/mt_union5.parquet        -- the union of the five chosen documents' eval questions.
  * meta.json                     -- doc order, paper_id map, per-subset scored-token counts.

Scored-token counts are computed with the SAME stock code path the eval harness uses
(`LossEvalDataset` -> elements -> topk_token_idxs), so the token-weighted decomposition
    L(full) = sum_d n_d * L_d / sum_d n_d
is exact (per-element attention is block-diagonal via `seq_ids`, and packing never splits
an element, so per-example CE is packing-invariant).
"""

import hashlib
import json
import os
import random
import re
from pathlib import Path

import pandas as pd

REPO = Path("/localhome/local-triv/gated-continual-cartridges_explore")
OUT = Path("/tmp/perdoc")
SYNTH = REPO / "data/qasper/train/qwen_qasper_MT_task_8192.parquet"
EVAL_MT = REPO / "data/qasper/eval/qasper_eval_MT.parquet"
CHOSEN = [0, 9, 11, 13, 15]  # doc_index (0-based) == k-1

(OUT / "synth").mkdir(parents=True, exist_ok=True)
(OUT / "eval").mkdir(parents=True, exist_ok=True)


def title_of(prompt: str) -> str:
    m = re.search(r"<title>(.*?)</title>", prompt or "", flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else (prompt or "")


# ------------------------------------------------------------------ documents --
synth = pd.read_parquet(SYNTH)
titles = synth["system_prompt"].map(title_of)
order = []  # first-appearance order == cartridges' dict-insertion order
for t in titles:
    if t not in order:
        order.append(t)
assert len(order) == 16, len(order)

# --------------------------------------------------------------- eval subsets --
ev = pd.read_parquet(EVAL_MT)
ev_titles = ev["metadata"].map(lambda m: m["title"].strip())
ev_pids = ev["metadata"].map(lambda m: m["paper_id"])

meta = {"documents": [], "chosen_doc_indices": CHOSEN}
for i, t in enumerate(order):
    slug = f"doc-{i:03d}-{hashlib.sha1(t.encode('utf-8')).hexdigest()[:8]}"
    mask_ev = ev_titles == t.strip()
    pids = sorted(set(ev_pids[mask_ev]))
    meta["documents"].append(
        {
            "doc_index": i,
            "k": i + 1,
            "slug": slug,
            "title": t,
            "paper_id": pids[0] if len(pids) == 1 else pids,
            "n_synth_rows": int((titles == t).sum()),
            "n_eval_questions": int(mask_ev.sum()),
        }
    )
    sub = ev[mask_ev].reset_index(drop=True)
    sub.to_parquet(OUT / f"eval/mt_doc{i:03d}.parquet")

union_mask = ev_titles.isin([order[i].strip() for i in CHOSEN])
ev[union_mask].reset_index(drop=True).to_parquet(OUT / "eval/mt_union5.parquet")
meta["n_union5_questions"] = int(union_mask.sum())

# --------------------------------------------------------------- synth subsets --
for i in CHOSEN:
    t = order[i]
    sub = synth[titles == t].reset_index(drop=True)
    sub.to_parquet(OUT / f"synth/doc{i:03d}.parquet")
    if i == CHOSEN[-1]:  # reference-draw control on the last (late) document
        perm = list(range(len(sub)))
        random.Random(20260729).shuffle(perm)
        sub.iloc[perm].reset_index(drop=True).to_parquet(OUT / f"synth/doc{i:03d}_shuf.parquet")

# ---------------------------------------------- canonical-prompt identity check --
import sys

sys.path.insert(0, "/tmp/amsnap_perdoc")
from cartridges.am.reference_data import (  # noqa: E402
    canonical_document_prompt,
    group_conversations_by_document,
    limit_conversations,
    load_conversations,
)

full_groups = group_conversations_by_document(load_conversations(str(SYNTH)))
full_titles = list(full_groups.keys())
prompt_ok = {}
ref_overlap = {}
for i in CHOSEN:
    ref_full = canonical_document_prompt(full_groups[full_titles[i]])
    solo = load_conversations(str(OUT / f"synth/doc{i:03d}.parquet"))
    solo_groups = group_conversations_by_document(solo)
    assert len(solo_groups) == 1, (i, len(solo_groups))
    ref_solo = canonical_document_prompt(list(solo_groups.values())[0])
    prompt_ok[str(i)] = {
        "identical": ref_full == ref_solo,
        "sha256_in_sequence": hashlib.sha256(ref_full.encode()).hexdigest()[:16],
        "sha256_solo": hashlib.sha256(ref_solo.encode()).hexdigest()[:16],
        "n_chars": len(ref_full),
    }
    # how much does the reference draw differ? in-sequence uses seed=doc_index,
    # a solo run uses seed=0 (the document is index 0 of its own parquet).
    a = limit_conversations(full_groups[full_titles[i]], 32, seed=i)
    b = limit_conversations(list(solo_groups.values())[0], 32, seed=0)
    ida = {id(x) for x in a}
    # compare by content hash, the objects differ across loads
    def h(c):
        return hashlib.sha1(
            (str(c.system_prompt) + "||" + str([m.content for m in c.messages])).encode()
        ).hexdigest()

    ha, hb = {h(x) for x in a}, {h(x) for x in b}
    ref_overlap[str(i)] = {"n_a": len(ha), "n_b": len(hb), "overlap": len(ha & hb)}

    if i == CHOSEN[-1]:
        shuf = load_conversations(str(OUT / f"synth/doc{i:03d}_shuf.parquet"))
        sg = list(group_conversations_by_document(shuf).values())[0]
        ref_shuf = canonical_document_prompt(sg)
        prompt_ok[f"{i}_shuf"] = {
            "identical": ref_full == ref_shuf,
            "sha256_solo": hashlib.sha256(ref_shuf.encode()).hexdigest()[:16],
        }
        hs = {h(x) for x in limit_conversations(sg, 32, seed=0)}
        ref_overlap[f"{i}_shuf_vs_solo"] = {"overlap": len(hb & hs), "n": len(hs)}

meta["canonical_prompt_identity"] = prompt_ok
meta["reference_draw_overlap_solo_vs_in_sequence"] = ref_overlap

# ------------------------------------------------- scored-token counts per subset --
from transformers import AutoTokenizer  # noqa: E402

from cartridges.datasets import DataSource, LossEvalDataset  # noqa: E402

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")


def scored_tokens(path: str):
    ds = LossEvalDataset(
        LossEvalDataset.Config(
            data_source=DataSource(path=path, type="local"),
            packed_seq_length=2048,
        ),
        tokenizer=tok,
        seed=0,
    )
    n = 0
    n_trunc = 0
    for e in ds.elements:
        idxs = e.topk_token_idxs
        n += int((idxs < 2048).sum()) if len(e.input_ids) > 2048 else int(len(idxs))
        n_trunc += int(len(e.input_ids) > 2048)
    return {"n_elements": len(ds.elements), "n_scored_tokens": n, "n_elements_over_2048": n_trunc}


counts = {"full_MT": scored_tokens(str(EVAL_MT))}
for i in range(16):
    counts[f"doc{i:03d}"] = scored_tokens(str(OUT / f"eval/mt_doc{i:03d}.parquet"))
counts["union5"] = scored_tokens(str(OUT / "eval/mt_union5.parquet"))
meta["scored_token_counts"] = counts
meta["token_count_decomposition_check"] = {
    "sum_per_doc": sum(counts[f"doc{i:03d}"]["n_scored_tokens"] for i in range(16)),
    "full_MT": counts["full_MT"]["n_scored_tokens"],
}

json.dump(meta, open(OUT / "meta.json", "w"), indent=1)
print(json.dumps({k: v for k, v in meta.items() if k != "documents"}, indent=1))
for d in meta["documents"]:
    print(d["doc_index"], d["slug"], d["paper_id"], "nq=", d["n_eval_questions"],
          "ntok=", counts[f"doc{d['doc_index']:03d}"]["n_scored_tokens"])
