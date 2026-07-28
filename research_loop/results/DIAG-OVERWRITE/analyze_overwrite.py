"""DIAG-OVERWRITE (B-OVERWRITE) Q1/Q2/Q4: collision, survival, and doc-disagreement.

CPU only.  Reads the artefacts the canonical top-32 run already writes:
  am_doc_*.pt        -> per-document per-layer selected slots + the tf scores that chose them
  cache-after-*.pt   -> the cache snapshot after each document's write (16 of them)
plus the untouched Phase-1 cache as the "before document 1" reference.

Q1  per-layer union size, mean pairwise Jaccard, writes-per-slot histogram, and the
    number of slots written by >=2 / >=4 / >=8 documents.
Q2  survival: for each document d, the fraction of the slots it wrote that no later
    document re-selected (combinatorial), and how much of the value it actually wrote
    is still present at the end (numeric drift + projection of the surviving delta).
Q4  rank correlation between different documents' per-layer slot scores.

Usage (env): RUN_DIR, PHASE1_CACHE, OUT_JSON
"""

from __future__ import annotations

import json
import os
from glob import glob
from itertools import combinations

import numpy as np
import torch

RUN_DIR = os.environ["RUN_DIR"]
PHASE1_CACHE = os.environ["PHASE1_CACHE"]
OUT_JSON = os.environ["OUT_JSON"]


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average-tie ranks (scipy-free)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / d) if d > 0 else 0.0


def main():
    doc_files = sorted(glob(os.path.join(RUN_DIR, "am_doc_*.pt")))
    cache_files = sorted(glob(os.path.join(RUN_DIR, "cache-after-*.pt")))
    assert len(doc_files) == 16, f"expected 16 am_doc files, got {len(doc_files)}"
    assert len(cache_files) == len(doc_files), (len(cache_files), len(doc_files))
    n_docs = len(doc_files)

    sel: list[dict[int, list[int]]] = []   # sel[d][layer] = sorted trainable slot idxs
    tf: list[np.ndarray] = []              # tf[d] = (n_layers, n_tokens)
    tfidf: list[np.ndarray] = []
    for f in doc_files:
        p = torch.load(f, map_location="cpu", weights_only=False)
        mask = p["am_stats"].mask
        assert mask.granularity == "per_layer"
        sel.append(
            {
                int(l): sorted(int(x) for x in pos.flatten().tolist())
                for l, pos in mask.positions_per_layer.items()
            }
        )
        ri = p["ranking_info"]
        tf.append(ri.tf.float().numpy())
        tfidf.append(ri.tfidf.float().numpy())
    layers = sorted(sel[0].keys())
    n_layers = len(layers)
    n_tokens = tf[0].shape[1]
    top_t = len(sel[0][layers[0]])

    # ---------------- Q1: collision -------------------------------------------------
    q1_per_layer = []
    for l in layers:
        sets = [set(sel[d][l]) for d in range(n_docs)]
        union = sorted(set().union(*sets))
        counts = np.zeros(n_tokens, dtype=np.int64)
        for s in sets:
            for p in s:
                counts[p] += 1
        written = counts[counts > 0]
        jac = [
            len(sets[i] & sets[j]) / len(sets[i] | sets[j])
            for i, j in combinations(range(n_docs), 2)
        ]
        hist = np.bincount(counts, minlength=n_docs + 1)[: n_docs + 1]
        q1_per_layer.append(
            {
                "layer": int(l),
                "union_size": len(union),
                "writes_total": int(counts.sum()),
                "writes_per_slot_mean": float(written.mean()),
                "writes_per_slot_max": int(written.max()),
                "mean_pairwise_jaccard": float(np.mean(jac)),
                "max_pairwise_jaccard": float(np.max(jac)),
                "min_pairwise_jaccard": float(np.min(jac)),
                "n_slots_ge2": int((counts >= 2).sum()),
                "n_slots_ge4": int((counts >= 4).sum()),
                "n_slots_ge8": int((counts >= 8).sum()),
                "n_slots_all16": int((counts >= n_docs).sum()),
                "writes_per_slot_hist": hist.tolist(),  # index = #docs writing that slot
            }
        )
    hist_tot = np.sum([r["writes_per_slot_hist"] for r in q1_per_layer], axis=0)

    def mean_of(rows, k):
        return float(np.mean([r[k] for r in rows]))

    q1_summary = {
        "n_documents": n_docs,
        "top_t_per_doc": top_t,
        "n_tokens_trainable": int(n_tokens),
        "union_size_mean": mean_of(q1_per_layer, "union_size"),
        "union_size_min": int(min(r["union_size"] for r in q1_per_layer)),
        "union_size_max": int(max(r["union_size"] for r in q1_per_layer)),
        "writes_per_slot_mean": mean_of(q1_per_layer, "writes_per_slot_mean"),
        "writes_per_slot_max": int(max(r["writes_per_slot_max"] for r in q1_per_layer)),
        "mean_pairwise_jaccard": mean_of(q1_per_layer, "mean_pairwise_jaccard"),
        "n_slots_ge2_mean": mean_of(q1_per_layer, "n_slots_ge2"),
        "n_slots_ge4_mean": mean_of(q1_per_layer, "n_slots_ge4"),
        "n_slots_ge8_mean": mean_of(q1_per_layer, "n_slots_ge8"),
        "n_slots_all16_mean": mean_of(q1_per_layer, "n_slots_all16"),
        "writes_per_slot_hist_all_layers": hist_tot.tolist(),
        "frac_of_writes_into_slots_ge8": float(
            sum(i * hist_tot[i] for i in range(8, n_docs + 1)) / (n_docs * top_t * n_layers)
        ),
    }

    # ---------------- Q2: survival --------------------------------------------------
    # combinatorial: doc d's slot survives iff no later document selects it
    surv_comb = np.zeros((n_docs, n_layers))
    for li, l in enumerate(layers):
        for d in range(n_docs):
            later = set().union(*[set(sel[e][l]) for e in range(d + 1, n_docs)]) if d + 1 < n_docs else set()
            mine = set(sel[d][l])
            surv_comb[d, li] = len(mine - later) / len(mine)

    # numeric: how much of what document d wrote is still in the final cache
    def load_values(path):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        tv = sd["trainable_values"]
        return [torch.as_tensor(t).detach().float() for t in tv]

    caches = [load_values(PHASE1_CACHE)] + [load_values(f) for f in cache_files]
    V_final = caches[-1]

    surv_num = np.zeros((n_docs, n_layers))       # frac of entries with rel drift < 1e-3
    drift_med = np.zeros((n_docs, n_layers))      # median rel drift ||Vf-Vd||/||Vd||
    write_keep = np.zeros((n_docs, n_layers))     # <Df, Dd>/||Dd||^2 (UNBOUNDED)
    write_cos = np.zeros((n_docs, n_layers))
    val_cos = np.zeros((n_docs, n_layers))        # cos(V_final, V_after_d) on d's slots
    subsequent_over_own = np.zeros((n_docs, n_layers))
    n_later_writes = np.zeros((n_docs, n_layers))
    for li, l in enumerate(layers):
        for d in range(n_docs):
            S = torch.tensor(sel[d][l], dtype=torch.long)
            Vb = caches[d][l][0][:, S, :]         # before doc d (cache index d)
            Vd = caches[d + 1][l][0][:, S, :]     # after doc d
            Vf = V_final[l][0][:, S, :]
            nrm_d = Vd.norm(dim=-1)
            drift = (Vf - Vd).norm(dim=-1) / nrm_d.clamp(min=1e-12)
            surv_num[d, li] = float((drift < 1e-3).float().mean())
            drift_med[d, li] = float(drift.median())
            cs = torch.nn.functional.cosine_similarity(Vf, Vd, dim=-1)
            val_cos[d, li] = float(cs.mean())
            Dd = (Vd - Vb).reshape(-1)
            Df = (Vf - Vb).reshape(-1)
            dd = float(Dd @ Dd)
            write_keep[d, li] = float((Df @ Dd) / dd) if dd > 0 else float("nan")
            nn = float(Dd.norm() * Df.norm())
            write_cos[d, li] = float((Df @ Dd) / nn) if nn > 0 else float("nan")
            own = float((Vd - Vb).norm())
            subsequent_over_own[d, li] = (
                float((Vf - Vd).norm() / own) if own > 0 else float("nan")
            )
            mine = set(sel[d][l])
            n_later_writes[d, li] = sum(
                len(mine & set(sel[e][l])) for e in range(d + 1, n_docs)
            ) / max(len(mine), 1)

    q2 = {
        "definition": {
            "survival_combinatorial": "frac of doc d's selected slots that NO later document re-selects",
            "survival_numeric_1e-3": "frac of doc d's written (head,slot) value vectors whose relative change from end-of-doc-d to end-of-run is < 1e-3",
            "median_rel_drift": "median over (head,slot) of ||V_final - V_after_d|| / ||V_after_d||",
            "write_retained_fraction": "<V_final - V_before_d, V_after_d - V_before_d> / ||V_after_d - V_before_d||^2 on doc d's slots (1 = the whole increment survives). UNBOUNDED above -- >1 means later writes moved the same slots further in the same direction.",
            "value_cosine_final_vs_after_d": "mean cos(V_final, V_after_d) over doc d's written (head,slot) value vectors",
            "subsequent_change_over_own_write": "||V_final - V_after_d|| / ||V_after_d - V_before_d|| -- how big the later documents' net change to d's slots is, relative to the change d itself made",
            "mean_later_writes_per_own_slot": "average number of LATER documents that re-select each of doc d's slots",
        },
        "per_doc_mean_over_layers": {
            "survival_combinatorial": surv_comb.mean(1).tolist(),
            "survival_numeric_1e-3": surv_num.mean(1).tolist(),
            "median_rel_drift": drift_med.mean(1).tolist(),
            "write_retained_fraction": write_keep.mean(1).tolist(),
            "write_delta_cosine": write_cos.mean(1).tolist(),
            "value_cosine_final_vs_after_d": val_cos.mean(1).tolist(),
            "subsequent_change_over_own_write": subsequent_over_own.mean(1).tolist(),
            "mean_later_writes_per_own_slot": n_later_writes.mean(1).tolist(),
        },
        "per_layer_mean_over_docs": {
            "survival_combinatorial": surv_comb.mean(0).tolist(),
            "survival_numeric_1e-3": surv_num.mean(0).tolist(),
            "write_retained_fraction": write_keep.mean(0).tolist(),
        },
        "survival_combinatorial_matrix_doc_by_layer": surv_comb.round(5).tolist(),
        "summary": {
            "mean_survival_combinatorial_docs_1_to_15": float(surv_comb[:-1].mean()),
            "mean_survival_combinatorial_all": float(surv_comb.mean()),
            "survival_doc1": float(surv_comb[0].mean()),
            "survival_doc8": float(surv_comb[7].mean()),
            "survival_doc16": float(surv_comb[-1].mean()),
            "mean_write_retained_docs_1_to_15": float(np.nanmean(write_keep[:-1])),
            "write_retained_doc1": float(np.nanmean(write_keep[0])),
            "write_retained_doc16": float(np.nanmean(write_keep[-1])),
            "mean_median_rel_drift_docs_1_to_15": float(drift_med[:-1].mean()),
            "mean_value_cosine_docs_1_to_15": float(np.nanmean(val_cos[:-1])),
            "value_cosine_doc1": float(np.nanmean(val_cos[0])),
            "mean_subsequent_over_own_docs_1_to_15": float(np.nanmean(subsequent_over_own[:-1])),
            "mean_later_writes_per_own_slot_doc1": float(n_later_writes[0].mean()),
        },
    }

    # ---------------- Q4: do documents disagree? ------------------------------------
    sp_layer, pe_layer, top32_overlap_vs_mean = [], [], []
    for li, l in enumerate(layers):
        ranks = [rankdata(tf[d][l]) for d in range(n_docs)]
        sp = [corr(ranks[i], ranks[j]) for i, j in combinations(range(n_docs), 2)]
        pe = [corr(tf[i][l], tf[j][l]) for i, j in combinations(range(n_docs), 2)]
        mean_tf = np.mean([tf[d][l] for d in range(n_docs)], axis=0)
        mean_top = set(np.argsort(-mean_tf)[:top_t].tolist())
        ov = [len(mean_top & set(sel[d][l])) / top_t for d in range(n_docs)]
        sp_layer.append(
            {
                "layer": int(l),
                "spearman_mean": float(np.mean(sp)),
                "spearman_min": float(np.min(sp)),
                "spearman_max": float(np.max(sp)),
                "pearson_mean": float(np.mean(pe)),
                "overlap_with_doc_independent_top32_mean": float(np.mean(ov)),
            }
        )
        pe_layer.append(float(np.mean(pe)))
        top32_overlap_vs_mean.append(float(np.mean(ov)))
    q4 = {
        "definition": {
            "spearman": "rank correlation between two documents' per-layer slot scores (ranking_info.tf, 511 slots), averaged over all 120 document pairs",
            "overlap_with_doc_independent_top32": "|top-32 of the ACROSS-DOCUMENT MEAN score  &  doc d's own top-32| / 32 -- 1.0 means the selection carries no document-specific information",
        },
        "per_layer": sp_layer,
        "summary": {
            "spearman_mean": float(np.mean([r["spearman_mean"] for r in sp_layer])),
            "spearman_min_over_layers": float(min(r["spearman_min"] for r in sp_layer)),
            "pearson_mean": float(np.mean(pe_layer)),
            "overlap_with_doc_independent_top32_mean": float(np.mean(top32_overlap_vs_mean)),
        },
    }

    # tf-idf == tf sanity (USE_IDF=0)
    idf_is_identity = bool(
        all(np.allclose(tf[d], tfidf[d], rtol=0, atol=0) for d in range(n_docs))
    )

    out = {
        "run_dir": RUN_DIR,
        "phase1_cache": PHASE1_CACHE,
        "n_documents": n_docs,
        "top_t_per_doc": top_t,
        "n_layers": n_layers,
        "use_idf_is_identity": idf_is_identity,
        "q1_collision": {"per_layer": q1_per_layer, "summary": q1_summary},
        "q2_survival": q2,
        "q4_document_disagreement": q4,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"q1": q1_summary, "q2": q2["summary"], "q4": q4["summary"]}, indent=2))
    print(f"[done] wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
