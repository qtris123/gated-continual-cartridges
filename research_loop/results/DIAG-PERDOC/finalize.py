"""DIAG-PERDOC finalizer: per-document acquisition ceiling, uncertainty, bundles.

Reads research_loop/results/DIAG-PERDOC/{writes.tsv,evals.tsv,wrapper*.log} and
/tmp/perdoc/meta.json, writes
  research_loop/state/diagnostics/DIAG-PERDOC.json
  research_loop/results/DIAG-PERDOC/result.json
and logs one summary wandb run (group B-OVERWRITE, tag `diagnostic`), also tagging the
individual eval/write runs `diagnostic`.

No source file is touched; this only reads artefacts.
"""

import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path("/localhome/local-triv/gated-continual-cartridges_explore")
RESDIR = REPO / "research_loop/results/DIAG-PERDOC"
SNAP = Path("/tmp/amsnap_perdoc")
DATA = Path("/tmp/perdoc")
DOCS = ["000", "009", "011", "013", "015"]

meta = json.load(open(DATA / "meta.json"))
TOK = {k: v["n_scored_tokens"] for k, v in meta["scored_token_counts"].items()}
T_FULL = TOK["full_MT"]
docmeta = {f"{d['doc_index']:03d}": d for d in meta["documents"]}

# ------------------------------------------------------------------- tables ---
evals = {}
urls = {}
ckpts = {}
with open(RESDIR / "evals.tsv") as f:
    f.readline()
    for line in f:
        ctag, sub, loss, ckpt, url = line.rstrip("\n").split("\t")
        if loss == "MISSING":
            continue
        evals[(ctag, sub)] = float(loss)
        urls[f"{ctag}/{sub}"] = url
        ckpts[ctag] = ckpt

writes = {}
with open(RESDIR / "writes.tsv") as f:
    f.readline()
    for line in f:
        tag, idx, rope, synth, run_dir, cache, secs = line.rstrip("\n").split("\t")
        writes[tag] = dict(doc_index=int(idx), rope=rope, synth=synth,
                           run_dir=run_dir, cache=cache, secs=int(secs))

wrap = (RESDIR / "wrapper.log").read_text(errors="ignore")
wrap3 = (RESDIR / "wrapper_phase3.log").read_text(errors="ignore") if (RESDIR / "wrapper_phase3.log").exists() else ""


def _field(txt, key):
    for l in txt.splitlines():
        if l.startswith(key + "="):
            return l.split("=", 1)[1]
    return None


# in-run full-MT eval printed by each write (a second, independent read of the same number)
write_mt = {}
write_wandb = {}
write_mass = {}
for tag, w in writes.items():
    log = RESDIR / f"logs/write_{tag}.log"
    if not log.exists():
        continue
    txt = log.read_text(errors="ignore")
    hits = [l for l in txt.splitlines() if "Eval loss - " in l]
    if hits:
        write_mt[tag] = float(hits[-1].split("Eval loss - ")[1].strip())
    u = [w2 for w2 in txt.split() if w2.startswith("https://wandb.ai/") and "/runs/" in w2]
    if u:
        write_wandb[tag] = u[-1].rstrip(".,")
    stats_p = Path(w["run_dir"]) / "per_document_am_stats.pt"
    if stats_p.exists():
        a = torch.load(stats_p, map_location="cpu", weights_only=False)
        rec = a["per_document"][0]
        m = rec.get("extra", {}).get("ref_mass_on_S_per_layer", {})
        write_mass[tag] = {
            "mean_mse": rec.get("mean_mse"),
            "n_queries": rec.get("n_queries"),
            "ref_mass_on_S_mean": (sum(m.values()) / len(m)) if m else None,
            "ref_mass_on_S_min": min(m.values()) if m else None,
            "ref_mass_on_S_max": max(m.values()) if m else None,
            "ref_mass_on_S_per_layer": {str(k): float(v) for k, v in m.items()},
        }

# ----------------------------------------------------- bit-identity of doc 1 ---
OVW_RUN = REPO / "outputs/2026-07-28-21-43-17-continual_am_sparse/4b0edcab-d1c5-4313-bc43-309d7ff77097"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


ROPEB_RUN = REPO / "outputs/2026-07-28-21-23-35-continual_am_sparse/5ff66d3f-84dc-466e-b539-d2ef01a53785"

bit_identity = {}
_BITPAIRS = [
    ("A_doc000_vs_in_sequence_k1_rope1e4", "A_doc000", OVW_RUN / "cache-after-doc-000-ae4e8fb7.pt"),
    ("B_doc000_vs_in_sequence_k1_rope5e6", "B_doc000", ROPEB_RUN / "cache-after-doc-000-ae4e8fb7.pt"),
    ("A_doc015_vs_A_doc015shuf_replicate", "A_doc015", None),
]
for _name, _tag, _ref in _BITPAIRS:
    if _tag not in writes:
        continue
    solo = Path(writes[_tag]["cache"])
    ref = _ref if _ref is not None else Path(writes.get("A_doc015shuf", {}).get("cache", "/nonexistent"))
    if solo.exists() and Path(ref).exists():
        a = torch.load(solo, map_location="cpu", weights_only=False)
        b = torch.load(ref, map_location="cpu", weights_only=False)

        def tensors(o, pre=""):
            # checkpoints store torch.nn.ParameterList objects -> iterate them too
            out = {}
            if torch.is_tensor(o):
                out[pre[:-1]] = o
            elif isinstance(o, dict):
                for k, v in o.items():
                    out.update(tensors(v, f"{pre}{k}."))
            elif isinstance(o, (list, tuple)) or hasattr(o, "__iter__"):
                try:
                    for i, v in enumerate(o):
                        out.update(tensors(v, f"{pre}{i}."))
                except TypeError:
                    pass
            return out

        ta, tb = tensors(a), tensors(b)
        same = [k for k in ta if k in tb and torch.equal(ta[k], tb[k])]
        diff = [k for k in ta if k in tb and not torch.equal(ta[k], tb[k])]
        rel = None
        if diff:
            k0 = diff[0]
            rel = float((ta[k0].float() - tb[k0].float()).norm() / tb[k0].float().norm())
        bit_identity[_name] = {
            "cache_a": str(solo),
            "cache_b": str(ref),
            "n_tensors_compared": len(ta),
            "n_bitwise_identical": len(same),
            "n_different": len(diff),
            "all_identical": len(diff) == 0 and len(ta) > 0,
            "different_keys_sample": diff[:8],
            "rel_fro_first_differing_tensor": rel,
        }

# ------------------------------------------------------ per-document results ---
MT_PHASE1_KNOWN = 3.7825491428375244   # DIAG-SEQUENCE, same harness
MT_K16_KNOWN = 2.5524158477783203
MT_K12_KNOWN = 2.435236930847168


def others_loss(l_full, l_own, t_own):
    """CE on the OTHER documents' eval questions, from the token-weighted identity."""
    if l_full is None or l_own is None:
        return None
    return (l_full * T_FULL - l_own * t_own) / (T_FULL - t_own)


per_doc = {}
for d in DOCS:
    t_own = TOK[f"doc{d}"]
    ph1 = evals.get(("PH1", f"doc{d}"))
    k16a = evals.get(("K16A", f"doc{d}"))
    k12a = evals.get(("K12A", f"doc{d}"))
    k16b = evals.get(("K16B", f"doc{d}"))
    k12b = evals.get(("K12B", f"doc{d}"))
    soloA = evals.get((f"A_doc{d}", f"doc{d}"))
    soloA_full = evals.get((f"A_doc{d}", "MTfull"))
    soloB = evals.get((f"B_doc{d}", f"doc{d}"))
    soloB_full = evals.get((f"B_doc{d}", "MTfull"))
    ph1_full = evals.get(("PH1", "MTfull"), MT_PHASE1_KNOWN)
    rec = {
        "doc_index": int(d),
        "k_in_write_order": int(d) + 1,
        "slug": docmeta[d]["slug"],
        "paper_id": docmeta[d]["paper_id"],
        "title": docmeta[d]["title"],
        "n_eval_questions": docmeta[d]["n_eval_questions"],
        "n_scored_tokens": t_own,
        "own_subset_loss": {
            "phase1_floor": ph1,
            "solo_rope1e4": soloA,
            "k12_rope1e4": k12a,
            "k16_rope1e4": k16a,
            "solo_rope5e6": soloB,
            "k12_rope5e6": k12b,
            "k16_rope5e6": k16b,
        },
        "full_MT_loss": {
            "phase1_floor": ph1_full,
            "solo_rope1e4": soloA_full,
            "solo_rope5e6": soloB_full,
            "k16_rope1e4": evals.get(("K16A", "MTfull"), MT_K16_KNOWN),
        },
    }
    if None not in (ph1, soloA, k16a):
        rec["deltas_rope1e4"] = {
            "solo_minus_phase1": soloA - ph1,
            "k16_minus_phase1": k16a - ph1,
            "k12_minus_phase1": (k12a - ph1) if k12a is not None else None,
            "solo_minus_k16": soloA - k16a,
            "solo_minus_k12": (soloA - k12a) if k12a is not None else None,
        }
    if None not in (ph1, soloB, k16b):
        rec["deltas_rope5e6"] = {
            "solo_minus_phase1": soloB - ph1,
            "k16_minus_phase1": k16b - ph1,
            "k12_minus_phase1": (k12b - ph1) if k12b is not None else None,
            "solo_minus_k16": soloB - k16b,
        }
    # content-free component: what the solo write does to the OTHER 15 documents
    lo_solo = others_loss(soloA_full, soloA, t_own)
    lo_ph1 = others_loss(ph1_full, ph1, t_own)
    if lo_solo is not None and lo_ph1 is not None:
        rec["off_target_rope1e4"] = {
            "others_loss_solo": lo_solo,
            "others_loss_phase1": lo_ph1,
            "others_delta": lo_solo - lo_ph1,
            "own_delta": soloA - ph1,
            "share_of_full_MT_gain_that_is_off_target": (
                ((lo_ph1 - lo_solo) * (T_FULL - t_own))
                / max(1e-12, (ph1_full - soloA_full) * T_FULL)
            ),
        }
    per_doc[f"doc{d}"] = rec

# ------------------------------------------------- aggregate + uncertainty -----
def agg(key, arm="deltas_rope1e4"):
    vals = [per_doc[f"doc{d}"].get(arm, {}).get(key) for d in DOCS]
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return {"n": len(vals), "values": vals}
    return {
        "n": len(vals),
        "values": vals,
        "mean": statistics.fmean(vals),
        "sd": statistics.stdev(vals),
        "se": statistics.stdev(vals) / math.sqrt(len(vals)),
        "min": min(vals),
        "max": max(vals),
    }


aggregate = {
    "rope1e4": {k: agg(k) for k in
                ["solo_minus_phase1", "k16_minus_phase1", "k12_minus_phase1",
                 "solo_minus_k16", "solo_minus_k12"]},
    "rope5e6": {k: agg(k, "deltas_rope5e6") for k in
                ["solo_minus_phase1", "k16_minus_phase1", "k12_minus_phase1", "solo_minus_k16"]},
}

# token-weighted aggregate over the five documents (the honest pooled number:
# it is exactly what a 5-document, 27-question eval split would report)
def pooled(tag_fn):
    num = den = 0.0
    for d in DOCS:
        v = tag_fn(d)
        if v is None:
            return None
        num += v * TOK[f"doc{d}"]
        den += TOK[f"doc{d}"]
    return num / den


pooled_losses = {
    "phase1_floor": pooled(lambda d: evals.get(("PH1", f"doc{d}"))),
    "solo_rope1e4": pooled(lambda d: evals.get((f"A_doc{d}", f"doc{d}"))),
    "k12_rope1e4": pooled(lambda d: evals.get(("K12A", f"doc{d}"))),
    "k16_rope1e4": pooled(lambda d: evals.get(("K16A", f"doc{d}"))),
    "solo_rope5e6": pooled(lambda d: evals.get((f"B_doc{d}", f"doc{d}"))),
    "k12_rope5e6": pooled(lambda d: evals.get(("K12B", f"doc{d}"))),
    "k16_rope5e6": pooled(lambda d: evals.get(("K16B", f"doc{d}"))),
}

# decomposition check: union5 must equal the token-weighted mean of the five subsets
union_pred = pooled_losses["phase1_floor"]
union_obs = evals.get(("PH1", "union5"))
decomposition_check = {
    "union5_measured_phase1": union_obs,
    "union5_predicted_from_5_subsets": union_pred,
    "abs_error": (abs(union_obs - union_pred) if (union_obs and union_pred) else None),
    "note": ("the harness loss is a token-weighted mean over scored tokens and per-example "
             "CE is packing-invariant (block-diagonal seq_ids, packing never splits an "
             "element), so subsets decompose exactly"),
}

# replicate control: doc-015 written alone twice with independent reference draws
replicate = {}
if ("A_doc015shuf", "doc015") in evals and ("A_doc015", "doc015") in evals:
    replicate = {
        "doc015_own_subset_draw1": evals[("A_doc015", "doc015")],
        "doc015_own_subset_draw2_shuffled": evals[("A_doc015shuf", "doc015")],
        "abs_diff_own_subset": abs(evals[("A_doc015", "doc015")] - evals[("A_doc015shuf", "doc015")]),
        "doc015_full_MT_draw1": evals.get(("A_doc015", "MTfull")),
        "doc015_full_MT_draw2_shuffled": evals.get(("A_doc015shuf", "MTfull")),
        "reference_draw_overlap": meta["reference_draw_overlap_solo_vs_in_sequence"].get("15_shuf_vs_solo"),
        "what_it_measures": ("two nominally identical single-document writes that differ only "
                             "in WHICH 32 of the document's ~532 synthesis conversations were "
                             "drawn as reference queries -- a same-condition variability floor"),
    }
    if replicate["doc015_full_MT_draw1"] and replicate["doc015_full_MT_draw2_shuffled"]:
        replicate["abs_diff_full_MT"] = abs(
            replicate["doc015_full_MT_draw1"] - replicate["doc015_full_MT_draw2_shuffled"])

# leave-one-out -> exact per-example CE on doc-015's 7 questions
loo = {}
try:
    sys.path.insert(0, str(SNAP))
    from transformers import AutoTokenizer

    from cartridges.datasets import DataSource, LossEvalDataset

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

    def ntok(p):
        ds = LossEvalDataset(
            LossEvalDataset.Config(data_source=DataSource(path=str(p), type="local"),
                                   packed_seq_length=2048),
            tokenizer=tok, seed=0)
        return sum(int(len(e.topk_token_idxs)) for e in ds.elements)

    per_example = {}
    for ctag in ["A_doc015", "K16A"]:
        L_S = evals.get((ctag, "doc015"))
        if L_S is None:
            continue
        T_S = TOK["doc015"]
        vals = []
        for i in range(7):
            L_loo = evals.get((ctag, f"loo_e{i}"))
            if L_loo is None:
                vals.append(None)
                continue
            T_loo = ntok(DATA / f"eval/loo_doc015_e{i}.parquet")
            t_e = T_S - T_loo
            c_e = L_S * T_S - L_loo * T_loo
            vals.append(c_e / t_e if t_e > 0 else None)
        per_example[ctag] = vals
    loo["per_example_loss"] = per_example
    if "A_doc015" in per_example and "K16A" in per_example:
        a, b = per_example["A_doc015"], per_example["K16A"]
        pairs = [(x - y) for x, y in zip(a, b) if x is not None and y is not None]
        if len(pairs) >= 2:
            loo["paired_delta_solo_minus_k16"] = {
                "per_example": pairs,
                "mean": statistics.fmean(pairs),
                "sd": statistics.stdev(pairs),
                "se_of_mean": statistics.stdev(pairs) / math.sqrt(len(pairs)),
                "n": len(pairs),
                "n_examples_favouring_solo": sum(1 for p in pairs if p < 0),
            }
        for tag, v in per_example.items():
            vv = [x for x in v if x is not None]
            if len(vv) >= 2:
                loo.setdefault("per_example_spread", {})[tag] = {
                    "mean": statistics.fmean(vv), "sd": statistics.stdev(vv),
                    "se_of_subset_mean": statistics.stdev(vv) / math.sqrt(len(vv)),
                    "min": min(vv), "max": max(vv),
                }
except Exception as e:  # noqa: BLE001
    loo["error"] = repr(e)

# ------------------------------------------------------------- provenance -----
head_now = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
manifest_now = subprocess.run(
    "find cartridges examples -type f -name '*.py' | LC_ALL=C sort | xargs sha256sum "
    "| awk '{print $1\"  \"$2}' | LC_ALL=C sort | sha256sum | cut -d' ' -f1",
    shell=True, cwd=SNAP, capture_output=True, text=True).stdout.strip()

provenance = {
    "snapshot_path": str(SNAP),
    "snapshot_contents": f"git archive HEAD {{cartridges,examples}} at HEAD={_field(wrap, 'REPO_HEAD_AT_LAUNCH')}",
    "snapshot_manifest_sha256_at_launch": _field(wrap, "SNAPSHOT_MANIFEST"),
    "snapshot_manifest_sha256_after_phase12": _field(wrap, "SNAPSHOT_MANIFEST_AFTER"),
    "snapshot_manifest_sha256_after_phase3": _field(wrap3, "SNAPSHOT_MANIFEST_AFTER"),
    "snapshot_manifest_sha256_now": manifest_now,
    "snapshot_manifest_recipe": ("cd $SNAP && find cartridges examples -type f -name '*.py' | "
                                 "LC_ALL=C sort | xargs sha256sum | awk '{print $1\"  \"$2}' | "
                                 "LC_ALL=C sort | sha256sum"),
    "verified_import_path": _field(wrap, "CARTRIDGES_IMPORT_PATH") or
                            "/tmp/amsnap_perdoc/cartridges (printed from cwd=/tmp in wrapper.log)",
    "CARTRIDGES_DIR": str(SNAP),
    "repo_HEAD_at_launch": _field(wrap, "REPO_HEAD_AT_LAUNCH"),
    "repo_HEAD_after": head_now,
    "gpu_claimed_via_flock": _field(wrap, "PERDOC_CLAIMED_GPU"),
    "instrumentation": "NONE - no source file edited; MECH-KEYS was editing cartridges/am/key_select.py concurrently",
    "key_select_differed_from_HEAD_during_run": True,
    "single_document_route": (
        "the MT synthesis parquet was filtered to the rows of ONE paper title and passed as "
        "SYNTH_DATA_PATH; the stock per-document driver then sees exactly one document. "
        "canonical_document_prompt() output is byte-identical to the in-sequence write for all "
        "five documents (sha256 recorded in /tmp/perdoc/meta.json)."
    ),
    "known_difference_vs_in_sequence_write": (
        "cartridges/am/continual.py seeds limit_conversations() and build_reference_dataloader() "
        "with doc_idx, which is 0 for a solo run. For doc_index=0 that is identical (32/32 same "
        "reference conversations -> the solo cache is byte-identical to cache-after-doc-000 and "
        "its full-MT loss reproduces DIAG-SEQUENCE k=1 to 16 digits). For the other documents the "
        "solo run draws a DIFFERENT 32 of the ~500 conversations (overlap 1-4 of 32); the "
        "A_doc015shuf replicate bounds that effect."
    ),
    "baselines": {
        "phase1": str(REPO / "outputs/phase1_selfdistill_qwen512/cache_last.pt"),
        "k12_k16_rope1e4": str(OVW_RUN) + " (DIAG-OVERWRITE canonical 16-document run, AM_ROPE_THETA unset = 1e4)",
        "k12_k16_rope5e6": "outputs/2026-07-28-21-23-35-continual_am_sparse/5ff66d3f-84dc-466e-b539-d2ef01a53785 (DIAG-ROPE arm B)",
    },
    "eval_convention": ("standalone examples/qasper2/train/eval_forgetting.py from the snapshot, "
                        "BATCH_SIZE=4 (default), MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507, one fresh "
                        "process per (checkpoint, subset); 'Eval loss' mean-CE parsed from the log, "
                        "then the PID killed (RUNBOOK §1)"),
    "doc_mapping_source": ("research_loop/state/diagnostics/DIAG-SEQUENCE.json document_write_order "
                           "(title <-> write order); paper_id recovered by exact title match against "
                           "qasper_eval_MT.parquet metadata and cross-checked against DIAG-SEQUENCE's "
                           "per-document question counts (16/16 exact) and the sha1 slugs (16/16 exact)"),
}

# every eval log prints the checkpoint it loaded -> verify
loaded_ok = {}
for (ctag, sub) in evals:
    log = RESDIR / f"logs/eval_{ctag}__{sub}.log"
    if log.exists():
        loaded_ok[f"{ctag}/{sub}"] = ckpts[ctag] in log.read_text(errors="ignore")
provenance["each_eval_loaded_intended_checkpoint"] = loaded_ok
provenance["all_evals_loaded_intended_checkpoint"] = all(loaded_ok.values()) if loaded_ok else False

gates = {
    "solo_doc1_full_MT_equals_DIAG_SEQUENCE_k1": {
        "measured": write_mt.get("A_doc000"),
        "expected": 3.7445480823516846,
        "matches_16_digits": (write_mt.get("A_doc000") == 3.7445480823516846),
    },
    "phase1_full_MT": {"measured": evals.get(("PH1", "MTfull")), "expected": MT_PHASE1_KNOWN},
    "k16_full_MT": {"measured": evals.get(("K16A", "MTfull")), "expected": MT_K16_KNOWN},
    "k12_full_MT": {"measured": evals.get(("K12A", "MTfull")), "expected": MT_K12_KNOWN},
    "solo_doc1_cache_bitwise_identical_to_in_sequence":
        bit_identity.get("A_doc000_vs_in_sequence_k1_rope1e4", {}).get("all_identical"),
    "solo_doc1_rope5e6_cache_bitwise_identical_to_DIAG_ROPE_armB_k1":
        bit_identity.get("B_doc000_vs_in_sequence_k1_rope5e6", {}).get("all_identical"),
}

out = {
    "id": "DIAG-PERDOC",
    "board_entry": "B-OVERWRITE",
    "question": ("Can 32 slots hold ONE document? Measure the per-document acquisition ceiling: "
                 "write a single document alone from the Phase-1 cache and score it on that "
                 "document's own MT eval questions, against the Phase-1 floor, the k=12 cache and "
                 "the 16-document cache on the same questions."),
    "provenance": provenance,
    "correctness_gates": gates,
    "bit_identity_doc1": bit_identity,
    "per_document": per_doc,
    "pooled_over_5_documents_token_weighted": pooled_losses,
    "aggregate_paired_deltas": aggregate,
    "decomposition_check": decomposition_check,
    "replicate_control": replicate,
    "leave_one_out_uncertainty": loo,
    "write_side": {
        "in_run_full_MT_eval": write_mt,
        "mass_on_S_and_solve": write_mass,
        "wall_clock_s": {k: v["secs"] for k, v in writes.items()},
        "run_dirs": {k: v["run_dir"] for k, v in writes.items()},
        "caches": {k: v["cache"] for k, v in writes.items()},
    },
    "wandb_urls": {"evals": urls, "writes": write_wandb},
    "eval_subsets": {f"doc{d}": {"n_questions": docmeta[d]["n_eval_questions"],
                                 "n_scored_tokens": TOK[f"doc{d}"],
                                 "paper_id": docmeta[d]["paper_id"]} for d in DOCS},
    "reference_points": {
        "phase1_full_MT": MT_PHASE1_KNOWN,
        "k12_full_MT": MT_K12_KNOWN,
        "k16_full_MT": MT_K16_KNOWN,
        "oracle_write_perfect_values_full_MT": 2.381,
        "dense_4ep_bar_full_MT": 1.8725,
        "noise_band_full_set": 0.15,
    },
}

diagpath = REPO / "research_loop/state/diagnostics/DIAG-PERDOC.json"
diagpath.parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(diagpath, "w"), indent=1)
print("wrote", diagpath)

print("\n=== own-subset loss (rope 1e4) ===")
print(f"{'doc':<8}{'nq':>4}{'ntok':>6}{'phase1':>10}{'solo':>10}{'k12':>10}{'k16':>10}"
      f"{'solo-k16':>10}{'solo-ph1':>10}")
for d in DOCS:
    r = per_doc[f"doc{d}"]["own_subset_loss"]
    dl = per_doc[f"doc{d}"].get("deltas_rope1e4", {})
    def f(x):
        return f"{x:10.4f}" if isinstance(x, float) else f"{'--':>10}"
    print(f"doc{d:<5}{per_doc[f'doc{d}']['n_eval_questions']:>4}{TOK[f'doc{d}']:>6}"
          f"{f(r['phase1_floor'])}{f(r['solo_rope1e4'])}{f(r['k12_rope1e4'])}{f(r['k16_rope1e4'])}"
          f"{f(dl.get('solo_minus_k16'))}{f(dl.get('solo_minus_phase1'))}")
print("pooled:", json.dumps(pooled_losses, indent=1))
print("aggregate:", json.dumps(aggregate, indent=1))
print("gates:", json.dumps(gates, indent=1, default=str))
print("replicate:", json.dumps(replicate, indent=1))
print("loo:", json.dumps(loo, indent=1))
print("off_target:", json.dumps({d: per_doc[f"doc{d}"].get("off_target_rope1e4") for d in DOCS}, indent=1))

# ------------------------------------------------------------------ wandb ----
if os.environ.get("SKIP_WANDB") == "1":
    raise SystemExit(0)

import wandb  # noqa: E402

api = wandb.Api()
ENTITY, PROJECT = "vqtri-purdue-university", "SEACrowd"
tagged, failed = 0, 0
for u in list(urls.values()) + list(write_wandb.values()):
    if not u or "/runs/" not in u:
        continue
    rid = u.rstrip("/").split("/runs/")[-1]
    try:
        r = api.run(f"{ENTITY}/{PROJECT}/{rid}")
        if "diagnostic" not in r.tags:
            r.tags = list(r.tags) + ["diagnostic"]
            r.update()
        tagged += 1
    except Exception:  # noqa: BLE001
        failed += 1
print(f"tagged {tagged} runs `diagnostic` ({failed} failed)")

summary = wandb.init(
    project=PROJECT, entity=ENTITY, name="DIAG-PERDOC_summary",
    group="B-OVERWRITE", tags=["diagnostic", "DIAG-PERDOC", "B-OVERWRITE"],
    notes="per-document acquisition ceiling: one document written alone into 32 slots vs the same document written as one of 16",
    config={"top_t": 32, "granularity": "per_layer", "slot_selection": "tfidf", "use_idf": 0,
            "key_mode": "freeze", "enable_beta": 0, "ridge_lambda": 1e-4, "ridge_scale": "spectral",
            "delta_weight": 1e-2, "max_queries_per_head": 64, "n_documents_written": 1,
            "documents": DOCS, "snapshot_manifest": provenance["snapshot_manifest_sha256_at_launch"]},
)
tbl = wandb.Table(columns=["doc", "n_questions", "n_tokens", "phase1", "solo_1e4", "k12_1e4",
                           "k16_1e4", "solo_minus_k16", "solo_minus_phase1", "solo_5e6", "k16_5e6"])
for d in DOCS:
    r = per_doc[f"doc{d}"]["own_subset_loss"]
    dl = per_doc[f"doc{d}"].get("deltas_rope1e4", {})
    tbl.add_data(f"doc{d}", per_doc[f"doc{d}"]["n_eval_questions"], TOK[f"doc{d}"],
                 r["phase1_floor"], r["solo_rope1e4"], r["k12_rope1e4"], r["k16_rope1e4"],
                 dl.get("solo_minus_k16"), dl.get("solo_minus_phase1"),
                 r["solo_rope5e6"], r["k16_rope5e6"])
    summary.log({"diag/doc_index": int(d),
                 "diag/own_phase1": r["phase1_floor"], "diag/own_solo": r["solo_rope1e4"],
                 "diag/own_k12": r["k12_rope1e4"], "diag/own_k16": r["k16_rope1e4"]},
                step=int(d))
summary.log({"diag/per_document": tbl})
flat = {}
for arm, kv in aggregate.items():
    for k, v in kv.items():
        if isinstance(v, dict) and "mean" in v:
            flat[f"diag/{arm}/{k}_mean"] = v["mean"]
            flat[f"diag/{arm}/{k}_sd"] = v["sd"]
            flat[f"diag/{arm}/{k}_se"] = v["se"]
for k, v in pooled_losses.items():
    if v is not None:
        flat[f"diag/pooled/{k}"] = v
for tag, m in write_mass.items():
    if m.get("ref_mass_on_S_mean") is not None:
        flat[f"diag/mass_on_S/{tag}"] = m["ref_mass_on_S_mean"]
        flat[f"diag/mean_mse/{tag}"] = m["mean_mse"]
if loo.get("paired_delta_solo_minus_k16"):
    flat["diag/loo_paired_delta_mean"] = loo["paired_delta_solo_minus_k16"]["mean"]
    flat["diag/loo_paired_delta_se"] = loo["paired_delta_solo_minus_k16"]["se_of_mean"]
if replicate:
    flat["diag/replicate_abs_diff_own_subset"] = replicate["abs_diff_own_subset"]
    if "abs_diff_full_MT" in replicate:
        flat["diag/replicate_abs_diff_full_MT"] = replicate["abs_diff_full_MT"]
flat["diag/gate_solo_doc1_matches_k1_16_digits"] = bool(
    gates["solo_doc1_full_MT_equals_DIAG_SEQUENCE_k1"]["matches_16_digits"])
flat["diag/gate_solo_doc1_cache_bitwise_identical"] = bool(
    bit_identity.get("A_doc000_vs_in_sequence_k1_rope1e4", {}).get("all_identical"))
summary.summary.update(flat)
url, rid = summary.url, summary.id
summary.finish()
print("SUMMARY_WANDB_URL=", url)
print("SUMMARY_WANDB_ID=", rid)
json.dump({"summary_wandb_url": url, "summary_wandb_id": rid},
          open(RESDIR / "summary_wandb.json", "w"), indent=1)
out["summary_wandb_url"] = url
out["summary_wandb_id"] = rid
json.dump(out, open(diagpath, "w"), indent=1)
