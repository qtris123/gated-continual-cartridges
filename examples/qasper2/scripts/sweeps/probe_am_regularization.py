#!/usr/bin/env python
"""Probe TF-IDF absolute-mass vulnerability and regularization ablations.

Usage:
  CARTRIDGES_DIR=... PHASE1_CACHE_PATH=... BG_STATS_PATH=... \\
  python examples/qasper2/scripts/sweeps/probe_am_regularization.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from cartridges.am_reference_data import (
    build_reference_dataloader,
    canonical_document_prompt,
    cleanup_reference_parquet,
    group_conversations_by_document,
    limit_conversations,
    load_conversations,
)
from cartridges.am_stability_probe import (
    DocProbeSummary,
    probe_sparse_solve,
    probe_tfidf_vs_absolute_mass,
    summarize_slot_mass_probes,
    summarize_solve_probes,
)
from cartridges.am_teacher import (
    compute_teacher_targets,
    concat_teacher_kv,
    prefill_document_kv_cache,
)
from cartridges.attention_matching import compute_attention_weights
from cartridges.attention_matching_finetuning import (
    AttentionMatchingFinetuningConfig,
    _collect_reference_queries,
)
from cartridges.cache import TrainableCache
from cartridges.models import FlexQwen3ForCausalLM
from cartridges.sparse_cache_finetuning import BackgroundAccessTracker, CacheTFIDFRanker
from cartridges.train import CacheAndModel


REGIMES = [
    {
        "name": "spectral_baseline",
        "ridge_scale": "spectral",
        "ridge_lambda": 1e-4,
        "ridge_lambda_min": 0.0,
        "delta_weight": 0.0,
    },
    {
        "name": "spectral_floor_1e-4",
        "ridge_scale": "spectral",
        "ridge_lambda": 1e-4,
        "ridge_lambda_min": 1e-4,
        "delta_weight": 0.0,
    },
    {
        "name": "fixed_1e-4",
        "ridge_scale": "fixed",
        "ridge_lambda": 1e-4,
        "ridge_lambda_min": 0.0,
        "delta_weight": 0.0,
    },
    {
        "name": "delta_1e-2",
        "ridge_scale": "spectral",
        "ridge_lambda": 1e-4,
        "ridge_lambda_min": 0.0,
        "delta_weight": 1e-2,
    },
    {
        "name": "spectral_floor_1e-4_delta_1e-2",
        "ridge_scale": "spectral",
        "ridge_lambda": 1e-4,
        "ridge_lambda_min": 1e-4,
        "delta_weight": 1e-2,
    },
]


def _clone_cache(path: str, device) -> TrainableCache:
    return TrainableCache.from_pretrained(path, device=device)


def _layer_tensors(cache: TrainableCache, layer_idx: int, head_idx: int, device):
    n_frozen = getattr(cache, "_num_frozen_tokens", 0)
    trainable_keys = cache.trainable_keys[layer_idx][0, head_idx].detach()
    trainable_values = cache.trainable_values[layer_idx][0, head_idx].detach()
    if n_frozen > 0:
        keys = torch.cat(
            [cache.frozen_keys[layer_idx][0, head_idx].detach(), trainable_keys],
            dim=0,
        )
        values = torch.cat(
            [cache.frozen_values[layer_idx][0, head_idx].detach(), trainable_values],
            dim=0,
        )
    else:
        keys, values = trainable_keys, trainable_values
    return keys.to(device), values.to(device), n_frozen


def main():
    cartridges_dir = Path(os.environ["CARTRIDGES_DIR"])
    phase1 = Path(os.environ["PHASE1_CACHE_PATH"])
    bg_stats = Path(os.environ["BG_STATS_PATH"])
    mt_path = os.environ.get(
        "SYNTH_DATA_PATH",
        str(cartridges_dir / "data/qasper/train/qwen_qasper_MT_task_8192.parquet"),
    )
    out_dir = Path(
        os.environ.get(
            "PROBE_OUT_DIR",
            str(cartridges_dir / "outputs/e2e_qa_to_mt_20260721/am_stability_probe"),
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    max_docs = int(os.environ.get("PROBE_MAX_DOCS", "4"))
    max_ref = int(os.environ.get("MAX_REF_EXAMPLES_PER_DOC", "8"))
    top_t = int(os.environ.get("TOP_T", "64"))
    probe_layers = [int(x) for x in os.environ.get("PROBE_LAYERS", "0,18,35").split(",")]
    max_queries = int(os.environ.get("MAX_QUERIES_PER_HEAD", "64"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model/cache...")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    model = (
        FlexQwen3ForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
        .to(device)
        .to(torch.bfloat16)
    )
    for p in model.parameters():
        p.requires_grad = False

    base_cache = TrainableCache.from_pretrained(str(phase1), device=device)
    attn_config = base_cache.config
    n_layers = attn_config.n_layers
    n_kv = attn_config.n_heads
    head_dim = attn_config.head_dim

    cfg = AttentionMatchingFinetuningConfig(
        enabled=True,
        top_t=top_t,
        granularity="per_layer",
        use_idf=True,
        background_indices_path=str(bg_stats),
        queries_per_batch="all_tokens",
        max_queries_per_head=max_queries,
        ridge_lambda=1e-4,
        key_mode="freeze",
        enable_beta=False,
        target_mode="cartridge_plus_doc",
        max_ref_examples_per_doc=max_ref,
    )
    bg = BackgroundAccessTracker(num_batches=999999999, granularity="per_layer")
    bg.load(str(bg_stats))
    ranker = CacheTFIDFRanker(
        background_tracker=bg,
        use_idf=True,
        smoothing=1.0,
        top_k_per_batch=128,
        granularity="per_layer",
    )

    convs = load_conversations(mt_path)
    groups = list(group_conversations_by_document(convs).items())[:max_docs]
    print(f"Probing {len(groups)} documents, layers={probe_layers}")

    # Independent caches per regularization regime for recurrent multi-doc write.
    caches = {r["name"]: _clone_cache(str(phase1), device) for r in REGIMES}
    wrappeds = {
        name: CacheAndModel(caches[name], model, am_config=cfg).to(device)
        for name in caches
    }

    all_docs_tfidf = []
    all_docs_regimes = {r["name"]: [] for r in REGIMES}

    for doc_idx, (doc_id, doc_convs) in enumerate(groups):
        slug = f"doc-{doc_idx:03d}"
        system_prompt = canonical_document_prompt(doc_convs)
        limited = limit_conversations(doc_convs, max_ref, seed=doc_idx)
        print(f"\n=== {slug} title={doc_id[:60]!r} n={len(limited)} ===")

        # Collect queries / TF-IDF once from spectral_baseline cache state
        # (access ranking depends weakly on values; use current baseline cache).
        baseline_name = "spectral_baseline"
        wrapped = wrappeds[baseline_name]
        cache = caches[baseline_name]

        doc_kv = prefill_document_kv_cache(
            model,
            tok,
            system_prompt,
            attn_config,
            device,
            cartridge_cache=cache,
        )
        loader, tmp = build_reference_dataloader(limited, tok, seed=doc_idx)
        try:
            qacc, _, nbat = _collect_reference_queries(
                wrapped,
                cache,
                loader,
                cfg,
                n_layers,
                n_kv,
                head_dim,
                device,
                max_batches=len(loader),
            )
        finally:
            cleanup_reference_parquet(tmp)

        access = qacc.get_access_scores()  # (n_layers, T_trainable)
        mask, ranking_info = ranker.rank_positions(
            access, top_t=top_t, step=doc_idx + 1, return_info=True
        )
        tf = ranking_info.tf
        tfidf = ranking_info.tfidf

        doc_summary = DocProbeSummary(doc_index=doc_idx, slug=slug, n_queries=0)
        regime_doc_probes = {r["name"]: [] for r in REGIMES}

        for layer_idx in probe_layers:
            selected = mask.positions_per_layer[layer_idx].to(device)
            # Use head 0 as representative; also aggregate a few heads for mass confirm.
            heads = list(range(min(4, n_kv)))
            for head_idx in heads:
                keys, values, n_frozen = _layer_tensors(cache, layer_idx, head_idx, device)
                selected_full = selected + n_frozen
                queries = qacc.get_layer_head_queries(
                    layer_idx, head_idx, n_q_heads=32, n_kv_heads=n_kv
                ).to(device=device, dtype=keys.dtype)
                if queries.numel() == 0:
                    continue
                if queries.shape[0] > max_queries:
                    queries = queries[:max_queries]
                doc_summary.n_queries = max(doc_summary.n_queries, queries.shape[0])

                k_doc = doc_kv[layer_idx][0][head_idx].to(device=device, dtype=keys.dtype)
                v_doc = doc_kv[layer_idx][1][head_idx].to(device=device, dtype=values.dtype)
                k_t, v_t = concat_teacher_kv(keys, values, k_doc, v_doc)
                targets = compute_teacher_targets(
                    queries,
                    k_t,
                    v_t,
                    head_dim,
                    n_cartridge_keys=keys.shape[0],
                    doc_rope_offset=k_doc.shape[0],
                )
                alpha = compute_attention_weights(queries, keys, head_dim)
                alpha_t = compute_attention_weights(
                    queries,
                    k_t,
                    head_dim,
                    doc_key_start=keys.shape[0],
                    doc_rope_offset=k_doc.shape[0],
                )
                mass_doc = float(alpha_t[:, keys.shape[0] :].sum(-1).mean().item())

                slot_probe = probe_tfidf_vs_absolute_mass(
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    selected_indices=selected,  # trainable indices
                    alpha=alpha,
                    abs_access=access[layer_idx].to(device),
                    tfidf=tfidf[layer_idx].to(device),
                    n_frozen=n_frozen,
                )
                doc_summary.slot_mass.append(slot_probe)

                # Regularization A/B on this head, using each regime's own cache state.
                for regime in REGIMES:
                    rname = regime["name"]
                    rcache = caches[rname]
                    rkeys, rvalues, rn_frozen = _layer_tensors(
                        rcache, layer_idx, head_idx, device
                    )
                    assert rn_frozen == n_frozen
                    # Rebuild teacher against this regime's current values.
                    rk_t, rv_t = concat_teacher_kv(rkeys, rvalues, k_doc, v_doc)
                    rtargets = compute_teacher_targets(
                        queries,
                        rk_t,
                        rv_t,
                        head_dim,
                        n_cartridge_keys=rkeys.shape[0],
                        doc_rope_offset=k_doc.shape[0],
                    )
                    new_vals, solve_probe = probe_sparse_solve(
                        layer_idx=layer_idx,
                        head_idx=head_idx,
                        keys=rkeys,
                        values=rvalues,
                        queries=queries,
                        selected_indices=selected_full,
                        targets=rtargets,
                        head_dim=head_dim,
                        ridge_lambda=regime["ridge_lambda"],
                        ridge_scale=regime["ridge_scale"],
                        ridge_lambda_min=regime["ridge_lambda_min"],
                        delta_weight=regime["delta_weight"],
                        teacher_mass_on_doc_mean=mass_doc,
                    )
                    regime_doc_probes[rname].append(solve_probe)
                    # Commit only probed heads' selected rows into that regime cache.
                    with torch.no_grad():
                        rcache.trainable_values[layer_idx][0, head_idx].copy_(
                            new_vals[n_frozen:].to(rcache.trainable_values[layer_idx].dtype)
                        )

        all_docs_tfidf.append(
            {
                "doc": doc_summary.to_dict(),
                "slot_mass_summary": summarize_slot_mass_probes(doc_summary.slot_mass),
            }
        )
        for rname, probes in regime_doc_probes.items():
            all_docs_regimes[rname].append(
                {
                    "doc_index": doc_idx,
                    "slug": slug,
                    "summary": summarize_solve_probes(probes),
                    "probes": [asdict_probe(p) for p in probes],
                }
            )

        print("TF-IDF vs mass:", summarize_slot_mass_probes(doc_summary.slot_mass))
        for rname in all_docs_regimes:
            print(f"  {rname}:", all_docs_regimes[rname][-1]["summary"])

    # Aggregate final cartridge value growth per regime vs Phase1.
    phase1_cpu = TrainableCache.from_pretrained(str(phase1), device="cpu")
    growth_report = {}
    for rname, cache in caches.items():
        growth_report[rname] = {}
        for layer_idx in probe_layers:
            v0 = phase1_cpu.trainable_values[layer_idx].float()
            v1 = cache.trainable_values[layer_idx].detach().float().cpu()
            growth_report[rname][f"layer_{layer_idx}"] = {
                "absmax_phase1": float(v0.abs().max().item()),
                "absmax_final": float(v1.abs().max().item()),
                "absmean_phase1": float(v0.abs().mean().item()),
                "absmean_final": float(v1.abs().mean().item()),
                "growth_max": float(
                    (v1.abs().max() / v0.abs().max().clamp_min(1e-8)).item()
                ),
            }

    report = {
        "config": {
            "phase1_cache": str(phase1),
            "bg_stats": str(bg_stats),
            "mt_path": mt_path,
            "max_docs": max_docs,
            "top_t": top_t,
            "probe_layers": probe_layers,
            "regimes": REGIMES,
        },
        "tfidf_vs_absolute_mass": all_docs_tfidf,
        "regularization_by_doc": all_docs_regimes,
        "final_value_growth": growth_report,
    }

    out_json = out_dir / "probe_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    # Human-readable markdown summary
    md = out_dir / "PROBE_RESULTS.md"
    lines = [
        "# AM Stability Probe Results",
        "",
        f"- Phase1 cache: `{os.environ['PHASE1_CACHE_PATH']}`",
        f"- Docs probed: {max_docs}",
        f"- Layers: {probe_layers}",
        f"- top_t: {top_t}",
        "",
        "## 1) TF-IDF vs absolute attention mass",
        "",
        "| doc | mass_on_S_mean | frac heads mass<0.01 | frac S below median abs-access | TFIDF_S/global | abs_S/global |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in all_docs_tfidf:
        s = row["slot_mass_summary"]
        lines.append(
            f"| {row['doc']['slug']} | {s.get('mass_on_S_mean', float('nan')):.4g} | "
            f"{s.get('frac_heads_mass_lt_0_01', float('nan')):.3f} | "
            f"{s.get('mean_frac_S_below_median_access', float('nan')):.3f} | "
            f"{s.get('mean_tfidf_on_S_over_global', float('nan')):.2f} | "
            f"{s.get('mean_abs_access_on_S_over_global', float('nan')):.2f} |"
        )

    lines += [
        "",
        "## 2) Regularization A/B (per-doc mean over probed heads/layers)",
        "",
    ]
    for rname, docs in all_docs_regimes.items():
        lines.append(f"### `{rname}`")
        lines.append(
            "| doc | mean_mass_S | eff_λ mean | max V growth | max \|V\| after | mean mse |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for d in docs:
            s = d["summary"]
            lines.append(
                f"| {d['slug']} | {s.get('mean_mass_on_S', float('nan')):.4g} | "
                f"{s.get('mean_effective_lambda', float('nan')):.3g} | "
                f"{s.get('max_v_growth_max', float('nan')):.3g} | "
                f"{s.get('max_v_absmax_after', float('nan')):.3g} | "
                f"{s.get('mean_mse_after', float('nan')):.3g} |"
            )
        lines.append("")

    lines += ["## 3) Final value growth vs Phase 1", ""]
    lines.append("| regime | L0 growth | L18 growth | L35 growth | L18 final max | L35 final max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for rname, g in growth_report.items():
        lines.append(
            f"| {rname} | {g['layer_0']['growth_max']:.3g} | {g['layer_18']['growth_max']:.3g} | "
            f"{g['layer_35']['growth_max']:.3g} | {g['layer_18']['absmax_final']:.3g} | "
            f"{g['layer_35']['absmax_final']:.3g} |"
        )

    md.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_json}")
    print(f"Wrote {md}")

    for w in wrappeds.values():
        w.remove_hooks()


def asdict_probe(p):
    from dataclasses import asdict

    return asdict(p)


if __name__ == "__main__":
    main()
