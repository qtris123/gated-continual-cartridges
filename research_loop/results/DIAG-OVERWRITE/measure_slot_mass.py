"""DIAG-OVERWRITE (B-OVERWRITE) Q3: per-slot cartridge attention mass, and the
bandwidth of the tf-idf written union vs a mass-ranked per-layer top-32.

Standalone (nothing under `cartridges/` is modified), forward passes only, on the
UNTOUCHED Phase-1 cartridge.  Independent reimplementation of the SCOUT-KEYS number
(15.2% vs 70.4%) -- it does its own forward pass rather than re-reading
DIAG-ROUTING.npz, and it additionally separates the frozen attention-sink slot 0,
which no AM config can write.

Convention (identical to ORACLE-WRITE / DIAG-ROUTING):
    w         = softmax over [ 512 cartridge slots (always visible) || causal prefix ]
    mass_cart = w[..., :T_c].sum(-1)
    a         = w[..., :T_c] / mass_cart          (cartridge-normalised routing)

Accumulated per (layer, kv_head, slot):
    sum_w[l,h,s] = sum over query rows of w[..., s]          (unnormalised)
    sum_a[l,h,s] = sum over query rows of a[..., s]          (normalised; sums to cnt)

Reported per layer, for several slot sets S:
    share_mean_norm(S)  = mean_q sum_{s in S} a_s              (DIAG-ROUTING convention)
    share_ratio(S)      = sum_w(S) / sum_w(all cartridge)      (ORACLE-WRITE convention)
    abs_mass(S)         = mean_q sum_{s in S} w_s              (absolute attention mass)

Slot sets: the tf-idf written union (from am_doc_*.pt), a mass-ranked per-layer top-32
(with and without the frozen sink), a mass-ranked per-HEAD top-32, and a budget-matched
mass-ranked set of exactly |union| slots.

Usage (env): PHASE1_CACHE, EVAL_SPECS "QA:path,MT:path", SLOTS_RUN_DIR, OUT_JSON, OUT_NPZ
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from glob import glob

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import cartridges
from cartridges.cache import TrainableCache
from cartridges.datasets import DataSource, LossEvalDataset
from cartridges.models import FlexQwen3ForCausalLM, HFModelConfig
from cartridges.sparse_cache_finetuning import _apply_rotary_pos_emb

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
PHASE1_CACHE = os.environ["PHASE1_CACHE"]
EVAL_SPECS = os.environ["EVAL_SPECS"]
SLOTS_RUN_DIR = os.environ["SLOTS_RUN_DIR"]
OUT_JSON = os.environ["OUT_JSON"]
OUT_NPZ = os.environ["OUT_NPZ"]
DEVICE = "cuda"

print(f"[provenance] cartridges.__file__ = {os.path.dirname(cartridges.__file__)}", flush=True)


def install_qk_capture(model):
    captured: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    handles = []
    for layer_idx, decoder_layer in enumerate(model.model.layers):
        attn_module = decoder_layer.self_attn

        def make_hook(idx, attn_mod):
            def hook_fn(module, args, output):
                batch = args[0] if args else None
                if batch is None:
                    return
                hidden_states = batch.hidden_states
                hidden_shape = (*hidden_states.shape[:-1], -1, attn_mod.head_dim)
                with torch.no_grad():
                    if hasattr(attn_mod, "q_norm"):
                        q = attn_mod.q_norm(
                            attn_mod.q_proj(hidden_states).view(hidden_shape)
                        ).transpose(1, 2)
                        k = attn_mod.k_norm(
                            attn_mod.k_proj(hidden_states).view(hidden_shape)
                        ).transpose(1, 2)
                    else:
                        q = attn_mod.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                        k = attn_mod.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    cos, sin = batch.position_embeddings
                    q = _apply_rotary_pos_emb(q, cos, sin)
                    k = _apply_rotary_pos_emb(k, cos, sin)
                captured[idx] = (q.detach(), k.detach())

            return hook_fn

        handles.append(attn_module.register_forward_hook(make_hook(layer_idx, attn_module)))
    return captured, handles


def per_doc_slots(run_dir: str) -> tuple[dict[int, list[list[int]]], list[str]]:
    """Per-document per-layer selected positions (TRAINABLE index space)."""
    files = sorted(glob(os.path.join(run_dir, "am_doc_*.pt")))
    if not files:
        raise FileNotFoundError(f"no am_doc_*.pt under {run_dir}")
    per_layer: dict[int, list[list[int]]] = defaultdict(list)
    for f in files:
        payload = torch.load(f, map_location="cpu", weights_only=False)
        mask = payload["am_stats"].mask
        assert mask.granularity == "per_layer", mask.granularity
        for layer_idx, pos in mask.positions_per_layer.items():
            per_layer[int(layer_idx)].append(sorted(int(x) for x in pos.flatten().tolist()))
    return per_layer, files


@torch.no_grad()
def accumulate(model, cache, dataloader, n_layers: int):
    captured, handles = install_qk_capture(model)
    n_frozen = cache._num_frozen_tokens
    T_c = cache.num_cartridge_tokens()
    n_kv = cache.trainable_keys[0].shape[1]
    sum_w = torch.zeros(n_layers, n_kv, T_c, dtype=torch.float64, device=DEVICE)
    sum_a = torch.zeros(n_layers, n_kv, T_c, dtype=torch.float64, device=DEVICE)
    cnt = torch.zeros(n_layers, n_kv, dtype=torch.float64, device=DEVICE)
    try:
        for batch in dataloader:
            captured.clear()
            input_ids = batch.input_ids.to(DEVICE)
            seq_ids = batch.element_ids.to(DEVICE)
            position_ids = batch.position_ids.to(DEVICE)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                model(
                    input_ids=input_ids,
                    seq_ids=seq_ids,
                    position_ids=position_ids,
                    use_cache=True,
                    past_key_values=cache,
                )
            L = input_ids.shape[-1]
            ar = torch.arange(L, device=DEVICE)
            seq_vis = (seq_ids[:, None] == seq_ids[None, :]) & (ar[:, None] >= ar[None, :])
            for layer_idx in range(n_layers):
                q, k_seq = captured[layer_idx]
                k_cart = (
                    torch.cat(
                        [cache.frozen_keys[layer_idx], cache.trainable_keys[layer_idx]], dim=2
                    )[0]
                    if n_frozen > 0
                    else cache.trainable_keys[layer_idx][0]
                )
                n_q = q.shape[1]
                g = n_q // n_kv
                head_dim = k_cart.shape[-1]
                scaling = head_dim ** -0.5
                for h in range(n_kv):
                    qh = q[0, h * g : (h + 1) * g].float()
                    Kh = torch.cat([k_cart[h], k_seq[0, h]], dim=0).float()
                    s = (qh @ Kh.T) * scaling
                    s[..., T_c:] = s[..., T_c:].masked_fill(~seq_vis, float("-inf"))
                    w = torch.softmax(s, dim=-1)
                    w_c = w[..., :T_c]                       # (g, L, T_c)
                    m_c = w_c.sum(-1, keepdim=True).clamp(min=1e-30)
                    a = w_c / m_c
                    sum_w[layer_idx, h] += w_c.reshape(-1, T_c).sum(0).double()
                    sum_a[layer_idx, h] += a.reshape(-1, T_c).sum(0).double()
                    cnt[layer_idx, h] += w_c.shape[0] * w_c.shape[1]
                    del s, w, w_c, a
            cache.clear()
    finally:
        for hd in handles:
            hd.remove()
    return sum_w.cpu().numpy(), sum_a.cpu().numpy(), cnt.cpu().numpy()


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = (
        HFModelConfig(pretrained_model_name_or_path=MODEL_NAME, model_cls=FlexQwen3ForCausalLM)
        .instantiate()
        .to(DEVICE)
        .to(torch.bfloat16)
    )
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    evals = {}
    for spec in EVAL_SPECS.split(","):
        label, path = spec.split(":", 1)
        ds = LossEvalDataset.Config(
            data_source=DataSource(path=path, type="local"),
            packed_seq_length=2048,
        ).instantiate(tokenizer=tokenizer, seed=0)
        evals[label] = DataLoader(ds, batch_size=1, collate_fn=lambda b: b[0], num_workers=0)
        print(f"[eval] {label}: n={len(ds)} batches", flush=True)

    cache = TrainableCache.from_pretrained(PHASE1_CACHE, device=DEVICE).to(DEVICE)
    n_layers = cache.config.n_layers
    n_frozen = cache._num_frozen_tokens
    T_c = cache.num_cartridge_tokens()

    doc_slots, files = per_doc_slots(SLOTS_RUN_DIR)
    n_docs = len(files)
    # union per layer, in FULL cartridge index space (trainable idx + n_frozen)
    union_full = {
        l: sorted({p + n_frozen for doc in doc_slots[l] for p in doc}) for l in doc_slots
    }
    print(
        f"[slots] {n_docs} docs; union/layer min={min(len(v) for v in union_full.values())} "
        f"max={max(len(v) for v in union_full.values())} "
        f"mean={np.mean([len(v) for v in union_full.values()]):.1f}",
        flush=True,
    )

    out = {
        "provenance": {
            "cartridges_import_path": os.path.dirname(cartridges.__file__),
            "phase1_cache": PHASE1_CACHE,
            "slots_run_dir": SLOTS_RUN_DIR,
            "n_docs": n_docs,
            "n_frozen": int(n_frozen),
            "T_c": int(T_c),
        },
        "splits": {},
    }
    npz: dict[str, np.ndarray] = {}

    for label, dl in evals.items():
        print(f"[measure] split={label}", flush=True)
        sw, sa, cnt = accumulate(model, cache, dl, n_layers)
        a_bar = sa / cnt[:, :, None]              # (L, H, T_c), each row sums to 1
        w_bar = sw / cnt[:, :, None]              # (L, H, T_c), sums to mass_on_cart
        a_layer = a_bar.mean(1)                   # (L, T_c)
        w_layer = w_bar.mean(1)                   # (L, T_c)
        npz[f"a_bar_{label}"] = a_bar.astype(np.float32)
        npz[f"w_bar_{label}"] = w_bar.astype(np.float32)

        per_layer = []
        for l in range(n_layers):
            S = np.array(union_full[l], dtype=np.int64)
            k = 32
            order_all = np.argsort(-a_layer[l])
            top32_all = order_all[:k]
            # writable-only ranking (exclude the frozen sink slots)
            order_wr = np.array([i for i in order_all if i >= n_frozen], dtype=np.int64)
            top32_wr = order_wr[:k]
            topU_wr = order_wr[: len(S)]
            # per-head mass-ranked top-32 (each head keeps its own set)
            ph_norm = float(
                np.mean([a_bar[l, h][np.argsort(-a_bar[l, h])[:k]].sum() for h in range(a_bar.shape[1])])
            )
            ph_norm_wr = float(
                np.mean(
                    [
                        a_bar[l, h][
                            np.array([i for i in np.argsort(-a_bar[l, h]) if i >= n_frozen])[:k]
                        ].sum()
                        for h in range(a_bar.shape[1])
                    ]
                )
            )
            tot_w = float(w_layer[l].sum())
            rec = {
                "layer": l,
                "n_union": int(len(S)),
                "mass_on_cart": tot_w,
                "share_union_norm": float(a_layer[l][S].sum()),
                "share_union_ratio": float(w_layer[l][S].sum() / tot_w),
                "abs_union": float(w_layer[l][S].sum()),
                "share_mass32_norm": float(a_layer[l][top32_all].sum()),
                "share_mass32_ratio": float(w_layer[l][top32_all].sum() / tot_w),
                "abs_mass32": float(w_layer[l][top32_all].sum()),
                "share_mass32_writable_norm": float(a_layer[l][top32_wr].sum()),
                "abs_mass32_writable": float(w_layer[l][top32_wr].sum()),
                "share_massU_writable_norm": float(a_layer[l][topU_wr].sum()),
                "share_mass32_perhead_norm": ph_norm,
                "share_mass32_perhead_writable_norm": ph_norm_wr,
                "share_sink_norm": float(a_layer[l][:n_frozen].sum()),
                "abs_sink": float(w_layer[l][:n_frozen].sum()),
                "n_union_in_top32_mass": int(len(set(S.tolist()) & set(top32_all.tolist()))),
                "top1_slot": int(order_all[0]),
                "top1_share_norm": float(a_layer[l][order_all[0]]),
            }
            per_layer.append(rec)

        def m(key):
            return float(np.mean([r[key] for r in per_layer]))

        out["splits"][label] = {
            "per_layer": per_layer,
            "summary": {k: m(k) for k in per_layer[0] if k not in ("layer", "top1_slot")},
        }
        s = out["splits"][label]["summary"]
        print(
            f"[measure] {label}: union {s['share_union_norm']:.4f} | "
            f"mass32 {s['share_mass32_norm']:.4f} | mass32-writable "
            f"{s['share_mass32_writable_norm']:.4f} | sink {s['share_sink_norm']:.4f} | "
            f"cart {s['mass_on_cart']:.4f}",
            flush=True,
        )

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(OUT_NPZ, **npz)
    print(f"[done] wrote {OUT_JSON} and {OUT_NPZ}", flush=True)


if __name__ == "__main__":
    main()
