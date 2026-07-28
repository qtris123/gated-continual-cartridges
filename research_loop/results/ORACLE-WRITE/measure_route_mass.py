"""B-ROUTE diagnostic: eval-time attention mass on the rewritten cartridge slots.

Standalone (nothing in `cartridges/` is modified by this file), so it cannot perturb
any training/eval path. For each (cartridge checkpoint, slot-set, eval split) it runs
the SAME forward the perplexity eval runs, recomputes post-RoPE q/k per layer from the
hidden states, rebuilds the exact attention distribution the model sees
    softmax over [ cartridge (always visible) || own-sequence causal prefix ]
and reports the probability mass that lands on the rewritten slots S, per layer.

Reported per layer:
  mass_on_S        - mean over query tokens/heads of sum_{j in S} alpha_j
  mass_on_cart     - mean total mass on the whole 512-slot cartridge
  share_of_cart    - mass_on_S / mass_on_cart
  *_scored         - same, restricted to the query positions that the CE is scored at
                     (i.e. the assistant tokens: topk_token_idxs - 1)

Usage (env):
  CACHE_SPECS   "label:cache.pt:slots_run_dir,..."   slots_run_dir may be '-' (=> use
                                                     the previous spec's slot set)
  EVAL_SPECS    "QA:path.parquet,MT:path.parquet"
  OUT_JSON      output path
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from glob import glob

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from cartridges.cache import TrainableCache
from cartridges.datasets import DataSource, LossEvalDataset
from cartridges.models import FlexQwen3ForCausalLM, HFModelConfig
from cartridges.sparse_cache_finetuning import _apply_rotary_pos_emb

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
CACHE_SPECS = os.environ["CACHE_SPECS"]
EVAL_SPECS = os.environ["EVAL_SPECS"]
OUT_JSON = os.environ["OUT_JSON"]
DEVICE = "cuda"


def install_qk_capture(model):
    captured: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    handles = []
    layers = model.model.layers
    for layer_idx, decoder_layer in enumerate(layers):
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


def slot_union_from_run_dir(run_dir: str) -> dict[int, torch.Tensor]:
    """Union over documents of the per-layer selected positions (trainable index space)."""
    files = sorted(glob(os.path.join(run_dir, "am_doc_*.pt")))
    if not files:
        raise FileNotFoundError(f"no am_doc_*.pt under {run_dir}")
    per_layer: dict[int, set] = defaultdict(set)
    per_doc_counts = []
    for f in files:
        payload = torch.load(f, map_location="cpu", weights_only=False)
        mask = payload["am_stats"].mask
        assert mask.granularity == "per_layer", mask.granularity
        n = 0
        for layer_idx, pos in mask.positions_per_layer.items():
            p = pos.flatten().tolist()
            per_layer[int(layer_idx)].update(p)
            n = len(p)
        per_doc_counts.append(n)
    out = {l: torch.tensor(sorted(v), dtype=torch.long) for l, v in per_layer.items()}
    print(
        f"[slots] {run_dir}: {len(files)} docs, top_t/doc={per_doc_counts[0]}, "
        f"union/layer min={min(len(v) for v in out.values())} "
        f"max={max(len(v) for v in out.values())}",
        flush=True,
    )
    return out


@torch.no_grad()
def measure(model, cache, slots, dataloader, n_layers: int):
    captured, handles = install_qk_capture(model)
    n_frozen = cache._num_frozen_tokens
    T_c = cache.num_cartridge_tokens()
    acc = {
        l: dict(
            sum_S=0.0, sum_cart=0.0, cnt=0,
            sum_S_sc=0.0, sum_cart_sc=0.0, cnt_sc=0,
        )
        for l in range(n_layers)
    }
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
            scored = None
            if batch.topk_token_idxs is not None:
                scored = (batch.topk_token_idxs.to(DEVICE).flatten() - 1).clamp(min=0).unique()

            for layer_idx in range(n_layers):
                q, k_seq = captured[layer_idx]
                k_cart = torch.cat(
                    [cache.frozen_keys[layer_idx], cache.trainable_keys[layer_idx]], dim=2
                )[0] if n_frozen > 0 else cache.trainable_keys[layer_idx][0]
                n_kv = k_cart.shape[0]
                n_q = q.shape[1]
                g = n_q // n_kv
                head_dim = k_cart.shape[-1]
                scaling = head_dim ** -0.5
                S = (slots[layer_idx].to(DEVICE) + n_frozen)
                for h in range(n_kv):
                    qh = q[0, h * g : (h + 1) * g].float()          # (g, L, d)
                    Kh = torch.cat([k_cart[h], k_seq[0, h]], dim=0).float()  # (T_c+L, d)
                    s = (qh @ Kh.T) * scaling                        # (g, L, T_c+L)
                    s[..., T_c:] = s[..., T_c:].masked_fill(~seq_vis, float("-inf"))
                    w = torch.softmax(s, dim=-1)
                    m_S = w[..., S].sum(-1)                          # (g, L)
                    m_c = w[..., :T_c].sum(-1)                       # (g, L)
                    a = acc[layer_idx]
                    a["sum_S"] += m_S.sum().item()
                    a["sum_cart"] += m_c.sum().item()
                    a["cnt"] += m_S.numel()
                    if scored is not None and scored.numel() > 0:
                        a["sum_S_sc"] += m_S[:, scored].sum().item()
                        a["sum_cart_sc"] += m_c[:, scored].sum().item()
                        a["cnt_sc"] += m_S[:, scored].numel()
                    del s, w
            cache.clear()
    finally:
        for h in handles:
            h.remove()

    out = {"per_layer": {}, "n_slots_per_layer": {}}
    for l in range(n_layers):
        a = acc[l]
        mS = a["sum_S"] / max(a["cnt"], 1)
        mC = a["sum_cart"] / max(a["cnt"], 1)
        mSs = a["sum_S_sc"] / max(a["cnt_sc"], 1)
        mCs = a["sum_cart_sc"] / max(a["cnt_sc"], 1)
        out["per_layer"][l] = {
            "mass_on_S": mS,
            "mass_on_cart": mC,
            "share_of_cart": mS / mC if mC > 0 else 0.0,
            "mass_on_S_scored": mSs,
            "mass_on_cart_scored": mCs,
            "share_of_cart_scored": mSs / mCs if mCs > 0 else 0.0,
        }
        out["n_slots_per_layer"][l] = int(slots[l].numel())
    vals = [out["per_layer"][l]["mass_on_S"] for l in range(n_layers)]
    vals_sc = [out["per_layer"][l]["mass_on_S_scored"] for l in range(n_layers)]
    cart = [out["per_layer"][l]["mass_on_cart"] for l in range(n_layers)]
    out["summary"] = {
        "mean_mass_on_S": sum(vals) / len(vals),
        "mean_mass_on_S_scored": sum(vals_sc) / len(vals_sc),
        "mean_mass_on_cart": sum(cart) / len(cart),
        "max_mass_on_S": max(vals),
        "argmax_layer": int(max(range(n_layers), key=lambda l: vals[l])),
    }
    return out


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

    results = {"model": MODEL_NAME, "runs": {}}
    prev_slots = None
    for spec in CACHE_SPECS.split(","):
        label, cache_path, slots_dir = spec.split(":", 2)
        slots = prev_slots if slots_dir == "-" else slot_union_from_run_dir(slots_dir)
        prev_slots = slots
        cache = TrainableCache.from_pretrained(cache_path, device=DEVICE)
        cache = cache.to(DEVICE)
        n_layers = cache.config.n_layers
        for eval_label, dl in evals.items():
            print(f"[measure] cache={label} split={eval_label}", flush=True)
            r = measure(model, cache, slots, dl, n_layers)
            r["cache_path"] = cache_path
            r["slots_from"] = slots_dir
            results["runs"][f"{label}|{eval_label}"] = r
            print(
                f"[measure] {label}|{eval_label}: mean_mass_on_S="
                f"{r['summary']['mean_mass_on_S']:.5f} "
                f"scored={r['summary']['mean_mass_on_S_scored']:.5f} "
                f"cart={r['summary']['mean_mass_on_cart']:.5f}",
                flush=True,
            )
        del cache
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
