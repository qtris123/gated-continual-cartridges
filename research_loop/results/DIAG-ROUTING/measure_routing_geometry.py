"""DIAG-ROUTING (B-ROUTE / B-CAP): geometry of the simplex routing vectors.

Standalone diagnostic — nothing under ``cartridges/`` is modified, and no cache is
written. Forward passes only.

The "key" of our closed-form write is the **simplex routing vector**
``a_S(q) = alpha[:, S]`` (the design matrix ``X`` at ``cartridges/am/value_solve.py:86``),
where ``alpha = softmax(qK_cart^T/sqrt(d))`` is the softmax restricted to the 512
cartridge slots.  At ``top_t = 512`` (all slots) ``a(q)`` is the full cartridge routing
distribution.

At eval time the model's softmax runs over ``[512 cartridge slots || own-sequence causal
prefix]`` (ORACLE-WRITE's convention).  Both objects are recovered from the SAME logits:

    w        = softmax over [cartridge || prefix]         (eval-time distribution)
    mass_cart= w[..., :T_c].sum(-1)                       (= mass_on_S at top_t=512)
    a        = w[..., :T_c] / mass_cart                   (= value_solve.py's alpha, exactly,
                                                           because softmax restricted to a
                                                           subset is the renormalised subset
                                                           softmax)

Measured, per (layer, KV-head), on the Phase-1 cartridge, over both eval splits:

 1. QA routing Gram   ``C0 = E_{q in QA}[a a^T]``  -> eigenspectrum, effective rank at
    several thresholds, approximate-null-space dimension.
 2. ``rho(tau)`` = fraction of MT routing ENERGY inside QA's approximate null space,
    ``rho = tr(P C_MT)/tr(C_MT)`` with ``P = I - U_r U_r^T``.  The QA self-leakage
    ``rho_QA`` is computed identically as the control.
 3. ``mass_on_S`` (both at top_t=512 = whole cartridge, and on the ORACLE-WRITE tf-idf
    top-32 slot union so the numbers are directly comparable) and routing entropy,
    MT queries vs QA queries.
 4. QA/MT mean-routing overlap: cosine of the mean routing vectors, histogram
    intersection (LIT-013), and top-k slot-set overlap.

Usage (env):
  PHASE1_CACHE   path to cache_last.pt
  EVAL_SPECS     "QA:path.parquet,MT:path.parquet"
  SLOTS_RUN_DIR  run dir with am_doc_*.pt (for the ORACLE-WRITE-comparable mass_on_S32)
  OUT_JSON / OUT_NPZ
"""

from __future__ import annotations

import json
import os
import time
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
SLOTS_RUN_DIR = os.environ.get("SLOTS_RUN_DIR", "")
OUT_JSON = os.environ["OUT_JSON"]
OUT_NPZ = os.environ["OUT_NPZ"]
CHUNK = int(os.environ.get("QCHUNK", "256"))
DEVICE = "cuda"

# relative eigenvalue thresholds: r(tau) = #{lambda_i > tau * lambda_max}
TAUS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
# GPM energy rule: smallest r with cumsum(lambda)_r >= eps * sum(lambda)
EPSS = [0.90, 0.95, 0.99, 0.999, 0.9999]
# explicit "forbid the top-r QA directions" sweep
RDIMS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 448, 480, 496, 504, 511]


# --------------------------------------------------------------------------------------
# capture post-RoPE q/k exactly as ORACLE-WRITE/measure_route_mass.py does
# --------------------------------------------------------------------------------------
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


def slot_union_from_run_dir(run_dir: str) -> dict[int, torch.Tensor]:
    files = sorted(glob(os.path.join(run_dir, "am_doc_*.pt")))
    if not files:
        raise FileNotFoundError(f"no am_doc_*.pt under {run_dir}")
    per_layer: dict[int, set] = defaultdict(set)
    for f in files:
        payload = torch.load(f, map_location="cpu", weights_only=False)
        mask = payload["am_stats"].mask
        assert mask.granularity == "per_layer", mask.granularity
        for layer_idx, pos in mask.positions_per_layer.items():
            per_layer[int(layer_idx)].update(pos.flatten().tolist())
    out = {l: torch.tensor(sorted(v), dtype=torch.long) for l, v in per_layer.items()}
    print(
        f"[slots] {run_dir}: {len(files)} docs, union/layer "
        f"min={min(len(v) for v in out.values())} max={max(len(v) for v in out.values())}",
        flush=True,
    )
    return out


# --------------------------------------------------------------------------------------
# pass 1: accumulate the routing Gram + mass/entropy statistics
# --------------------------------------------------------------------------------------
@torch.no_grad()
def collect(model, cache, dataloader, n_layers, S32, label):
    captured, handles = install_qk_capture(model)
    n_frozen = cache._num_frozen_tokens
    T_c = cache.num_cartridge_tokens()

    gram = None      # (L, H, T_c, T_c) float64
    mean_vec = None  # (L, H, T_c) float64
    n_rows = 0
    n_rows_sc = 0
    acc = None       # scalar sums, (L, H, K)
    KEYS = [
        "mass_cart", "mass_S32", "H_a", "H_full", "sq_norm",
        "mass_cart_sc", "mass_S32_sc", "H_a_sc", "H_full_sc", "sq_norm_sc",
    ]
    t0 = time.time()
    n_batches = 0
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
            scored_mask = torch.zeros(L, dtype=torch.bool, device=DEVICE)
            if batch.topk_token_idxs is not None:
                idx = (batch.topk_token_idxs.to(DEVICE).flatten() - 1).clamp(min=0).unique()
                scored_mask[idx] = True

            for layer_idx in range(n_layers):
                q, k_seq = captured[layer_idx]
                k_cart = (
                    torch.cat(
                        [cache.frozen_keys[layer_idx], cache.trainable_keys[layer_idx]], dim=2
                    )[0]
                    if n_frozen > 0
                    else cache.trainable_keys[layer_idx][0]
                )  # (H, T_c, d)
                H, _, d = k_cart.shape
                n_q = q.shape[1]
                g = n_q // H
                scaling = d ** -0.5
                if gram is None:
                    gram = torch.zeros(n_layers, H, T_c, T_c, dtype=torch.float64, device=DEVICE)
                    mean_vec = torch.zeros(n_layers, H, T_c, dtype=torch.float64, device=DEVICE)
                    acc = torch.zeros(n_layers, H, len(KEYS), dtype=torch.float64, device=DEVICE)
                qh = q[0].reshape(H, g, L, d).float()
                ks = k_seq[0].float().contiguous()          # (H, L, d)
                kc = k_cart.float().contiguous()            # (H, T_c, d)
                Ssel = (S32[layer_idx].to(DEVICE) + n_frozen)

                for c0 in range(0, L, CHUNK):
                    c1 = min(c0 + CHUNK, L)
                    C = c1 - c0
                    qc = qh[:, :, c0:c1, :].reshape(H, g * C, d)
                    s_cart = torch.bmm(qc, kc.transpose(1, 2)) * scaling      # (H, g*C, T_c)
                    s_seq = torch.bmm(qc, ks.transpose(1, 2)) * scaling       # (H, g*C, L)
                    vis = seq_vis[c0:c1].unsqueeze(0).expand(g, C, L).reshape(1, g * C, L)
                    s_seq = s_seq.masked_fill(~vis, float("-inf"))
                    s = torch.cat([s_cart, s_seq], dim=-1)
                    w = torch.softmax(s, dim=-1)
                    del s, s_cart, s_seq
                    w_c = w[..., :T_c]
                    mass_cart = w_c.sum(-1)                                   # (H, g*C)
                    mass_S32 = w[..., Ssel].sum(-1)
                    H_full = -(torch.xlogy(w, w)).sum(-1)
                    del w
                    a = w_c / mass_cart.clamp_min(1e-30).unsqueeze(-1)
                    del w_c
                    H_a = -(torch.xlogy(a, a)).sum(-1)
                    sq = (a * a).sum(-1)

                    gram[layer_idx] += torch.bmm(a.transpose(1, 2), a).double()
                    mean_vec[layer_idx] += a.sum(1).double()

                    sm = scored_mask[c0:c1].unsqueeze(0).expand(g, C).reshape(1, g * C)
                    smf = sm.to(a.dtype)
                    stack = torch.stack(
                        [
                            mass_cart.sum(1), mass_S32.sum(1), H_a.sum(1),
                            H_full.sum(1), sq.sum(1),
                            (mass_cart * smf).sum(1), (mass_S32 * smf).sum(1),
                            (H_a * smf).sum(1), (H_full * smf).sum(1), (sq * smf).sum(1),
                        ],
                        dim=-1,
                    ).double()
                    acc[layer_idx] += stack
                    if layer_idx == 0:
                        n_rows += g * C
                        n_rows_sc += int(sm.sum().item())
                    del a, mass_cart, mass_S32, H_a, H_full, sq
            cache.clear()
            n_batches += 1
            print(f"[collect:{label}] batch {n_batches} done ({time.time()-t0:.1f}s)", flush=True)
    finally:
        for h in handles:
            h.remove()

    print(
        f"[collect:{label}] {n_batches} batches, rows/head={n_rows}, scored rows/head={n_rows_sc}, "
        f"{time.time()-t0:.1f}s",
        flush=True,
    )
    stats = {KEYS[i]: (acc[..., i] / max(n_rows, 1)).cpu().numpy() for i in range(5)}
    for i in range(5, len(KEYS)):
        stats[KEYS[i]] = (acc[..., i] / max(n_rows_sc, 1)).cpu().numpy()
    return {
        "gram": gram / max(n_rows, 1),
        "mean_vec": mean_vec / max(n_rows, 1),
        "n_rows": n_rows,
        "n_rows_scored": n_rows_sc,
        "stats": stats,
        "n_batches": n_batches,
    }


# --------------------------------------------------------------------------------------
def spectral_analysis(C0: torch.Tensor, CMT: torch.Tensor):
    """C0, CMT: (H, T, T) float64. Returns per-head arrays."""
    evals, U = torch.linalg.eigh(C0)                 # ascending
    evals = evals.flip(-1).clamp_min(0.0)            # descending, (H, T)
    U = U.flip(-1)                                   # (H, T, T) columns = eigvecs
    # MT energy along each QA eigendirection: e_i = u_i^T C_MT u_i
    e_mt = ((CMT @ U) * U).sum(1).clamp_min(0.0)     # (H, T)
    return evals, e_mt, U


def rank_metrics(evals: np.ndarray):
    """evals: (..., T) descending, nonneg. Returns dict of rank statistics."""
    T = evals.shape[-1]
    lmax = evals[..., :1]
    tot = evals.sum(-1, keepdims=True)
    out = {}
    for tau in TAUS:
        out[f"r_tau{tau:g}"] = (evals > tau * lmax).sum(-1).astype(np.float64)
    cs = np.cumsum(evals, axis=-1) / np.maximum(tot, 1e-300)
    for eps in EPSS:
        out[f"r_energy{eps:g}"] = (cs < eps).sum(-1).astype(np.float64) + 1.0
    p = evals / np.maximum(tot, 1e-300)
    out["erank_participation"] = (tot[..., 0] ** 2) / np.maximum((evals ** 2).sum(-1), 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.where(p > 0, p * np.log(p), 0.0).sum(-1)
    out["erank_entropy"] = np.exp(ent)
    out["lambda_max"] = evals[..., 0]
    out["trace"] = tot[..., 0]
    out["cond_max_over_min"] = evals[..., 0] / np.maximum(evals[..., -1], 1e-300)
    out["T"] = np.full(evals.shape[:-1], float(T))
    return out


def rho_at(e_mt: np.ndarray, ranks: np.ndarray):
    """fraction of MT energy outside the top-r QA eigendirections. ranks broadcastable."""
    cs = np.cumsum(e_mt, axis=-1)
    tot = cs[..., -1]
    r = np.clip(ranks.astype(int), 0, e_mt.shape[-1])
    inside = np.take_along_axis(
        np.concatenate([np.zeros(e_mt.shape[:-1] + (1,)), cs], axis=-1),
        r[..., None],
        axis=-1,
    )[..., 0]
    return 1.0 - inside / np.maximum(tot, 1e-300)


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int):
    ia = np.argsort(-a, axis=-1)[..., :k]
    ib = np.argsort(-b, axis=-1)[..., :k]
    out = np.zeros(a.shape[:-1])
    it = np.ndindex(*a.shape[:-1])
    for idx in it:
        out[idx] = len(set(ia[idx].tolist()) & set(ib[idx].tolist())) / k
    return out


def main():
    print(f"[import] cartridges={os.path.dirname(cartridges.__file__)}", flush=True)
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

    cache = TrainableCache.from_pretrained(PHASE1_CACHE, device=DEVICE).to(DEVICE)
    n_layers = cache.config.n_layers
    T_c = cache.num_cartridge_tokens()
    n_frozen = cache._num_frozen_tokens
    print(f"[cache] {PHASE1_CACHE} n_layers={n_layers} T_c={T_c} n_frozen={n_frozen}", flush=True)

    if SLOTS_RUN_DIR:
        S32 = slot_union_from_run_dir(SLOTS_RUN_DIR)
    else:
        S32 = {l: torch.arange(T_c - n_frozen) for l in range(n_layers)}

    evals_dl = {}
    for spec in EVAL_SPECS.split(","):
        label, path = spec.split(":", 1)
        ds = LossEvalDataset.Config(
            data_source=DataSource(path=path, type="local"), packed_seq_length=2048
        ).instantiate(tokenizer=tokenizer, seed=0)
        evals_dl[label] = DataLoader(ds, batch_size=1, collate_fn=lambda b: b[0], num_workers=0)
        print(f"[eval] {label}: n={len(ds)} batches", flush=True)

    res = {}
    for label, dl in evals_dl.items():
        res[label] = collect(model, cache, dl, n_layers, S32, label)

    del model
    torch.cuda.empty_cache()

    QA, MT = res["QA"], res["MT"]
    H = QA["gram"].shape[1]
    T = QA["gram"].shape[2]

    per_head = {
        "evals_qa": np.zeros((n_layers, H, T)),
        "e_mt": np.zeros((n_layers, H, T)),
    }
    rank_arrays = defaultdict(lambda: np.zeros((n_layers, H)))
    rho_mt = {f"r{r}": np.zeros((n_layers, H)) for r in RDIMS}
    rho_qa = {f"r{r}": np.zeros((n_layers, H)) for r in RDIMS}

    for l in range(n_layers):
        C0 = QA["gram"][l]
        CMT = MT["gram"][l]
        ev, emt, _U = spectral_analysis(C0, CMT)
        ev_np = ev.cpu().numpy()
        emt_np = emt.cpu().numpy()
        per_head["evals_qa"][l] = ev_np
        per_head["e_mt"][l] = emt_np
        rk = rank_metrics(ev_np)
        for k, v in rk.items():
            rank_arrays[k][l] = v
        for r in RDIMS:
            rho_mt[f"r{r}"][l] = rho_at(emt_np, np.full(H, r))
            rho_qa[f"r{r}"][l] = rho_at(ev_np, np.full(H, r))
        print(f"[spec] layer {l} done", flush=True)

    # rho at each rank RULE (per-head rank varies), for the thresholds that matter
    rho_rule_mt, rho_rule_qa = {}, {}
    for key in list(rank_arrays.keys()):
        if not (key.startswith("r_tau") or key.startswith("r_energy")):
            continue
        m = np.zeros((n_layers, H))
        qa = np.zeros((n_layers, H))
        for l in range(n_layers):
            m[l] = rho_at(per_head["e_mt"][l], rank_arrays[key][l])
            qa[l] = rho_at(per_head["evals_qa"][l], rank_arrays[key][l])
        rho_rule_mt[key] = m
        rho_rule_qa[key] = qa

    # ---- item 4: mean-routing overlap -------------------------------------------------
    a_qa = QA["mean_vec"].cpu().numpy()   # (L, H, T), each row sums to 1
    a_mt = MT["mean_vec"].cpu().numpy()
    cos = (a_qa * a_mt).sum(-1) / np.maximum(
        np.linalg.norm(a_qa, axis=-1) * np.linalg.norm(a_mt, axis=-1), 1e-300
    )
    hist_int = np.minimum(a_qa, a_mt).sum(-1)
    ovl = {f"top{k}": topk_overlap(a_qa, a_mt, k) for k in (16, 32, 64, 128)}

    # ---- pooled (Gram averaged over all 288 heads) ------------------------------------
    C0_pool = QA["gram"].mean(dim=(0, 1))
    CMT_pool = MT["gram"].mean(dim=(0, 1))
    ev_p, emt_p, _ = spectral_analysis(C0_pool[None], CMT_pool[None])
    ev_p = ev_p.cpu().numpy()[0]
    emt_p = emt_p.cpu().numpy()[0]
    rk_p = {k: float(v[0]) for k, v in rank_metrics(ev_p[None]).items()}
    rho_p_mt = {f"r{r}": float(rho_at(emt_p[None], np.array([r]))[0]) for r in RDIMS}
    rho_p_qa = {f"r{r}": float(rho_at(ev_p[None], np.array([r]))[0]) for r in RDIMS}
    rho_p_rule_mt = {
        k: float(rho_at(emt_p[None], np.array([rk_p[k]]))[0])
        for k in rk_p if k.startswith("r_tau") or k.startswith("r_energy")
    }

    def pl(x):  # per-layer mean over heads
        return [float(v) for v in np.asarray(x).mean(axis=-1)]

    st_qa, st_mt = QA["stats"], MT["stats"]
    out = {
        "id": "DIAG-ROUTING",
        "provenance": {
            "snapshot_path": os.environ.get("AMSNAP", ""),
            "cartridges_import_path": os.path.dirname(cartridges.__file__),
            "git_head": os.environ.get("SNAP_GIT_HEAD", ""),
            "snapshot_manifest_sha256": os.environ.get("SNAP_SHA", ""),
            "model": MODEL_NAME,
            "cache": PHASE1_CACHE,
            "eval_specs": EVAL_SPECS,
            "slots_run_dir": SLOTS_RUN_DIR,
            "n_layers": n_layers,
            "n_kv_heads": int(H),
            "T_cartridge": int(T),
            "n_query_rows_per_head_QA": QA["n_rows"],
            "n_query_rows_per_head_MT": MT["n_rows"],
            "n_scored_rows_per_head_QA": QA["n_rows_scored"],
            "n_scored_rows_per_head_MT": MT["n_rows_scored"],
            "routing_vector_definition": (
                "a(q) = softmax(qK_cart^T/sqrt(d)) over the 512 cartridge slots, recovered as "
                "w[:, :T_c]/w[:, :T_c].sum() from the FULL eval-time softmax over "
                "[512 slots || own-sequence causal prefix]; identical to value_solve.py:86 X at top_t=512"
            ),
            "gram_definition": "C = (1/N) sum_q a(q) a(q)^T, N = all query positions x all query heads",
        },
        "thresholds": {"taus": TAUS, "energy_eps": EPSS, "rdims": RDIMS},
        "per_layer": {
            "rank": {k: pl(v) for k, v in rank_arrays.items()},
            "rho_mt_at_rdim": {k: pl(v) for k, v in rho_mt.items()},
            "rho_qa_at_rdim": {k: pl(v) for k, v in rho_qa.items()},
            "rho_mt_at_rule": {k: pl(v) for k, v in rho_rule_mt.items()},
            "rho_qa_at_rule": {k: pl(v) for k, v in rho_rule_qa.items()},
            "mass_on_S_top512_QA": pl(st_qa["mass_cart"]),
            "mass_on_S_top512_MT": pl(st_mt["mass_cart"]),
            "mass_on_S_top512_QA_scored": pl(st_qa["mass_cart_sc"]),
            "mass_on_S_top512_MT_scored": pl(st_mt["mass_cart_sc"]),
            "mass_on_S32union_QA": pl(st_qa["mass_S32"]),
            "mass_on_S32union_MT": pl(st_mt["mass_S32"]),
            "mass_on_S32union_QA_scored": pl(st_qa["mass_S32_sc"]),
            "mass_on_S32union_MT_scored": pl(st_mt["mass_S32_sc"]),
            "routing_entropy_nats_QA": pl(st_qa["H_a"]),
            "routing_entropy_nats_MT": pl(st_mt["H_a"]),
            "routing_entropy_nats_QA_scored": pl(st_qa["H_a_sc"]),
            "routing_entropy_nats_MT_scored": pl(st_mt["H_a_sc"]),
            "full_softmax_entropy_nats_QA": pl(st_qa["H_full"]),
            "full_softmax_entropy_nats_MT": pl(st_mt["H_full"]),
            "eff_slots_expH_QA": pl(np.exp(st_qa["H_a"])),
            "eff_slots_expH_MT": pl(np.exp(st_mt["H_a"])),
            "sq_norm_a_QA": pl(st_qa["sq_norm"]),
            "sq_norm_a_MT": pl(st_mt["sq_norm"]),
            "mean_routing_cosine_QA_MT": pl(cos),
            "mean_routing_hist_intersection_QA_MT": pl(hist_int),
            **{f"mean_routing_top{k}_overlap": pl(v) for k, v in
               [(kk.replace("top", ""), vv) for kk, vv in ovl.items()]},
            "n_slots_S32union": [int(S32[l].numel()) for l in range(n_layers)],
        },
        "pooled": {
            "rank": rk_p,
            "rho_mt_at_rdim": rho_p_mt,
            "rho_qa_at_rdim": rho_p_qa,
            "rho_mt_at_rule": rho_p_rule_mt,
            "spectrum_normalised_head": [float(x) for x in (ev_p / max(ev_p[0], 1e-300))[:64]],
        },
        "summary": {
            "mean_over_heads": {
                **{f"rank_{k}": float(np.mean(v)) for k, v in rank_arrays.items()},
                **{f"rho_mt_{k}": float(np.mean(v)) for k, v in rho_mt.items()},
                **{f"rho_qa_{k}": float(np.mean(v)) for k, v in rho_qa.items()},
                **{f"rho_mt_rule_{k}": float(np.mean(v)) for k, v in rho_rule_mt.items()},
                **{f"rho_qa_rule_{k}": float(np.mean(v)) for k, v in rho_rule_qa.items()},
                "mass_on_S_top512_QA": float(np.mean(st_qa["mass_cart"])),
                "mass_on_S_top512_MT": float(np.mean(st_mt["mass_cart"])),
                "mass_on_S_top512_QA_scored": float(np.mean(st_qa["mass_cart_sc"])),
                "mass_on_S_top512_MT_scored": float(np.mean(st_mt["mass_cart_sc"])),
                "mass_on_S32union_QA": float(np.mean(st_qa["mass_S32"])),
                "mass_on_S32union_MT": float(np.mean(st_mt["mass_S32"])),
                "mass_on_S32union_QA_scored": float(np.mean(st_qa["mass_S32_sc"])),
                "mass_on_S32union_MT_scored": float(np.mean(st_mt["mass_S32_sc"])),
                "mt_over_qa_mass_ratio_S32union": float(
                    np.mean(st_mt["mass_S32"]) / max(np.mean(st_qa["mass_S32"]), 1e-30)
                ),
                "mt_over_qa_mass_ratio_top512": float(
                    np.mean(st_mt["mass_cart"]) / max(np.mean(st_qa["mass_cart"]), 1e-30)
                ),
                "routing_entropy_nats_QA": float(np.mean(st_qa["H_a"])),
                "routing_entropy_nats_MT": float(np.mean(st_mt["H_a"])),
                "eff_slots_expH_QA": float(np.mean(np.exp(st_qa["H_a"]))),
                "eff_slots_expH_MT": float(np.mean(np.exp(st_mt["H_a"]))),
                "mean_routing_cosine_QA_MT": float(np.mean(cos)),
                "mean_routing_hist_intersection_QA_MT": float(np.mean(hist_int)),
                **{f"mean_routing_top{k}_overlap": float(np.mean(v))
                   for k, v in [(kk.replace("top", ""), vv) for kk, vv in ovl.items()]},
            }
        },
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(
        OUT_NPZ,
        evals_qa=per_head["evals_qa"].astype(np.float32),
        e_mt=per_head["e_mt"].astype(np.float32),
        mean_routing_qa=a_qa.astype(np.float32),
        mean_routing_mt=a_mt.astype(np.float32),
        **{f"rank_{k}": v.astype(np.float32) for k, v in rank_arrays.items()},
        **{f"rho_mt_{k}": v.astype(np.float32) for k, v in rho_mt.items()},
        **{f"rho_qa_{k}": v.astype(np.float32) for k, v in rho_qa.items()},
    )
    print(f"[done] wrote {OUT_JSON} and {OUT_NPZ}", flush=True)

    s = out["summary"]["mean_over_heads"]
    for k in sorted(s):
        print(f"  SUMMARY {k} = {s[k]:.6g}", flush=True)

    # ------------------------------------------------------------------ wandb
    if os.environ.get("WANDB_DISABLED", "0") != "1":
        import wandb

        run = wandb.init(
            project=os.environ.get("CARTRIDGES_WANDB_PROJECT", "SEACrowd"),
            entity=os.environ.get("CARTRIDGES_WANDB_ENTITY", None),
            name=os.environ.get("RUN_NAME", "DIAG-ROUTING_geometry"),
            group=os.environ.get("WANDB_GROUP", "B-ROUTE"),
            tags=["diagnostic", "B-ROUTE", "B-CAP", "DIAG-ROUTING"],
            notes=os.environ.get("WANDB_NOTES", ""),
            config=out["provenance"] | {"thresholds": out["thresholds"]},
        )
        wandb.summary.update({f"diag/{k}": v for k, v in s.items()})
        wandb.summary.update({f"diag/pooled_{k}": v for k, v in rk_p.items()})
        wandb.summary.update({f"diag/pooled_rho_mt_{k}": v for k, v in rho_p_mt.items()})
        for l in range(n_layers):
            row = {"layer": l}
            for k, v in out["per_layer"].items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        row[f"diag/{k}/{k2}"] = v2[l]
                else:
                    row[f"diag/{k}"] = v[l]
            wandb.log(row, step=l)
        print(f"WANDB_RUN_URL={run.url}", flush=True)
        print(f"WANDB_RUN_ID={run.id}", flush=True)
        wandb.finish()


if __name__ == "__main__":
    main()
