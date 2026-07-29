"""DIAG-IMPORTANCE (B-GATE): six per-slot scores for the 511 writable cartridge slots.

Standalone diagnostic. Nothing under ``cartridges/`` is modified, no cache is written,
and **no optimizer step is taken** (the Fisher pass runs ``backward()`` purely to read
``.grad``; ``gradient_steps`` stays 0).

Everything is measured on the untouched Phase-1 cartridge
(``outputs/phase1_selfdistill_qwen512/cache_last.pt``) over the QA and MT eval splits.

Conventions (identical to DIAG-ROUTING/measure_routing_geometry.py, which agrees with
ORACLE-WRITE to 4-6 s.f.):

    w         = softmax over [512 cartridge slots || own-sequence causal prefix]
                (the model's real eval-time attention distribution, per query head)
    mass_cart = w[..., :T_c].sum(-1)
    a         = w[..., :T_c] / mass_cart            (= value_solve.py:86 X at top_t=512)

Slot indices: cartridge position 0 is the frozen sink (unwritable); positions 1..511 are
the writable trainable slots and are what every array here reports.

Scores (per layer l, per writable slot j), all accumulated PER EVAL EXAMPLE so that a
bootstrap over examples is possible:

 1. tf_mass          E_{h,q}[a_j]                     (the incumbent ranker's signal)
    ranker_tf        the *exact* incumbent score: softmax of the GQA-group-mean query
                     over the 511 trainable keys only (query_accum.py:80-95)
 2. entropy          H_j = -sum_q p_jq log p_jq, p_jq = a_jq / sum_q a_jq
                     computed exactly from the sufficient statistics
                     s_j = sum_q a_jq and t_j = sum_q a_jq log a_jq  =>  H_j = log s_j - t_j/s_j
 3. kl_loo           E_q[ KL(a^{(-j)} || a) ] with a^{(-j)} the FULL eval-time attention
                     distribution with slot j deleted and renormalised.  Exactly
                     KL = -log(1 - w_j), so kl_loo_j = E_q[-log1p(-w_j)].
 4. fisher           E_e[ (dL_e/dv_{l,:,j,:})^2 summed over kv-heads and head dims ],
                     L_e = mean CE over eval example e's scored tokens (DIAG-NOISE's
                     per-example loss).  Empirical diagonal Fisher.
 5. redundancy       1 - r_j^2/||v_j||^2 where r_j^2 = min ||v_j - sum_{i != j} c_i v_i||^2
                     over the OTHER 511 slots, v_i in R^{H*d}=R^1024 (heads concatenated).
                     Closed form: r_j^2 = 1/(G^-1)_jj with G = V V^T (512x512).
 6. contrast         log(tf_mass_mt_j / tf_mass_qa_j)

Usage (env): PHASE1_CACHE, EVAL_SPECS, SLOTS_RUN_DIR, OUT_JSON, OUT_NPZ.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import cartridges
from cartridges.cache import TrainableCache
from cartridges.datasets import DataSource, LossEvalDataset
from cartridges.models import FlexQwen3ForCausalLM, HFModelConfig
from cartridges.sparse_cache_finetuning import _apply_rotary_pos_emb
from cartridges.train import CacheAndModel, _collate_first

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
PHASE1_CACHE = os.environ["PHASE1_CACHE"]
EVAL_SPECS = os.environ["EVAL_SPECS"]
SLOTS_RUN_DIR = os.environ.get("SLOTS_RUN_DIR", "")
OUT_JSON = os.environ["OUT_JSON"]
OUT_NPZ = os.environ["OUT_NPZ"]
CHUNK = int(os.environ.get("QCHUNK", "256"))
NBOOT = int(os.environ.get("NBOOT", "500"))
TOPK = int(os.environ.get("TOPK", "32"))
RIDGE_REL = float(os.environ.get("REDUNDANCY_RIDGE_REL", "1e-6"))
FP32_CHECK = os.environ.get("FP32_CHECK", "0") == "1"
MAXB = int(os.environ["MAXB"]) if os.environ.get("MAXB") else None   # smoke-test cap only
DEVICE = "cuda"
RNG = np.random.default_rng(20260729)


# ======================================================================================
# pass A: attention statistics (no grad)
# ======================================================================================
def install_qk_capture(model):
    """post-RoPE q/k capture, byte-for-byte the DIAG-ROUTING / ORACLE-WRITE hook."""
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


@torch.no_grad()
def collect_attention(model, cache, ds, n_layers, label):
    captured, handles = install_qk_capture(model)
    n_frozen = cache._num_frozen_tokens
    T_c = cache.num_cartridge_tokens()
    T_tr = T_c - n_frozen
    n_el = len(ds.elements)
    dl = DataLoader(ds, batch_size=1, collate_fn=_collate_first, num_workers=0)

    # per-element sufficient statistics, float64 on GPU, laid out (layer, element, slot)
    # so that S[layer] is a contiguous view for index_add_.
    S1 = torch.zeros(n_layers, n_el, T_c, dtype=torch.float64, device=DEVICE)  # sum a
    S2 = torch.zeros(n_layers, n_el, T_c, dtype=torch.float64, device=DEVICE)  # sum a log a
    S3 = torch.zeros(n_layers, n_el, T_c, dtype=torch.float64, device=DEVICE)  # sum -log1p(-w)
    S4 = torch.zeros(n_layers, n_el, T_c, dtype=torch.float64, device=DEVICE)  # sum w
    RK = torch.zeros(n_layers, n_el, T_tr, dtype=torch.float64, device=DEVICE)  # ranker score
    PH = None                                                       # (L, H, T_c) per-head sum a
    nrow = torch.zeros(n_el, dtype=torch.float64, device=DEVICE)   # query rows (heads x tokens)
    nrow_rk = torch.zeros(n_el, dtype=torch.float64, device=DEVICE)  # ranker rows (kv-heads x tok)
    ntok = torch.zeros(n_el, dtype=torch.float64, device=DEVICE)

    t0 = time.time()
    n_batches = 0
    H_kv = None
    try:
        for bi, batch in enumerate(dl):
            if MAXB is not None and bi >= MAXB:
                break
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
            gmap = torch.tensor(ds.batches[bi], dtype=torch.long, device=DEVICE)
            glob = gmap[seq_ids]                       # (L,) global element index per position
            ar = torch.arange(L, device=DEVICE)
            seq_vis = (seq_ids[:, None] == seq_ids[None, :]) & (ar[:, None] >= ar[None, :])
            ntok.index_add_(0, glob, torch.ones(L, dtype=torch.float64, device=DEVICE))

            for layer_idx in range(n_layers):
                q, k_seq = captured[layer_idx]
                k_tr = cache.trainable_keys[layer_idx][0]           # (H, T_tr, d)
                k_cart = (
                    torch.cat(
                        [cache.frozen_keys[layer_idx], cache.trainable_keys[layer_idx]], dim=2
                    )[0]
                    if n_frozen > 0
                    else k_tr
                )                                                   # (H, T_c, d)
                H, _, d = k_cart.shape
                H_kv = H
                n_q = q.shape[1]
                g = n_q // H
                scaling = d ** -0.5
                if PH is None:
                    PH = torch.zeros(n_layers, H, T_c, dtype=torch.float64, device=DEVICE)
                qh = q[0].reshape(H, g, L, d).float()
                ks = k_seq[0].float().contiguous()
                kc = k_cart.float().contiguous()
                ktr = k_tr.float().contiguous()

                for c0 in range(0, L, CHUNK):
                    c1 = min(c0 + CHUNK, L)
                    C = c1 - c0
                    gl = glob[c0:c1]
                    qc = qh[:, :, c0:c1, :].reshape(H, g * C, d)
                    s_cart = torch.bmm(qc, kc.transpose(1, 2)) * scaling
                    s_seq = torch.bmm(qc, ks.transpose(1, 2)) * scaling
                    vis = seq_vis[c0:c1].unsqueeze(0).expand(g, C, L).reshape(1, g * C, L)
                    s_seq = s_seq.masked_fill(~vis, float("-inf"))
                    s = torch.cat([s_cart, s_seq], dim=-1)
                    w = torch.softmax(s, dim=-1)
                    del s, s_cart, s_seq
                    w_c = w[..., :T_c].double()                      # (H, g*C, T_c)
                    del w
                    mass_cart = w_c.sum(-1, keepdim=True)
                    a = w_c / mass_cart.clamp_min(1e-300)

                    # -> (C, T_c): sum over kv-heads AND the g query heads in each group
                    def red(x):
                        return x.reshape(H, g, C, T_c).sum(dim=(0, 1))

                    S1[layer_idx].index_add_(0, gl, red(a))
                    S2[layer_idx].index_add_(0, gl, red(torch.xlogy(a, a)))
                    S3[layer_idx].index_add_(
                        0, gl, red(-torch.log1p(-w_c.clamp(max=1.0 - 1e-12)))
                    )
                    S4[layer_idx].index_add_(0, gl, red(w_c))
                    PH[layer_idx] += a.reshape(H, g, C, T_c).sum(dim=(1, 2))

                    # ---- incumbent ranker score (query_accum.py:80-95) ----
                    q_mean = qc.reshape(H, g, C, d).mean(dim=1)      # group mean BEFORE softmax
                    rk = torch.softmax(
                        torch.bmm(q_mean, ktr.transpose(1, 2)) * scaling, dim=-1
                    ).double()                                       # (H, C, T_tr)
                    RK[layer_idx].index_add_(0, gl, rk.sum(dim=0))

                    if layer_idx == 0:
                        nrow.index_add_(
                            0,
                            gl,
                            torch.full((C,), float(H * g), dtype=torch.float64, device=DEVICE),
                        )
                        nrow_rk.index_add_(
                            0,
                            gl,
                            torch.full((C,), float(H), dtype=torch.float64, device=DEVICE),
                        )
                    del a, w_c, mass_cart, rk, q_mean
            cache.clear()
            n_batches += 1
            print(f"[attn:{label}] batch {n_batches}/{len(ds.batches)} ({time.time()-t0:.1f}s)",
                  flush=True)
    finally:
        for h in handles:
            h.remove()
    print(f"[attn:{label}] done in {time.time()-t0:.1f}s, n_el={n_el}", flush=True)
    tr = lambda X: X.permute(1, 0, 2).contiguous().cpu().numpy()   # (L,E,T) -> (E,L,T)
    return {
        "S1": tr(S1), "S2": tr(S2), "S3": tr(S3), "S4": tr(S4), "RK": tr(RK),
        "PH": (PH / max(float(nrow.sum().item()) / H_kv, 1.0)).cpu().numpy(),
        "nrow": nrow.cpu().numpy(), "nrow_rk": nrow_rk.cpu().numpy(),
        "ntok": ntok.cpu().numpy(), "n_batches": n_batches,
        "secs": time.time() - t0,
    }


# ======================================================================================
# pass B: empirical diagonal Fisher of the eval CE w.r.t. each slot's value vector
# ======================================================================================
def collect_fisher(cam, cache, ds, n_layers, label, fp32=False, max_batches=None):
    """Returns (n_el, n_layers, T_tr) array of squared per-example value gradients.

    NOT an optimizer step: no optimizer is constructed, `.grad` is read then zeroed, and
    the cache tensors are never modified (`gradient_steps` stays 0).
    """
    n_el = len(ds.elements)
    T_tr = cache._num_trainable_tokens
    Fsh = np.zeros((n_el, n_layers, T_tr), dtype=np.float64)
    per_ex_loss = np.full(n_el, np.nan)
    per_ex_tok = np.zeros(n_el, dtype=np.int64)
    dl = DataLoader(ds, batch_size=1, collate_fn=_collate_first, num_workers=0)

    params = [cache.trainable_values[l] for l in range(n_layers)]
    prev_rg = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(True)
        p.grad = None

    def forward_ce(batch):
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=not fp32):
            out = cam(
                input_ids=batch.input_ids.to(DEVICE),
                seq_ids=batch.element_ids.to(DEVICE),
                position_ids=batch.position_ids.to(DEVICE),
            )
        lp = F.log_softmax(out.logits.float(), dim=-1)[
            0,
            batch.topk_token_idxs.to(DEVICE) - 1,
            batch.topk_token_ids.to(DEVICE),
        ]
        p = batch.topk_logprobs.to(DEVICE).exp()
        ce = -p * lp                                              # (n_scored,)
        cache.clear()
        return ce

    t0 = time.time()
    mode = ["retain_graph"]
    for bi, batch in enumerate(dl):
        if max_batches is not None and bi >= max_batches:
            break
        gmap = torch.tensor(ds.batches[bi], dtype=torch.long)
        idx = batch.topk_token_idxs.to(torch.long)
        glob = gmap[batch.element_ids[idx]]
        uniq = torch.unique(glob).tolist()
        masks = {e: (glob == e) for e in uniq}

        ce = forward_ce(batch)
        ce_d = ce.detach().double().cpu()
        for e in uniq:
            m_cpu = masks[e]
            t_e = int(m_cpu.sum().item())
            per_ex_tok[e] = t_e
            per_ex_loss[e] = float(ce_d[m_cpu].sum().item()) / t_e
            if mode[0] != "retain_graph" and ce is None:
                ce = forward_ce(batch)
            sel = m_cpu.to(DEVICE).to(ce.dtype)
            L_e = (ce * sel).sum() / t_e
            for p in params:
                p.grad = None
            try:
                L_e.backward(retain_graph=(mode[0] == "retain_graph"))
            except RuntimeError as exc:
                print(f"[fisher:{label}] retain_graph unsupported ({exc}); "
                      f"switching to one-forward-per-example", flush=True)
                mode[0] = "refresh"
                for p in params:
                    p.grad = None
                ce = forward_ce(batch)
                sel = m_cpu.to(DEVICE).to(ce.dtype)
                L_e = (ce * sel).sum() / t_e
                L_e.backward()
            for l in range(n_layers):
                gr = params[l].grad
                if gr is not None:   # (1, n_kv_heads, T_tr, head_dim)
                    Fsh[e, l] = (gr.double() ** 2).sum(dim=(0, 1, 3)).cpu().numpy()
            if mode[0] != "retain_graph":
                ce = None
        del ce
        for p in params:
            p.grad = None
        torch.cuda.empty_cache()
        print(f"[fisher:{label}] batch {bi+1}/{len(ds.batches)} "
              f"({len(uniq)} elements, {time.time()-t0:.1f}s, mode={mode[0]})", flush=True)

    for p, rg in zip(params, prev_rg):
        p.requires_grad_(rg)
        p.grad = None
    print(f"[fisher:{label}] done in {time.time()-t0:.1f}s", flush=True)
    return Fsh, per_ex_loss, per_ex_tok, time.time() - t0


# ======================================================================================
# redundancy
# ======================================================================================
@torch.no_grad()
def compute_redundancy(cache, n_layers, ridge_rel):
    n_frozen = cache._num_frozen_tokens
    out = {}
    for tag, rel in [("", ridge_rel), ("_ridge1e-4", 1e-4), ("_ridge1e-8", 1e-8)]:
        red = np.zeros((n_layers, cache._num_trainable_tokens))
        resid = np.zeros_like(red)
        vnorm = np.zeros_like(red)
        for l in range(n_layers):
            V = torch.cat(
                [cache.frozen_values[l], cache.trainable_values[l]], dim=2
            )[0] if n_frozen > 0 else cache.trainable_values[l][0]     # (H, T_c, d)
            Hh, T_c, d = V.shape
            Vf = V.permute(1, 0, 2).reshape(T_c, Hh * d).double()      # (T_c, H*d)
            G = Vf @ Vf.T
            lam = rel * float(torch.diagonal(G).mean().item())
            Gi = torch.linalg.inv(G + lam * torch.eye(T_c, dtype=G.dtype, device=G.device))
            r2 = 1.0 / torch.diagonal(Gi).clamp_min(1e-300)
            gdiag = torch.diagonal(G)
            rr = (1.0 - r2 / gdiag.clamp_min(1e-300)).clamp(0.0, 1.0)
            red[l] = rr[n_frozen:].cpu().numpy()
            resid[l] = r2[n_frozen:].sqrt().cpu().numpy()
            vnorm[l] = gdiag[n_frozen:].sqrt().cpu().numpy()
        out["redundancy" + tag] = red
        if tag == "":
            out["resid_norm"] = resid
            out["value_norm"] = vnorm
    return out


# ======================================================================================
# statistics helpers
# ======================================================================================
def rankdata(x, axis=-1):
    """average-rank (ties) along `axis`."""
    from scipy.stats import rankdata as _rd
    return _rd(x, axis=axis)


def spearman(a, b):
    ra, rb = rankdata(a.ravel()), rankdata(b.ravel())
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def topk_set(x, k, largest=True):
    """indices of the top-k along the last axis."""
    s = -x if largest else x
    return np.argsort(s, axis=-1, kind="stable")[..., :k]


def set_overlap(ia, ib, k):
    out = np.zeros(ia.shape[:-1])
    for idx in np.ndindex(*ia.shape[:-1]):
        out[idx] = len(set(ia[idx].tolist()) & set(ib[idx].tolist())) / k
    return out


def slot_union_from_run_dir(run_dir):
    files = sorted(glob(os.path.join(run_dir, "am_doc_*.pt")))
    if not files:
        raise FileNotFoundError(run_dir)
    per_doc = []
    for f in files:
        payload = torch.load(f, map_location="cpu", weights_only=False)
        mask = payload["am_stats"].mask
        assert mask.granularity == "per_layer", mask.granularity
        per_doc.append({int(l): p.flatten().tolist() for l, p in mask.positions_per_layer.items()})
    return per_doc


# ======================================================================================
def build_metrics(A, Fsh, sel=None):
    """Per-(layer, writable slot) scores from per-element sufficient statistics.

    `sel` = index array of eval examples to include (bootstrap resample) or None (all).
    """
    S1, S2, S3, S4, RK = A["S1"], A["S2"], A["S3"], A["S4"], A["RK"]
    if sel is None:
        s1, s2, s3, s4, rk = S1.sum(0), S2.sum(0), S3.sum(0), S4.sum(0), RK.sum(0)
        nr, nrk = A["nrow"].sum(), A["nrow_rk"].sum()
        fi = Fsh.mean(0) if Fsh is not None else None
    else:
        s1, s2, s3 = S1[sel].sum(0), S2[sel].sum(0), S3[sel].sum(0)
        s4, rk = S4[sel].sum(0), RK[sel].sum(0)
        nr, nrk = A["nrow"][sel].sum(), A["nrow_rk"][sel].sum()
        fi = Fsh[sel].mean(0) if Fsh is not None else None
    w = slice(1, None)      # drop the frozen sink slot 0
    ent = np.log(np.maximum(s1, 1e-300)) - s2 / np.maximum(s1, 1e-300)
    return {
        "tf_mass": (s1 / nr)[:, w],
        "w_mass": (s4 / nr)[:, w],
        "ranker_tf": (rk / nrk),
        "entropy": ent[:, w],
        "entropy_norm": (ent / np.log(max(nr, 2.0)))[:, w],
        "kl_loo": (s3 / nr)[:, w],
        "fisher": fi,
        "n_rows": float(nr),
    }


def main():
    print(f"[import] cartridges={os.path.dirname(cartridges.__file__)}", flush=True)
    t_start = time.time()
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
    T_tr = cache._num_trainable_tokens
    print(f"[cache] n_layers={n_layers} T_c={T_c} n_frozen={n_frozen} T_tr={T_tr}", flush=True)

    datasets = {}
    for spec in EVAL_SPECS.split(","):
        label, path = spec.split(":", 1)
        datasets[label] = LossEvalDataset(
            LossEvalDataset.Config(
                data_source=DataSource(path=path, type="local"), packed_seq_length=2048
            ),
            tokenizer=tokenizer,
            seed=0,
        )
        print(f"[eval] {label}: n_elements={len(datasets[label].elements)} "
              f"n_batches={len(datasets[label].batches)}", flush=True)

    # ---- pass A ----------------------------------------------------------------------
    A = {lab: collect_attention(model, cache, ds, n_layers, lab) for lab, ds in datasets.items()}

    # ---- pass B ----------------------------------------------------------------------
    cam = CacheAndModel(cache=cache, model=model)
    FI, fisher_secs, per_ex = {}, {}, {}
    for lab, ds in datasets.items():
        f, pl, pt, secs = collect_fisher(cam, cache, ds, n_layers, lab, max_batches=MAXB)
        FI[lab] = f
        fisher_secs[lab] = secs
        per_ex[lab] = {"loss": pl.tolist(), "tokens": pt.tolist(),
                       "micro_mean_ce": float(np.nansum(pl * pt) / max(pt.sum(), 1))}
        print(f"[fisher:{lab}] micro-mean CE from per-example buckets = "
              f"{per_ex[lab]['micro_mean_ce']:.6f}", flush=True)

    fp32_check = {"ran": False, "note": "set FP32_CHECK=1 to enable (recompiles flex_attention "
                                        "for fp32, costs several minutes)"}
    if FP32_CHECK:
        try:
            model.float()
            cache.float()
            f32, _, _, s32 = collect_fisher(cam, cache, datasets["QA"], n_layers, "QA-fp32",
                                            fp32=True, max_batches=1)
            keep = np.where(f32.sum(axis=(1, 2)) > 0)[0]
            fp32_check = {
                "ran": True,
                "n_elements": int(len(keep)),
                "spearman_bf16_vs_fp32_pooled": spearman(
                    FI["QA"][keep].mean(0), f32[keep].mean(0)
                ),
                "secs": s32,
                "note": "one packed QA batch recomputed with model+cache in float32 and autocast "
                        "off; the primary Fisher uses the eval harness's own bf16 autocast path.",
            }
            print(f"[fp32] {fp32_check}", flush=True)
        except Exception as exc:  # noqa: BLE001
            fp32_check = {"ran": False, "error": repr(exc)}
            print(f"[fp32] FAILED: {exc!r}", flush=True)
        finally:
            model.to(torch.bfloat16)
            cache.to(torch.bfloat16)

    del cam
    del model
    torch.cuda.empty_cache()

    # ---- redundancy ------------------------------------------------------------------
    RED = compute_redundancy(cache, n_layers, RIDGE_REL)
    print(f"[redundancy] mean={RED['redundancy'].mean():.4f} "
          f"range=[{RED['redundancy'].min():.4f},{RED['redundancy'].max():.4f}]", flush=True)

    # ---- assemble point-estimate metric table ---------------------------------------
    MQ = build_metrics(A["QA"], FI["QA"])
    MM = build_metrics(A["MT"], FI["MT"])
    eps = 1e-30
    M = {
        "tf_mass_qa": MQ["tf_mass"],
        "tf_mass_mt": MM["tf_mass"],
        "entropy": MQ["entropy"],
        "kl_loo": MQ["kl_loo"],
        "fisher": MQ["fisher"],
        "redundancy": RED["redundancy"],
        "contrast": np.log((MM["tf_mass"] + eps) / (MQ["tf_mass"] + eps)),
    }
    EXTRA = {
        "entropy_mt": MM["entropy"],
        "kl_loo_mt": MM["kl_loo"],
        "fisher_mt": MM["fisher"],
        "ranker_tf_qa": MQ["ranker_tf"],
        "ranker_tf_mt": MM["ranker_tf"],
        "w_mass_qa": MQ["w_mass"],
        "w_mass_mt": MM["w_mass"],
        "entropy_norm_qa": MQ["entropy_norm"],
        "value_norm": RED["value_norm"],
        "resid_norm": RED["resid_norm"],
    }
    names = list(M.keys())

    # ---- cross-validation against DIAG-ROUTING --------------------------------------
    ph_qa, ph_mt = A["QA"]["PH"], A["MT"]["PH"]          # (L, H, T_c) mean routing per head
    xval = {
        "per_head_top32_overlap_QA_MT_all512": float(
            set_overlap(topk_set(ph_qa, 32), topk_set(ph_mt, 32), 32).mean()
        ),
        "per_head_mean_routing_cosine_QA_MT": float(
            np.mean(
                (ph_qa * ph_mt).sum(-1)
                / np.maximum(np.linalg.norm(ph_qa, axis=-1) * np.linalg.norm(ph_mt, axis=-1), 1e-300)
            )
        ),
        "per_head_hist_intersection_QA_MT": float(np.minimum(ph_qa, ph_mt).sum(-1).mean()),
        "DIAG-ROUTING_reference": {"top32_overlap": 0.914, "cosine": 0.99899,
                                   "hist_intersection": 0.963},
    }
    print(f"[xval] {json.dumps({k: v for k, v in xval.items() if k != 'DIAG-ROUTING_reference'})}",
          flush=True)

    # ---- Spearman matrices -----------------------------------------------------------
    def corr_matrix(mats):
        P = np.zeros((len(names), len(names)))
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                P[i, j] = spearman(mats[ni], mats[nj])
        return P

    rho_pooled = corr_matrix(M)
    rho_per_layer = np.zeros((n_layers, len(names), len(names)))
    for l in range(n_layers):
        rho_per_layer[l] = corr_matrix({k: v[l] for k, v in M.items()})

    # ---- top-32 overlaps -------------------------------------------------------------
    # incumbent = the actual TF ranker. Two realisations:
    #   (i)  reproduced on the MT eval queries with the exact ranker convention
    #   (ii) the canonical run's own per-document selections (am_doc_*.pt)
    inc_eval = topk_set(EXTRA["ranker_tf_mt"], TOPK)            # (L, 32)
    sel_dir = {                    # direction that makes the metric a *selector*
        "tf_mass_qa": True, "tf_mass_mt": True, "entropy": False, "kl_loo": False,
        "fisher": False, "redundancy": True, "contrast": True,
    }
    safe_dir_doc = {
        "tf_mass_qa": "descending (slots QA attends to most)",
        "tf_mass_mt": "descending (slots MT attends to most = an attention-only selector)",
        "entropy": "ascending (narrowly-used slots first)",
        "kl_loo": "ascending (least irreplaceable first)",
        "fisher": "ascending (least QA-important first)",
        "redundancy": "descending (most linearly reconstructible first)",
        "contrast": "descending (slots MT wants more than QA)",
    }
    ovl_vs_incumbent, ovl_vs_incumbent_desc = {}, {}
    for nm in names:
        s_sel = topk_set(M[nm], TOPK, largest=sel_dir[nm])
        s_desc = topk_set(M[nm], TOPK, largest=True)
        ovl_vs_incumbent[nm] = set_overlap(inc_eval, s_sel, TOPK)
        ovl_vs_incumbent_desc[nm] = set_overlap(inc_eval, s_desc, TOPK)

    # pairwise (descending) top-32 overlaps between metrics
    desc_sets = {nm: topk_set(M[nm], TOPK, largest=True) for nm in names}
    ovl_matrix = np.zeros((len(names), len(names)))
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            ovl_matrix[i, j] = set_overlap(desc_sets[ni], desc_sets[nj], TOPK).mean()

    # incumbent (ii): the canonical run's own per-document top-32
    ovl_vs_actual = {}
    if SLOTS_RUN_DIR:
        per_doc = slot_union_from_run_dir(SLOTS_RUN_DIR)
        for nm in names:
            s_sel = topk_set(M[nm], TOPK, largest=sel_dir[nm])
            vals = []
            for doc in per_doc:
                v = [len(set(doc[l]) & set(s_sel[l].tolist())) / TOPK for l in range(n_layers)]
                vals.append(float(np.mean(v)))
            ovl_vs_actual[nm] = {"mean_over_16_docs": float(np.mean(vals)), "per_doc": vals}
        ovl_vs_actual["_incumbent_eval_reproduction"] = float(
            np.mean([
                np.mean([len(set(doc[l]) & set(inc_eval[l].tolist())) / TOPK
                         for l in range(n_layers)])
                for doc in per_doc
            ])
        )

    # ---- safe acquisition budget -----------------------------------------------------
    def safe_budget(mt_sel, imp, low_is_safe):
        """fraction of MT-wanted slots sitting in the safe tail of QA importance."""
        out = {}
        r = rankdata(imp if low_is_safe else -imp, axis=-1)     # rank 1 = safest
        n = imp.shape[-1]
        # the 32 MOST IMPORTANT slots under this metric: highest value when a high value
        # means "important" (low_is_safe), lowest value otherwise (e.g. redundancy).
        top_imp = topk_set(imp, TOPK, largest=low_is_safe)
        out["not_in_qa_top32"] = float(
            np.mean([1.0 - len(set(mt_sel[l].tolist()) & set(top_imp[l].tolist())) / TOPK
                     for l in range(n_layers)])
        )
        for q, tag in [(TOPK / n, "in_qa_safest_32"), (0.25, "in_qa_bottom25pct"),
                       (0.50, "in_qa_bottom50pct")]:
            thr = q * n
            out[tag] = float(np.mean([
                np.mean(r[l][mt_sel[l]] <= thr) for l in range(n_layers)
            ]))
        return out

    mt_sets = {
        "tf_mass_mt_top32": topk_set(M["tf_mass_mt"], TOPK),
        "ranker_tf_mt_top32": inc_eval,
    }
    qa_importance = {
        "tf_mass_qa": (M["tf_mass_qa"], True),
        "entropy": (M["entropy"], True),
        "kl_loo": (M["kl_loo"], True),
        "fisher": (M["fisher"], True),
        "redundancy": (M["redundancy"], False),   # high redundancy == safe
    }
    budgets = {
        mk: {ik: safe_budget(ms, im, lo) for ik, (im, lo) in qa_importance.items()}
        for mk, ms in mt_sets.items()
    }

    # ---- concentration ---------------------------------------------------------------
    def concentration(x, shift_min=True):
        y = x - x.min(axis=-1, keepdims=True) if shift_min else x.copy()
        y = np.maximum(y, 0.0)
        tot = y.sum(-1, keepdims=True)
        srt = -np.sort(-y, axis=-1)
        cs = np.cumsum(srt, axis=-1) / np.maximum(tot, 1e-300)
        pr = (tot[..., 0] ** 2) / np.maximum((y ** 2).sum(-1), 1e-300)  # participation ratio
        return {
            "frac_in_top32": [float(v) for v in cs[:, TOPK - 1]],
            "frac_in_top64": [float(v) for v in cs[:, 63]],
            "frac_in_top128": [float(v) for v in cs[:, 127]],
            "participation_ratio": [float(v) for v in pr],
            "mean_frac_in_top32": float(cs[:, TOPK - 1].mean()),
            "mean_frac_in_top64": float(cs[:, 63].mean()),
            "mean_frac_in_top128": float(cs[:, 127].mean()),
            "mean_participation_ratio": float(pr.mean()),
        }

    conc = {
        "fisher_qa": concentration(M["fisher"], shift_min=False),
        "tf_mass_qa": concentration(M["tf_mass_qa"], shift_min=False),
        "kl_loo_qa": concentration(M["kl_loo"], shift_min=False),
        "fisher_mt": concentration(EXTRA["fisher_mt"], shift_min=False),
        "entropy_qa_excess": concentration(M["entropy"], shift_min=True),
    }

    # ---- split-half reliability ------------------------------------------------------
    def split_half(A_, F_, nm_builder):
        n_el = A_["S1"].shape[0]
        perm = RNG.permutation(n_el)
        h1, h2 = perm[: n_el // 2], perm[n_el // 2:]
        m1 = nm_builder(A_, F_, h1)
        m2 = nm_builder(A_, F_, h2)
        return {k: spearman(m1[k], m2[k]) for k in ("tf_mass", "entropy", "kl_loo", "fisher")}

    reliab = {
        "QA": split_half(A["QA"], FI["QA"], lambda a, f, s: build_metrics(a, f, s)),
        "MT": split_half(A["MT"], FI["MT"], lambda a, f, s: build_metrics(a, f, s)),
    }
    print(f"[split-half] {json.dumps(reliab)}", flush=True)

    # ---- bootstrap over eval examples ------------------------------------------------
    nq, nm_ = A["QA"]["S1"].shape[0], A["MT"]["S1"].shape[0]
    boot_keys = [("tf_mass_qa", "fisher"), ("tf_mass_qa", "kl_loo"),
                 ("tf_mass_qa", "entropy"), ("tf_mass_qa", "redundancy"),
                 ("tf_mass_qa", "contrast"), ("tf_mass_qa", "tf_mass_mt"),
                 ("fisher", "kl_loo"), ("fisher", "redundancy"), ("fisher", "entropy")]
    boot_rho = {f"{a}|{b}": [] for a, b in boot_keys}
    boot_ovl = {nm: [] for nm in names}
    boot_budget = {ik: [] for ik in qa_importance}
    t0 = time.time()
    for b in range(NBOOT):
        sq = RNG.integers(0, nq, nq)
        sm = RNG.integers(0, nm_, nm_)
        bq = build_metrics(A["QA"], FI["QA"], sq)
        bm = build_metrics(A["MT"], FI["MT"], sm)
        Mb = {
            "tf_mass_qa": bq["tf_mass"], "tf_mass_mt": bm["tf_mass"],
            "entropy": bq["entropy"], "kl_loo": bq["kl_loo"], "fisher": bq["fisher"],
            "redundancy": M["redundancy"],
            "contrast": np.log((bm["tf_mass"] + eps) / (bq["tf_mass"] + eps)),
        }
        for a_, b_ in boot_keys:
            boot_rho[f"{a_}|{b_}"].append(spearman(Mb[a_], Mb[b_]))
        inc_b = topk_set(bm["ranker_tf"], TOPK)
        for nm in names:
            boot_ovl[nm].append(
                float(set_overlap(inc_b, topk_set(Mb[nm], TOPK, largest=sel_dir[nm]), TOPK).mean())
            )
        mtsel_b = topk_set(Mb["tf_mass_mt"], TOPK)
        for ik, (_, lo) in qa_importance.items():
            boot_budget[ik].append(
                safe_budget(mtsel_b, Mb[ik] if ik != "redundancy" else M["redundancy"], lo)
            )
        if (b + 1) % 100 == 0:
            print(f"[boot] {b+1}/{NBOOT} ({time.time()-t0:.1f}s)", flush=True)

    def ci(v):
        v = np.asarray(v, dtype=float)
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                "lo95": float(np.percentile(v, 2.5)), "hi95": float(np.percentile(v, 97.5))}

    boot = {
        "spearman": {k: ci(v) for k, v in boot_rho.items()},
        "top32_overlap_vs_incumbent": {k: ci(v) for k, v in boot_ovl.items()},
        "safe_budget_tf_mass_mt_top32": {
            ik: {tag: ci([d[tag] for d in lst]) for tag in lst[0]}
            for ik, lst in boot_budget.items()
        },
        "n_bootstrap": NBOOT,
        "resampling_unit": "eval example (QA n=%d, MT n=%d), splits resampled independently"
                           % (nq, nm_),
    }

    # ======================================================================= outputs ===
    def pl(x):
        return [float(v) for v in np.asarray(x)]

    headline = {
        "spearman_tf_mass_qa_vs_fisher_pooled": rho_pooled[names.index("tf_mass_qa"),
                                                           names.index("fisher")],
        "spearman_tf_mass_qa_vs_kl_loo_pooled": rho_pooled[names.index("tf_mass_qa"),
                                                           names.index("kl_loo")],
        "spearman_tf_mass_qa_vs_entropy_pooled": rho_pooled[names.index("tf_mass_qa"),
                                                            names.index("entropy")],
        "spearman_tf_mass_qa_vs_redundancy_pooled": rho_pooled[names.index("tf_mass_qa"),
                                                               names.index("redundancy")],
        "spearman_tf_mass_qa_vs_contrast_pooled": rho_pooled[names.index("tf_mass_qa"),
                                                             names.index("contrast")],
        "spearman_tf_mass_qa_vs_tf_mass_mt_pooled": rho_pooled[names.index("tf_mass_qa"),
                                                               names.index("tf_mass_mt")],
        "spearman_fisher_vs_kl_loo_pooled": rho_pooled[names.index("fisher"),
                                                       names.index("kl_loo")],
        "spearman_tf_mass_qa_vs_fisher_perlayer_mean": float(
            rho_per_layer[:, names.index("tf_mass_qa"), names.index("fisher")].mean()
        ),
        "spearman_tf_mass_qa_vs_kl_loo_perlayer_mean": float(
            rho_per_layer[:, names.index("tf_mass_qa"), names.index("kl_loo")].mean()
        ),
    }
    headline = {k: float(v) for k, v in headline.items()}

    out = {
        "id": "DIAG-IMPORTANCE",
        "board_entry": "B-GATE",
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
            "n_kv_heads": 8,
            "T_cartridge": T_c,
            "n_writable_slots": T_tr,
            "frozen_sink_slot": 0,
            "n_eval_examples": {k: int(A[k]["S1"].shape[0]) for k in A},
            "n_query_rows_per_layer": {k: float(A[k]["nrow"].sum()) for k in A},
            "routing_vector_definition":
                "a(q) = softmax(qK_cart^T/sqrt(d)) over the 512 cartridge slots, recovered as "
                "w[:, :T_c]/w[:, :T_c].sum() from the FULL eval-time softmax over "
                "[512 slots || own-sequence causal prefix]; == value_solve.py:86 X at top_t=512. "
                "Identical to DIAG-ROUTING.",
            "per_layer_aggregation":
                "scores are summed over the 8 KV heads AND the 4 query heads in each GQA group, "
                "matching the incumbent ranker's per_layer granularity",
            "ranker_score_definition":
                "softmax(mean_over_GQA_group(q) @ trainable_keys^T / sqrt(d)) summed over kv-heads "
                "and query positions -- the exact convention of query_accum.py:80-95 "
                "(group mean taken BEFORE the softmax; frozen slot 0 excluded)",
            "gradient_steps": 0,
            "no_source_file_edited": True,
            "no_cache_written": True,
        },
        "metric_names": names,
        "headline": headline,
        "spearman_pooled": {names[i]: {names[j]: float(rho_pooled[i, j]) for j in range(len(names))}
                            for i in range(len(names))},
        "spearman_per_layer_mean": {
            names[i]: {names[j]: float(rho_per_layer[:, i, j].mean()) for j in range(len(names))}
            for i in range(len(names))
        },
        "spearman_per_layer_sd": {
            names[i]: {names[j]: float(rho_per_layer[:, i, j].std(ddof=1))
                       for j in range(len(names))}
            for i in range(len(names))
        },
        "top32_overlap_vs_incumbent_eval_ranker": {
            nm: {"mean": float(ovl_vs_incumbent[nm].mean()),
                 "per_layer": pl(ovl_vs_incumbent[nm]),
                 "direction": safe_dir_doc[nm],
                 "mean_if_taken_descending": float(ovl_vs_incumbent_desc[nm].mean())}
            for nm in names
        },
        "top32_overlap_vs_actual_run_selections": ovl_vs_actual,
        "top32_overlap_matrix_descending": {
            names[i]: {names[j]: float(ovl_matrix[i, j]) for j in range(len(names))}
            for i in range(len(names))
        },
        "safe_acquisition_budget": budgets,
        "concentration": conc,
        "split_half_reliability_spearman": reliab,
        "bootstrap": boot,
        "cross_validation_vs_DIAG_ROUTING": xval,
        "fisher_cost": {
            "wall_clock_s": {k: float(v) for k, v in fisher_secs.items()},
            "total_s": float(sum(fisher_secs.values())),
            "n_backward_passes": {k: int(len(datasets[k].elements)) for k in datasets},
            "fp32_precision_control": fp32_check,
            "gradient_steps": 0,
        },
        "attention_pass_cost": {k: float(A[k]["secs"]) for k in A},
        "per_example_losses_from_fisher_pass": per_ex,
        "redundancy_ridge_sensitivity": {
            "spearman_default_vs_1e-4": spearman(RED["redundancy"], RED["redundancy_ridge1e-4"]),
            "spearman_default_vs_1e-8": spearman(RED["redundancy"], RED["redundancy_ridge1e-8"]),
            "mean_default": float(RED["redundancy"].mean()),
            "mean_1e-4": float(RED["redundancy_ridge1e-4"].mean()),
            "mean_1e-8": float(RED["redundancy_ridge1e-8"].mean()),
            "ridge_rel": RIDGE_REL,
        },
        "per_layer_means": {
            **{f"{nm}_mean": pl(M[nm].mean(-1)) for nm in names},
            **{f"{nm}_mean": pl(EXTRA[nm].mean(-1)) for nm in EXTRA},
            "spearman_tf_mass_qa_vs_fisher": pl(
                rho_per_layer[:, names.index("tf_mass_qa"), names.index("fisher")]
            ),
            "spearman_tf_mass_qa_vs_kl_loo": pl(
                rho_per_layer[:, names.index("tf_mass_qa"), names.index("kl_loo")]
            ),
            "spearman_tf_mass_qa_vs_redundancy": pl(
                rho_per_layer[:, names.index("tf_mass_qa"), names.index("redundancy")]
            ),
            "fisher_frac_in_top32": conc["fisher_qa"]["frac_in_top32"],
            "fisher_frac_in_top64": conc["fisher_qa"]["frac_in_top64"],
        },
        "wall_clock_total_s": time.time() - t_start,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(
        OUT_NPZ,
        metric_names=np.array(names),
        **{f"score_{nm}": M[nm].astype(np.float32) for nm in names},
        **{f"score_{nm}": EXTRA[nm].astype(np.float32) for nm in EXTRA},
        spearman_pooled=rho_pooled,
        spearman_per_layer=rho_per_layer.astype(np.float32),
        top32_overlap_matrix_descending=ovl_matrix,
        **{f"ovl_vs_incumbent_{nm}": ovl_vs_incumbent[nm].astype(np.float32) for nm in names},
        mean_routing_per_head_qa=A["QA"]["PH"].astype(np.float32),
        mean_routing_per_head_mt=A["MT"]["PH"].astype(np.float32),
        fisher_per_example_qa=FI["QA"].astype(np.float32),
        fisher_per_example_mt=FI["MT"].astype(np.float32),
        redundancy_ridge1e4=RED["redundancy_ridge1e-4"].astype(np.float32),
        redundancy_ridge1e8=RED["redundancy_ridge1e-8"].astype(np.float32),
    )
    print(f"[done] wrote {OUT_JSON} and {OUT_NPZ}", flush=True)
    for k, v in headline.items():
        print(f"  HEADLINE {k} = {v:.6g}", flush=True)

    # ------------------------------------------------------------------------- wandb --
    if os.environ.get("WANDB_DISABLED", "0") != "1":
        import wandb

        run = wandb.init(
            project=os.environ.get("CARTRIDGES_WANDB_PROJECT", "SEACrowd"),
            entity=os.environ.get("CARTRIDGES_WANDB_ENTITY", None),
            name=os.environ.get("RUN_NAME", "DIAG-IMPORTANCE_slot-metrics"),
            group=os.environ.get("WANDB_GROUP", "B-GATE"),
            tags=["diagnostic", "B-GATE", "DIAG-IMPORTANCE"],
            notes=os.environ.get("WANDB_NOTES", ""),
            config=out["provenance"],
        )
        summ = {f"diag/{k}": v for k, v in headline.items()}
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                if j > i:
                    summ[f"diag/rho_pooled/{ni}__{nj}"] = float(rho_pooled[i, j])
                    summ[f"diag/rho_perlayer_mean/{ni}__{nj}"] = float(
                        rho_per_layer[:, i, j].mean()
                    )
                    summ[f"diag/top32ovl/{ni}__{nj}"] = float(ovl_matrix[i, j])
        for nm in names:
            summ[f"diag/ovl_vs_incumbent/{nm}"] = float(ovl_vs_incumbent[nm].mean())
            if nm in ovl_vs_actual:
                summ[f"diag/ovl_vs_actual_run/{nm}"] = ovl_vs_actual[nm]["mean_over_16_docs"]
        for mk, d in budgets.items():
            for ik, dd in d.items():
                for tag, v in dd.items():
                    summ[f"diag/safebudget/{mk}/{ik}/{tag}"] = float(v)
        for ck, d in conc.items():
            for tag in ("mean_frac_in_top32", "mean_frac_in_top64", "mean_frac_in_top128",
                        "mean_participation_ratio"):
                summ[f"diag/concentration/{ck}/{tag}"] = float(d[tag])
        for k, v in boot["spearman"].items():
            for s, vv in v.items():
                summ[f"diag/boot_rho/{k.replace('|','__')}/{s}"] = vv
        for k, v in boot["top32_overlap_vs_incumbent"].items():
            for s, vv in v.items():
                summ[f"diag/boot_ovl/{k}/{s}"] = vv
        for ik, d in boot["safe_budget_tf_mass_mt_top32"].items():
            for tag, v in d.items():
                for s, vv in v.items():
                    summ[f"diag/boot_budget/{ik}/{tag}/{s}"] = vv
        for k, v in xval.items():
            if isinstance(v, float):
                summ[f"diag/xval/{k}"] = v
        for split, d in reliab.items():
            for k, v in d.items():
                summ[f"diag/splithalf/{split}/{k}"] = float(v)
        summ["diag/fisher_wall_clock_s"] = float(sum(fisher_secs.values()))
        summ["diag/gradient_steps"] = 0
        wandb.summary.update(summ)
        for l in range(n_layers):
            row = {"layer": l}
            for k, v in out["per_layer_means"].items():
                row[f"diag/per_layer/{k}"] = v[l]
            wandb.log(row, step=l)
        print(f"WANDB_RUN_URL={run.url}", flush=True)
        print(f"WANDB_RUN_ID={run.id}", flush=True)
        wandb.finish()


if __name__ == "__main__":
    main()
