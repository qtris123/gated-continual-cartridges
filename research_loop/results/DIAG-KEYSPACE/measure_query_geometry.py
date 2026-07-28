"""DIAG-KEYSPACE (B-ROUTE, key side): geometry of the 128-dim per-head QUERY space.

Standalone diagnostic — nothing under ``cartridges/`` is modified, no training, no
solve, no cache writes.  Forward passes only.  Companion to DIAG-ROUTING, which
measured the *post-softmax 512-dim routing simplex*; this measures the *pre-softmax
128-dim per-head query* geometry, i.e. the space a synthesised KEY lives in.

Measured on the Phase-1 cartridge over both eval splits, per (layer, KV-head):

 1. ``Q0 = E_q[q q^T]`` (uncentered second moment of the POST-RoPE queries, pooled over
    the ``g`` query heads that share a KV head, because a key is shared by the group).
    ``rho_key(r) = tr(P_perp_r Q0^MT) / tr(Q0^MT)`` with ``P_perp_r = I - U_r U_r^T`` and
    ``U_r`` the top-r eigenvectors of ``Q0^QA``.  Controls at the same r:
      - in-sample QA tail  ``tr(P_perp_r Q0^QA)/tr(Q0^QA)``  (QA's own spectral floor),
      - HELD-OUT split-half: ``U_r`` from ``Q0^{QA,A}`` applied to ``Q0^{QA,B}``, where the
        halves are disjoint *by eval element id* (not by token), plus ``rho_key`` measured
        against the same half-A basis so sample sizes match.
 2. eigenspectrum / effective rank of ``Q0^QA`` and ``Q0^MT``; principal angles between
    their top-r subspaces.
 3. ``||qbar_MT - qbar_QA|| / ||qbar_MT||`` and the QA/MT mean-query cosine.
 4. ACHIEVABLE SELECTIVITY BOUND.  For every sampled query row we also store the
    log-sum-exp ``lse(q)`` of *all* logits it already sees at eval time
    (512 cartridge slots || own-sequence causal prefix).  A hypothetical extra slot with
    post-RoPE key ``k`` then receives, exactly under the frozen-context assumption,
    ``mass(q;k) = sigmoid(q.k/sqrt(d) - lse(q))``; for m keys,
    ``mass = sigmoid(logsumexp_j(q.k_j/sqrt(d)) - lse(q))``.  We maximise
    ``log E_MT[mass] - log E_QA[mass]`` (and, separately, ``log E_MT[mass]``) over
    ``||k_j|| = c`` by projected Adam, sweeping ``c`` relative to the median incumbent
    cartridge key norm, and also evaluate fixed closed-form constructions
    (``qbar_MT``, ``qbar_MT - qbar_QA``, ``P_perp_r qbar_MT`` = LIT-020's null-space key).

Also re-measured for cross-validation with DIAG-ROUTING/ORACLE-WRITE: total cartridge
mass and mass on the tf-idf written-slot union, QA vs MT (the incumbent 1.04-1.05 ratio).

Usage (env): PHASE1_CACHE, EVAL_SPECS="QA:...,MT:...", SLOTS_RUN_DIR, OUT_JSON, OUT_NPZ.
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
N_SUB = int(os.environ.get("N_SUB", "2048"))       # sampled query rows per (layer, head)
SEED = int(os.environ.get("DIAG_SEED", "0"))
DEVICE = "cuda"

TAUS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
EPSS = [0.90, 0.95, 0.99, 0.999, 0.9999]
RDIMS = [1, 2, 4, 8, 16, 32, 64, 96, 112, 127]
ANGLE_RDIMS = [1, 2, 4, 8, 16, 32, 64]
CSCALES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]          # x median incumbent key norm
MKEYS = [1, 32]                                    # keys placed jointly
OPT_STEPS = int(os.environ.get("OPT_STEPS", "400"))


# --------------------------------------------------------------------------------------
# capture post-RoPE q/k exactly as DIAG-ROUTING / ORACLE-WRITE do
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
# collection
# --------------------------------------------------------------------------------------
@torch.no_grad()
def collect(model, cache, dataloader, n_layers, S32, label):
    """One forward pass over a split.  Returns per-(layer, KV-head) query second moments
    (all rows / split-half A / split-half B / scored rows), mean queries, per-slot masses,
    and a uniform reservoir subsample of raw (q, lse) rows for the selectivity bound."""
    captured, handles = install_qk_capture(model)
    n_frozen = cache._num_frozen_tokens
    T_c = cache.num_cartridge_tokens()
    rng = np.random.default_rng(SEED)

    Q0 = {k: None for k in ("all", "A", "B", "scored")}
    mean_q = {k: None for k in ("all", "A", "B", "scored")}
    n_rows = {k: 0 for k in ("all", "A", "B", "scored")}
    slot_mass = None      # (L, H, T_c) sum of attention mass per slot
    slot_mass_sc = None
    acc = None            # (L, H, K) scalar sums
    KEYS = ["mass_cart", "mass_S32", "mass_cart_sc", "mass_S32_sc", "qnorm2"]

    # reservoir buffers (row set is identical for every (layer, head))
    sub_q = None          # (L, H, N_SUB, d) float32
    sub_lse = None        # (L, H, N_SUB) float32
    sub_scored = np.zeros(N_SUB, dtype=bool)
    sub_half = np.zeros(N_SUB, dtype=np.int8)
    sub_group = np.zeros(N_SUB, dtype=np.int8)     # global document index mod 4
    n_seen = 0
    doc_offset = 0

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
            # GLOBAL document index (element ids can restart per packed batch), then
            # split-half / 4-way group by DOCUMENT, never by token
            uniq = sorted({int(x) for x in seq_ids.tolist()})
            docmap = {u: doc_offset + i for i, u in enumerate(uniq)}
            doc_of_pos = torch.tensor(
                [docmap[int(x)] for x in seq_ids.tolist()], device=DEVICE, dtype=torch.long
            )
            doc_offset += len(uniq)
            half_mask = (doc_of_pos % 2 == 0)       # True -> half A
            grp_mask = (doc_of_pos % 4)             # 0..3, document-disjoint groups

            # ---- reservoir selection for this batch, computed ONCE (row set is the same
            # for every (layer, head): row = group_idx * L + position) ------------------
            H0 = cache.config.n_heads
            g0 = captured[0][0].shape[1] // H0
            sm_row0 = scored_mask.unsqueeze(0).expand(g0, L).reshape(-1).cpu().numpy()
            hf_row0 = half_mask.unsqueeze(0).expand(g0, L).reshape(-1).cpu().numpy()
            gr_row0 = grp_mask.unsqueeze(0).expand(g0, L).reshape(-1).cpu().numpy()
            sel_by_chunk = {}
            for c0 in range(0, L, CHUNK):
                c1 = min(c0 + CHUNK, L)
                C = c1 - c0
                m = g0 * C
                # within-chunk row r <-> (gi, p) with r = gi*C + (p - c0)
                gi = np.arange(m) // C
                pp = c0 + (np.arange(m) % C)
                glob = gi * L + pp
                t_idx = np.arange(n_seen, n_seen + m)
                j = np.where(t_idx < N_SUB, t_idx, rng.integers(0, t_idx + 1))
                keep = j < N_SUB
                rows_k = np.nonzero(keep)[0]
                slots_k = j[keep]
                if rows_k.size:
                    _, last = np.unique(slots_k[::-1], return_index=True)
                    sel = rows_k.size - 1 - last
                    rows_k, slots_k = rows_k[sel], slots_k[sel]
                    sub_scored[slots_k] = sm_row0[glob[rows_k]]
                    sub_half[slots_k] = np.where(hf_row0[glob[rows_k]], 1, 2)
                    sub_group[slots_k] = gr_row0[glob[rows_k]]
                    sel_by_chunk[c0] = (
                        torch.as_tensor(rows_k, device=DEVICE, dtype=torch.long),
                        torch.as_tensor(slots_k, device=DEVICE, dtype=torch.long),
                    )
                else:
                    sel_by_chunk[c0] = None
                n_seen += m
                n_rows["all"] += m
                n_rows["A"] += int(hf_row0[glob].sum())
                n_rows["B"] += int((~hf_row0[glob]).sum())
                n_rows["scored"] += int(sm_row0[glob].sum())

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
                if Q0["all"] is None:
                    for kk in Q0:
                        Q0[kk] = torch.zeros(n_layers, H, d, d, dtype=torch.float64, device=DEVICE)
                        mean_q[kk] = torch.zeros(n_layers, H, d, dtype=torch.float64, device=DEVICE)
                    slot_mass = torch.zeros(n_layers, H, T_c, dtype=torch.float64, device=DEVICE)
                    slot_mass_sc = torch.zeros(n_layers, H, T_c, dtype=torch.float64, device=DEVICE)
                    acc = torch.zeros(n_layers, H, len(KEYS), dtype=torch.float64, device=DEVICE)
                    sub_q = torch.zeros(n_layers, H, N_SUB, d, dtype=torch.float32, device=DEVICE)
                    sub_lse = torch.zeros(n_layers, H, N_SUB, dtype=torch.float32, device=DEVICE)
                qh = q[0].reshape(H, g, L, d).float()
                ks = k_seq[0].float().contiguous()
                kc = k_cart.float().contiguous()
                Ssel = (S32[layer_idx].to(DEVICE) + n_frozen)

                # row bookkeeping: row r = (group gi, position p) -> gi*L + p
                sm_row = scored_mask.unsqueeze(0).expand(g, L).reshape(-1)          # (g*L,)
                hf_row = half_mask.unsqueeze(0).expand(g, L).reshape(-1)

                for c0 in range(0, L, CHUNK):
                    c1 = min(c0 + CHUNK, L)
                    C = c1 - c0
                    qc = qh[:, :, c0:c1, :].reshape(H, g * C, d)
                    s_cart = torch.bmm(qc, kc.transpose(1, 2)) * scaling      # (H, g*C, T_c)
                    s_seq = torch.bmm(qc, ks.transpose(1, 2)) * scaling       # (H, g*C, L)
                    vis = seq_vis[c0:c1].unsqueeze(0).expand(g, C, L).reshape(1, g * C, L)
                    s_seq = s_seq.masked_fill(~vis, float("-inf"))
                    s = torch.cat([s_cart, s_seq], dim=-1)
                    lse = torch.logsumexp(s, dim=-1)                          # (H, g*C)
                    w = torch.softmax(s, dim=-1)
                    del s, s_cart, s_seq
                    w_c = w[..., :T_c]
                    mass_cart = w_c.sum(-1)
                    mass_S32 = w[..., Ssel].sum(-1)
                    del w

                    sm = sm_row.reshape(g, L)[:, c0:c1].reshape(-1)           # (g*C,)
                    hf = hf_row.reshape(g, L)[:, c0:c1].reshape(-1)
                    smf = sm.to(qc.dtype)

                    qd = qc.double()
                    Q0["all"][layer_idx] += torch.bmm(qd.transpose(1, 2), qd)
                    mean_q["all"][layer_idx] += qd.sum(1)
                    for tag, msk in (("A", hf), ("B", ~hf), ("scored", sm)):
                        if msk.any():
                            qm = qc[:, msk, :].double()
                            Q0[tag][layer_idx] += torch.bmm(qm.transpose(1, 2), qm)
                            mean_q[tag][layer_idx] += qm.sum(1)
                    del qd

                    slot_mass[layer_idx] += w_c.sum(1).double()
                    slot_mass_sc[layer_idx] += (w_c * smf.unsqueeze(0).unsqueeze(-1)).sum(1).double()
                    acc[layer_idx] += torch.stack(
                        [
                            mass_cart.sum(1), mass_S32.sum(1),
                            (mass_cart * smf).sum(1), (mass_S32 * smf).sum(1),
                            (qc * qc).sum(-1).sum(1),
                        ],
                        dim=-1,
                    ).double()

                    if sel_by_chunk[c0] is not None:
                        rk, sk = sel_by_chunk[c0]
                        sub_q[layer_idx][:, sk, :] = qc[:, rk, :]
                        sub_lse[layer_idx][:, sk] = lse[:, rk].float()

                    del qc, w_c, mass_cart, mass_S32, lse
            cache.clear()
            n_batches += 1
            print(f"[collect:{label}] batch {n_batches} done ({time.time()-t0:.1f}s)", flush=True)
    finally:
        for h in handles:
            h.remove()

    print(
        f"[collect:{label}] {n_batches} batches, rows/head all={n_rows['all']} "
        f"A={n_rows['A']} B={n_rows['B']} scored={n_rows['scored']}, {time.time()-t0:.1f}s",
        flush=True,
    )
    stats = {
        "mass_cart": (acc[..., 0] / max(n_rows["all"], 1)).cpu().numpy(),
        "mass_S32": (acc[..., 1] / max(n_rows["all"], 1)).cpu().numpy(),
        "mass_cart_sc": (acc[..., 2] / max(n_rows["scored"], 1)).cpu().numpy(),
        "mass_S32_sc": (acc[..., 3] / max(n_rows["scored"], 1)).cpu().numpy(),
        "mean_qnorm2": (acc[..., 4] / max(n_rows["all"], 1)).cpu().numpy(),
    }
    out = {
        "Q0": {k: Q0[k] / max(n_rows[k], 1) for k in Q0},
        "mean_q": {k: mean_q[k] / max(n_rows[k], 1) for k in mean_q},
        "n_rows": n_rows,
        "stats": stats,
        "slot_mass": (slot_mass / max(n_rows["all"], 1)).cpu().numpy(),
        "slot_mass_sc": (slot_mass_sc / max(n_rows["scored"], 1)).cpu().numpy(),
        "sub_q": sub_q,
        "sub_lse": sub_lse,
        "sub_scored": sub_scored,
        "sub_half": sub_half,
        "n_batches": n_batches,
    }
    return out


# --------------------------------------------------------------------------------------
# spectral helpers (same conventions as DIAG-ROUTING)
# --------------------------------------------------------------------------------------
def eigh_desc(C: torch.Tensor):
    evals, U = torch.linalg.eigh(C)
    return evals.flip(-1).clamp_min(0.0), U.flip(-1)


def rank_metrics(evals: np.ndarray):
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
    out["lambda_max_over_trace"] = evals[..., 0] / np.maximum(tot[..., 0], 1e-300)
    out["lambda_max"] = evals[..., 0]
    out["trace"] = tot[..., 0]
    out["T"] = np.full(evals.shape[:-1], float(T))
    return out


def energy_along(U: torch.Tensor, C: torch.Tensor):
    """diag(U^T C U) — energy of C along each column of U.  U,(H,d,d); C,(H,d,d)."""
    return ((C @ U) * U).sum(1).clamp_min(0.0)          # (H, d)


def rho_from_energy(e: np.ndarray, r: int):
    """fraction of energy outside the first r directions of the basis e is expressed in."""
    tot = e.sum(-1)
    inside = e[..., :r].sum(-1)
    return 1.0 - inside / np.maximum(tot, 1e-300)


def principal_angles(Ua: torch.Tensor, Ub: torch.Tensor, r: int):
    """Ua, Ub: (H, d, d) eigenvector matrices (descending).  Returns cos of principal
    angles between the two top-r subspaces, (H, r)."""
    M = Ua[:, :, :r].transpose(1, 2) @ Ub[:, :, :r]     # (H, r, r)
    s = torch.linalg.svdvals(M).clamp(0.0, 1.0)
    return s


# --------------------------------------------------------------------------------------
# item 4: achievable selectivity bound
# --------------------------------------------------------------------------------------
def mass_stats(q, lse, k, d_sqrt, w_scored=None):
    """q (P,n,d) f32, lse (P,n), k (P,m,d) -> mean mass per population (P,)."""
    logit = torch.einsum("pnd,pmd->pmn", q, k) / d_sqrt
    agg = torch.logsumexp(logit, dim=1)                 # (P, n)
    mass = torch.sigmoid(agg - lse)
    if w_scored is None:
        return mass.mean(dim=1)
    w = w_scored / w_scored.sum().clamp_min(1.0)
    return (mass * w).sum(dim=1)


def optimise_key(qA, lA, qB, lB, c, m, d_sqrt, objective, init, steps=OPT_STEPS, lr=0.05):
    """Maximise (objective) over ||k_j|| = c.  qA/lA = MT (target), qB/lB = QA (avoid).
    objective: 'ratio' -> log E_MT[mass] - log E_QA[mass]; 'mass' -> log E_MT[mass]."""
    P, n, d = qA.shape
    u = init.clone().float()
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    u = u.reshape(P, m, d).contiguous().requires_grad_(True)
    opt = torch.optim.Adam([u], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        k = c.view(P, 1, 1) * u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        mMT = mass_stats(qA, lA, k, d_sqrt).clamp_min(1e-30)
        mQA = mass_stats(qB, lB, k, d_sqrt).clamp_min(1e-30)
        obj = torch.log(mMT) - torch.log(mQA) if objective == "ratio" else torch.log(mMT)
        (-obj.sum()).backward()
        opt.step()
    with torch.no_grad():
        k = c.view(P, 1, 1) * u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        mMT = mass_stats(qA, lA, k, d_sqrt)
        mQA = mass_stats(qB, lB, k, d_sqrt)
    return k.detach(), mMT, mQA


# --------------------------------------------------------------------------------------
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
    print(f"[cache] {PHASE1_CACHE} n_layers={n_layers} T_c={T_c} n_frozen={n_frozen}", flush=True)

    S32 = slot_union_from_run_dir(SLOTS_RUN_DIR) if SLOTS_RUN_DIR else {
        l: torch.arange(T_c - n_frozen) for l in range(n_layers)
    }

    # incumbent key-norm scale (post-RoPE cartridge keys), per (layer, KV-head)
    key_norm_med = np.zeros((n_layers, cache.config.n_heads))
    key_norm_max = np.zeros_like(key_norm_med)
    with torch.no_grad():
        for l in range(n_layers):
            kc = (
                torch.cat([cache.frozen_keys[l], cache.trainable_keys[l]], dim=2)[0]
                if n_frozen > 0 else cache.trainable_keys[l][0]
            ).detach().float()
            nrm = kc.norm(dim=-1)
            key_norm_med[l] = nrm.median(dim=-1).values.cpu().numpy()
            key_norm_max[l] = nrm.max(dim=-1).values.cpu().numpy()

    evals_dl = {}
    for spec in EVAL_SPECS.split(","):
        label, path = spec.split(":", 1)
        ds = LossEvalDataset.Config(
            data_source=DataSource(path=path, type="local"), packed_seq_length=2048
        ).instantiate(tokenizer=tokenizer, seed=0)
        evals_dl[label] = DataLoader(ds, batch_size=1, collate_fn=lambda b: b[0], num_workers=0)
        print(f"[eval] {label}: n={len(ds)} batches", flush=True)

    t_collect = time.time()
    res = {}
    for label, dl in evals_dl.items():
        res[label] = collect(model, cache, dl, n_layers, S32, label)
    collect_s = time.time() - t_collect

    del model
    torch.cuda.empty_cache()

    QA, MT = res["QA"], res["MT"]
    H = QA["Q0"]["all"].shape[1]
    d = QA["Q0"]["all"].shape[2]
    P = n_layers * H
    d_sqrt = float(d) ** 0.5

    # ---------------- item 1 + 2: spectra, rho_key, principal angles -------------------
    t_spec = time.time()
    ev_qa = np.zeros((n_layers, H, d))
    ev_mt = np.zeros((n_layers, H, d))
    ev_qaA = np.zeros((n_layers, H, d))
    e_mt_in_qa = np.zeros((n_layers, H, d))       # MT energy along QA(all) eigenbasis
    e_mt_in_qaA = np.zeros((n_layers, H, d))      # MT energy along QA(half A) eigenbasis
    e_qaB_in_qaA = np.zeros((n_layers, H, d))     # held-out QA energy along QA(half A) basis
    e_qa_in_mt = np.zeros((n_layers, H, d))       # reverse direction
    ang_cos = {r: np.zeros((n_layers, H, r)) for r in ANGLE_RDIMS}

    for l in range(n_layers):
        Cqa, Cmt = QA["Q0"]["all"][l], MT["Q0"]["all"][l]
        CqaA, CqaB = QA["Q0"]["A"][l], QA["Q0"]["B"][l]
        eq, Uq = eigh_desc(Cqa)
        em, Um = eigh_desc(Cmt)
        eqA, UqA = eigh_desc(CqaA)
        ev_qa[l] = eq.cpu().numpy()
        ev_mt[l] = em.cpu().numpy()
        ev_qaA[l] = eqA.cpu().numpy()
        e_mt_in_qa[l] = energy_along(Uq, Cmt).cpu().numpy()
        e_mt_in_qaA[l] = energy_along(UqA, Cmt).cpu().numpy()
        e_qaB_in_qaA[l] = energy_along(UqA, CqaB).cpu().numpy()
        e_qa_in_mt[l] = energy_along(Um, Cqa).cpu().numpy()
        for r in ANGLE_RDIMS:
            ang_cos[r][l] = principal_angles(Uq, Um, r).cpu().numpy()
        print(f"[spec] layer {l} done", flush=True)
    spec_s = time.time() - t_spec

    def rho_curve(e, ranks):
        return {f"r{r}": rho_from_energy(e, r) for r in ranks}

    rho_key = rho_curve(e_mt_in_qa, RDIMS)                 # PRIMARY: MT vs full-QA basis
    rho_qa_insample = rho_curve(ev_qa, RDIMS)              # QA's own spectral tail
    rho_key_A = rho_curve(e_mt_in_qaA, RDIMS)              # MT vs half-A basis
    rho_qa_heldout = rho_curve(e_qaB_in_qaA, RDIMS)        # CONTROL: QA-B vs half-A basis
    rho_qa_rev = rho_curve(e_qa_in_mt, RDIMS)              # QA vs MT basis

    # rho at per-head rank RULES
    rk_qa = rank_metrics(ev_qa)
    rk_mt = rank_metrics(ev_mt)
    rho_rule = {}
    for key, rr in rk_qa.items():
        if not (key.startswith("r_tau") or key.startswith("r_energy")):
            continue
        mt_v = np.zeros((n_layers, H))
        qa_v = np.zeros((n_layers, H))
        ho_v = np.zeros((n_layers, H))
        for l in range(n_layers):
            for h in range(H):
                r = int(rr[l, h])
                mt_v[l, h] = rho_from_energy(e_mt_in_qa[l, h], r)
                qa_v[l, h] = rho_from_energy(ev_qa[l, h], r)
                ho_v[l, h] = rho_from_energy(e_qaB_in_qaA[l, h], r)
        rho_rule[key] = {"mt": mt_v, "qa_insample": qa_v, "qa_heldout": ho_v}

    # energy-weighted pooled rho (well-defined across heads, unlike averaging Grams)
    def pooled_rho(e, ranks):
        tot = e.sum(-1)
        return {
            f"r{r}": float(1.0 - e[..., :r].sum(-1).sum() / max(tot.sum(), 1e-300)) for r in ranks
        }

    # ---------------- item 3: mean-query separation -------------------------------------
    mq_qa = QA["mean_q"]["all"].cpu().numpy()      # (L, H, d)
    mq_mt = MT["mean_q"]["all"].cpu().numpy()
    mq_qaA = QA["mean_q"]["A"].cpu().numpy()
    mq_qaB = QA["mean_q"]["B"].cpu().numpy()
    nrm = lambda x: np.linalg.norm(x, axis=-1)
    sep_rel = nrm(mq_mt - mq_qa) / np.maximum(nrm(mq_mt), 1e-300)
    sep_rel_ctrl = nrm(mq_qaB - mq_qaA) / np.maximum(nrm(mq_qaB), 1e-300)
    cos_mean = (mq_qa * mq_mt).sum(-1) / np.maximum(nrm(mq_qa) * nrm(mq_mt), 1e-300)
    cos_mean_ctrl = (mq_qaA * mq_qaB).sum(-1) / np.maximum(nrm(mq_qaA) * nrm(mq_qaB), 1e-300)
    # per-unit-norm logit gap: a unit key sees mean logits differing by ||dmu||/sqrt(d)
    dmu_logit_per_unit = nrm(mq_mt - mq_qa) / d_sqrt

    # ---------------- item 4: achievable selectivity bound ------------------------------
    t_opt = time.time()
    qMT = MT["sub_q"].reshape(P, N_SUB, d)
    lMT = MT["sub_lse"].reshape(P, N_SUB)
    qQA = QA["sub_q"].reshape(P, N_SUB, d)
    lQA = QA["sub_lse"].reshape(P, N_SUB)
    c_med = torch.as_tensor(key_norm_med.reshape(P), device=DEVICE, dtype=torch.float32)

    dmu = torch.as_tensor((mq_mt - mq_qa).reshape(P, d), device=DEVICE, dtype=torch.float32)
    qbar_mt = torch.as_tensor(mq_mt.reshape(P, d), device=DEVICE, dtype=torch.float32)
    # LIT-020 construction: P_perp_r qbar_MT with U_r from Q0^QA
    U_qa = torch.zeros(P, d, d, device=DEVICE)
    for l in range(n_layers):
        _, Uq = eigh_desc(QA["Q0"]["all"][l])
        U_qa[l * H:(l + 1) * H] = Uq.float()
    nullspace_dirs = {}
    for r in (4, 8, 16, 32, 64):
        Ur = U_qa[:, :, :r]
        proj = torch.einsum("pdr,pd->pr", Ur, qbar_mt)
        nullspace_dirs[r] = qbar_mt - torch.einsum("pdr,pr->pd", Ur, proj)

    bound = {"c_scales": CSCALES, "m_keys": MKEYS, "results": {}}
    bound_ph = {}       # per-head arrays for the npz
    for m in MKEYS:
        for cs in CSCALES:
            c = c_med * cs
            tag = f"m{m}_c{cs:g}"
            entry = {}
            # fixed constructions
            fixed = {
                "qbar_MT": qbar_mt,
                "dmu": dmu,
                **{f"nullspace_r{r}": v for r, v in nullspace_dirs.items()},
            }
            for name, v in fixed.items():
                vv = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                k = (c.view(P, 1, 1) * vv.view(P, 1, d)).expand(P, m, d).contiguous()
                if m > 1:   # m identical keys == one key with logit + log m; keep as-is
                    k = k.clone()
                with torch.no_grad():
                    mMT = mass_stats(qMT, lMT, k, d_sqrt)
                    mQA = mass_stats(qQA, lQA, k, d_sqrt)
                entry[name] = {
                    "mass_MT": float(mMT.mean()), "mass_QA": float(mQA.mean()),
                    "ratio_meanhead": float((mMT / mQA.clamp_min(1e-30)).mean()),
                    "ratio_pooled": float(mMT.mean() / mQA.mean().clamp_min(1e-30)),
                }
                bound_ph[f"{tag}_{name}_massMT"] = mMT.cpu().numpy().reshape(n_layers, H)
                bound_ph[f"{tag}_{name}_massQA"] = mQA.cpu().numpy().reshape(n_layers, H)
            # optimised (multi-start: dmu / qbar_MT / null-space r=32; per-head best)
            gen = torch.Generator(device=DEVICE).manual_seed(SEED)
            for obj in ("ratio", "mass"):
                best = None
                for v0 in (dmu, qbar_mt, nullspace_dirs[32]):
                    init = v0.view(P, 1, d).expand(P, m, d).contiguous().clone()
                    if m > 1:
                        init = init + 0.3 * init.norm(dim=-1, keepdim=True) * torch.randn(
                            P, m, d, device=DEVICE, generator=gen) / (d ** 0.5)
                    _k, mMT, mQA = optimise_key(qMT, lMT, qQA, lQA, c, m, d_sqrt, obj, init)
                    val = (torch.log(mMT.clamp_min(1e-30)) - torch.log(mQA.clamp_min(1e-30))
                           if obj == "ratio" else torch.log(mMT.clamp_min(1e-30)))
                    if best is None:
                        best = (val, mMT, mQA)
                    else:
                        take = val > best[0]
                        best = (torch.where(take, val, best[0]),
                                torch.where(take, mMT, best[1]),
                                torch.where(take, mQA, best[2]))
                _val, mMT, mQA = best
                entry[f"opt_{obj}"] = {
                    "mass_MT": float(mMT.mean()), "mass_QA": float(mQA.mean()),
                    "ratio_meanhead": float((mMT / mQA.clamp_min(1e-30)).mean()),
                    "ratio_pooled": float(mMT.mean() / mQA.mean().clamp_min(1e-30)),
                    "per_head_ratio_max": float((mMT / mQA.clamp_min(1e-30)).max()),
                    "per_head_ratio_median": float((mMT / mQA.clamp_min(1e-30)).median()),
                    "per_head_mass_MT_median": float(mMT.median()),
                }
                bound_ph[f"{tag}_opt_{obj}_massMT"] = mMT.cpu().numpy().reshape(n_layers, H)
                bound_ph[f"{tag}_opt_{obj}_massQA"] = mQA.cpu().numpy().reshape(n_layers, H)
            bound["results"][tag] = entry
            print(f"[bound] {tag}: opt_ratio={entry['opt_ratio']['ratio_pooled']:.4f} "
                  f"massMT={entry['opt_ratio']['mass_MT']:.4f} | "
                  f"opt_mass massMT={entry['opt_mass']['mass_MT']:.4f} "
                  f"ratio={entry['opt_mass']['ratio_pooled']:.4f}", flush=True)
    opt_s = time.time() - t_opt

    # incumbent per-slot selectivity: what the existing 512 keys already achieve
    sm_qa = QA["slot_mass"]      # (L, H, T_c)
    sm_mt = MT["slot_mass"]
    slot_ratio = sm_mt / np.maximum(sm_qa, 1e-30)
    # restrict to slots carrying non-trivial MT mass (>= 1e-4) so the ratio is meaningful
    valid = sm_mt >= 1e-4
    inc_best = np.array([
        np.max(slot_ratio[l, h][valid[l, h]]) if valid[l, h].any() else np.nan
        for l in range(n_layers) for h in range(H)
    ]).reshape(n_layers, H)
    inc_top1mass = np.array([
        slot_ratio[l, h][np.argmax(sm_mt[l, h])] for l in range(n_layers) for h in range(H)
    ]).reshape(n_layers, H)

    st_qa, st_mt = QA["stats"], MT["stats"]

    def pl(x):
        return [float(v) for v in np.asarray(x).mean(axis=-1)]

    summary = {
        # ---- item 1
        **{f"rho_key_{k}": float(np.mean(v)) for k, v in rho_key.items()},
        **{f"rho_qa_insample_{k}": float(np.mean(v)) for k, v in rho_qa_insample.items()},
        **{f"rho_qa_heldout_{k}": float(np.mean(v)) for k, v in rho_qa_heldout.items()},
        **{f"rho_key_vs_halfA_{k}": float(np.mean(v)) for k, v in rho_key_A.items()},
        **{f"rho_qa_rev_{k}": float(np.mean(v)) for k, v in rho_qa_rev.items()},
        **{f"rho_key_rule_{k}": float(np.mean(v["mt"])) for k, v in rho_rule.items()},
        **{f"rho_qa_heldout_rule_{k}": float(np.mean(v["qa_heldout"])) for k, v in rho_rule.items()},
        # ---- item 2
        **{f"rank_qa_{k}": float(np.mean(v)) for k, v in rk_qa.items()},
        **{f"rank_mt_{k}": float(np.mean(v)) for k, v in rk_mt.items()},
        **{f"princ_angle_deg_max_r{r}": float(np.mean(np.degrees(np.arccos(
            np.clip(ang_cos[r].min(-1), -1, 1))))) for r in ANGLE_RDIMS},
        **{f"princ_angle_deg_mean_r{r}": float(np.mean(np.degrees(np.arccos(
            np.clip(ang_cos[r], -1, 1))))) for r in ANGLE_RDIMS},
        # ---- item 3
        "mean_query_sep_rel": float(np.mean(sep_rel)),
        "mean_query_sep_rel_median": float(np.median(sep_rel)),
        "mean_query_sep_rel_qa_control": float(np.mean(sep_rel_ctrl)),
        "mean_query_cosine_QA_MT": float(np.mean(cos_mean)),
        "mean_query_cosine_QA_control": float(np.mean(cos_mean_ctrl)),
        "dmu_logit_per_unit_key": float(np.mean(dmu_logit_per_unit)),
        "incumbent_key_norm_median": float(np.mean(key_norm_med)),
        "dmu_logit_at_median_keynorm": float(np.mean(dmu_logit_per_unit * key_norm_med)),
        # ---- cross-validation with DIAG-ROUTING / ORACLE-WRITE
        "mass_cart_QA": float(np.mean(st_qa["mass_cart"])),
        "mass_cart_MT": float(np.mean(st_mt["mass_cart"])),
        "mass_S32union_QA": float(np.mean(st_qa["mass_S32"])),
        "mass_S32union_MT": float(np.mean(st_mt["mass_S32"])),
        "mt_over_qa_mass_ratio_S32union": float(
            np.mean(st_mt["mass_S32"]) / max(np.mean(st_qa["mass_S32"]), 1e-30)),
        "mt_over_qa_mass_ratio_cart": float(
            np.mean(st_mt["mass_cart"]) / max(np.mean(st_qa["mass_cart"]), 1e-30)),
        "incumbent_best_slot_ratio_mean": float(np.nanmean(inc_best)),
        "incumbent_top1mass_slot_ratio_mean": float(np.nanmean(inc_top1mass)),
        "mean_qnorm_QA": float(np.mean(np.sqrt(st_qa["mean_qnorm2"]))),
        "mean_qnorm_MT": float(np.mean(np.sqrt(st_mt["mean_qnorm2"]))),
    }

    out = {
        "id": "DIAG-KEYSPACE",
        "provenance": {
            "snapshot_path": os.environ.get("AMSNAP", ""),
            "cartridges_import_path": os.path.dirname(cartridges.__file__),
            "git_head_at_launch": os.environ.get("SNAP_GIT_HEAD", ""),
            "snapshot_manifest_sha256_pre": os.environ.get("SNAP_SHA", ""),
            "model": MODEL_NAME,
            "cache": PHASE1_CACHE,
            "eval_specs": EVAL_SPECS,
            "slots_run_dir": SLOTS_RUN_DIR,
            "n_layers": n_layers, "n_kv_heads": int(H), "head_dim": int(d),
            "T_cartridge": int(T_c), "n_frozen": int(n_frozen),
            "n_query_rows_per_head": {"QA": QA["n_rows"], "MT": MT["n_rows"]},
            "n_subsampled_rows_per_head": N_SUB,
            "seed": SEED,
            "query_definition": (
                "post-RoPE q after q_norm, captured at every attention layer, pooled over the "
                "g=n_q_heads/n_kv_heads query heads that share a KV head (a key is shared by "
                "the whole GQA group). Q0 = (1/N) sum_q q q^T, UNCENTERED (LIT-020)."
            ),
            "splithalf_definition": "QA rows split by eval element id parity (disjoint documents)",
            "bound_definition": (
                "one extra cartridge slot with post-RoPE key k receives, holding all other "
                "logits fixed, mass(q;k)=sigmoid(q.k/sqrt(d)-lse(q)) where lse(q) is the "
                "logsumexp over the 512 existing cartridge slots and the visible causal prefix; "
                "for m keys, mass=sigmoid(logsumexp_j(q.k_j/sqrt(d))-lse(q))."
            ),
            "timings_s": {"collect": collect_s, "spectra": spec_s, "bound_opt": opt_s,
                          "total": time.time() - t_start},
        },
        "thresholds": {"taus": TAUS, "energy_eps": EPSS, "rdims": RDIMS,
                       "angle_rdims": ANGLE_RDIMS, "c_scales": CSCALES, "m_keys": MKEYS},
        "pooled_energy_weighted": {
            "rho_key": pooled_rho(e_mt_in_qa, RDIMS),
            "rho_qa_insample": pooled_rho(ev_qa, RDIMS),
            "rho_qa_heldout": pooled_rho(e_qaB_in_qaA, RDIMS),
            "rho_key_vs_halfA": pooled_rho(e_mt_in_qaA, RDIMS),
        },
        "per_layer": {
            "rho_key": {k: pl(v) for k, v in rho_key.items()},
            "rho_qa_insample": {k: pl(v) for k, v in rho_qa_insample.items()},
            "rho_qa_heldout": {k: pl(v) for k, v in rho_qa_heldout.items()},
            "rank_qa": {k: pl(v) for k, v in rk_qa.items()},
            "rank_mt": {k: pl(v) for k, v in rk_mt.items()},
            "mean_query_sep_rel": pl(sep_rel),
            "mean_query_sep_rel_qa_control": pl(sep_rel_ctrl),
            "mean_query_cosine_QA_MT": pl(cos_mean),
            "princ_angle_deg_max": {f"r{r}": pl(np.degrees(np.arccos(
                np.clip(ang_cos[r].min(-1), -1, 1)))) for r in ANGLE_RDIMS},
            "mass_cart_QA": pl(st_qa["mass_cart"]),
            "mass_cart_MT": pl(st_mt["mass_cart"]),
            "mass_S32union_QA": pl(st_qa["mass_S32"]),
            "mass_S32union_MT": pl(st_mt["mass_S32"]),
            "incumbent_key_norm_median": pl(key_norm_med),
            "incumbent_best_slot_ratio": pl(inc_best),
        },
        "selectivity_bound": bound,
        "summary": {"mean_over_heads": summary},
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(
        OUT_NPZ,
        evals_qa=ev_qa.astype(np.float32),
        evals_mt=ev_mt.astype(np.float32),
        evals_qa_halfA=ev_qaA.astype(np.float32),
        e_mt_in_qa_basis=e_mt_in_qa.astype(np.float32),
        e_mt_in_qaA_basis=e_mt_in_qaA.astype(np.float32),
        e_qaB_in_qaA_basis=e_qaB_in_qaA.astype(np.float32),
        e_qa_in_mt_basis=e_qa_in_mt.astype(np.float32),
        mean_q_qa=mq_qa.astype(np.float32),
        mean_q_mt=mq_mt.astype(np.float32),
        mean_q_qa_A=mq_qaA.astype(np.float32),
        mean_q_qa_B=mq_qaB.astype(np.float32),
        mean_query_sep_rel=sep_rel.astype(np.float32),
        mean_query_sep_rel_qa_control=sep_rel_ctrl.astype(np.float32),
        key_norm_median=key_norm_med.astype(np.float32),
        key_norm_max=key_norm_max.astype(np.float32),
        slot_mass_qa=sm_qa.astype(np.float32),
        slot_mass_mt=sm_mt.astype(np.float32),
        incumbent_best_slot_ratio=inc_best.astype(np.float32),
        **{f"rho_key_{k}": v.astype(np.float32) for k, v in rho_key.items()},
        **{f"rho_qa_insample_{k}": v.astype(np.float32) for k, v in rho_qa_insample.items()},
        **{f"rho_qa_heldout_{k}": v.astype(np.float32) for k, v in rho_qa_heldout.items()},
        **{f"princ_cos_r{r}": v.astype(np.float32) for r, v in ang_cos.items()},
        **{f"bound_{k}": v.astype(np.float32) for k, v in bound_ph.items()},
    )
    print(f"[done] wrote {OUT_JSON} and {OUT_NPZ}", flush=True)
    for k in sorted(summary):
        print(f"  SUMMARY {k} = {summary[k]:.6g}", flush=True)

    # ------------------------------------------------------------------ wandb
    if os.environ.get("WANDB_DISABLED", "0") != "1":
        import wandb

        run = wandb.init(
            project=os.environ.get("CARTRIDGES_WANDB_PROJECT", "SEACrowd"),
            entity=os.environ.get("CARTRIDGES_WANDB_ENTITY", None),
            name=os.environ.get("RUN_NAME", "DIAG-KEYSPACE_query-geometry"),
            group=os.environ.get("WANDB_GROUP", "B-ROUTE"),
            tags=["diagnostic", "B-ROUTE", "DIAG-KEYSPACE", "keyspace"],
            notes=os.environ.get("WANDB_NOTES", ""),
            config=out["provenance"] | {"thresholds": out["thresholds"]},
        )
        wandb.summary.update({f"diag/{k}": v for k, v in summary.items()})
        for k, v in out["pooled_energy_weighted"].items():
            wandb.summary.update({f"diag/pooled_{k}_{k2}": v2 for k2, v2 in v.items()})
        for tag, entry in bound["results"].items():
            for name, vals in entry.items():
                for stat, x in vals.items():
                    wandb.summary.update({f"bound/{tag}/{name}/{stat}": x})
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
