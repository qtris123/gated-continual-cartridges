"""Key stage: which key vectors occupy the cartridge slots being written.

Holds the two selection rules (highest-attention / OMP), which ``initial/``
uses to build a cartridge from scratch, and the Phase-2 support rewrite, which
``continual/`` uses to swap document keys into the selected slots.
"""

from __future__ import annotations

import math
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from pydrantic import ObjectConfig

from cartridges.am.components.beta import nnls_projected_gradient
from cartridges.am.core import _attention_scores, _rope_reposition

# ---------------------------------------------------------------------------
# NNLS box + iteration budget for the Phase-1 key-selection beta fit.
#
# These now follow the AM reference implementation's own algorithm configs
# (`compaction/evaluation/configs/algorithms/`), which pair the box with the
# iteration count as a single recipe per selection mode:
#   highest_attention_keys.py -> nnls_iters=2, lower=exp(-3), upper=exp(3)
#   summarize_then_compact.py -> nnls_iters=0, upper=exp(7)   (OMP; no lower,
#                                which the reference `_nnls_pg` floors at 1e-12)
# Paper backing: App. C.2 "Stabilizing beta" / Algorithm 3.
#
# PREVIOUS VALUES used by every Phase-1 experiment in this tree before
# 2026-07-30 (kept here so old runs stay interpretable, and so a sweep can go
# back by passing them explicitly):
#   highest_attention: n_iters=200, lower=1e-12, upper=None
#   omp:               n_iters=200, lower=1e-12, upper=None
# With upper=None the fit is unbounded above and the lower bound of 1e-12 puts
# a killed key at beta = log(1e-12) = -27.6, i.e. the "beta ~= -inf, key can no
# longer contribute regardless of Cv" state App. C.2 introduces the box to
# prevent (LIT-002).
# ---------------------------------------------------------------------------
HA_NNLS_ITERS = 2
HA_W_LOWER = math.exp(-3.0)
HA_W_UPPER = math.exp(3.0)
OMP_NNLS_ITERS = 0
OMP_W_LOWER = 1e-12
OMP_W_UPPER = math.exp(7.0)


def select_keys_highest_attention(
    keys: torch.Tensor,
    queries: torch.Tensor,
    t: int,
    head_dim: int,
    score_method: Literal["rms", "mean", "max"] = "rms",
    doc_key_start: Optional[int] = None,
    doc_rope_offset: Optional[int] = None,
    rope_theta: float = 10000.0,
    nnls_iters: int = HA_NNLS_ITERS,
    w_lower: float = HA_W_LOWER,
    w_upper: Optional[float] = HA_W_UPPER,
    fit_beta: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """Select top-t keys by attention score (highest-attention-keys AM).

    Args:
        nnls_iters / w_lower / w_upper: the beta fit's PGD budget and weight box.
            Defaults are the reference config's ``nnls2_-3_3``; see the module
            header for the previous (unbounded, 200-iteration) values.

    Returns:
        C1: (t, d) selected keys
        beta: (t,) log-weights from NNLS on mass
        indices: list of selected key indices
    """
    scores = _attention_scores(
        queries,
        keys,
        head_dim,
        doc_key_start=doc_key_start,
        doc_rope_offset=doc_rope_offset,
        rope_theta=rope_theta,
    )
    attn = F.softmax(scores, dim=-1)  # (n, T)

    if score_method == "rms":
        key_scores = torch.sqrt((attn ** 2).mean(dim=0))
    elif score_method == "mean":
        key_scores = attn.mean(dim=0)
    else:
        key_scores = attn.max(dim=0).values

    T = keys.shape[0]
    t = min(t, T)
    _, top_idx = torch.topk(key_scores, k=t, largest=True)
    indices = top_idx.tolist()

    C1 = keys[indices]
    if not fit_beta:
        # Callers that zero or discard beta (phase 1 with beta off, key rewrite)
        # were paying a (n x t) least squares whose result was thrown away.
        return C1, torch.zeros(C1.shape[0], device=keys.device, dtype=torch.float32), indices

    # NNLS for beta (mass matching)
    exp_scores = torch.exp(scores - scores.max(dim=1, keepdim=True).values)
    target_mass = exp_scores.sum(dim=1)
    Phi = exp_scores[:, indices]
    w = nnls_projected_gradient(
        Phi,
        target_mass,
        n_iters=nnls_iters,
        lower_bound=w_lower,
        upper_bound=w_upper,
    )
    beta = torch.log(w.clamp(min=w_lower, max=w_upper))

    return C1, beta, indices


def select_keys_omp(
    keys: torch.Tensor,
    queries: torch.Tensor,
    t: int,
    head_dim: int,
    n_iters: int = OMP_NNLS_ITERS,
    doc_key_start: Optional[int] = None,
    doc_rope_offset: Optional[int] = None,
    rope_theta: float = 10000.0,
    w_lower: float = OMP_W_LOWER,
    w_upper: Optional[float] = OMP_W_UPPER,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Greedy OMP key selection on attention-mass features (AM paper Algorithm 1).

    Args:
        keys: (T, d) candidate keys
        queries: (n, d) reference queries
        t: number of keys to select
        n_iters / w_lower / w_upper: the inner NNLS budget and weight box, used
            both for the greedy residual solves and the final fit. Defaults are
            the reference config's ``nnls0_-inf_7``; ``n_iters=0`` means clamped
            least squares with no PGD refinement, so this also changes which
            keys the greedy loop picks. See the module header for the previous
            values.

    Returns:
        selected_keys: (t, d)
        beta: (t,) log-weights from NNLS mass fit
        indices: selected row indices into keys
    """
    scores = _attention_scores(
        queries,
        keys,
        head_dim,
        doc_key_start=doc_key_start,
        doc_rope_offset=doc_rope_offset,
        rope_theta=rope_theta,
    )
    phi = torch.exp(scores - scores.max(dim=1, keepdim=True).values)  # (n, T)
    target_mass = phi.sum(dim=1)

    T = keys.shape[0]
    t = min(t, T)
    selected: list[int] = []
    residual = target_mass.clone()

    for _ in range(t):
        if not selected:
            col_scores = (phi.T @ target_mass).abs()
        else:
            phi_s = phi[:, selected]
            w = nnls_projected_gradient(
                phi_s,
                target_mass,
                n_iters=n_iters,
                lower_bound=w_lower,
                upper_bound=w_upper,
            )
            residual = target_mass - phi_s @ w
            col_scores = (phi.T @ residual).abs()
            col_scores[selected] = -1.0

        next_idx = int(torch.argmax(col_scores).item())
        if next_idx in selected:
            remaining = [i for i in range(T) if i not in selected]
            if not remaining:
                break
            next_idx = remaining[0]
        selected.append(next_idx)

    phi_s = phi[:, selected]
    w = nnls_projected_gradient(
        phi_s,
        target_mass,
        n_iters=n_iters,
        lower_bound=w_lower,
        upper_bound=w_upper,
    )
    beta = torch.log(w.clamp(min=w_lower, max=w_upper))
    selected_keys = keys[selected]
    return selected_keys, beta, selected


def rewrite_keys_on_support(
    keys: torch.Tensor,
    selected_indices: torch.Tensor,
    candidate_keys: torch.Tensor,
    queries: torch.Tensor,
    mode: Literal["highest_attention", "omp"],
    head_dim: int,
    doc_key_start: Optional[int] = None,
    doc_rope_offset: Optional[int] = None,
    rope_theta: float = 10000.0,
    reposition: bool = False,
    doc_baked_pos_base: Optional[int] = None,
    info: Optional[dict] = None,
) -> torch.Tensor:
    """Replace cartridge key rows at selected_indices using teacher candidates.

    Args:
        keys: (T, d) current cartridge keys (single head)
        selected_indices: (t,) support indices in cartridge
        candidate_keys: (T_cand, d) pool, typically concat(cartridge, doc)
        queries: (n, d) reference queries
        reposition: **opt-in RoPE counter-rotation (B-ROPE hazard H2 / LIT-026).**
            The candidate pool is ``[cartridge_support || doc]`` and the two blocks
            are scored in *different* rotary frames: ``_attention_scores`` scores the
            cartridge block with the raw reference query, and the doc block with the
            query rotated FORWARD by ``doc_rope_offset = T_doc``.  A doc key that wins
            selection is then installed into a cartridge slot, where the student (and
            eval) will score it with the **raw** query -- so the logit it was chosen
            for is not the logit it delivers, and the mismatch is a rotation by
            thousands of positions.  With ``reposition=True`` every doc-sourced row is
            counter-rotated by ``-doc_rope_offset`` (``core._rope_reposition``, the
            same operator ``initial/compaction`` already uses) so that
            ``<q, k_installed> == <R_offset q, k_doc>`` exactly.  Cartridge-sourced
            rows are already in the student frame and are left untouched.
            Default ``False`` -> bit-identical to the historical behaviour.
        info: optional dict; filled in-place with rewrite diagnostics
            (``n_selected``, ``n_from_doc``, ``n_from_cartridge``, ``n_changed``,
            ``n_repositioned``, ``rope_delta``).

    Returns:
        new_keys: (T, d) with rows at selected_indices rewritten
    """
    device = keys.device
    dtype = keys.dtype
    selected_indices = selected_indices.to(device=device, dtype=torch.long)
    t = selected_indices.numel()
    if t == 0:
        return keys.clone()

    if doc_key_start is None and doc_rope_offset is not None:
        doc_key_start = candidate_keys.shape[0] - doc_rope_offset
    if mode == "highest_attention":
        new_k, _, sel_idx = select_keys_highest_attention(
            candidate_keys,
            queries,
            t,
            head_dim,
            score_method="rms",
            doc_key_start=doc_key_start,
            doc_rope_offset=doc_rope_offset,
            rope_theta=rope_theta,
            fit_beta=False,
        )
    elif mode == "omp":
        new_k, _, sel_idx = select_keys_omp(
            candidate_keys,
            queries,
            t,
            head_dim,
            doc_key_start=doc_key_start,
            doc_rope_offset=doc_rope_offset,
            rope_theta=rope_theta,
        )
    else:
        raise ValueError(f"Unsupported key rewrite mode: {mode}")

    sel_src = torch.as_tensor(sel_idx, device=device, dtype=torch.long)
    from_doc = (
        sel_src >= doc_key_start
        if doc_key_start is not None
        else torch.zeros_like(sel_src, dtype=torch.bool)
    )
    if info is not None:
        info["n_selected"] = int(t)
        info["n_from_doc"] = int(from_doc.sum().item())
        info["n_from_cartridge"] = int(t - from_doc.sum().item())
        info["n_repositioned"] = 0
        info["rope_delta"] = 0.0

    if (
        reposition
        and doc_rope_offset
        and doc_key_start is not None
        and bool(from_doc.any())
    ):
        # H2 fix. Doc row m was baked at absolute RoPE position m; the teacher scored
        # it against a query rotated forward by `doc_rope_offset`, i.e. it behaved as
        # a key at position `m - doc_rope_offset` in the student's (cartridge/eval)
        # frame. Re-base it to exactly that position -> a uniform phase shift of
        # `-doc_rope_offset`, the AM.pdf App. C.3 / StreamingLLM operator. Done in
        # float32 so the bf16 cache does not eat the rotation.
        rows = new_k[from_doc].to(torch.float32)
        m = (sel_src[from_doc] - doc_key_start).to(torch.float32)
        if doc_baked_pos_base is None:
            # Historical analytical frame: the key was treated as baked at m
            # and the teacher query was rotated forward by T_doc.
            to_pos = m - float(doc_rope_offset)
        else:
            # Keys were baked at ``doc_baked_pos_base + m``. A clean prefill
            # uses base 0. Land the key on the slot it is about to occupy,
            # the same rebase phase 1 uses.
            slot = selected_indices.to(device=device, dtype=torch.float32)
            if slot.shape[0] != new_k.shape[0]:
                raise RuntimeError(
                    "rewrite_keys_on_support: selected slots "
                    f"({slot.shape[0]}) != rewritten keys ({new_k.shape[0]})"
                )
            to_pos = slot[from_doc]
            m = m + float(doc_baked_pos_base)
        rebased = _rope_reposition(
            rows,
            from_pos=m,
            to_pos=to_pos,
            head_dim=head_dim,
            rope_theta=rope_theta,
        )
        if rebased.shape != rows.shape:
            raise RuntimeError(
                "rewrite_keys_on_support: _rope_reposition changed shape "
                f"{tuple(rows.shape)} -> {tuple(rebased.shape)}"
            )
        if not torch.isfinite(rebased).all():
            raise RuntimeError(
                "rewrite_keys_on_support: non-finite key after RoPE reposition "
                f"({int((~torch.isfinite(rebased)).sum().item())} of {rebased.numel()} "
                f"entries; doc_rope_offset={doc_rope_offset}, rope_theta={rope_theta})"
            )
        new_k = new_k.clone()
        new_k[from_doc] = rebased.to(new_k.dtype)
        if info is not None:
            info["n_repositioned"] = int(from_doc.sum().item())
            info["rope_delta"] = (
                None if doc_baked_pos_base is not None else -float(doc_rope_offset)
            )
            info["doc_baked_pos_base"] = doc_baked_pos_base

    out = keys.clone()
    out[selected_indices] = new_k.to(device=device, dtype=dtype)
    if info is not None:
        changed = (out[selected_indices] != keys[selected_indices]).any(dim=-1)
        info["n_changed"] = int(changed.sum().item())
    return out


class KeyWriter:
    """Installs teacher-sourced keys into the selected cartridge slots."""

    class Config(ObjectConfig):
        _pass_as_config = True

        key_mode: Literal["freeze", "highest_attention", "omp"] = "freeze"
        # MECH-005: counter-rotate a document key by `-doc_rope_offset` when it is
        # installed into a cartridge slot, so the logit that selected it is the
        # logit it delivers. Only meaningful with `key_mode != "freeze"`, and only
        # correct when `rope_theta` is the model's own base.
        key_reposition: bool = False

    def __init__(self, config: Config):
        self.config = config

    @property
    def enabled(self) -> bool:
        """Whether this run rewrites keys at all."""
        return self.config.key_mode != "freeze"

    def write(
        self,
        keys: torch.Tensor,
        selected_indices: torch.Tensor,
        candidate_keys: torch.Tensor,
        queries: torch.Tensor,
        *,
        head_dim: int,
        doc_key_start: Optional[int] = None,
        doc_rope_offset: Optional[int] = None,
        rope_theta: float,
        doc_baked_pos_base: Optional[int] = None,
        info: Optional[dict] = None,
    ) -> torch.Tensor:
        return rewrite_keys_on_support(
            keys,
            selected_indices,
            candidate_keys,
            queries,
            mode=self.config.key_mode,
            head_dim=head_dim,
            doc_key_start=doc_key_start,
            doc_rope_offset=doc_rope_offset,
            rope_theta=rope_theta,
            reposition=self.config.key_reposition,
            doc_baked_pos_base=doc_baked_pos_base,
            info=info,
        )
