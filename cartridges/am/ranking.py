"""Slot ranking strategies (TF-IDF / attention-mass / residual-budget).

MECH-008 / B-GATE adds three *information-theoretic* selections on top of the
attention-mass family. They exist because DIAG-IMPORTANCE measured that the
incumbent ranker's premise is false: Spearman(`tf_mass_qa`, `fisher`) = 0.579
pooled (0.666 +- 0.122 per layer), so attention mass does **not** capture which
slots carry Phase-1 (QA) knowledge.

    redundancy         -- how well slot j's value vector is linearly
                          reconstructed from the OTHER slots. High = safe to
                          overwrite. Gradient-free, data-free: one 512x512
                          inverse per layer. Best gradient-free proxy for Fisher
                          (rho = -0.648).
    fisher             -- diagonal empirical Fisher of the QA loss w.r.t. the
                          slot's value. Lowest = safest. Requires a *diagnostic*
                          backward pass over QA data, which is NOT an optimizer
                          step (`gradient_steps` stays 0) but is a real cost, so
                          the scores are computed once offline and loaded from
                          disk here.
    mass_x_redundancy  -- slots the new task WANTS *and* that are redundant.

NOT implemented on purpose: `kl_loo`. DIAG-IMPORTANCE showed the exact
leave-one-out KL of dropping slot j is `-log(1 - w_j)`, a strictly monotone
function of the slot's own attention weight, so ranking by LOO-KL *is* ranking
by attention mass (measured rho = 0.968). Use `attention_mass`; do not rebuild
it as a separate mode.
"""

from __future__ import annotations

import os
from logging import getLogger
from typing import TYPE_CHECKING, Optional

import torch

from cartridges.sparse_cache_finetuning import (
    CacheTFIDFRanker,
    GradientMask,
    TFIDFRankingInfo,
)

if TYPE_CHECKING:
    from cartridges.am.finetune import AttentionMatchingFinetuningConfig

logger = getLogger(__name__)

# The MECH-008 selections. Each needs the *cache* (redundancy) and/or a cached
# score array on disk (fisher), neither of which the attention-mass family uses.
SLOT_PRIOR_SELECTIONS = ("redundancy", "fisher", "mass_x_redundancy")


def _rank_attention_mass_per_layer(
    access_scores: torch.Tensor,
    top_t: int,
) -> tuple[GradientMask, TFIDFRankingInfo]:
    scores = access_scores.float().cpu()
    n_layers, n_tokens = scores.shape
    k = min(top_t, n_tokens)
    totals = scores.sum(-1, keepdim=True).clamp(min=1e-12)
    tf = scores / totals
    _, top = torch.topk(tf, k=k, dim=-1, largest=True)
    mask = GradientMask(
        granularity="per_layer",
        positions_per_layer={l: top[l] for l in range(n_layers)},
        n_tokens=n_tokens,
        top_t=k,
    )
    info = TFIDFRankingInfo(step=-1, tf=tf, tfidf=tf, mask=mask)
    return mask, info


def _rank_residual_budget_per_layer(
    access_scores: torch.Tensor,
    top_t: int,
    old_access_scores: Optional[torch.Tensor] = None,
    idf_scores: Optional[torch.Tensor] = None,
    idf_prior_weight: float = 0.0,
    min_top_t_per_layer: int = 1,
) -> tuple[GradientMask, TFIDFRankingInfo]:
    """Allocate layer budgets from access pressure and old/new conflict.

    This is the first AM-native budget proxy. Until a non-self target supplies
    true AM residuals, access conflict is used as a cheap residual-pressure
    surrogate.
    """
    scores = access_scores.float().cpu()
    n_layers, n_tokens = scores.shape
    totals = scores.sum(-1, keepdim=True).clamp(min=1e-12)
    tf = scores / totals

    if old_access_scores is not None:
        old = old_access_scores.float().cpu()
        old_tf = old / old.sum(-1, keepdim=True).clamp(min=1e-12)
        conflict = (tf - old_tf).abs().sum(-1)
        layer_pressure = conflict.clamp(min=1e-6)
    else:
        layer_pressure = tf.std(-1).clamp(min=1e-6)

    total_budget = min(top_t * n_layers, n_tokens * n_layers)
    raw_budget = layer_pressure / layer_pressure.sum() * total_budget
    budgets = raw_budget.round().long().clamp(min=min_top_t_per_layer, max=n_tokens)

    # Keep total budget near top_t * n_layers after rounding/clamping.
    while budgets.sum().item() > total_budget:
        idx = torch.argmax(budgets.float()).item()
        if budgets[idx] <= min_top_t_per_layer:
            break
        budgets[idx] -= 1
    while budgets.sum().item() < total_budget:
        idx = torch.argmax(layer_pressure).item()
        if budgets[idx] >= n_tokens:
            break
        budgets[idx] += 1

    score = tf
    if idf_scores is not None and idf_prior_weight > 0:
        idf = idf_scores.float().cpu()
        score = score + idf_prior_weight * (idf / idf.max().clamp(min=1e-12))

    positions = {}
    for layer_idx in range(n_layers):
        k = int(min(max(budgets[layer_idx].item(), 1), n_tokens))
        _, top = torch.topk(score[layer_idx], k=k, largest=True)
        positions[layer_idx] = top

    mask = GradientMask(
        granularity="per_layer",
        positions_per_layer=positions,
        n_tokens=n_tokens,
        top_t=top_t,
    )
    info = TFIDFRankingInfo(step=-1, tf=tf, tfidf=score, mask=mask)
    return mask, info


# ======================================================================================
# MECH-008 / B-GATE — information-theoretic slot priors
# ======================================================================================
@torch.no_grad()
def compute_slot_redundancy(cache, ridge_rel: float = 1e-6) -> torch.Tensor:
    """Linear reconstructibility of each slot's value from the other slots.

    Verbatim the definition in `research_loop/results/DIAG-IMPORTANCE/METRICS.md`
    §5 (`measure_slot_importance.py::compute_redundancy`), so the selector and the
    diagnostic rank the same slots.

    For layer `l`, stack the slot value vectors with the KV heads CONCATENATED,
    `V in R^{T_c x (H*d)}` (concatenating heads is what makes the question
    non-degenerate: per head the vectors live in R^128 and 511 of them span the
    whole space, so the residual would be identically zero)::

        r_j^2 = min_c || v_j - sum_{i != j} c_i v_i ||^2  =  1 / (G^-1)_{jj}
        G     = V V^T + lam I,   lam = ridge_rel * mean(diag(V V^T))
        redundancy[l, j] = 1 - r_j^2 / ||v_j||^2   in [0, 1]

    which is exactly the uncentred, no-intercept R^2 of regressing slot `j` on all
    other slots. The frozen sink is available as a REGRESSOR but is not returned.
    `lam` is a numerical guard only (the Gram's condition number is 3.5e3-2.2e4,
    so lam_rel in {1e-8, 1e-6, 1e-4} agree to ~4 decimal places).

    Split-independent and gradient-free: a property of the cartridge values, not
    of any query distribution. One `T_c x T_c` float64 inverse per layer.

    Returns:
        (n_layers, n_trainable_tokens) float32 CPU tensor in [0, 1].
    """
    if not (ridge_rel > 0.0):
        raise ValueError(
            f"redundancy_ridge_rel must be > 0, got {ridge_rel!r}: it is the "
            "relative ridge that keeps the Gram matrix invertible."
        )
    values = getattr(cache, "trainable_values", None)
    if values is None:
        raise ValueError(
            "compute_slot_redundancy needs a TrainableCache with "
            "`trainable_values`; got "
            f"{type(cache).__name__} without it."
        )
    n_frozen = int(getattr(cache, "_num_frozen_tokens", 0) or 0)
    frozen_values = getattr(cache, "frozen_values", None)

    rows = []
    for layer_idx in range(len(values)):
        v = values[layer_idx]  # (1, H, T_train, d)
        if n_frozen > 0 and frozen_values is not None:
            V = torch.cat([frozen_values[layer_idx], v], dim=2)[0]
        else:
            V = v[0]
        n_heads, n_slots, head_dim = V.shape
        Vf = V.permute(1, 0, 2).reshape(n_slots, n_heads * head_dim).double()
        G = Vf @ Vf.T
        gdiag = torch.diagonal(G)
        lam = float(ridge_rel) * float(gdiag.mean().item())
        Gi = torch.linalg.inv(
            G + lam * torch.eye(n_slots, dtype=G.dtype, device=G.device)
        )
        r2 = 1.0 / torch.diagonal(Gi).clamp_min(1e-300)
        rr = (1.0 - r2 / gdiag.clamp_min(1e-300)).clamp(0.0, 1.0)
        rows.append(rr[n_frozen:].float().cpu())

    red = torch.stack(rows, dim=0)
    if not torch.isfinite(red).all():
        raise RuntimeError(
            "compute_slot_redundancy produced non-finite scores; the Gram matrix "
            "is likely degenerate. Raise `redundancy_ridge_rel`."
        )
    return red


_FISHER_KEYS = ("score_fisher", "fisher", "slot_fisher", "score_fisher_qa")


def load_slot_fisher_scores(
    path: str,
    n_layers: int,
    n_tokens: int,
) -> torch.Tensor:
    """Load a cached (n_layers, n_tokens) diagonal-Fisher array from disk.

    The Fisher of the QA loss w.r.t. each slot's value needs a *diagnostic*
    backward pass over QA data. That is not an optimizer step -- no optimizer is
    ever constructed and the cache is never updated, so `gradient_steps` stays 0
    -- but it IS a real cost, so it is paid ONCE offline (see
    `research_loop/results/MECH-INFOGATE/compute_slot_fisher.py`, which is
    `DIAG-IMPORTANCE::collect_fisher`) and cached. This function only reads.

    Accepts a `.npz` (any of `score_fisher` / `fisher` / `slot_fisher` /
    `score_fisher_qa`) or a `.pt` holding a tensor or a dict with one of those
    keys. Fails loudly on a missing file, a wrong shape, or non-finite/negative
    entries -- a silently wrong prior would look like a plausible experiment.
    """
    if not path:
        raise ValueError(
            "slot_selection='fisher' (or a fisher-weighted mode) requires "
            "`slot_fisher_path`. Set AM_SLOT_FISHER_PATH to a cached score file; "
            "generate one with "
            "research_loop/results/MECH-INFOGATE/compute_slot_fisher.py."
        )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"slot_fisher_path={path!r} does not exist. Generate it with "
            "research_loop/results/MECH-INFOGATE/compute_slot_fisher.py "
            "(one diagnostic backward pass over the QA split; gradient_steps stays 0)."
        )

    if path.endswith(".npz"):
        import numpy as np

        with np.load(path) as z:
            key = next((k for k in _FISHER_KEYS if k in z.files), None)
            if key is None:
                raise KeyError(
                    f"{path!r} has none of {_FISHER_KEYS}; found {list(z.files)[:12]}"
                )
            arr = torch.from_numpy(np.asarray(z[key]))
    else:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            key = next((k for k in _FISHER_KEYS if k in obj), None)
            if key is None:
                raise KeyError(
                    f"{path!r} has none of {_FISHER_KEYS}; found {list(obj)[:12]}"
                )
            arr = obj[key]
        else:
            arr = obj
        arr = torch.as_tensor(arr)

    arr = arr.float().cpu()
    if tuple(arr.shape) != (n_layers, n_tokens):
        raise ValueError(
            f"slot_fisher_path={path!r} holds shape {tuple(arr.shape)} but this "
            f"cache has {n_layers} layers x {n_tokens} trainable slots. A Fisher "
            "array measured on a different cartridge cannot be reused."
        )
    if not torch.isfinite(arr).all():
        raise ValueError(f"slot_fisher_path={path!r} contains non-finite entries.")
    if (arr < 0).any():
        raise ValueError(
            f"slot_fisher_path={path!r} contains negative entries; the diagonal "
            "Fisher is a sum of squares and must be >= 0."
        )
    return arr


def _unit_rank(scores: torch.Tensor) -> torch.Tensor:
    """Per-layer ordinal rank mapped to (0, 1]; 1.0 = the layer's largest score.

    Rank space is used (rather than the raw values) because the two axes being
    combined live on incomparable scales -- `tf` sums to 1 over 511 slots while
    `redundancy` is an R^2 crowded near 1 -- so any additive or multiplicative
    rule on the raw numbers is silently dominated by one of them.
    Ties are broken by slot index (`stable=True`), i.e. ordinal not average ranks.
    """
    n = scores.shape[-1]
    order = torch.argsort(scores, dim=-1, stable=True)  # ascending
    ranks = torch.empty_like(order)
    ar = torch.arange(1, n + 1, device=scores.device).expand_as(order)
    ranks.scatter_(-1, order, ar)
    return ranks.to(scores.dtype) / float(n)


def _mask_from_topk(
    score: torch.Tensor,
    top_t: int,
    largest: bool,
) -> GradientMask:
    n_layers, n_tokens = score.shape
    k = min(top_t, n_tokens)
    _, top = torch.topk(score, k=k, dim=-1, largest=largest)
    return GradientMask(
        granularity="per_layer",
        positions_per_layer={l: top[l] for l in range(n_layers)},
        n_tokens=n_tokens,
        top_t=k,
    )


def _rank_slot_prior_per_layer(
    access_scores: torch.Tensor,
    top_t: int,
    mode: str,
    cache,
    redundancy_ridge_rel: float = 1e-6,
    mass_redundancy_alpha: float = 0.5,
    slot_fisher_path: Optional[str] = None,
) -> tuple[GradientMask, TFIDFRankingInfo]:
    """MECH-008: pick the top-t slots by an information-theoretic prior.

    `access_scores` is (n_layers, n_tokens) and is only used by the modes that
    need the new task's demand (`mass_x_redundancy`).
    """
    scores = access_scores.float().cpu()
    n_layers, n_tokens = scores.shape
    totals = scores.sum(-1, keepdim=True).clamp(min=1e-12)
    tf = scores / totals

    if mode == "redundancy":
        # High redundancy => the slot's value is reconstructible from the others
        # => cheapest to overwrite. DESCENDING.
        prior = compute_slot_redundancy(cache, ridge_rel=redundancy_ridge_rel)
        _check_prior_shape(prior, n_layers, n_tokens, mode)
        mask = _mask_from_topk(prior, top_t, largest=True)
        select_score = prior

    elif mode == "fisher":
        # Low Fisher => the QA loss is flat in that slot's value => safest to
        # overwrite. ASCENDING.
        prior = load_slot_fisher_scores(slot_fisher_path, n_layers, n_tokens)
        _check_prior_shape(prior, n_layers, n_tokens, mode)
        mask = _mask_from_topk(prior, top_t, largest=False)
        select_score = -prior

    elif mode == "mass_x_redundancy":
        # The direct attempt to beat the measured anti-alignment: slots the new
        # task WANTS *and* that are redundant.
        #
        #   u_tf, u_red = per-layer ordinal ranks in (0, 1]  (1 = best)
        #   score = u_tf^(1 - alpha) * u_red^alpha
        #
        # A PRODUCT (geometric mean at alpha=0.5) rather than a sum, because the
        # semantics wanted are AND, not OR: a slot that is last on either axis
        # gets a score of 1/511 and cannot be rescued by the other axis. The
        # weighted-geometric form also degenerates exactly to the two pure modes
        # -- alpha=0 reproduces `attention_mass` and alpha=1 reproduces
        # `redundancy`, both bit-for-bit up to tie-breaking -- so alpha is a
        # genuine interpolation knob, not a fourth arbitrary selector.
        if not (0.0 <= mass_redundancy_alpha <= 1.0):
            raise ValueError(
                "mass_redundancy_alpha must be in [0, 1], got "
                f"{mass_redundancy_alpha!r} (0 = pure attention mass, "
                "1 = pure redundancy)."
            )
        prior = compute_slot_redundancy(cache, ridge_rel=redundancy_ridge_rel)
        _check_prior_shape(prior, n_layers, n_tokens, mode)
        u_tf = _unit_rank(tf)
        u_red = _unit_rank(prior)
        a = float(mass_redundancy_alpha)
        select_score = u_tf.pow(1.0 - a) * u_red.pow(a)
        mask = _mask_from_topk(select_score, top_t, largest=True)

    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"unknown slot prior mode {mode!r}")

    if not torch.isfinite(select_score).all():
        raise RuntimeError(f"slot_selection={mode!r} produced non-finite scores")

    info = TFIDFRankingInfo(step=-1, tf=tf, tfidf=select_score, mask=mask)
    return mask, info


def _check_prior_shape(prior, n_layers: int, n_tokens: int, mode: str) -> None:
    if tuple(prior.shape) != (n_layers, n_tokens):
        raise ValueError(
            f"slot_selection={mode!r}: prior has shape {tuple(prior.shape)} but "
            f"access_scores are ({n_layers}, {n_tokens}). The prior must be "
            "measured on this cartridge."
        )


def rank_am_slots(
    access_scores: torch.Tensor,
    top_t: int,
    tfidf_ranker: CacheTFIDFRanker,
    config: "AttentionMatchingFinetuningConfig",
    old_access_scores: Optional[torch.Tensor] = None,
    step: int = 0,
    cache=None,
) -> tuple[GradientMask, TFIDFRankingInfo]:
    if config.slot_selection == "tfidf":
        return tfidf_ranker.rank_positions(
            access_scores,
            top_t=top_t,
            step=step,
            return_info=True,
        )

    if config.slot_selection in SLOT_PRIOR_SELECTIONS:
        # MECH-008 / B-GATE. Fail loud rather than fall back: a silent downgrade
        # to TF-IDF would look like a null result for the mechanism.
        if config.granularity != "per_layer":
            raise ValueError(
                f"slot_selection={config.slot_selection!r} is defined per layer "
                f"only, but granularity={config.granularity!r}. The priors are "
                "per-(layer, slot) arrays (DIAG-IMPORTANCE aggregates over all "
                "32 query heads), so there is no per-head or global variant."
            )
        if cache is None:
            raise ValueError(
                f"slot_selection={config.slot_selection!r} needs the cartridge "
                "cache (redundancy is a property of the slot VALUES, and the "
                "Fisher array is validated against the cache's shape), but "
                "`rank_am_slots` was called without `cache=`."
            )
        mask, info = _rank_slot_prior_per_layer(
            access_scores,
            top_t,
            mode=config.slot_selection,
            cache=cache,
            redundancy_ridge_rel=float(
                getattr(config, "redundancy_ridge_rel", 1e-6) or 1e-6
            ),
            mass_redundancy_alpha=float(
                getattr(config, "mass_redundancy_alpha", 0.5)
            ),
            slot_fisher_path=getattr(config, "slot_fisher_path", None),
        )
        info.step = step
        return mask, info

    if config.granularity != "per_layer":
        logger.warning(
            "%s selection currently supports per_layer only; falling back to TF-IDF",
            config.slot_selection,
        )
        return tfidf_ranker.rank_positions(
            access_scores,
            top_t=top_t,
            step=step,
            return_info=True,
        )

    if config.slot_selection == "attention_mass":
        mask, info = _rank_attention_mass_per_layer(access_scores, top_t)
    else:
        mask, info = _rank_residual_budget_per_layer(
            access_scores,
            top_t,
            old_access_scores=old_access_scores,
            idf_scores=tfidf_ranker.get_idf_scores(),
            idf_prior_weight=config.idf_prior_weight,
            min_top_t_per_layer=config.min_top_t_per_layer,
        )
    info.step = step
    return mask, info
