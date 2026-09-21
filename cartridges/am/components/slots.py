"""Slot stage: which cartridge slots a write is allowed to overwrite.

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

MECH-009 / B-GATE adds a fourth, `constrained_mass`, for a different reason.
MECH-INFOGATE measured that all three of the above LOSE acquisition badly, and
diagnosed why: DIAG-IMPORTANCE's promising projection (~34.6% of writable MT
routing mass at t=64) described a selector nobody had run -- best-t **mass
ranked WITHIN** the safest fraction of slots, i.e. a hard safety CONSTRAINT
followed by the incumbent's own ranking -- whereas the three modes above rank by
the safety metric ITSELF, which is 2.4-3.2x below that. `constrained_mass` is
that missing variant, with the constraint strength exposed as a knob.

    constrained_mass   -- restrict to the safest `safe_fraction` of slots per
                          layer (by `safe_metric` in {redundancy, fisher}), then
                          take the top-t by ATTENTION MASS within that set.
                          Degenerate by construction: safe_fraction = 1.0 is
                          exactly `attention_mass`, and safe_fraction small
                          enough that the candidate set is only top_t slots is
                          exactly the pure-safety selector (`redundancy` /
                          `fisher`). So it interpolates between the incumbent's
                          ranking and MECH-008's, and nothing else.

NOT implemented on purpose: `kl_loo`. DIAG-IMPORTANCE showed the exact
leave-one-out KL of dropping slot j is `-log(1 - w_j)`, a strictly monotone
function of the slot's own attention weight, so ranking by LOO-KL *is* ranking
by attention mass (measured rho = 0.968). Use `attention_mass`; do not rebuild
it as a separate mode.
"""

from __future__ import annotations

import math
import os
from logging import getLogger
from typing import Literal, Optional

import torch
from pydrantic import ObjectConfig

from cartridges.sparse_cache_finetuning import (
    CacheTFIDFRanker,
    GradientMask,
    TFIDFRankingInfo,
)

logger = getLogger(__name__)

# The MECH-008 selections plus MECH-009's `constrained_mass`. Each needs the
# *cache* (redundancy) and/or a cached score array on disk (fisher), neither of
# which the attention-mass family uses.
SLOT_PRIOR_SELECTIONS = (
    "redundancy",
    "fisher",
    "mass_x_redundancy",
    "constrained_mass",
)


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


def _safety_prior(
    metric: str,
    cache,
    n_layers: int,
    n_tokens: int,
    redundancy_ridge_rel: float,
    slot_fisher_path: Optional[str],
) -> tuple[torch.Tensor, bool]:
    """The MECH-008 scorers, re-used as a *safety* ordering (MECH-009).

    Returns `(scores, safest_is_largest)`. The two metrics disagree on direction:
    high `redundancy` means the slot is reconstructible from the others (safe to
    overwrite), while LOW `fisher` means the QA loss is flat in that slot (safe
    to overwrite).
    """
    if metric == "redundancy":
        prior = compute_slot_redundancy(cache, ridge_rel=redundancy_ridge_rel)
        safest_is_largest = True
    elif metric == "fisher":
        prior = load_slot_fisher_scores(slot_fisher_path, n_layers, n_tokens)
        safest_is_largest = False
    else:
        raise ValueError(
            f"safe_metric must be 'redundancy' or 'fisher', got {metric!r}. "
            "Those are the only two QA-importance scorers MECH-008 built; a "
            "third would need its own DIAG-IMPORTANCE validation."
        )
    _check_prior_shape(prior, n_layers, n_tokens, f"constrained_mass/{metric}")
    return prior, safest_is_largest


def _rank_slot_prior_per_layer(
    access_scores: torch.Tensor,
    top_t: int,
    mode: str,
    cache,
    redundancy_ridge_rel: float = 1e-6,
    mass_redundancy_alpha: float = 0.5,
    slot_fisher_path: Optional[str] = None,
    safe_fraction: float = 1.0,
    safe_metric: str = "redundancy",
) -> tuple[GradientMask, TFIDFRankingInfo]:
    """MECH-008: pick the top-t slots by an information-theoretic prior.

    `access_scores` is (n_layers, n_tokens) and is only used by the modes that
    need the new task's demand (`mass_x_redundancy`, `constrained_mass`).
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

    elif mode == "constrained_mass":
        # MECH-009. A HARD safety constraint, then the incumbent's own ranking
        # inside it -- the variant DIAG-IMPORTANCE's projection actually
        # described and that MECH-INFOGATE never ran.
        #
        #   candidates_l = the `n_safe` safest slots of layer l by `safe_metric`
        #   selection_l  = top-k of `tf` restricted to candidates_l
        #
        # `n_safe = floor(safe_fraction * n_tokens)`, floored at k so the budget
        # is always fillable. FLOOR, not round, so that safe_fraction = 0.25 on
        # 511 writable slots gives 127 -- DIAG-IMPORTANCE's own "safest
        # quartile" size, which is what makes its published 4.62% / 0.28%
        # tradeoff row directly reproducible by this mode.
        # The two degeneracies are exact and are the reason
        # this is an interpolation knob rather than a fifth ad-hoc selector:
        #   safe_fraction = 1.0    -> n_safe = n_tokens -> no slot is excluded,
        #                             `select_score` IS `tf`, so the selection is
        #                             bit-for-bit `attention_mass`;
        #   safe_fraction <= k/n   -> n_safe = k -> the candidate set has exactly
        #                             k members, so mass ranking is a no-op and
        #                             the selection is the PURE safety selector
        #                             (`redundancy` / `fisher`).
        # Excluded slots get the sentinel -1.0 rather than -inf: `tf >= 0`
        # always (it is a normalised attention mass), so -1.0 orders strictly
        # below every candidate while keeping `select_score` finite for the
        # finiteness assert below and for the saved `ranking_info`.
        if not (0.0 < safe_fraction <= 1.0):
            raise ValueError(
                "safe_fraction must be in (0, 1], got "
                f"{safe_fraction!r} (1.0 = unconstrained = attention_mass; "
                "small = only the safest slots are candidates)."
            )
        k = min(top_t, n_tokens)
        n_safe = int(math.floor(float(safe_fraction) * n_tokens))
        n_safe = max(k, min(n_safe, n_tokens))
        safety, safest_is_largest = _safety_prior(
            safe_metric,
            cache,
            n_layers,
            n_tokens,
            redundancy_ridge_rel,
            slot_fisher_path,
        )
        if n_safe >= n_tokens:
            select_score = tf
        else:
            _, cand = torch.topk(
                safety, k=n_safe, dim=-1, largest=safest_is_largest
            )
            keep = torch.zeros_like(tf, dtype=torch.bool)
            keep.scatter_(-1, cand, True)
            select_score = torch.where(keep, tf, torch.full_like(tf, -1.0))
        logger.info(
            "MECH-009 constrained_mass: safe_metric=%s safe_fraction=%.4f -> "
            "%d/%d candidate slots per layer, top_t=%d",
            safe_metric,
            safe_fraction,
            n_safe,
            n_tokens,
            k,
        )
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
    config: "SlotSelector.Config",
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
            redundancy_ridge_rel=float(config.redundancy_ridge_rel or 1e-6),
            mass_redundancy_alpha=float(config.mass_redundancy_alpha),
            slot_fisher_path=config.slot_fisher_path,
            # MECH-009. Passed straight through with NO `or` fallback: that would
            # silently turn an (invalid) safe_fraction=0.0 into the unconstrained
            # selector instead of raising.
            safe_fraction=config.safe_fraction,
            safe_metric=config.safe_metric,
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


class SlotSelector:
    """Ranks cartridge slots and returns the write mask for one document."""

    class Config(ObjectConfig):
        _pass_as_config = True

        top_t: int = 64
        granularity: Literal["global", "per_layer", "per_head"] = "per_layer"
        slot_selection: Literal[
            "tfidf",
            "attention_mass",
            "residual_budget",
            # MECH-008 / MECH-009 information-theoretic priors.
            "redundancy",
            "fisher",
            "mass_x_redundancy",
            "constrained_mass",
        ] = "tfidf"
        use_idf: bool = True
        idf_smoothing: float = 1.0
        background_top_k_per_batch: int = 128
        background_indices_path: Optional[str] = None
        num_background_batches: int = 1000

        # MECH-008: inert unless `slot_selection` names a prior mode.
        redundancy_ridge_rel: float = 1e-6
        mass_redundancy_alpha: float = 0.5  # 0 = pure mass, 1 = pure redundancy
        slot_fisher_path: Optional[str] = None  # cached (n_layers, n_slots) array

        # MECH-009: mass ranking restricted to the safest slots. Inert elsewhere,
        # and `safe_fraction=1.0` excludes nothing, so the default
        # `constrained_mass` is identical to `attention_mass`.
        safe_fraction: float = 1.0
        safe_metric: Literal["redundancy", "fisher"] = "redundancy"

        # `residual_budget` only.
        idf_prior_weight: float = 0.0
        min_top_t_per_layer: int = 1

        # Soft slot locality (SOFT-LOCALITY). A dynamic, self-referential penalty
        # on slots this run has already written, applied to the selection score
        # BEFORE top-k. `usage_penalty_lambda=0.0` is a bit-identical no-op (it
        # multiplies every score by 1.0), so it recovers the incumbent selector
        # exactly. Larger lambda pushes writes off already-used slots, sliding
        # from full sharing (lambda=0, the observed collision) to hard isolation
        # (lambda large). This is distinct from IDF, which is a STATIC background
        # prior identical for every document; `usage` is updated online after
        # every write. Requires granularity='per_layer'.
        usage_penalty_lambda: float = 0.0
        usage_penalty_mode: Literal["mult", "sub"] = "mult"
        # Per-stage exponential fade of the accumulated usage counts, applied
        # once when the stage loads the carried-over state. 1.0 = full memory
        # (no decay); 0.0 = reset every stage (within-stage spreading only).
        usage_decay: float = 1.0

    def __init__(self, config: Config):
        self.config = config

    def build_ranker(self) -> CacheTFIDFRanker:
        """The TF-IDF ranker this selection is scored against.

        Loading the background corpus is the one setup cost of this stage, so it
        is done once per run and the ranker is handed back to the caller rather
        than cached on `self` (components stay stateless across documents).
        """
        # Imported here rather than at module scope: `BackgroundAccessTracker` is
        # only needed when a background corpus was actually collected.
        from cartridges.sparse_cache_finetuning import BackgroundAccessTracker

        bg_tracker = None
        if self.config.background_indices_path is not None:
            bg_tracker = BackgroundAccessTracker(
                num_batches=self.config.num_background_batches,
                granularity=self.config.granularity,
            )
            bg_tracker.load(self.config.background_indices_path)
        return CacheTFIDFRanker(
            background_tracker=bg_tracker,
            use_idf=self.config.use_idf,
            smoothing=self.config.idf_smoothing,
            top_k_per_batch=self.config.background_top_k_per_batch,
            granularity=self.config.granularity,
        )

    def select(
        self,
        access_scores: torch.Tensor,
        tfidf_ranker: CacheTFIDFRanker,
        *,
        cache=None,
        old_access_scores: Optional[torch.Tensor] = None,
        step: int = 0,
        usage: Optional[torch.Tensor] = None,
    ) -> tuple[GradientMask, TFIDFRankingInfo]:
        access_scores = self._apply_usage_penalty(access_scores, usage)
        return rank_am_slots(
            access_scores,
            self.config.top_t,
            tfidf_ranker,
            self.config,
            old_access_scores=old_access_scores,
            step=step,
            # MECH-008 slot priors; ignored by every other selection.
            cache=cache,
        )

    def _apply_usage_penalty(
        self,
        access_scores: torch.Tensor,
        usage: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Adjust slot selection scores based on prior usage.

        When lambda > 0 and exponent is positive (+lam), already-written slots
        receive a multiplicative bonus `score * (1 + usage) ** (lambda)` to
        encourage overlapping/slot reuse across stages. `lambda <= 0` or
        `usage is None` is a no-op.
        """
        lam = float(self.config.usage_penalty_lambda)
        if lam <= 0.0 or usage is None:
            return access_scores
        if self.config.granularity != "per_layer":
            raise ValueError(
                "usage_penalty_lambda>0 requires granularity='per_layer'; got "
                f"{self.config.granularity!r}."
            )
        u = usage.to(device=access_scores.device, dtype=access_scores.dtype)
        if tuple(u.shape) != tuple(access_scores.shape):
            raise ValueError(
                f"usage shape {tuple(u.shape)} != access_scores shape "
                f"{tuple(access_scores.shape)}; usage must be (n_layers, n_slots)."
            )
        if self.config.usage_penalty_mode == "mult":
            # Reversed exponent (+lam): encourage reusing previously written slots across stages
            return access_scores * torch.pow(1.0 + u, lam)
        raise NotImplementedError(
            f"usage_penalty_mode={self.config.usage_penalty_mode!r} is not "
            "implemented; use 'mult'."
        )
