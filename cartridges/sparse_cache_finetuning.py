"""
Sparse cache finetuning for cartridge continual learning.

Adapts the TF-IDF sparse memory finetuning approach (Lin et al., 2025) to
trainable KV-cache cartridges. Instead of tracking FFN memory slot activations,
we track attention scores from input tokens to trainable cache positions.

Usage:
    config = SparseCacheFinetuningConfig(enabled=True, top_t=256)
    tracker = CacheAccessTracker(n_trainable_tokens=1024, device="cuda")

    # During forward: accumulate attention-based access scores
    tracker.accumulate(cache, model)

    # On optimizer step: mask gradients so only top_t positions update
    mask_cache_gradients(cache, tracker, ranker, config)
    optimizer.step()
    tracker.reset()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
from typing import Dict, Literal, Optional, Union

import torch
import torch.nn as nn
from pydrantic import BaseConfig

logger = getLogger(__name__)


class SparseCacheFinetuningConfig(BaseConfig):
    enabled: bool = False
    top_t: int = 256
    background_indices_path: Optional[str] = None
    num_background_batches: int = 1000
    use_idf: bool = True
    idf_smoothing: float = 1.0
    collect_background_stats: bool = False
    background_top_k_per_batch: int = 128
    
    # Momentum handling for masked (non-top-t) positions.
    #
    # "soft"     - Do nothing extra. Gradient is 0 so momentum decays exponentially
    #              (v_t = momentum * v_{t-1}) and the param still drifts for ~20 steps.
    #              Paper's original approach (Lin et al., 2025).
    #
    # "hard"     - Zero momentum before optimizer.step(). Param immediately stops.
    #              Discards accumulated momentum for masked positions entirely.
    #
    # "freeze"   - Save param AND momentum for non-top-t positions before step,
    #              restore both after step. Param is frozen and momentum is
    #              preserved exactly as-is (neither decays nor updates).
    #              Good when you expect positions to re-enter top-t and want to
    #              resume from where they left off.
    #
    # "decouple" - Save param only for non-top-t positions before step, restore
    #              after step. Momentum decays naturally (v_t = momentum * v_{t-1})
    #              but the param never moves. Momentum stays warm without drift.
    momentum_masking: Literal["soft", "hard", "freeze", "decouple"] = "soft"

    # Whether to freeze key matrices (zero their gradients post-backward).
    # True  (default) - keys are never updated; only value positions in top_t change.
    # False           - keys receive gradients and are updated by the optimizer like values.
    #                   All momentum masking modes respect this flag: key momentum is
    #                   only zeroed (hard mode) when freeze_keys=True.
    freeze_keys: bool = True

    # Granularity at which TF-IDF scores are computed and gradient masks are applied.
    #
    # "global"    - one shared top-t mask for every layer and every head (original).
    # "per_layer" - each layer has its own top-t mask (TF summed over heads and docs).
    # "per_head"  - each (layer, head) pair has its own top-t mask (TF summed over docs only).
    #
    # Finer granularity means each sub-component protects its own knowledge independently,
    # at the cost of higher memory for IDF storage and slightly more masking overhead.
    granularity: Literal["global", "per_layer", "per_head"] = "global"


class CacheAccessTracker:
    """Accumulates per-position attention scores across microbatches.

    After each forward pass, call `accumulate_from_hooks()` to add the attention
    scores from the last token of each packed sequence to the running totals.
    On `do_step`, call `get_access_counts()` then `reset()`.

    The `granularity` parameter controls which dimensions are preserved:
      "global"    - scores summed over layers, heads, and docs → (n_tokens,)
      "per_layer" - scores summed over heads and docs per layer → (n_layers, n_tokens)
      "per_head"  - scores summed over docs only → (n_layers, n_kv_heads, n_tokens)
    """

    def __init__(
        self,
        n_trainable_tokens: int,
        device: torch.device,
        granularity: Literal["global", "per_layer", "per_head"] = "global",
        n_layers: int = 1,
        n_kv_heads: int = 1,
    ):
        self.n_trainable_tokens = n_trainable_tokens
        self.device = device
        self.granularity = granularity
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads

        if granularity == "global":
            self._scores = torch.zeros(n_trainable_tokens, device=device)
        elif granularity == "per_layer":
            self._scores = torch.zeros(n_layers, n_trainable_tokens, device=device)
        else:  # per_head
            self._scores = torch.zeros(n_layers, n_kv_heads, n_trainable_tokens, device=device)
        self._num_accumulations = 0

    def accumulate_from_hooks(
        self,
        captured_queries: dict[int, torch.Tensor],
        cache: nn.Module,
        scaling: float,
        seq_ids: torch.Tensor,
    ):
        """Compute attention scores from captured post-RoPE queries to cache keys.

        Extracts the last token of *each* packed sequence (identified by seq_ids)
        and computes its attention to all trainable cache positions.

        Args:
            captured_queries: {layer_idx: query_states} where query_states is
                (1, n_q_heads, seq_len, head_dim) -- post-RoPE, pre-flex_attention.
            cache: TrainableCache with trainable_keys[layer_idx] of shape
                (1, n_kv_heads, n_trainable_tokens, head_dim).
            scaling: attention scaling factor (head_dim ** -0.5).
            seq_ids: (seq_len,) tensor of element/sequence IDs for the packed batch.
        """
        with torch.no_grad():
            # Find the last token index of each unique sequence in the pack.
            # seq_ids looks like [0,0,0,1,1,1,2,2,2,2] where each number identifies
            # a distinct document in the packed window. We pick the final position of
            # each document as its representative query for cache access scoring.
            unique_ids = seq_ids.unique()
            last_indices = []
            for uid in unique_ids:
                positions = (seq_ids == uid).nonzero(as_tuple=True)[0]
                last_indices.append(positions[-1].item())
            last_indices = torch.tensor(last_indices, device=seq_ids.device)

            for layer_idx, q in captured_queries.items():
                cache_k = cache.trainable_keys[layer_idx]
                n_kv_heads = cache_k.shape[1]
                n_q_heads = q.shape[1]
                groups = n_q_heads // n_kv_heads

                # Select last token of each sequence: (1, n_q_heads, n_seqs, head_dim)
                q_selected = q[:, :, last_indices, :]
                n_seqs = q_selected.shape[2]

                # Group query heads → KV heads by averaging:
                # (1, n_q_heads, n_seqs, d) → (1, n_kv_heads, groups, n_seqs, d)
                q_grouped = q_selected.view(1, n_kv_heads, groups, n_seqs, -1)
                # (1, n_kv_heads, n_seqs, head_dim)
                q_mean = q_grouped.mean(dim=2)

                # attn_weights: (1, n_kv_heads, n_seqs, n_trainable_tokens)
                raw_scores = torch.matmul(q_mean, cache_k.transpose(-1, -2)) * scaling
                attn_weights = raw_scores.softmax(dim=-1)

                if self.granularity == "global":
                    # Sum over batch(=1), kv_heads, docs → (n_tokens,)
                    self._scores += attn_weights.sum(dim=(0, 1, 2))
                elif self.granularity == "per_layer":
                    # Sum over batch(=1), kv_heads, docs per layer → (n_tokens,) at [layer_idx]
                    self._scores[layer_idx] += attn_weights.sum(dim=(0, 1, 2))
                else:  # per_head
                    # Sum over batch(=1) and docs per (layer, head) → (n_kv_heads, n_tokens) at [layer_idx]
                    self._scores[layer_idx] += attn_weights.sum(dim=(0, 2))

        self._num_accumulations += 1

    def get_access_counts(self) -> torch.Tensor:
        """Returns accumulated scores (on device).

        Shape:
          global:    (n_tokens,)
          per_layer: (n_layers, n_tokens)
          per_head:  (n_layers, n_kv_heads, n_tokens)
        """
        return self._scores

    def reset(self):
        self._scores.zero_()
        self._num_accumulations = 0


class BackgroundAccessTracker:
    """Tracks per-batch ranked cache position scores for flexible IDF computation.

    For each background batch, stores a full ranking of cache positions by
    attention score (descending) along the last dimension. The granularity
    determines the shape of scores accepted and rankings stored:

      "global"    - (n_tokens,)               → rankings: (n_batches, n_tokens)
      "per_layer" - (n_layers, n_tokens)       → rankings: (n_batches, n_layers, n_tokens)
      "per_head"  - (n_layers, n_kv_heads, n_tokens) → rankings: (n_batches, n_layers, n_kv_heads, n_tokens)

    At finetuning time, any top-k cutoff can be applied to compute document
    frequencies without re-running collection.
    """

    def __init__(
        self,
        num_batches: int = 1000,
        granularity: Literal["global", "per_layer", "per_head"] = "global",
    ):
        self.num_batches = num_batches
        self.granularity = granularity
        self.batch_ranked_positions: list[torch.Tensor] = []
        self.current_batch_id = 0

    def add_batch_scores(self, scores: torch.Tensor):
        """Record the full position ranking for this batch.

        Args:
            scores: attention-based access scores. Shape matches granularity:
                global:    (n_tokens,)
                per_layer: (n_layers, n_tokens)
                per_head:  (n_layers, n_kv_heads, n_tokens)
        """
        if self.current_batch_id >= self.num_batches:
            return
        # Sort descending along the last dimension (positions)
        _, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        self.batch_ranked_positions.append(sorted_indices.cpu())
        self.current_batch_id += 1

    def compute_document_frequencies(self, top_k: int) -> torch.Tensor:
        """Compute document frequency for every position at a given top-k.

        Args:
            top_k: how many top positions per batch count as "present".

        Returns:
            Long tensor whose shape matches the leading dims of the stored rankings
            (without the n_batches prefix):
                global:    (n_tokens,)
                per_layer: (n_layers, n_tokens)
                per_head:  (n_layers, n_kv_heads, n_tokens)

            Entry [..., i] is the number of batches in which position i appeared
            within the top-k for that (layer, head) combination.
        """
        if not self.batch_ranked_positions:
            return torch.zeros(0, dtype=torch.long)

        # stacked: (n_batches, [*leading_dims], n_positions)
        stacked = torch.stack(self.batch_ranked_positions)
        n_batches = stacked.shape[0]
        n_positions = stacked.shape[-1]
        leading_shape = stacked.shape[1:-1]  # () / (n_layers,) / (n_layers, n_kv_heads)

        k = min(top_k, n_positions)
        # top_slices: (n_batches, [*leading_dims], k) — positions are already sorted desc
        top_slices = stacked[..., :k]

        # Flatten leading dims so we can use a single vectorised scatter_add_ loop
        D = 1
        for d in leading_shape:
            D *= d
        top_flat = top_slices.view(n_batches, D, k)   # (n_batches, D, k)
        df_flat = torch.zeros(D, n_positions, dtype=torch.long)

        for d_idx in range(D):
            indices = top_flat[:, d_idx, :].reshape(-1)  # (n_batches * k,)
            df_flat[d_idx].scatter_add_(0, indices, torch.ones(len(indices), dtype=torch.long))

        if leading_shape:
            return df_flat.view(*leading_shape, n_positions)
        else:
            return df_flat.view(n_positions)

    def save(self, path: str):
        ranked_tensor = torch.stack(self.batch_ranked_positions)
        torch.save(
            {
                "batch_ranked_positions": ranked_tensor,
                "num_batches": self.current_batch_id,
                "granularity": self.granularity,
            },
            path,
        )
        logger.info(f"Saved background access stats to {path} (granularity={self.granularity})")

    def load(self, path: str):
        data = torch.load(path, weights_only=False)
        ranked_tensor = data["batch_ranked_positions"]
        self.batch_ranked_positions = list(ranked_tensor.unbind(0))
        self.current_batch_id = data["num_batches"]
        saved_granularity = data.get("granularity", "global")
        if saved_granularity != self.granularity:
            logger.warning(
                f"BackgroundAccessTracker: loaded granularity '{saved_granularity}' "
                f"does not match current granularity '{self.granularity}'. "
                "IDF computation may be incorrect."
            )
        logger.info(
            f"Loaded background stats: shape={tuple(ranked_tensor.shape)}, "
            f"{self.current_batch_id} batches, granularity={saved_granularity}"
        )


@dataclass
class GradientMask:
    """Encapsulates which cache value positions to keep gradients for, per layer/head.

    This is the central contract between the TF-IDF ranker and all masking
    functions. It supports three granularities:

      "global"    – same top_t positions shared by every layer and head.
      "per_layer" – each layer has its own top_t positions, broadcast across heads.
      "per_head"  – each (layer, head) pair has its own top_t positions.

    positions_per_layer layout:
      "global" / "per_layer"  → dict[layer_idx → LongTensor(top_t,)]   (CPU)
      "per_head"              → dict[layer_idx → LongTensor(n_kv_heads, top_t)]  (CPU)
    """

    granularity: Literal["global", "per_layer", "per_head"]
    positions_per_layer: Dict[int, torch.Tensor]
    n_tokens: int
    top_t: int

    def build_4d_mask(self, layer_idx: int, n_kv_heads: int, device) -> torch.Tensor:
        """Build a (1, n_kv_heads, n_tokens, 1) float keep-mask for this layer.

        Returns 1.0 where gradients should be kept, 0.0 where they should be zeroed.
        For global/per_layer the same positions are used for every head (broadcast).
        For per_head each head has an independent set of kept positions.
        """
        mask = torch.zeros(1, n_kv_heads, self.n_tokens, 1, device=device)
        pos = self.positions_per_layer[layer_idx]
        if self.granularity in ("global", "per_layer"):
            # pos: (top_t,) — identical for all heads
            mask[0, :, pos.to(device), 0] = 1.0
        else:  # per_head: pos is (n_kv_heads, top_t)
            pos_dev = pos.to(device)
            head_idx = torch.arange(n_kv_heads, device=device).unsqueeze(1).expand_as(pos_dev)
            mask[0, head_idx, pos_dev, 0] = 1.0
        return mask

    def get_complement_positions(self, layer_idx: int, device) -> torch.Tensor:
        """1-D LongTensor of positions NOT in the top-t for this layer.

        Only valid for "global" and "per_layer" granularities (where all heads
        share the same positions). Raises for "per_head".
        """
        if self.granularity == "per_head":
            raise ValueError(
                "get_complement_positions is not defined for per_head granularity; "
                "use build_4d_mask directly."
            )
        pos = self.positions_per_layer[layer_idx].to(device)
        keep = torch.zeros(self.n_tokens, dtype=torch.bool, device=device)
        keep[pos] = True
        return (~keep).nonzero(as_tuple=True)[0]

    def get_union_top_positions(self, layer_idx: int) -> torch.Tensor:
        """Union of kept positions across all heads for this layer (1-D CPU LongTensor).

        Useful for summary logging.
        """
        pos = self.positions_per_layer[layer_idx]
        if self.granularity in ("global", "per_layer"):
            return pos
        return pos.reshape(-1).unique()


@dataclass
class TFIDFRankingInfo:
    """Per-step TF-IDF ranking information for logging and analysis."""

    step: int
    # tf and tfidf shapes depend on granularity:
    #   global:    (n_tokens,)
    #   per_layer: (n_layers, n_tokens)
    #   per_head:  (n_layers, n_kv_heads, n_tokens)
    tf: torch.Tensor
    tfidf: torch.Tensor
    mask: GradientMask  # which positions were selected this step


class CacheTFIDFRanker:
    """Ranks cache positions by TF-IDF score and produces a GradientMask.

    TF = normalized attention score for position i in the current global batch,
         computed at the configured granularity (global / per_layer / per_head).
    IDF = log((|B| + s) / (df(i) + s)) where |B| is the number of background
         batches and df(i) is how many background batches had position i in top-k.
         Shape matches the granularity of the BackgroundAccessTracker used.

    The top_k_per_batch parameter controls how deep into each batch's ranked
    list to look when computing document frequency, allowing flexible trade-off
    between selectivity and coverage without re-running background collection.
    """

    def __init__(
        self,
        background_tracker: Optional[BackgroundAccessTracker] = None,
        use_idf: bool = True,
        smoothing: float = 1.0,
        top_k_per_batch: int = 128,
        granularity: Literal["global", "per_layer", "per_head"] = "global",
    ):
        self.use_idf = use_idf and background_tracker is not None
        self.smoothing = smoothing
        self.granularity = granularity
        self.num_background_batches = (
            background_tracker.current_batch_id if background_tracker else 0
        )
        self._idf: Optional[torch.Tensor] = None
        self._df: Optional[torch.Tensor] = None
        if self.use_idf:
            self._df = background_tracker.compute_document_frequencies(top_k_per_batch)
            self._idf = torch.log(
                (self.num_background_batches + smoothing) / (self._df.float() + smoothing)
            )
            # _idf shape matches granularity:
            #   global:    (n_tokens,)
            #   per_layer: (n_layers, n_tokens)
            #   per_head:  (n_layers, n_kv_heads, n_tokens)

    def get_idf_scores(self) -> Optional[torch.Tensor]:
        """Returns the IDF scores (constant across all batches)."""
        return self._idf.clone() if self._idf is not None else None

    def get_document_frequencies(self) -> Optional[torch.Tensor]:
        """Returns document frequencies used to compute IDF."""
        return self._df.clone() if self._df is not None else None

    def rank_positions(
        self,
        access_scores: torch.Tensor,
        top_t: int,
        step: int = -1,
        return_info: bool = False,
    ) -> Union[GradientMask, tuple[GradientMask, TFIDFRankingInfo]]:
        """Compute TF-IDF and select top-t positions, returning a GradientMask.

        Args:
            access_scores: accumulated attention scores with shape matching granularity:
                global:    (n_tokens,)
                per_layer: (n_layers, n_tokens)
                per_head:  (n_layers, n_kv_heads, n_tokens)
            top_t: number of positions to select per granularity unit.
            step: optimizer step index (for logging in TFIDFRankingInfo).
            return_info: if True, also return TFIDFRankingInfo with TF/TF-IDF tensors.

        Returns:
            GradientMask (and optionally TFIDFRankingInfo if return_info=True).
        """
        scores = access_scores.float().cpu()
        n_tokens = scores.shape[-1]
        idf_cpu = self._idf.cpu() if (self.use_idf and self._idf is not None) else None

        if self.granularity == "global":
            mask, tf, tfidf = self._rank_global(scores, top_t, n_tokens, idf_cpu)

        elif self.granularity == "per_layer":
            mask, tf, tfidf = self._rank_per_layer(scores, top_t, n_tokens, idf_cpu)

        else:  # per_head
            mask, tf, tfidf = self._rank_per_head(scores, top_t, n_tokens, idf_cpu)

        if return_info:
            info = TFIDFRankingInfo(step=step, tf=tf, tfidf=tfidf, mask=mask)
            return mask, info
        return mask

    # ------------------------------------------------------------------
    # Internal helpers – one per granularity
    # ------------------------------------------------------------------

    def _rank_global(
        self,
        scores: torch.Tensor,   # (n_tokens,)
        top_t: int,
        n_tokens: int,
        idf: Optional[torch.Tensor],  # (n_tokens,) or None
    ) -> tuple[GradientMask, torch.Tensor, torch.Tensor]:
        total = scores.sum()
        if total == 0:
            top = torch.arange(min(top_t, n_tokens))
            tf = torch.zeros_like(scores)
            positions = {0: top}  # layer key 0 is a sentinel; build_4d_mask broadcasts
        else:
            tf = scores / total
            tfidf = tf * idf if idf is not None else tf
            k = min(top_t, n_tokens)
            _, top = torch.topk(tfidf, k=k, largest=True)
            positions = {0: top}

        # For global, all layers point to the same position tensor (key 0 is canonical;
        # build_4d_mask must be called with any layer_idx, but we'll populate all lazily).
        # Store as a single shared entry and let build_4d_mask use key 0 via __missing__.
        tfidf_out = (tf * idf) if (idf is not None and total != 0) else tf
        mask = _GlobalGradientMask(positions_per_layer=positions, n_tokens=n_tokens, top_t=top_t)
        return mask, tf, tfidf_out

    def _rank_per_layer(
        self,
        scores: torch.Tensor,   # (n_layers, n_tokens)
        top_t: int,
        n_tokens: int,
        idf: Optional[torch.Tensor],  # (n_layers, n_tokens) or None
    ) -> tuple[GradientMask, torch.Tensor, torch.Tensor]:
        n_layers = scores.shape[0]
        totals = scores.sum(-1, keepdim=True).clamp(min=1e-12)  # (n_layers, 1)
        tf = scores / totals  # (n_layers, n_tokens)
        tfidf = tf * idf if idf is not None else tf
        k = min(top_t, n_tokens)
        _, top = torch.topk(tfidf, k=k, dim=-1, largest=True)  # (n_layers, top_t)
        positions = {l: top[l] for l in range(n_layers)}
        mask = GradientMask(
            granularity="per_layer",
            positions_per_layer=positions,
            n_tokens=n_tokens,
            top_t=top_t,
        )
        return mask, tf, tfidf

    def _rank_per_head(
        self,
        scores: torch.Tensor,   # (n_layers, n_kv_heads, n_tokens)
        top_t: int,
        n_tokens: int,
        idf: Optional[torch.Tensor],  # (n_layers, n_kv_heads, n_tokens) or None
    ) -> tuple[GradientMask, torch.Tensor, torch.Tensor]:
        n_layers, n_kv_heads, _ = scores.shape
        totals = scores.sum(-1, keepdim=True).clamp(min=1e-12)  # (n_layers, n_kv_heads, 1)
        tf = scores / totals  # (n_layers, n_kv_heads, n_tokens)
        tfidf = tf * idf if idf is not None else tf
        k = min(top_t, n_tokens)
        _, top = torch.topk(tfidf, k=k, dim=-1, largest=True)  # (n_layers, n_kv_heads, top_t)
        positions = {l: top[l] for l in range(n_layers)}  # {l: (n_kv_heads, top_t)}
        mask = GradientMask(
            granularity="per_head",
            positions_per_layer=positions,
            n_tokens=n_tokens,
            top_t=top_t,
        )
        return mask, tf, tfidf


class _GlobalGradientMask(GradientMask):
    """GradientMask for 'global' granularity where all layers share the same positions.

    Overrides positions_per_layer lookup so any layer_idx returns the single
    canonical top-t tensor stored under key 0.
    """

    def __init__(self, positions_per_layer: dict, n_tokens: int, top_t: int):
        super().__init__(
            granularity="global",
            positions_per_layer=positions_per_layer,
            n_tokens=n_tokens,
            top_t=top_t,
        )

    def build_4d_mask(self, layer_idx: int, n_kv_heads: int, device) -> torch.Tensor:
        mask = torch.zeros(1, n_kv_heads, self.n_tokens, 1, device=device)
        pos = self.positions_per_layer[0].to(device)  # always use key 0
        mask[0, :, pos, 0] = 1.0
        return mask

    def get_complement_positions(self, layer_idx: int, device) -> torch.Tensor:
        pos = self.positions_per_layer[0].to(device)
        keep = torch.zeros(self.n_tokens, dtype=torch.bool, device=device)
        keep[pos] = True
        return (~keep).nonzero(as_tuple=True)[0]

    def get_union_top_positions(self, layer_idx: int) -> torch.Tensor:
        return self.positions_per_layer[0]


def _trainable_value_layer_indices(n_layers: int) -> set[int]:
    """Layer indices whose value matrices may receive sparse updates.

    Currently updates all value layers (0 .. n_layers-1). Keys stay frozen.
    To restrict to the middle third, use e.g. range(n_layers // 3, 2 * n_layers // 3 + 1).
    """
    middle_start = 0
    middle_end = n_layers - 1
    return set(range(middle_start, middle_end + 1))


def freeze_key_gradients(
    cache: nn.Module,
    n_layers: int,
):
    """Zero all key gradients post-backward. Keys are never updated.

    Position-level masking for values is handled post-backward by
    apply_gradient_mask_to_cache.  This function handles the orthogonal
    concern: keys are always frozen regardless of position or layer.

    If layer-selective value freezing is needed in future (e.g. only middle
    layers), add that logic here alongside the key zeroing.

    Args:
        cache: TrainableCache with trainable_keys as nn.ParameterList.
        n_layers: number of layers.
    """
    frozen = 0
    no_grad = 0

    for layer_idx in range(n_layers):
        k_param = cache.trainable_keys[layer_idx]
        if k_param.grad is not None:
            k_param.grad.zero_()
            frozen += 1
        else:
            no_grad += 1

    logger.info(
        f"freeze_key_gradients: {frozen} key params zeroed, {no_grad} had no grad"
    )


def zero_momentum_for_masked_positions(
    cache: nn.Module,
    mask: GradientMask,
    n_layers: int,
    optimizer: torch.optim.Optimizer,
    freeze_keys: bool = True,
):
    """Zero momentum buffers for positions NOT selected by the GradientMask (hard freeze).

    Masked positions (non-top-t values, and keys when freeze_keys=True) have their
    momentum zeroed so they stop updating immediately with no residual drift.

    Args:
        cache: TrainableCache with trainable_keys/values as nn.ParameterList.
        mask: GradientMask specifying which value positions to KEEP.
        n_layers: number of layers.
        optimizer: Optimizer with momentum state to modify.
        freeze_keys: if True, zero all key momentum (keys are not being trained).
                     if False, leave key momentum untouched (keys are being trained).
    """
    device = cache.trainable_keys[0].device
    trainable_value_layers = _trainable_value_layer_indices(n_layers)
    momentum_masked = 0

    for layer_idx in range(n_layers):
        k_param = cache.trainable_keys[layer_idx]
        v_param = cache.trainable_values[layer_idx]

        if freeze_keys:
            momentum_masked += _mask_momentum_buffer(optimizer, k_param, None, freeze_all=True)

        if layer_idx in trainable_value_layers:
            n_kv_heads = v_param.shape[1]
            keep_4d = mask.build_4d_mask(layer_idx, n_kv_heads, device)
            momentum_masked += _mask_momentum_buffer(optimizer, v_param, keep_4d)
        else:
            momentum_masked += _mask_momentum_buffer(optimizer, v_param, None, freeze_all=True)

    logger.info(
        f"zero_momentum_for_masked_positions (HARD): "
        f"{momentum_masked} momentum buffers masked, "
        f"top_t={mask.top_t}, granularity={mask.granularity}, freeze_keys={freeze_keys}"
    )


# Keys used to store optimizer state snapshots for "freeze" / "decouple" modes.
_MOMENTUM_KEYS = ("momentum_buffer", "exp_avg", "exp_avg_sq")


def save_non_top_t_state(
    cache: nn.Module,
    mask: GradientMask,
    n_layers: int,
    optimizer: torch.optim.Optimizer,
    save_momentum: bool,
) -> dict:
    """Snapshot param values (and optionally momentum) for non-top-t positions.

    Call this BEFORE optimizer.step(). Pass the returned dict to
    restore_non_top_t_state() AFTER optimizer.step() to undo the update
    for those positions.

    For "global" and "per_layer" granularities a compact index-based snapshot is
    used (only the non-top positions are stored). For "per_head" the full value
    tensor is cloned because each head has a different set of kept positions.

    Args:
        cache: TrainableCache.
        mask: GradientMask specifying which positions are allowed to update.
        n_layers: number of model layers.
        optimizer: the optimizer (needed when save_momentum=True).
        save_momentum: if True, also snapshot optimizer momentum buffers.

    Returns:
        saved: dict keyed by layer_idx. Each entry contains the data needed by
        restore_non_top_t_state(); the internal format depends on granularity.
    """
    device = cache.trainable_keys[0].device
    trainable_value_layers = _trainable_value_layer_indices(n_layers)
    saved: dict = {"granularity": mask.granularity}

    for layer_idx in range(n_layers):
        if layer_idx not in trainable_value_layers:
            continue

        v_param = cache.trainable_values[layer_idx]

        if mask.granularity != "per_head":
            # Compact index-based save: only store non-top positions.
            non_top = mask.get_complement_positions(layer_idx, device)
            entry: dict = {
                "non_top_indices": non_top,
                "v_param": v_param.data[:, :, non_top, :].clone(),
            }
            if save_momentum and v_param in optimizer.state:
                state = optimizer.state[v_param]
                entry["v_momentum"] = {
                    k: state[k][:, :, non_top, :].clone()
                    for k in _MOMENTUM_KEYS
                    if k in state
                }
        else:
            # Full-tensor save: different heads have different non-top positions.
            n_kv_heads = v_param.shape[1]
            not_top_4d = (1.0 - mask.build_4d_mask(layer_idx, n_kv_heads, device)).cpu()
            entry = {
                "not_top_4d": not_top_4d,  # (1, n_kv_heads, n_tokens, 1) on CPU
                "v_param_full": v_param.data.clone(),
            }
            if save_momentum and v_param in optimizer.state:
                state = optimizer.state[v_param]
                entry["v_momentum_full"] = {
                    k: state[k].clone() for k in _MOMENTUM_KEYS if k in state
                }

        saved[layer_idx] = entry

    return saved


def restore_non_top_t_state(
    cache: nn.Module,
    saved: dict,
    n_layers: int,
    optimizer: torch.optim.Optimizer,
    restore_momentum: bool,
):
    """Restore param values (and optionally momentum) for non-top-t positions.

    Call this AFTER optimizer.step() to undo the update for masked positions.

    Args:
        cache: TrainableCache.
        saved: dict returned by save_non_top_t_state().
        n_layers: number of model layers.
        optimizer: the optimizer (needed when restore_momentum=True).
        restore_momentum: if True, also restore optimizer momentum buffers
                          to their pre-step values ("freeze" mode).
                          If False, only param is restored; momentum keeps its
                          post-step value ("decouple" mode).
    """
    trainable_value_layers = _trainable_value_layer_indices(n_layers)
    granularity = saved.get("granularity", "global")

    for layer_idx in range(n_layers):
        if layer_idx not in trainable_value_layers or layer_idx not in saved:
            continue

        entry = saved[layer_idx]
        v_param = cache.trainable_values[layer_idx]
        device = v_param.device

        if granularity != "per_head":
            non_top = entry["non_top_indices"]
            v_param.data[:, :, non_top, :] = entry["v_param"].to(device)
            if restore_momentum and v_param in optimizer.state:
                state = optimizer.state[v_param]
                for k, snap in entry.get("v_momentum", {}).items():
                    if k in state:
                        state[k][:, :, non_top, :] = snap.to(device)
        else:
            # Restore only the non-top positions using the saved not_top mask.
            not_top = entry["not_top_4d"].to(device=device, dtype=v_param.dtype)
            top = 1.0 - not_top
            v_param.data.mul_(top).add_(entry["v_param_full"].to(device=device, dtype=v_param.dtype) * not_top)
            if restore_momentum and v_param in optimizer.state:
                state = optimizer.state[v_param]
                for k, snap in entry.get("v_momentum_full", {}).items():
                    if k in state:
                        buf = state[k]
                        not_top_buf = not_top.to(dtype=buf.dtype)
                        buf.mul_(1.0 - not_top_buf).add_(snap.to(device=device, dtype=buf.dtype) * not_top_buf)



def _mask_momentum_buffer(
    optimizer: torch.optim.Optimizer,
    param: nn.Parameter,
    mask_4d: Optional[torch.Tensor],
    freeze_all: bool = False,
) -> int:
    """Mask or zero the momentum buffer for a parameter.
    
    For SGD with momentum, the state dict has 'momentum_buffer'.
    For Adam/AdamW, the state dict has 'exp_avg' (first moment) and 'exp_avg_sq' (second moment).
    
    Args:
        optimizer: The optimizer containing the state.
        param: The parameter whose momentum buffer to mask.
        mask_4d: The mask to apply (1.0 for positions to keep, 0.0 to zero).
                 If None and freeze_all=True, zeros the entire buffer.
        freeze_all: If True, zero the entire buffer regardless of mask.
    
    Returns:
        1 if a buffer was masked, 0 otherwise.
    """
    if param not in optimizer.state:
        return 0
    
    state = optimizer.state[param]
    masked_count = 0
    
    # SGD momentum buffer
    if 'momentum_buffer' in state:
        if freeze_all:
            state['momentum_buffer'].zero_()
        elif mask_4d is not None:
            state['momentum_buffer'].mul_(mask_4d)
        masked_count = 1
    
    # Adam first moment (momentum equivalent)
    if 'exp_avg' in state:
        if freeze_all:
            state['exp_avg'].zero_()
        elif mask_4d is not None:
            state['exp_avg'].mul_(mask_4d)
        masked_count = 1
    
    # Adam second moment (RMSprop-like term)
    if 'exp_avg_sq' in state:
        if freeze_all:
            state['exp_avg_sq'].zero_()
        elif mask_4d is not None:
            state['exp_avg_sq'].mul_(mask_4d)
        masked_count = 1
    
    return masked_count


def apply_gradient_mask_to_cache(
    cache: nn.Module,
    mask: GradientMask,
    n_layers: int,
):
    """Zero gradients for non-top-t value positions after backward.

    This replaces the paper's pre-forward gradient trick, eliminating the 1-step lag.
    Because the mask is applied to already-computed .grad tensors, the selected
    positions are derived from the CURRENT batch's attention scores.

    Works correctly with gradient accumulation: all microbatch gradients have
    already been summed into .grad by the time this is called (at do_step).

    Args:
        cache: TrainableCache with trainable_values as nn.ParameterList.
        mask: GradientMask specifying which positions keep their gradient.
        n_layers: number of model layers.
    """
    device = cache.trainable_keys[0].device
    trainable_value_layers = _trainable_value_layer_indices(n_layers)
    masked_count = 0

    for layer_idx in range(n_layers):
        if layer_idx not in trainable_value_layers:
            continue
        v_param = cache.trainable_values[layer_idx]
        if v_param.grad is None:
            continue
        n_kv_heads = v_param.shape[1]
        keep_4d = mask.build_4d_mask(layer_idx, n_kv_heads, device)
        v_param.grad.mul_(keep_4d.to(v_param.grad.dtype))
        masked_count += 1

    logger.info(
        f"apply_gradient_mask_to_cache: {masked_count} value layers masked, "
        f"top_t={mask.top_t}, granularity={mask.granularity}"
    )


def _apply_rotary_pos_emb(q, cos, sin):
    """Apply RoPE to query tensor. Identical logic for both Llama and Qwen3."""
    cos = cos.unsqueeze(1)  # unsqueeze_dim=1 for (batch, heads, seq, dim)
    sin = sin.unsqueeze(1)
    x1 = q[..., : q.shape[-1] // 2]
    x2 = q[..., q.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (q * cos) + (rotated * sin)


def install_query_capture_hooks(model: nn.Module) -> tuple[dict, list]:
    """Register forward hooks on attention layers to capture post-RoPE queries.

    Works with both FlexLlama and FlexQwen3 model architectures.
    Hooks fire after each attention module's forward, recomputing the query
    projection from the *input* hidden states (captured via the hook's input
    args) under torch.no_grad(). This avoids interfering with the compiled
    flex_attention kernel.

    Returns:
        captured: dict that will be populated with {layer_idx: query_tensor}
            during each forward pass.
        handles: list of hook handles (call .remove() to uninstall).
    """
    captured: dict[int, torch.Tensor] = {}
    handles = []

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        raise ValueError(f"Cannot find decoder layers in model of type {type(model)}")

    for layer_idx, decoder_layer in enumerate(layers):
        attn_module = decoder_layer.self_attn

        def make_hook(idx, attn_mod):
            def hook_fn(module, args, output):
                batch = args[0] if args else None
                if batch is None:
                    return

                hidden_states = batch.hidden_states
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, attn_mod.head_dim)

                with torch.no_grad():
                    if hasattr(attn_mod, "q_norm"):
                        q = attn_mod.q_norm(
                            attn_mod.q_proj(hidden_states).view(hidden_shape)
                        ).transpose(1, 2)
                    else:
                        q = attn_mod.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    cos, sin = batch.position_embeddings
                    q = _apply_rotary_pos_emb(q, cos, sin)

                captured[idx] = q.detach()

            return hook_fn

        h = attn_module.register_forward_hook(make_hook(layer_idx, attn_module))
        handles.append(h)

    return captured, handles


def collect_background_stats(
    wrapped_model: nn.Module,
    cache: nn.Module,
    dataloader,
    config: SparseCacheFinetuningConfig,
    local_rank,
    save_path: str,
    is_ddp: bool = False,
):
    """Run forward passes over a dataloader to collect background access stats.

    Iterates the dataloader (up to config.num_background_batches), running
    forward-only passes and accumulating attention-based cache access scores.
    Saves a BackgroundAccessTracker to save_path.

    Call this after training completes to produce the IDF statistics needed
    for use_idf=True in subsequent training phases. The granularity of the
    saved stats matches config.granularity.
    """
    from cartridges.datasets import DatasetBatch

    n_trainable = cache._num_trainable_tokens
    n_layers = len(cache.trainable_keys)
    n_kv_heads = cache.trainable_keys[0].shape[1]

    tracker = CacheAccessTracker(
        n_trainable_tokens=n_trainable,
        device=local_rank,
        granularity=config.granularity,
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
    )
    bg_tracker = BackgroundAccessTracker(
        num_batches=config.num_background_batches,
        granularity=config.granularity,
    )
    scaling = cache.config.head_dim ** -0.5

    _cam = wrapped_model.module if is_ddp else wrapped_model
    had_hooks = _cam._sparse_enabled
    if not had_hooks:
        captured, handles = install_query_capture_hooks(_cam.model)
        _cam._captured_queries = captured
    else:
        captured = _cam._captured_queries

    logger.info(
        f"Collecting background stats: {config.num_background_batches} batches"
    )

    wrapped_model.eval()
    batch_count = 0
    with torch.no_grad():
        for batch in dataloader:
            if batch_count >= config.num_background_batches:
                break

            batch: DatasetBatch
            if captured is not None:
                captured.clear()

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                wrapped_model(
                    input_ids=batch.input_ids.to(local_rank),
                    seq_ids=batch.element_ids.to(local_rank),
                    position_ids=batch.position_ids.to(local_rank),
                )

            if captured:
                tracker.accumulate_from_hooks(
                    captured, cache,
                    scaling=scaling,
                    seq_ids=batch.element_ids.to(local_rank),
                )

            bg_tracker.add_batch_scores(tracker.get_access_counts())
            tracker.reset()
            cache.clear()
            batch_count += 1

            if (batch_count) % 100 == 0:
                logger.info(f"Background collection: {batch_count}/{config.num_background_batches}")

    if not had_hooks:
        for h in handles:
            h.remove()
        _cam._captured_queries = None

    wrapped_model.train()
    bg_tracker.save(save_path)
    logger.info(f"Background stats saved to {save_path} ({batch_count} batches)")
