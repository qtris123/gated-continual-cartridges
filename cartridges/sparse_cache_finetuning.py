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

import math
from collections import defaultdict
from logging import getLogger
from typing import Dict, Optional, Set

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


class CacheAccessTracker:
    """Accumulates per-position attention scores across microbatches.

    After each forward pass, call `accumulate()` to add the attention scores
    from the last token of each packed sequence to the running totals.
    On `do_step`, call `get_access_counts()` then `reset()`.
    """

    def __init__(self, n_trainable_tokens: int, device: torch.device):
        self.n_trainable_tokens = n_trainable_tokens
        self.device = device
        self._scores = torch.zeros(n_trainable_tokens, device=device)
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
            # seq_ids looks like [0,0,0,1,1,1,2,2,2,2,pad,pad] where pad has
            # the same id as the previous real token after packing truncation.
            # We detect boundaries where the id changes or is the very last position.
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

                # scores: (1, n_kv_heads, n_seqs, n_trainable_tokens)
                scores = torch.matmul(q_mean, cache_k.transpose(-1, -2)) * scaling
                attn_weights = scores.softmax(dim=-1)

                # Sum across sequences and heads → (n_trainable_tokens,)
                self._scores += attn_weights.sum(dim=(0, 1, 2))

        self._num_accumulations += 1

    def get_access_counts(self) -> torch.Tensor:
        """Returns accumulated scores (on device). Shape: (n_trainable_tokens,)."""
        return self._scores

    def reset(self):
        self._scores.zero_()
        self._num_accumulations = 0


class BackgroundAccessTracker:
    """Tracks which cache positions are accessed across background corpus batches.

    For each cache token position, records the set of batch IDs where that
    position appeared in the top-attended set. This builds the document
    frequency (DF) needed for IDF computation.
    """

    def __init__(self, num_batches: int = 1000, top_k_per_batch: int = 128):
        self.num_batches = num_batches
        self.top_k_per_batch = top_k_per_batch
        self.position_to_batches: Dict[int, Set[int]] = defaultdict(set)
        self.current_batch_id = 0

    def add_batch_scores(self, scores: torch.Tensor):
        """Record which positions were most attended in this batch.

        Args:
            scores: (n_trainable_tokens,) attention-based access scores.
        """
        if self.current_batch_id >= self.num_batches:
            return

        k = min(self.top_k_per_batch, len(scores))
        _, top_positions = torch.topk(scores, k=k, largest=True)
        for pos in top_positions.cpu().tolist():
            self.position_to_batches[pos].add(self.current_batch_id)

        self.current_batch_id += 1

    def get_document_frequency(self, position: int) -> int:
        return len(self.position_to_batches.get(position, set()))

    def save(self, path: str):
        torch.save(
            {
                "position_to_batches": {
                    k: list(v) for k, v in self.position_to_batches.items()
                },
                "num_batches": self.current_batch_id,
                "top_k_per_batch": self.top_k_per_batch,
            },
            path,
        )
        logger.info(f"Saved background access stats to {path}")

    def load(self, path: str):
        data = torch.load(path, weights_only=False)
        self.position_to_batches = defaultdict(
            set, {k: set(v) for k, v in data["position_to_batches"].items()}
        )
        self.current_batch_id = data["num_batches"]
        self.top_k_per_batch = data.get("top_k_per_batch", 128)
        logger.info(
            f"Loaded background stats: {len(self.position_to_batches)} positions "
            f"from {self.current_batch_id} batches"
        )


class CacheTFIDFRanker:
    """Ranks cache positions by TF-IDF score.

    TF = normalized attention score for position i in current global batch.
    IDF = log((|B| + s) / (df(i) + s)) where |B| is the number of background
    batches and df(i) is how many background batches had position i in top-k.
    """

    def __init__(
        self,
        background_tracker: Optional[BackgroundAccessTracker] = None,
        use_idf: bool = True,
        smoothing: float = 1.0,
    ):
        self.background_tracker = background_tracker
        self.use_idf = use_idf and background_tracker is not None
        self.smoothing = smoothing
        self.num_background_batches = (
            background_tracker.current_batch_id if background_tracker else 0
        )

    def rank_positions(
        self,
        access_scores: torch.Tensor,
        top_t: int,
    ) -> torch.Tensor:
        """Returns top-t position indices sorted by TF-IDF score (highest first).

        Args:
            access_scores: (n_trainable_tokens,) accumulated attention scores.
            top_t: number of positions to select.

        Returns:
            (top_t,) tensor of position indices on CPU.
        """
        scores = access_scores.float().cpu()
        total = scores.sum()
        if total == 0:
            return torch.arange(min(top_t, len(scores)))

        tf = scores / total

        if self.use_idf:
            n = len(scores)
            idf = torch.zeros(n)
            for i in range(n):
                df = self.background_tracker.get_document_frequency(i)
                idf[i] = math.log(
                    (self.num_background_batches + self.smoothing)
                    / (df + self.smoothing)
                )
            tfidf = tf * idf
        else:
            tfidf = tf

        k = min(top_t, len(tfidf))
        _, top_positions = torch.topk(tfidf, k=k, largest=True)
        return top_positions


def mask_cache_gradients(
    cache: nn.Module,
    top_positions: torch.Tensor,
    n_layers: int,
):
    """Zero out gradients for cache positions NOT in top_positions.

    Operates in-place on .grad of cache.trainable_keys and cache.trainable_values.

    Args:
        cache: TrainableCache with trainable_keys/values as nn.ParameterList.
        top_positions: (top_t,) indices of positions to keep.
        n_layers: number of layers.
    """
    n_tokens = cache._num_trainable_tokens
    device = cache.trainable_keys[0].device

    mask = torch.zeros(n_tokens, device=device)
    mask[top_positions.to(device)] = 1.0
    # Broadcast shape: (1, 1, n_tokens, 1) for (batch, heads, tokens, head_dim)
    mask_4d = mask[None, None, :, None]

    for layer_idx in range(n_layers):
        k_param = cache.trainable_keys[layer_idx]
        v_param = cache.trainable_values[layer_idx]
        if k_param.grad is not None:
            k_param.grad.mul_(mask_4d)
        if v_param.grad is not None:
            v_param.grad.mul_(mask_4d)


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
    for use_idf=True in subsequent training phases.
    """
    from cartridges.datasets import DatasetBatch

    n_trainable = cache._num_trainable_tokens
    tracker = CacheAccessTracker(n_trainable_tokens=n_trainable, device=local_rank)
    bg_tracker = BackgroundAccessTracker(
        num_batches=config.num_background_batches,
        top_k_per_batch=config.background_top_k_per_batch,
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
        f"Collecting background stats: {config.num_background_batches} batches, "
        f"top_k_per_batch={config.background_top_k_per_batch}"
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
