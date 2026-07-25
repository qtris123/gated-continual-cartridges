"""Query/target accumulators and teacher-attention capture hooks for AM."""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

from cartridges.models.attention import flex_attention_forward
from cartridges.sparse_cache_finetuning import (
    _apply_rotary_pos_emb,
    install_query_capture_hooks,
)

__all__ = [
    "AMQueryAccumulator",
    "AMTargetAccumulator",
    "install_teacher_attention_capture_hooks",
    "install_query_capture_hooks",
]


class AMQueryAccumulator:
    """Accumulates reference queries from captured hooks across microbatches."""

    def __init__(
        self,
        granularity: Literal["global", "per_layer", "per_head"] = "per_layer",
        queries_per_batch: Literal["last_token", "all_tokens"] = "all_tokens",
        n_layers: int = 1,
        n_kv_heads: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        self.granularity = granularity
        self.queries_per_batch = queries_per_batch
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.device = device
        # {layer_idx: list of (n_q_heads, n_tokens, head_dim) tensors}
        self._queries: dict[int, list[torch.Tensor]] = {l: [] for l in range(n_layers)}
        self._access_scores_layer: Optional[torch.Tensor] = None

    def reset(self):
        self._queries = {l: [] for l in range(self.n_layers)}
        self._access_scores_layer = None

    def accumulate_from_hooks(
        self,
        captured_queries: dict[int, torch.Tensor],
        cache: nn.Module,
        scaling: float,
        seq_ids: torch.Tensor,
    ):
        """Extract queries and TF scores from captured post-RoPE queries."""
        with torch.no_grad():
            unique_ids = seq_ids.unique()
            if self.queries_per_batch == "last_token":
                token_indices = []
                for uid in unique_ids:
                    positions = (seq_ids == uid).nonzero(as_tuple=True)[0]
                    token_indices.append(positions[-1].item())
                token_indices_t = torch.tensor(token_indices, device=seq_ids.device)
            else:
                token_indices_t = None

            for layer_idx, q in captured_queries.items():
                if layer_idx >= self.n_layers:
                    continue
                # q: (1, n_q_heads, seq_len, head_dim)
                if self.queries_per_batch == "last_token":
                    q_sel = q[:, :, token_indices_t, :]
                else:
                    q_sel = q

                self._queries[layer_idx].append(q_sel.detach().cpu())

                # TF scores for TF-IDF ranking (same as CacheAccessTracker per_layer)
                cache_k = cache.trainable_keys[layer_idx]
                n_kv_heads = cache_k.shape[1]
                n_q_heads = q.shape[1]
                groups = n_q_heads // n_kv_heads

                if self.queries_per_batch == "last_token":
                    q_for_score = q[:, :, token_indices_t, :]
                else:
                    q_for_score = q

                q_grouped = q_for_score.view(1, n_kv_heads, groups, -1, q_for_score.shape[-1])
                q_mean = q_grouped.mean(dim=2)  # (1, n_kv_heads, n_tokens, d)
                raw_scores = torch.matmul(q_mean, cache_k.transpose(-1, -2)) * scaling
                attn_weights = raw_scores.softmax(dim=-1)
                if self.granularity == "per_head":
                    # Sum batch and query tokens, preserving each KV head.
                    # Shape: (n_kv_heads, n_cache_tokens).
                    layer_score = attn_weights.sum(dim=(0, 2))
                else:
                    layer_score = attn_weights.sum(dim=(0, 1, 2))

                if self._access_scores_layer is None:
                    if self.granularity == "per_head":
                        self._access_scores_layer = torch.zeros(
                            self.n_layers,
                            self.n_kv_heads,
                            layer_score.shape[-1],
                            device=self.device,
                        )
                    else:
                        self._access_scores_layer = torch.zeros(
                            self.n_layers, layer_score.shape[-1], device=self.device
                        )
                self._access_scores_layer[layer_idx] += layer_score.to(self.device)

    def get_access_scores(self) -> torch.Tensor:
        """TF scores shaped for TF-IDF ranker."""
        if self.granularity == "global":
            return self._access_scores_layer.sum(dim=0)
        elif self.granularity == "per_layer":
            return self._access_scores_layer
        else:
            return self._access_scores_layer

    def get_layer_head_queries(
        self,
        layer_idx: int,
        head_idx: int,
        n_q_heads: int,
        n_kv_heads: int,
    ) -> torch.Tensor:
        """Return (n, head_dim) queries for a specific KV head."""
        if not self._queries[layer_idx]:
            return torch.zeros(0, 0)

        groups = n_q_heads // n_kv_heads
        head_qs = []
        for q_batch in self._queries[layer_idx]:
            # q_batch: (1, n_q_heads, n_tokens, head_dim)
            q_h = q_batch[0, head_idx * groups : (head_idx + 1) * groups]
            q_h = q_h.reshape(-1, q_h.shape[-1])  # (n_tokens * groups, d)
            head_qs.append(q_h)

        if not head_qs:
            return torch.zeros(0, 0)
        return torch.cat(head_qs, dim=0)


class AMTargetAccumulator:
    """Accumulates per-query, per-head AM targets in the same order as queries."""

    def __init__(
        self,
        n_layers: int = 1,
        n_kv_heads: int = 1,
    ):
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self._targets: dict[int, list[torch.Tensor]] = {l: [] for l in range(n_layers)}

    def reset(self):
        self._targets = {l: [] for l in range(self.n_layers)}

    def accumulate_from_hooks(self, captured_targets: dict[int, torch.Tensor]):
        with torch.no_grad():
            for layer_idx, target in captured_targets.items():
                if layer_idx >= self.n_layers:
                    continue
                # target: (1, n_q_heads, n_tokens, head_dim)
                self._targets[layer_idx].append(target.detach().cpu())

    def get_layer_head_targets(
        self,
        layer_idx: int,
        head_idx: int,
        n_q_heads: int,
        n_kv_heads: int,
    ) -> torch.Tensor:
        """Return (n, head_dim) targets aligned to AMQueryAccumulator order."""
        if not self._targets[layer_idx]:
            return torch.zeros(0, 0)

        groups = n_q_heads // n_kv_heads
        head_targets = []
        for target_batch in self._targets[layer_idx]:
            target_h = target_batch[0, head_idx * groups : (head_idx + 1) * groups]
            target_h = target_h.reshape(-1, target_h.shape[-1])
            head_targets.append(target_h)

        if not head_targets:
            return torch.zeros(0, 0)
        return torch.cat(head_targets, dim=0)


def install_teacher_attention_capture_hooks(model: nn.Module) -> tuple[dict, list]:
    """Capture no-cache teacher attention outputs before the output projection.

    The sparse AM value solve operates per KV head in head-dim space. Capturing
    pre-``o_proj`` outputs keeps the target dimension compatible with the solver.
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
                        query_states = attn_mod.q_norm(
                            attn_mod.q_proj(hidden_states).view(hidden_shape)
                        ).transpose(1, 2)
                        key_states = attn_mod.k_norm(
                            attn_mod.k_proj(hidden_states).view(hidden_shape)
                        ).transpose(1, 2)
                    else:
                        query_states = attn_mod.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                        key_states = attn_mod.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_states = attn_mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    cos, sin = batch.position_embeddings
                    query_states = _apply_rotary_pos_emb(query_states, cos, sin)
                    key_states = _apply_rotary_pos_emb(key_states, cos, sin)

                    attn_output = flex_attention_forward(
                        attn_mod,
                        query_states,
                        key_states,
                        value_states,
                        attention_mask=batch.attention_mask,
                        scaling=attn_mod.scaling,
                        mode=batch.mode,
                    )
                    # flex_attention_forward returns (batch, tokens, q_heads, head_dim).
                    captured[idx] = attn_output.transpose(1, 2).detach()

            return hook_fn

        h = attn_module.register_forward_hook(make_hook(layer_idx, attn_module))
        handles.append(h)

    return captured, handles
