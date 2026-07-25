"""Teacher KV construction for AM Phase 2 (Design A: [cartridge || ICL(doc)])."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from cartridges.am.core import (
    _attention_scores,
    compute_attention_output,
)
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.initialization.tokenization_utils import MODEL_TO_SYSTEM_PROMPT_TOKENIZER
from cartridges.sparse_cache_finetuning import _apply_rotary_pos_emb


def _tokenize_system_prompt(
    tokenizer,
    system_prompt: str,
    max_tokens: Optional[int] = None,
) -> torch.Tensor:
    model_key = tokenizer.name_or_path.lower()
    if model_key not in MODEL_TO_SYSTEM_PROMPT_TOKENIZER:
        raise ValueError(f"No system-prompt tokenizer registered for {tokenizer.name_or_path}")
    fn = MODEL_TO_SYSTEM_PROMPT_TOKENIZER[model_key]
    return fn(tokenizer=tokenizer, content=system_prompt, max_tokens=max_tokens)


def prefill_document_kv_cache(
    model: nn.Module,
    tokenizer,
    system_prompt: str,
    attn_config: AttnConfig,
    device: torch.device,
    cartridge_cache: Optional[TrainableCache] = None,
    max_tokens: Optional[int] = None,
    position_offset: int = 0,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Capture document KV while prefilling against the current cartridge.

    Returns per-layer (K_D, V_D) with shape (n_kv_heads, T_doc, head_dim), post-RoPE.

    ``position_offset`` shifts the RoPE positions of this document so multiple
    documents can be prefilled separately yet carry globally-consistent absolute
    positions (avoids cross-document position collisions when concatenated).
    """
    device = torch.device(device)
    input_ids = _tokenize_system_prompt(tokenizer, system_prompt, max_tokens=max_tokens)
    input_ids = input_ids.squeeze(0).to(device)
    seq_len = input_ids.shape[-1]

    captured: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    handles = []
    decoder = model.model if hasattr(model, "model") else model
    if not hasattr(decoder, "layers"):
        raise ValueError(f"Cannot find decoder layers in model of type {type(model)}")

    for layer_idx, decoder_layer in enumerate(decoder.layers):
        attn_module = decoder_layer.self_attn

        def make_hook(idx, attn_mod):
            def hook_fn(module, args, output):
                batch = args[0] if args else None
                if batch is None:
                    return
                hidden_states = batch.hidden_states
                hidden_shape = (*hidden_states.shape[:-1], -1, attn_mod.head_dim)
                with torch.no_grad():
                    if hasattr(attn_mod, "k_norm"):
                        key_states = attn_mod.k_norm(
                            attn_mod.k_proj(hidden_states).view(hidden_shape)
                        ).transpose(1, 2)
                    else:
                        key_states = attn_mod.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_states = attn_mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    cos, sin = batch.position_embeddings
                    key_states = _apply_rotary_pos_emb(key_states, cos, sin)
                    captured[idx] = (
                        key_states[0].detach(),
                        value_states[0].detach(),
                    )

            return hook_fn

        handles.append(attn_module.register_forward_hook(make_hook(layer_idx, attn_module)))

    try:
        with torch.no_grad():
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                model(
                    input_ids=input_ids,
                    seq_ids=torch.zeros(seq_len, dtype=torch.long, device=device),
                    position_ids=torch.arange(
                        position_offset, position_offset + seq_len,
                        dtype=torch.long, device=device,
                    ),
                    use_cache=cartridge_cache is not None,
                    past_key_values=cartridge_cache,
                    mode="train",
                )
    finally:
        for handle in handles:
            handle.remove()

    for layer_idx in range(attn_config.n_layers):
        if layer_idx not in captured:
            raise RuntimeError(f"Document prefill produced no KV at layer {layer_idx}")
    return captured


def concat_teacher_kv(
    k_cartridge: torch.Tensor,
    v_cartridge: torch.Tensor,
    k_doc: torch.Tensor,
    v_doc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate cartridge and document KV along the token dimension.

    Args:
        k_cartridge, v_cartridge: (T_c, d)
        k_doc, v_doc: (T_d, d) for a single KV head
    """
    k_teacher = torch.cat([k_cartridge, k_doc], dim=0)
    v_teacher = torch.cat([v_cartridge, v_doc], dim=0)
    return k_teacher, v_teacher


def compute_teacher_targets(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    head_dim: int,
    attention_bias: Optional[torch.Tensor] = None,
    n_cartridge_keys: Optional[int] = None,
    doc_rope_offset: Optional[int] = None,
    rope_theta: float = 10000.0,
) -> torch.Tensor:
    """Attention outputs Attn(q; keys, values) for teacher reference queries."""
    return compute_attention_output(
        queries,
        keys,
        values,
        head_dim,
        attention_bias=attention_bias,
        doc_key_start=n_cartridge_keys,
        doc_rope_offset=doc_rope_offset,
        rope_theta=rope_theta,
    )


def compute_teacher_log_mass(
    queries: torch.Tensor,
    keys: torch.Tensor,
    head_dim: int,
    attention_bias: Optional[torch.Tensor] = None,
    n_cartridge_keys: Optional[int] = None,
    doc_rope_offset: Optional[int] = None,
    rope_theta: float = 10000.0,
) -> torch.Tensor:
    """Log unnormalized attention mass for stable, consistently-scaled NNLS."""
    scores = _attention_scores(
        queries,
        keys,
        head_dim,
        doc_key_start=n_cartridge_keys,
        doc_rope_offset=doc_rope_offset,
        rope_theta=rope_theta,
    )
    if attention_bias is not None:
        scores = scores + attention_bias.to(torch.float32)
    return torch.logsumexp(scores, dim=-1)


def compute_teacher_mass(
    queries: torch.Tensor,
    keys: torch.Tensor,
    head_dim: int,
    attention_bias: Optional[torch.Tensor] = None,
    n_cartridge_keys: Optional[int] = None,
    doc_rope_offset: Optional[int] = None,
    rope_theta: float = 10000.0,
) -> torch.Tensor:
    """Unnormalized attention mass; prefer log mass for fitting."""
    return torch.exp(
        compute_teacher_log_mass(
            queries,
            keys,
            head_dim,
            attention_bias=attention_bias,
            n_cartridge_keys=n_cartridge_keys,
            doc_rope_offset=doc_rope_offset,
            rope_theta=rope_theta,
        )
    )
