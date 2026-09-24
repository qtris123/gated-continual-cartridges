"""Teacher stage: the target the write is fitted against.

Design A for Phase 2: the teacher is the concatenation ``[cartridge || ICL(doc)]``,
so the write asks "what would the model output if it could still see the
document?" rather than "reproduce yourself".
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from pydrantic import ObjectConfig

from cartridges.am.components.queries import document_key, full_document_prompt
from cartridges.am.core import _attention_scores, compute_attention_output
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.initialization.tokenization_utils import MODEL_TO_SYSTEM_PROMPT_TOKENIZER
from cartridges.sparse_cache_finetuning import _apply_rotary_pos_emb


def eval_aligned_system_ids(tokenizer, content: str) -> torch.Tensor:
    """Token ids of ``content`` as the eval chat template's system message.

    Llama's ``apply_chat_template`` inserts a BOS token plus a dated preamble
    (``Cutting Knowledge Date`` / ``Today Date``). Eval questions are tokenized
    with the message converter, which does not. Document keys have to use that
    same prefix or the continual write fits a different prompt than phase 1 and
    eval.
    """
    from cartridges.datasets import MODEL_TO_MESSAGE_CONVERTER
    from cartridges.structs import Conversation

    converter = MODEL_TO_MESSAGE_CONVERTER[tokenizer.name_or_path.lower()]

    def _msg(role: str, text: str) -> Conversation.Message:
        return Conversation.Message(content=text, role=role, token_ids=None)

    joint = converter(
        [_msg("system", content), _msg("user", "."), _msg("assistant", ".")],
        retokenize=True,
        tokenizer=tokenizer,
    )
    tail = converter(
        [_msg("user", "."), _msg("assistant", ".")],
        retokenize=True,
        tokenizer=tokenizer,
    )
    ids = joint.input_ids.reshape(-1)
    tail_ids = tail.input_ids.reshape(-1)
    if (
        tail_ids.numel() == 0
        or ids.numel() <= tail_ids.numel()
        or not torch.equal(ids[-tail_ids.numel() :], tail_ids)
    ):
        raise RuntimeError(
            "Eval chat template system turn is not a stable prefix; "
            "cannot align in-context targets with question-only queries."
        )
    return ids[: ids.numel() - tail_ids.numel()].contiguous()


def _tokenize_system_prompt(
    tokenizer,
    system_prompt: str,
    max_tokens: Optional[int] = None,
) -> torch.Tensor:
    model_key = tokenizer.name_or_path.lower()
    from cartridges.datasets import MODEL_TO_MESSAGE_CONVERTER

    if model_key in MODEL_TO_MESSAGE_CONVERTER:
        # Same system turn eval and phase 1 use. The legacy chat-template
        # tokenizer adds a BOS token and a date preamble the questions never see.
        ids = eval_aligned_system_ids(tokenizer, system_prompt)
    else:
        if model_key not in MODEL_TO_SYSTEM_PROMPT_TOKENIZER:
            raise ValueError(
                f"No system-prompt tokenizer registered for {tokenizer.name_or_path}"
            )
        ids = MODEL_TO_SYSTEM_PROMPT_TOKENIZER[model_key](
            tokenizer=tokenizer, content=system_prompt, max_tokens=max_tokens
        ).reshape(-1)
        return ids
    if max_tokens is not None and ids.numel() > max_tokens:
        ids = torch.cat([ids[: max_tokens - 1], ids[-1:]])
    return ids.reshape(1, -1)


def prefill_document_kv_cache(
    model: nn.Module,
    tokenizer,
    system_prompt: str,
    attn_config: AttnConfig,
    device: torch.device,
    cartridge_cache: Optional[TrainableCache] = None,
    max_tokens: Optional[int] = None,
    position_offset: int = 0,
    capture_device: Optional[torch.device] = None,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Capture document KV while prefilling against the current cartridge.

    Returns per-layer (K_D, V_D) with shape (n_kv_heads, T_doc, head_dim), post-RoPE.

    ``position_offset`` shifts the RoPE positions of this document so multiple
    documents can be prefilled separately yet carry globally-consistent absolute
    positions (avoids cross-document position collisions when concatenated).

    ``capture_device`` receives each layer's KV as it comes off the hook. Left as
    ``None`` the KV stays where the model computed it, which means all
    ``n_layers`` of it is resident alongside the forward pass (1.3 GiB for a 8.9k
    token paper on Qwen3-4B); callers that only want it off-device should say so.
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
                    k_out = key_states[0].detach()
                    v_out = value_states[0].detach()
                    if capture_device is not None:
                        k_out = k_out.to(capture_device)
                        v_out = v_out.to(capture_device)
                    captured[idx] = (k_out, v_out)

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
                    # KV comes off the hooks; the vocab projection is discarded. Keeping
                    # every position costs seq_len x vocab (2.7 GiB for a 8.9k-token
                    # QASPER paper on Qwen3) at the same moment the doc KV is live.
                    logits_to_keep=1,
                )
    finally:
        for handle in handles:
            handle.remove()

    for layer_idx in range(attn_config.n_layers):
        if layer_idx not in captured:
            raise RuntimeError(f"Document prefill produced no KV at layer {layer_idx}")
    return captured


def cache_from_layer_kv(
    attn_config: AttnConfig,
    layer_kv: dict[int, tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device | str,
) -> TrainableCache:
    """Build a cartridge whose tokens are the captured per-layer KV.

    ``layer_kv[layer]`` is ``(K, V)`` with shape ``(n_kv_heads, T, head_dim)``
    or ``(1, n_kv_heads, T, head_dim)``. Sequence ids are the cartridge id, so
    every query can attend to every token.
    """
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    for layer_idx in range(attn_config.n_layers):
        k, v = layer_kv[layer_idx]
        if k.dim() == 3:
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
        keys.append(k.detach().to(device=device).contiguous())
        values.append(v.detach().to(device=device).contiguous())
    return TrainableCache(
        config=attn_config,
        init_keys=keys,
        init_values=values,
        num_frozen_tokens=0,
    ).to(device)


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


class TeacherTarget:
    """Builds the ``[cartridge || doc]`` teacher and reads targets off it.

    Two phases: ``prefill`` captures the document KV once per document, then
    ``targets`` / ``log_mass`` evaluate that teacher per (layer, head). The
    rotary base is NOT a field here -- three stages consume it, so it lives on
    ``AMContinualConfig.rope_theta`` and is passed in (MECH-003 / MECH-005).
    """

    class Config(ObjectConfig):
        _pass_as_config = True

        # Backward-compatible default: all historical configs are QASPER.
        dataset: str = "qasper"
        # Which QASPER topic to resolve document titles against. `all` searches
        # every topic in `TOPIC_TO_IDS` (titles are unique across them).
        qasper_topic: str = "MT"
        # Required for QuALITY so a title cannot silently resolve in another phase.
        quality_phase: Optional[int] = None
        # Five-phase index for the phase-keyed datasets (finqa, techqa).
        phase: Optional[int] = None

    def __init__(self, config: Config):
        self.config = config

    def document_prompt(self, conversations: list) -> str:
        """The complete document text for one synthesis document group."""
        return full_document_prompt(
            document_key(conversations[0], self.config.dataset),
            dataset=self.config.dataset,
            qasper_topic=self.config.qasper_topic,
            quality_phase=self.config.quality_phase,
            phase=self.config.phase,
        )

    def prefill(
        self,
        model: nn.Module,
        tokenizer,
        system_prompt: str,
        *,
        attn_config: AttnConfig,
        device: torch.device,
        cartridge_cache: Optional[TrainableCache] = None,
        max_tokens: Optional[int] = None,
        position_offset: int = 0,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        return prefill_document_kv_cache(
            model=model,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            attn_config=attn_config,
            device=device,
            cartridge_cache=cartridge_cache,
            max_tokens=max_tokens,
            position_offset=position_offset,
        )

    def targets(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        head_dim: int,
        *,
        attention_bias: Optional[torch.Tensor] = None,
        n_cartridge_keys: Optional[int] = None,
        doc_rope_offset: Optional[int] = None,
        rope_theta: float,
    ) -> torch.Tensor:
        return compute_teacher_targets(
            queries,
            keys,
            values,
            head_dim,
            attention_bias=attention_bias,
            n_cartridge_keys=n_cartridge_keys,
            doc_rope_offset=doc_rope_offset,
            rope_theta=rope_theta,
        )

    def log_mass(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        head_dim: int,
        *,
        attention_bias: Optional[torch.Tensor] = None,
        n_cartridge_keys: Optional[int] = None,
        doc_rope_offset: Optional[int] = None,
        rope_theta: float,
    ) -> torch.Tensor:
        """The mass flavour of the same target, consumed by ``BetaFitter.fit``."""
        return compute_teacher_log_mass(
            queries,
            keys,
            head_dim,
            attention_bias=attention_bias,
            n_cartridge_keys=n_cartridge_keys,
            doc_rope_offset=doc_rope_offset,
            rope_theta=rope_theta,
        )
