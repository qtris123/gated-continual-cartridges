"""Phase 1: build a cartridge by compacting a teacher KV cache (classic AM).

Two layers live here. The C1/C2 building blocks solve one KV head::

    softmax(qK^T)V  ~=  softmax(qC1^T + beta) C2

and ``compact_cache_am_phase1`` drives them over a whole corpus: prefill every
QA document into a teacher KV, collect the reference queries the compacted
cartridge must serve, then select ``num_tokens`` teacher keys per (layer, head)
and ridge-fit the values that reproduce the teacher's attention output.
"""

from __future__ import annotations

import dataclasses
import time
from logging import getLogger
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F

from cartridges.am.components.keys import (
    select_keys_highest_attention,
    select_keys_omp,
)
from cartridges.am.components.queries import (
    AMQueryAccumulator,
    AMTargetAccumulator,
    build_reference_dataloader,
    cleanup_reference_parquet,
    full_document_prompt,
    group_conversations_by_document,
    install_teacher_attention_capture_hooks,
    limit_conversations,
    load_conversations,
)
from cartridges.am.components.teacher import cache_from_layer_kv, prefill_document_kv_cache
from cartridges.datasets import MODEL_TO_MESSAGE_CONVERTER
from cartridges.structs import Conversation
from cartridges.am.core import (
    _inv_sqrt_d,
    _ridge_lstsq,
    _rope_reposition,
    bind_model_rope,
    compute_attention_output,
    compute_attention_weights,
)
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.sparse_cache_finetuning import (
    _apply_rotary_pos_emb,
    install_query_capture_hooks,
)

logger = getLogger(__name__)


# ======================================================================================
# C1 / C2 building blocks
# ======================================================================================
def naive_compaction_c2_update(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    selected_indices: torch.Tensor,
    head_dim: Optional[int] = None,
    ridge_lambda: float = 1e-4,
) -> Tuple[torch.Tensor, dict]:
    """WRONG approach for sparse-in-full-cache: attention over subset only.

    Included for unit tests demonstrating why compaction C2 must not be used
    directly for sparse finetuning.
    """
    if head_dim is None:
        head_dim = keys.shape[-1]

    device = values.device
    dtype = values.dtype
    selected_indices = selected_indices.to(device=device, dtype=torch.long)

    C1 = keys[selected_indices]
    t = C1.shape[0]
    n = queries.shape[0]

    inv_sqrt_d = _inv_sqrt_d(head_dim)
    scores = (queries @ C1.T).to(torch.float32) * inv_sqrt_d
    X = F.softmax(scores, dim=-1)  # (n, t) — WRONG: ignores other keys

    targets = compute_attention_output(queries, keys, values, head_dim)

    V_sel_new = _ridge_lstsq(X, targets, ridge_lambda=ridge_lambda)

    new_values = values.clone()
    sel_mask = torch.zeros(keys.shape[0], dtype=torch.bool, device=device)
    sel_mask[selected_indices] = True
    new_values[sel_mask] = V_sel_new.to(dtype)

    alpha_new = compute_attention_weights(queries, keys, head_dim)
    output_new = alpha_new @ new_values.to(torch.float32)
    mse = F.mse_loss(output_new, targets).item()

    return new_values, {"mse": mse, "n_selected": t}


def compute_compaction_c2(
    C1: torch.Tensor,
    beta: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    head_dim: int,
    ridge_lambda: float = 1e-4,
    ridge_scale: str = "spectral",
    targets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Standard compaction C2 solve (valid when compacted block replaces full block).

    softmax(qK^T)V ≈ softmax(qC1^T + beta) C2

    ``targets``, when given, replace ``softmax(qK^T)V``. Phase-1 passes the
    in-context flex attention output: on Llama 3 the analytical document-only
    softmax is a different vector from the residual the model actually uses.
    """
    inv_sqrt_d = _inv_sqrt_d(head_dim)

    if targets is None:
        sK = (queries @ keys.T).to(torch.float32) * inv_sqrt_d
        attn_K = F.softmax(sK, dim=-1)
        Y = attn_K @ values.to(torch.float32)
    else:
        Y = targets.to(torch.float32)

    sC = (queries @ C1.T).to(torch.float32) * inv_sqrt_d + beta.to(torch.float32)
    X = F.softmax(sC, dim=-1)

    return _ridge_lstsq(X, Y, ridge_lambda=ridge_lambda, ridge_scale=ridge_scale).to(values.dtype)


def compact_kv_head(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    t: int,
    head_dim: int,
    ridge_lambda: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, list, dict]:
    """Full AM compaction for one KV head (Phase 1 init).

    When t == T, selects all keys and refines values via C2 lstsq.

    Returns:
        new_keys: (t, d) — selected keys (subset of original)
        new_values: (t, d) — fitted values
        indices: selected indices in original cache
        stats: reconstruction metrics
    """
    T = keys.shape[0]
    t = min(t, T)

    if t == T:
        indices = list(range(T))
        C1 = keys
        beta = torch.zeros(T, device=keys.device, dtype=torch.float32)
    else:
        C1, beta, indices = select_keys_highest_attention(
            keys, queries, t, head_dim, score_method="rms"
        )

    C2 = compute_compaction_c2(
        C1, beta, keys, values, queries, head_dim, ridge_lambda=ridge_lambda
    )

    # Evaluate reconstruction on queries
    inv_sqrt_d = _inv_sqrt_d(head_dim)
    target = compute_attention_output(queries, keys, values, head_dim)

    sC = (queries @ C1.T).to(torch.float32) * inv_sqrt_d + beta.to(torch.float32)
    X = F.softmax(sC, dim=-1)
    approx = X @ C2.to(torch.float32)
    mse = F.mse_loss(approx, target).item()

    stats = {"mse": mse, "t": t, "T": T, "n_queries": queries.shape[0]}
    return C1, C2, indices, stats


# ======================================================================================
# Corpus-level compaction
# ======================================================================================
def _chat_system_prefix_ids(tokenizer, content: str) -> torch.Tensor:
    """Token ids of ``content`` as the eval chat template's system message."""
    from cartridges.am.components.teacher import eval_aligned_system_ids

    return eval_aligned_system_ids(tokenizer, content)


def _capture_post_rope_kv(model, input_ids: torch.Tensor, device) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Post-RoPE K and V for one document, shaped ``(n_kv_heads, T, head_dim)``."""
    input_ids = input_ids.to(device)
    captured: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    handles = []
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn

        def _make(idx, attn_mod):
            def _hook(module, args, output):
                hidden = args[0].hidden_states
                shape = (*hidden.shape[:-1], -1, attn_mod.head_dim)
                projected = attn_mod.k_proj(hidden).view(shape)
                # Qwen3 RMS-normalizes K before RoPE. Skipping it leaves deep-layer
                # keys unnormalized (absmax ~80), the value solve blows up, and
                # phase-1 eval loss sits near 13 nats.
                if hasattr(attn_mod, "k_norm"):
                    projected = attn_mod.k_norm(projected)
                keys = projected.transpose(1, 2)
                values = attn_mod.v_proj(hidden).view(shape).transpose(1, 2)
                cos, sin = args[0].position_embeddings
                keys = _apply_rotary_pos_emb(keys, cos, sin)
                captured[idx] = (
                    keys[0].detach().to("cpu", torch.bfloat16),
                    values[0].detach().to("cpu", torch.bfloat16),
                )

            return _hook

        handles.append(attn.register_forward_hook(_make(layer_idx, attn)))
    try:
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            model(
                input_ids=input_ids,
                seq_ids=torch.zeros_like(input_ids),
                position_ids=torch.arange(input_ids.numel(), device=input_ids.device),
                use_cache=False,
                mode="train",
            )
    finally:
        for handle in handles:
            handle.remove()
    return captured


def _shift_key_positions(keys, src_start: int, dst_start: int, head_dim: int, rope_theta: float):
    """Move post-RoPE keys from positions ``src_start`` to ``dst_start``."""
    if src_start == dst_start:
        return keys
    length = keys.shape[1]
    from_pos = torch.arange(src_start, src_start + length, device=keys.device)
    to_pos = torch.arange(dst_start, dst_start + length, device=keys.device)
    return torch.stack(
        [
            _rope_reposition(
                keys[head].float(), from_pos, to_pos, head_dim, rope_theta=rope_theta
            ).to(keys.dtype)
            for head in range(keys.shape[0])
        ],
        0,
    )


def _forward_capture(model, input_ids, position_ids, cache, kind: str, seq_ids=None):
    if kind == "query":
        captured, handles = install_query_capture_hooks(model)
    else:
        captured, handles = install_teacher_attention_capture_hooks(model)
    if seq_ids is None:
        seq_ids = torch.zeros_like(input_ids)
    try:
        if cache is not None:
            cache.clear()
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            model(
                input_ids=input_ids,
                seq_ids=seq_ids,
                position_ids=position_ids,
                use_cache=cache is not None,
                past_key_values=cache,
                mode="train",
            )
    finally:
        for handle in handles:
            handle.remove()
        if cache is not None:
            cache.clear()
    return captured


def _collect_flex_phase1_queries(
    model,
    tokenizer,
    ref_groups: dict,
    local_docs: dict,
    attn_config: AttnConfig,
    num_tokens: int,
    queries_per_batch: str,
    device,
    collect_targets: bool = True,
):
    """Question-only queries at eval positions, targets from the document cache.

    Queries are captured at ``num_tokens + local_pos`` because eval adds the
    cartridge length to ``position_ids``. Targets are the flex attention outputs
    of those same tokens when the document KV is visible.
    """
    converter = MODEL_TO_MESSAGE_CONVERTER[tokenizer.name_or_path.lower()]
    n_layers = attn_config.n_layers
    query_acc = AMQueryAccumulator(
        granularity="per_layer",
        queries_per_batch="all_tokens",
        n_layers=n_layers,
        n_kv_heads=attn_config.n_heads,
        device=device,
    )
    target_acc = AMTargetAccumulator(n_layers=n_layers, n_kv_heads=attn_config.n_heads)
    n_used = 0
    for doc_id, convos in ref_groups.items():
        if doc_id not in local_docs:
            logger.warning("No teacher KV for reference document %s", doc_id)
            continue
        encoded = []
        for convo in convos:
            messages = [m for m in convo.messages if m.role != "system"]
            if not messages:
                continue
            elem = converter(list(messages), retokenize=True, tokenizer=tokenizer)
            encoded.append(elem.input_ids)
        if not encoded:
            continue
        # One flex shape per document. Padding uses a different seq id, so the
        # real tokens' attention matches the unpadded question.
        bucket = max(int(ids.numel()) for ids in encoded)
        bucket = max(128, ((bucket + 127) // 128) * 128)
        doc_cache = (
            cache_from_layer_kv(attn_config, local_docs[doc_id], device=device)
            if collect_targets
            else None
        )
        try:
            for ids in encoded:
                ids = ids.to(device)
                length = int(ids.numel())
                if length < bucket:
                    pad = torch.zeros(bucket - length, dtype=ids.dtype, device=device)
                    ids = torch.cat([ids, pad])
                    seq = torch.cat([
                        torch.zeros(length, dtype=torch.long, device=device),
                        torch.ones(bucket - length, dtype=torch.long, device=device),
                    ])
                else:
                    seq = torch.zeros(length, dtype=torch.long, device=device)
                if queries_per_batch == "last_token":
                    token_idx = slice(length - 1, length)
                else:
                    token_idx = slice(0, length)
                span = int(ids.numel())
                q_pos = torch.arange(num_tokens, num_tokens + span, device=device)
                y_pos = torch.arange(span, device=device)
                q_cap = _forward_capture(model, ids, q_pos, None, "query", seq)
                y_cap = (
                    _forward_capture(model, ids, y_pos, doc_cache, "target", seq)
                    if collect_targets
                    else {}
                )
                for layer_idx, query in q_cap.items():
                    if layer_idx >= n_layers:
                        continue
                    query_acc._queries[layer_idx].append(query[:, :, token_idx, :].detach().cpu())
                for layer_idx, target in y_cap.items():
                    if layer_idx >= n_layers:
                        continue
                    target_acc._targets[layer_idx].append(target[:, :, token_idx, :].detach().cpu())
                n_used += 1
        finally:
            del doc_cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    query_acc._n_ref_batches = n_used
    return query_acc, target_acc, n_used


def _collect_compaction_queries(
    model,
    dataloader,
    n_layers: int,
    n_kv_heads: int,
    queries_per_batch: str,
    max_ref_batches: int,
    attn_config: AttnConfig,
    local_rank,
) -> AMQueryAccumulator:
    """Capture post-RoPE reference queries by reading QA text with no cartridge.

    Queries are the model's own query vectors while reading the QA reference
    sequences causally (no cartridge attended to). These represent the query
    distribution the compacted cartridge must serve.
    """
    query_acc = AMQueryAccumulator(
        granularity="per_layer",
        queries_per_batch=queries_per_batch,
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        device=local_rank,
    )
    captured, handles = install_query_capture_hooks(model)
    model.eval()

    batch_count = 0
    try:
        with torch.no_grad():
            for batch in dataloader:
                if batch_count >= max_ref_batches:
                    break
                captured.clear()
                seq_ids = batch.element_ids.to(local_rank)
                with torch.amp.autocast(
                    device_type="cuda" if torch.cuda.is_available() else "cpu",
                    dtype=torch.bfloat16,
                    enabled=torch.cuda.is_available(),
                ):
                    # No cartridge → reference tokens attend causally to
                    # themselves only (same call convention as teacher prefill).
                    model(
                        input_ids=batch.input_ids.to(local_rank),
                        seq_ids=seq_ids,
                        position_ids=batch.position_ids.to(local_rank),
                        use_cache=False,
                        past_key_values=None,
                        mode="train",
                    )
                if not captured:
                    batch_count += 1
                    continue
                valid_len = getattr(batch, "valid_len", None)
                if valid_len is None:
                    valid_len = seq_ids.shape[0]

                # Store raw per-layer queries without needing a cartridge for
                # access scores (compaction does not use TF access scores).
                if queries_per_batch == "last_token":
                    valid_seq_ids = seq_ids[:valid_len]
                    unique_ids = valid_seq_ids.unique()
                    token_indices = [
                        (valid_seq_ids == uid).nonzero(as_tuple=True)[0][-1].item()
                        for uid in unique_ids
                    ]
                    tok_idx = torch.tensor(token_indices, device=seq_ids.device)
                else:
                    tok_idx = torch.arange(valid_len, device=seq_ids.device)
                for layer_idx, q in captured.items():
                    if layer_idx >= n_layers:
                        continue
                    q_sel = q[:, :, tok_idx, :]
                    query_acc._queries[layer_idx].append(q_sel.detach().cpu())
                batch_count += 1
    finally:
        for h in handles:
            h.remove()

    query_acc._n_ref_batches = batch_count
    return query_acc


def compact_cache_am_phase1(
    model,
    tokenizer,
    qa_data_path: str,
    attn_config: AttnConfig,
    dataset: str = "qasper",
    qasper_topic: str = "QA",
    quality_phase: Optional[int] = None,
    phase: Optional[int] = None,
    num_tokens: int = 512,
    key_select: Literal["highest_attention", "omp"] = "highest_attention",
    ridge_lambda: float = 1e-4,
    ridge_scale: str = "spectral",
    enable_beta: bool = True,
    max_ref_batches: int = 50,
    queries_per_batch: str = "all_tokens",
    max_queries_per_head: int = 64,
    max_teacher_tokens: Optional[int] = None,
    max_ref_examples: Optional[int] = 256,
    rebake_key_positions: bool = True,
    strip_reference_system_prompt: bool = True,
    global_teacher_positions: bool = False,
    rope_theta: float = 10000.0,
    nnls_iters: Optional[int] = None,
    beta_w_lower: Optional[float] = None,
    beta_w_upper: Optional[float] = None,
    teacher_kv_device: str = "cpu",
    local_rank="cuda",
) -> Tuple[TrainableCache, dict]:
    """Classic AM compaction Phase 1 over all QA documents.

    1. Group QA docs, prefill each into a teacher KV, concatenate across docs.
    2. Collect QA reference queries (causal, no cartridge).
    3. Per (layer, KV head): select ``num_tokens`` teacher keys + ridge-fit values
       to reproduce the teacher's attention output on the reference queries.
    4. Assemble a TrainableCache with teacher-derived keys (and optional beta).

    ``dataset`` selects the complete-document resolver. Historical callers
    default to QASPER; QuALITY callers must also provide ``quality_phase``.

    ``nnls_iters`` / ``beta_w_lower`` / ``beta_w_upper`` override the beta fit's
    PGD budget and weight box; ``None`` keeps the per-mode reference default from
    ``am.components.keys`` (highest_attention: 2 iters, ``[e^-3, e^3]``; omp:
    0 iters, ``[1e-12, e^7]``).

    ``rope_theta`` is only consumed by the ``rebake_key_positions`` rotation, and
    that rotation lands on the intended slot **only** when it is the base the keys
    were baked with. The default is the historical hard-coded value, not the
    model's own base (Qwen3-4B-Instruct-2507 = 5e6) -- see MECH-003 / MECH-005.

    ``teacher_kv_device`` holds the concatenated corpus KV off-accelerator; the
    per-(layer, head) solve below pulls one layer back at a time, so the full
    corpus never has to fit in device memory.

    Returns (cartridge, stats). No Phase-2 stabilizers are used (one-shot compaction).
    """
    t0 = time.time()
    beta_fit_kwargs = {
        name: value
        for name, value in (
            ("n_iters" if key_select == "omp" else "nnls_iters", nnls_iters),
            ("w_lower", beta_w_lower),
            ("w_upper", beta_w_upper),
        )
        if value is not None
    }
    n_layers = attn_config.n_layers
    n_kv_heads = attn_config.n_heads
    head_dim = attn_config.head_dim
    inv_sqrt_d = _inv_sqrt_d(head_dim)
    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_dtype = torch.bfloat16

    bound = bind_model_rope(model)
    vanilla = 1.0 / (
        float(rope_theta)
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    logger.info(
        "RoPE rebake bound to model frequencies (ratio to vanilla theta min=%.4f max=%.4f)",
        float((bound / vanilla).min()),
        float((bound / vanilla).max()),
    )
    model_rope_theta = getattr(getattr(model, "config", None), "rope_theta", None)
    rope_theta_matches_model = (
        model_rope_theta is None or abs(float(model_rope_theta) - rope_theta) < 1e-6
    )
    if rebake_key_positions and not rope_theta_matches_model:
        logger.warning(
            "rebake_key_positions=True with rope_theta=%s but the model's rotary base "
            "is %s. Composing two RoPE rotations only lands on an absolute position "
            "when both use the same base, so the rebaked keys will sit at no "
            "well-defined position. Pass the model's base.",
            rope_theta, model_rope_theta,
        )

    # ---- 1. Build the teacher KV from all QA documents -------------------
    conversations = load_conversations(qa_data_path)
    doc_groups = group_conversations_by_document(conversations, dataset)
    logger.info("Compaction Phase 1: %d unique QA documents", len(doc_groups))

    teacher_k: dict[int, list[torch.Tensor]] = {l: [] for l in range(n_layers)}
    teacher_v: dict[int, list[torch.Tensor]] = {l: [] for l in range(n_layers)}
    per_doc_tokens: dict[int, int] = {l: 0 for l in range(n_layers)} # tokens per layer
    teacher_pos_chunks: list[torch.Tensor] = []  # per-doc RoPE positions (layer-agnostic)
    doc_token_counts = []
    running_offset = 0
    # Unrotated (positions 0..T-1) KV, one entry per document, for flex targets.
    local_docs: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
    for doc_idx, (doc_id, group) in enumerate(doc_groups.items()):
        # 1.1 Get the system prompt for the document
        system_prompt = full_document_prompt(
            doc_id,
            dataset=dataset,
            qasper_topic=qasper_topic,
            quality_phase=quality_phase,
            phase=phase,
        )
        if not system_prompt.strip():
            continue
        # 1.2 Prefill in the eval chat format so keys live at positions 0..T-1,
        # the geometry a question attends to when this KV is the cartridge.
        pos_offset = running_offset if global_teacher_positions else 0
        prefix_ids = _chat_system_prefix_ids(tokenizer, system_prompt)
        doc_kv = _capture_post_rope_kv(model, prefix_ids, local_rank)
        local_docs[doc_id] = doc_kv
        t_doc = doc_kv[0][0].shape[1]
        doc_token_counts.append(t_doc)
        appended_len = None
        # 1.3 Append the teacher KV cache to the teacher KV caches across layers
        for layer_idx in range(n_layers):
            k_d, v_d = doc_kv[layer_idx]  # (n_kv_heads, T_doc, head_dim)
            if pos_offset:
                k_d = _shift_key_positions(
                    k_d, 0, pos_offset, head_dim, rope_theta
                )
            if max_teacher_tokens is not None and per_doc_tokens[layer_idx] >= max_teacher_tokens:
                appended_len = 0
                continue
            if max_teacher_tokens is not None:
                room = max_teacher_tokens - per_doc_tokens[layer_idx]
                if k_d.shape[1] > room:
                    k_d = k_d[:, :room]
                    v_d = v_d[:, :room]
            teacher_k[layer_idx].append(k_d.to(teacher_kv_device))
            teacher_v[layer_idx].append(v_d.to(teacher_kv_device))
            per_doc_tokens[layer_idx] += k_d.shape[1]
            appended_len = k_d.shape[1]
        if appended_len:
            teacher_pos_chunks.append(
                torch.arange(pos_offset, pos_offset + appended_len, dtype=torch.long)
            )
            running_offset += appended_len
    # 1.4 Concatenate the teacher KV caches across layers
    teacher_K = {l: torch.cat(teacher_k[l], dim=1) for l in range(n_layers)}
    teacher_V = {l: torch.cat(teacher_v[l], dim=1) for l in range(n_layers)}
    # Per-token absolute RoPE position within its source document (restarts per doc).
    teacher_pos = torch.cat(teacher_pos_chunks).to(device=local_rank)
    T_teacher = teacher_K[0].shape[1]
    logger.info(
        "Teacher KV built: T_teacher=%d tokens (per-doc counts=%s)",
        T_teacher, doc_token_counts,
    )

    # ---- 2. Collect QA reference queries and in-context flex targets -------
    # Eval questions carry no document and sit at positions ``num_tokens + i``
    # (the model adds the cartridge length). The value target is the flex
    # attention output of those tokens when this document's KV is visible,
    # which is the residual that actually predicts the answer. The analytical
    # document-only softmax is a different vector on Llama 3.
    ref_convos = limit_conversations(conversations, max_ref_examples, seed=0)
    ref_groups = group_conversations_by_document(ref_convos, dataset)
    if not strip_reference_system_prompt:
        logger.warning(
            "strip_reference_system_prompt=False; question queries still drop "
            "system-role turns so the cartridge, not an in-prompt document, "
            "has to carry the paper."
        )
    # Qwen3 qk-norm cannot fit queries placed at ``num_tokens + i``. Both the
    # flex residual and the analytical softmax of those queries leave deep-layer
    # recon mse around 20 and eval loss around 13 nats. The published Qwen
    # matrices used question-only queries at the dataloader's own positions.
    from cartridges.am.components.teacher import model_uses_qk_norm
    use_flex_targets = not model_uses_qk_norm(model)
    query_position_offset = int(num_tokens)
    if use_flex_targets:
        query_acc, target_acc, n_ref_batches = _collect_flex_phase1_queries(
            model,
            tokenizer,
            ref_groups,
            local_docs,
            attn_config,
            num_tokens,
            queries_per_batch,
            local_rank,
            collect_targets=True,
        )
    else:
        if strip_reference_system_prompt:
            ref_convos = [dataclasses.replace(c, system_prompt="") for c in ref_convos]
        ref_loader, ref_tmp = build_reference_dataloader(ref_convos, tokenizer, seed=0)
        try:
            query_acc = _collect_compaction_queries(
                model,
                ref_loader,
                n_layers,
                n_kv_heads,
                queries_per_batch,
                max_ref_batches,
                attn_config,
                local_rank,
            )
        finally:
            cleanup_reference_parquet(ref_tmp)
        target_acc = None
        n_ref_batches = getattr(query_acc, "_n_ref_batches", 0)
        query_position_offset = 0
    logger.info(
        "Reference queries collected for %d batches (flex_targets=%s, position offset %d)",
        n_ref_batches,
        use_flex_targets,
        query_position_offset,
    )

    # ---- 3. Compact per (layer, KV head) ---------------------------------
    keys_out: list[torch.Tensor] = []
    values_out: list[torch.Tensor] = []
    beta_out: list[torch.Tensor] = []
    mse_per_layer: dict[int, float] = {}
    device = torch.device(local_rank) if not isinstance(local_rank, str) else local_rank

    for layer_idx in range(n_layers):
        logger.info("Compaction solve layer %d/%d", layer_idx + 1, n_layers)
        K_T = teacher_K[layer_idx].to(device=device, dtype=model_dtype)  # (n_kv_heads, T, d)
        V_T = teacher_V[layer_idx].to(device=device, dtype=model_dtype)
        dtype = model_dtype
        layer_keys = torch.zeros(1, n_kv_heads, num_tokens, head_dim, device=device, dtype=dtype)
        layer_values = torch.zeros(1, n_kv_heads, num_tokens, head_dim, device=device, dtype=dtype)
        layer_beta = torch.zeros(1, n_kv_heads, num_tokens, device=device, dtype=torch.float32)
        layer_mses = []

        if not query_acc._queries[layer_idx]:
            logger.warning("No reference queries for layer %d; using zeros", layer_idx)
            keys_out.append(layer_keys)
            values_out.append(layer_values)
            beta_out.append(layer_beta)
            continue

        q_batch = query_acc._queries[layer_idx][0]
        n_q_heads = q_batch.shape[1]

        for head_idx in range(n_kv_heads):
            K_head = K_T[head_idx]  # (T, d)
            V_head = V_T[head_idx]
            queries = query_acc.get_layer_head_queries(
                layer_idx, head_idx, n_q_heads=n_q_heads, n_kv_heads=n_kv_heads,
            ).to(device=device, dtype=dtype)
            flex_targets = None
            if use_flex_targets:
                flex_targets = target_acc.get_layer_head_targets(
                    layer_idx, head_idx, n_q_heads=n_q_heads, n_kv_heads=n_kv_heads,
                ).to(device=device, dtype=torch.float32)
            if queries.numel() == 0:
                continue
            if flex_targets is not None and flex_targets.shape[0] != queries.shape[0]:
                raise RuntimeError(
                    f"Flex targets ({flex_targets.shape[0]}) are not aligned with "
                    f"queries ({queries.shape[0]}) at layer {layer_idx} head {head_idx}"
                )
            if queries.shape[0] > max_queries_per_head:
                idx = torch.randperm(queries.shape[0], device=device)[:max_queries_per_head]
                queries = queries[idx]
                if flex_targets is not None:
                    flex_targets = flex_targets[idx]

            # 3.1 Select the keys for the compacted block
            if key_select == "omp":
                C1, beta, sel_idx = select_keys_omp(
                    K_head, queries, num_tokens, head_dim, **beta_fit_kwargs,
                )
            else: # highest_attention
                C1, beta, sel_idx = select_keys_highest_attention(
                    K_head, queries, num_tokens, head_dim, score_method="rms",
                    fit_beta=enable_beta,
                    **beta_fit_kwargs,
                )

            # 3.2 Compute the compaction C2 solve
            if not enable_beta:
                beta = torch.zeros(C1.shape[0], device=device, dtype=torch.float32)

            # 3.3 Fixing the RoPE positional information
            if rebake_key_positions:
                # Re-base selected teacher keys onto sequential cartridge slot
                # positions so eval-query↔cartridge RoPE geometry matches how a
                # normal front cartridge (e.g. KVFromText) behaves.
                t_sel = C1.shape[0]
                from_pos = teacher_pos[torch.tensor(sel_idx, device=device)]
                to_pos = torch.arange(t_sel, device=device)
                C1 = _rope_reposition(C1, from_pos, to_pos, head_dim, rope_theta=rope_theta)

            # 3.4 Compute the compaction solve for the values
            C2 = compute_compaction_c2(
                C1, beta, K_head, V_head, queries, head_dim,
                ridge_lambda=ridge_lambda, ridge_scale=ridge_scale,
                targets=flex_targets,
            )

            # Reconstruction MSE against the target the solve just fitted.
            if flex_targets is None:
                target = compute_attention_output(queries, K_head, V_head, head_dim)
            else:
                target = flex_targets
            sC = (queries @ C1.T).to(torch.float32) * inv_sqrt_d + beta.to(torch.float32)
            approx = F.softmax(sC, dim=-1) @ C2.to(torch.float32)
            layer_mses.append(F.mse_loss(approx, target).item())

            t_sel = C1.shape[0]
            layer_keys[0, head_idx, :t_sel] = C1.to(dtype)
            layer_values[0, head_idx, :t_sel] = C2.to(dtype)
            layer_beta[0, head_idx, :t_sel] = beta.to(torch.float32)

        keys_out.append(layer_keys)
        values_out.append(layer_values)
        beta_out.append(layer_beta)
        if layer_mses:
            mse_per_layer[layer_idx] = sum(layer_mses) / len(layer_mses)
        # Free the teacher for this layer.
        del K_T, V_T
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- 4. Assemble the cartridge ---------------------------------------
    cache = TrainableCache(
        config=attn_config,
        init_keys=keys_out,
        init_values=values_out,
        num_frozen_tokens=0,
    ).to(device)

    if enable_beta and cache.trainable_beta is not None:
        with torch.no_grad():
            for layer_idx in range(n_layers):
                cache.trainable_beta[layer_idx].copy_(
                    beta_out[layer_idx].to(cache.trainable_beta[layer_idx].dtype)
                )
        cache.enable_attention_bias(True)

    with torch.no_grad():
        value_absmax = max(float(v.detach().float().abs().max()) for v in values_out)
        key_absmax = max(float(k.detach().float().abs().max()) for k in keys_out)

    teacher_pos_note = (
        "Each doc prefilled at a running offset, so absolute RoPE positions are "
        "unique corpus-wide."
        if global_teacher_positions
        else "Per-doc prefill at position 0; docs concatenated so absolute RoPE "
        "positions overlap across docs (each doc self-consistent)."
    )
    if not rebake_key_positions:
        rebake_note = "Keys installed at their teacher positions (no rebake)."
    elif rope_theta_matches_model:
        rebake_note = (
            f"Selected keys rebaked onto sequential slots at rope_theta={rope_theta:g}, "
            "matching the model's base."
        )
    else:
        rebake_note = (
            f"Selected keys rebaked at rope_theta={rope_theta:g} but the model's base "
            f"is {model_rope_theta:g}, so the rebake does NOT land on the slot."
        )

    mse_values = list(mse_per_layer.values())
    stats = {
        "value_absmax": value_absmax,
        "key_absmax": key_absmax,
        "n_documents": len(doc_groups),
        "T_teacher": int(T_teacher),
        "doc_token_counts": doc_token_counts,
        "num_tokens": num_tokens,
        "key_select": key_select,
        "dataset": dataset,
        "qasper_topic": qasper_topic,
        "quality_phase": quality_phase,
        "phase": phase,
        "enable_beta": bool(enable_beta),
        "rebake_key_positions": bool(rebake_key_positions),
        "global_teacher_positions": bool(global_teacher_positions),
        "rope_theta": float(rope_theta),
        "model_rope_theta": None if model_rope_theta is None else float(model_rope_theta),
        "rope_theta_matches_model": bool(rope_theta_matches_model),
        "strip_reference_system_prompt": bool(strip_reference_system_prompt),
        "ridge_lambda": ridge_lambda,
        "ridge_scale": ridge_scale,
        "n_ref_batches": n_ref_batches,
        "flex_targets": use_flex_targets,
        "query_position_offset": int(query_position_offset),
        "recon_mse_mean": float(sum(mse_values) / max(len(mse_values), 1)) if mse_values else None,
        "recon_mse_max": float(max(mse_values)) if mse_values else None,
        "recon_mse_per_layer": {int(k): float(v) for k, v in mse_per_layer.items()},
        "wall_clock_s": time.time() - t0,
        "rope_note": " ".join([teacher_pos_note, rebake_note]),
        "stabilizers": "none (one-shot compaction: no delta_weight, no old-ref guard)",
    }
    logger.info(
        "Compaction Phase 1 done in %.1fs: recon_mse_mean=%s max=%s",
        stats["wall_clock_s"], stats["recon_mse_mean"], stats["recon_mse_max"],
    )
    return cache, stats
