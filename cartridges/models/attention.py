import os

import torch

from typing import Optional, Union, Literal
from torch.nn.attention.flex_attention import create_block_mask, flex_attention, BlockMask

from cartridges.cache import TrainableCache

# The compiled generate-path kernels (dynamic flex_attention + compiled
# create_block_mask) hit a Triton "illegal memory access" on some torch/triton/GPU
# combinations for the short shapes decoding produces (cartridge KV + a ~80-token
# query). Generation shapes are tiny, so running that path eagerly is both cheap and
# correct; training stays on the tuned compiled kernels. Opt in per-process (e.g. the
# accuracy eval driver) with CARTRIDGES_EAGER_GENERATE=1.
_EAGER_GENERATE = os.environ.get("CARTRIDGES_EAGER_GENERATE", "0") == "1"



# SE (07/21): `dynamic=False` is necessary to avoid a "PassManager::run failed" error
# when interacting with torch.amp.autocast during training. This is okay since we pack
# all sequences to the same length during training.
# SE (07/22): The `mode="max-autotune-no-cudagraphs"` gives a ~2x speedup on 
# backward running on a single A100.

# `dynamic=False` costs one compiled variant per distinct sequence length, and the AM
# document paths prefill each document at its own length (16 for a QASPER phase). Past
# dynamo's recompile limit the call degrades to `sdpa_dense`, whose score matrix is
# quadratic in the sequence length -- 10 GiB for one 8.9k-token paper, ~87 GiB for a
# 26k-token TechQA technote -- so the failure reads as an OOM, not as a compile miss.
# The default limit is 8. A single continual stage can prefill ~100 documents, each a
# distinct length (and dynamo also recompiles on autocast-state changes), so 64 was not
# enough for TechQA; keep the ceiling well above the per-stage document count.
_DYNAMO_VARIANT_LIMIT = 1024
torch._dynamo.config.cache_size_limit = max(
    torch._dynamo.config.cache_size_limit, _DYNAMO_VARIANT_LIMIT
)
# `cache_size_limit` is the per-code-object ceiling (aliased to `recompile_limit` in
# newer torch); `accumulated_cache_size_limit` caps variants across all guarded frames.
if hasattr(torch._dynamo.config, "recompile_limit"):
    torch._dynamo.config.recompile_limit = max(
        torch._dynamo.config.recompile_limit, _DYNAMO_VARIANT_LIMIT
    )
if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
    torch._dynamo.config.accumulated_cache_size_limit = max(
        torch._dynamo.config.accumulated_cache_size_limit, 8 * _DYNAMO_VARIANT_LIMIT
    )
if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
    torch._dynamo.config.accumulated_recompile_limit = max(
        torch._dynamo.config.accumulated_recompile_limit, 8 * _DYNAMO_VARIANT_LIMIT
    )

flex_attention_train = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")

# # SE (07/25): For generation, we need to use `dynamic=True` to avoid a "PassManager::run failed" error
flex_attention_generate = (
    flex_attention if _EAGER_GENERATE else torch.compile(flex_attention, dynamic=True)
)

# Eager `create_block_mask` materialises the dense (Q_LEN, KV_LEN) mask before reducing
# it to blocks: 0.75 GiB for one 8.9k-token document, against 0.008 GiB compiled, for a
# bit-identical block mask.
create_block_mask_compiled = (
    create_block_mask if _EAGER_GENERATE else torch.compile(create_block_mask)
)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_key_value_heads * n_rep, slen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def create_block_mask_w_cache(
    cache: Optional[TrainableCache],
    seq_ids: torch.LongTensor, # [sum(seq_lens)]
    device: torch.device,
):
    cache_len = cache.num_tokens() if cache is not None else 0

    # Build the block mask
    # --- begin build block mask ---
    kv_seq_ids = seq_ids
    if cache_len > 0:
        kv_seq_ids = torch.cat([cache.seq_ids(), kv_seq_ids])

    def mask_func(_, _h, q_idx, kv_idx):
        return (kv_seq_ids[kv_idx] == -1) | ((seq_ids[q_idx] == kv_seq_ids[kv_idx]) & (q_idx + cache_len >= kv_idx))
    
    block_mask = create_block_mask_compiled(
        mask_func, B=1, H=1, Q_LEN=len(seq_ids), KV_LEN=len(seq_ids) + cache_len, 
        device=device,
    )
    return block_mask
    # --- end build block mask ---


def flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Union[torch.Tensor, "BlockMask"],
    scaling: Optional[float] = None,
    mode: Literal["train", "generate"] = "train",
    cartridge_beta: Optional[torch.Tensor] = None,
    num_cartridge_tokens: int = 0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:

    if kwargs.get("dropout", 0.0) > 0:
        raise ValueError(
            "`flex_attention` does not support `dropout`. Please use it with inference"
            " only (`model.eval()`) or turn off the attention dropout in the respective config."
        )

    block_mask = None
    if isinstance(attention_mask, BlockMask):
        block_mask = attention_mask

    enable_gqa = True
    num_local_query_heads = query.shape[1]

    # SE (07/25): For grouped-query attention, to work, the `Number of shared query 
    # heads sharing the same KV head must be power of 2`
    # This is the case with 3.2-3B (24 qheads and 8 kvheads), so we need to repeat 
    # the key  and value and turn off GQA. :( 
    if not ((num_local_query_heads & (num_local_query_heads - 1)) == 0):
        key = repeat_kv(key, query.shape[1] // key.shape[1])
        value = repeat_kv(value, query.shape[1] // value.shape[1])
        enable_gqa = False

    kernel_options = kwargs.get("kernel_options", None)
    attn = flex_attention_train if mode == "train" else flex_attention_generate
    
    # SE (07/26): This helps to avoid recompiles, since during prefix tuning, the first
    # layer's query does not require grad.
    if key.requires_grad and not query.requires_grad:
        query.requires_grad = True

    score_mod = None
    if cartridge_beta is not None and num_cartridge_tokens > 0:
        # cartridge_beta: (1, n_kv_heads, num_cartridge_tokens)
        beta_h = cartridge_beta[0].to(torch.float32)
        head_group_size = query.shape[1] // beta_h.shape[0]

        def score_mod(score, b, h, q_idx, kv_idx):
            kv_head = h // head_group_size
            safe_kv_idx = torch.clamp(kv_idx, max=num_cartridge_tokens - 1)
            bias = beta_h[kv_head, safe_kv_idx]
            return score + torch.where(kv_idx < num_cartridge_tokens, bias, 0.0)

    attn_output = attn(
        query,
        key,
        value,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
        scale=scaling,
        kernel_options=kernel_options,
        return_lse=False,
        score_mod=score_mod,
    )    
    attn_output = attn_output.transpose(1, 2).contiguous()


    return attn_output

