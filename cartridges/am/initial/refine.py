"""Phase 1, legacy variant: self-match refinement of an existing cartridge.

Refines all value positions of a ``KVFromText`` init cartridge against its own
self-study reference queries -- no compression, no key selection. Kept for
backward compatibility; ``initial.compaction`` is the classic AM construction.
"""

from __future__ import annotations

from logging import getLogger

import torch

from cartridges.am.components.objective import refine_kv_head_values
from cartridges.am.components.queries import AMQueryAccumulator
from cartridges.sparse_cache_finetuning import install_query_capture_hooks

logger = getLogger(__name__)


def refine_cache_am_phase1(
    cache,
    model,
    dataloader,
    n_am_passes: int = 3,
    max_batches: int = 50,
    ridge_lambda: float = 1e-4,
    queries_per_batch: str = "all_tokens",
    local_rank: torch.device = torch.device("cuda"),
) -> dict:
    """Phase 1 AM: refine all cache values using self-study reference queries.

    Runs multiple passes over the dataloader, updating all value positions
    per (layer, head) via closed-form AM (no gradients).
    """
    from cartridges.datasets import DatasetBatch

    n_layers = len(cache.trainable_values)
    n_kv_heads = cache.trainable_keys[0].shape[1]
    head_dim = cache.config.head_dim

    captured, handles = install_query_capture_hooks(model)
    model.eval()

    query_acc = AMQueryAccumulator(
        granularity="per_layer",
        queries_per_batch=queries_per_batch,
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        device=local_rank,
    )

    stats_history = []

    for pass_idx in range(n_am_passes):
        query_acc.reset()
        batch_count = 0

        with torch.no_grad():
            for batch in dataloader:
                if batch_count >= max_batches:
                    break
                batch: DatasetBatch
                captured.clear()

                model(
                    input_ids=batch.input_ids.to(local_rank),
                    seq_ids=batch.element_ids.to(local_rank),
                    position_ids=batch.position_ids.to(local_rank),
                    use_cache=True,
                    past_key_values=cache,
                )

                if captured:
                    query_acc.accumulate_from_hooks(
                        captured, cache,
                        scaling=head_dim ** -0.5,
                        seq_ids=batch.element_ids.to(local_rank),
                    )
                batch_count += 1

        pass_mses = []
        for layer_idx in range(n_layers):
            for head_idx in range(n_kv_heads):
                k_param = cache.trainable_keys[layer_idx]
                v_param = cache.trainable_values[layer_idx]
                keys = k_param[0, head_idx].detach()
                values = v_param[0, head_idx].detach()

                if query_acc._queries[layer_idx]:
                    q_batch = query_acc._queries[layer_idx][0]
                    n_q_heads = q_batch.shape[1]
                    queries = query_acc.get_layer_head_queries(
                        layer_idx, head_idx, n_q_heads, n_kv_heads,
                    ).to(device=local_rank, dtype=keys.dtype)
                else:
                    continue

                if queries.numel() == 0:
                    continue

                new_values, stats = refine_kv_head_values(
                    keys, values, queries, head_dim, ridge_lambda=ridge_lambda,
                )
                with torch.no_grad():
                    v_param[0, head_idx].copy_(new_values.to(v_param.dtype))
                pass_mses.append(stats["mse"])

        mean_mse = sum(pass_mses) / max(len(pass_mses), 1)
        stats_history.append({"pass": pass_idx, "mean_mse": mean_mse, "n_batches": batch_count})
        logger.info(f"AM Phase 1 pass {pass_idx}: mean_mse={mean_mse:.6f}, batches={batch_count}")

    for h in handles:
        h.remove()

    return {"passes": stats_history, "final_mean_mse": stats_history[-1]["mean_mse"] if stats_history else 0.0}
