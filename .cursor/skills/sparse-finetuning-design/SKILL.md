---
name: sparse-finetuning-design
description: Architecture knowledge for the cartridge sparse finetuning system (continual learning on KV-cache cartridges). Use when interpreting experiment results, proposing design changes, debugging training behavior, or reasoning about forgetting vs. plasticity trade-offs in Phase 2 training.
disable-model-invocation: true
---

# Sparse Cache Finetuning — Design Knowledge

## Research Goal

A **cartridge** is a trainable KV-cache that stores document knowledge for an LLM.
- **Phase 1**: train a cartridge on a set of documents.
- **Phase 2 (continual)**: update the cartridge with *new* documents **without forgetting Phase 1**.

The core failure mode is **catastrophic forgetting**: naive full finetuning overwrites the Phase 1 cache positions that encode old knowledge.

---

## Why Sparse Updates Work

The key insight: if Phase 1 knowledge is distributed across many cache positions, new knowledge only *needs* to change a small subset. By identifying and updating only those positions, everything else is implicitly frozen.

The two questions sparse finetuning must answer:
1. **Which positions does the new data need?** → answered by attention scores (TF)
2. **Which of those are already critical to Phase 1?** → answered by background statistics (IDF)

---

## TF-IDF Position Scoring

**Term Frequency (TF)**: normalized attention score from the current batch's queries to each cache position, summed across layers and heads.

**Inverse Document Frequency (IDF)**: `log((|B| + s) / (df(i) + s))`
- `|B|` = number of background (Phase 1) batches
- `df(i)` = how many background batches had position `i` in their top-k
- `s` = smoothing constant (default 1.0)

**TF-IDF** = TF × IDF. This selects positions that:
- The new data actively attends to (high TF)
- Were NOT frequently activated during Phase 1 (high IDF)

**Why not TF alone?** High-TF positions may be fundamental structural positions always active in Phase 1. Updating them causes forgetting.

**Why not IDF alone?** IDF is static — it doesn't know what the current batch actually needs.

**Implementation**: `CacheTFIDFRanker` in `sparse_cache_finetuning.py`. Background stats are collected once from Phase 1 data and saved to `bg_stats.pt`.

---

## Gradient Masking: Post-Backward (Current Design)

After backward, `apply_gradient_mask_to_cache` directly zeros `.grad` for non-top-t value positions.

**Why post-backward, not pre-forward (paper's original trick)?**

The paper's approach inserted a differentiable identity into the forward graph (`out = param * mask + detach — (param * mask).detach()`). This required the mask to be set *before* `forward()`, which forced a **1-step lag**: the mask for step N used batch N-1's attention patterns.

The problem: the current batch determines which positions *should* update, but the mask came from the previous batch's view — a fundamental misalignment.

Post-backward masking:
```
forward(batch N)         → collect attention scores
loss.backward()          → compute full gradients
rank_positions(scores)   → top_positions from batch N ← correct
zero non-top .grads      → gradient mask applied now
optimizer.step()         → only top-t positions update
```

**With gradient accumulation**: all microbatch gradients are already summed in `.grad` by the time `apply_gradient_mask_to_cache` is called at `do_step` — no special handling needed.

---

## Freeze Keys (Default: True)

**Keys** = the attention routing structure. They determine *which* queries attend to *which* cache positions.

**Values** = the content stored at each position. They determine *what* is retrieved.

`freeze_keys=True` zeros all key gradients post-backward. Only values are updated (sparsely at top-t positions). This preserves the attention routing that Phase 1 knowledge depends on.

`freeze_keys=False` allows keys to update. All key positions update (no position-level masking for keys — only values are position-masked). This changes the addressing structure globally while only changing values locally. More expressive, but risks breaking Phase 1 retrieval patterns.

**Note**: `freeze_keys` interacts with `hard` momentum masking. When `freeze_keys=False`, key momentum is intentionally left untouched. When `freeze_keys=True`, key momentum is zeroed.

---

## Momentum Masking Modes

SGD with momentum means even zero-gradient positions drift due to accumulated momentum: `v_t = 0.9 × v_{t-1}`. After `n` steps without a gradient, a position accumulates drift proportional to `v_0 × (0.9 + 0.9² + ... + 0.9ⁿ)`. The modes control this:

| Mode | What happens to non-top-t positions | Forgetting risk | Plasticity |
|---|---|---|---|
| **soft** | grad=0, momentum decays → param drifts ~20 steps | Slow, gradual | High |
| **hard** | momentum zeroed before step → param stops immediately | Minimal | Low |
| **freeze** | param + momentum both restored post-step → exact freeze | None | Lowest |
| **decouple** | param restored post-step, momentum keeps decaying | None (param) | Medium |

**Hypotheses for each mode:**
- `soft`: Allows gradual repurposing of positions. Risk is subtle, slow forgetting of Phase 1.
- `hard`: Cleanest separation. When a position re-enters top-t, it starts with zero momentum — convergence at that position is slower.
- `freeze`: Perfect isolation. The saved momentum lets positions "resume" correctly when they re-enter top-t. Best when top-t sets are stable and non-overlapping across steps.
- `decouple`: Param is frozen (no drift), but momentum warms up for when the position is re-selected. Likely the best balance: zero forgetting + warm restart.

---

## Attention Score Collection

Per forward pass, post-RoPE queries are captured via hooks (under `torch.no_grad()`, separate from the compiled flex_attention graph). The **last token of each packed sequence** is used as the representative query — it has attended to all preceding context, making it the best proxy for "what does this document need from the cache?"

Scores are summed across all layers, all heads (with GQA head grouping), and all sequences in the microbatch. They accumulate across microbatches within one optimizer step.

---

## The 8-Experiment Design Space

| `momentum_masking` | `freeze_keys` | Hypothesis |
|---|---|---|
| soft | True | Baseline (paper's approach adapted). Gradual drift risk. |
| soft | False | Baseline + key updates. High forgetting risk from both drift and key changes. |
| hard | True | No drift. Clean but cold-starts on re-entry. |
| hard | False | No value drift, but key momentum also zeroed. Aggressive. |
| freeze | True | Perfect param+momentum preservation. Best for stable top-t sets. |
| freeze | False | Perfect value freeze + free keys. Keys evolve; values are frozen for non-top. |
| decouple | True | Param frozen, momentum warm. Likely best balance for values-only training. |
| decouple | False | Param frozen for values, keys update freely. Warm momentum for value re-entry. |

---

## Key Files

| File | Role |
|---|---|
| `cartridges/sparse_cache_finetuning.py` | `SparseCacheFinetuningConfig`, `CacheAccessTracker`, `CacheTFIDFRanker`, `apply_gradient_mask_to_cache`, momentum helpers |
| `cartridges/train.py` | Training loop — where masking, momentum, and optimizer step are orchestrated |
| `cartridges/cache.py` | `TrainableCache` — holds `trainable_keys` and `trainable_values` as `nn.ParameterList` |
| `examples/shared/train/continual_perplexity.py` | Phase 2 config wired to env vars |
| `examples/qasper/pipelines/train_continual.sh` | Single-experiment launcher |
| `examples/qasper/sweeps/pool_continual_sparse.sh` | Submits all 8 experiments as parallel SLURM jobs |

## What to Look for in Results

When analyzing experiment outputs, compare:
- **Phase 1 eval loss** (before and after Phase 2): measures forgetting
- **Phase 2 eval loss**: measures new knowledge acquisition
- **Slot overlap across steps** (from `sparse_slot_log`): how stable are the selected top-t positions? High stability favors `freeze`; high turnover favors `decouple`.
- **Loss curves shape**: `hard` and `freeze` may show sharper, noisier learning curves due to cold-start momentum. `soft` and `decouple` should be smoother.
