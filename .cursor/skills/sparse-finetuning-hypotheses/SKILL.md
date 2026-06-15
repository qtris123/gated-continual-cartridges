---
name: sparse-finetuning-hypotheses
description: >-
  Unconfirmed observations, working theories, open questions, and planned experiments
  for the cartridge sparse finetuning system. Companion to sparse-finetuning-design,
  which holds confirmed architecture knowledge. Use this skill when planning new
  experiments, reasoning about unexpected results, or deciding what to ablate next.
disable-model-invocation: true
---

# Sparse Finetuning — Hypotheses, Open Questions, and Experiment Plans

This skill captures **what we think is happening and why** — not confirmed architecture facts.
For the confirmed system design, see `sparse-finetuning-design`.

---

## Confirmed Results to Date

### Momentum ablation: Phase 2 acquisition (eval on MT task, ~550 steps, lr=2.0, top-t=512)

| Mode | Final MT eval ppl (training curves) | Final MT eval ppl (forgetting eval) |
|---|---|---|
| freeze | ~24.8 | **24.92** |
| soft | ~25.5 | 25.13 |
| decouple | ~26.0 | 25.33 |
| hard | ~26.5 | 26.07 |

Ranking: **freeze > soft > decouple > hard** (consistent across both measurement methods).

### Forgetting eval: Phase 1 QA perplexity after Phase 2 training (job 456626)

| Run | QA ppl | Delta vs baseline |
|---|---|---|
| phase1_baseline | 5.584 | — |
| phase2_freeze | 5.416 | **−0.168** |
| phase2_soft | 5.428 | −0.156 |
| phase2_decouple | 5.434 | −0.150 |
| phase2_hard | 5.474 | −0.110 |

**Headline finding: no catastrophic forgetting detected.** All Phase 2 modes show slightly *improved*
Phase 1 QA perplexity after Phase 2 training. The improvement is small (~0.11–0.17 ppl) and
the eval set is very small (n=5 batches), so statistical reliability is uncertain. But the
direction is consistent: no mode degraded Phase 1.

**freeze is Pareto-dominant**: it wins on both Phase 2 acquisition (lowest MT ppl) and Phase 1
retention (lowest QA ppl). No trade-off observed.

**hard is Pareto-dominated**: worst on both metrics.

---

## Active Theories and Explanations

### Theory 1: Why freeze outperforms decouple (the decay-to-zero problem)

The design skill predicted decouple would be "the best balance" because it keeps params frozen
while allowing momentum to stay warm for re-entry. This prediction was wrong.

**The mechanism explaining the failure:**

When a position exits top-t, its gradient is zeroed. The SGD momentum update becomes:
```
v_t = momentum * v_{t-1} + 0 = 0.9 * v_{t-1}
```
After `n` steps without re-entry: `v_n ≈ v_0 * 0.9^n → 0`.

With `momentum=0.9`, a position has near-zero effective momentum after:
- 10 steps: 34% of original
- 20 steps: 12% of original
- 40 steps: 1.5% of original

**Conclusion**: if positions cycle out of top-t for more than ~20 steps before re-entering,
decouple's "warm restart" advantage has completely evaporated. In practice:
`decouple ≈ hard` for any position that was inactive for >20 steps.

**Freeze** avoids this by stopping time: momentum is restored exactly on re-entry regardless
of how long the position was inactive. This is only better when positions *do* re-enter —
if positions never re-enter top-t, freeze and hard are equivalent.

**Implication**: the fact that freeze beats decouple significantly suggests that
**top-t positions cycle with a period longer than ~20 steps**. This is testable
via the slot overlap / Jaccard analysis (see Planned Experiments).

---

### Theory 2: Why soft outperforms decouple (benign drift as implicit warm-restart)

Soft allows residual momentum drift for ~20 steps after a position exits top-t.
This slow drift is the opposite of what decouple does (param frozen, momentum decays).
Yet soft beats decouple.

**Possible explanation:** Soft's drift moves non-top-t params in the direction of their
last active gradient signal. When a position re-enters top-t, it has already moved
slightly in the right direction — a form of implicit lookahead. This is only useful
when the loss surface is locally smooth enough that the old gradient direction
is still beneficial.

**Alternative explanation:** The drift slightly changes non-top-t params and thereby
changes the effective loss surface for top-t params, acting as a form of regularization
that improves generalization.

**Counter-argument:** This drift is exactly what the design says causes forgetting.
But the forgetting eval shows *no forgetting at all* across any mode — so the Phase 1
forgetting concern may be overstated at this task/model/scale combination.

**Status: unconfirmed**. Need slot overlap analysis and longer training runs to distinguish
these explanations.

---

### Theory 3: Why hard is worst on both Phase 1 and Phase 2

Hard zeros momentum before each step. When a position re-enters top-t, it starts with
zero momentum — a cold start that requires ~10 steps to rebuild useful momentum.

**Cold-start tax**: if positions cycle in/out frequently, hard spends a disproportionate
fraction of its budget rebuilding momentum rather than making progress.

**Why is hard also worst on Phase 1 retention?**

This is the most surprising result. Hard should be the most conservative mode.
A possible explanation: the cold-start creates noisy, underdamped updates when positions
re-enter top-t. These noisy updates may briefly perturb neighboring positions in the
value tensor through the attention mechanism, causing more structural disruption than
the smoother updates of other modes.

**Status: speculative**. The QA ppl differences are small enough that this could be noise
given n=5 eval batches.

---

### Theory 4: The QA improvement paradox

All Phase 2 modes slightly *improved* Phase 1 QA perplexity. This seems wrong — Phase 2
training should at best not affect Phase 1.

**Possible explanations:**
1. **Task overlap**: QA and MT are both question-answering tasks on the same Qasper corpus.
   Phase 2 may be training cache positions that are actually useful for Phase 1 too
   (positions with high IDF = rarely used in Phase 1 but general enough to help).
2. **Continued regularization**: Phase 2 training with SGD + sparse mask may act as
   regularization that improves generalization on the eval set.
3. **n=5 noise**: The QA eval has only 5 batches. A 0.16 ppl difference on 5 examples
   is within noise margin. **Do not over-interpret the improvement direction.**
4. **Eval set leakage**: If Phase 2 training data and Phase 1 eval data share documents
   or topics, Phase 2 training may directly improve Phase 1 eval.

**Status: unclear**. Priority action is to run forgetting eval on a larger eval set
(not just n=5) before drawing conclusions about the direction of forgetting.

---

### Theory 5: Train perplexity oscillation — LR too high

All 4 momentum modes show identical train perplexity behavior: violent oscillations
between ~1.4 and ~1.75 across all 500+ steps, no convergence trend. This is the same
regardless of momentum mode.

**Key observation**: the mode-independence of the train curve means the instability
is in the **gradient signal** or **learning rate**, not in the momentum handling.

**Most likely culprit**: LR = 2.0 for SGD. The cosine schedule decays to 0.1 × 2.0 = 0.2
at step 500 — still aggressive for KV-cache values. The packed-sequence packing with
per-step TF-IDF mask changes creates a highly non-stationary effective loss surface,
making stability harder.

**What the oscillations cost**: all 4 runs reached good eval ppl despite the oscillation,
suggesting the optimizer is not diverging. But the noisy trajectory means convergence
is slower and the final checkpoint may not be at the true minimum. Better LR could
yield meaningfully lower perplexity.

**Status: testable**. Planned: run freeze with lr=0.5 and lr=1.0.

---

## Open Questions (Unresolved)

### Q1: How stable is the top-t selection across steps?

The freeze-beats-decouple result implies high position turnover (>20 steps between
re-entries). But we have not measured this.

**How to test**: load `sparse_slot_log.pt` from any Phase 2 run and compute:
```python
# Jaccard similarity between consecutive steps
def jaccard(a, b):
    a, b = set(a.tolist()), set(b.tolist())
    return len(a & b) / len(a | b)

# Load and compute
log = torch.load("sparse_slot_log.pt")
similarities = [jaccard(log[i], log[i+1]) for i in range(len(log)-1)]
print(f"Mean Jaccard: {np.mean(similarities):.3f}")
```

If mean Jaccard < 0.5, positions are volatile → confirms Theory 1.
If Jaccard > 0.8, positions are stable → freeze and decouple should be similar,
and the gap is explained by something else.

### Q2: Is the QA perplexity eval reliable with n=5?

The forgetting eval ran on only 5 eval batches. This makes the QA ppl comparison
statistically marginal. The QA ppl range across modes is only 0.06 — likely noise.

**How to test**: Extend the QA eval dataset to have at least 50–100 batches.
Check `qasper_eval_QA.parquet` for how many examples it contains.

### Q3: Does soft's outperformance of decouple hold with lower LR?

At LR=2.0, soft's residual drift is very small per step (gradient ≈ 0 → drift ≈ 0.9^n × v0).
At lower LR, all updates are smaller, so the relative contribution of drift vs. gradient
changes. The relative ordering of soft vs. decouple may reverse at lower LR.

### Q4: Are non-top-t positions truly unchanged under freeze?

The implementation saves and restores non-top-t param values after each step.
But are there any numerical precision issues in the save/restore cycle at bfloat16?
Accumulated tiny rounding errors over 500+ steps could create subtle drift.

### Q5: Is the MT perplexity eval also too small?

The MT eval has n=5 batches too. The MT ppl differences (freeze: 24.92 vs hard: 26.07,
delta = 1.15) are much larger in absolute terms, making them more trustworthy — but
still worth validating with a larger eval set.

---

## Planned Experiments (Prioritized)

### P1: LR ablation — freeze with lr=0.5 and lr=1.0

**Goal**: determine if train curve instability is costing acquisition performance.

**Setup**: `MOMENTUM_MASKING=freeze`, `LR=0.5` and `LR=1.0`, all else unchanged.
Compare:
- Phase 2 MT eval ppl at step 500
- Phase 1 QA eval ppl (forgetting eval)
- Train perplexity curve smoothness

**Expected outcome**: lower LR → smoother train curve → lower final MT ppl.
If the improvement is large, all future experiments should use lr=1.0.

**Cost**: 2 training runs.

---

### P2: Slot stability analysis on existing checkpoints

**Goal**: measure top-t position Jaccard similarity across steps to confirm/deny Theory 1.

**Setup**: no new training needed. Load `sparse_slot_log.pt` from any Phase 2 run
(e.g., the freeze run). Compute consecutive-step Jaccard. Plot distribution over training.

**Expected outcome**: mean Jaccard < 0.5 (volatile top-t) → explains freeze > decouple.

**Cost**: zero compute. Pure analysis.

---

### P3: Larger QA eval set for forgetting measurement

**Goal**: make Phase 1 forgetting measurement statistically reliable.

**Setup**: check the size of `qasper_eval_QA.parquet`. If it has many more rows,
rerun `eval_forgetting.sh` with the full dataset (remove the n=5 limit if one exists,
or create a larger eval parquet).

**Cost**: one re-run of `eval_forgetting.sh`.

---

### P4: EMA of attention scores for position selection

**Goal**: test whether making top-t selection more persistent would let decouple
recover its theoretical "warm restart" advantage.

**Design**: in `CacheAccessTracker`, instead of resetting scores to zero after each step:
```python
# Current: self._scores.zero_()
# Proposed EMA:
self._scores.mul_(alpha)  # decay old scores
# then accumulate new scores as normal
```
Try `alpha=0.8`. This makes the top-t set sticky — positions that were in top-t
recently stay for longer before being replaced.

**Prediction**: with EMA, decouple's performance should improve toward freeze's level
(positions stay active long enough for momentum to actually warm up).
If EMA-decouple ≈ freeze, it confirms that decouple's failure was purely the
position-turnover issue.

**Cost**: code change (~5 lines) + 2 training runs (EMA-decouple, EMA-freeze as control).

---

### P5: Per-layer top-t budgets

**Goal**: test whether a global top-t ranking misallocates budget across layers.

**Hypothesis**: early layers (syntax/structure) and late layers (task specialization)
have different IDF distributions. The global top-t may be dominated by a subset of
layers, leaving others over- or under-updated.

**Diagnostic first**: plot per-layer TF-IDF score concentration (entropy of score
distribution per layer). Load `tfidf_ranking_log.pt` from an existing run.

**If diagnostic confirms layer skew**: implement per-layer top-t budgets in
`_trainable_value_layer_indices` and `apply_gradient_mask_to_cache`:
- `top_t_per_layer = top_t // n_layers` per layer
- Rank positions independently within each layer

**Cost**: diagnostic is zero compute. If promising: code change + 1 training run.

---

### P6: Re-rank every K steps instead of every step

**Goal**: test whether reducing mask-change frequency stabilizes training and
equalizes the freeze/decouple gap.

**Hypothesis**: changing the top-t mask at every step is maximally non-stationary.
Re-ranking every K=5 or K=10 steps means each position stays active long enough for
decouple's momentum to meaningfully warm up.

**Prediction**: at K=10, decouple should approach freeze performance. The train curve
may also become smoother (lower oscillation) since the effective loss surface changes
less frequently.

**Cost**: small code change in the training loop + 2-3 training runs.

---

### P7: freeze_keys=False ablation (the remaining half of the 8-experiment grid)

**Goal**: determine whether allowing key updates helps or hurts.

**Current state**: all 4 momentum mode runs used `freeze_keys=True` (value-only).
The 8-experiment grid includes 4 `freeze_keys=False` variants — none have been run
with the current setup (top-t=512, lr=2.0, 10 epochs).

**Expectation**: `freeze_keys=False` allows the attention routing to change,
giving more expressiveness but higher forgetting risk. With our TF-IDF position
selection (which already tries to avoid Phase 1 positions), key updates may be
safe enough to improve Phase 2 acquisition.

**Priority**: lower than P1–P3 because the current value-only approach is already
working well.

---

## Architecture Change Ideas (Longer-Term)

### Idea A: Hierarchical sparse updates (layer groups)

Instead of a flat position space shared across all layers, divide layers into groups
(early / middle / late) and compute separate TF-IDF rankings per group with
allocated budgets. This respects the functional heterogeneity of the transformer:
- Early layers: syntactic features — likely need less Phase 2 updating
- Middle layers: semantic content — main locus of task-specific knowledge
- Late layers: task output formatting — might need Phase 2 updating for new task format

**First step**: run the per-layer diagnostic (P5) to see if the assumption holds.

---

### Idea B: Persistent top-t with re-ranking on demand

Instead of always re-ranking at every step, maintain a "hot set" of positions that
remain in top-t until their TF-IDF score drops below a threshold. Re-rank only when:
1. Every K steps (periodic)
2. When a position's score drops below threshold (adaptive)

This is a generalization of P6 and would naturally let decouple's momentum warm up
without any other changes.

---

### Idea C: Gradient accumulation across mask changes

Current design: accumulate gradients within a step, then apply mask once at `do_step`.
The mask comes from the same step's attention scores — correct.

Alternative: accumulate gradients over N steps *with the same mask*, then update.
This would be N steps of gradient accumulation with a frozen mask. The top-t set
would only change every N steps. Equivalent to P6 but implemented differently.

Trade-off: mask becomes less responsive to data variation, but each position gets
N gradient steps before the mask changes — more stable convergence per position.

---

## Known Experimental Caveats

1. **n=5 eval batches**: Both QA and MT eval sets have only 5 batches. The MT ppl
   differences are large enough to be credible; the QA ppl differences are not.

2. **One random seed**: all Phase 2 runs used the same Phase 1 checkpoint as starting
   point. Variance across seeds is unknown.

3. **One task pair (QA → MT)**: all experiments use the same Phase 1 (QA) → Phase 2 (MT)
   transfer. It is unknown whether the forgetting pattern changes for other task pairs
   (e.g., QA → SA, SA → MT).

4. **Fixed top-t=512 on n=1023 tokens**: top-t is ~50% of total tokens. This is a
   very high sparsity (50% update rate). Most of the "sparse" protection comes from
   the IDF weighting, not from the sparsity itself. Testing top-t=128 and top-t=256
   would show how much protection the sparsity level adds.

5. **Mislabeled hard run**: the run `10107dc6` (2026-06-03-21-19-48) has
   `name: qasper_phase2_soft` in its config but `momentum_masking: hard`.
   This was excluded from analysis. When submitting future grid runs, verify
   that the wandb name and `MOMENTUM_MASKING` env var agree.
