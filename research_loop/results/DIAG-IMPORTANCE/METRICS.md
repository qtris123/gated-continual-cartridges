# DIAG-IMPORTANCE — metric definitions (reproduction guide)

Everything below is measured **on the untouched Phase-1 cartridge**
`outputs/phase1_selfdistill_qwen512/cache_last.pt` (Qwen3-4B-Instruct-2507, 36 layers,
8 KV heads × 4 query heads per group, head_dim 128, 512 cartridge slots = 1 frozen sink
+ **511 writable**), over the two eval parquets
`data/qasper/eval/qasper_eval_{QA,MT}.parquet` packed at `packed_seq_length=2048`
(`LossEvalDataset`, seed 0 → QA 78 examples / 6 packed batches, MT 69 / 5).

No source file under `cartridges/` was edited. No cache was written. **`gradient_steps = 0`**
— the Fisher pass calls `backward()` only to read `.grad`; no optimizer is ever constructed
and the cache parameters are never updated.

Single entry point: `research_loop/results/DIAG-IMPORTANCE/measure_slot_importance.py`
(launched by `launch_diag_importance.sh`, which claims one GPU by `flock` and pins
`PYTHONPATH` to the frozen `git archive HEAD` snapshot `/tmp/amsnap_DIAG-IMPORTANCE`,
RUNBOOK §9c-bis).

---

## 0. The shared attention object

All attention-derived scores come from **one** quantity, captured in
`collect_attention()` with the same post-RoPE q/k hook that DIAG-ROUTING and ORACLE-WRITE
use (`install_qk_capture`, mirroring `cartridges/sparse_cache_finetuning.py::install_query_capture_hooks`):

```
s   = [ q Kcartᵀ ‖ q Kseqᵀ(causal, same-element) ] / sqrt(d)     # 512 slots ‖ own prefix
w   = softmax(s, dim=-1)                     # the model's REAL eval-time distribution
m   = w[..., :512].sum(-1)                   # = mass_on_S at top_t=512
a   = w[..., :512] / m                       # = the design matrix X at value_solve.py:86
```

`a` is exactly DIAG-ROUTING's routing vector (a softmax restricted to a subset is the
renormalised subset softmax), which is why this file reproduces DIAG-ROUTING's per-head
QA/MT top-32 overlap, mean-routing cosine and histogram intersection as a cross-check
(`cross_validation_vs_DIAG_ROUTING` in the JSON).

**Per-layer aggregation.** The incumbent ranker's granularity is `per_layer`
(`GRANULARITY=per_layer`), so every score is summed over the 8 KV heads *and* the 4 query
heads in each GQA group, i.e. over all 32 query heads × all query token positions. Slot 0
(the frozen sink) is dropped from every reported array; arrays are `(36 layers, 511 slots)`.

**Per-example bookkeeping.** Every sum is accumulated into per-eval-example buckets
(`index_add_` on the global element id, `ds.batches[bi][batch.element_ids]`), so all
statistics can be recomputed on a bootstrap resample of examples without a second forward
pass.

---

## 1. `tf_mass_qa` / `tf_mass_mt` — the incumbent ranker's signal

```
tf_mass_X[l, j] = (1/N_X) Σ_{q ∈ split X} a_j(q)        N_X = 32 query heads × #tokens
```
Mean attention mass on slot *j*, in DIAG-ROUTING's convention. Code: `collect_attention`
(accumulator `S1`) → `build_metrics` key `tf_mass`. Within a layer this is a positive
rescaling of the ranker's `tf = scores/scores.sum(-1)` (`cartridges/am/ranking.py:29-31`,
`CacheTFIDFRanker.rank_positions`), so it induces the same ordering and the same top-t set.

**`ranker_tf_qa` / `ranker_tf_mt` (secondary, exact).** Because the production ranker does
*not* use `a`, the exact incumbent score is also computed:
`softmax( mean_over_GQA_group(q) · trainable_keysᵀ / √d )` summed over KV heads and
positions — the **GQA-group mean query taken before the softmax**, over the **511 trainable
keys only** (no frozen slot, no prefix). That is `cartridges/am/query_accum.py:80-95`
verbatim. Code: `collect_attention` accumulator `RK`. The "incumbent selection" used in
every overlap number is per-layer top-32 of `ranker_tf_mt` — the closest possible stand-in
for what the ranker would pick — and is separately checked against the canonical run's
**actual** saved per-document selections (`am_doc_*.pt` → `GradientMask.positions_per_layer`).

**`w_mass_qa` / `w_mass_mt` (secondary).** The same mean taken on `w` instead of `a`
(mass as a fraction of the *whole* softmax including the prefix), accumulator `S4`.

## 2. `entropy` — how broadly a slot is used across QA queries

Treat slot *j*'s attention across the query population as a distribution
`p_j(q) = a_j(q) / Σ_q a_j(q)`:

```
H_j = − Σ_q p_j(q) log p_j(q) = log s_j − t_j / s_j
      s_j = Σ_q a_j(q)              (accumulator S1)
      t_j = Σ_q a_j(q) log a_j(q)   (accumulator S2, torch.xlogy)
```
This is **exact**, not binned: `s_j` and `t_j` are both plain sums over query rows, so the
entropy of any subset of examples (including a bootstrap resample) is recoverable from the
per-example sufficient statistics. Reported in nats; `entropy_norm_qa` divides by `log N`
(the maximum, attained by a slot used equally by every query). Low `H` = narrowly used
(a few queries own the slot); high `H` = broadly used. Code: `build_metrics` key `entropy`.

## 3. `kl_loo` — leave-one-out irreplaceability

Delete slot *j* from the attention distribution and renormalise:
`a^{(−j)}_i = w_i / (1 − w_j)` for `i ≠ j`. Then

```
KL( a^{(−j)} ‖ w ) = Σ_{i≠j} (w_i/(1−w_j)) log( (w_i/(1−w_j)) / w_i ) = − log(1 − w_j)

kl_loo[l, j] = (1/N) Σ_q − log1p( − w_j(q) )        (accumulator S3)
```

Two things to be honest about:

* **The KL itself is exact**, not approximated — the closed form falls out because
  renormalisation multiplies every surviving entry by the same constant. (The reverse
  direction `KL(w ‖ a^{(−j)})` is `+∞` by construction, which is why the forward direction
  is used.)
* **The approximation is structural, and it is first-order:** the removal is *not*
  propagated. We do not re-run the network with slot *j* ablated, so this measures the
  local perturbation of one layer/head's attention distribution and ignores every
  downstream effect (changed attention *output* → changed residual stream → changed queries
  at layers l+1…35). It also ignores what the slot's *value* contains: two slots with equal
  `w_j` get equal `kl_loo` even if one holds a value identical to the cartridge mean and the
  other holds a unique one. Note the consequence: since `−log(1−x) = x + x²/2 + …`, `kl_loo`
  is a **convex reweighting of `w_mass`** and is therefore expected to rank-correlate with
  attention mass almost perfectly; it differs from the mean only by up-weighting queries on
  which the slot is individually peaked. Read a high `tf_mass ↔ kl_loo` correlation as an
  algebraic identity, not as evidence.

## 4. `fisher` — empirical diagonal Fisher of the QA loss w.r.t. the slot's value

```
L_e   = (1/t_e) Σ_{scored tokens of example e} ce_by_token        # DIAG-NOISE's per-example loss
fisher[l, j] = (1/E) Σ_{e=1..E} Σ_{h=1..8} Σ_{c=1..128} ( ∂L_e / ∂ v[l, h, j, c] )²
```

`ce_by_token = −p·log q` is copied token-for-token from
`cartridges/train.py::evaluate_perplexity` (lines 948-958), and the per-example bucketing is
DIAG-NOISE's (`ds.batches[bi][batch.element_ids[topk_token_idxs]]`); the pass prints the
recovered micro-average `Σc_e/Σt_e` so it can be compared against the harness's published
`Eval loss`. One `backward()` per eval example (78 QA, 69 MT) against
`cache.trainable_values[l]` (shape `(1, 8, 511, 128)`), squared and summed over heads and
head dims → one scalar per slot. **This is a diagnostic gradient: no optimizer is created,
`.grad` is zeroed after each read, and the cache is never modified.** Code: `collect_fisher`.

Caveats. (i) It is the *empirical* Fisher (gradient of the observed loss), not the true
Fisher (expectation under the model's own predictive distribution) — the standard EWC-style
approximation. (ii) The forward/backward runs in the eval harness's own
`autocast(bfloat16)` path, so the gradients carry bf16 rounding; the `split_half_reliability`
entry bounds sampling + numerical noise together, and `FP32_CHECK=1` recomputes one packed
batch with model and cache in float32 as a precision control. (iii) `E = 78` is small; every
Fisher-derived number is reported with a bootstrap CI over examples.

## 5. `redundancy` — linear reconstructibility of a slot's value from the others

For layer *l*, stack the slot value vectors with the KV heads **concatenated**:
`V ∈ R^{512 × 1024}`, row *j* = `[v[l,0,j,:] ‖ … ‖ v[l,7,j,:]]`. (Concatenating heads is
what makes the question non-degenerate: per head the vectors live in R^128 and 511 of them
span the whole space, so the residual would be identically zero.)

```
r_j² = min_c ‖ v_j − Σ_{i≠j} c_i v_i ‖²   s.t. the sum excludes j
     = 1 / (G⁻¹)_{jj}          with G = V Vᵀ + λI,  λ = 1e-6 · mean(diag(V Vᵀ))
redundancy[l, j] = 1 − r_j² / ‖v_j‖²   ∈ [0, 1]
```

The identity is the standard one: minimising `uᵀGu` subject to `u_j = 1` (with `u = e_j − c`)
gives `u = G⁻¹e_j /(G⁻¹)_{jj}` and optimum `1/(G⁻¹)_{jj}`; `redundancy` is then exactly the
`R²` of regressing slot *j* on all other slots (uncentred, no intercept). One 512×512
double-precision inverse per layer. The frozen sink is available as a regressor but is not
reported. λ is a numerical guard only — `redundancy_ridge_sensitivity` in the JSON reports
the same array at λ_rel ∈ {1e-8, 1e-6, 1e-4} (the Gram's condition number is 3.5e3–2.2e4, so
the three agree to ~4 decimal places). Split-independent: it is a property of the cartridge
values, not of any query distribution. Code: `compute_redundancy`. `resid_norm` (= `r_j`) and
`value_norm` (= `‖v_j‖`) are dumped alongside.

## 6. `contrast` — PMI-style MT-over-QA preference

```
contrast[l, j] = log( tf_mass_mt[l, j] / tf_mass_qa[l, j] )
```
Positive = MT queries route to the slot more than QA queries do, in log-ratio terms; this is
the only *asymmetric* signal the measured routing geometry leaves available (DIAG-ROUTING:
mean-routing cosine 0.99899, top-32 overlap 0.914). Code: `main()`, key `contrast`.

---

## Derived quantities

**Spearman matrices.** `spearman()` = Pearson on average ranks (`scipy.stats.rankdata`).
`spearman_pooled` ranks all 36×511 = 18 396 (layer, slot) pairs jointly;
`spearman_per_layer_mean/sd` rank within each layer and then summarise over the 36 layers.
Pooled values are inflated by between-layer scale differences — **the per-layer numbers are
the honest ones** for a per-layer selector.

**Top-32 overlaps.** `top32_overlap_vs_incumbent_eval_ranker[m]` = mean over layers of
`|top32(ranker_tf_mt) ∩ sel32(m)| / 32`, where `sel32(m)` takes each metric in its
*acquisition-favouring* direction (recorded per metric in the `direction` field):
descending for `tf_mass_*`, `redundancy`, `contrast`; **ascending** for `entropy`, `kl_loo`,
`fisher` (least-loaded slots first). `mean_if_taken_descending` gives the other direction.
`top32_overlap_matrix_descending` is the all-pairs matrix with every metric taken
descending. `top32_overlap_vs_actual_run_selections` compares against the canonical
top-32 run's own 16 saved per-document masks instead of the eval-query reproduction.

**Safe acquisition budget.** For the set the new task wants (per-layer top-32 by
`tf_mass_mt`, and again by `ranker_tf_mt`), and for each QA-importance metric *m*:
`not_in_qa_top32` = fraction of that set outside *m*'s own per-layer top-32 most-important
slots (the direct analogue of the ~9% that DIAG-ROUTING's 0.914 overlap implies);
`in_qa_safest_32` / `in_qa_bottom25pct` / `in_qa_bottom50pct` = fraction lying in the safest
32 slots / safest quartile / safest half under *m*.

**Concentration.** Per layer, the fraction of the metric's total mass carried by its top
32 / 64 / 128 slots, plus the participation ratio `(Σx)²/Σx²`.

**Attenuation correction.** `attenuation_corrected_spearman` divides each observed Spearman by
`sqrt(rel_x · rel_y)`, where `rel` is the split-half reliability corrected to full sample size by
Spearman–Brown (`2r/(1+r)`). It is an **upper bound** on the true correlation — the number to
quote when asking "could the observed 0.58 really be a 0.9 blurred by noise?". `redundancy` is
deterministic, so its reliability is 1.

**Joint availability & the selector trade-off** (added read-only after the GPU run, from the npz;
same data, no new forward passes). `joint_availability[want][safety]` counts, per layer, how many
slots are simultaneously in the top-{32,64,128} of what the new task wants and in the safest
{25%, 50%} of a QA-importance metric; the `chance_*` entries give the value expected if the two
were independent. `selector_tradeoff_table` evaluates four concrete 32-slot-per-layer selectors on
two axes at once: `frac_writable_MT_routing_mass` (write bandwidth — the share of the 511 writable
slots' mean MT routing mass that the selection captures) and `frac_total_QA_Fisher_mass`
(retention exposure — the share of that layer's total QA Fisher mass sitting in the selection).
`bandwidth_under_a_fisher_constraint` is the same comparison stated as ratios.

**Uncertainty.** `NBOOT` (default 500) bootstrap resamples **over eval examples** (QA and MT
resampled independently), recomputing every metric from the per-example sufficient
statistics and re-deriving the correlations, overlaps and budgets on each resample; 95%
percentile intervals. `redundancy` is deterministic and carries no example-bootstrap
interval. `split_half_reliability_spearman` splits the examples in half and correlates the
two halves' scores — a ceiling on how well any of these metrics is even estimated at this
eval size. Loss-scale noise (DIAG-NOISE: ±0.049 MT / ±0.054 QA paired) does **not** apply to
these statistics; they get their own intervals.

## Raw arrays

`research_loop/state/diagnostics/DIAG-IMPORTANCE.npz` holds `score_<name>` `(36, 511)` for
all six metrics plus the secondaries, `spearman_pooled` `(7, 7)`, `spearman_per_layer`
`(36, 7, 7)`, `top32_overlap_matrix_descending`, `ovl_vs_incumbent_<name>` `(36,)`, the
per-head mean routing vectors `(36, 8, 512)` for the DIAG-ROUTING cross-check, the
**per-example** Fisher tensors `(E, 36, 511)`, and the two redundancy ridge variants.
