# LITREV-04 — Algorithms & mathematics for *measuring* overwriting / interference

**Scout:** W2 (literature) · **Date:** 2026-07-29 · **No GPU, no source edited, no commit.**
**Grounding read:** `notes/2026-07-29-am-investigation-synthesis.md`, `research_loop/GLOSSARY.md`,
`research_loop/results/DIAG-IMPORTANCE/METRICS.md`, `cartridges/am/ranking.py`,
`cartridges/am/value_solve.py`, `research_loop/results/DIAG-IMPORTANCE/result.json`.
**20 papers read** (list at L04-1 … L04-20). **Read-only re-analysis** of
`research_loop/state/diagnostics/DIAG-IMPORTANCE.npz` — every new number below is reproducible from
that file with the snippet in the appendix; nothing was written, no GPU touched, `gradient_steps = 0`.

This thread was asked to audit the **instrument**. It found three estimator faults, two of them of the
`kl_loo` kind (algebraic, not empirical), and one free estimator that dominates two of our six metrics.

---

## Executive summary — what we measured wrong or incompletely

1. **`redundancy` uses the wrong Gram matrix, and the literature names the right one.** Our score is
   `1 − r_j²/‖v_j‖²` with `G = VVᵀ` (the **value**-space Gram). The quantity that actually governs
   "damage of overwriting slot *j* when the other slots may be re-solved to compensate" is *exactly* the
   OBS / SparseGPT saliency `‖v_j‖² / (H⁻¹)_jj` with **`H = E_q[a aᵀ]`, the routing second moment**
   (L04-3, L04-4). Same algebraic form, different index space. Two errors compound: (i) the wrong Gram,
   (ii) dividing out `‖v_j‖²`, which is the *only* factor OBS keeps. A corollary is decisive:
   **`redundancy` is not a function of any query distribution** (METRICS.md §5 calls this
   "split-independent" as a *feature*), so it cannot be a QA-importance metric — it is a property of the
   store, not of the read.
2. **"Redundancy is the best gradient-free proxy for Fisher (ρ = −0.648)" is an ecological correlation
   and is false at the granularity the selector runs at.** Pooled −0.648; **within-layer −0.323**;
   layer-level ρ(mean redundancy, mean Fisher) over the 36 layers = **−0.850**, and
   ρ(layer index, mean redundancy) = **+0.752**. The product of the two layer trends
   (0.752 × −0.811) = −0.61 accounts for essentially all of the pooled −0.648. DIAG-IMPORTANCE's own
   METRICS.md warns that pooled numbers are inflated and "the per-layer numbers are the honest ones" —
   but the pooled number is what got quoted in the synthesis, the GLOSSARY and the `ranking.py` docstring.
   Redundancy agrees with Fisher's own safest-32 set on only **25.4%** of slots.
3. **There is a second `kl_loo`-style collapse, and it is a theorem, not a correlation.** *Slot-importance
   collapse lemma:* the cartridge attention output is **linear in the values**, `o(q) = Σ_j a_j(q) v_j`, so
   for any score of the form `S_j = E_q[φ(∂g/∂v_j)]` where `g` depends on the layer's own output,
   `∂g/∂v_j = a_j(q)·(∂g/∂o)` — hence **every first-order slot score is a query-reweighted attention
   mass**, `S_j = E_q[a_j(q)·ψ(q)]` with `ψ` slot-independent. Corollaries: `kl_loo` collapses
   (ψ = 1 + w/2 + …, measured ρ = 0.968 — already known); **MAS (L04-6) collapses** (ψ = 2‖o(q)‖ — so do
   not build it); and at second order the **K-FAC/GGN diagonal collapses to `E_q[a_j²]`** (L04-2).
   Escaping this family requires the routing **covariance** (off-diagonal of `E[a aᵀ]`) or genuine
   downstream propagation. Of our six metrics, four (`tf_mass`, `kl_loo`, `entropy`, `contrast`) are
   functions of the *marginal* of `a_j` alone; only `redundancy` and `fisher` escape, and `redundancy`
   escapes into the wrong space.
4. **The 98.8 s diagnostic Fisher pass bought little that a free statistic already contained.** The
   K-FAC prediction is testable from arrays we already saved: `E[w²] ≈ 2·(kl_loo − w_mass)`, because
   `−log(1−w) = w + w²/2 + …`. Result: **per-layer Spearman(E[w²], fisher) = 0.847 ± 0.056**, versus
   `tf_mass` **0.666 ± 0.121** and `redundancy` **−0.323 ± 0.113**. Agreement with Fisher's safest-32 set:
   **E[w²] 0.753**, tf_mass 0.549, redundancy 0.254. A rank-regression of Fisher on two free numbers
   (rank `E[w]`, rank `CV²`) gives **rank-R² 0.708 ± 0.090** against a reliability ceiling of ≈ 0.95
   (split-half 0.908 → Spearman-Brown 0.952) — i.e. **~74 % of the *reliable* rank variance of the
   diagonal Fisher is already in the free attention pass**.
5. **Diagonal *empirical* Fisher is the wrong curvature estimator at this operating point, and Kunstner
   et al. (L04-1) say exactly why.** The empirical Fisher approximates the true Fisher only when
   residuals are near zero. Our QA CE is **2.2388 nats** (perplexity ≈ 9.4) — a large-residual regime.
   For cross-entropy, `∇_v L = Jᵀ(p − y)`; the empirical Fisher is the GGN weighted by `(p−y)(p−y)ᵀ`
   instead of `diag(p) − ppᵀ`. So `fisher` scores "slots routed to by the QA examples the model gets
   **wrong**", not "slots carrying QA knowledge". With E = 78 examples it is also a rank-≤78 object.
6. **The better estimator makes our negative result *stronger*, not weaker.** Recomputing the
   headline anti-alignment with the GGN-correct safety metric: MT-top-32 ∩ safest-quartile is
   **0.17 slots/layer vs 8.0 at chance = 48× below chance** (reproducing the published Fisher figure
   0.72 / 11.1× exactly as a control). Redundancy's much-celebrated "at chance" (7.33 / 1.1×) is not a
   discovery — **it is the null**, and it is what independence between a value-space score and a
   routing-space set predicts a priori.
7. **`contrast` is our least identifiable metric and has a hard measurement ceiling.** It is a log-ratio
   of two quantities that correlate at ρ_true ≈ 0.99. Propagating the published split-half reliabilities
   (QA 0.9898, MT 0.9945 after Spearman-Brown) gives a reliability for `contrast` of **0.28–0.70**
   depending on ρ_true ∈ [0.997, 0.982], i.e. an **absolute ceiling of |ρ| ≈ 0.52–0.83 on any
   correlation it can ever show**. Its bootstrap CI with `tf_mass` spanning zero is therefore a real
   null, not attenuation — but no future experiment should treat `contrast` as a usable ranking.
8. **The 0.6× "doc-0 optimism" is a known, named, correctable effect.** It is the gap between a
   *one-shot* mask scored against the initial Hessian and a *sequential* mask with **Hessian downdates
   after every removal** — the entire algorithmic content of OBC/SparseGPT (L04-3, L04-4). The
   continual-learning literature says the same thing twice: Synaptic Intelligence (L04-7) accumulates
   importance as a **path integral along the trajectory** precisely because endpoint estimates are wrong
   under drift; online EWC (L04-8) replaces per-task anchors with a **discounted running estimate**
   `F* ← γF* + F` for the same reason. Our metrics are recomputed live but the *projection* was made once
   at doc 0 — that is the mis-specification.
9. **`k ≈ 12` is not a capacity number, and three theories agree.** Linear associative memory capacity =
   number of linearly independent keys (≤ 128/head, 512 slots); modern Hopfield capacity ≈ 2^(d/2) with
   d = 128 (L04-15); superposition/JL gives exponentially many ε-interfering directions (L04-16).
   Every bound is 10–40× above 12. What Hopfield theory *does* name is our exact failure mode:
   patterns that are **not well separated** produce **metastable states that average groups of
   patterns** — which is verbatim the synthesis's "cumulative, largely document-agnostic adaptation".
   The binding quantity is **separation / effective rank of the routing matrix**, not slot count. The
   achievable function class of a value-only write is exactly `rowspace(A)`, so the number of
   independently addressable documents is bounded by the **numerical rank of `A`**, which nobody has
   measured. (First principal angle between mean QA and MT routing: `arccos(0.99899) = 2.58°`.)
10. **Our own headline law does not survive its one out-of-range point.** Fitting MECH-CONSTRAINED's
    controlled dose-response gives `MT = 2.0613 − 0.1575·log(mass)` (Pearson −0.9775 over
    mass ∈ [0.10, 0.28]). Extrapolated to `mass = 1.0` it predicts **MT 2.061 — still above the 2.02
    target**, and reaching 2.02 would need `mass = 1.30 > 1`. But the measured full-support point
    (`ORACLE-WRITE-512`, mass = 1 by construction) is **MT 2.576** (solved control) / **3.237** (oracle) —
    the law is wrong by +0.5 to +1.2 nats at the endpoint, 10–24× the paired resolution. **The curve is
    U-shaped; `log(MT routing mass)` is a local linearisation, not a mechanism, and must stop being
    quoted as if it bounds the family.** Either reading kills the family, but for different reasons, and
    the difference matters for what to try next.

---

## A. The two analytic results, stated precisely

### A.1 Slot-importance collapse lemma (generalises the `kl_loo` finding)

For query `q`, cartridge head `h`, layer `l`, the attention output is
`o(q) = Σ_{j=1}^{512} a_j(q) v_j + (prefix)`, **linear in `v`** with coefficient `a_j(q)`.
Let `g` be any differentiable functional of `o` (the layer's own attention distribution, its output
norm, its output). Then

```
∂g/∂v_j  =  a_j(q) · (∂g/∂o)          ⇒     S_j := E_q[ φ(∂g/∂v_j) ]  =  E_q[ a_j(q) · ψ(q) ]
```

with `ψ` **independent of `j`**. Any such score is a query-reweighted attention mass. Instances:

| score | ψ(q) | status |
|---|---|---|
| `tf_mass` | 1 | the incumbent |
| `kl_loo` = `E[−log(1−w_j)]` | `1 + w_j/2 + …` (convex reweight) | **collapsed** — measured ρ = 0.968, already removed |
| **MAS** `Ω_j = E‖∂‖o‖²/∂v_j‖` (L04-6) | `2‖o(q)‖` | **collapses — do not build it** |
| K-FAC / GGN diagonal (L04-2) | second order ⇒ `E[a_j²]` | **collapses to a convex reweight of mass** |
| `entropy` | — (a dispersion of the same marginal `a_j`) | not independent information; ρ(CV², entropy) = **−0.783** |
| `contrast` | ratio of two marginals | same family, reliability 0.28–0.70 |

**Consequence.** Five of our six metrics live on a two-dimensional manifold of the *marginal* of `a_j`
(its mean and its dispersion). Indeed `E[a²] = E[a]²(1 + CV²)`, and the rank-regression of Fisher on
(rank `E[w]`, rank `CV²`) reaches **rank-R² 0.708** — i.e. the "six independent slot metrics" are close
to two. **To get information that is not attention mass you must use the routing covariance
`E[a aᵀ]` off-diagonals, or propagate the perturbation downstream.** Neither of our metrics does.

### A.2 The redundancy metric is the right formula in the wrong space

Ask the question `redundancy` was *meant* to answer: how much QA damage does overwriting slot *j* do if
the remaining slots may be re-solved to compensate? Erasing `v_j` and compensating with `Δv_i`, `i ≠ j`:

```
Damage_j = min_{Δv}  E_q ‖ a_j(q) v_j − Σ_{i≠j} a_i(q) Δv_i ‖²
```

The target is a scalar function of `q` times the fixed vector `v_j`, so the optimum is rank-1,
`Δv_i = c_i v_j`, and

```
Damage_j = ‖v_j‖² · min_c E_q ( a_j(q) − Σ_{i≠j} c_i a_i(q) )²
         = ‖v_j‖² / (H⁻¹)_jj ,        H = E_q[ a aᵀ ] + λI
```

which is **verbatim the OBS/SparseGPT saliency `w² / [H⁻¹]_qq`** (L04-3, L04-4). Our metric is

```
redundancy_j = 1 − (1/(G⁻¹)_jj)/‖v_j‖² ,     G = V Vᵀ + λI
```

— identical algebra, but `G` is the Gram over **value vectors** rather than **routing columns**, and the
`‖v_j‖²` that OBS *multiplies by* is instead *divided out*. Because `H` depends on the query split and
`G` does not, `redundancy` is structurally incapable of distinguishing QA-safe from MT-wanted; its
"at-chance" intersection with MT-wanted (7.33 vs 8.0) is the null hypothesis, not evidence.

**Sanity check on `redundancy` as a geometry statistic (it is fine at that job):** an i.i.d. Gaussian
`V ∈ R^{512×1024}` gives mean redundancy **0.5003** (theory `511/1024 = 0.4990`); the Phase-1 cartridge
measures **0.7759**, per-layer means spanning **0.659 → 0.961**, with ρ(layer index, mean redundancy)
= **+0.752**. So the cartridge values *are* markedly more collinear than random — but the signal is
overwhelmingly *which layer you are in*, which is exactly why the pooled correlation with Fisher is an
artefact.

### A.3 The free estimator we should have used

`E_q[w_j²]` — the diagonal of the routing second moment — is recoverable from arrays already on disk:

```
kl_loo − w_mass  =  E[w²]/2 + E[w³]/3 + …     ⇒     E[w²] ≈ 2·(kl_loo − w_mass)
```

| predictor of `fisher` (QA) | per-layer ρ | pooled ρ | agreement w/ Fisher-safest-32 | cost |
|---|---|---|---|---|
| `tf_mass_qa` = `E[a]` | +0.666 ± 0.121 | +0.579 | 0.549 | free |
| `w_mass_qa` = `E[w]` | +0.724 ± 0.092 | +0.511 | — | free |
| `kl_loo` | +0.738 ± 0.088 | +0.518 | — | free |
| **`E[w²]` (K-FAC/GGN diagonal)** | **+0.847 ± 0.056** | +0.491 | **0.753** | **free — already saved** |
| `redundancy` | −0.323 ± 0.113 | −0.648 *(ecological)* | 0.254 | 36 × 512³ inverse |
| `entropy` | −0.198 ± 0.139 | +0.137 | 0.225 | free |
| `value_norm²` | +0.252 ± 0.159 | −0.464 | 0.222 | free |
| `‖v‖²·E[w²]` (OBS-diagonal *saliency*) | +0.847 ± 0.048 | +0.151 | — | free |

Note the consistency check in the last row: adding `‖v‖²` does **not** improve the fit to `fisher`,
exactly as the theory says (Fisher is curvature; `‖v‖²` belongs to the *saliency*, not the curvature).

**Corrected trade-off table** (same definitions as `selector_tradeoff_table`; incumbent row reproduces
0.3736 / 0.2959):

| selector (32/layer) | MT bandwidth | QA Fisher exposure | QA GGN exposure |
|---|---|---|---|
| incumbent `ranker_tf_mt` top-32 | 0.3736 | 0.2959 | 0.4298 |
| best-32 within safest 25 % by **fisher** | 0.0451 | 0.0027 | 0.0103 |
| best-32 within safest 25 % by **E[w²]** | **0.0302** | 0.0048 | **0.0024** |
| best-32 within safest 25 % by **redundancy** | 0.1710 | 0.1132 | 0.1424 |
| best-32 within safest 50 % by **redundancy** | 0.2340 | 0.1536 | 0.2051 |

The corrected safety metric costs **more** bandwidth than Fisher for the same protection. The audit
tightens the negative.

---

## B. Papers

### L04-1: Empirical Fisher vs true Fisher — *Limitations of the Empirical Fisher Approximation for Natural Gradient Descent* (Kunstner, Hennig, Balles; arXiv 1905.12558; NeurIPS 2019) — https://arxiv.org/abs/1905.12558
- **What it measures.** The gap between `F̂ = (1/n)Σ ∇ℓ_i ∇ℓ_iᵀ` (empirical Fisher), `F = E_{y~p_θ}[∇log p ∇log pᵀ]` (true/model Fisher) and the GGN `JᵀH_ℓ J`. Key result: `F̂` is *not* a curvature matrix; it equals the Fisher only under (i) **near-zero residuals** and (ii) near-linearity in parameters. For least squares, `H = 2XᵀX` but `F̂ = (2/n)Xᵀ e eᵀ X` — equal only if `e eᵀ ∝ I`.
- **Better than what we used?** It *is* what we used, indicted. Our `fisher` is `F̂`'s diagonal at a point where QA CE = **2.2388 nats** — the large-residual regime the paper says breaks it. For cross-entropy `∇_v L = Jᵀ(p−y)`, so `F̂` is the GGN reweighted by the *residual outer product* rather than the predictive covariance `diag(p) − ppᵀ`. `fisher` therefore ranks slots by "routed to by QA examples the model is currently **wrong** on", which is a different quantity from "carries QA knowledge".
- **Cost.** Their fix (MC-Fisher: sample `ŷ ~ p_θ`, backprop that) costs the **same 147 backward passes / 98.8 s** we already pay — one label swap in `collect_fisher`. The GGN route is cheaper still here (see L04-2): the cartridge output is linear in `v`, so the GGN factorises exactly.
- **Maps to our code.** `measure_slot_importance.py::collect_fisher` (uses `ce_by_token` against the **observed** tokens); `MECH-INFOGATE/compute_slot_fisher.py`; consumed by `ranking.py::load_slot_fisher_scores`.
- **What it would have predicted.** That `fisher`'s split-half reliability would be the worst of the six (measured **0.908** vs 0.98 for the mass family) and that its ordering would be partly residual noise — consistent with `fisher`'s ρ with the *free* GGN proxy being only 0.847 rather than ≈ 1.
- **Verdict: adopt the critique, do not adopt more empirical Fisher.** If a gradient estimate is wanted at all, spend the same 98.8 s on the **MC-Fisher** (one-line change) and report both. Better: skip it — see L04-2.

### L04-2: Kronecker curvature — *Optimizing Neural Networks with Kronecker-factored Approximate Curvature* (Martens & Grosse; arXiv 1503.05671; ICML 2015) — https://arxiv.org/abs/1503.05671
- **What it measures.** Block Fisher per layer as `F_l ≈ E[ā āᵀ] ⊗ E[g gᵀ]` under an independence assumption between activations and backpropagated derivatives; `(A⊗G)⁻¹ = A⁻¹⊗G⁻¹` makes it cheap.
- **Better than what we used? Decisively, and it yields a closed form.** Our "activations" are literally the **routing weights** `a` and our layer is **exactly linear** in `v`, so the Kronecker structure here is not an architectural approximation — it is the true structure of the block, up to the independence assumption. Consequence: the diagonal of the K-FAC Fisher for slot `j`, summed over the 128 head dims, is `E_q[a_j²] · tr(G_l)`, and `tr(G_l)` is **constant within a layer**. **So the "correct" diagonal Fisher of a slot's value is, within a layer, a monotone function of `E_q[a_j²]` alone.** This is the second `kl_loo`-grade collapse and it is analytic. Measured: per-layer ρ(`E[w²]`, `fisher`) = **0.847 ± 0.056** vs `tf_mass`'s 0.666 ± 0.121. The 0.579 pooled `tf_mass↔fisher` correlation the whole B-GATE family was launched on is therefore *understated* as an argument: the right mass statistic (second moment, not first) recovers most of Fisher.
- **Cost at 511 × 36 × 8, no optimizer step.** The `Ā` factor `E[a aᵀ]` is **512×512 per (layer, head)**: `AᵀA` over 393 216 QA query rows ≈ 2·10¹¹ FLOP/layer, ~7·10¹² FLOP total — seconds, fused into the 3.2 s QA attention pass we already run. Storage 37.7 MB (heads summed, matching the per-layer selector granularity). The `G` factor needs the same backward pass we already pay. **The diagonal version costs literally nothing — it is already on disk.**
- **Maps to our code.** `measure_slot_importance.py::collect_attention` — add one accumulator `S5 += aᵀa` next to `S1/S2/S3/S4`. `cartridges/am/query_accum.py:80-95` for the production-side accumulation. In `ranking.py`, a new `safe_metric='routing_second_moment'` is a three-line addition to `_safety_prior`.
- **What it would have predicted.** That Spearman(`tf_mass`, `fisher`) ≈ 0.6 is not "attention mass ≠ importance" but "**first** moment ≠ importance, **second** moment ≈ importance"; and that any selector built on the *difference* between mass and Fisher is mostly chasing `CV²` plus estimation noise. With the corrected metric the anti-alignment goes from 11× to **48× below chance** — i.e. B-GATE's premise was even weaker than measured.
- **Verdict: adopt the diagonal immediately (free); adopt the full Kronecker only if the routing Gram is wanted for L04-3/L04-4 anyway (it is).**

### L04-3: Exact OBS with compensation — *Optimal Brain Compression* (Frantar & Alistarh; arXiv 2208.11580; NeurIPS 2022) — https://arxiv.org/abs/2208.11580
- **What it measures.** Layer-wise `argmin_Ŵ ‖WX − ŴX‖²` under a sparsity constraint. Saliency
  `w_p = argmin_p w_p²/[H⁻¹]_pp` with **`H = 2XXᵀ`**; optimal compensation of the survivors
  `δ_p = −(w_p/[H⁻¹]_pp)·H⁻¹_{:,p}`. ExactOBS is `O(d_row·d_col³)` via Gaussian-elimination downdates of `H⁻¹`.
- **Better than what we used?** This *is* the corrected form of `redundancy` (§A.2): same
  `1/(Gram⁻¹)_jj` algebra, but the Gram is the **input second moment** — and our layer's "input" to the
  value mixture is the routing matrix `A`, not the value matrix `V`. It also keeps the `w²` factor we
  divide away, and — crucially — it is the *with-compensation* damage, which is the right question
  because the AM value solve **is** the compensation step.
- **Cost at 511 × 36 × 8.** One `H = AᵀA` accumulation (see L04-2) + 36 float64 512×512 inverses
  (4.8·10⁹ FLOP, milliseconds) — i.e. **the same order as `compute_slot_redundancy` today**, which
  already does 36 512×512 inverses. Sequential downdates for a greedy mask: 36 × 32 rank-1 downdates of
  a 512×512 inverse ≈ 3·10⁸ FLOP. Negligible. Zero backward passes.
- **Maps to our code.** `ranking.py::compute_slot_redundancy` — replace `Vf @ Vf.T` with the routing Gram
  and return `‖v_j‖²/(G⁻¹)_jj` instead of `1 − r²/‖v‖²`. `value_solve.py:67` already materialises the
  full `alpha` (n × 512) and line 86 slices `X = alpha[:, selected_mask]`; **`alpha.T @ alpha` is one
  extra matmul on a tensor already in memory at exactly the point the write happens.**
- **What it would have predicted.** That a data-free value-space score would land at chance against a
  routing-derived target set (measured: 7.33 vs 8.00). And that a mask chosen **once** against the
  initial Hessian under-delivers against one recomputed with downdates — our **0.6× doc-0 optimism**.
- **Verdict: adopt. This is the single highest-value import in this thread.** It is a drop-in
  replacement for `compute_slot_redundancy` at the same cost, in the right space, with the compensation
  semantics the AM solve actually implements.

### L04-4: One-shot LLM-scale OBS — *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot* (Frantar & Alistarh; arXiv 2301.00774; ICML 2023) — https://arxiv.org/abs/2301.00774
- **What it measures.** Same objective as L04-3 with `H = XXᵀ + λI`, error `ε_m = w_m²/[H⁻¹]_mm`, update
  `δ_m = −(w_m/[H⁻¹]_mm)·H⁻¹_{:,m}`, plus **adaptive mask selection in blocks of B_s = 128 columns**
  rather than a single global mask, and `O(d³_hidden)` via Hessian synchronisation across rows.
- **Better than what we used?** The *adaptive/blocked* mask is the direct answer to our doc-0 optimism.
  SparseGPT's whole point is that a mask fixed up front against a stale `H⁻¹` is measurably worse than
  one re-decided as the Hessian is downdated. Our projection was made on doc 0 and realised
  **0.749 / 0.598 / 0.567 / 0.585 / 0.612** of it across five arms — a mean ≈ 0.62. That is precisely the
  one-shot-vs-adaptive gap.
- **Cost.** `O(512³)` per layer plus 32 downdates — milliseconds per layer, 36 layers. No backward pass.
  The λ ridge is already our `AM_REDUNDANCY_RIDGE_REL = 1e-6` convention.
- **Maps to our code.** `ranking.py::_rank_slot_prior_per_layer` branch `constrained_mass` — the
  candidate set is currently computed **once per document** from a per-slot score; the SparseGPT form
  recomputes the score **after each of the 32 selections** within the document.
- **What it would have predicted.** The 0.6× factor, quantitatively and by name. Also that `mass_on_S`
  (already computed at `value_solve.py:146`) is the right realised statistic, not a projected one.
- **Verdict: adopt the *sequential* selection idea; it is the correction for the 0.6×.** Note the
  honest caveat: our board has already shown that *more* MT mass is not monotonically better beyond the
  observed range (see bullet 10), so this fixes the estimator, not necessarily the outcome.

### L04-5: Cheapest data-aware saliency — *A Simple and Effective Pruning Approach for Large Language Models* (Wanda; Sun, Liu, Bevilacqua, Kolter; arXiv 2306.11695; ICLR 2024) — https://arxiv.org/abs/2306.11695
- **What it measures.** `S_ij = |W_ij| · ‖X_j‖₂` — weight magnitude × input-activation norm, compared
  **per output row**, no weight update, one forward pass, `O(d²)` vs SparseGPT's `O(d³)`. The authors show
  it approximates SparseGPT's `w²/[H⁻¹]` when λ = 0 and only the **diagonal** of `H` is kept.
- **Better than what we used?** It is exactly the diagonal ablation of L04-3 and is therefore the cheapest
  correct thing: **`‖v_j‖ · √E[a_j²]`**. Our `redundancy` uses neither factor correctly; our `tf_mass`
  uses `E[a_j]` and drops `‖v_j‖` entirely. Measured here: `‖v‖²·E[w²]` tracks `fisher` at per-layer
  **0.847 ± 0.048**, the joint best of everything tested.
- **Cost.** Zero beyond one attention forward — and `score_value_norm` is *already* in
  `DIAG-IMPORTANCE.npz`. This is a **two-line** score.
- **Maps to our code.** `ranking.py::_safety_prior`, new metric; `measure_slot_importance.py` already dumps
  `value_norm` and (implicitly, via `kl_loo − w_mass`) `E[w²]`.
- **What it would have predicted.** Wanda's core observation — that magnitude alone misranks when input
  scales vary by orders of magnitude — maps here to: a value-only score (`redundancy`, `value_norm`) and
  a routing-only score (`tf_mass`) each drop half the saliency. Measured `value_norm²` alone: per-layer
  ρ with Fisher only **+0.252**. Neither half works; the product does.
- **Verdict: adopt as the cheap default; keep L04-3 as the exact version.**

### L04-6: Function-sensitivity importance — *Memory Aware Synapses* (Aljundi, Babiloni, Elhoseiny, Rohrbach, Tuytelaars; arXiv 1711.09601; ECCV 2018) — https://arxiv.org/abs/1711.09601
- **What it measures.** `Ω_ij = (1/N) Σ_k ‖ ∂[ℓ₂²(F(x_k;θ))] / ∂θ_ij ‖` — sensitivity of the **learned
  function**, not the loss; label-free, so it can be computed on unlabelled or streaming data. Local
  variant for one layer with ReLU: `g_ij = 2 y_i y_j`, i.e. Hebbian.
- **Better than what we used? No — and here is the proof it cannot be.** Apply MAS at the cartridge
  attention output `o = Σ_i a_i v_i`: `∂‖o‖²/∂v_j = 2 a_j o`, so
  `Ω_j = 2·E_q[a_j(q)·‖o(q)‖]` — **a query-reweighted attention mass**, by the collapse lemma (§A.1).
  MAS would land on the same axis as `tf_mass` and `kl_loo`. It is *label-free*, which is genuinely
  attractive given that our Fisher was scored on the QA **eval** split (an admitted optimistic bound) —
  but the label-freeness buys nothing that `tf_mass` does not already have.
- **Cost.** One backward per example over `‖o‖²`, ≈ the 98.8 s Fisher pass. Wasted.
- **Maps to our code.** Would have gone in `MECH-INFOGATE/compute_slot_fisher.py` as a second scorer.
- **What it would have predicted.** ρ(MAS, `tf_mass`) ≈ 0.95+ per layer, by construction — the same
  outcome as `kl_loo`'s 0.968.
- **Verdict: not worth it. Register it in the GLOSSARY next to `kl_loo` as a second analytic collapse,
  so nobody builds it.** Its *label-free* motivation, however, is a live criticism of our Fisher (see
  METRICS.md's own caveat that `fisher` was scored on the QA eval split).

### L04-7: Importance under drift, part 1 — *Continual Learning Through Synaptic Intelligence* (Zenke, Poole, Ganguli; arXiv 1703.04200; ICML 2017) — https://arxiv.org/abs/1703.04200
*(equations read via the alphaXiv full-text overview of v3; the arXiv PDF did not extract cleanly)*
- **What it measures.** A **path-integral** importance `ω_k^μ = −∫_{t_{μ-1}}^{t_μ} g_k(θ(t)) θ'_k(t) dt`
  accumulated online during learning, normalised by the parameter's own total displacement
  `Ω_k^μ = Σ_{ν<μ} ω_k^ν / ((Δ_k^ν)² + ξ)`, then used in a surrogate quadratic
  `L̃_μ = L_μ + c Σ_k Ω_k^μ (θ̃_k − θ_k)²`.
- **Better than what we used? Structurally yes, for the drift problem.** SI exists precisely because an
  importance estimated **at one point** (EWC's endpoint Fisher) misprices a parameter whose relevance
  changes along the update path. Our `redundancy` *is* recomputed live at every document (good), but the
  quantity being predicted — the fraction of MT routing mass a selection will capture over the whole
  16-document run — was projected once at doc 0. SI's normalisation by `(Δ_k)²` is the exact analogue of
  what we lack: importance **per unit of change actually made**.
- **Cost.** SI proper needs the optimizer trajectory; we take **no** optimizer steps, so the literal
  algorithm does not apply. The transferable piece is the *bookkeeping*: accumulate, per slot,
  `Σ_docs (realised Δ in QA attention output) / (Δv_j)²` from the 16 `cache-after-doc-*.pt` snapshots
  we already save (`SAVE_AFTER_EACH_DOCUMENT=1`). **Zero new forward passes** — pure arithmetic on
  saved caches.
- **Maps to our code.** `cartridges/am/continual.py` (the per-document loop), the
  `cache-after-doc-*.pt` snapshots, `am_doc_*.pt::GradientMask.positions_per_layer`.
- **What it would have predicted.** That a doc-0 projection under-delivers by a roughly constant factor,
  and that the factor is a function of accumulated displacement — matching our tight
  0.567–0.749 realised/predicted band.
- **Verdict: adopt the *measurement* (path-accumulated importance from the saved snapshots), not the
  algorithm.** It costs nothing and directly diagnoses the 0.6×.

### L04-8: Importance under drift, part 2 — *Progress & Compress* / online EWC (Schwarz et al.; arXiv 1805.06370; ICML 2018) — https://arxiv.org/abs/1805.06370
- **What it measures.** Replaces EWC's growing sum of per-task penalties anchored at each `θ*_j` with a
  **single discounted running Fisher** `F*_i = γ·F*_{i-1} + F_i` and one penalty
  `½‖θ − θ*_{i-1}‖²_{γF*_{i-1}}` centred at the **most recent** parameters. Stated motivation: penalties
  anchored at stale parameters become **mis-specified** as the parameters drift.
- **Better than what we used?** It names our exact failure and gives the one-line correction. Our safety
  prior is currently either a **frozen file** (`AM_SLOT_FISHER_PATH` — measured once on the untouched
  Phase-1 cartridge, then used unchanged for all 16 documents) or **fully recomputed** (`redundancy`,
  from the live cache). Neither is the right thing: the frozen one is stale by exactly the amount the
  cartridge has moved; the fully-recomputed one has **no memory of what Phase-1 used to look like**,
  which is the thing we are trying to protect. Online EWC's `γ`-discounted running estimate is the
  interpolation, and `γ` is a single knob.
- **Cost.** Free for the routing-Gram / `E[w²]` family (an EMA over documents); for a Fisher family it
  would need one 98.8 s pass per document = 16 × 98.8 s ≈ 26 min. `gradient_steps` still 0.
- **Maps to our code.** `ranking.py::_safety_prior` — currently returns a fresh or a frozen array; add an
  EMA carried on the config/state across `continual.py`'s document loop.
- **What it would have predicted.** That the **fisher** arm (frozen prior) and the **redundancy** arm
  (live prior) would drift apart over documents in opposite directions — and that neither's doc-0
  projection would hold. Both observed.
- **Verdict: adopt the running-estimate discipline for whatever safety metric survives. Cheap, and it
  is the literature's standard answer to our question 4.**

### L04-9: Subspace capacity — *Gradient Projection Memory for Continual Learning* (Saha, Garg, Roy; arXiv 2103.09762; ICLR 2021 oral) — https://arxiv.org/abs/2103.09762
- **What it measures.** Per layer, build the representation matrix `R^l_τ = [x^l_1 … x^l_{n_s}]` from
  activations, SVD it, keep the smallest `k` with `‖R^l_k‖²_F ≥ ε^l_th ‖R^l‖²_F`, and store those bases
  in `M^l`; new-task gradients are projected orthogonal to `span(M^l)`. Before appending, project out
  what is already there: `R̂^l_τ = R^l_τ − M^l (M^l)ᵀ R^l_τ`. They report that after five dissimilar
  tasks **78 % of the gradient space is already constrained**, and that `|M^l|` is capped by architecture.
- **Better than what we used? This is the measurement that would have pre-empted the whole B-GATE
  family.** GPM asks the right question in the right units: *how many dimensions are left?* Our slot
  metrics ask a per-coordinate question and then take a top-k, which is a **rank-1-per-slot** proxy for a
  subspace question. Applied to our setting: the QA-preserving write must satisfy
  `A_QA[:,S] · Δv = 0`; if `A_QA[:,S]` has full column rank 32 (generic with 393 216 query rows), the
  null space is `{0}` and **exact QA preservation forces a zero write**. GPM's ε-relaxation is what makes
  it non-trivial — and the writable dimension is then `32 − k(ε)`, a *directly measurable number nobody
  has measured*.
- **Cost at 511 × 36 × 8.** One 512×512 SVD/eigendecomposition per layer of the routing Gram from L04-2:
  36 × ~5·10⁸ FLOP. **Seconds. No backward pass.**
- **Maps to our code.** `measure_slot_importance.py::collect_attention` (accumulate `AᵀA`);
  `DIAG-KEYSPACE` measured the analogous thing in *query* space (ρ_key) and found it at its control
  floor — GPM's version in *slot/routing* space has never been run.
- **What it would have predicted.** With a per-head QA/MT top-32 routing overlap of **0.914** and a mean
  routing cosine of **0.99899** (first principal angle **2.58°**), GPM's criterion says the MT
  representation lies essentially **inside** the QA subspace, so `k(ε) ≈ 32` and the orthogonal writable
  dimension ≈ 0. **That is the 11×-below-chance anti-alignment, predicted from the routing geometry
  alone, before any Fisher was computed.**
- **Verdict: adopt as a diagnostic.** It converts our set-overlap statistics into a capacity number with
  units, and it is the cheapest honest test of the project's central premise.

### L04-10: Task interference, exactly — *A Theoretical Analysis of Catastrophic Forgetting through the NTK Overlap Matrix* (Doan, Bennani, Mazoure, Rabusseau, Alquier; arXiv 2010.04003; AISTATS 2021) — https://arxiv.org/abs/2010.04003
- **What it measures.** `O^{τ_S→k} = V_{τ_S}ᵀ V_k`, built from the right singular vectors of the feature
  maps `φ(X^{τ_S})`, `φ(X^k)`. Theorem 1 writes forgetting as
  `Δ^{τ_S→τ_T} = ‖Σ_k U_{τ_S} Σ_{τ_S} O^{τ_S→k} M_k ỹ_k‖²`; Corollary 1 bounds it by the **cosines of the
  principal angles** between the two task subspaces. PCA-OGD stores the top-`d` eigenvectors, reducing
  forgetting to the residual `σ²_{d+1}`.
- **Better than what we used?** It is the theory our set-overlap statistics are a crude proxy for. Our
  "top-32 overlap = 0.914" is a **set** statistic; the NTK overlap matrix is the **angle** statistic, and
  it is what appears in the bound. Also important: it predicts forgetting **and its converse** — high
  overlap means high interference *and* high transfer, which is why our content-free control shows
  28–73 % of MT gain from documents with zero MT content and ICL shows 53.3 %.
- **Cost.** For a value-only write the "feature map" is the routing matrix; the overlap matrix is
  `V_QAᵀ V_MT` from two 512×512 eigendecompositions per layer. **Seconds, no backward pass.**
- **Maps to our code.** Same accumulator as L04-2/L04-9; `DIAG-ROUTING`'s per-head mean routing vectors
  are already in the npz (`mean_routing_per_head_{qa,mt}`), but only the **means** — the second moments
  needed for the angles were never saved.
- **What it would have predicted.** `arccos(0.99899) = 2.58°` for the mean-routing direction and
  `arccos(0.914) = 23.9°` for the set-overlap-implied angle. With principal angles that small, Corollary 1
  puts us at the **maximum-interference** end: any write in the MT direction moves the QA function almost
  fully. The 11×-below-chance intersection is not a surprising empirical fact; it is the set-language
  restatement of a 2.58° angle.
- **Verdict: adopt.** This is the correct formalism for our question 2 and the one that would have
  answered it in advance, from data DIAG-ROUTING had already collected.

### L04-11: Interference as a constraint, and the CL metric set — *Gradient Episodic Memory* (Lopez-Paz & Ranzato; arXiv 1706.08840; NeurIPS 2017) — https://arxiv.org/abs/1706.08840
- **What it measures.** `ACC = (1/T)Σ_i R_{T,i}`, `BWT = (1/(T−1))Σ_{i<T}[R_{T,i} − R_{i,i}]`,
  `FWT = (1/(T−1))Σ_{i≥2}[R_{i−1,i} − b̄_i]`; and the interference constraint
  `⟨g, g_k⟩ ≥ 0 ∀k<t`, enforced by the dual QP `min_v ½vᵀGGᵀv + gᵀGᵀv, v ≥ 0`.
- **Better than what we used? Yes, in framing.** GEM does **not** rank old parameters by importance. It
  imposes an **inequality** that binds only when there is actual conflict, and otherwise takes the *full*
  new-task step. Our `constrained_mass` at `q < 1` removes slots from candidacy **whether or not** writing
  them would hurt QA — an EWC-style always-on quadratic penalty in disguise. The measured monotone
  dose-response (MT 2.2720 → 2.4388 as `q` falls, Pearson −0.9775) is what an always-on penalty on the
  binding axis must do.
- **Cost.** The GEM analogue here is *free of QP*: after selecting the incumbent top-32 (max MT mass),
  solve the value write **subject to** `‖A_QA[:,S] Δv‖² ≤ ε` — a single equality/inequality-constrained
  least squares on the same 32×128 unknowns, i.e. a second ridge solve per (layer, head, document).
  Roughly **2× the existing solve cost** (`solve_s` is currently 170–340 s), no backward pass.
- **Maps to our code.** `cartridges/am/value_solve.py:86-115` — `X_solve`/`Y_solve` already accept
  appended rows for the `DELTA_WEIGHT` trust region; a QA-preservation block is the *same* mechanism with
  `A_QA[:,S]` rows and zero targets.
- **What it would have predicted.** That protecting via **selection** costs bandwidth linearly while
  protecting via **projection** costs it only where the constraint binds — and that with a 2.58° angle
  the constraint binds nearly everywhere, so even the projection form has little room.
- **Verdict: adopt BWT/FWT as reporting metrics immediately** (see "Did we measure the right thing?"),
  and treat the constrained-solve form as the one untried protective mechanism that is *not* a selector.

### L04-12: Gradient conflict — *Gradient Surgery for Multi-Task Learning* (PCGrad; Yu, Kumar, Gupta, Levine, Hausman, Finn; arXiv 2001.06782; NeurIPS 2020) — https://arxiv.org/abs/2001.06782
- **What it measures.** Conflicting gradients = negative cosine; the "tragic triad" is
  **conflict + high curvature + large gradient-magnitude difference**; the fix is
  `g_i ← g_i − (g_i·g_j/‖g_j‖²) g_j`. Theorem 2 gives sufficient conditions:
  `cos φ₁₂ ≤ −Φ(g₁,g₂)`, curvature `ℓ ≥ ξ(g₁,g₂)L`, large enough step.
- **Better than what we used?** As a *measurement*, yes: cosine between the QA and MT descent directions
  in **value space** is a one-number interference test that we never ran, and it is the natural
  companion to our routing-cosine 0.99899. The theory is instructive in the negative direction: PCGrad
  helps only when the cosine is **negative**. Our routing cosine is +0.999 and QA/MT Fisher correlate at
  per-layer **0.874**, so the two tasks are **aligned, not conflicting** — PCGrad's precondition fails,
  and so, by the same logic, does the premise that a gating rule can separate them.
- **Cost.** Two backward passes (one per split) at the cartridge values = 147 backwards ≈ 98.8 s, which
  we already pay; the cosine is then free from the per-example tensors **already in the npz**
  (`fisher_per_example_{qa,mt}` are squared gradients, so a signed version needs one re-run).
- **Maps to our code.** `measure_slot_importance.py::collect_fisher` currently squares and discards the
  signed gradient — **one line** (`accumulate g, not g²`) would have given the QA/MT gradient cosine.
- **What it would have predicted.** Positive QA↔MT alignment ⇒ the "reverse trade" the board discovered
  empirically (protecting QA buys the axis already 0.573 ahead).
- **Verdict: adopt the one-line signed-gradient accumulation next time the Fisher pass runs.** Do not
  adopt PCGrad itself — its precondition is measurably absent.

### L04-13: Representation-overlap measurement — *Similarity of Neural Network Representations Revisited* (CKA; Kornblith, Norouzi, Lee, Hinton; arXiv 1905.00414; ICML 2019) — https://arxiv.org/abs/1905.00414
- **What it measures.** `CKA(X,Y) = ‖YᵀX‖²_F / (‖XᵀX‖_F ‖YᵀY‖_F)`; invariant to orthogonal transforms
  and isotropic scaling, **deliberately not** to arbitrary invertible linear maps. Equivalent form:
  `Σ_ij λ^i_X λ^j_Y ⟨u^i_X, u^j_Y⟩² / normaliser` — alignment of principal components **weighted by
  their eigenvalues**. CCA/SVCCA are shown to be degenerate when `p ≥ n` and fail the layer-matching
  sanity check (1.4 % vs CKA's 99.3 %).
- **Better than what we used? Yes, and it fixes a specific weakness.** Our QA/MT comparison statistics
  (top-32 overlap 0.914, histogram intersection 0.963, mean-routing cosine 0.99899) are all
  **unweighted or first-moment** comparisons. CKA on the two routing Grams `H_QA`, `H_MT` gives a single
  scalar that weights each direction by how much routing energy it carries — which is exactly the axis
  along which a value-only write can act.
- **Cost.** Two 512×512 Grams per layer (L04-2) then a Frobenius product: **36 × ~10⁶ FLOP.** Free.
- **Maps to our code.** `DIAG-ROUTING`; `measure_slot_importance.py::cross_validation_vs_DIAG_ROUTING`.
- **What it would have predicted.** Given cosine 0.99899 between mean routing vectors, linear CKA between
  the QA and MT routing Grams will be **≈ 1**, i.e. the two tasks route through the *same* eigen-directions
  with the *same* weights — no writable direction exists that MT uses and QA does not. This is the same
  conclusion `DIAG-KEYSPACE` reached in query space (ρ_key at its control floor) and it would have been
  reachable in slot space for free.
- **Verdict: adopt as a one-line summary statistic; note explicitly that SVCCA/CCA are *not* usable here
  (p = 512 ≥ effective n per head), which is a mild but real correction to the brief's suggestion.**

### L04-14: Where forgetting lives, and the similarity curve — *Anatomy of Catastrophic Forgetting: Hidden Representations and Task Semantics* (Ramasesh, Dyer, Raghu; arXiv 2007.07400; ICLR 2021) — https://arxiv.org/abs/2007.07400
- **What it measures.** CKA before/after sequential training, layer by layer. Findings: **deeper layers
  are disproportionately the source of forgetting**; lower layers stay stable; EWC and replay both work
  by *stabilising deeper representations*; and — the non-obvious one — **intermediate task similarity
  causes maximal forgetting**, with near-identical and very dissimilar task pairs both forgetting less.
- **Better than what we used?** It supplies the missing *layer-resolved* reading of our own data. Our
  per-layer Spearman(`tf_mass`, `fisher`) ranges **0.471 at L5 to 0.924 at L35**; ρ(layer index, mean
  redundancy) = **+0.752** and ρ(layer index, mean Fisher) = **−0.811**. We have a strong depth gradient
  in every metric and have never used it — every selector spends a **uniform 32 slots per layer**.
- **Cost.** Zero — the layer-resolved arrays are already in the npz.
- **Maps to our code.** `ranking.py::_rank_residual_budget_per_layer` is the *only* existing mechanism
  that allocates a non-uniform per-layer budget, and it uses "access pressure" as the allocator, not
  depth or measured importance. It is currently unused by the AM path's default.
- **What it would have predicted.** That QA and MT sitting at 2.58° apart puts us at the **near-identical**
  end of their similarity curve — the **low**-forgetting, **high**-transfer regime. That is exactly our
  situation (QA 0.573 ahead of budget; content-free documents reproducing 28–73 % of the MT gain). Their
  curve says: with tasks this similar you should **not** expect to buy much by protecting, and you should
  expect the *acquisition* side to be transfer-dominated rather than storage-dominated. Both hold.
- **Verdict: adopt the depth-aware reading.** The concrete unused lever it points at is a
  **non-uniform per-layer budget**, which is orthogonal to every selector tried.

### L04-15: Capacity of the store — *Hopfield Networks is All You Need* (Ramsauer et al.; arXiv 2008.02217; ICLR 2021) — https://arxiv.org/abs/2008.02217
- **What it measures.** Energy `E = −lse(β, Xᵀξ) + ½ξᵀξ + β⁻¹log N + ½M²`; update
  `ξ^new = X·softmax(βXᵀξ)` — **identical to transformer attention** `softmax(QKᵀ/√d)V`. Storage capacity
  is **exponential in the dimension**, `C ≈ 2^{d/2}` (vs 0.14·d classically), for **well-separated**
  patterns, with retrieval in one update. For patterns that are *not* well separated the fixed points are
  **metastable states that average groups of patterns**; β controls separation vs averaging.
- **Better than what we used?** It answers question 3 directly and reframes it. Our store is
  512 slots × 128 dims/head: classical linear-associative capacity = number of linearly independent keys
  (≤ 128/head, 512 slots); modern-Hopfield capacity is astronomically larger. **`k ≈ 12` is 10–40× below
  every capacity bound — it is not a capacity limit.** What Hopfield theory *does* predict is our
  actual symptom: with non-separated patterns you get **averaging, not retrieval**, which is verbatim the
  synthesis's "cumulative, largely document-agnostic adaptation" and "writing a paper in makes the model
  worse at that paper".
- **Cost.** Free — the separation `Δ_i` is a property of the keys, computable from the cache.
- **Maps to our code.** `MECH-BETA` tested exactly the β knob this theory says controls the
  separation/averaging trade — and found β **query-independent**, moving selectivity only 1.043 → 1.051
  while raising `mass_on_S` 4.23×. That is the theory's own prediction of a **failure mode**: raising β
  without raising *pattern separation* moves mass, not selectivity.
- **What it would have predicted.** Given documents whose induced query patterns are near-collinear
  (routing cosine 0.99899 across *tasks*; per-document separation never measured), the retrieval regime
  is metastable-averaging, and adding documents past the point where averaging dominates **degrades**
  rather than saturates — which is the k-curve's shape (minimum at 12, degradation to 16), not a
  saturation at all.
- **Verdict: adopt as the theoretical frame for question 3. Answer: 12 is anomalous versus capacity, and
  is not a capacity number. Measure separation / effective rank instead.**

### L04-16: Interference under sharing — *Toy Models of Superposition* (Elhage et al., Anthropic, 2022) — https://transformer-circuits.pub/2022/toy_model/index.html
- **What it measures.** When a `d`-dimensional space stores `N > d` features, the residual cross-terms are
  **interference**; sparsity is what makes it tolerable (inactive features contribute nothing). The
  Johnson–Lindenstrauss link gives exponentially many almost-orthogonal directions in `d` dims. Feature
  dimensionality `D_i = ‖W_i‖² / Σ_j (Ŵ_i·Ŵ_j)²` measures how "purely" a feature is represented; there
  are sharp phase changes between dedicated / superposed / dropped.
- **Better than what we used? Yes — `D_i` is a strictly better-posed version of `redundancy`.** `D_i`
  measures a direction's *exclusivity of use* against all others, normalised by its own norm, which is
  the quantity `redundancy` gropes at; but `D_i` is defined on the **read directions**, not the stored
  vectors. Also relevant: the theory says interference is governed by **activation sparsity**, and our
  routing is not sparse — mean participation ratio of `tf_mass_qa` is **113 of 511**, i.e. each query
  spreads over ~113 slots. **In superposition terms we are in the dense regime, where interference is
  maximal and superposition is *not* a good deal.**
- **Cost.** `D_i` on routing directions: one 512×512 Gram per layer. Free.
- **Maps to our code.** `ranking.py::compute_slot_redundancy`; `concentration_of_qa_importance` in the
  result JSON (participation ratios).
- **What it would have predicted.** With PR ≈ 113 (not ≈ 1–5) the model is *not* using the cartridge as a
  sparse addressable store, so per-slot allocation reasoning — the entire premise of "don't overwrite
  important slots" — is category-mistaken. The right unit is a **direction**, not a slot.
- **Verdict: adopt the framing (directions, not slots). This is the strongest theoretical argument that
  our slot-level instrument was mis-specified from the start.**

### L04-17: Attention as a sparse addressable memory — *Attention Approximates Sparse Distributed Memory* (Bricken & Pehlevan; arXiv 2111.05498; NeurIPS 2021) — https://arxiv.org/abs/2111.05498
- **What it measures.** Kanerva's SDM read/write maps onto attention: the circle-intersection weighting
  `I(d_v,d,n) ≈ c₁ exp(−c₂·d(p_a,ξ))` is approximately the softmax, with β setting the effective Hamming
  radius. They derive three optimal radii — `d*_SNR` (query fidelity), `d*_Mem` (capacity), `d*_CD`
  (noise tolerance) — which for transformer-scale settings (n = 64, m ≤ 1024) are 11 / 5 / 15, and find
  trained models learn β ∈ [10, 25], interpolating between them.
- **Better than what we used?** It gives a **capacity-vs-fidelity dial with named optima**, which is what
  our β sweep lacked. Crucially it says capacity and query fidelity want *different* β — so "more mass on
  S" (β up) and "cleaner per-document retrieval" are in tension **by construction**, not by accident.
- **Cost.** Free — β is a scalar already exposed (`ENABLE_BETA`, `AM_BETA_BOX`).
- **Maps to our code.** `cartridges/am/key_select.py` (β fitting, MECH-004), `MECH-BETA`.
- **What it would have predicted.** MECH-BETA's result exactly: raising β raises `mass_on_S` (moving
  toward `d*_Mem`) while making both axes worse (moving away from `d*_SNR`). And that β cannot fix
  selectivity, because β is a *radius*, not an *address*.
- **Verdict: adopt as the explanation of the β negative; not a new mechanism.**

### L04-18: Per-example attribution without a Hessian — *Estimating Training Data Influence by Tracing Gradient Descent* (TracIn; Pruthi, Liu, Kale, Sundararajan; arXiv 2002.08484; NeurIPS 2020) — https://arxiv.org/abs/2002.08484
- **What it measures.** `TracIn(z,z') = Σ_{t: z∈B_t} (η_t/b)·∇ℓ(w_t,z')·∇ℓ(w_t,z)`; practical
  `TracInCP` sums over saved **checkpoints**. No Hessian inversion, no convergence assumption (unlike
  influence functions). Last-layer and random-projection approximations make it cheap. Positive =
  proponent, negative = opponent.
- **Better than what we used? For a question we have not been asking, yes — and we already have the
  checkpoints.** The AM run saves `cache-after-doc-*.pt` for all 16 documents. TracInCP over those 16
  checkpoints gives, per document, its signed influence on **QA loss** and on **MT loss** — i.e. our
  per-document BWT and FWT — without any training. That is the direct measurement of "which document's
  write hurt which task", which nothing in the loop has measured; `DIAG-CONTENT` measured it only in
  aggregate (28–73 % content-free).
- **Cost at 511 × 36 × 8.** 16 checkpoints × (78 QA + 69 MT) backward passes ≈ 16 × 98.8 s ≈ **26 min**,
  one GPU, `gradient_steps = 0`. With the last-layer/projection trick, far less.
- **Maps to our code.** `MECH-INFOGATE/compute_slot_fisher.py` (loop over checkpoints instead of one),
  `cartridges/am/continual.py` (`SAVE_AFTER_EACH_DOCUMENT=1`).
- **What it would have predicted.** It is the natural instrument for the two most surprising facts in the
  synthesis — "a solo write into uncontested slots is **+1.101 worse** on its own document than writing
  it among 16" and "71–93 % of a solo write's gain lands on *other* documents' questions". TracIn would
  attribute those cross-document gains **per pair**, turning a puzzling aggregate into a matrix.
- **Verdict: adopt.** (Related, read only at abstract level: Grosse et al., *Studying LLM Generalization
  with Influence Functions*, arXiv 2308.03296, which uses EK-FAC to make the Hessian-inverse form
  tractable at LLM scale — noted for completeness; TracIn is the affordable one here.)

### L04-19: Fisher as a *task* descriptor — *Task2Vec: Task Embedding for Meta-Learning* (Achille et al.; arXiv 1902.03545; ICCV 2019) — https://arxiv.org/abs/1902.03545
- **What it measures.** Embed a task as the **diagonal FIM** of a fixed probe network,
  `F = E_{x,y}[∇_w log p_w(y|x) ∇_w log p_w(y|x)ᵀ]`, filter-averaged. Symmetric distance
  `d_sym = d_cos(F_a/(F_a+F_b), F_b/(F_a+F_b))` predicts taxonomic similarity; asymmetric
  `d_asym(a→b) = d_sym(a,b) − α·d_sym(a,t_0)` predicts transferability. Caveats stated by the authors:
  the diagonal ignores correlations, and embeddings from different probes are incomparable.
- **Better than what we used?** It is the one place in the literature where a **diagonal Fisher is the
  right tool** — as a *task descriptor*, not a per-parameter importance. That is a useful licence: our
  `fisher_qa` and `fisher_mt` arrays, which exist and are unused as a pair, form exactly a Task2Vec
  embedding of the two tasks in cartridge-slot space. Measured here: per-layer ρ(`fisher_qa`,
  `fisher_mt`) = **0.874**, and ρ(`E[w²]_QA`, `E[w²]_MT`) = **0.888**.
- **Cost.** Zero — both arrays are in the npz.
- **Maps to our code.** `measure_slot_importance.py` already computes `score_fisher_mt`; nothing consumes it.
- **What it would have predicted.** `d_sym(QA, MT) ≈ 0` in cartridge space. Task2Vec's own experiments say
  that at that distance you are in the **maximum-transfer, minimum-separability** regime — you should
  expect strong positive transfer (observed: content-free documents give 28–73 % of MT gain) and no
  ability to protect one task from the other by parameter selection (observed: 11×/48×-below-chance).
- **Verdict: adopt as a *reporting* statistic (one number: QA↔MT task distance). Do not adopt diagonal
  Fisher for per-slot importance — see L04-1.**

### L04-20: Hessian-free curvature, honestly costed — *PyHessian: Neural Networks Through the Lens of the Hessian* (Yao, Gholami, Keutzer, Mahoney; arXiv 1912.07145; IEEE BigData 2020) — https://arxiv.org/abs/1912.07145
- **What it measures.** Hessian-vector products by double backprop, `∂(g_θᵀv)/∂θ = Hv`, at the cost of
  **one gradient backprop each**; top eigenvalue by power iteration (10–20 matvecs); trace by Hutchinson,
  `Tr(H) = E[vᵀHv]` with Rademacher `v` (5–10 matvecs); full spectral density by stochastic Lanczos
  quadrature (20–50 iterations × several vectors).
- **Better than what we used?** It bounds the honest price of the exact answer. For us one HVP over the
  QA split ≈ one Fisher pass ≈ **98.8 s**; a top-eigenvalue estimate ≈ **25–33 min**; a full SLQ spectrum
  ≈ **hours**. Against that, the K-FAC/GGN route (L04-2) is analytically exact in structure here (the
  layer is *linear* in `v`) and costs **seconds**. So the verdict is: we do not need Hessian-free methods,
  we need the closed form we did not notice we had.
- **Cost.** As above. `gradient_steps` stays 0 throughout (double backprop constructs no optimizer).
- **Maps to our code.** Would slot into `measure_slot_importance.py::collect_fisher` with
  `create_graph=True`.
- **What it would have predicted.** Nothing specific about our numbers; it is the cost oracle that says
  the expensive route is not worth taking given L04-2.
- **Verdict: not worth it here.** Useful only as the fallback if the Kronecker independence assumption is
  ever contested — and one cheap test of that assumption is the ρ(`E[w²]`, `fisher`) = 0.847 we already
  have (a perfect Kronecker structure would give ≈ the reliability ceiling 0.976).

---

## C. Did we measure the right thing?

**No. "Importance to Phase-1" is a retention quantity, and retention is not the binding constraint.**
QA is **0.573 ahead** of budget; MT is **0.248 short**. Every one of the six metrics is a function of the
QA distribution or of the Phase-1 store. We built a high-precision instrument pointed at the axis we had
already won. The instrument then worked correctly and told us, in three independent ways (11× below
chance, 106×-for-8.1× Pareto, a monotone dose-response with Pearson −0.9775), that there is nothing there.

Four sharper statements, in decreasing order of how much I would stake on them:

**(1) The slot is the wrong unit; the direction is the right one.** By the collapse lemma (§A.1), any
first-order per-slot score is a query-reweighted attention mass. The cartridge is not being used as a
sparse addressable store — participation ratio ≈ **113 of 511** slots per query (L04-16 says that is the
dense, maximum-interference regime) — so "which slots are important" has no sharp answer. GPM (L04-9)
and the NTK overlap matrix (L04-10) ask the question in the right units: **how many directions of the
routing subspace are free?** With a first principal angle of **2.58°** between QA and MT routing, the
answer is "approximately none", and that is derivable from `DIAG-ROUTING`'s own 2026 numbers **without
any Fisher computation at all**.

**(2) The quantity to measure is *marginal acquisition per unit of interference*, conditioned on what has
already been written — not a per-slot prior.** Three literatures converge on this. OBC/SparseGPT
(L04-3/L04-4): greedy selection with a Hessian **downdate** after every choice, because per-item saliency
computed once against the initial Hessian systematically over-promises — our measured **0.6×**. SI
(L04-7): importance must be normalised by the change **actually made**, `ω/(Δ)²`. GEM (L04-11): don't
rank the old task's parameters at all; impose an inequality that binds only on real conflict, and
otherwise take the full new-task step. Our `constrained_mass` is the opposite of all three: a static,
unconditioned, always-on exclusion.

**(3) The quantity we should be *reporting* is BWT/FWT (L04-11), per document.** Our results.csv carries
two endpoint losses. The CL literature's `R` matrix — accuracy on task *j* after learning task *i* — is
what separates "we forgot" from "we never learned", and we have every ingredient: 16
`cache-after-doc-*.pt` snapshots and two eval splits. The k-curve is a **one-dimensional slice** of that
matrix. Filling in the rest is free (evals only) and would immediately answer whether the k≈12
degradation is forgetting of earlier documents (negative BWT) or failure to acquire later ones (low
`R_{i,i}`) — a distinction the project has never made and which changes what to build next.

**(4) The project's own best predictor is over-trusted, and this is an estimator-hygiene failure of the
same class as `kl_loo`.** `log(MT routing mass) → MT loss` at Pearson −0.877 / −0.9775 / −0.986 is quoted
three times in the synthesis as near-proof that the incumbent is optimal. Fitting the controlled arm
(MECH-CONSTRAINED, only `q` moves) gives `MT = 2.0613 − 0.1575·log(mass)` over mass ∈ [0.100, 0.280].
Two things follow:
  - Extrapolated to `mass = 1.0` (capture **all** writable MT routing mass) the law predicts **MT 2.061 —
    still above the 2.02 target**; reaching 2.02 would require `mass = 1.30 > 1`. Under the law, the
    fixed-slot value-write family is **bounded away from PASS**. (Slope sensitivity: a 30 % steeper slope
    would put the requirement at mass = 0.96, so this is suggestive, not a proof.)
  - But the law is **contradicted by the one measured point outside its range**. `ORACLE-WRITE-512` writes
    all 511 writable slots (mass = 1 by construction) and gives **MT 2.5757** (solved control) /
    **3.2370** (teacher-value oracle) — the law is wrong by **+0.515 to +1.176 nats**, i.e. 10–24× the
    ±0.049 paired resolution.
  So the relation is **U-shaped**, `log(mass)` is a local linearisation on its descending branch, and the
  argument "the incumbent maximises MT routing mass by construction, therefore nothing can beat it" only
  establishes that **nothing which *reduces* mass can beat it**. That is a much weaker claim than the one
  in the synthesis, and it leaves a specific gap: mass is confounded with **inter-document contention**
  (at full support the 16 documents "mutually annihilate"). The uncontested quantity — **mass net of
  contention**, i.e. per-document mass on slots not subsequently overwritten — has never been measured,
  and it is the natural object for the GEM/PCGrad pairwise formalism.

**What I would measure instead of "importance to Phase-1":** (a) the **effective rank** of the routing
matrix — the hard bound on how many documents a value-only write can address; (b) **marginal MT mass
conditional on the written union**, sequentially with downdates; (c) **per-document BWT/FWT**. All three
are cheap, none needs training, and each is a bound rather than a heuristic.

---

## D. Top 3 measurements I would run next

**1. Effective rank / eigen-spectrum of the routing second moment `H_l = E_q[a aᵀ]`, per layer, both
splits.** *Cost:* one extra accumulator in the 3.2 s QA attention pass + 36 eigendecompositions of
512×512 — **seconds, no backward pass, `gradient_steps = 0`.** Report the participation ratio
`(Σλ)²/Σλ²`, the `ε = 0.99` k-rank (GPM's criterion, L04-9), and the **principal angles** between `H_QA`
and `H_MT` (L04-10) plus linear CKA (L04-13).
*Rules in/out:* the achievable function class of a value-only write is exactly `rowspace(A)`, so
`r_eff` is a **hard ceiling on the number of independently addressable documents**. If `r_eff ≈ 10–15`,
the k ≈ 12 saturation is **explained and the family is closed by a capacity theorem** rather than by an
exhausted sweep — the strongest possible form of the current negative, and it retires "capacity" as a
hypothesis permanently. If `r_eff ≫ 100`, the saturation is *not* an addressing limit, capacity
reasoning is wrong, and the search should move to the solve/target rather than the slots. Either answer
is decisive; neither costs a GPU-hour.

**2. Swap the Gram: rescore `redundancy` as the OBS routing-space saliency `‖v_j‖²/(H⁻¹)_jj` and
re-derive the whole selector trade-off table.** *Cost:* one `AᵀA` matmul + 36 float64 512×512 inverses —
**the same cost as `compute_slot_redundancy` today**, zero backward passes, purely offline scoring on the
untouched Phase-1 cartridge (no training run needed to get the table).
*Rules in/out:* it decides whether `redundancy`'s failure was the *idea* (protecting Phase-1) or the
*estimator* (wrong Gram). Prediction from §A.3: the corrected metric is a **much better** QA-safety
metric (safest-32 agreement with Fisher 0.753 vs 0.254) and therefore an even **worse** gating metric —
the anti-alignment should deepen from 11× to ≈ 48× below chance and the MT bandwidth at fixed protection
should *fall* (0.171 → 0.030 at the safest quartile). If that is confirmed, B-GATE is closed by a
**correct** instrument rather than a broken one, and the GLOSSARY's "redundancy is the best gradient-free
proxy for Fisher (ρ = −0.648)" must be corrected to "within-layer ρ = −0.323; the pooled figure is an
ecological correlation". If instead the corrected metric *widens* the safe budget, the family reopens.

**3. Fill in the CL `R` matrix from the checkpoints we already have: per-document BWT/FWT (L04-11), plus
signed QA/MT gradient cosine (L04-12).** *Cost:* evaluations only against the 16
`cache-after-doc-*.pt` snapshots — the MT/QA eval is ~1 min each, so ≈ 32 evals ≈ 40 min, **zero
training, `gradient_steps = 0`**; the gradient cosine is a one-line change (accumulate `g`, not `g²`) to
the next Fisher pass we run.
*Rules in/out:* it separates the two hypotheses the k-curve cannot distinguish — **degradation at k > 12
is forgetting of earlier documents (negative BWT)** vs **failure to acquire later ones (low `R_{i,i}`)**.
If BWT ≈ 0 and `R_{i,i}` falls with `i`, then the "don't overwrite" premise is refuted directly by
measurement (there is no overwriting to prevent; there is a saturating acquisition channel), and the
whole gating literature is the wrong shelf. If BWT is strongly negative between *documents* (as opposed
to between tasks), then intra-Phase-2 interference is real and the correct mechanism is per-document
isolation, not QA protection — a target no experiment in the loop has aimed at. The gradient cosine
adjudicates PCGrad's precondition in one number: **positive ⇒ the tasks are aligned and no surgery-style
method can help** (predicted, given per-layer ρ(`fisher_qa`, `fisher_mt`) = 0.874).

---

## Appendix — reproducing every new number in this note

Read-only, CPU, ~5 s. Nothing is written.

```python
import numpy as np; from scipy.stats import rankdata
z = np.load('research_loop/state/diagnostics/DIAG-IMPORTANCE.npz')
sp = lambda a,b: np.corrcoef(rankdata(a), rankdata(b))[0,1]
pl = lambda a,b: np.mean([sp(a[l],b[l]) for l in range(36)])
kl, wq, fis = z['score_kl_loo'], z['score_w_mass_qa'], z['score_fisher']
m2q = 2.0*(kl - wq)                       # E[w^2] from -log(1-w) = w + w^2/2 + ...
pl(m2q, fis)          # 0.8474  <- K-FAC/GGN diagonal, FREE
pl(z['score_tf_mass_qa'], fis)            # 0.6662
pl(z['score_redundancy'], fis)            # -0.3229   (pooled -0.6483 is ecological)
sp(z['score_redundancy'].mean(1), fis.mean(1))   # -0.8497  the layer-level trend
```

| number in this note | value |
|---|---|
| per-layer ρ(`E[w²]`, `fisher`) | **+0.8474 ± 0.0562** |
| per-layer ρ(`tf_mass_qa`, `fisher`) | +0.6662 ± 0.1206 |
| per-layer ρ(`redundancy`, `fisher`) | −0.3229 ± 0.1125 (pooled −0.6483) |
| layer-level ρ(mean redundancy, mean Fisher) over 36 layers | −0.8497 |
| ρ(layer index, mean redundancy) / (layer index, mean Fisher) | +0.752 / −0.811 |
| safest-32 agreement with Fisher-safest-32 | `E[w²]` 0.753 · `tf_mass` 0.549 · `redundancy` 0.254 · `entropy` 0.225 · `value_norm` 0.222 |
| rank-R²(`fisher` \| rank `E[w]`, rank `CV²`) | 0.7079 ± 0.0896 (vs 0.5325 on `E[w]` alone; reliability ceiling ≈ 0.952) |
| ρ(`CV²`, `entropy`) per layer | −0.7833 |
| MT-top32 ∩ safest-quartile (want = `tf_mass_mt`) | `fisher` 0.72 (**11.1×** below chance — reproduces published) · `E[w²]` **0.17 (48.0×)** · `redundancy` 7.33 (1.1×) |
| MT-top32 ∩ safest-quartile (want = `ranker_tf_mt`) | `fisher` 1.31 (6.1×) · `E[w²]` 0.31 (26.2×) · `redundancy` 8.06 (1.0×) |
| redundancy of i.i.d. Gaussian `V(512×1024)` | 0.5003 (theory 511/1024 = 0.4990); measured cartridge **0.7759**, per-layer 0.659–0.961 |
| `contrast` reliability (propagated from published split-half) | 0.28–0.70 ⇒ \|ρ\| ceiling **0.52–0.83** |
| first principal angle, mean QA vs MT routing | `arccos(0.99899)` = **2.58°** (set-overlap-implied `arccos(0.914)` = 23.94°) |
| MECH-CONSTRAINED law | `MT = 2.0613 − 0.1575·log(mass)`, r = −0.9775; at mass = 1.0 → **2.061**; MT ≤ 2.02 needs mass = **1.30** |
| measured at mass = 1.0 (`ORACLE-WRITE-512`) | control **2.5757**, oracle **3.2370** ⇒ law off by +0.52 / +1.18 nats |
| corrected trade-off, best-32 within safest 25 % | `fisher` MT 0.0451 / QAF 0.0027 · `E[w²]` MT **0.0302** / QAF 0.0048 · `redundancy` MT 0.1710 / QAF 0.1132 |

*(The `E[w²]` recovery is a truncation of `−log(1−w) = w + w²/2 + w³/3 + …`; with `E[w] = 4.29e−4` and
`E[w²] ≈ 7.0e−5` the `w³` term is ~4 orders down, and the recovered array is strictly positive
everywhere — min 1.61e−10, 0/18396 non-positive. The exact quantity is one extra accumulator away.)*
