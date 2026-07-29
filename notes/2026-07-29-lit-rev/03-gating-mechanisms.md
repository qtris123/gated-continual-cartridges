# LITREV-03 — Gating & routing mechanisms for continual learning on a compressed KV cache

**Worker:** W2 SCOUT (literature) · **Date:** 2026-07-29 · **No GPU, no source modified, no commit.**
**Grounding:** `../2026-07-29-am-investigation-synthesis.md`, `../../research_loop/GLOSSARY.md`,
`../../research_loop/state/bottleneck_board.md`.
**Baseline this file predicts against:** seed-mean best **QA 1.9468 / MT 2.2681** (k=12 single-seed
QA 1.9560 / MT 2.2720); incumbent MT routing mass **0.2799**; MT/QA selectivity **1.0424–1.0545**
(untouched Phase-1 cartridge **1.0812**); paired resolution **±0.049 MT / ±0.054 QA**;
target MT ≤ 2.02, dense bar MT 1.8725.

**Papers read: 16** (17 arXiv/venue entries; DeltaNet and Fast Weight Programmers share one entry).

---

## The number that governs every prediction in this file

MECH-CONSTRAINED's dose–response is four points of a single-variable sweep. Fitting them:

| `AM_SAFE_FRACTION` q | MT routing mass m | MT loss |
|---|---|---|
| 1.00 (= incumbent) | 0.2799 | 2.2720 |
| 0.75 | 0.1661 | 2.3288 |
| 0.50 | 0.1327 | 2.3693 |
| 0.25 | 0.1000 | 2.4388 |

```
MT  =  2.0613  −  0.1574 · ln(m)          r = −0.9774  (reproduces the board's −0.9775), R² = 0.955
```

Three consequences I will use throughout, all of them **extrapolations of a within-family fit**, not
measurements:

1. **One doubling of MT routing mass is worth −0.109 MT.** Not −0.248. The gap to the bar is 2.25
   doublings, i.e. **4.6× the incumbent's mass ⇒ m = 1.29**.
2. **m is a probability. It cannot exceed 1.0.** The law's own ceiling at *perfect* routing —
   every MT attention query landing entirely on written slots — is **MT = 2.0613**, which is
   **0.041 above the 2.02 target** and 0.189 above the dense bar. Reaching the dense bar would need
   m = e^{(2.0613−1.8725)/0.1574} = **3.32**, which does not exist.
3. Therefore: **no operator that obeys this law can pass, however good its gate is.** The only
   mechanisms worth listing below are ones that *leave the family the law was fitted on*. The law was
   fitted at fixed content (only q moved), and the board already recorded one escape from it — key
   installation raised m 0.0711 → 0.2430 and improved both axes, while β raised m to 0.30 and made both
   worse ("bandwidth-to-the-right-content is a lever; generic bandwidth is not"). A **query-dependent**
   gate changes the content-conditional, so it is outside the fit. That is the whole remaining opening,
   and it is narrow.

---

## Executive summary (10 bullets)

1. **(a) Yes — query-dependent gating formulations exist, and three of them are not bounded by our β
   result.** Quest's per-page criticality `Σ_i max(q_i m_i, q_i M_i)` (L03-8), InduceKV's prefix-conditioned
   retrieval weight `α_i(x) = softmax(⟨r(x), r_i⟩/τ)` (L03-16), and Routing-Transformer cluster
   assignment (L03-7) all make the *selectivity itself* a function of the incoming query. β is an
   additive per-key log-bias with no `q` in it, which is why 4.23× more mass moved selectivity only
   1.043 → 1.051; none of these three has that defect.
2. **(b) Yes — the decisive one is fully gradient-free.** Quest requires **no training at all**: per-slot
   min/max key bounds are computed once from the cartridge, and the score is evaluated per query at
   inference. Routing Transformer's centroids are **EMA k-means, not gradients**. BASE Layers solves a
   **linear assignment problem with the auction algorithm** — a combinatorial solve, not an optimiser
   step. Hash Layers show a **fixed, unlearned router beats a learned one by 0.4–0.6 ppl**. NTM content
   addressing and the delta rule are both closed-form. So "a gate must be learned" is false.
3. **(c) Expert-choice routing does change the overwrite arithmetic — and our own data predicts it
   loses.** Slots-pick-documents with capacity c collapses 9.83 writes/slot to c and survival 4.7% → ~100%.
   But DIAG-PERDOC measured a solo write into **uncontested** slots is **+1.101 worse**, 5/5 documents,
   and ORACLE-WRITE-512 measured that at full support the 16 documents mutually annihilate. Expert-choice
   is worth running as a **falsification test of DIAG-PERDOC**, not as a fix. Predicted MT ≥ 2.45.
4. **The single largest untried structural idea is the direction reversal, and it is cheap.** Every
   selector this project has built is **token-choice** (each document takes its top-32 slots). Nobody
   has run **expert-choice** (each slot takes its top-c documents) or **Soft MoE / Slot Attention's
   softmax-over-slots** (slots *compete* for documents). It is ~30 lines in `ranking.py`. Its value is
   diagnostic: it is the cleanest single-variable probe of "is collision helping or hurting".
5. **Query-independent gates are all bounded by the β result and I am marking them unusable, not
   hopeful.** That is: HAT (L03-11, task-embedding only), Switch/ST-MoE aux-loss balancing (L03-3),
   SPLADE's document-side expansion (L03-10), NSA's per-branch gate `g = σ(MLP(x_t))` insofar as it is
   applied to *our* fixed slot set post-hoc, Hash Layers (L03-5), and InduceKV's per-layer value gate
   `λ_ℓ = σ(φ_ℓ)`. Each of these can only rescale mass, and rescaling mass is exactly the axis the
   0.1574 slope prices.
6. **The delta rule is a query-dependent *write*, and it is the one write-side idea we have not tried —**
   `S_t = S_{t−1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ` erases the value already stored *at that key* before
   writing. But our own ORACLE-WRITE result ("better fitting is anti-correlated with CE: 18× lower MSE →
   +13 loss") predicts it fails here, and DIAG-PERDOC says the interference it removes is load-bearing.
   Candidate with a **negative** prediction (L03-13).
7. **Product-key / NTM addressing gives the mechanism a query-dependent gate would need, but both need a
   learned query network.** PKM's usage collapse (25.8% of slots used without batch-norm) is our
   87.4%-document-blind-top-32 pathology under a different name; PKM's fix (BatchNorm on the query
   network) is a *training-time* fix and does not transfer to a gradient-free write (L03-14, L03-15).
8. **Two 2026 papers solve our exact problem and both do it by giving up the fixed slot set.**
   Cartridges-at-Scale (L03-17) trains cartridges jointly under mixed-visibility and then **concatenates**
   the retrieved ones; InduceKV (L03-16) keeps a fixed *budget* B but the memory is a **retrieved,
   concatenated set of entries** with a query-dependent weight, not a 512-slot cache being overwritten.
   Neither writes 16 documents into the same 512 slots. This is convergent external evidence for the
   synthesis's own "where a win most likely lives: non-fixed-slot formulations".
9. **Honest conclusion the human asked for, stated plainly.** Within a **fixed 512-slot, gradient-free,
   overwrite** formulation, gating cannot reach MT ≤ 2.02: the measured law caps the family at **2.061 at
   100% routing mass**, and the one operator class that escapes the law (query-dependent selectivity)
   would have to more than **quadruple** effective MT mass to close 0.248 — which the mass bound forbids
   *if* it obeys the law, and which nothing in the literature achieves *even when trained*. **Useful
   gating here needs either gradients on a router or a cache that grows/retrieves.** The scope should be
   opened on that basis.
10. **But there is one cheap, decisive experiment before opening scope**, and it is an *eval-time-only*
    oracle: `cartridges/models/attention.py:106` already gives `score_mod(score, b, h, q_idx, kv_idx)`
    a batch index `b` and a query index `q_idx`. A per-example, per-slot bias is a 3-line change and
    **zero** change to the write path. Measure the *ceiling* of query-dependent gating on the cartridge
    we already have before building anything. See "Top 3 experiments".

---

## Papers

### L03-1: Soft MoE (soft, fully differentiable slot assignment) — *From Sparse to Soft Mixtures of Experts* (arXiv 2308.00951, ICLR 2024, https://arxiv.org/abs/2308.00951)
- **Claim / mechanism.** Replaces discrete routing with a soft assignment through `n·p` **slots**
  (the paper's word is literally ours). With `Φ ∈ R^{d×(n·p)}`, dispatch `D_ij = softmax_over_tokens((XΦ)_ij)`
  and combine `C_ij = softmax_over_slots((XΦ)_ij)`; each slot receives `X̃ = DᵀX`, a convex combination of
  **all** input tokens. No token is dropped, no capacity factor, no auxiliary balance loss, and the whole
  thing is differentiable. It outperforms both Tokens-Choice and Experts-Choice on ViT.
- **Is it query-dependent?** **Partly, and in the wrong direction for us.** `D` depends on the *inputs
  being written* (our documents), not on the *queries that will later read* (our MT/QA questions). It is a
  write-side allocation rule, so as a gate over our cartridge it is **query-independent** and inherits the
  β bound. The one genuinely new thing it gives us is the **softmax axis**: normalising over tokens makes
  slots compete for documents, which is the direction reversal we have never run.
- **Gradient-free?** No — `Φ` is learned. But the *degenerate* form is closed-form: set `Φ` to the
  cartridge's own keys, and `D` becomes softmax over documents of the document-to-slot attention logits,
  computable from quantities `query_accum.py` already produces. That amortised variant is what I would
  actually implement.
- **Maps to our code.** `cartridges/am/ranking.py::_rank_slot_prior_per_layer`, new branch
  `SLOT_SELECTION=soft_dispatch`; the top-k mask becomes a soft weight vector consumed by
  `finetune.py::apply_document_am_write_to_cache` as a per-slot write weight (interpolating old value
  and solved value) rather than a hard overwrite. Flag: `AM_SOFT_DISPATCH_TAU`.
- **Prediction about OUR signal.** MT routing mass essentially unchanged (**0.27–0.29**), because the
  support is the same slots with soft edges; selectivity unchanged at **1.04–1.06** (query-independent);
  MT **2.25–2.32**, i.e. inside or slightly worse than ±0.049. The soft write is a *convex blend*, so it
  is strictly less aggressive than the current overwrite, and DIAG-PERDOC says less aggressive collision
  = less acquisition.
- **Why it might NOT transfer.** The paper's own stated limitation is fatal for us: Soft MoE "makes the
  use of Soft MoEs in auto-regressive decoders difficult, since causality … has to be preserved." Our
  eval is autoregressive. Also, Soft MoE buys its wins from *scaling experts*; we have a fixed 512 slots.
- **Verdict: candidate, low priority.** Worth it only as the mechanism that carries the softmax-axis
  reversal; on its own it predicts a null.

### L03-2: Expert-choice routing (slots pick documents, with capacity) — *Mixture-of-Experts with Expert Choice Routing* (arXiv 2202.09368, NeurIPS 2022, https://arxiv.org/abs/2202.09368)
- **Claim / mechanism.** `S = softmax(X W_g) ∈ R^{n×e}`, then `G, I = TopK(Sᵀ, k)`, `P = Onehot(I)` —
  the top-k is taken **over the token axis**, so each expert takes exactly `k = n·c/e` tokens (c = capacity
  factor). Perfect load balance with **no auxiliary loss**. Tokens get a *variable* number of experts
  (measured: most get 1–2, 23% get 3–4) and some get none. An optional entropy-regularised LP variant
  `max_A ⟨Sᵀ, A⟩ + λH(A)` with a per-token cap `b` is solved by Dykstra's algorithm (λ=0.001, ≤100 iters).
- **Is it query-dependent?** No — it is a *write-side* assignment between documents and slots. It does not
  change what an MT question attends to. So by itself it is bounded by the β result. **Its value here is
  entirely about capacity/overwrite, not about selectivity.**
- **Gradient-free?** The routing *rule* is; only `W_g` is learned, and we already have the affinity matrix
  (`tf`, the per-document per-slot normalised access score, `ranking.py:404`). So `S` needs no training:
  `S[d, j] = tf_d[l, j]`, then top-c over the document axis per slot. **Fully gradient-free.** The
  entropy-regularised variant is a Sinkhorn/Dykstra solve — also gradient-free.
- **Maps to our code.** `cartridges/am/ranking.py` — new `SLOT_SELECTION=expert_choice` plus
  `AM_CAPACITY_FACTOR`. Needs one structural change: `rank_am_slots` is currently called **per document**
  (`continual.py` loop), so the 16 documents' `tf` vectors must be materialised before assignment. Two
  honest options — (i) a *two-pass* run (pass 1 records all 16 `tf` matrices, pass 2 assigns and writes),
  or (ii) an online greedy with per-slot capacity counters, which is the sequential/causal analogue and
  keeps the single-pass structure.
- **Prediction about OUR signal.** With `n=16` documents, `e=511` writable slots, `c=1`: each slot takes
  its single best document ⇒ **writes/slot = 1.00** (from 9.83), union → ~511 slots/layer (from 54.4),
  survival → **~100%** (from 4.7%), Jaccard → **0.0** (from 0.713). Union MT routing mass rises toward
  ~0.70 but **per-document** captured mass collapses. Predicted **MT ≥ 2.45, most likely 2.52–2.60**, i.e.
  at or worse than the k=16 endpoint 2.552, because this is arithmetically the ORACLE-WRITE-512 regime
  (full support, mutual annihilation) reached by a nicer route. Selectivity **unchanged, 1.04–1.06**.
  QA likely *improves* (writes spread thinner over high-Fisher slots), consistent with the fisher
  selector's QA 1.8641 — the axis already 0.573 ahead.
- **Why it might NOT transfer.** Expert-choice's entire selling point is load balance, and load balance
  is exactly what DIAG-PERDOC measured to be *harmful* here (+1.101 for the uncontested write). MoE
  experts are *parameters that get trained after routing*; our slots are *values written once and never
  trained*, so a balanced assignment cannot be repaired downstream.
- **Verdict: candidate — as a falsification test only.** It is the cleanest single-variable probe of
  DIAG-PERDOC. If MT ≤ 2.219 it refutes DIAG-PERDOC and reopens the whole allocation family; I predict
  it will not.

### L03-3: Token-choice top-k, capacity factor, load-balancing loss — *Switch Transformers* (arXiv 2101.03961, JMLR 2022, https://arxiv.org/abs/2101.03961)
- **Claim / mechanism.** Top-1 token-choice routing; `expert_capacity = (tokens_per_batch/num_experts) ×
  capacity_factor`; overflow tokens are **dropped** (passed through the residual); balance is enforced by
  an auxiliary loss `α · N · Σ_i f_i · P_i` with `α = 10⁻²`. Establishes that k=1 suffices, contra the
  prior belief that k>1 was needed for non-trivial router gradients.
- **Is it query-dependent?** No. The router reads the token being routed. Applied to our setting this is
  *precisely* the incumbent: `tf` top-32 per document **is** token-choice top-k with `k=32` and no capacity
  constraint. Switch adds only (i) a capacity cap and (ii) an auxiliary balance loss.
- **Gradient-free?** The capacity cap is; the auxiliary loss is not (and it needs an optimiser step,
  which would break `gradient_steps = 0`).
- **Maps to our code.** The capacity cap is a 5-line addition to `ranking.py` (`AM_SLOT_CAPACITY`:
  a running per-slot write counter in `continual.py`, slots at capacity get the −1.0 sentinel already used
  by `constrained_mass`). The auxiliary loss has **no legal home** in this project.
- **Prediction about OUR signal.** Capacity cap `c` per slot over 16 documents: at `c=4`, writes/slot
  9.83 → ~4, union ~130 slots/layer, per-document captured MT mass falls to roughly the level
  `constrained_mass` q=0.5 reached (**m ≈ 0.13–0.17**) because later documents are pushed off their
  best slots ⇒ predicted **MT 2.33–2.37** by the law. Selectivity **1.04–1.06**.
- **Why it might NOT transfer.** The capacity factor exists to avoid *dropping* tokens under a hardware
  constraint; we have no hardware constraint, and nothing is dropped. Importing it imports the cost with
  none of the benefit.
- **Verdict: unusable as a fix** — the auxiliary-loss half needs gradients, and the capacity half is a
  strictly-worse special case of the expert-choice experiment (L03-2), which achieves the same spread
  optimally instead of greedily. Listed because the brief asked for capacity factors and load balancing,
  and because it identifies the incumbent as textbook token-choice.

### L03-4: Balanced assignment as a combinatorial solve — *BASE Layers* (arXiv 2103.16716, ICML 2021, https://arxiv.org/abs/2103.16716)
- **Claim / mechanism.** Formulates routing as a **linear assignment problem**: `maximize Σ_t h_t·w_{a_t}`
  subject to `∀e: Σ_t 1[a_t = e] = T/E`. Solved with **Bertsekas's auction algorithm** (chosen over
  Hungarian for GPU efficiency). No auxiliary loss, no capacity factor, no dropped tokens — balance is a
  *hard constraint solved exactly*. At test time it degrades gracefully to greedy top-1.
- **Is it query-dependent?** No — same write-side assignment as L03-2. But BASE answers a different
  question that matters to us: **"if you want a balanced assignment, is greedy or optimal the right
  solver?"** Answer: optimal, and it is cheap.
- **Gradient-free?** **Yes for the assignment.** The auction algorithm is a combinatorial solve over an
  affinity matrix we already have; `gradient_steps` stays 0. Only the affinity *scores* are learned in
  the paper, and we substitute `tf`.
- **Maps to our code.** `ranking.py` — `SLOT_SELECTION=base_assign`, using `scipy.optimize.linear_sum_assignment`
  on the (16 docs × 511 slots) `tf` matrix per layer, replicated 32× to give each document 32 slots
  (a transportation problem: supply 32/doc, capacity 1/slot). Same two-pass requirement as L03-2.
- **Prediction about OUR signal.** This is L03-2 with an optimal rather than greedy solver, so:
  writes/slot **1.00**, union 512 slots/layer, and a *higher* total assigned affinity than expert-choice
  greedy — I'd predict the optimal assignment recovers **5–15% more** summed `tf` than greedy top-c.
  MT: **2.42–2.55**, i.e. better than greedy expert-choice by ≤0.05 but still well short. Selectivity
  **1.04–1.06**.
- **Why it might NOT transfer.** Same as L03-2: it optimises balance, and balance was measured to hurt.
  Also BASE's own train/test mismatch (balanced during training, greedy at test) has no analogue here —
  our write *is* the test.
- **Verdict: candidate only as the solver for L03-2.** If the expert-choice experiment is run at all, run
  it with this solver; do not run it as a separate arm.

### L03-5: Routing with no learned router at all — *Hash Layers For Large Sparse Models* (arXiv 2106.04426, NeurIPS 2021, https://arxiv.org/abs/2106.04426)
- **Claim / mechanism.** Route each token by a **fixed hash of its token id**, `h_t = FFN_{hash(x_t)}(h̄_t)`.
  No router parameters, no balancing loss. Beats Switch by **0.4–0.6 perplexity** at 751M–1.28B, and the
  gap *widens* with more experts. Their diagnosis of why learned routing is hard is directly relevant to
  us: "during training membership for each expert is changing while it is trying to learn the mapping for
  those members." Notably, **clustered** hashing (k-means on token embeddings) was *worse* than random
  (23.90 vs 23.16 ppl), and **dispersed** hashing (deliberately separating similar tokens) restored it.
- **Is it query-dependent?** **No — it is not even input-dependent beyond the token id.** By the brief's
  own rule this is bounded by the β result.
- **Gradient-free?** Completely. This is the existence proof for "a useful router need not be learned."
- **Maps to our code.** `ranking.py::SLOT_SELECTION=hash` — assign document `d`'s 32 slots by
  `hash(doc_id, layer, i) mod 511`, ignoring `tf` entirely. ~15 lines, no new state.
- **Prediction about OUR signal.** MT routing mass **≈ 32/511 = 0.063** at best (random slots carry
  average mass), i.e. **4.4× below** the incumbent's 0.2799 ⇒ by the law, MT ≈ 2.0613 − 0.1574·ln(0.063)
  = **2.497**, +0.23 worse. Selectivity **1.00 ± 0.01** (random support has no task preference).
  Survival ~100%, writes/slot ~1.0 — so it is *also* a cheap probe of the capacity question, at 1/50th
  the implementation cost of L03-2.
- **Why it might NOT transfer.** Hash layers work because the *experts are then trained* on whatever they
  receive. Our slots receive a one-shot closed-form value and are never trained. Random routing with no
  downstream training is just noise. And the paper's own "dispersed beats clustered" finding argues
  *against* our whole `mass_x_redundancy` line of reasoning, which is interesting but not actionable.
- **Verdict: unusable as a fix** (query-independent, and predicted −0.23). **Usable as a control:**
  it is the correct null for "does the selection matter at all", which this project has never run —
  every arm so far has been a *variant* of mass ranking.

### L03-6: Slots compete for inputs (the softmax-axis reversal, distilled) — *Object-Centric Learning with Slot Attention* (arXiv 2006.15055, NeurIPS 2020, https://arxiv.org/abs/2006.15055)
- **Claim / mechanism.** `attn_{i,j} = softmax_j( k(inputs) q(slots)ᵀ / √D )` — **the softmax is over the
  slots axis**, so slots *compete* for each input element; then `W_{i,j} = attn_{i,j} / Σ_l attn_{l,j}`
  (a weighted mean over inputs) and `slots = GRU(slots, W ᵀ v(inputs))`, iterated T=3 times. The paper is
  explicit that the softmax axis is what "forces competition: slots must specialize to explain distinct
  input regions, **preventing redundancy**". Fixed K slots, permutation-symmetric, random init.
- **Is it query-dependent?** No (write-side). But this is the *cleanest statement in the literature* of
  the thing our brief calls "slots choose documents vs documents choose slots", and unlike expert-choice
  it gives a **soft, iterative** version with an explicit anti-redundancy argument.
- **Gradient-free?** The module is normally trained, but **the iteration itself is a fixed-point algorithm**:
  given the frozen projections (which we have — the cartridge keys are `q(slots)`, the document keys are
  `k(inputs)`), 3 iterations of softmax-over-slots + normalised mean is pure linear algebra. **Runnable
  with `gradient_steps = 0`.** The GRU is the only learned part and can be replaced by a convex update
  `slots ← (1−η)·slots + η·updates`.
- **Maps to our code.** `ranking.py` — `SLOT_SELECTION=slot_competition`; the competition weights replace
  the top-32 mask, and `finetune.py`'s per-document write becomes a weighted write. Shares the two-pass
  requirement with L03-2 (all 16 documents must be present to compete).
- **Prediction about OUR signal.** The competition softmax redistributes each document away from slots
  already claimed. Predicted union 54.4 → **150–300 slots/layer**, writes/slot 9.83 → **1.7–3.4**,
  Spearman between documents' slot scores **0.958 → < 0.5** (this is the one arm I expect to actually move
  that number). MT routing mass falls to **0.15–0.22** ⇒ MT **2.30–2.37**. Selectivity **1.04–1.06**.
- **Why it might NOT transfer.** Slot Attention's competition is valuable because the slots are then
  *decoded* and supervised; the anti-redundancy it buys is a representational property, and DIAG-PERDOC
  says representational separation is not what we lack. Also our documents are 16, not thousands — the
  competition is barely constrained.
- **Verdict: candidate, medium priority.** It is the *only* mechanism here that I expect to break the
  0.958 document-blindness, and that number is the project's own stated symptom. But breaking it is
  predicted to *cost* MT, which would be a clean and informative negative.

### L03-7: Query-dependent routing by online clustering — *Efficient Content-Based Sparse Attention with Routing Transformers* (arXiv 2003.05997, TACL 2021, https://arxiv.org/abs/2003.05997)
- **Claim / mechanism.** Project queries and keys with a **shared** matrix `R`, ℓ2-normalise, and assign
  both to the nearest of `k` centroids; a query attends only to keys in **its own cluster**. Centroids are
  updated by EMA, `μ ← λμ + ((1−λ)/2)Σ_{i: μ(Q_i)=μ} Q_i + ((1−λ)/2)Σ_{j: μ(K_j)=μ} K_j`, λ=0.999 —
  **online spherical k-means, not gradient descent**. Balanced clusters by taking the top n/k tokens per
  centroid. `k = √n` gives O(n^1.5 d).
- **Is it query-dependent?** **Yes, decisively.** Which keys a query sees is a function of *that query's*
  cluster assignment. This is a genuine counter-example to β and it costs no gradients.
- **Gradient-free?** **Yes** for the centroids (EMA). Only `R` is learned in the paper; we can set `R = I`
  (the head's own 128-dim space) and cluster the raw queries/keys, which is exactly the space DIAG-KEYSPACE
  already measured in.
- **Maps to our code.** Two places. (i) **Write side**: `key_select.py` — cluster the 511 cartridge keys
  and the document's reference queries jointly, and write each document into the slots of the clusters its
  queries fall in (`AM_ROUTE_MODE=cluster`). (ii) **Eval side**: `cartridges/models/attention.py:100–110`
  — a `score_mod` that adds `−∞` (or a bounded −B) to slots outside the query's cluster.
- **Prediction about OUR signal — and this is the entry where our own diagnostics bite hardest.**
  DIAG-KEYSPACE measured QA/MT mean-routing cosine **0.99899**, top-32 overlap **0.914**, histogram
  intersection **0.963**, mean task separation **2.5× smaller than within-QA sampling noise**, and
  **0/288 heads** running the other way. **In that geometry, k-means cannot produce clusters that separate
  MT from QA queries** — there is nothing to cluster on. Predicted selectivity **1.03–1.09**, i.e. the
  same band, and MT within ±0.049 of 2.2681 at best. The one number I would expect to move: per-query
  routing entropy, which nobody has recorded.
- **Why it might NOT transfer.** Routing Transformer's clusters separate *content* in a 4k-token
  sequence with genuine topical structure. Our 512 slots are a distilled cartridge whose routing profile
  is measured to be nearly the same for both tasks. The mechanism is sound; the substrate has no signal.
- **Verdict: unusable *here*, for a measured reason** — ρ_key at its held-out control floor and 0.99899
  routing cosine mean the clustering has nothing to separate. Marked unusable rather than listed hopefully,
  per the brief's rule. Worth re-reading if the query distribution is ever changed (the other open escape).

### L03-8: Query-aware, training-free criticality over a KV cache — *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference* (arXiv 2406.10774, ICML 2024, https://arxiv.org/abs/2406.10774)
- **Claim / mechanism.** Partition the KV cache into pages (16 KV pairs). For each page store element-wise
  **min `m_i`** and **max `M_i`** of the keys. At decode, score each page for the *current* query `Q`:
  `Score = Σ_i max(Q_i·m_i, Q_i·M_i)`, an **upper bound on the maximum attention logit in that page**
  (since `m_i ≤ k_i ≤ M_i`, the per-channel max brackets every possible product). Take top-K pages.
  **No training whatsoever.** 7.03× self-attention speedup at 32K/2048-budget, near-full-attention recall.
  The paper's central argument is exactly our problem statement: query-**agnostic** eviction (H2O,
  StreamingLLM, TOVA) "permanently discard tokens based on historical data or recency" and therefore
  "parts of KV caches that are important for distant future tokens may be discarded"; the same token gets
  a different importance depending on which query attends to it.
- **Is it query-dependent?** **Yes — maximally so.** The score is a function of `Q`. There is no `Q`-free
  reduction of it. **This is the formulation that is not bounded by our β result**, and it is the one to
  build on.
- **Gradient-free?** **Completely.** Two reductions (min, max) over the cartridge keys, computed once; a
  dot-product per query at eval. `gradient_steps` stays 0 and there is not even a diagnostic backward.
- **Maps to our code.** Purely eval-side, **the write path is untouched**:
  `cartridges/cache.py:203 get_cartridge_beta` currently returns a static `(1, n_kv_heads, T_c)` bias, and
  `cartridges/models/attention.py:106` closes over it in `score_mod(score, b, h, q_idx, kv_idx)` — which
  **already receives `b` and `q_idx`**. A query-dependent bias is a batch/query-indexed lookup in that
  closure. Flag: `AM_QUERY_GATE=quest`, `AM_QUERY_GATE_TOPK`, `AM_QUERY_GATE_BOX` (reuse `AM_BETA_BOX=3.0`
  so the operator is bounded exactly like β was — that makes the β comparison *paired*).
- **Prediction about OUR signal.** With page size 1 (our slots are already 1 token each, so Quest degenerates
  to exact per-query top-K, which is *free* at our scale — 511 slots, not 32K), a top-K gate that promotes
  the written slots for MT queries and demotes them for QA queries can, within a ±3.0 log-box, multiply the
  MT/QA mass ratio by up to `e^{6}` in principle. Realistically: **selectivity 1.05 → 1.4–2.5** (this is the
  number to watch; anything < 1.10 kills the family), MT routing mass **0.2799 → 0.45–0.60** ⇒ by the law
  **MT 2.196–2.150**, a gain of **−0.07 to −0.12**, i.e. **1.4–2.4× the ±0.049 resolution**. QA should be
  roughly neutral or better (mass is *removed* from written slots for QA queries; QA is 0.573 ahead so it
  can afford it). **This does not reach 2.02** — see bullet 9 of the summary.
- **Why it might NOT transfer.** Quest selects *which of the real, uncompressed KV to load* — it never
  changes the attention distribution's *content*, only skips provably-low-logit pages. Our gate would be
  *changing* the distribution, i.e. using the same arithmetic for a different purpose, and the frozen model
  never saw a biased cartridge. Second: DIAG-KEYSPACE's 0.99899 routing cosine says the *incumbent* keys
  give MT and QA nearly identical logits, so a Quest score computed on the current keys will rank slots
  nearly identically for both — **the gate would need the document keys installed** (`KEY_MODE=highest_attention
  AM_KEY_REPOSITION=1`, which is already our best point and does move `mass_on_S` 0.0711 → 0.2430). Run it
  on top of the key-installed cartridge, not the frozen-key one.
- **Verdict: CANDIDATE — the strongest in this file.** Only query-dependent, fully gradient-free,
  eval-time-only, ~30 lines, and it is a *paired* test against β on the same operator budget.

### L03-9: Learned key-side selection, trained with the sparsity in the loop — *Native Sparse Attention* (arXiv 2502.11089, ACL 2025, https://arxiv.org/abs/2502.11089)
- **Claim / mechanism.** Three branches — compression (block MLP `φ`), **selection** (blockwise top-n), and
  sliding window — combined by a learned per-branch gate `o*_t = Σ_c g_t^c · Attn(q_t, K̃_t^c, Ṽ_t^c)`,
  `g_t^c = σ(MLP(x_t))`. The selection scores are **reused from the compression branch's own attention**:
  `p_t^cmp = softmax(q_tᵀ K̃_t^cmp)`, spatially re-aggregated to selection blocks, and summed over query
  heads within a GQA group (`p_t^slc' = Σ_h p_t^slc,(h)`) so the whole group loads the same blocks.
  The headline methodological claim: **post-hoc sparsity fails** — "applying sparsity post-hoc forces
  models to deviate from their pretrained optimization trajectory" — so NSA is trained from pretraining.
- **Is it query-dependent?** **Yes for the selection** (`p_t^cmp` contains `q_t`). The *gate* `g_t^c` is a
  function of the hidden state `x_t`, so it is also query-dependent in the token sense — but it gates
  *branches*, not slots, so as an operator on our fixed slot set it collapses to a per-layer scalar, which
  is β-shaped and bounded.
- **Gradient-free?** **No, and the paper's central claim is that this matters.** The compression MLP `φ`
  and the branch gates are learned jointly with the model. There is no closed-form variant; the closest
  gradient-free relative is exactly L03-8 (Quest), which NSA cites the family of.
- **Maps to our code.** The GQA-consistent aggregation is directly relevant to a known suspect:
  `cartridges/am/query_accum.py:87-89` builds selection scores from a **GQA-group mean query taken before
  the softmax**. NSA aggregates **after** the softmax (`Σ_h p_t^slc,(h)` are attention probabilities, not
  logits). That is a real, cheap, single-variable fix: `AM_GQA_AGG=post_softmax` in `query_accum.py`.
- **Prediction about OUR signal.** The GQA aggregation fix alone: MT routing mass **0.2799 → 0.29–0.34**
  (the board measured a per-head selection carries 0.2598 vs per-layer 0.1954 on the writable set, so the
  head-resolved signal is ~1.3× richer) ⇒ MT **2.24–2.27**, i.e. **−0.00 to −0.03: inside the resolution**.
  Selectivity unchanged 1.04–1.06. Prediction: a real but sub-resolution improvement.
- **Why it might NOT transfer.** NSA's entire result rests on training with the sparse pattern; we cannot.
  Its own argument implies that bolting NSA's selection onto a frozen Qwen3-4B is the failure mode it was
  designed to avoid.
- **Verdict: unusable as a mechanism** (needs gradients, no closed form). **Usable as one concrete
  micro-fix** — the post-softmax GQA aggregation — which is worth doing but is predicted sub-resolution.

### L03-10: Learned sparse retrieval / key-side expansion — *SPLADE v2* (arXiv 2109.10086; SPLADE arXiv 2107.05720, SIGIR 2021, https://arxiv.org/abs/2107.05720)
- **Claim / mechanism.** Predict a sparse vocabulary-space representation for documents and queries via the
  MLM head: `w_j = Σ_i log(1 + ReLU(w_ij))`. The log saturation prevents any single term dominating and
  induces sparsity; a **FLOPS regulariser** balances the posting-list distribution (outperforming ℓ1,
  which produces skewed term frequencies). **Document expansion**: the key side learns to light up
  vocabulary entries the document does not literally contain, so future queries route to it.
- **Is it query-dependent?** **No, and this is the instructive part.** The *document* representation is
  computed once, independently of any query — it is precisely a **query-independent key-side operator**,
  the same class as β. Its selectivity comes entirely from the *match* being an inner product, i.e. from
  the query side, which SPLADE does not modify at index time. So SPLADE is the retrieval literature's
  version of our β, and it works there only because the expansion is **learned against the query
  distribution** with gradients and a ranking loss.
- **Gradient-free?** No. There is no closed-form SPLADE. The nearest gradient-free relative is BM25/TF-IDF
  — which is, exactly, our incumbent `tfidf` selector.
- **Maps to our code.** `cartridges/am/key_select.py::rewrite_keys_on_support` — a "key expansion" analogue
  would move installed document keys toward a direction that raises MT logits and lowers QA logits. Flag:
  `AM_KEY_EXPAND`. This is the mechanism DIAG-KEYSPACE already bounded.
- **Prediction about OUR signal.** DIAG-KEYSPACE measured that an **optimally placed key buys selectivity
  1.22** against an incumbent 1.083. SPLADE-style expansion is a *constrained* version of optimal placement
  (it must remain a plausible key), so predicted selectivity **≤ 1.22**, realistically **1.08–1.15**;
  by the mass↔loss law a selectivity gain of 1.15/1.05 = 1.10× in mass buys ΔMT = −0.1574·ln(1.10) =
  **−0.015**, which is **3× below the ±0.049 resolution**. Unmeasurable.
- **Why it might NOT transfer.** Beyond the above: SPLADE's FLOPS regulariser exists to balance an inverted
  index across millions of documents; our 511 slots and 16 documents have no index-skew problem that a
  regulariser fixes — our skew (87.4% document-blind overlap) is a *signal* problem, not a *balance* problem.
- **Verdict: unusable.** Query-independent by construction, needs gradients, and its ceiling is already
  measured at 1.22 selectivity ⇒ ≤ −0.015 MT.

### L03-11: Per-task hard gates — *HAT: Overcoming Catastrophic Forgetting with Hard Attention to the Task* (arXiv 1801.01423, ICML 2018, https://arxiv.org/abs/1801.01423)
- **Claim / mechanism.** A per-layer, almost-binary gate `a_t^l = σ(s · e_t^l)` where `e_t^l` is a **learned
  task embedding**; `s` is annealed from `1/s_max` to `s_max` so the gate hardens; gradients for previous
  tasks are masked by the cumulative max `a_{≤t}^l = max(a_t^l, a_{≤t−1}^l)` via
  `g'_{l,ij} = [1 − min(a_{≤t}^l,i, a_{≤t−1}^l,j)] · g_{l,ij}`; a compressibility coefficient `c` trades
  capacity against sparsity.
- **Is it query-dependent?** **No — it is conditioned on the task identifier alone.** The paper is explicit:
  "the task definition or, more pragmatically, its identifier, is crucial", and **task identity is required
  at inference**. Under the brief's rule this is bounded by the β result: a task-conditional gate is a
  *constant per task*, which is exactly what β is per document.
- **Gradient-free?** No — `e_t^l` is learned by backprop concurrently with the task, and the annealing
  schedule is a training-time device with no inference-time analogue.
- **Maps to our code.** The *masking* half already exists here in a stronger form: `SLOT_SELECTION=fisher`
  is HAT's protection idea computed exactly rather than learned, and it produced the best retention this
  project has ever measured (**QA 1.8641** at 0.04% QA-Fisher exposure).
- **Prediction about OUR signal.** HAT's protection axis is measured: `fisher` gives QA **1.8641**
  (−0.083 vs baseline 1.9468, **inside** the ±0.054 QA resolution) and **MT +0.397** (8.1× the MT
  resolution). A HAT-style learned mask would sit on the same anti-aligned frontier: predicted
  **ΔQA ≈ −0.05 to −0.10, ΔMT ≈ +0.15 to +0.40**. It buys the axis that is already **0.573 ahead** and
  pays the axis that is **0.248 short**. Selectivity **unchanged**.
- **Why it might NOT transfer.** Two hard blockers beyond query-independence: (i) our eval provides **no
  task label** — QA and MT are scored from the same cartridge with no oracle, so a task-conditional gate
  is not even *evaluable* here without a task classifier (see L03-12); (ii) it requires gradients.
- **Verdict: unusable.** Query-independent, gradient-requiring, needs a task oracle we do not have, and its
  measured proxy (`fisher`) already spends the wrong axis. Marked unusable rather than listed hopefully.

### L03-12: Input-conditioned gates + a learned task classifier — *Conditional Channel Gated Networks for Task-Aware Continual Learning* (arXiv 2004.00070, CVPR 2020, https://arxiv.org/abs/2004.00070)
- **Claim / mechanism.** The fix for HAT's task oracle. Gates are produced by a small MLP (16 hidden units,
  BN, ReLU) **from the input feature map**: `h^{l+1} = G_t^l(h^l) ⊙ h^{l+1}`, relaxed with Gumbel-softmax +
  straight-through. An ℓ1 sparsity term `L_sparse = (λ_s/L) Σ ‖G_t^l(h^l)‖₁ / c_out^l` preserves capacity for
  future tasks. In class-incremental settings a shallow MLP **task classifier** over the concatenated
  per-task feature streams infers the task at test time. Kernels relevant to past tasks are **frozen**;
  irrelevant ones are re-initialised.
- **Is it query-dependent?** **Yes — the gate reads the input, not the task id.** This is the CL literature's
  proof that the query-dependence we need is achievable within a CL gating framework, and that the task
  oracle is removable.
- **Gradient-free?** **No.** The gating MLPs, the Gumbel relaxation and the task classifier are all trained.
  There is no closed form. The *amortised* substitute that survives our constraints: replace the learned
  classifier with a **frozen-model retrieval key** — exactly InduceKV's `r(x)` (L03-16) — which needs no
  training at all. That substitution is the bridge between this paper and something we can actually run.
- **Maps to our code.** No write-path change; this is an eval-time gate at
  `cartridges/models/attention.py:100–110` with a per-example bias, plus a per-slot "owning document" map
  (already recoverable from the saved `am_doc_*.pt` selections). Flag: `AM_QUERY_GATE=doc_router`.
- **Prediction about OUR signal.** A perfect task classifier (QA vs MT) applied as a ±3.0 log-box gate over
  each document's written slots would give selectivity bounded by how *separable* the slot sets are —
  and our own measurement is that they are not: QA/MT top-32 overlap **0.914**, histogram intersection
  **0.963**. So even an **oracle** task label can only re-weight the 8.6% of slots that differ ⇒ predicted
  selectivity **1.05 → 1.12–1.20**, mass 0.2799 → 0.31–0.34 ⇒ **ΔMT −0.016 to −0.030: inside ±0.049**.
  This is the single most important negative number in this file, and it is worth stating as a theorem-shaped
  claim: **a gate that is only task-conditional cannot beat the 0.914 overlap, whatever it is conditioned on.**
  Only a gate conditioned on the *individual query* (L03-8, L03-16) escapes that bound.
- **Why it might NOT transfer.** Channel gating gates *units of a trainable network*; we gate *values in a
  frozen cache*. Their freezing step has no analogue (our slots are overwritten, not frozen), and their
  re-initialisation step is what our overwrite already does destructively.
- **Verdict: candidate for its *architecture* (input-conditioned gate, no task oracle), unusable as
  implemented** (gradients). Its main contribution here is the 0.914-overlap bound above.

### L03-13: Query-dependent *writes* — the delta rule — *Linear Transformers Are Secretly Fast Weight Programmers* (arXiv 2102.11174, ICML 2021) + *Parallelizing Linear Transformers with the Delta Rule* (arXiv 2406.06484, NeurIPS 2024) (https://arxiv.org/abs/2102.11174, https://arxiv.org/abs/2406.06484)
- **Claim / mechanism.** Replace the additive outer-product write `W ← W + v kᵀ` with the **delta rule**
  `W_i = W_{i−1} + β_i (v_i − W_{i−1} φ(k_i)) φ(k_i)ᵀ`, equivalently
  `S_t = S_{t−1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ` — a **generalised Householder**: *retrieve what is
  currently stored at this key, subtract it, then write*. `β_t ∈ [0,1]` is a network-generated **write
  strength** (β=1 fully replaces, β=0 leaves memory untouched). The stated motivation is exactly our
  B-OVERWRITE: "a purely additive update rule makes it difficult to deallocate past key–value associations,
  eventually leading to key collisions when L > d". DeltaNet gives a chunkwise-parallel WY form and shows
  large associative-recall gains over additive linear attention.
- **Is it query-dependent?** **Yes — this is the sharpest contrast with β in the whole file.** Our β is a
  per-key scalar log-bias with no `q`; the delta rule's write is *defined by* the key it is retrievable at,
  and the correction term `W_{i−1}φ(k_i)` is a **read performed with the write's own key**. The write is
  chosen so the read comes back right.
- **Gradient-free?** **Yes, as an update rule** — it is closed-form given `(k, v, β)`. Only `β_t`'s generator
  is learned in the papers, and we can set `β_t = 1` (full replacement) or grid-search a scalar.
- **Maps to our code.** `cartridges/am/value_solve.py` + `finetune.py::apply_document_am_write_to_cache`.
  Today the write installs a solved value into selected slots. The delta form would instead write
  `v_new = v_solved − (current cartridge's own read at this key)`, i.e. a **residual** write.
  Flag: `AM_WRITE_RULE=delta`, `AM_DELTA_BETA`. Note `beta_target='residual'` (`finetune.py:90`) is a
  *related but different* thing — it subtracts the non-selected slots' mass from the fit target; the delta
  rule subtracts what the **selected** slots already return.
- **Prediction about OUR signal.** I predict this **loses**, and I will say so numerically. Two of our
  measurements point the same way: (i) ORACLE-WRITE found "better fitting is anti-correlated with CE" —
  18× lower MSE → **+13 loss** — and the delta rule's entire purpose is a more exact fit;
  (ii) DIAG-PERDOC found a solo, non-interfering write is **+1.101 worse**, and the delta rule's purpose is
  to remove interference. Predicted: solve MSE **falls 2–10×**, survival rises **4.7% → 20–60%**, MT routing
  mass roughly unchanged (**0.26–0.30**), and **MT +0.05 to +0.25** (worse). Selectivity **1.04–1.06**.
  **Falsifier:** MT ≤ 2.219 would refute both ORACLE-WRITE's anti-correlation and DIAG-PERDOC in one shot,
  and would be the most important result the project could get.
- **Why it might NOT transfer.** DeltaNet's state is a `d×d` matrix updated over a *sequence* with the model
  trained around the rule; ours is 512 discrete slots written 16 times by a frozen model. Their capacity
  argument (`L > d_key` ⇒ collisions) does not bind us: 16 documents × 32 slots into 511 slots of 128 dims
  is not an overcapacity regime in their sense — our collisions are caused by the ranker choosing the same
  slots (Spearman 0.958), not by dimensional saturation.
- **Verdict: candidate with a negative prediction.** Cheap (a residual subtraction in an existing solve),
  single-variable, and its outcome is decision-relevant either way.

### L03-14: Explicit learned addressing over a fixed slot set — *Large Memory Layers with Product Keys* (arXiv 1907.05242, NeurIPS 2019, https://arxiv.org/abs/1907.05242) [+ *Mixture of A Million Experts / PEER*, arXiv 2407.04153]
- **Claim / mechanism.** Keys are a Cartesian product `K = {(c, c′) : c ∈ C, c′ ∈ C′}`, so exact top-k over
  `|C|²` keys costs `O(√|K|)`: split the query, take top-k in each half-codebook, rank the `k²` combinations.
  Selected slots are combined by a softmax-weighted sum of their values. A **query network** with
  **BatchNorm** is essential: without it, memory usage collapses to **25.8%** of a 1M-slot memory (with it,
  80.3%), tracked by the KL divergence of the access distribution from uniform.
- **Is it query-dependent?** **Yes** — the address is `top-k(q · K)`. This is textbook query-dependent
  addressing over a *fixed* slot set, which is structurally the closest thing in the literature to what our
  cartridge is.
- **Gradient-free?** **The addressing is** (it is a top-k over inner products, exactly what our frozen
  attention already computes). **The query network is not**, and the BatchNorm fix is a training-time
  device with no closed form. PEER extends this to expert routing at a million experts, same dependency.
- **Maps to our code.** Structurally, our attention over 511 slots **already is** PKM addressing with the
  query network fixed to the frozen model's `q_proj`. So the PKM lesson is not a new mechanism but a
  **diagnosis**: their usage-collapse number (25.8% of slots ever used) is our **87.4% document-blind
  top-32 overlap** and **54.4/511 = 10.6% union** under a different name — and their fix requires training
  the query network, which is frozen for us.
- **Prediction about OUR signal.** If we imported their *diagnostic* rather than their fix: the access-KL
  from uniform over our 511 slots is the natural summary statistic for B-OVERWRITE and has never been
  recorded. Predicted slot-usage (fraction of writable slots ever written across 16 docs) **10.6%** —
  measured — vs PKM's pre-fix 25.8%: **our addressing is 2.4× more collapsed than the pathology PKM
  considered severe enough to require an architectural fix.** No MT prediction: PKM offers no gradient-free
  intervention, so per the brief's rule it does not qualify as a mechanism.
- **Why it might NOT transfer.** PKM adds memory *capacity* to a trainable network; we have a fixed cache
  in a frozen network. Their slot values are learned by gradient descent over the whole corpus.
- **Verdict: unusable as a mechanism** (its only lever is training the query network, which is frozen).
  **Valuable as the naming of the pathology** and for the access-KL statistic.

### L03-15: Content-addressed write with an explicit erase vector — *Neural Turing Machines* (arXiv 1410.5401, 2014) [+ DNC, Nature 538:471, 2016] (https://arxiv.org/abs/1410.5401)
- **Claim / mechanism.** Write addressing is content-based: `w_t^c(i) = softmax_i(β_t K[k_t, M_t(i)])` with
  cosine similarity and a sharpening `β_t`, optionally interpolated with the previous weighting (gate `g`),
  shifted, and sharpened by `γ_t`. The write is **erase-then-add**:
  `M̃_t(i) = M_{t−1}(i)[1 − w_t(i) e_t]`, then `M_t(i) = M̃_t(i) + w_t(i) a_t`, with `e_t ∈ (0,1)^d`.
  DNC adds a **usage-based allocation** weighting (write to the least-used location) alongside content
  addressing.
- **Is it query-dependent?** The *write* address is content-dependent on the write key — same class as the
  delta rule (L03-13). The *read* is query-dependent. But the erase vector `e_t` is emitted by the
  controller and is **not** a function of the future read query, so as a gate over already-written slots it
  is query-independent.
- **Gradient-free?** The addressing and erase/add arithmetic are closed-form; only the controller that emits
  `(k_t, β_t, g, γ_t, e_t)` is learned. DNC's usage-based allocation is a **closed-form, gradient-free
  heuristic** (retention/usage vectors updated by a recurrence) — and it is the exact literature ancestor of
  our `redundancy` selector.
- **Maps to our code.** Two things. (i) The **element-wise erase vector** `e_t` is a finer-grained overwrite
  than anything we have: currently a slot is fully replaced, whereas `e_t ∈ (0,1)^{128}` would let a document
  overwrite only *some channels* of a slot. `finetune.py::apply_document_am_write_to_cache` +
  `AM_ERASE_MODE=partial`. (ii) DNC's usage allocation ≙ `SLOT_SELECTION=redundancy`, already built (MECH-008)
  and already measured to lose **+0.167 MT**.
- **Prediction about OUR signal.** The usage-allocation half is **already measured**: `redundancy` = +0.167 MT
  (3.4× the resolution), reproduced across three seed offsets (+0.167 / +0.163 / +0.200). The partial-erase
  half is new: a channel-wise soft overwrite `v ← (1−w⊙e)v_old + w a` with `e` set from per-channel value
  magnitude would raise survival (4.7% → 30–70%) while leaving MT routing mass unchanged (**0.27–0.29**) ⇒
  predicted **MT within ±0.05 of 2.2681, sign more likely positive (worse)**, on the same DIAG-PERDOC logic
  as L03-13.
- **Why it might NOT transfer.** NTM/DNC controllers are trained end-to-end for tens of thousands of steps to
  *learn* to use these addressing modes; the modes have no standalone value. And DNC's usage allocation has
  already been imported and refuted here under the name `redundancy`.
- **Verdict: unusable as a whole** (controller must be learned; its gradient-free half is already measured to
  lose). **The partial-erase idea is a candidate**, ranked below L03-13 because it shares the same predicted
  failure mode with less mechanistic motivation.

### L03-16: Query-dependent retrieval weighting over a fixed-budget KV memory, frozen backbone — *InduceKV: Fixed-Footprint Continual Adaptation of Multimodal LLMs via Inducing KV Memories* (arXiv 2607.02010, 2026-07-02, https://arxiv.org/abs/2607.02010)
- **Claim / mechanism.** Our setting, almost exactly, from three weeks ago. Freeze the backbone `θ`
  (**no gradient updates to `θ` at any time**); externalise each task increment as memory entries
  `e(x) = (r(x), {K̃^ℓ(x), Ṽ^ℓ(x)}_{ℓ=1..L})` where `r(x) = h̄(x)/‖h̄(x)‖` is a unit-norm pooled prefix
  representation and `K̃, Ṽ` are pooled layerwise KV payloads of fixed length `m`. At inference, compute
  the **query-dependent** retrieval weight
  `α_i(x; φ) = exp(⟨r(x), r_i⟩/τ) / Σ_j exp(⟨r(x), r_j⟩/τ)`, `τ = softplus(φ_τ)`,
  assemble `K_mem^ℓ(x) = Concat(√α_i · K̃_i^ℓ)`, `V_mem^ℓ(x) = Concat(√α_i · Ṽ_i^ℓ)`, and inject them into
  masked self-attention with a per-layer **value-side gate** `λ_ℓ = σ(φ_ℓ)`:
  `Attn^ℓ = softmax(Q[K_self; K_mem]ᵀ/√d_h + M)[V_self; λ_ℓ V_mem]`.
  Memory selection is a **bilevel** problem under a fixed budget `B`: inner level fits the tiny calibration
  `φ = {φ_τ, φ_1..φ_L}`; outer level picks selection weights `w ∈ R^N_{≥0}, Σw_i = B` minimising
  `J(w) = L_cur + β L_anc + γ Ω_spec`, where `Ω_spec(w) = −log det(C(w) + εI)` is a **log-determinant
  spectral-coverage penalty** on the weighted covariance of projected retrieval keys — i.e. an explicit
  *anti-redundancy* term over the selected memory. Top-B is committed. Regret bounds `O(√T)` static,
  `O(√T·√(D²+DV_T))` dynamic. Results: +0.88/+1.12 over HiDe-LLaVA on UCIT, AP 51.34 → **52.64** over
  CL-MoE on VQAv2-10task with AF 1.70; ablating **retrieval-weighted induction costs up to 4.20 AP**;
  memory attention mass rises from **≈0.018 (early layers) to ≈0.094 (late layers)**, and tasks with larger
  gains show stronger late-layer memory usage.
- **Is it query-dependent?** **Yes for `α_i(x)` — this is the second unbounded formulation in this file.**
  Note the operator form carefully: scaling `K̃_i` by `√α_i` scales the *logit* multiplicatively
  (`√α_i · q·k`), which is **not** an additive log-bias — it is sign-sensitive (boosts positive logits,
  suppresses negative ones) and its coefficient depends on `x`. β is additive and `x`-free. **`λ_ℓ` is
  query-independent and is exactly β's class** — and the paper's own ablation says the *retrieval weighting*,
  not `λ`, is what carries the gain (4.20 AP). That is independent corroboration of our β result.
- **Gradient-free?** **Almost.** `φ` has exactly `L+1` scalars — for us **37** — and is fit by a few inner
  steps; at that size it is grid-searchable or fittable by coordinate descent with **zero optimiser steps on
  the cartridge** (`gradient_steps` stays 0 by this project's own definition, §1 of the GLOSSARY, but the
  cost must be reported). The **outer** selection uses projected gradient on `w` — but `w` is a selection
  weight, not a cartridge parameter, and for `N ≈ 16–512` candidates it can be replaced by a greedy
  log-det / DPP maximisation, which is closed-form and submodular-greedy.
- **Maps to our code.** The α-gate is implementable *without touching the write path*, on the cartridge we
  already have:
  1. Build one retrieval key `r_d` per document (pooled final-layer hidden state of the document, frozen
     forward pass — `cartridges/am/teacher.py` already does document forwards).
  2. Build a per-slot owner map from the saved `am_doc_*.pt` selections (which document last wrote slot `j`).
  3. At eval, per example, compute `r(x)` from the prefill (one extra prefix pass, which
     `eval_forgetting.py` effectively already runs), then `α_d(x)`, then bias slot `j` by
     `½ ln α_{owner(j)}(x)` inside `cartridges/models/attention.py:106`'s `score_mod` — the closure already
     receives `b`, so a `(B, n_kv_heads, T_c)` bias needs no new plumbing (`cartridges/cache.py:203`
     currently returns `(1, …)`). Flag: `AM_QUERY_GATE=induce_alpha`, `AM_QUERY_GATE_TAU`.
- **Prediction about OUR signal.** With 16 owner-documents and a temperature `τ` tuned on a held-out split:
  MT/QA selectivity **1.05 → 1.25–1.9** (α concentrates on the MT documents for MT questions; QA questions
  see none of the 16 as relevant, so α is near-uniform and the bias is near-zero — an asymmetry that works
  in our favour); MT routing mass **0.2799 → 0.38–0.55** ⇒ by the law **MT 2.224–2.166**, i.e.
  **ΔMT −0.04 to −0.10** (0.9–2.0× the resolution). QA **neutral to −0.03**. The honest read: **plausibly
  the first mechanism to move selectivity out of the 1.04–1.05 band, and still ~0.15–0.20 short of 2.02.**
- **Why it might NOT transfer.** Three real risks. (i) **They concatenate; we overwrite.** InduceKV's memory
  is `|M|·m` *appended* tokens with per-entry keys intact; our 16 documents were written into overlapping
  slots at 9.83 writes/slot, so `owner(j)` is a lossy fiction — 87.3% of slot-writes land in slots touched by
  ≥8 documents. The gate can only be as good as the ownership map, and ours is 87% ambiguous. A cleaner
  variant would need per-document slot disjointness, which is L03-2, which we predict loses.
  (ii) Their α operates at the *prefix* level (one weight per example), not per attention query — so it
  cannot separate two questions inside one prefix. (iii) Their memory attention mass is **0.018–0.094**,
  far below our 0.2799 — their gains come from a *small*, well-targeted memory contribution, which is
  evidence against "more mass is the answer".
- **Verdict: CANDIDATE — joint strongest with L03-8**, and the two compose: use Quest's per-query criticality
  as the *within-document* term and InduceKV's α as the *across-document* term.

### L03-17: What happens when you stop overwriting — *Cartridges at Scale: Training Modular KV Caches over Large Document Collections* (arXiv 2606.04557, 2026, https://arxiv.org/abs/2606.04557)
- **Claim / mechanism.** Independently trained cartridges **catastrophically interfere** when combined —
  the paper reports collapse to near-chance. Their fix is **mixed-visibility training**: each training
  example sees the relevant cartridge alone with probability `P_iso = 0.75`, and otherwise the relevant
  cartridge plus `k` randomly sampled **distractor** cartridges, teaching the frozen model to attend
  selectively. At inference, cartridge selection is **training-free**: an off-the-shelf dense retriever maps
  retrieved document chunks to cartridges, and the selected KV caches are **concatenated** into the prefix.
  Scales to 496 documents / >1M tokens with a GPU "budget manager" rotating `B` cartridges. Base: context
  distillation on **Qwen3-8B**, frozen weights, cartridge KV the only trainable parameters.
- **Is it query-dependent?** **Yes, at the cartridge level** — the retriever reads the query. And the
  *within*-cartridge attention is the frozen model's own, so nothing is gated.
- **Gradient-free?** The **routing** is (dense retrieval, no training). The **cartridges** are not — and
  their central finding is that the cartridges must be trained *jointly, with distractors present*, or
  composition fails. That is a gradient requirement on the thing we are trying to build gradient-free.
- **Maps to our code.** No mapping into the current fixed-slot write path exists; this is a different
  regime (`N` cartridges concatenated, not 16 documents into 512 slots). The nearest legal probe on our
  side is `examples/qasper2/train/eval_forgetting.py` in **cartridge mode with a concatenated pair of
  caches** (Phase-1 QA cartridge ‖ a Phase-2 MT cartridge) — a 2-cartridge composition control with
  **zero** new mechanism.
- **Prediction about OUR signal.** Concatenating an independently-solved MT cartridge onto the Phase-1 QA
  cartridge, both untrained-for-composition: their result predicts **interference**, and ours predicts a
  *split* outcome — QA should return toward the untouched floor **2.2388** (the QA slots are no longer
  overwritten) and MT should land **between** the ICL number and the k=12 point. Given our content-free
  control (documents with zero MT content reproduce **28.3–73.2%** of the MT gain, and full-context ICL is
  **53.3%** content-free), I predict **QA 2.15–2.28, MT 2.10–2.45**, with the MT end dominated by whether
  the second cartridge is solved on-distribution. **A composed pair beating MT 2.2681 while keeping QA ≤ 2.52
  would be the project's first PASS-shaped result — and it would come from abandoning the fixed slot set.**
- **Why it might NOT transfer.** Their cartridges are *gradient-trained* with mixed visibility; ours would be
  closed-form and composition-naive, which is precisely the setting they report collapsing to near-chance.
  Also they use Qwen3-8B and doubled cache footprint, which changes the comparison's ruler.
- **Verdict: candidate — but it is a scope change, not a gate.** It is the strongest external evidence for
  the synthesis's own conclusion that the win lives in non-fixed-slot formulations, and it says the price of
  admission is joint training. Flagging for the human, not proposing it as a gating mechanism.

---

## Direct answers to the three questions

**(a) Is there a gating formulation whose selectivity is query-dependent, and therefore not bounded by our
β result?** Yes — three, in descending order of usability here: **Quest per-query criticality** (L03-8,
fully training-free, per-*query*), **InduceKV's α_i(x)** (L03-16, 37 scalars, per-*prefix*), and
**Routing-Transformer clustering** (L03-7, EMA k-means, per-query — but *dead on our substrate*, because
QA/MT routing cosine is 0.99899 and there is nothing to cluster on). The delta rule (L03-13) is a
query-dependent **write** rather than gate. Everything else in this file — HAT, SPLADE, Hash Layers, Switch,
NSA's branch gate, InduceKV's λ_ℓ, Soft MoE's dispatch as applied to us, expert-choice, BASE, Slot
Attention — is **query-independent and therefore bounded by the β result**, and I have marked each as such
rather than listing it hopefully.

**(b) Does any of them work without gradients, and if a gate must be learned, is there a closed-form or
amortised route?** Quest needs **no training at all** (two reductions over the cartridge keys + a dot
product per query). Routing Transformer's centroids are **EMA k-means**. BASE's assignment is an **auction
algorithm**. Hash Layers show a fixed unlearned router **beating** a learned one. NTM addressing, DNC usage
allocation and the delta rule are **closed-form**. InduceKV's gate is **37 scalars**, small enough to
grid-search, and its outer selection's log-det coverage objective is submodular ⇒ greedy is near-optimal
without gradients. **So "a useful gate must be learned" is false.** What *is* true: NSA and
Cartridges-at-Scale both argue explicitly that the *substrate* must be trained with the sparsity/composition
in the loop, and neither has a post-hoc variant.

**(c) Does expert-choice change the overwrite picture at 9.83 writes/slot?** Arithmetically yes —
capacity `c` caps writes/slot at `c`, taking 9.83 → 1.0 and survival 4.7% → ~100%. Causally, **our own
measurements predict it loses**: DIAG-PERDOC's uncontested solo write is **+1.101 worse** (5/5 documents,
5–14× the noise), 71–93% of a solo write's gain lands on *other* documents' questions, and ORACLE-WRITE-512
showed full support makes the 16 documents mutually annihilate. Expert-choice at `c=1` **is** the
full-support regime reached by a better route. Predicted MT ≥ 2.45. Run it to falsify DIAG-PERDOC, not to
fix acquisition.

**The honest conclusion the brief asked for.** Within a **fixed 512-slot, gradient-free, overwrite**
formulation, gating cannot reach MT ≤ 2.02. The measured law `MT = 2.0613 − 0.1574 ln(m)` caps the family at
**2.061 even at 100% routing mass**, and the dense bar would need `m = 3.32`. The only operator class that
can leave that law is query-dependent selectivity, and the best case I can construct from the literature
(Quest ∘ InduceKV on the key-installed cartridge) predicts **MT 2.15–2.22** — real, 2–4× the resolution, and
**still 0.13–0.20 short**. Useful *closing* gating in this setting requires either **gradients on a router**
(NSA, HAT, channel gating, PKM query network, Cartridges-at-Scale all say so) or a **cache that retrieves
and concatenates rather than overwrites** (InduceKV, Cartridges-at-Scale). The human may reasonably open the
scope on that basis; my recommendation is to spend one cheap oracle first (below) so the scope change is
made against a measured ceiling rather than a literature argument.

---

## Top 3 experiments I would run next

Baseline for all three: **QA 1.9468 / MT 2.2681** (seed-mean; k=12 single-seed 1.9560 / 2.2720),
MT routing mass **0.2799**, selectivity **1.0424–1.0545**, resolution **±0.049 MT / ±0.054 QA**.
Each is single-variable and each has a falsifier.

### E1 — `GATE-ORACLE`: the ceiling of *any* query-dependent gate, eval-time only, zero writes changed
**Variable:** the per-slot attention bias becomes **per-example and query-dependent**, oracle-fitted, vs the
static β. Nothing about the write, the solve, or the selection moves. **Not a mechanism — a bound.**
**How:** take the existing best cartridge (`cache-after-doc-011-*.pt` from the MECH-005 best point).
In `cartridges/models/attention.py:100–110`, `score_mod(score, b, h, q_idx, kv_idx)` already receives `b`;
supply a `(B, n_kv_heads, T_c)` bias instead of `(1, …)` (`cartridges/cache.py:203`). For each eval example,
compute the bias that **maximises that example's attention mass on the written union `S`**, box-clamped to
`|bias| ≤ AM_BETA_BOX = 3.0` so it is *the same operator budget β had* — making this a paired test against
MECH-BETA. Run on both splits. **Cost:** two evals, no solve, no training, `gradient_steps = 0`.
**Prediction:** MT routing mass **0.2799 → 0.55–0.80**; selectivity **1.05 → 1.6–3.0**; **MT 2.16–2.11**
(ΔMT −0.11 to −0.16, i.e. 2.2–3.3× resolution); QA neutral to −0.05. **It will not reach 2.02** —
the law's ceiling is 2.061 at m=1.
**Falsified if:** the *oracle* ΔMT is **> −0.049** (inside resolution) or oracle selectivity stays **< 1.10**.
Either outcome closes the entire query-dependent gating family with a measured cause, and the scope should
then be opened to non-fixed-slot formulations immediately.
**If it survives**, E2 is the realisable version and the oracle tells us what fraction of the ceiling a real
gate recovers.

### E2 — `GATE-ALPHA`: the realisable query-dependent gate (InduceKV α ∘ Quest criticality)
**Variable:** the same bias slot as E1, but computed from **retrieval only** — no oracle, no labels.
`bias_j(x) = ½ ln α_{owner(j)}(x)`, `α_d(x) = softmax_d(⟨r(x), r_d⟩ / τ)` with `r(·)` the frozen model's
unit-norm pooled prefix state (L03-16), optionally multiplied by a Quest criticality term
`Σ_i max(q_i m_i, q_i M_i)` over each slot's key (L03-8). `owner(j)` from the saved `am_doc_*.pt` selections.
Only `τ` is fit — one scalar, grid-searched on a held-out split; `gradient_steps = 0`.
**Prediction:** selectivity **1.05 → 1.25–1.9** (the *first* mechanism in this project to leave the
1.04–1.05 band); MT routing mass **0.38–0.55**; **MT 2.224–2.166** (ΔMT −0.04 to −0.10);
QA **−0.03 to +0.02**. Recovers **50–75%** of E1's oracle gap.
**Falsified if:** selectivity stays **< 1.10** (⇒ the 87.3%-ambiguous `owner(j)` map is the binding problem,
not the gate), **or** ΔMT is inside ±0.049. Either result means query-dependence is available in principle
(E1) but not recoverable from our overwritten cartridge — which is a *different and more actionable*
conclusion than "gating doesn't work", because it points at allocation, i.e. E3.

### E3 — `ROUTE-EXPERT`: expert choice (slots pick documents) with a capacity constraint
**Variable:** the **direction of the top-k only.** `SLOT_SELECTION=expert_choice` in
`cartridges/am/ranking.py`, using the existing `tf` scores; each of the 511 writable slots takes its top-`c`
documents (`AM_CAPACITY_FACTOR`, `c = 1` and `c = 4`), solved optimally with `linear_sum_assignment`
(L03-4) rather than greedily. Requires a two-pass driver in `continual.py` (pass 1 records all 16 `tf`
matrices, pass 2 assigns and writes) — the only structural code change of the three.
**Prediction (negative, and that is the point):** writes/slot **9.83 → 1.0** (c=1) / **~4** (c=4); union
54.4 → **~511** / **~200** slots/layer; survival **4.7% → ~100%** / **~40%**; Jaccard **0.713 → 0.00**;
per-document MT routing mass falls; **MT ≥ 2.45 at c=1** (most likely 2.52–2.60, matching the k=16 endpoint
2.552 and the ORACLE-WRITE-512 annihilation regime) and **2.33–2.40 at c=4**. QA **improves 0.05–0.15**.
**Falsified if:** **MT ≤ 2.219** at either capacity. That would refute DIAG-PERDOC's +1.101 and
ORACLE-WRITE-512 simultaneously and would reopen the allocation family, which the synthesis currently
records as closed. It is the cheapest available attack on the project's own strongest negative, and the
direction reversal (expert-choice vs token-choice) has literally never been run here.

**Not proposed, and why:** delta-rule writes (L03-13) and partial erase (L03-15) — both predicted to lose
for the same measured reason (better fitting and less interference are both anti-correlated with CE here),
and both cost a full re-solve; Routing-Transformer clustering (L03-7) — dead on a substrate with 0.99899
QA/MT routing cosine; anything requiring gradients on a router (L03-9, L03-11, L03-12, L03-14) — out of
scope by the mission's `gradient_steps = 0` rule.
