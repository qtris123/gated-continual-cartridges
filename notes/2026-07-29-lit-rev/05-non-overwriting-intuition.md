# LITREV-05 — The non-overwriting intuition itself: is it right?

**Thread:** W2 SCOUT (literature) · **Date:** 2026-07-29 · **Papers read:** 14
**Grounding:** `../2026-07-29-am-investigation-synthesis.md` · `../../research_loop/GLOSSARY.md` ·
`../../research_loop/state/bottleneck_board.md`
**Scope note:** no GPU, no source edits, no commits. Every claim below is either a quotation from a paper
I read or a number already on our board.

---

## Executive summary — the answer, stated plainly

> **The founding premise is mis-specified for this setting.** Not "true but untestable here", not "right
> in principle, wrong in the details" — **mis-specified**: it names the wrong quantity as the thing to
> conserve, and its single strongest prediction is measured on our own bench **with the wrong sign**.

The premise says *"don't overwrite slots that carry important Phase-1 knowledge, and you will retain
Phase-1 while still acquiring Phase-2."* Three of its four load-bearing assumptions fail here:

1. **It assumes retention and acquisition are separate objectives with separate knobs.** They are not.
   Hiratani (NeurIPS 2024, L05-3) derives, for exactly this class of mechanism, that random slot/activity
   gating *is* a single scalar knob α trading transfer against retention, and that Euclidean weight
   regularisation is **algebraically the same knob**. Biderman et al. (TMLR 2024, L05-4) measure the same
   single Pareto ray in an LLM: raising LoRA rank moves you "up and left — learning more and forgetting
   more". **Our MECH-CONSTRAINED dose-response is that ray**: q = 1.00→0.25 moves MT routing mass
   0.2799→0.1661→0.1327→0.1000 and MT loss 2.2720→2.3288→2.3693→2.4388 monotonically, Pearson −0.9775.
   There is no setting of the knob that buys retention for free, because there is no second knob.

2. **It assumes a protectable subspace exists.** Hiratani gives the exact condition under which
   Fisher-metric protection is free: retention is perfect *only* while `N_s ≪ N_x(1 − ρ_a²)`, where ρ_a is
   the **input-feature similarity** of the two tasks. Our ρ_a is the QA/MT mean-routing cosine
   **0.99899**, so `1 − ρ_a² = 0.00202` and with `N_x = 511` writable slots the protectable budget is
   **`N_s ≪ 1.03` slots per layer**. On the more generous top-32-overlap reading (ρ_a = 0.914) it is
   `N_s ≪ 84`, still under the 54.4-slot written union at 9.83 writes/slot. **The subspace the intuition
   needs does not exist by two to three orders of magnitude** — which is precisely what DIAG-KEYSPACE
   measured independently (ρ_key at its held-out control floor; QA/MT query separation 2.5× *smaller*
   than sampling noise). Hiratani also notes that the *diagonal* Fisher approximation "makes the metric
   full-rank even when the true metric has a low-rank structure", destroying the invariance — and MECH-008's
   `fisher` selector is a diagonal empirical Fisher. So we implemented the fragile variant of a method
   whose enabling condition we fail anyway.

3. **It assumes isolation is the ideal limit.** This is the prediction that dies. Non-overwriting, taken
   to its limit, *is* disjoint allocation: 100% survival, zero contention. We measured it. A solo write
   into 32 uncontested slots is **+1.101 worse on its own document's questions** (sd 0.186, 5/5
   documents, 5–14× the paired resolution) than the same document written as one of 16 — and 71–93% of a
   solo write's gain lands on the *other* documents' questions. **The best case for the intuition is our
   worst measured arm.** Lee et al. (ICML 2022, L05-1) predict exactly this and name the mechanism: strong
   consolidation "amplif[ies] the bias to fresh node activation", and at high λ "the task in the second
   phase of learning is akin to learning a new random teacher with a **tabula rasa node**". A tabula rasa
   node is a fresh uncontested slot. It is the *worst* way to learn a task that overlaps what you already
   have — and at ρ = 0.99899 ours is the maximally overlapping case.

4. The one assumption that survives is that **overwriting happens**. It does: Jaccard 0.713, 9.83
   writes/slot, 4.7% survival. It just doesn't cost what the premise says. Document 1's values drift by a
   median 99% of their own magnitude and document 1 still contributes **3.1%** of the total MT gain while
   documents 2–16 contribute **96.9%** — because, per Hendel et al. (L05-6) and Min et al. (L05-5), the
   thing carried is not a per-item store whose loss is proportional to slot survival.

**The narrow sense in which the intuition is not "wrong":** it correctly names a real mechanism
(interference), and there *is* a regime where it pays — low feature overlap, an item-level retrieval
metric, and a sub-capacity number of writes. **We are in none of those three.** So the correct summary is
not "protection was untested because retention was already satisfied". Protection was tested, works
exactly as advertised on the axis it protects (`fisher`: QA 1.8641, the best retention ever measured
here, at 0.04% QA-Fisher exposure), and is **strictly dominated** because in this regime the axis it
protects and the axis we need are the same axis with opposite signs.

**What replaces it** is in the final section: *cumulative shared adaptation, capped at capacity, measured
against a matched-domain control* — and one experiment that discriminates the two in a single sweep.

---

## The five facts, indexed (used by reference below)

| id | fact | source |
|---|---|---|
| **F1** | `fisher` selector: best retention ever measured (QA **1.8641**, 0.04% QA-Fisher exposure), pays **+0.397 MT**. Retention already **0.573 ahead** of budget; acquisition **0.248 short**. Dose-response Pearson `log(MT mass)`→MT loss **−0.9775**. | MECH-INFOGATE, MECH-CONSTRAINED |
| **F2** | Survival **4.7%**, Jaccard 0.713, **9.83 writes/slot**; doc 1 contributes **3.1%** of MT gain, docs 2–16 **96.9%**. | DIAG-OVERWRITE, DIAG-SEQUENCE |
| **F3** | Solo write into 32 uncontested slots: **+1.101 worse** on its own questions (5/5); **71–93%** of a solo write's gain lands on other documents' questions; solo buys **15.5%** of the 16-doc gain. | DIAG-PERDOC |
| **F4** | Zero-MT-content documents reproduce **28.3–73.2%** of the MT gain; full-context ICL reproduces **53.3%** under deliberate topic mismatch; writing the QA papers in **degrades QA by +0.652** on those very papers. | DIAG-CONTENT, D0-ICL |
| **F5** | QA/MT routing cosine **0.99899**, top-32 overlap **0.914**; query-space separation **2.5× smaller than sampling noise**. Curve saturates at **k≈12** then degrades; reversing document order costs **+0.340 MT**. | DIAG-KEYSPACE, DIAG-SEQUENCE, DIAG-CONTENT arm C |

---

## Papers

### L05-1: Protection converts re-use into activation — *Maslow's Hammer for Catastrophic Forgetting: Node Re-Use vs Node Activation* (arXiv:2205.09029, ICML 2022, PMLR 162:12455–12477)

- **Claim.** Forgetting is **non-monotonic** in task similarity: worst at *intermediate* similarity, not
  at maximal dissimilarity. The mechanism is a trade-off between two ways a network can learn task 2:
  **re-use** the units already specialised to task 1, or **activate** fresh, previously dormant units. At
  high similarity re-use is cheap and correct ("the student mostly re-uses the previously specialised node
  and ignores" the fresh one); at orthogonality the network naturally activates a fresh node and old
  knowledge survives untouched. Intermediate is worst because it does a costly hybrid. They then show what
  EWC actually does: "The effect of EWC is to intensify this ... bias to fresh node activation since
  movement in the weights contributing to the specialised node is penalised", and at large λ "the task in
  the second phase of learning is akin to learning a new random teacher with a **tabula rasa node**. Hence
  the learning trajectories collapse onto one." They also state the cost explicitly: "the conservatism
  induced by strong EWC (comparable in this setting to **freezing a node**), limits even the fully aligned
  setting". Bonus result — *catastrophic slowing*: in the intermediate regime, re-initialising the whole
  network at the task boundary beats interleaved replay, on both transfer *and* forgetting.
- **Which facts.** Explains **F3** and **F1** together, and it is the cleanest available account of both.
  Our `fisher`/`constrained_mass` selectors *are* the "freeze the important node" operation, in slot space
  instead of weight space. The paper's prediction for that operation is: it forces the write onto fresh,
  uncontested capacity, and that costs acquisition. F1 measures the cost (+0.397 MT) and F3 measures the
  limit case with the sign the paper predicts (+1.101 worse when the fresh capacity is *all* the write
  gets). At ρ = 0.99899 (**F5**) we are in the paper's "fully aligned" regime, where its verdict on EWC is
  that it "has little effect" on forgetting while still limiting acquisition — which is exactly F1's
  ΔQA −0.082 (inside our ±0.054 paired band on most arms) against ΔMT +0.397 (8× the band).
- **DO differently.** Stop treating slot contention as damage. The paper's own remedy for the aligned
  regime is to *encourage* re-use: in our code that means the exact opposite of `AM_SAFE_FRACTION < 1` —
  push toward a **document-independent** slot set (`ranking.py` already knows how: the across-document mean
  score's top-32 already agrees with each document's own top-32 at **87.4%**). Also: run the analogue of
  the paper's "catastrophic slowing" control — a **from-Phase-0** write (no Phase-1 cartridge to build on)
  — to test whether our Phase-1 initialisation is helping or slowing the Phase-2 write. We have never run it.
- **Verdict: undermines the intuition.** It supplies the mechanism by which protection *causes* the
  acquisition loss, rather than merely failing to prevent it.

### L05-2: Similarity buys fast transfer and blocks late transfer — *Continual Learning in the Teacher-Student Setup: Impact of Task Similarity* (arXiv:2107.04384, ICML 2021)

- **Claim.** The analytically solvable precursor to L05-1. Two contributions matter to us. (i) The same
  non-monotonicity: "when tasks depend on similar features, intermediate task similarity leads to greatest
  forgetting." (ii) A dissociation the later paper does not have: **feature similarity** (input→hidden) and
  **readout similarity** (hidden→output) act differently, and "transfer depends on readout similarity even
  for teachers with identical features". On transfer dynamics: "Just after the switch, higher overlap
  allows faster transfer. All students then reach a second plateau. Only students trained on tasks that
  are close to orthogonal break away from this second plateau ... We thus find that **task similarity aids
  short-term transfer but harms long-term transfer**."
- **Which facts.** Predicts **F5**'s saturation shape. Our whole procedure is a *short-horizon* write: 16
  closed-form solves, `gradient_steps = 0`, no long optimisation. The paper says that is exactly the
  regime where high overlap **helps** (fast transfer), and that the penalty for overlap only appears in the
  long-run plateau we never reach. It also gives a second, independent reading of **F5**'s k≈12 plateau:
  the "second plateau" that only near-orthogonal tasks escape. And the feature/readout split is a warning
  about our own diagnostics: DIAG-KEYSPACE measured *feature*-side similarity (query/routing space) only.
  QA and MT share the readout completely (same LM head, same CE), so on this paper's axes we are at
  ρ_feature ≈ 1, ρ_readout ≈ 1 — the corner where re-use dominates.
- **DO differently.** Two cheap diagnostics we lack: (a) a **readout-side** similarity measurement (cosine
  between QA and MT *output-logit* gradients w.r.t. the cartridge, not query-space overlap) to place us on
  this paper's 2-D map rather than its 1-D one; (b) since short-horizon writes favour overlap, test whether
  *iterating* the 16-document write (a second pass over the same documents) moves us from the fast-transfer
  regime toward the plateau — the paper predicts the overlap advantage should shrink if so.
- **Verdict: reframes the intuition.** Overlap is not uniformly harmful; its sign depends on horizon, and
  our horizon is the one where it is positive.

### L05-3: Gating *is* the retention/transfer knob, and Fisher protection has an enabling condition we fail — *Disentangling and mitigating the impact of task similarity for continual learning* (arXiv:2405.20236, NeurIPS 2024, Hiratani)

- **Claim.** A linear teacher–student model with latent structure, solved analytically, for exactly the
  interventions we built. Four results. (i) "high input feature similarity coupled with low readout
  similarity is catastrophic for both knowledge transfer and retention"; the converse is benign.
  (ii) **Random task-dependent activity gating** (keep a fraction α of units active per task) has
  transfer optimised at `α* = min{ρ_b/ρ_a, 1}` — "indicating that the input activity typically needs to be
  **dense**" — while "the retention performance is optimized at α → 0 limit where tasks do not interfere
  with each other". Verbatim: "random activity gating improves the retention at the cost of forward
  knowledge transfer". (iii) Euclidean weight regularisation with amplitude `(N_x/N_s)(1/γ − 1)` is
  "**mathematically equivalent** with random activity gating with sparsity γ, in terms of knowledge
  transfer" — and adds a `(1 − γ)²` term meaning "strong weight regularization not only prevent
  forgetting, but also **impairs task acquisition**". (iv) Regularisation in the **exact** Fisher metric is
  the one thing that escapes the trade — retention error is `O(N_s / (N_x(1 − ρ_a²)))`, i.e. perfect while
  `N_s ≪ N_x(1 − ρ_a²)`, "unless the two tasks share the same feature" — but "this invariance no longer
  holds when the Fisher information metric is approximated by its **diagonal** component, as is done in the
  elastic weight methods ... because the diagonal approximation makes the metric full-rank even when the
  true metric has a low-rank structure."
- **Which facts.** This is the theory of **F1**, and it quantifies **F5**. Our `AM_SAFE_FRACTION` q and our
  `TOP_T` are literally this paper's α; our measured dose-response (MT mass 0.2799→0.1000 as q 1.00→0.25,
  MT loss 2.2720→2.4388, Pearson −0.9775) is its Fig. 2A/2C read off a real model. Its α* rule says the
  optimum is **dense**, i.e. q = 1.0 and large `TOP_T` — which is the incumbent, and is what MECH-CONSTRAINED
  found by sweeping. And its enabling condition for free protection is the number I opened this note with:
  `N_x(1 − ρ_a²) = 511 × 0.00202 = 1.03` slots at ρ_a = 0.99899, or 84 at ρ_a = 0.914 — against a written
  union of 54.4 slots/layer. **There is no low-rank structure for the protection to hide in**, and our
  implementation is the diagonal approximation that this paper singles out as the fragile one.
- **DO differently.** (a) **Stop reporting `fisher` as a candidate mechanism** — the theory says a diagonal
  Fisher gate at ρ_a ≈ 1 can only trade, and our sweep confirms the slope; that family is closed twice
  over. (b) If anyone insists on one more protection experiment, it must be the **non-diagonal** form —
  regularise `‖(V − V_phase1) A_QA‖_F` (project onto the QA query second moment `Q₀`, which
  DIAG-KEYSPACE already computes) rather than a per-slot diagonal weight. The paper predicts it is free
  *only* if `N_s ≪ 1.03`, so this is a **falsification run, not a hope**: it should fail, and if it fails
  the whole protection family is closed analytically rather than empirically. (c) Its **adaptive gating**
  idea is the one constructive item: a probe trial that measures how well task-1's gating serves task 2 and
  reuses it with probability `1 − ε_probe/ε_0`. In our terms: reuse the *previous document's* slot set
  unless the residual says otherwise — a one-line change in `continual.py` and a genuinely untried arm.
- **Verdict: undermines the intuition** — and is the strongest single paper in this review, because it
  derives our measured trade-off from first principles and states the condition under which the intuition
  would have been right, in a form we can check with numbers we already have.

### L05-4: Constraining the update moves you *along* one Pareto ray, not off it — *LoRA Learns Less and Forgets Less* (arXiv:2405.09673, TMLR 08/2024)

- **Claim.** Large-scale LLM measurement of the exact trade. LoRA at low rank "substantially
  underperforms full finetuning" on code and math, and "LoRA forgets less than full finetuning, and the
  extent of forgetting is **controlled by rank**". The Pareto analysis is the key part: "As models train on
  more data, they learn more and forget more, traveling up and left in this space. **As we increase LoRA
  ranks, we find that the curves shift up and left as well, again, learning more and forgetting more**" —
  i.e. the constraint level is a position on the frontier, not a better frontier. They also find LoRA
  "mitigates forgetting more aggressively than classic regularization techniques such as weight decay and
  dropout", and that full finetuning's actual perturbation is high-rank ("10–100× greater than typical LoRA
  configurations"), i.e. the constrained parameterisation is not what the task wants.
- **Which facts.** Empirical confirmation of **F1** in an LLM at scale. Our `TOP_T`/q are their rank r; our
  MECH-CONSTRAINED table is their Fig. 3. Their "full finetuning finds high-rank perturbations" maps onto
  our finding that the useful write is broad and shared (**F2/F3**), not narrow and protected. Their honest
  conclusion — "it seems that LoRA can offer preferable learning-forgetting tradeoffs for code, while full
  finetuning can offer preferable tradeoffs for math" — is the general form of ours: which end of the ray
  is right is a property of the *budget*, and our budget is 0.573 of retention slack and 0.248 of
  acquisition debt.
- **DO differently.** Report every mechanism as a **point on the (QA, MT) frontier with its constraint
  level named**, not as a win/loss. `state/results.csv` already has both axes; add the constraint
  parameter (`TOP_T`, `AM_SAFE_FRACTION`, selector) as a column and plot the ray. Any future mechanism has
  to be shown to **shift the ray**, not slide along it — and the only thing that has ever shifted it here
  is MECH-005 (key installation + RoPE repositioning, ΔMT −0.1545 at unchanged QA), which is a
  *representation* change, not a protection change. That is the shape of thing to look for.
- **Verdict: undermines the intuition** in our budget regime; reframes it in general.

### L05-5: The content in a demonstration is worth ~2.6 points; the distribution is worth 3–16 — *Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?* (arXiv:2202.12837, EMNLP 2022)

- **Claim.** Replacing gold labels in ICL demonstrations with **random** labels costs almost nothing:
  "models see performance drop in the range of **0–5% absolute** ... less impact in ... multi-choice tasks
  (**1.7%** on average) than in classification tasks (**2.6% absolute**)"; for MetaICL, 0.1–0.9%. What
  *does* matter is the surrounding distribution: replacing the demonstration *inputs* with
  out-of-distribution text "significantly drops the performance ... by **3–16% in absolute**", and replacing
  the label set with random English words costs **5–16%** for direct models. Their conclusion: demonstrations
  work by specifying "(1) the label space, (2) the distribution of the input text, and (3) the overall
  format of the sequence", not by conveying the input→output mapping.
- **Which facts.** This is the published, gold-standard version of **F4**, and it settles the interpretation
  our board flagged as the "strongest surviving objection". Our arm B (QA-topic documents, MT eval) is
  their *OOD-input* ablation with the format and register held fixed — same QASPER corpus, same synthesis
  format, different topic. Their result says: format + input distribution carry the bulk, content carries a
  small remainder. Ours says the same with our numbers reversed in scale (28.3% content-free at k=16,
  73.2% at k=8; ICL 53.3% under topic mismatch). And our content term is **real and grows with k** — arm A
  beats arm B by 0.232/0.288/0.608/**0.882** at k=4/8/12/16, each outside the ±0.15 band. So: content is a
  minority of the *signal* and a majority of the *asymptote*.
- **DO differently.** **The MT loss as currently defined is not an acquisition metric.** It is
  content + register + format + prior shift, with content the minority term at every k below 16. Two fixes,
  both cheap and both already implementable in `eval_forgetting.py`: (i) always report
  **ΔMT(canonical) − ΔMT(matched-domain-different-document)** as the headline, i.e. make arm B a permanent
  control rather than a one-off diagnostic; (ii) add their *label-space* analogue — evaluate on MT questions
  whose *answers* are shuffled between documents. If our write is content, shuffling must destroy the gain;
  if it is register, it will not.
- **Verdict: reframes the intuition** — and reframes the target. It removes "the AM write is defective"
  from the table and replaces it with "the metric measures the wrong thing", which is a different project.

### L05-6: What ICL leaves behind is one vector, not a store — *In-Context Learning Creates Task Vectors* (arXiv:2310.15916, Findings of EMNLP 2023)

- **Claim.** ICL factorises into `A(S) → θ` and `f(x; θ)`: the demonstration set S is compressed into a
  **single hidden-state vector θ**, computed with a *dummy* query, and patched into one layer. "Across all
  models, our procedure maintains around **80–90% of the accuracy of regular ICL**, while the baseline
  reaches only 10–20%." The conflicting-task experiment is decisive: give the model demonstrations for task
  A but patch in the task vector for task B, and it answers **task B** at 77–95% — "the model mainly relies
  on θ, largely disregarding the demonstrations S".
- **Which facts.** Directly answers "is storage the right model of what this write does?" — **no**, and not
  only for our write: not even for the full-context baseline. It explains **F2** (why 4.7% slot survival
  costs 3.1% of the gain: a low-dimensional modulation is robust to losing most of its carriers) and
  **F4**'s ICL 53.3% (a task/register vector transfers across topic by construction). It predicts **F3**:
  if the write's product is a shared modulation rather than 16 separate records, then a single document
  written alone produces a *weaker version of the same shared thing* — which is exactly what we measured
  (solo buys 15.5% of the 16-document gain and 71–93% of it lands elsewhere).
- **DO differently.** Test the factorisation on our object, which is cheap and diagnostic: take the
  16-document cartridge, compute the **mean value-delta across the written slots** (rank-1 summary), write
  *only* that back into a fresh Phase-1 cartridge, and evaluate MT. If a rank-1/rank-k summary recovers most
  of the 1.230 MT gain, the write is a task vector, the entire slot-allocation family is irrelevant by
  construction, and the right search space is *which direction*, not *which slots*. This is a
  gradient-free, one-evening experiment on existing checkpoints and it discriminates two whole research
  programmes.
- **Verdict: undermines the intuition** at its root — "don't overwrite the slot" presupposes the slot is
  the unit of information. This paper says the unit is a direction, and directions superpose.

### L05-7: The content-free probe, and why order matters — *Calibrate Before Use: Improving Few-Shot Performance of Language Models* (arXiv:2102.09690, ICML 2021)

- **Claim.** Few-shot performance varies from near-chance to near-SOTA with the choice and **ordering** of
  demonstrations, and the cause is three biases: majority-label, **recency**, and common-token. Recency is
  measured directly: in 4-shot LAMA the model "overpredicts the answer from the 1st, 2nd, 3rd, and 4th
  training example by **8.5%, 8.3%, 14.3%, and 16.1%**" — a monotone gradient toward the end of the prompt.
  Their fix estimates the bias with a **content-free input** ("N/A", "[MASK]", empty string), fits
  `W = diag(p̂_cf)⁻¹`, and applies it at test time: up to 30.0% absolute gain, and "reduces variance across
  different choices of the prompt".
- **Which facts.** Two things at once. (i) It is the published parent of our **content-free control** —
  the "N/A input" trick *is* DIAG-CONTENT arm B, invented for the same purpose (measure how much of the
  apparent gain is prior shift). Their framing is the one we should adopt: a content-free probe is not a
  scandal, it is the **calibration constant**, and it should be subtracted, not agonised over. (ii) It
  explains **F5**'s order effect: reversing the 16 documents costs **+0.340 MT** with content held
  constant, and recency bias is the standard account — the last-written documents dominate the output
  distribution, which is also what DIAG-OVERWRITE's write-count asymmetry (doc 16 survives 100%, doc 8
  survives 1.0%) says at the mechanism level.
- **DO differently.** Two concrete items. (a) **Calibrate the MT metric.** Compute the model's MT-token
  distribution under a content-free cartridge write (arm B is already on disk) and report acquisition as
  the *calibrated* CE. This turns F4 from a confound into a subtractable constant. (b) **Neutralise order
  by averaging, not by ordering.** Their variance reduction comes from removing the bias, not from finding
  the good permutation; the analogue here is to write the 16 documents in a **random order per seed and
  report the mean** (MECH-007's `AM_SEED_OFFSET` already exists; extend it to permute document order). A
  0.340 order effect on a 0.248 deficit is not a nuisance — it is larger than the thing we are trying to
  measure.
- **Verdict: reframes the intuition** — the "acquisition" number we are optimising is partly an
  uncalibrated prior, and the order effect is a known, correctable artefact rather than evidence about
  overwriting.

### L05-8: How to define an acquisition metric that measures content — *Surface Form Competition: Why the Highest Probability Answer Isn't Always Right* (arXiv:2104.08315, EMNLP 2021)

- **Claim.** Raw string probability is a bad scoring function because answers compete for finite mass with
  paraphrases of themselves. The fix is **Domain Conditional PMI**:
  `exp PMI_DC(x, y, domain) = P(y | x, domain) / P(y | x_domain)`, where `x_domain` is a short "domain
  premise" — typically the tail of the prompt with the content stripped. Dividing by the domain-conditional
  prior "reweighs each option according to its a priori likelihood within the context of a specific task"
  and "achieves consistent gains ... over both calibrated and uncalibrated scoring functions on all GPT-2
  and GPT-3 models".
- **Which facts.** This is the direct, published answer to the thread's question 3: *how should an
  acquisition metric be designed so it measures content acquired rather than distribution shift?* Answer:
  **divide by the domain-conditional prior.** Our MT loss is `−log P(y | question, cartridge)`; the
  content-acquisition analogue is
  `−log P(y | question, cartridge_canonical) + log P(y | question, cartridge_matched-domain-control)`,
  i.e. per-example PMI against the arm-B cartridge instead of against nothing. That number, and only that
  number, is invariant to the 53.3% (**F4**) that full-context ICL also shows.
- **DO differently.** Add a `PMI_DC` mode to `eval_forgetting.py`: score every MT eval token twice — once
  under the canonical cartridge and once under a content-matched, topic-mismatched cartridge — and report
  the difference. Predicted immediate consequence, from the DIAG-CONTENT table: the canonical arm's
  calibrated gain is **0.232 / 0.288 / 0.608 / 0.882** at k = 4/8/12/16, so under a calibrated metric
  **k=16 is the best point, not k=12** — the k-curve's tail regression is largely a *content-free*
  regression. That single re-reading changes which checkpoint the project reports.
- **Verdict: reframes the intuition.** It does not speak to overwriting at all; it says our dependent
  variable is mis-specified, which is the same disease as the premise.

### L05-9: Domain adaptation transfers; off-domain adaptation does not — *Don't Stop Pretraining: Adapt Language Models to Domains and Tasks* (arXiv:2004.10964, ACL 2020)

- **Claim.** A second pretraining phase on in-domain text (DAPT) improves downstream task performance
  across four domains and eight tasks; a further phase on the task's own unlabeled data (TAPT) adds more.
  Critically for us they run the **¬DAPT control** — adapt to a deliberately *irrelevant* domain, "to
  control for the case in which the improvements ... might be attributed simply to exposure to more data"
  — and find "DAPT significantly outperforms adapting to an irrelevant domain ... we generally observe
  that **¬DAPT results in worse performance than even ROBERTA**".
- **Which facts.** This is the informative **contrast** to **F4**, and it pins down what our content-free
  gain actually is. In gradient-based DAPT, off-domain adaptation is *harmful* (below baseline). In our
  setting, "off-topic" documents deliver 28–73% of the gain and ICL delivers 53.3%. The difference is that
  our arm B is **not off-domain** — it is QASPER scientific papers in the same synthesis format, differing
  only in topic. So on this paper's taxonomy our content-free gain is a **domain/format** (DAPT-like) gain
  and our residual is the **document-level** (TAPT-like) gain. That is a much more precise statement than
  "the write is not a store", and it is testable: a genuinely off-domain arm (non-scientific text, same
  pipeline) should, per this paper, land **at or below the Phase-1 floor**. If it instead reproduces 28%
  of the gain, the effect is pure activation perturbation and the acquisition claim collapses further.
- **DO differently.** Run the missing third control. DIAG-CONTENT has arms A (canonical), B (same corpus,
  other topic) and C (reversed order). It needs **arm D: off-corpus** (e.g. news or code, same document
  count, same format wrapper). Cost: one write + one eval. It cleanly partitions the gain into
  {generic perturbation, corpus/format, topic, document}, and it is the only way to know which of the four
  the 0.248 deficit lives in.
- **Verdict: reframes the intuition** — and supplies the control that makes F4 interpretable instead of
  merely alarming.

### L05-10: More constituents = worse on your own task, better on everything else — *What Matters for Model Merging at Scale?* (arXiv:2410.03617)

- **Claim.** Merging up to 8 experts, 1B–64B. Held-in (each expert's own task) performance **degrades**
  with more experts — "merging eight 8B PaLM-2 models decreases performance from **0.66 to 0.39** when
  increasing the number of experts from 2 to 8" — while held-out (unseen task) generalisation
  **monotonically improves** with more experts, and past 6 experts at ≥24B "our merged model outperforms
  the multitask baseline". Strong base models both merge better and tolerate more experts.
- **Which facts.** This is **F3** and **F2** in a different medium, and it is the closest published analogue
  of the thing that most surprised us. "Held-in degrades with k, held-out improves with k" is exactly:
  writing document *i* among 16 makes the cartridge worse at document *i* than the k=2 case would, while
  71–93% of any single write's benefit shows up on the *other* documents' questions; and writing the QA
  papers in **degrades QA on those very papers by +0.652** (**F4**). Nobody in the merging literature calls
  that a bug — it is the expected shape. It also supplies a capacity reading of **F5**'s k≈12 turn: their
  held-in curves turn at a k that depends on base-model strength.
- **DO differently.** Adopt their two-axis reporting: for every cartridge, report **held-in** (mean CE on
  each written document's own questions) and **held-out** (CE on questions from documents *not* written).
  We have never separated these — the MT eval pools all 16. The board already contains the raw material
  (per-document eval subsets exist from DIAG-PERDOC). If our held-in curve turns at k≈12 while held-out
  keeps improving, then "stop at 12" is wrong and the right endpoint depends on which one we are selling.
- **Verdict: undermines the intuition.** In a literature built entirely on merging *interfering* updates,
  the interference is priced in and the aggregate still wins — and the per-item degradation we treated as
  pathology is the documented norm.

### L05-11: There is an overflow point, and the optimal policy is to store fewer items — *Scaling Laws for Associative Memories* (arXiv:2310.02984, ICLR 2024)

- **Claim.** For outer-product associative memories `W = Σ_x q(x) u_{f(x)} e_xᵀ` — the same algebra as our
  value write — error decomposes into a finite-data term and a **finite-capacity** term, and the model has
  two regimes: an "**overflow regime** where 3(d+1) ≤ N p*, and in essence the memory `W_q0` is too full to
  recover any signal in it", and an infinite-memory regime where all associations are storable near-
  orthogonally. The optimal scheme is **explicit thresholding**: `q̂_{ρ,[P]}` stores only the top-P
  associations, and "this maximum is reached for P ≃ d ... reminiscent of Hopfield networks which can only
  store d/log(d) patterns with a d by d matrix". Storing *everything* (ρ = 0, no threshold) is strictly
  suboptimal past capacity.
- **Which facts.** The capacity account of **F5**. Our MT curve falls strictly monotonically for 12 steps
  and then rises strictly monotonically for four (2.435 → 2.449 → … → 2.552); the content-free arm B turns
  even earlier, at k=8 (2.9978) and then degrades hard to 3.4343. Two curves, two different turning points,
  same shape — that is a capacity signature, not an interference signature (interference would scale with
  the *overlap* between arms, which is identical by construction). Note the theory's prescription is
  **not** "spread the writes out" — capacity is a property of the *key geometry*, and our keys are frozen
  Phase-1 keys with QA/MT top-32 overlap 0.914 (**F5**), so adding slots cannot add capacity. That is
  consistent with MECH-BUDGET-B, where t128 vs t64 gave ΔMT inside the resolution and **not even
  sign-consistent**.
- **DO differently.** Two things. (a) The theory says the right response to overflow is **thresholding
  over items**: select which *documents* to write, not which slots. We have never done document selection —
  a `q̂_{ρ,[P]}` analogue would rank documents by predicted marginal ΔMT and write only the top P. The board
  already shows the payoff is there: docs 1–3 hold 18.8% of eval questions and deliver **56.7%** of the MT
  improvement, and per-document marginal ΔMT correlates only 0.51 with eval share. (b) Since capacity is
  set by key geometry, MECH-005 (key installation + RoPE repositioning) is the *only* mechanism in the
  registry that could raise it — which is also the only one that ever moved MT (−0.1545). That is a
  coherent story worth pushing on.
- **Verdict: undermines the intuition** by supplying a competing, better-fitting explanation for the
  degradation the intuition claims. The k>12 rise is overflow, not overwriting — and overflow is fixed by
  writing *less*, not by protecting *more*.

### L05-12: Order is a first-class variable with a computable optimum — *The Effect of Task Ordering in Continual Learning* (arXiv:2205.13323)

- **Claim.** Permuting the task sequence changes forgetting far more than reseeding does: reseeding gives
  absolute change in average forgetting of **0.005 ± 0.003**, permutation gives a significant increase
  (p = 0.012), and one fixed order had "mean 0.11 higher average forgetting than order B, and 0.06 lower
  average accuracy". They then build orderings from an **asymmetric** task distance
  `c(j,k) = g_kᵀ H_j g_k` — the curvature of the *old* task's loss in the direction of the *new* task's
  gradient, computable by Hessian-vector product — and find, counter-intuitively, that the **maximum**-sum
  Hamiltonian path (consecutive tasks as *dissimilar* as possible) minimises forgetting: "a significant
  decrease (p = 0.008) ... corresponding to a mean decrease of −0.056" vs the minimum-sum path. They note
  "c is not symmetric, and moving from one task to another may be low curvature but the inverse high".
- **Which facts.** The literature answer to thread question 5. Our **F5** order effect (+0.340 MT from
  reversal alone, content held constant) is 7× larger relative to our own seed noise (±0.049 paired) than
  theirs is to theirs — so it is a *big* effect by CL standards, not an anomaly. Their asymmetry result is
  the mechanism: reversal is not a null operation because the pairwise cost is directed. And their
  max-sum-path result predicts the opposite of intuition for us too: consecutive documents should be as
  *dissimilar* as possible, not grouped by topic.
- **DO differently.** This is the cheapest untried lever on the board and it is worth **more than any
  gating result we have** (0.340 vs 0.167). Build the 16×16 directed distance matrix `c(j,k)` over the
  documents — with no gradients at all, using our existing per-document reference-query second moments as
  the curvature proxy, or with 16 diagnostic HVPs at ~100 s each (`gradient_steps` stays 0) — then run
  three arms: min-sum path, random (= canonical), max-sum path. Prediction from this paper: max-sum wins.
  If the max-sum path buys ≥0.1 MT, ordering alone is a bigger mechanism than everything in
  `mechanism_registry.md` except MECH-005.
- **Verdict: reframes the intuition.** The order effect is real, expected, directed, and *optimisable* —
  and it is orthogonal to overwriting (both arms have identical content and identical slot statistics).

### L05-13: Forgetting is a double-edged sword and sometimes the goal — *A Comprehensive Survey of Forgetting in Deep Learning Beyond Continual Learning* (arXiv:2307.09218, IEEE TPAMI 2024)

- **Claim.** The field's standing survey, and the only one that organises **beneficial forgetting** as a
  first-class category rather than an exception. Its three cases: selective forgetting mitigates
  overfitting; forgetting is *required* to learn new knowledge — "when a model contains outdated or
  unrelated knowledge, it can hinder its ability to effectively learn and generalize from new data ... **By
  freeing up capacity within the model, it becomes more receptive and adaptive to acquiring new
  knowledge**"; and machine unlearning for privacy. It surveys the forget-and-relearn paradigm ("adding a
  forgetting step can improve the generalization and effectiveness of model relearning") and Learn-to-Forget
  in meta-learning ("not all prior knowledge acquired through meta-learning is beneficial for learning new
  tasks").
- **Which facts.** Provides the field-level licence for the reading this note takes, and it names our
  situation exactly: capacity-limited (**F5**, k≈12 overflow), with old content occupying slots the new
  task needs, and a retention budget already met by 0.573. In this survey's taxonomy, our Phase-1 QA
  knowledge is not something to protect — 0.573 of it is **slack that should be spent**. The board already
  calls this "the reverse trade"; the survey says that is a recognised regime with its own literature,
  not an improvisation.
- **DO differently.** Formalise the reverse trade as the objective. Right now every selector is scored on
  (QA, MT) independently against fixed budgets. Score them instead on a **single scalarised objective that
  prices the slack**: e.g. `MT + max(0, QA − 2.52)·λ` with λ large. Under that objective `fisher`
  (QA 1.8641 / MT 2.6692) is not "best retention" — it is the worst arm in the bundle by 0.397, and
  MECH-INFOGATE's entire table reorders. This is a reporting change, not an experiment, and it prevents
  the next worker from re-discovering "protection works" and mistaking it for progress.
- **Verdict: undermines the intuition** as a *goal*, while conceding it as a *phenomenon*.

### L05-14: To keep learning, you must deliberately destroy — *Maintaining / Loss of Plasticity in Deep Continual Learning* (arXiv:2306.13812; Nature 632:768–774, 2024)

- **Claim.** The complementary failure to catastrophic forgetting, and the less well known one: networks
  trained continually **lose the ability to learn at all** — ImageNet binary classification "dropped from
  89% accuracy on an early task down to 77%, about the level of a linear network, on the 2000th task".
  Loss of plasticity survives architecture, optimiser, activation, batchnorm and dropout changes. Their fix,
  **continual backpropagation**, "selectively reinitializes low-utility units in the network" at a
  replacement rate ρ every step, using a contribution utility (running average of |outgoing weight ×
  activation|) and a maturity threshold protecting freshly reset units. In other words: the algorithm that
  sustains learning is one that **deliberately overwrites** the least useful storage, continually.
- **Which facts.** The strongest existence proof that the sign of the intuition can be inverted: here,
  *not* overwriting is the disease and scheduled overwriting is the cure. It reframes **F2** — 9.83
  writes/slot and 4.7% survival is not damage, it is a replacement rate — and it gives our `redundancy`
  score a second, opposite interpretation. MECH-008's redundancy (`1 − r_j²/‖v_j‖²`, gradient-free,
  ρ = −0.648 with Fisher) is structurally the same object as their **contribution utility**; we used it to
  decide what was *safe to overwrite* and it lost, because we spent the freed capacity on protection. Their
  use is the opposite: identify low-utility units and **reset them to fresh random values** to restore
  plasticity.
- **DO differently.** One genuinely new arm, and it costs nothing to build because the score already
  exists in `ranking.py::compute_slot_redundancy`: before each document's write, **reinitialise** the
  highest-redundancy ρ fraction of slots (to Phase-1 values, or to random draws from the cartridge's own
  value distribution) rather than merely permitting writes there. Prediction, if the plasticity account is
  right: the k>12 degradation (**F5**) is a plasticity collapse and a small replacement rate should push
  the turning point past 12 and lower the k=16 endpoint below 2.552. If the overflow account (L05-11) is
  right instead, resetting helps *only* by freeing capacity and a document-thresholding arm should match it.
  Those two are separable in one sweep.
- **Verdict: undermines the intuition** — the reference solution in this literature is scheduled,
  utility-ranked, deliberate overwriting.

---

## If the intuition is wrong, what replaces it?

### The alternative principle, stated so it can be falsified

> **Cumulative Shared Adaptation, capped by capacity, measured against a matched-domain control.**
>
> **(P1 — sharing beats isolation.)** The write does not deposit 16 retrievable records; it accumulates a
> single, largely document-agnostic adaptation whose quality grows with the number of documents that write
> into the *same* slots. Contention is therefore a **resource, not a cost**: the marginal value of a write
> is dominated by its contribution to the shared component, and isolation destroys that component.
> *Quantitative form:* per-document MT gain should increase, not decrease, with writes-per-slot, up to (P2).
>
> **(P2 — capacity, not interference, is the ceiling.)** Degradation past k≈12 is associative-memory
> overflow (L05-11) / merging saturation (L05-10), set by the frozen key geometry, not by slot survival.
> *Quantitative form:* the turning point should move with anything that changes key geometry (MECH-005) and
> **not** with anything that changes slot allocation (MECH-BUDGET-B: it didn't, ΔMT inside resolution and
> not sign-consistent at all four k).
>
> **(P3 — the metric is a PMI, not a CE.)** "Acquisition" must be scored against a matched-domain,
> mismatched-content control (L05-8, L05-9), because ≥53.3% of raw MT gain is available to a topic-mismatched
> context even for full-context ICL. *Quantitative form:* under calibration the k-curve's tail regression
> should largely vanish (canonical−control widens monotonically: 0.232 / 0.288 / 0.608 / 0.882 at
> k = 4/8/12/16), moving the reported best point from k=12 to k=16.

The two principles make **opposite predictions about the same knob**, which is what makes them testable:

| | non-overwriting (incumbent premise) | cumulative shared adaptation |
|---|---|---|
| best allocation | disjoint — each document its own slots | maximally shared — all documents the same slots |
| survival 4.7% | the bottleneck | irrelevant; it is a replacement rate |
| solo write | should be **best** on its own questions | should be **worst** (measured: +1.101 worse) |
| more slots (`TOP_T` ↑) | should help (less contention) | should be neutral (capacity is key-set-bound) |
| k>12 degradation | accumulated overwriting | capacity overflow |
| `fisher` gate | should win | must lose by construction (measured: +0.397) |

### The single discriminating experiment: **CONTENTION-SWEEP**

One knob, content and budget held exactly constant, gradient-free, no new mechanism required.

Fix the 16 documents, the order, and the per-document budget at `TOP_T = 32`. Vary **only the overlap
between documents' slot sets**, per layer:

| arm | slot assignment | writes/slot | survival | status |
|---|---|---|---|---|
| **D (disjoint)** | partition the 511 writable slots into 16 blocks of 32; document *i* writes block *i* | 1.00 | 100% | **never run** |
| **P (partial)** | the incumbent per-document tf ranking | 9.83 | 4.7% | = canonical, MT 2.4352 @ k=12 |
| **S (shared)** | one **document-independent** top-32 (across-document mean score), used by all 16 | 16.00 | 6.25% | **never run** — and already 87.4% identical to P |

**Predictions.**
*Non-overwriting:* MT improves monotonically **D < P < S** in loss terms reversed — i.e. D best, S worst,
with the D−P gap on the order of the survival deficit.
*Cumulative shared adaptation:* **S ≤ P ≪ D**, i.e. S best or tied with P, and D dramatically worse — by
roughly the DIAG-PERDOC margin (+1.101 per document on own questions) since D is 16 independent solo writes
that happen to share a cartridge.

**Why this is the right experiment and not a re-run.** DIAG-PERDOC's solo arm confounded *contention* with
*number of documents* (k=1 vs k=16); arm D fixes that — 16 documents, zero contention. ORACLE-WRITE-512
confounded it with *support* (full 512-slot support per document made the 16 writes mutually annihilate);
arm D holds support at 32. And arm S has never been run at all, despite the board's own measurement that
the incumbent is **already 87.4% a shared write** — which is itself weak evidence for the alternative, and
makes S nearly free to produce (`ranking.py`: replace the per-document prior with its across-document mean;
`_rank_slot_prior_per_layer` needs one branch).

**Cost.** Three writes + six evals (MT and QA at each arm's k-curve minimum), ~1 GPU-day at the canonical
config. Arm P is already on disk, so really two writes.

**Falsifier for the alternative principle.** If **D beats P** on MT by more than the ±0.049 paired
resolution, cumulative-shared-adaptation is refuted and the non-overwriting premise is vindicated — every
argument in this note fails at once. That is a clean, cheap, single-number kill test, and it is the one
number that would change my verdict.

**Secondary (metric) experiment, if there is budget for two.** Add `PMI_DC` scoring to
`eval_forgetting.py` (L05-8) and re-read the existing DIAG-CONTENT arms. Zero new GPU writes — arms A, B, C
are all on disk. Predicted outcome: the calibrated best point moves from k=12 to k=16, and the "acquisition
is 0.248 short" headline changes value, because 28–73% of what it is measuring is a prior we never
subtracted.

---

## Reading list (what was actually read)

| id | paper | venue | what I read |
|---|---|---|---|
| L05-1 | Lee, Sarao Mannelli, Clopath, Goldt, Saxe — *Maslow's Hammer for Catastrophic Forgetting* | arXiv:2205.09029, ICML 2022 | full text §1–5, esp. §4.1 EWC, §4.2 replay, catastrophic slowing |
| L05-2 | Lee, Goldt, Saxe — *Continual Learning in the Teacher-Student Setup* | arXiv:2107.04384, ICML 2021 | full text; §2 transfer dynamics, §3 feature/readout split |
| L05-3 | Hiratani — *Disentangling and mitigating the impact of task similarity for CL* | arXiv:2405.20236, NeurIPS 2024 | full text §5 gating, §6 weight regularisation incl. Fisher-metric condition |
| L05-4 | Biderman et al. — *LoRA Learns Less and Forgets Less* | arXiv:2405.09673, TMLR 08/2024 | §4.1–4.3, learning–forgetting Pareto |
| L05-5 | Min et al. — *Rethinking the Role of Demonstrations* | arXiv:2202.12837, EMNLP 2022 | §4 random labels, §5.1–5.3 all ablations with numbers |
| L05-6 | Hendel, Geva, Globerson — *In-Context Learning Creates Task Vectors* | arXiv:2310.15916, Findings EMNLP 2023 | §3.1–3.4, §5 conflicting-tasks |
| L05-7 | Zhao, Wallace, Feng, Klein, Singh — *Calibrate Before Use* | arXiv:2102.09690, ICML 2021 | §4 the three biases, §5 contextual calibration |
| L05-8 | Holtzman, West, Shwartz, Choi, Zettlemoyer — *Surface Form Competition* | arXiv:2104.08315, EMNLP 2021 | §3.2–3.3, PMI_DC definition and baselines |
| L05-9 | Gururangan et al. — *Don't Stop Pretraining* | arXiv:2004.10964, ACL 2020 | §3.2–3.3 incl. the ¬DAPT control, §4 TAPT |
| L05-10 | Yadav et al. — *What Matters for Model Merging at Scale?* | arXiv:2410.03617 | §4.3–4.5, Figs. 2, 5, 6 |
| L05-11 | Cabannes, Dohmatob, Bietti — *Scaling Laws for Associative Memories* | arXiv:2310.02984, ICLR 2024 | §3.1–3.2, Thm 1–2, overflow regime, thresholding scheme |
| L05-12 | Bell, Lawrence — *The Effect of Task Ordering in Continual Learning* | arXiv:2205.13323 | §3.3–3.5, curvature distance `c(j,k)`, min/max-sum paths |
| L05-13 | Wang, Yang, Shen, Huang — *A Comprehensive Survey of Forgetting in DL Beyond CL* | arXiv:2307.09218, TPAMI 2024 | §1.2 and §9 (beneficial forgetting) |
| L05-14 | Dohare, Hernandez-Garcia, Lan, Rahman, Mahmood, Sutton — *Loss of plasticity in deep continual learning* | arXiv:2306.13812; Nature 632:768–774 (2024) | §1, continual-backprop / contribution-utility sections |

## Caveats on this review

- **The strongest counter-argument to my verdict** is that F3 (uncontested slots are worse) has an
  alternative reading that does not need any of this literature: the solve is *underdetermined* differently
  in the disjoint case, so arm D of CONTENTION-SWEEP must log `mass_on_S`, solve MSE and determinacy per
  arm (MECH-BUDGET-B's protocol) or it will be uninterpretable. DIAG-PERDOC already partly closes this —
  every solo write landed `ref_mass_on_S` 0.1477–0.1535 against the 16-doc 0.15851 — but at 16 disjoint
  blocks the geometry is different again.
- **I did not find** a paper that studies overwriting in a *fixed-slot, gradient-free KV memory* with
  frozen keys. L05-3 and L05-11 are the closest formal analogues; both required me to map their variables
  onto ours (α ↔ q/`TOP_T`, N_x ↔ 511 writable slots, ρ_a ↔ routing cosine), and those mappings are
  arguments, not theorems.
- **Threads 01–04** cover compressed-cache CL, sparse finetuning, gating mechanisms and interference
  measurement. Where they recommend a protection mechanism, this note is the dissent: in *this* budget
  (retention +0.573, acquisition −0.248) protection is not a neutral option to be tuned, it is a cost to be
  avoided, and the papers above say so with mechanisms and with numbers.
