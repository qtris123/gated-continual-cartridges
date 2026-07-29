# LITREV-02 — Sparse finetuning & continual update: parameter/slot isolation

**Worker:** W2 SCOUT (literature, no GPU) · **Date:** 2026-07-29 · **Papers read:** 19
**Grounding:** `../2026-07-29-am-investigation-synthesis.md`, `../../research_loop/GLOSSARY.md`,
local `TF-IDF.pdf` (2510.15103, the direct ancestor) and `AM.pdf` (2602.16284, the base method).
**Baseline for every prediction:** QA **1.9468** / MT **2.2681** (seed-mean, k=12 snapshot reads
QA 1.9560 / MT 2.2720). Paired resolution **±0.049 MT / ±0.054 QA**. Budget QA ≤ 2.52, MT ≤ 2.02.

---

## Executive summary

1. **The honest headline: this entire family is bounded here, and the bound is not a hyperparameter.**
   Every method below selects the write support by an *old-task-importance* signal. In this system that
   signal is measured anti-aligned with acquisition (MT-wanted ∩ QA-safe is **11× below chance**), and
   acquisition is the only axis short of budget. The literature's lever buys the axis we are already
   **0.573 ahead** on.
2. **I fit the project's own law and it is tighter than the write-up claims.** OLS over the four
   MECH-CONSTRAINED arms gives **MT = 2.0613 − 0.15747·ln(realised MT routing mass)**, max residual
   **0.015** — 3.3× inside the ±0.049 resolution. It also predicts *out of family*: the pure-`fisher`
   arm (measured MT 2.665) is predicted at **2.626**, residual 0.039 < 0.049. The law is not an
   artifact of the constraint sweep.
3. **Every method in this review is a rule for shrinking `mass_on_S`.** PackNet/HAT/WSN/SupSup/Piggyback
   shrink it by disjointness; GPM/OWM/OGD/Adam-NSCL/O-LoRA/InfLoRA shrink it by orthogonality;
   TF-IDF shrinks it by background-frequency. So each one lands somewhere on that line, and the line's
   maximum is `q = 1.0` = the incumbent. **No paper in this review can beat the incumbent without
   changing what the line is made of.**
4. **The ancestor's premise breaks on three separate assumptions,** not one. (a) SMF's memory is
   *hard top-k routed* over 1M slots (3.2e-5 of the pool per token); ours is a *dense softmax* over 512
   slots — every slot is read by every query, so "unselected ⇒ untouched" is false. (b) SMF's `t` masks
   a **gradient**, so `t` is a learning-rate-like knob on the task loss; our `top_t` is the **support of
   a closed-form projection**, so it is a rank constraint, not a step size. (c) SMF's slots are
   *input-specific* (core set 100–500 of ~1e5 accessed); ours are **document-independent** — mean
   Spearman between documents' slot scores **0.958**, 87.4% overlap with a document-free top-32.
5. **`USE_IDF` is not a separate mechanism — it is `constrained_mass` with the constraint inverted,**
   and its measured damage is exactly what the law predicts. Our IDF is `log((|B|+s)/(df+s))` over a
   top-**128**-per-batch document frequency (`sparse_cache_finetuning.py:397–427`,
   `IDF_TOP_K=128`). Because the per-batch rankings are ~identical (ρ = 0.958), `df` is bimodal: ≈|B|
   for ~128 slots, ≈0 for the other ~383. So IDF ≈ 0 on exactly the highest-mass slots and ≈ ln(|B|+1)
   elsewhere — it does not *down-weight* the incumbent's picks, it **deletes** them. See L02-1.
6. **Parameter isolation is already refuted here by direct measurement, not by argument.** A solo
   write into **uncontested** slots is **+1.101 worse** on its own document than writing it among 16
   (allocation family, closed). Disjointness is the objective of PackNet/HAT/WSN/SupSup — this system
   measures interference as *helpful*. Note the arithmetic: 16 documents × 32 slots = **512** = the
   whole cartridge, so PackNet's capacity-exhaustion point is *exactly* our K=16.
7. **The one paper that reports our anti-alignment and does something about it is TRGP** (ICLR 2022):
   "naive orthogonal projection could possibly compromise the learning performance of the new task that
   is strongly correlated with old tasks." Its fix is to **relax the constraint and reuse** old
   subspaces. Our constraint knob is `AM_SAFE_FRACTION`, and TRGP's direction is `q → 1.0` — which is
   the incumbent and the **measured optimum** of the dose-response. TRGP's advice, applied here,
   returns the incumbent.
8. **The second paper that predicts our situation quantitatively is Hiratani (2405.20236).** Transfer
   Δε_TF = ρ_a(2ρ_b − ρ_a): "high feature similarity coupled with low readout similarity is
   catastrophic for both transfer and retention." Our ρ_a ≈ 1 (DIAG-KEYSPACE: QA/MT query separation is
   **2.5× smaller than sampling noise**, 0/288 heads separable) and ρ_b is low (writing a paper in makes
   the model *worse* at that paper). That is the theoretical worst cell of the 2×2, and it is the cell
   in which gating provably cannot help — because gating needs ρ_a < 1 to have anything to gate on.
9. **The field's answer, when the constraint binds, is unanimously "grow capacity"** — PackNet packs 4
   tasks then stops; GPM reports **78%** of the gradient space already constrained on 5-Datasets and
   states "after saturation, no new learning will be possible"; OWM's capacity *is* rank(P), monotone
   decreasing; WSN spends ~48.65% of the net on 40 tasks; SupSup needs a *new mask per task*; O-LoRA/
   InfLoRA add a branch per task. **This is out of scope (fixed 512 slots) and it is the honest
   finding.** Superposition theory (Cheung, ε ∝ (K−1)/M) says the same thing with a formula.
10. **What is genuinely untried and *not* bounded by the r ≈ −0.88 law:** nothing in the *selection*
    axis. The law's two exits are (a) full support — measured, 16 documents mutually annihilate — and
    (b) more slots — out of scope. Accordingly my three proposed experiments are designed to **test the
    law out of sample** and to tell the human whether opening the capacity scope would pay, not to beat
    the incumbent. If any of the three falsifies the law, the whole family reopens.

---

## A quantitative tool used throughout

Fit over the four MECH-CONSTRAINED arms (`q` = 1.00/0.75/0.50/0.25 → realised MT routing mass
0.2799/0.1661/0.1327/0.1000, MT 2.2720/2.3288/2.3693/2.4388):

```
MT_loss  ≈  2.0613  −  0.15747 · ln(mass_on_S)          r = −0.9775
residuals: +0.010, −0.015, −0.010, +0.015   (all ≤ 0.015, vs ±0.049 resolution)
inverse:   mass  ≈  exp( (2.0613 − MT_loss) / 0.15747 )
```

Two calibration facts that make it usable as a *predictor* from a doc-0 offline score, with **no GPU**:

* doc-0 projected bandwidth → realised `mass_on_S` ratio is **0.749 / 0.598 / 0.567 / 0.585 / 0.612**
  across five arms (mean ≈ 0.62, incumbent 0.749).
* incumbent top-32 projects **37.4%** of writable MT routing mass; a uniformly random top-32 projects
  **32/511 = 6.26%** by construction.

Every "Prediction about OUR signal" below is this formula applied to the paper's own selection rule.
Where a method cannot be scored offline I say so and mark it `unusable` rather than guess.

---

## Papers

### L02-1: sparse gradient mask on a top-k-routed memory layer — Continual Learning via Sparse Memory Finetuning (arXiv [2510.15103](https://arxiv.org/abs/2510.15103), Lin, Zettlemoyer, Ghosh, Yih, Markosyan, Berges, Oğuz; FAIR, Oct 2025) — *local `TF-IDF.pdf`, read in full*

- **Claim / mechanism.** Replace one FFN (layer 12 of 22, 1.3B base) with a memory layer (Berges 2024):
  keys `K ∈ R^{N×d}`, values `V ∈ R^{N×d}`, `N = 1M`, `k = 32` retrieved per token, 4 memory heads,
  value dim 1024. Per batch, count memory accesses `c(i)`; rank by
  `TF-IDF(i) = c(i)/Σ_j c(j) · log( (|B|+1) / (Σ_{b∈B} 1[c_b(i)>0] + 1) )` with `B` = 1000 DCLM
  background batches; make **only the top-`t`** values trainable via a detach trick
  (`mem = mem*mask + mem.detach() − (mem*mask).detach()`), then take **SGD steps** (10,000 of them).
  `t = 500` for facts, `t = 10,000` for documents. Result: NaturalQuestions F1 drops 89% (full FT) /
  71% (LoRA) / **11%** (SMF) at matched acquisition.
- **Selection signal.** *Both, factorised.* TF is pure **new-task utility** (this batch's accesses);
  IDF is pure **old-task importance** (background-corpus document frequency). Their Fig. 6 ablation is
  the load-bearing one: at `t = 500` TF-only "can retain target performance… but we observe more
  forgetting"; the TF-IDF-vs-TF gap **widens as `t` shrinks** (`t = 50`). Their Fig. 7 shows the
  background corpus matters: IDF from the *learning set* (TriviaQA) → similar learning, **more**
  forgetting.
- **Maps to our code.** This *is* our incumbent: `cartridges/sparse_cache_finetuning.py::CacheTFIDFRanker`
  (`_rank_per_layer`, `tfidf = tf * idf`, `IDF = log((|B|+s)/(df+s))`, `background_top_k_per_batch=128`)
  reached via `ranking.py::rank_am_slots` when `SLOT_SELECTION=tfidf`. Flags already exist:
  `USE_IDF`, `IDF_TOP_K`, `IDF_SMOOTHING`, `BG_STATS_PATH`, `TOP_T`.
- **Which assumptions break here — three, independently.**
  1. **Routing.** SMF's memory is *hard top-k*: each token touches 32 of 1e6 slots (3.2e-5), so a
     non-selected slot is genuinely not read on most inputs. Our cartridge is **dense softmax
     attention over 512 slots** — every slot is read by every query on every forward pass. "Sparse
     update ⇒ isolated update" is false by architecture. Direct evidence: mean pairwise Jaccard
     **0.713**, **9.83 writes/slot**, survival **4.7%**, and the untouched cartridge's own selectivity
     is only **1.0812**.
  2. **What `t` is.** In SMF, `t` masks a **gradient**, so larger `t` monotonically buys learning and
     costs forgetting (their Fig. 5 frontier is monotone in trainable-parameter count). In ours,
     `top_t` is the **support of a closed-form least-squares projection** (`value_solve.py:19`,
     `sparse_am_value_update`) — a rank constraint. Measured consequence: `t128` vs `t64` is *inside*
     resolution and **not sign-consistent** (MECH-BUDGET-B), and `TOP_T=64` at `q=0.5` is worse on
     **both** axes than `TOP_T=32`. `t` is not our learning knob.
  3. **Slot specificity.** SMF's "core set" is 100–500 indices out of ~1e5 accessed per batch, and
     Table 1 shows core-set indices "align with entity boundaries" — i.e. slots are *content-specific*.
     Ours are not: mean Spearman between different documents' slot scores **0.958**, and each
     document's top-32 overlaps a **document-independent** top-32 by **87.4%**. There is no core set
     to find. This is why "the gate is barely gating."
- **Prediction about OUR signal (the IDF forensic).** Because `df` is computed as "was this slot in the
  top-**128** of this background batch" and the per-batch rankings are ρ ≈ 0.958 identical, `df` is
  bimodal: ≈|B| for ~128 slots, ≈0 for the remaining ~383. So `idf ≈ 0` on precisely the high-mass
  slots and `≈ ln(|B|+1)` elsewhere, and `topk(tf·idf)` = **top-32 by `tf` among slots that are NOT in
  the background top-128** — algebraically a `constrained_mass` with the constraint set to the
  *complement* of the mass. Measured cost was **MT +0.465** → MT ≈ 2.733; the law then requires
  realised `mass_on_S` ≈ **0.0140** (band 0.0103–0.0192, i.e. **5–7% of the incumbent's 0.2799**).
  Checkable offline from any archived `USE_IDF=1` run's `ranking_info` / `mass_on_S_mean`.
  Selectivity: unchanged, **1.04–1.05**. Survival: unchanged (~4.7%; IDF changes *which* 32, not how
  many).
- **Cost.** Ranking is gradient-free either way. SMF itself is **not** gradient-free (10k SGD steps);
  only its *selection rule* is what we inherited.
- **Why it might NOT transfer.** Already covered — but the sharpest statement is that SMF's own
  ablation says IDF matters *more* as `t` shrinks, i.e. IDF is a device for protecting a *shared*
  memory when the update budget is tiny relative to the pool (`t/N = 5e-4`). Our `t/N = 32/511 =
  6.26%`, **125× denser**, and our pool is not shared with pretraining — it is a 512-slot cartridge
  that only ever serves one context. The regime IDF was designed for does not exist here.
- **Verdict.** **Incumbent (TF branch) / unusable (IDF branch).** The TF half is the measured optimum
  of the family. The IDF half is refuted (QA +0.383, MT +0.465) and I now have a mechanism for *why*
  that is checkable without a GPU.

---

### L02-2: iterative magnitude pruning + freeze — PackNet (arXiv [1711.05769](https://arxiv.org/abs/1711.05769), Mallya & Lazebnik, CVPR 2018)

- **Claim / mechanism.** After training task *t*, prune the lowest-magnitude 50–75% of weights per
  layer, **freeze** the surviving weights as task-*t*'s private set, retrain the network briefly to
  recover, and give the pruned (freed) weights to task *t+1*. Inference uses a per-task binary mask.
  Packs 4 tasks into VGG-16 at ~17–34 MB overhead on a 537 MB base.
- **Selection signal.** **Old-task importance** (weight magnitude as an importance proxy), applied
  *after* the task is learned. New-task utility never enters the allocation.
- **Maps to our code.** A disjoint-support scheduler in `ranking.py`: maintain a cumulative
  `used_mask[l, j]`, score `tf` only on `~used_mask`, take top-`t`, mark used. Proposed flag
  `SLOT_SELECTION=packnet` + `AM_DISJOINT_SUPPORT=1`. The "retrain to recover" half has **no analogue**
  — `gradient_steps = 0`.
- **Prediction about OUR signal.** Exact and unflattering: **16 documents × 32 slots/layer = 512 =
  the entire cartridge**, so PackNet's capacity-exhaustion point is *precisely* our K=16 and there is
  zero slack for a 17th. Each document then writes into 32 **uncontested** slots — which is exactly the
  arm the allocation family already ran: **+1.101 worse** on its own document's questions, with 71–93%
  of a solo write's gain landing on *other* documents' questions. Per-document survival → **100%**
  (from 4.7%), Jaccard → **0** (from 0.713), writes/slot → **1.0** (from 9.83). Per-document realised
  mass → ~0.042 (32/511 projected × 0.62) → the law's per-document value **≈ 2.56**; aggregate MT
  I predict **≥ 2.9**, direction certain, magnitude uncertain because the law was fit inside the
  overlapping regime and does not extrapolate to a disjoint union.
- **Cost.** Selection is gradient-free (magnitude only). The retraining phase is not, and is
  unavailable to us.
- **Why it might NOT transfer.** PackNet's premise is that a frozen weight *is not used* by the next
  task's forward pass in a harmful way. Ours are: every slot is read by every query. And PackNet
  assumes over-parameterisation ("redundancies in large deep networks"); at 32 of 511 with 16
  documents we are at exactly 1.0× capacity, not 10×.
- **Verdict.** **Unusable** — its objective (disjointness) is the arm this project measured at
  **+1.101**, i.e. 22× the resolution in the wrong direction.

---

### L02-3: learned per-task hard attention over units — HAT (arXiv [1801.01423](https://arxiv.org/abs/1801.01423), Serrà, Surís, Miron, Karatzoglou, ICML 2018)

- **Claim / mechanism.** Per-task embedding `e^t_l` gated to `a^t_l = σ(s·e^t_l)`, applied as
  `h'_l = a^t_l ⊙ h_l`. Cumulative mask `a^{≤t}_l = max(a^t_l, a^{≤t−1}_l)`; the gradient on weight
  (l,i,j) is scaled by `1 − min(a^{≤t}_{l,i}, a^{≤t−1}_{l−1,j})`. A compressibility regulariser
  `R = Σ a^t_{l,i}(1−a^{<t}_{l,i}) / Σ (1−a^{<t}_{l,i})` with coefficient `c` controls how much
  capacity each task claims (1–21% of the net).
- **Selection signal.** **Old-task importance only** — the `max` accumulation records prior
  allocation, and new-task utility does not un-protect anything.
- **Maps to our code.** A soft version of the disjoint scheduler: a per-slot cumulative
  `a[l,j] ∈ [0,1]` multiplied into `select_score` in `_rank_slot_prior_per_layer`. Proposed
  `SLOT_SELECTION=hat_cumulative` + `AM_HAT_C` (the compressibility coefficient). Note `c` is
  *literally* our `AM_SAFE_FRACTION` under a different parameterisation.
- **Prediction about OUR signal.** `c` and `q` traverse the same axis, so HAT lands on the same
  dose-response curve I fit: `c = 0` (no protection) → the incumbent MT 2.2720; any `c > 0` moves
  monotonically down the mass axis. Using the constrained_mass mapping, an HAT-style cumulative mask
  after 12 documents would have consumed ~12×32 = 384 of 511 slots, leaving 127 candidates ≡ `q ≈
  0.25` → predicted MT **2.42–2.44** for the last documents, i.e. **+0.15 to +0.17**. Selectivity
  unchanged at **1.04–1.05**. QA would improve by ≤ 0.05, i.e. **inside the ±0.054 resolution** — as
  every safety arm to date has been.
- **Cost.** HAT's masks are *learned by gradient*. A gradient-free port must replace the learned
  embedding with a heuristic, which throws away the paper's actual contribution.
- **Why it might NOT transfer.** HAT gates **units** in an FFN, where a zeroed unit truly contributes
  nothing. A "gated-off" cartridge slot still contributes `α_j v_j` to every attention output. The
  softmax denominator makes hard gating impossible without touching keys, and β (the only key-free
  mass knob) is **query-independent**, so it provably cannot make the gate task-dependent: β raised
  `mass_on_S` **4.23×** and moved selectivity only 1.043 → 1.051.
- **Verdict.** **Unusable** — algebraically the same knob as `AM_SAFE_FRACTION`, whose optimum is
  already measured to be "no constraint."

---

### L02-4: learned binary masks on a fully frozen backbone — Piggyback (arXiv [1801.06519](https://arxiv.org/abs/1801.06519), Mallya, Davis, Lazebnik, ECCV 2018)

- **Claim / mechanism.** Freeze the backbone entirely; learn a real-valued mask `m^r` per weight,
  threshold it to {0,1} in the forward pass, backprop through the threshold as a noisy estimator.
  1 bit/weight ≈ **3.12%** overhead per task; 10 tasks → 1.28× the backbone. Learned sparsity ranges
  from 4.51% (Flowers) to 37.59% (ImageNet).
- **Selection signal.** **New-task utility only** — masks are learned from the new task's loss, and the
  paper explicitly notes "our performance is agnostic to task ordering" and "the addition of a task
  does not affect performance on any other task," contrasting itself with PackNet. **This is the one
  masking paper whose signal points the right way for us.**
- **Maps to our code.** There is no gradient-free way to *learn* a mask, but there is a closed-form
  analogue: choose `S` to minimise the AM residual directly — i.e. **OMP over slots**, which the base
  method already implements for keys (`key_select.py:184 select_keys_omp`, `AM.pdf` Alg. 1). A
  slot-support OMP is `SLOT_SELECTION=omp_support` in `ranking.py`, greedily adding the slot whose
  column most reduces `‖X_S w − m‖²`.
- **Prediction about OUR signal.** OMP-by-residual is a *strictly better* new-task-utility selector
  than top-`tf`, so it should land at **≥** the incumbent's realised mass — and here is the catch: the
  incumbent already maximises mass by construction, and OMP maximises *residual reduction*, which on a
  single-block mass-matching problem is dominated by the same high-mass columns. I predict realised
  mass **0.28–0.32** (0–15% above 0.2799) → MT **2.245–2.272**, i.e. a best case of **−0.027**,
  **inside the ±0.049 resolution**. Selectivity unchanged. This is the strongest prediction in the
  review that a *correctly-signed* selector still cannot clear the bar.
- **Cost.** Gradient-free (greedy least-squares); ~4–8× the incumbent's ranking cost per `AM.pdf` C.1.
- **Why it might NOT transfer.** Piggyback's whole leverage is that a rich pretrained backbone
  contains a good subnetwork for many tasks. Our "backbone" is a 512-slot Phase-1 cartridge whose
  *values* are QA-specific; masking cannot synthesise MT content that is not in the cartridge, and the
  write-ceiling oracle (teacher's own KV, the best any value-only write can do) reaches only
  **MT 2.381** — *worse than the incumbent*.
- **Verdict.** **Candidate, low prior** — the only correctly-signed selector in the masking family, and
  the only one worth the ~1 GPU-hour, but I predict it lands inside the resolution. Rank it below the
  three experiments at the end.

---

### L02-5: fixed random weights + per-task supermask + entropy-based inference — SupSup (arXiv [2006.14769](https://arxiv.org/abs/2006.14769), Wortsman et al., NeurIPS 2020)

- **Claim / mechanism.** Never touch the weights (fixed at ±c Kaiming init). Per task, find a binary
  supermask by *edge-popup* (learn scores `S`, take `M = h(S)` top-k%, straight-through gradient). At
  test time with unknown task identity, take a weighted superposition `Σ α_i M^i` and infer the task by
  **one gradient step on the output entropy**: `argmax_i (−∂H(p(α))/∂α_i)`. Scales to **2500**
  PermutedMNIST tasks.
- **Selection signal.** **New-task utility** (masks are optimised against the current task's CE) with
  *no* old-task protection at all — isolation is structural (one mask per task, weights never change).
- **Maps to our code.** Requires storing one mask per document and *selecting the mask at eval time*.
  Our eval path has no such hook, and `finetune.py`'s write path is one cache. Proposed flag would be
  `AM_PER_DOC_MASKS=1` + an eval-time router — a substantial architectural change, not a flag.
- **Prediction about OUR signal.** The mechanism that makes SupSup work is that the entropy signal can
  *discriminate* which mask matches the input. Ours cannot: the analogous discriminator is the
  MT-vs-QA query separation, measured at **2.5× smaller than sampling noise**, **0/288 heads**
  separable, and ρ_key sitting **at its held-out control floor**. So SupSup's task-inference step
  would be at chance, and a chance-level `α` is a uniform mixture — which is the *current* system.
  Predicted ΔMT **0.000 ± 0.049**, selectivity **1.04–1.05** (i.e. exactly unchanged), by construction.
- **Cost.** Edge-popup is gradient-based; entropy inference is a gradient step at *eval*, which is
  worse than a diagnostic backward — it is an inference-time optimisation.
- **Why it might NOT transfer.** Superposition of masks requires near-orthogonal task signatures.
  Ours are Spearman **0.958** correlated. This is Cheung's (K−1)/M in disguise (see L02-17).
- **Verdict.** **Unusable** — its enabling condition (task-discriminable inputs) is measured absent,
  and the failure mode is provably a no-op.

---

### L02-6: score-learned top-c% subnetwork with reuse — WSN (Kang et al., [ICML 2022, PMLR 162:10734](https://proceedings.mlr.press/v162/kang22b/kang22b.pdf); extension SoftNet arXiv [2303.14962](https://arxiv.org/abs/2303.14962))

- **Claim / mechanism.** Learn a per-weight score `s`; take the top-`c%` as the task-`t` mask `m_t`;
  accumulate `M_{t−1} = ∨_{i<t} m_i` and mask the update `θ ← θ − η(∂L/∂θ ⊙ (1 − M_{t−1}))`, so
  previously-selected weights may be **reused in the forward pass but never updated**. `c ∈
  {3,5,10,30,50,70}%`; TinyImageNet with `c=10%` reaches ~**48.65%** capacity after **40 tasks**.
  SoftNet softens `m ∈ [0,1]` (top-c% = 1, rest ~U(0,1)) because hard masks overfit in few-shot.
- **Selection signal.** **New-task utility** for *which* weights (score is trained on the current
  loss), **old-task protection** for *which are writable*. This is the closest published analogue of
  our `constrained_mass`: a hard eligibility constraint plus a new-task ranking inside it.
- **Maps to our code.** `SLOT_SELECTION=constrained_mass` with `AM_SAFE_METRIC` replaced by a
  cumulative-use mask; i.e. `AM_SAFE_METRIC=used_before` + `AM_SAFE_FRACTION` derived from
  `1 − (k·top_t)/511`.
- **Prediction about OUR signal.** WSN's reuse-but-don't-update rule is a *time-varying* `q`:
  after `k` documents, `n_safe = 511 − 32k`, so `q(k) = 1 − 0.0626k`. That crosses the measured
  dose-response at `q=0.75` around **k = 4** and `q=0.25` around **k = 12**. Interpolating the fit
  arm-by-arm and integrating over the 16-document schedule, I predict best-MT ≈ **2.36 ± 0.05**
  (**+0.09**, clearing resolution) with QA improving ≤ 0.05 (inside resolution) — i.e. the same shape
  as every safety arm. Survival → 100%, Jaccard → 0, writes/slot → 1.0.
- **Cost.** Score learning is gradient-based. A gradient-free port degenerates to PackNet (L02-2).
- **Why it might NOT transfer.** WSN's `c=10%` on 40 tasks works because the network is 10×
  over-provisioned. Our `top_t/n = 6.26%` × 16 documents = **100.2%** — we are at the exhaustion point
  on the first pass, which is the regime WSN explicitly does not operate in.
- **Verdict.** **Unusable** — it is `constrained_mass` with `q` annealed downward over documents, and
  the measured `q`-curve is monotone in the wrong direction.

---

### L02-7: SVD of activations → project gradients orthogonal — GPM (arXiv [2103.09762](https://arxiv.org/abs/2103.09762), Saha, Garg, Roy, ICLR 2021)

- **Claim / mechanism.** After task *t*, build a representation matrix `R` from activations, SVD it,
  keep the top-*k* left singular vectors satisfying `‖R_k‖²_F ≥ ε_th‖R‖²_F` as the Gradient Projection
  Memory `M`. Future gradients are projected: `∇W ← ∇W − (∇W)MMᵀ`. `ε_th` "mediates the
  stability–plasticity dilemma."
- **Selection signal.** **Old-task importance** — the principal directions of old activations.
- **Maps to our code.** The AM analogue is exact and already partially present: constrain the solved
  values so the *QA* reference queries' outputs are unchanged, i.e. solve
  `min ‖X_MT V_S − Y_MT‖²` s.t. `X_QA V_S = X_QA V_S^old`. Implementable as a projector inside
  `value_solve.py::sparse_am_value_update`, flag `AM_NULLSPACE_QA=1` (+ `AM_NULLSPACE_EPS`). Note the
  *soft* version already exists: `guarded_sparse_am_value_update` (`value_solve.py:284`) appends the
  QA reference queries to the design matrix with weight `√OLD_REFERENCE_WEIGHT`.
- **Prediction about OUR signal.** Sharp, and it is the most important negative in this review:
  `ENABLE_OLD_REFERENCE_GUARD` defaults to **0**, and `continual_am_sparse.py:636` passes
  `old_reference_weight = OLD_REFERENCE_WEIGHT if ENABLE_OLD_REFERENCE_GUARD else 0.0`. **The
  incumbent already runs at λ = 0** — zero old-task preservation in the solve. So GPM's knob is
  already at the plasticity extreme and can only be turned the *wrong* way. Turning it on can only
  reduce the MT fit; since MT loss is *not* monotone in fit quality (18× lower MSE → +13 loss), I
  cannot even promise the QA gain, but the MT direction is certain: **ΔMT > 0**. Because the QA and MT
  query spaces are not separable (ρ_key at its control floor), the QA null space contains almost no MT
  energy, so a *hard* null-space constraint would leave near-zero writable direction: predicted
  realised mass collapse and MT → **> 2.6**.
- **Cost.** Gradient-free — it is a linear constraint on a least-squares solve. **This is the one
  family whose port is genuinely free of gradients.**
- **Why it might NOT transfer.** GPM's premise is that old and new tasks occupy *different* subspaces
  and the null space is non-empty. DIAG-KEYSPACE measured that premise false at the query level
  (0/288 heads), and GPM itself reports **78%** of the gradient space already constrained after 5
  dissimilar tasks with the caveat "after saturation, no new learning will be possible."
- **Verdict.** **Unusable as a win, valuable as a diagnosis.** It formalises why the QA slack cannot be
  converted into MT: the exchange rate is set by ρ_key, which sits at its control floor.

---

### L02-8: Adam updates projected into the null space of feature covariance — Adam-NSCL (arXiv [2103.07113](https://arxiv.org/abs/2103.07113), Wang et al., CVPR 2021)

- **Claim / mechanism.** Maintain the uncentred feature covariance `X̄ᵀX̄` of all previous tasks per
  layer; SVD it; take the singular vectors `U₂` with the *smallest* singular values
  (`λ ≤ a·λ_min`) as an **approximate** null space; project the Adam update `Δw = U₂U₂ᵀ g`. Accuracy
  73.77% / 75.95% / 58.28% on 10/20-split CIFAR-100 and 25-split TinyImageNet; BWT −1.6% / −3.66% /
  −6.05%.
- **Selection signal.** **Old-task importance** (feature covariance), with `a` as the explicit
  plasticity dial: "larger `a` … larger approximate null space … reducing stability" while "increasing
  the plasticity"; accuracy is **non-monotone** in `a` (their Fig. 5).
- **Maps to our code.** Identical port to L02-7 but with the **relaxation** that makes it usable:
  instead of the exact QA null space, take the eigendirections of `Q_QA ᵀQ_QA` below `a·λ_min`.
  Flag `AM_NULLSPACE_QA=1 AM_NULLSPACE_A=<a>`.
- **Prediction about OUR signal.** The relaxation parameter `a` is the same axis as `AM_SAFE_FRACTION`
  and `OLD_REFERENCE_WEIGHT`. Adam-NSCL's own finding — accuracy rises then falls in `a` — is the
  *mirror* of our measured monotone `q` curve, and the reason for the difference is diagnostic: their
  optimum is interior because both axes are short of budget; **ours is at the boundary because one axis
  has 0.573 of slack**. Predicted: any `a` giving a non-trivial constraint costs **ΔMT ≥ +0.06**;
  `a → ∞` returns the incumbent. Selectivity unchanged.
- **Cost.** Gradient-free in our port (the SVD is of a query Gram matrix, ~free next to the 512×512
  inverse `compute_slot_redundancy` already does per layer).
- **Why it might NOT transfer.** Their "R < 0.05 of explained variance in `U₂`" justification says the
  null space is *nearly* empty even in vision. With QA/MT queries indistinguishable at 2.5× below
  noise, ours is emptier.
- **Verdict.** **Unusable** — same axis, boundary optimum, already swept as MECH-CONSTRAINED.

---

### L02-9: project new gradients orthogonal to stored old prediction-gradients — OGD (arXiv [1910.07104](https://arxiv.org/abs/1910.07104), Farajtabar, Azizan, Mott, Li, AISTATS 2020)

- **Claim / mechanism.** Store an orthonormal basis of `∇f(x;w)` (the **model**, not loss, gradients)
  from old-task samples (200/task in their MNIST runs); project each new-task loss gradient onto its
  orthogonal complement. Three variants — OGD-ALL / OGD-AVE / **OGD-GTL** (ground-truth logit only,
  which they find "slightly outperforms").
- **Selection signal.** **Old-task importance**, unweighted — every stored gradient is treated equally;
  no importance ranking at all.
- **Maps to our code.** Store `∂(QA output)/∂v[l,·,j,·]` for a sample of QA references — one
  diagnostic backward pass (the same 98.8 s already paid for `slot_fisher_qa_phase1.npz`) — then
  project the solved `V_sel_new` in `value_solve.py`. Flag `AM_OGD_BASIS=<path>`.
- **Prediction about OUR signal.** OGD's basis is the *span* of the same quantity our `fisher` selector
  takes the diagonal of; the `fisher` arm gave the best retention ever measured here (**QA 1.8641**,
  0.04% QA-Fisher exposure) at **MT +0.397**. Projecting rather than *excluding* is a strictly weaker
  constraint, so I predict it lands between the incumbent and `fisher`: **MT +0.10 to +0.35**,
  QA −0.02 to −0.08. The QA gain will not clear ±0.054; the MT loss will clear ±0.049.
- **Cost.** One diagnostic backward pass (`gradient_steps` stays 0), ~99 s, cacheable — genuinely
  affordable. The projection itself is free.
- **Why it might NOT transfer.** OGD explicitly relies on over-parameterisation — "there always exist a
  direction that conforms to the orthogonality condition" — and gives **no** analysis of saturation.
  With 32 writable slots per layer and a QA basis of rank ≫ 32, the orthogonal complement inside
  `span(S)` is likely **empty**, in which case the projection is the zero map and MT reverts to the
  Phase-1 floor **3.7825**. That is a concrete, testable failure mode.
- **Verdict.** **Unusable** — dominated by the already-run `fisher` arm on both the mechanism and the
  measured axis, with a plausible degenerate-to-floor mode.

---

### L02-10: recursive-least-squares projector on the input space — OWM (Zeng, Chen, Cui, Yu, *Nature Machine Intelligence* 1:364–372, 2019; arXiv [1810.01256](https://arxiv.org/abs/1810.01256))

- **Claim / mechanism.** `P = I − A(AᵀA + αI)⁻¹Aᵀ` from the previous tasks' **inputs** `A`, updated
  recursively (RLS/Kalman gain) so `A` is never stored; `ΔW = κ·P·ΔW^{BP}`. Learns 3,755 Chinese
  characters sequentially at ~92%, 1000 ImageNet classes, 40 face attributes. The paper is explicit
  that capacity **is** `rank(P)`, and `rank(P_{i+1}) = rank(P_i) − rank(ΔP_{i+1})` is monotone
  decreasing; performance degrades as it approaches zero. A separate **context-dependent processing**
  module rotates features by a context signal `C` so identical inputs give different outputs.
- **Selection signal.** **Old-task importance**, via the input second moment — the same object as
  DIAG-KEYSPACE's `Q₀`.
- **Maps to our code.** The projector is a rank-`r` version of the L02-7 port and `α` is literally
  `RIDGE_LAMBDA`. The interesting half is the **CDP module**, which maps to `key_select.py` — it is a
  *key-side* mechanism: it makes routing context-dependent rather than restricting the write.
- **Prediction about OUR signal.** The projector half: identical to L02-7/8, ΔMT > 0. The CDP half is
  the **only idea in this review that targets our actually-broken quantity** — selectivity **1.04–1.05**
  vs the untouched cartridge's **1.0812**. But DIAG-KEYSPACE bounds it: an *optimally placed* key buys
  ρ_key **1.22** against an incumbent **1.083**, so the entire achievable CDP gain is ~13% of a ratio
  that is already ~1, and MECH-005 (key installation + RoPE repositioning, the only confirmed win,
  **ΔMT −0.1545**) has already collected most of it. Predicted residual headroom from a CDP-style
  context rotation: **ΔMT −0.02 to −0.06**, straddling the resolution.
- **Cost.** Projector: gradient-free. CDP: needs a learned context encoder → not gradient-free.
- **Why it might NOT transfer.** OWM's context signal is *given* (an explicit task cue). We have no
  task cue at eval — the eval queries are just QA or MT questions, and they are measured
  indistinguishable.
- **Verdict.** **Unusable** (projector half) / **already harvested** (CDP half is MECH-005). Worth
  recording because it is the paper that states the capacity law most cleanly: capacity = rank(P).

---

### L02-11: relax the orthogonality constraint and *reuse* old subspaces — TRGP (arXiv [2202.02931](https://arxiv.org/abs/2202.02931), Lin, Yang, Fang, Gao, Kong, ICLR 2022)

- **Claim / mechanism.** **The paper that names our problem.** It criticises GPM directly: "by
  modifying the model only in the orthogonal direction to the input space of old tasks, the
  optimization space of learning the new task could be more restrictive, resulting in compromised
  performance of the new task," and conjectures that "naive orthogonal projection could possibly
  compromise the learning performance of the new task that is **strongly correlated** with old tasks."
  Fix: put old task *j* in a layer-wise **trust region** for the new task when
  `‖Proj_{S_j}(∇L_t)‖ ≥ ε‖∇L_t‖`, then let the new task **reuse** that frozen subspace through a
  learned scaling matrix `Q_{j,t}` rather than avoid it. +1.4 to +2.4 points over GPM, and *better*
  BWT too (−0.8% vs −3% on PMNIST).
- **Selection signal.** **Both, and explicitly:** old-task subspaces are the candidates, but the
  *trigger* is new-task gradient alignment. It is the field's only clean "when old and new want the
  same thing, share it" rule.
- **Maps to our code.** The trust-region trigger is measurable offline in `ranking.py`: our analogue of
  `‖Proj_{S_j}(∇L_t)‖ / ‖∇L_t‖` is exactly `mass_on_S` / total MT mass, already computed at
  `value_solve.py:146`. The "reuse" half maps to `AM_SAFE_FRACTION → 1.0` plus a per-slot rescale —
  which is `β` (`key_select.py:247 refit_beta_nnls`, `ENABLE_BETA`).
- **Prediction about OUR signal.** TRGP's prescription, applied literally, is **`q = 1.0`** — the
  incumbent, which is the measured argmax of the dose-response. Its second half (rescale rather than
  re-solve) is **β**, which was run: β raised `mass_on_S` **4.23×** and made **both axes worse**,
  moving selectivity only 1.043 → 1.051 because β is query-independent, whereas TRGP's `Q_{j,t}` is
  *task-conditioned*. Predicted ΔMT for a faithful port: **0.000 ± 0.049** (it is the incumbent) —
  or worse if β is enabled to implement the rescale.
- **Cost.** `Q_{j,t}` is learned by gradient. The trigger is free.
- **Why it might NOT transfer.** TRGP assumes there exist *some* uncorrelated old tasks worth avoiding
  and *some* correlated ones worth reusing. Ours is a single degenerate cell: **everything is
  correlated** (documents' slot scores ρ = 0.958, QA/MT queries inseparable). Its decision rule has no
  work to do.
- **Verdict.** **Unusable as a method, load-bearing as evidence.** It is the field's own statement of
  our anti-alignment, and its answer — relax the constraint — is the setting we already run.

---

### L02-12: per-task LoRA with an orthogonality penalty — O-LoRA (arXiv [2310.14152](https://arxiv.org/abs/2310.14152), Wang et al., EMNLP 2023 Findings)

- **Claim / mechanism.** Each task gets `{A_t, B_t}`; the columns of `A_t` span the task's update
  subspace; penalise `L_orth = Σ_{j,k} ‖(A_iᵀA_t)_{jk}‖²` against all frozen previous `A_i`, weight
  `λ₁ = 0.5`. Previous LoRAs frozen; all merged into `W_init` at the end. 75.8% avg (vs LFPT5 72.7%);
  69.6% on a 15-task sequence.
- **Selection signal.** **Old-task importance** (previous subspaces), with no new-task term.
  Parameters **grow** linearly with tasks during training (merged only afterwards).
- **Maps to our code.** Per-document orthogonality of the *written value deltas*: add
  `Σ_{i<k} ‖ΔV_iᵀ ΔV_k ‖²` to the solve — closed-form, since it is a quadratic penalty on a
  least-squares problem. Flag `AM_ORTH_DELTA=<λ>` in `value_solve.py`.
- **Prediction about OUR signal.** This is the direct test of "should documents avoid each other."
  The allocation family already answered: mutual overwriting is not the problem (survival 4.7% "costs
  almost nothing"), and forced non-overlap is **+1.101**. Orthogonality is a softer version of the same
  thing, so I predict a monotone penalty in `λ`: **ΔMT ≈ +0.05·ln(1+λ/λ₀)** with the incumbent at
  `λ=0`; concretely **λ = 0.5 → MT ≈ 2.33 ± 0.05** (+0.06, clearing resolution). Per-document survival
  would rise from 4.7% toward ~30–50% (deltas become mutually invisible rather than disjoint).
- **Cost.** Gradient-free in our port (quadratic penalty → normal equations).
- **Why it might NOT transfer.** O-LoRA's tasks are genuinely different NLP tasks with different
  readouts. Our 16 documents are 16 samples from **one** distribution — and the content-free control
  shows a document with **zero MT content** still reproduces 28.3–73.2% of the MT gain (ICL itself
  shows 53.3%). Forcing them orthogonal destroys the shared component that is doing most of the work.
- **Verdict.** **Unusable** — predicted to lose, for a reason this project has already measured twice
  (allocation family; content-free control).

---

### L02-13: pre-designed LoRA down-projection inside (new-gradient ∩ old-orthogonal) — InfLoRA (arXiv [2404.00228](https://arxiv.org/abs/2404.00228), Liang & Li, CVPR 2024)

- **Claim / mechanism.** **The best-designed method in this review, and the one our data refutes most
  directly.** Do not *learn* the subspace — *design* it. Constrain every row of the down-projection
  `B_t` to lie in `N_t ∩ M_t^⊥`, where `N_t` is the new task's gradient space (from its input
  activations) and `M_t^⊥` is the orthogonal complement of the old tasks' gradient space (via
  DualGPM). Computed as: project `H_t` to remove old components (`Ĥ_t = H_t − M_tM_tᵀH_t`), SVD, take
  the top-`r` singular vectors. **Fixed capacity** — branches are merged after each task, so overhead
  is constant `(d_I+d_O)·r` regardless of task count. Ablation: both constraints together beat either
  alone by ~2–5%.
- **Selection signal.** **Both, by intersection.** This is exactly our `mass_x_redundancy`
  (`ranking.py`, `u_tf^{1−α}·u_red^α`, "slots the new task WANTS *and* that are redundant") and exactly
  our `constrained_mass`.
- **Maps to our code.** `SLOT_SELECTION=mass_x_redundancy` (α as the mixing knob) and
  `SLOT_SELECTION=constrained_mass` (`AM_SAFE_FRACTION` as the hard-constraint knob). Both exist and
  both were run.
- **Prediction about OUR signal.** InfLoRA's enabling condition is `N_t ∩ M_t^⊥ ≠ ∅` with rank ≥ `r`.
  We measured that intersection: **MT-wanted ∩ QA-safe is 11× below chance**, and selecting for
  QA-safety at fixed budget buys 106× less exposure for **8.1× less bandwidth**. So `r` = 32 slots
  cannot be found inside the intersection — the SVD would return directions with near-zero MT energy.
  Quantitatively: bandwidth `37.4%/8.1 = 4.62%` projected → realised ≈ 0.028 → the law gives
  **MT ≈ 2.626**; the measured pure-`fisher` arm is **2.665** (residual 0.039, inside ±0.049). **The
  law predicts InfLoRA's port before it is run.**
- **Cost.** Gradient-free in our port (SVD of a query Gram + the existing 512×512 inverse).
- **Why it might NOT transfer.** InfLoRA works in vision-transformer CL where old and new tasks are
  *different classes* with separable features. It presupposes the intersection is non-degenerate. Our
  Spearman(`tf_mass`, `fisher`) = **0.579** looked like it left room — that is exactly why MECH-008 was
  built — but the room is on the *wrong* axis: low correlation between two rankings does not create an
  intersection when one ranking's top is where all the mass is.
- **Verdict.** **Unusable — already run, twice, and it lost by 2.2–8.1× the resolution.** Recorded so
  nobody re-imports it as a fresh idea.

---

### L02-14: the learning/forgetting frontier is one curve — LoRA Learns Less and Forgets Less (arXiv [2405.09673](https://arxiv.org/abs/2405.09673), Biderman et al., TMLR 2024)

- **Claim / mechanism.** Not a selection method — a **measurement**, and the one that legitimises this
  project's framing. Across code/math CPT and IFT on 7B/13B Llama-2, LoRA and full finetuning "seem to
  occupy the same Pareto curve, with the LoRA models on the lower right — learning less and forgetting
  less." Spectral analysis shows full FT learns **rank 10–100× higher** than typical LoRA ranks (~1500–
  2000 for 90% variance on a 4096² matrix), refuting the low-rank premise. HumanEval 0.175 (LoRA) vs
  0.263 (full FT); GSM8K 0.622 vs 0.642.
- **Selection signal.** N/A — it is the paper that says the *choice of signal* is second-order to
  **where on the frontier you sit**.
- **Maps to our code.** It is the frame for `research_loop/state/results.csv`: every arm should be
  plotted (QA, MT), not adjudicated per-axis.
- **Prediction about OUR signal.** Direct: our ten gating arms should lie on **one** monotone curve in
  (QA, MT), and the incumbent should be the MT-extreme endpoint of it. That is what the data shows —
  `fisher` at (1.8641, 2.665) and the incumbent at (1.9468, 2.2681) are two points on one line, and the
  `q`-sweep traces it. **Predicted correlation between ΔQA and ΔMT across the ten arms: r < −0.8.**
  This is worth computing from `results.csv` and putting in the synthesis — it converts "gating fails"
  into "gating moves us along a frontier that does not pass through the target."
- **Cost.** Free — it is a re-plot of existing numbers.
- **Why it might NOT transfer.** SMF's own headline (Fig. 5) is that sparse memory finetuning
  **Pareto-dominates** rather than slides along. That domination came from the *architecture* (sparse
  routing), not the selector — which is precisely the assumption we do not have.
- **Verdict.** **Candidate (as an analysis, not a mechanism).** Zero cost, and it is the cleanest way
  to state the project's negative result to a reader.

---

### L02-15: closed-form least-squares memory edit with an old-key covariance term — MEMIT (arXiv [2210.07229](https://arxiv.org/abs/2210.07229), Meng, Sharma, Andonian, Belinkov, Bau, ICLR 2023)

- **Claim / mechanism.** **The closest published relative of our *write*, as opposed to our
  *selection*.** Treat an MLP output matrix as a linear associative memory and solve in closed form:
  `Δ = R K₁ᵀ (C₀ + K₁K₁ᵀ)⁻¹` with `R = M₁ − W₀K₁` the residual, `K₁` the new keys, and
  `C₀ = λ·E[kkᵀ]` the covariance of **previously stored** keys (λ ≈ 1.5e4, estimated from WikiText).
  Layers chosen by causal tracing (GPT-J: layers 3–8); the residual is spread as
  `(z_i − h_i^L)/(L − l + 1)`. Scales to **10,000** simultaneous edits at 98.9% efficacy where ROME
  degrades from n=32.
- **Selection signal.** Layer choice is **causal-tracing** (old-task importance, but as *localisation*
  not protection). `C₀` is the old-task-preservation term, and λ is the dial: "specificity and fluency
  increase monotonically with λ."
- **Maps to our code.** Term for term:
  `C₀ ↔ guarded_sparse_am_value_update`'s appended QA reference block (`value_solve.py:360–363`,
  `scale = old_reference_weight**0.5`); `λ ↔ OLD_REFERENCE_WEIGHT`; the ridge `↔ RIDGE_LAMBDA` /
  `RIDGE_SCALE`; the proximal term `delta_weight·‖V_S − V_S^old‖² ↔ DELTA_WEIGHT`.
- **Prediction about OUR signal — and this is the review's most useful single check.** MEMIT's
  prescription for our situation (QA 0.573 ahead, MT 0.248 short) is **lower λ**: trade specificity for
  efficacy. But `ENABLE_OLD_REFERENCE_GUARD` defaults to **0** and
  `continual_am_sparse.py:636` forces `old_reference_weight = 0.0` when it is off — **λ is already 0**.
  And the proximal term cannot be lowered either: the write-ceiling oracle (a *perfect* write, no
  proximal term) reaches only **MT 2.381**, and "18× lower MSE → +13 loss". So **both** of MEMIT's
  dials are already pinned at the plasticity extreme, and one of them is measured to be *positively*
  load-bearing in the preservation direction. Predicted `ΔMT` from any MEMIT-faithful change:
  **≥ +0.05**, i.e. strictly worse.
- **Cost.** Gradient-free for `Δ` (normal equations); MEMIT's `z_i` targets need 25 gradient steps,
  but our targets come from the teacher, so our port is fully gradient-free.
- **Why it might NOT transfer.** MEMIT edits a **weight matrix** whose key space is
  `d_mlp`-dimensional and shared across all inputs, with a covariance estimated from 100k WikiText
  samples. Our "keys" are 512 cached vectors and the read is a *softmax convex combination*, so
  `Σ_j α_j = 1` is a hard budget — MEMIT's `Δ` has no such normalisation and can add capacity;
  ours can only reallocate it.
- **Verdict.** **Unusable as a change / decisive as a diagnosis.** It proves the write side is already
  at MEMIT's plasticity limit, which is why acquisition cannot be bought there.

---

### L02-16: a side memory with random-mask sharding, because editing the main memory cannot work — WISE (arXiv [2405.14768](https://arxiv.org/abs/2405.14768), Wang et al., NeurIPS 2024)

- **Claim / mechanism.** Argues an "impossible triangle": reliability + generalisation + locality
  cannot be had by editing long-term (main) parameters, nor by retrieval-only working memory. So: copy
  one FFN value matrix into a **side memory**, route between main and side at inference by an
  activation-difference threshold `ε`, and split the edit stream into `k` **shards**, each trained in a
  *random* gradient subspace (binary mask, ratio ρ), then **Ties-merge** the shards. 100–3000
  sequential edits; 0.83 avg at T=1000 on LLaMA-2-7B, +18% over competitors.
- **Selection signal.** **Neither — deliberately random.** They argue random masks are *preferable*
  because they "do not harm network performance while regularizing optimization in a subspace," and
  because near-orthogonality of random masks is what prevents shard conflict. This is the one paper
  that argues *against* importance-based selection.
- **Maps to our code.** Two pieces. (a) `SLOT_SELECTION=random` in `ranking.py` — ~8 lines, a seeded
  `torch.randperm` per layer, plus `AM_RANDOM_SEED`. (b) The side-memory/routing half has **no**
  in-scope port: it grows capacity, which is the out-of-scope direction.
- **Prediction about OUR signal — numeric, and this is my experiment #2.** Random top-32 projects
  `32/511 = 6.262%` of writable MT routing mass; applying the measured realised/projected ratio
  (0.60–0.75) gives realised `mass_on_S` **0.038–0.047**, and the law then gives
  **MT = 2.54–2.58** (point estimate **2.561**), i.e. **+0.29** — *worse than every importance-based
  selector ever run here*, including `fisher` (2.665 is worse still, consistent). QA should be the
  best-but-one ever measured (random selection has ~29.6%×(6.26/37.4) ≈ 5% QA-Fisher exposure vs the
  incumbent's 29.6% and `fisher`'s 0.04%), so predicted **QA ≈ 1.88–1.92**. Selectivity **1.04–1.05**.
  Survival **~4.7%** (random 32-of-511 across 16 documents gives expected Jaccard 0.032, so survival
  should *rise* to ~65% — a second, independent falsifier).
- **Cost.** Free, gradient-free, no data.
- **Why it might NOT transfer.** WISE's random masks are near-orthogonal *because* the space is
  high-dimensional (FFN-sized); at 32-of-511 they are near-orthogonal too, which is exactly why I
  predict it loses — it destroys the shared cumulative adaptation that the content-free control showed
  is doing 28–73% of the work.
- **Verdict.** **Candidate — as a control, not a win.** It is the cheapest out-of-sample test of the
  law at a mass ~6× below anything tried. See "Top 3 experiments," #2.

---

### L02-17: context vectors put K models in one weight matrix — Superposition of Many Models Into One / PSP (arXiv [1902.05522](https://arxiv.org/abs/1902.05522), Cheung, Terekhov, Chen, Agrawal, Olshausen, NeurIPS 2019)

- **Claim / mechanism.** Store `W = Σ_i W_i C_i^{-1}` with per-task context `C_i` (binary ±1, complex
  unit-phase, or full rotation), retrieved as `y_k = W(C_k ⊙ x)`. **Proposition 2** gives the capacity
  law: retrieval interference variance scales as `≈ (K−1)/M` with `M` the layer width, and destructive
  interference suppresses non-target models to `~1/√M` of signal. 50 PermutedMNIST tasks in a 256-unit
  net: **97.6%** vs 61.8% baseline.
- **Selection signal.** **Neither** — contexts are random and fixed; nothing is selected. Isolation is
  bought by *near-orthogonality of contexts*, not by disjointness of parameters.
- **Maps to our code.** The context `C_k` is a *query-side rotation*, so the port is key-side:
  `key_select.py` + `AM_KEY_REPOSITION` already does a RoPE rotation, and a per-document phase offset
  would be `AM_DOC_CONTEXT_PHASE`. Nothing in `ranking.py` implements this.
- **Prediction about OUR signal.** The law says interference ∝ (K−1)/M. Ours: K = 16, M = 512 →
  `(K−1)/M = 0.0293`, which is *small*, i.e. **superposition theory says 16 documents in 512 slots
  should be nearly free** — and indeed survival at 4.7% "costs almost nothing" and the k-curve's
  minimum is at k≈12. But the law's precondition is **near-orthogonal contexts**, and ours are measured
  ρ = **0.958** correlated. With contexts that aligned, the effective K is ~1: all documents write into
  the same channel, which is *exactly* the observed "cumulative, largely document-agnostic adaptation."
  Prediction: the k-curve minimum at k≈12 is a **capacity** signature and should scale with M, not with
  the document count — testable (experiment #3). If a per-document context phase could be installed
  with even modest decorrelation, predicted **ΔMT −0.05 to −0.15**; but DIAG-KEYSPACE bounds the
  achievable decorrelation (optimal key placement buys 1.22 vs incumbent 1.083), so I hold this at
  **−0.02 to −0.06** and inside/straddling the resolution.
- **Cost.** Gradient-free (contexts are fixed random rotations).
- **Why it might NOT transfer.** PSP needs a **task cue at inference** to pick `C_k`. We have none, and
  QA/MT queries are indistinguishable at 2.5× below sampling noise. Without the cue, the superposition
  is read with a uniform context — the current system.
- **Verdict.** **Unusable directly / the correct theory of our failure.** Its formula explains both why
  16 documents fit and why they do not stay separate: capacity is fine, *addressing* is not.

---

### L02-18: how many features fit in m dimensions as a function of sparsity — Toy Models of Superposition (Elhage et al., [transformer-circuits.pub](https://transformer-circuits.pub/2022/toy_model/index.html), 2022)

- **Claim / mechanism.** Define capacity as dimensions-per-feature. At **low sparsity**, a model
  represents at most `m` features (one per dimension, orthogonal); as **sparsity rises**, superposition
  becomes favourable and the number of representable features grows well beyond `m`, in geometric
  arrangements (antipodal pairs, pentagons, tetrahedra). Interference is set by (i) co-occurrence
  probability, (ii) direction alignment, (iii) the model's tolerance. **Importance** determines which
  features get a dedicated dimension and which get superposed.
- **Selection signal.** **Old-task importance** in the only place it belongs: deciding what deserves a
  private direction. This is the field's cleanest normative answer to "what is the right sparsity level
  for a given capacity."
- **Maps to our code.** Not a method; a *sizing rule* for `TOP_T`. It says the right `t` is set by how
  sparse the write-events are, not by a hyperparameter sweep.
- **Prediction about OUR signal.** Apply the rule. Our write events are **not sparse**: each document's
  top-32 overlaps a document-independent top-32 by **87.4%**, i.e. the "features" (documents)
  co-activate almost always. In the toy-model phase diagram that is the **dense** regime, where
  capacity is exactly `m` features and superposition provides **no** gain — you get one document per
  dedicated direction and no more. With 32 slots/layer of budget and 16 documents at 87.4%
  co-activation, the predicted usable capacity is ~**1–2** effective documents' worth of private
  structure, with everything else shared. That is a *quantitative* restatement of "the write produces
  a cumulative, largely document-agnostic adaptation" and of "the curve saturates at k≈12 then
  degrades." Predicted: no setting of `TOP_T` changes this, because `TOP_T` does not change
  co-activation. Consistent with MECH-BUDGET-B (t128 vs t64 inside resolution, not sign-consistent).
- **Cost.** Free (analysis).
- **Why it might NOT transfer.** The toy model's features are *inputs* to a bottleneck; our slots are
  *memory*, and the analogy between "feature sparsity" and "document co-activation" is mine, not the
  paper's. Treat the number (1–2 effective documents) as an order-of-magnitude claim.
- **Verdict.** **Candidate (as the sizing argument that closes `TOP_T`).** It gives the principled
  reason the budget sweep had to come out flat, which the project currently states only empirically.

---

### L02-19: top-k activation as an organic continual learner — Sparse Distributed Memory is a Continual Learner (arXiv [2303.11934](https://arxiv.org/abs/2303.11934), Bricken, Davies, Singh, Krotov, Kreiman, ICLR 2023)

- **Claim / mechanism.** An MLP whose neurons are SDM "addresses" on an L2 hypersphere, with a **Top-K**
  activation and a **GABA switch** (anneal `k` from max down to target, subtracting the (k+1)-th
  activation) to avoid dead neurons. Top-K "causes the `k` neurons most activated by an input to
  specialize towards this input, resulting in the formation of specialized subnetworks." Split-CIFAR-10
  class-incremental, no replay: SDMLP **71%**, +EWC **86%** (SoTA for no-replay), ReLU baseline 21%,
  oracle 93%. They state "the smaller `k` is, the less catastrophic forgetting occurs," and offer **no
  closed-form optimal `k`** — it is empirical and manifold-dependent.
- **Selection signal.** **Purely input-driven (new-task utility)** — cosine similarity of the current
  input to each neuron's address. No task labels, no boundaries, no old-task importance. Importance
  (EWC) is bolted on separately and is what takes 71% → 86%.
- **Maps to our code.** This *is* our incumbent's selection rule with the labels changed: our `tf` is
  the softmax attention weight, i.e. input-driven similarity to each slot's key. `SLOT_SELECTION=tfidf`
  with `USE_IDF=0` **is** SDM Top-K.
- **Prediction about OUR signal.** SDM's enabling condition is that Top-K *specialises* — different
  inputs must select different neurons, which the GABA switch exists to enforce. Ours does not: mean
  Spearman **0.958** between documents' slot scores. So the port is already at its ceiling and the
  paper's own advice ("smaller `k` forgets less") pushes toward the axis we do not need. Predicted:
  reducing `TOP_T` from 32 → 16 buys QA ≤ 0.04 (inside ±0.054) and costs MT — realised mass roughly
  halves at the margin (the top-32 mass is concentrated, so more like 0.2799 → ~0.21) → **MT ≈ 2.317**
  (+0.045, straddling the resolution). Their `+EWC` result maps to our `fisher` arm, which gave the
  best QA ever measured (**1.8641**) and **+0.397 MT** — the same 71→86 direction, on the axis we do
  not need.
- **Cost.** Gradient-free selection; GABA annealing needs training.
- **Why it might NOT transfer.** SDM works because neuron addresses **move** during learning (they
  "tile the manifold"). Our keys are frozen by default; MECH-005 moves them and is the only confirmed
  win (**ΔMT −0.1545**), which is consistent with SDM's story — but DIAG-KEYSPACE bounds how far that
  can go (optimal key placement 1.22 vs incumbent 1.083).
- **Verdict.** **Unusable as a change / strong support for the key-side direction.** It is the paper
  that says the *addresses*, not the update rule, are what make sparsity work — and address quality
  (selectivity 1.04–1.05 vs floor 1.0812) is this system's actual defect.

---

## Bonus: two papers read for framing, not proposed

* **Hiratani, "Disentangling and mitigating the impact of task similarity for continual learning"**
  (arXiv [2405.20236](https://arxiv.org/abs/2405.20236)). Two similarity axes — input/feature `ρ_a`
  and readout `ρ_b` — with `Δε_TF = ρ_a(2ρ_b − ρ_a)` and `Δε_RT = 1 − ρ_a²(ρ_a² − 2ρ_aρ_b + 1)`.
  Central result: "**high feature similarity coupled with low readout similarity is catastrophic for
  both knowledge transfer and retention**," and transfer is **non-monotone** in `ρ_a` — past a
  threshold, *more* input similarity makes transfer *worse*. Our cell: `ρ_a ≈ 1` (QA/MT query
  separation 2.5× below sampling noise, 0/288 heads), `ρ_b` low (writing a paper in makes the model
  worse at that paper; 71–93% of a solo write's gain lands on other documents). Then
  `Δε_TF ≈ 2ρ_b − 1 < 0`. Their prescribed mitigations are *activity gating with a probe trial* (needs
  a task cue we do not have) and *Fisher-metric weight regularisation* (retention — the axis with
  slack). **This is the theoretical statement of why our gate cannot gate.**
* **Konishi et al., SPG, "Parameter-Level Soft-Masking for Continual Learning"**
  (arXiv [2306.14775](https://arxiv.org/abs/2306.14775), ICML 2023). Notable only for its abstract's
  diagnosis of the whole isolation family: hard masking is achieved "by letting each task monopolize a
  sub-network in a shared network, which **seriously limits knowledge transfer** and causes
  **over-consumption of the network capacity**, i.e., as more tasks are learned, the performance
  deteriorates." SPG's fix is soft gradient scaling by `(1 − importance)` — algebraically a smooth
  `AM_SAFE_FRACTION`, so it lands on the same dose-response. `unusable`.

---

## The bottom line, stated plainly

**Yes — this entire family is bounded here by the `r ≈ −0.88` relationship, and I can now say why in
one sentence per axis.**

* **Why the *selection* axis is bounded.** Every method above is a rule for choosing a subset `S` of
  the 512 slots. In this system, the only property of `S` that predicts MT loss is `mass_on_S`
  (Pearson −0.877 across selectors, −0.9775 across constraint strength, −0.986 across seeds; my OLS
  fit has max residual 0.015 and predicts the out-of-family `fisher` arm to 0.039). The incumbent
  maximises `mass_on_S` **by construction**. Therefore every method above is, at best, a tie and
  generically a loss. The only escapes from the *law itself* are (a) full support — run, and 16
  documents mutually annihilate — and (b) more slots, which is out of scope.
* **Why the *write* axis is bounded.** MEMIT's two dials both exist in our code and are both already
  pinned at the plasticity extreme: `OLD_REFERENCE_WEIGHT` is forced to 0.0 because
  `ENABLE_OLD_REFERENCE_GUARD` defaults off, and the proximal `DELTA_WEIGHT` cannot be lowered because
  the *perfect* write (teacher's own KV, no proximal term) reaches only MT **2.381**.
* **Why the *isolation* premise is not merely unhelpful but wrong-signed.** Parameter isolation
  maximises non-interference. This system measured non-interference at **+1.101 worse**. Sixteen
  documents mutually overwriting each other (Jaccard 0.713, 9.83 writes/slot, 4.7% survival) is not a
  bug being tolerated — it is where the acquisition comes from.
* **What the field says to do instead, unanimously: grow capacity.** PackNet stops at 4 tasks; GPM
  reports 78% of the space consumed after 5 and states learning then stops; OWM's capacity *is*
  rank(P); WSN spends 48.65% on 40 tasks; SupSup/O-LoRA/InfLoRA/WISE each add per-task structure or a
  side memory. Cheung gives the formula (`ε ∝ (K−1)/M`) and Elhage gives the sizing rule (dense
  co-activation ⇒ capacity is exactly `m`). **This is out of scope at a fixed 512 slots. It is also
  the honest finding, and it agrees with the synthesis's own "where a win most likely lives:
  non-fixed-slot formulations."**

The one thing this review adds that the synthesis does not already contain: **the anti-alignment is not
a property of our selectors, it is the generic case for this architecture**, because dense softmax
attention makes the routing weights nearly input-independent (ρ = 0.958), which is Hiratani's
`ρ_a ≈ 1, ρ_b low` cell — theoretically the worst cell for *both* transfer and retention, and the one
cell in which no gate can help.

---

## Top 3 experiments I would run next

Baseline **QA 1.9468 / MT 2.2681**; resolution **±0.049 MT / ±0.054 QA**. All three are designed to
**test the law**, because a fourth arm that confirms it is worth less than one that could break it.
None of the three is expected to beat the incumbent; #1 costs no GPU at all.

### 1. IDF forensic — is `USE_IDF` on the same line? (zero GPU, ~1 hour)
**Single variable:** none — a re-read of existing artefacts.
**Do:** load an archived `USE_IDF=1` run's `am_doc_*.pt` `ranking_info` (or recompute `tf·idf` offline
on `outputs/phase1_selfdistill_qwen512/cache_last.pt` with `IDF_TOP_K=128`) and read
`mass_on_S_mean` from `value_solve.py:146`. Also report `|topk(tf·idf) ∩ topk(tf)|` per layer.
**Prediction:** IDF's measured MT (+0.465 → ≈ 2.733) requires realised `mass_on_S` ≈ **0.0140**
(band **0.0103–0.0192**), i.e. **5–7%** of the incumbent's 0.2799; and the top-32 overlap with the
incumbent should be **≈ 0**, because IDF is `constrained_mass` with the constraint set to the
*complement* of the background top-128.
**Falsifier:** realised mass ≥ **0.10** while MT ≈ 2.73 → the single-variable log-mass law is broken,
there is a second axis, and **the entire gating family reopens**. Also falsified if the overlap is
large (then my mechanism for IDF is wrong even if the law holds).
**Why first:** it is the only cheap test of the project's central claim against an arm that was *not*
in the regression that produced it.

### 2. Random-support control — the law at 6× lower mass (~1 GPU-hour)
**Single variable:** `SLOT_SELECTION=random` (≈8 lines in `ranking.py`: seeded `randperm` per layer,
`AM_RANDOM_SEED`), everything else at the confirmed best point (`TOP_T=32 GRANULARITY=per_layer
KEY_MODE=highest_attention AM_KEY_REPOSITION=1 AM_ROPE_THETA=5000000 DELTA_WEIGHT=1e-2
MAX_QUERIES_PER_HEAD=64`), 3 `AM_SEED_OFFSET`s, read the k-curve.
**Prediction:** projected bandwidth `32/511 = 6.262%` → realised `mass_on_S` **0.038–0.047** →
**MT 2.54–2.58** (point **2.561**, i.e. **+0.293**) — which places random *between* the worst
constrained arm (`q=0.25`, 2.4388) and the pure-`fisher` arm (2.665), exactly where the law says a
selector with mass 0.042 belongs. **QA 1.88–1.92** (QA-Fisher exposure ≈ 5% vs incumbent 29.6%,
`fisher` 0.04%). Selectivity
**1.04–1.05**. Per-document survival should **rise to ~60–70%** (expected pairwise Jaccard 0.032).
**Falsifier:** MT ≤ **2.45** (i.e. ≥ 0.09 better than predicted, ~2× the resolution) → the law does not
hold at low mass, mass is not the binding variable, and the whole selection family reopens. Symmetric
falsifier: MT ≥ 2.70 with mass ≈ 0.04 → the law under-penalises and something else (support geometry)
matters.
**Why:** it is the only single-variable arm that probes the law **6× outside** the range it was fit on,
and WISE (L02-16) is a published argument that random can beat importance. If it lands on the line,
the family is closed to the standard of the constraint dose-response.

### 3. Document-count scaling — is `k* ≈ 12` a capacity limit or a data artefact? (~3 GPU-hours)
**Single variable:** number of documents written, `K ∈ {8, 16, 32}` from the same `SYNTH_DATA_PATH`,
everything else at the confirmed best point; read the k-curve from `cache-after-doc-*.pt`.
**Prediction (from Cheung `ε ∝ (K−1)/M` + Elhage's dense-regime sizing):** if the k≈12 minimum is a
**capacity** signature of M=512, then `k*` should be **K-independent at 12 ± 2** and the best MT should
be **2.25–2.29 for all three** — i.e. K=32 buys nothing beyond k≈12, and K=8 tops out *worse*
(**MT ≈ 2.35–2.45**) because it never reaches k*.
**Falsifier:** K=32 reaching **MT < 2.20** at some k > 14 → not capacity-limited, more documents help,
and the "grow M" recommendation is wrong. Conversely K=8 reaching **MT ≤ 2.28** at k=8 → `k*` tracks
the data, not the slots, so the 512-slot budget is **not** the binding constraint and opening the
capacity scope would **not** pay.
**Why:** this is the only experiment here that gives the human a defensible answer to the one open
question the synthesis leaves — *"out of scope, and where a win most likely lives: non-fixed-slot
formulations."* It costs 3 GPU-hours to learn whether that hope is founded.

**What I would *not* run:** any further importance-aware selector. Between MECH-INFOGATE (3 selectors),
MECH-CONSTRAINED (a 4-point monotone dose-response), and VERIFY-GATE (3 seeds, 12/12 matched-k cells),
this family has been tested to a higher standard than the project's own positive result. The remaining
literature ideas — GPM/OWM/Adam-NSCL/O-LoRA/InfLoRA/HAT/WSN — all reduce to `AM_SAFE_FRACTION < 1.0`
under a different name, and its optimum is 1.0.
