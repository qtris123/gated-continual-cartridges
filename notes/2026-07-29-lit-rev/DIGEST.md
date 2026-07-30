# DIGEST — every claim, its evidence, and its source

**Purpose:** one file to catch up. Five scout threads, 88 papers, distilled to claims you can check.
**Convention:** ⚑ = changes what we do · ⚠ = corrects something we previously believed · 🔒 = closes a direction
**Our numbers referenced throughout** are defined in `../../research_loop/GLOSSARY.md`; the investigation
they came from is `../2026-07-29-am-investigation-synthesis.md`. Action list: `00-CONSOLIDATED.md`.

---
## 0. The seven claims that matter

| # | claim | strength |
|---|---|---|
| **C1** | ⚠🔒 The "no operator can pass" ceiling **is retracted** — the mass law is U-shaped, refuted by our own high-mass measurement | our data |
| **C2** | ⚑ We run at **≈428× compaction**; the AM paper itself says AM loses at **100×**, for the reason we measured | paper + our data |
| **C3** | 🔒 **Nobody overwrites a fixed compressed cache.** Composition is by concatenation, universally | 8 papers |
| **C4** | 🔒 Our (ρ_a, ρ_b) cell is the one theory names as **catastrophic for both transfer and retention** | Hiratani |
| **C5** | ⚠ Protection is **wrong-signed** here, not merely useless — theory predicts our +1.101 result | Lee et al. |
| **C6** | ⚠ Three of our six slot metrics are **algebraically mis-specified**; fixing them **strengthens** the negative | derivation + our arrays |
| **C7** | ⚑ The field's answer is **gradients or growth** — and one paper prices the gradient at 1 step | GradMem |

---
## 1. ⚠ RETRACTION — the closed-form ceiling does not hold

**What was claimed (threads 02 + 03, independently, to four digits):**
`MT = 2.0613 − 0.1575·ln(mass_on_S)`, max residual 0.015, and it predicts out-of-family (the `fisher`
arm: measured 2.665, predicted 2.626). Since mass is a probability ≤ 1, the ceiling is **MT 2.061 > 2.02**
⇒ no operator obeying the law can pass.

**Why it fails (thread 04):** `ORACLE-WRITE-512` measured **mass = 1.0** — the exact extrapolation point —
at **MT 2.576 / 3.237**. The law predicts 2.061 there; it is off by **0.5–1.2 nats = 10–24× the paired
resolution (±0.049)**. **The curve is U-shaped, not monotone.**

**What survives:** the law is sound *inside its fitted range* (mass ≈ 0.10–0.28). The correct, narrower
statement is: **nothing that *reduces* routing mass can beat the incumbent.** Operators that *raise* mass
are unconstrained by it — and our one measurement up there says loss rises again.

> **Lesson:** two independent agents agreeing to four digits is not evidence about a region neither
> sampled. The refuting datapoint was already on disk.

---
## 2. Where we actually sit (thread 01)

| fact | value | source |
|---|---|---|
| Our compaction ratio (**measured from the parquets, first time**) | Phase-1 ≈123k + Phase-2 ≈96k tokens → 512 slots = **≈428×, 16 slots/doc** | our data |
| AM's own stated breaking point | beats Cartridges at **50×**, **loses at 100×** | `AM.pdf` §6 |
| AM's stated reason | *"gradient-based optimization is not restricted to selecting keys from the original cache — increasingly important as the budget shrinks"* | `AM.pdf` §6 |
| Cartridges-at-Scale per-document budget | **≈1.2K tokens/doc — 75× ours** | [2606.04557](https://arxiv.org/html/2606.04557) |
| Monolithic vs per-document cartridge (FinQA, matched budget) | **14.1% vs 62.7%**; 20 isolated cartridges loaded together collapse 73.6 → 26.0% | 2606.04557 |

**⚑ C2:** our 0.248 acquisition shortfall is the **source paper's own predicted regime**, four times past
where its authors said the method breaks. We proved carefully something the paper had already stated.

**🔒 C3 — nobody has our problem.** Composition of compressed caches is by **concatenation**, universally:
Cartridges §5.4 · ICAE Tab. 8 · AutoCompressor (`σ_{<i} = Concat(...)`) · Activation Beacon · C²KV ·
KBLaM · InfLLM · InduceKV. **Overwriting a fixed compressed cache with a new domain has no published
instance.**

**Our capacity is not bad — it is 1–2 orders better than published:**

| system | density | note |
|---|---|---|
| MemoryLLM (fixed pool, genuinely gradient-free write) | **2.6 tok/slot** (<20k tokens in 7,680 slots/layer) | |
| Larimar (closed-form write, **K=512 — our exact shape**) | **~1 item/slot**, cliff at N=K | |
| ICAE | breaks past **4–8×** | |
| **ours** | **141 tok/slot** at the k-curve minimum | DIAG-CONTENT says 28–73% is content-free |

**Non-monotone curves under repeated writing are the norm:** Still collapses 51.0%@32k → **1.5%@128k**
(*below the no-context floor*); DualPrompt selection drops **<50% by task 10**.

---
## 3. 🔒 The theory that closes the gating family

### C4 — Hiratani, *Disentangling and Mitigating the Impact of Task Similarity* (NeurIPS 2024, [2405.20236](https://arxiv.org/abs/2405.20236))
- **Enabling condition for Fisher-metric protection:** free only while `N_s ≪ N_x(1 − ρ_a²)`.
  With our routing cosine **ρ_a = 0.99899** and 511 writable slots ⇒ **N_s ≪ 1.03 slots/layer**.
  We write **54.4**. Even on the generous 0.914-overlap reading the bound is ≪ 84.
- **`Δε_TF = ρ_a(2ρ_b − ρ_a)`** — "high feature similarity coupled with low readout similarity is
  catastrophic for **both** transfer and retention." Our ρ_a ≈ 1 (query separation **2.5× below sampling
  noise**), ρ_b low. **That is the worst cell, and the one where no gate helps.**
- Slot/activity gating and weight regularisation are **the same scalar knob** trading transfer for
  retention — which *is* our MECH-CONSTRAINED dose-response (r = −0.9775).
- Singles out the **diagonal Fisher approximation** as the fragile variant that loses low-rank invariance.
  **MECH-008's `fisher` is exactly that variant.**

### TRGP (ICLR 2022) — the field says *relax* the constraint
> *"naive orthogonal projection could possibly compromise the learning performance of the new task that is
> strongly correlated with old tasks"*

Its fix is to **relax** the constraint — i.e. `q → 1.0`, **which is the incumbent.**

### C5 — Lee et al., *Maslow's Hammer* (ICML 2022) — protection is wrong-signed
Strong consolidation "amplif[ies] the bias to fresh node activation"; at high λ the new task is learned
"with a **tabula rasa node**". **A fresh uncontested slot is a tabula rasa node** — our **+1.101**
solo-write result, with the sign the paper predicts, in the maximally-aligned regime where its verdict on
EWC is "little effect on forgetting, still limits acquisition".

### PackNet arithmetic (thread 02)
16 documents × 32 slots = **512** = the whole cartridge. PackNet's capacity-exhaustion point **is** our
K=16 — and that regime is the measured **+1.101** arm. Isolation isn't unhelpful here, it's **wrong-signed**.

### The field's unanimous answer when the constraint binds: **grow capacity**
GPM (78% of space consumed ⇒ "no new learning will be possible") · OWM (capacity *is* rank(P)) ·
Cheung superposition (ε ∝ (K−1)/M). **Out of scope at fixed 512 slots** — reported as the honest finding.

---
## 4. ⚠ C6 — our instrument was mis-specified (thread 04)

| # | error | consequence |
|---|---|---|
| **E1** | **`redundancy` uses the wrong Gram.** Correct damage-of-overwriting is `‖v_j‖²/(H⁻¹)_jj` with **`H = E_q[a aᵀ]`** (routing second moment) — verbatim the **OBS / SparseGPT** saliency. Ours used `G = VVᵀ` (value space) and *divided* `‖v_j‖²` out instead of multiplying in. **`G` is query-independent ⇒ `redundancy` cannot be a QA-importance metric at all** | its "at chance vs MT-wanted" (7.33 vs 8.0) is **the null, not a discovery**; MECH-009's dose-response is a dose over *value-space redundancy*, not QA importance |
| **E2** | **ρ = −0.648 "best Fisher proxy" is an ecological correlation.** Within-layer **−0.323**; ρ(layer index, mean redundancy) = **+0.752** explains the pooled figure. Set agreement with Fisher's safest-32: **25.4%** | `METRICS.md` warned about pooling; the GLOSSARY, the synthesis and `ranking.py`'s docstring all quote the pooled number — **my error, propagated three places** |
| **E3** | **Lemma:** attention output is **linear in the values** ⇒ *every* first-order slot score is a query-reweighted attention mass | proves `kl_loo`'s collapse; predicts **MAS would collapse too — do not build it**; shows **K-FAC/GGN diagonal collapses to `E_q[a_j²]`** |

**A free estimator dominates two of our six.** `E[w²] ≈ 2(kl_loo − w_mass)`, from arrays already on disk:

| metric | per-layer ρ with Fisher | Fisher-safest-32 agreement |
|---|---|---|
| **free surrogate** | **0.847 ± 0.056** | **0.753** |
| `tf_mass` | 0.666 | — |
| `redundancy` | **−0.323** | 0.254 |

Rank-R² of Fisher on two free numbers = **0.708** against a reliability ceiling of 0.95 ⇒ the **98.8 s
backward pass bought ~26%** of the reliable signal. **Recomputing the headline anti-alignment with the
corrected metric deepens it from 11× to 48× below chance** (the published 0.72 / 11.1× reproduced exactly
as a control). **The audit strengthens the negative.**

**Other instrument findings:** `k ≈ 12` is **not** a capacity number — linear associative memory, modern
Hopfield (2^(d/2)) and superposition/JL all put capacity **10–40× higher**. What Hopfield *does* name is
our **symptom**: non-separated patterns give **metastable averaging** — verbatim the synthesis's
"document-agnostic adaptation". `contrast` has a measurement ceiling of |ρ| ≈ 0.52–0.83 and should not be
used as a ranking.

**Did we measure the right thing? No.** "Importance to Phase-1" is a *retention* quantity; retention is
**0.573 ahead** of budget. **GPM's k-rank criterion and Doan's NTK overlap would have predicted the 0.914
overlap and the anti-alignment in advance**, from DIAG-ROUTING's own numbers, with **no Fisher at all**
(first principal angle **2.58°**).

---
## 5. The β bound, and what escapes it (thread 03)

**Why β failed is algebra, not tuning.** `w_j(q) = softmax_j(q·k_j/√d + β_j)`; `β_j` is a **per-slot
constant**, identical for every query. So MT and QA mass pass through the **same monotone map**: β moves
**bandwidth**, never **selectivity**. Predicted before the run (+2.32 nats: mass 0.090 → 0.500, ratio
1.083 → **1.046**); measured: mass **4.23×**, ratio **1.043 → 1.051**, both axes worse.
**Every mechanism we tried had lower selectivity than doing nothing** (untouched cartridge: **1.0812**).

**The filter that follows:** any operator whose eval-time effect is a per-slot constant *or a static slot
choice* inherits the bound. **Marked unusable on that ground:** HAT · SPLADE · Hash Layers · Switch ·
Soft MoE · expert-choice · BASE · Slot Attention · NSA's branch gate · InduceKV's own `λ_ℓ`.

**Query-dependent, therefore live:**

| mechanism | rule | trainable? |
|---|---|---|
| **Quest** ([2406.10774](https://arxiv.org/abs/2406.10774)) | per-query page criticality `Σᵢ max(qᵢmᵢ, qᵢMᵢ)` | **no training at all** — two reductions + one dot product, eval-time only |
| **InduceKV** ([2607.02010](https://arxiv.org/abs/2607.02010)) | `α_i(x) = softmax(⟨r(x), r_i⟩/τ)` | gate is **37 scalars**, grid-searchable |
| **Routing Transformer** ([2003.05997](https://arxiv.org/abs/2003.05997)) | online k-means clustering | EMA, no backprop through routing |

**Why keys escaped:** changing `k_j` changes `q·k_j`, which **varies with q**. That is why MECH-005 moved
MT by −0.155 where β moved nothing.

**⚑ Cheapest test in the whole review:** `cartridges/models/attention.py:106` **already passes `score_mod`
the batch index `b` and `q_idx`** — a per-example, query-dependent bias is a **3-line change with zero
modification to the write path**, testable **eval-time only** on the cartridge we already have.

**On expert-choice:** it fixes the arithmetic (9.83 → 1.0 writes/slot, survival 4.7% → ~100%) but our own
data predicts it loses — at capacity 1 it **is** the ORACLE-WRITE-512 regime, and DIAG-PERDOC measured
uncontested writes at **+1.101 worse**. Predicted MT ≥ 2.45. Worth running only as a falsifier.

---
## 6. ⚑ C7 — the field's answer is gradients or growth, and it is priced

**GradMem ([2603.13875](https://arxiv.org/abs/2603.13875)) — same architecture, 8 memory vectors:**

| write | accuracy |
|---|---|
| forward-only | **19.3–45.5%** |
| **one gradient step** | **58.6–96.3%** |
| five steps | 99.1–100% |

and *"repeating the forward-only write yields weak or inconsistent improvements"* — **our k=12→16 tail,
verbatim.**

**Two independent validations of our own work:** Still and C²KV **both** strip RoPE before compaction and
re-apply at placement — that is **MECH-005**, invented independently by two 2026 papers. KBLaM adds
`log C − log M` to **cap** a memory block's mass as it grows — **the field's mass lever runs opposite to
ours**, corroborating MECH-BETA.

**A genuinely novel negative:** the literature assumes **overlap is interference** (InduceKV's coverage/DPP
term, predicted here at MT ≈ 2.39; L2P's frequency penalty, predicted at MT ≈ 2.44). **We measured that
overlap is the mechanism** — uncontested slots are *worse*.

---
## 7. Is the founding intuition right? (thread 05) — **no, mis-specified for this setting**

Three of its four load-bearing assumptions fail:

1. **Protection works and buys nothing we need** — `fisher` gave the best retention ever measured here
   (QA 1.8641, 0.04% Fisher exposure) and paid **+0.397 MT**.
2. **Overwriting doesn't cost what the intuition says** — survival 4.7%, yet doc 1 contributes **3.1%** of
   the MT gain and docs 2–16 contribute **96.9%**. *Slot survival ≠ information survival.*
3. **Uncontested slots are WORSE** (+1.101), and **71–93%** of a solo write's gain lands on *other*
   documents' questions.
4. **The write may not be a store** — Hendel et al. recover **80–90%** of ICL accuracy from a *single
   vector* computed with a dummy query; Min et al. measure content at **0–5%** vs distribution **3–16%**;
   Yadav et al.'s merging-at-scale reproduces our strangest fact (held-in degrades 0.66 → 0.39 at 8
   experts while held-out improves monotonically).

**Two untried levers, both larger than anything we optimised:**
- **Order.** Our order effect is **0.340** — *twice* our best gating result (0.167) — and **never
  optimised**. Bell & Lawrence give a computable asymmetric distance `c(j,k) = g_kᵀH_jg_k` whose max-sum
  path minimises forgetting.
- **Metric.** Holtzman's **PMI_DC**: calibrated against the matched-domain control, **our own DIAG-CONTENT
  table says k=16 is the best point, not k=12** (canonical−control widens 0.232 → 0.288 → 0.608 → 0.882).
  **The tail "regression" is largely content-free.** Zero GPU to re-derive.

**The discriminating experiment — CONTENTION-SWEEP.** Arms: **D** disjoint-16 (511 slots partitioned into
16 blocks of 32; zero contention; **never run** — DIAG-PERDOC tested **solo** writes, which conflates
contention with the presence of the other 15 documents), **P** incumbent, **S** one document-independent
top-32 shared by all (also never run, and already 87.4% identical to P by our own measurement).
Non-overwriting predicts **D** best; the alternative predicts **S ≤ P ≪ D**.
**Falsifier: if D beats P by more than ±0.049, every argument in thread 05 dies.**

---
## 8. Sources

**Local:** `AM.pdf` (Fast KV Compaction via Attention Matching, 2602.16284) · `TF-IDF.pdf` (Continual
Learning via Sparse Memory Finetuning, [2510.15103](https://arxiv.org/abs/2510.15103))

**Compressed cache / memory:** Cartridges 2506.06266 · [Cartridges at Scale 2606.04557](https://arxiv.org/html/2606.04557) ·
[InduceKV 2607.02010](https://arxiv.org/abs/2607.02010) · [GradMem 2603.13875](https://arxiv.org/abs/2603.13875) ·
ICAE · AutoCompressor · Activation Beacon · C²KV · KBLaM · InfLLM · StreamingLLM · Still · MemoryLLM · Larimar

**Sparse finetuning / CL:** PackNet · [HAT 1801.01423](https://arxiv.org/abs/1801.01423) · SupSup · Piggyback ·
WSN · GPM · OWM · OGD · Adam-NSCL · O-LoRA · InfLoRA · TRGP (ICLR 2022) · WISE (NeurIPS 2024) · Cheung superposition

**Gating / routing:** [Soft MoE 2308.00951](https://arxiv.org/abs/2308.00951) · [Expert Choice 2202.09368](https://arxiv.org/abs/2202.09368) ·
[Switch 2101.03961](https://arxiv.org/abs/2101.03961) · [BASE 2103.16716](https://arxiv.org/abs/2103.16716) ·
[Hash Layers 2106.04426](https://arxiv.org/abs/2106.04426) · [Slot Attention 2006.15055](https://arxiv.org/abs/2006.15055) ·
[Routing Transformer 2003.05997](https://arxiv.org/abs/2003.05997) · [Quest 2406.10774](https://arxiv.org/abs/2406.10774) ·
[NSA 2502.11089](https://arxiv.org/abs/2502.11089) · [SPLADE 2107.05720](https://arxiv.org/abs/2107.05720) ·
[Conditional Channel Gated Nets 2004.00070](https://arxiv.org/abs/2004.00070) ·
[Fast Weight Programmers 2102.11174](https://arxiv.org/abs/2102.11174) · [DeltaNet 2406.06484](https://arxiv.org/abs/2406.06484) ·
[Product-Key Memory 1907.05242](https://arxiv.org/abs/1907.05242) · [NTM 1410.5401](https://arxiv.org/abs/1410.5401) · L2P · DualPrompt

**Theory / measurement:** [Hiratani 2405.20236](https://arxiv.org/abs/2405.20236) · Lee et al. *Maslow's Hammer* (ICML 2022) ·
OBS · SparseGPT · K-FAC · influence functions · TracIn · CKA · SVCCA · Doan NTK overlap · modern Hopfield ·
Hendel et al. (task vectors) · Min et al. (rethinking demonstrations) · Yadav et al. (merging at scale) ·
Bell & Lawrence (task order) · Holtzman (PMI_DC / surface form competition)

**Full per-paper entries with formulas, code mappings and numeric predictions:** `01`–`05` in this folder
(3,900 lines total).
