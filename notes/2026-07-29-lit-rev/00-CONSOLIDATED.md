# Consolidated review — five threads, 88 papers, and one retracted claim

**Date:** 2026-07-29 · Threads: `01`–`05` in this folder · Context: `../2026-07-29-am-investigation-synthesis.md`

---
## 0. RETRACTION — the "closed-form ceiling" does not hold

Threads 02 and 03 independently fitted the dose-response arms and got the same law to four digits:

> `MT = 2.0613 − 0.1575·ln(mass_on_S)`, max residual 0.015

and both concluded: since routing mass is a probability ≤ 1, the family's ceiling is **MT 2.061 > 2.02**,
so **no operator obeying the law can pass**. The orchestrator relayed that as the review's most decisive
result.

**Thread 04 refutes it with our own data.** `ORACLE-WRITE-512` measured **mass = 1.0** — the exact
extrapolation point — at **MT 2.576 / 3.237**. The law predicts 2.061 there. It is off by **0.5–1.2 nats,
10–24× the paired resolution**. **The curve is U-shaped, not monotone.**

**What survives:** the law is well-fitted *inside its range* (mass ≈ 0.10–0.28) and predicts out-of-family
there (the `fisher` arm, measured 2.665, predicted 2.626). So the correct statement is narrower:
**nothing that *reduces* routing mass can beat the incumbent.** It says nothing about operators that
*raise* mass — and the one measurement we have at high mass says the loss goes *up* again.

---
## 1. Three estimator errors, all algebraic (thread 04)

| # | error | consequence |
|---|---|---|
| **E1** | **`redundancy` uses the wrong Gram.** Correct damage-of-overwriting is `‖v_j‖²/(H⁻¹)_jj` with **`H = E_q[a aᵀ]`** (routing second moment) — the OBS/SparseGPT saliency. Ours used `G = VVᵀ` (value space) and *divided* `‖v_j‖²` out instead of multiplying in. `G` is **query-independent**, so `redundancy` **cannot** be a QA-importance metric | its "at chance vs MT-wanted" (7.33 vs 8.0) is **the null, not a finding**; MECH-009's dose-response is a dose over *value-space redundancy*, not over QA importance |
| **E2** | **"`redundancy` is the best gradient-free proxy for Fisher (ρ = −0.648)" is an ecological correlation.** Within-layer it is **−0.323**; the pooled figure is driven by layer-index trends (ρ = +0.752). Set agreement with Fisher's safest-32 is **25.4%** | `METRICS.md` warned about pooling; the GLOSSARY, the synthesis and `ranking.py`'s docstring all quote the pooled number. **Orchestrator error, propagated three places** |
| **E3** | **Lemma:** attention output is **linear in the values**, so *every* first-order slot score is a query-reweighted attention mass | proves `kl_loo`'s collapse, proves **MAS would collapse too (do not build it)**, and shows the **K-FAC / GGN diagonal collapses to `E_q[a_j²]`** |

**And a free estimator that dominates two of the six.** `E[w²] ≈ 2(kl_loo − w_mass)`, computable from
arrays already on disk, gives per-layer ρ with Fisher = **0.847 ± 0.056** vs `tf_mass` 0.666 and
`redundancy` **−0.323**; Fisher-safest-32 agreement **0.753**. Rank-R² of Fisher on two free numbers is
**0.708** against a reliability ceiling of 0.95 — the 98.8 s backward pass bought ~26% of the reliable
signal. **Recomputing the headline anti-alignment with the corrected metric deepens it from 11× to 48×
below chance.** *The audit strengthens the negative.*

---
## 2. What the five threads agree on

1. **We are 4× past where the method's own authors said it breaks.** Measured for the first time
   (thread 01): Phase-1 ≈123k + Phase-2 ≈96k tokens into 512 slots ⇒ **≈428× compaction, 16 slots/doc**.
   **`AM.pdf` §6**: AM beats Cartridges at 50× but **loses at 100×**, "because gradient-based optimization
   is not restricted to selecting keys from the original cache — increasingly important as the budget
   shrinks." Cartridges-at-Scale budgets **1.2K tokens/doc — 75× ours**.
2. **Nobody overwrites a fixed compressed cache.** Composition is by **concatenation**, universally
   (Cartridges §5.4, ICAE, AutoCompressor, Activation Beacon, C²KV, KBLaM, InfLLM, InduceKV). Our problem
   has **no published instance**.
3. **Our theoretical cell is the one where no gate helps.** Hiratani (2405.20236): protection is free only
   while `N_s ≪ N_x(1 − ρ_a²)`; at our ρ_a = 0.99899 that is **≪ 1.03 slots/layer** against 54.4 written.
   `Δε_TF = ρ_a(2ρ_b − ρ_a)` names "high feature similarity, low readout similarity" as catastrophic for
   **both** transfer and retention. TRGP says the fix is to **relax** the constraint — i.e. `q → 1.0`,
   which *is* the incumbent.
4. **Protection is wrong-signed here, not merely useless.** Lee et al. (ICML 2022): strong consolidation
   makes the new task be learned "with a **tabula rasa node**" — our +1.101 uncontested-slot result, with
   the predicted sign. PackNet's capacity-exhaustion point is arithmetically ours (16 × 32 = 512).
5. **Gradients are the field's answer, and it is priced.** GradMem (2603.13875), same architecture, 8
   memory vectors: forward-only **19.3–45.5%**, one gradient step **58.6–96.3%**, five steps **99.1–100%**
   — and *"repeating the forward-only write yields weak or inconsistent improvements"*, our k=12→16 tail.
6. **Two independent validations of our own work.** Still and C²KV **both** strip RoPE before compaction
   and re-apply at placement — that is **MECH-005**, invented independently by two 2026 papers. KBLaM adds
   `log C − log M` to **cap** a memory block's mass — the field's mass lever runs *opposite* to ours,
   corroborating MECH-BETA.
7. **`k ≈ 12` is not a capacity number.** Three theories put capacity **10–40× higher**. But Hopfield names
   our *symptom*: non-separated patterns give **metastable averaging** — verbatim the synthesis's
   "document-agnostic adaptation".
8. **We measured a retention quantity while acquisition was the constraint.** GPM's k-rank criterion and
   Doan's NTK overlap would have predicted the 0.914 overlap and the anti-alignment **in advance**, from
   DIAG-ROUTING's own numbers, with **no Fisher at all** (first principal angle **2.58°**).

---
## 3. Ranked next steps

**Zero-GPU, do first (they can change the conclusions):**
- **N1 — swap the Gram.** Recompute `redundancy` with `H = E_q[a aᵀ]` and re-derive the trade-off table.
  Same cost as today's metric. Fixes E1; MECH-009's dose-response must be re-read against it.
- **N2 — free Fisher surrogate.** `E[w²] ≈ 2(kl_loo − w_mass)` from arrays on disk; ρ 0.847 vs Fisher.
- **N3 — IDF forensic.** Thread 02 derives that `USE_IDF=1` is `constrained_mass` **with the constraint
  inverted** (`IDF_TOP_K=128` + ρ≈0.958 rankings ⇒ `idf ≈ 0` on the highest-mass slots ⇒ **deletion**).
  Predicts its realised mass ≈ **0.014**. Read it off an archived run. Falsifier: mass ≥ 0.10 ⇒ a second
  axis exists and the gating family reopens.
- **N4 — PMI_DC re-scoring.** Calibrated against the matched-domain control, our own DIAG-CONTENT table
  says **k=16 is the best point, not k=12** (canonical−control widens 0.232 → 0.882): the tail regression
  is largely content-free.
- **N5 — effective rank of `E_q[a aᵀ]`.** Seconds. Could close the family by a capacity theorem.

**Cheap GPU, high information:**
- **N6 — `SLOT_SELECTION=random`** (~10 lines). **Nominated independently by threads 01 and 02.** Tests the
  mass law **6× outside its fit range**; MemoryLLM's actual policy; WISE argues random should *beat*
  importance here. Predictions differ across threads (2.40–2.58) — a real disagreement to resolve.
  **Also the right null for TF-IDF:** if random ≈ incumbent (or beats it), the ranking signal is not buying
  anything; if random lands where the mass law predicts (~2.54–2.58, i.e. clearly worse), TF/attention-mass
  selection is load-bearing. Do this before further selector work.
- **N7 — `CONTENTION-SWEEP`** (thread 05). Arms **D** disjoint-16 (511 slots partitioned; *never run* —
  DIAG-PERDOC tested **solo** writes, which conflates contention with the other 15 documents' presence),
  **P** incumbent, **S** one shared document-independent top-32. Falsifier: **D beats P by > ±0.049** kills
  the whole "overlap is the mechanism" reading.
- **N8 — `GATE-ORACLE`** (thread 03). `cartridges/models/attention.py:106` **already passes `score_mod` the
  batch index and `q_idx`** — a per-example query-dependent bias is a **3-line, eval-time-only** change.
  Tests whether *any* query-dependent gate can move selectivity past 1.05.
- **N9 — `GRAD-LADDER {0,1,2,5}`** — prices `gradient_steps = 0` in MT loss. GradMem predicts ~2.02–2.10
  at one step.

**Scope decision (needs the human):** every thread concludes the win is outside fixed-512-slot
gradient-free overwrite. Thread 03's best constructible gate predicts **MT 2.15–2.22** — real, 2–4×
resolution, still 0.13–0.20 short. The two published routes are **concatenation/growth** (Cartridges at
Scale, InduceKV) or **a few gradient steps** (GradMem) — which is the point the *prior* loop had already
reached, now with a measured reason rather than an assumption.
