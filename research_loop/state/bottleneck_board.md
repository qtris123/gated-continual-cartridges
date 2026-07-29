# BOTTLENECK BOARD — the loop's central artifact

> **What this file is:** the live causal picture of *why* gradient-free AM caps at MT ~2.54.
> The orchestrator rewrites it every cycle (ORCHESTRATOR.md STEP 7.2). It must never go stale.
> **Rule: no mechanism is built for a cause that has no measured number here.**

**Stages:** `A MEASURE` (suspected, no number) → `B SEARCH` (measured, need a fix from the literature)
→ `C BUILD` (fix chosen, needs code) → `D TEST` (built, needs a run) → `E VERIFY` (result needs
adversarial confirmation) → **CLOSED** (`confirmed+fixed` / `confirmed+capped` / `refuted`).

**Status vocabulary:** `suspected` (argument only) · `measured` (has a number) · `confirmed` ·
`refuted` · `capped` (real, and it bounds the method family) · `fixed`.

**Closing rule:** a cause may only be closed with a **mechanistic sentence** — what is happening, with
the number that shows it. "It didn't help" never closes anything.

---
## 🔴 B-OVERWRITE — the 16 documents may be overwriting each other AT top-32 TOO (opened 2026-07-28)
- **Stage:** A MEASURE (dispatched) · **Status:** suspected, and the arithmetic is alarming
- **The observation that opened it:** ORACLE-WRITE-512 failed because at full support each document
  writes all 511 slots, so only document 16 survives. But ORACLE-WRITE recorded that at `top_t=32` the
  **union of all 16 documents' selections is only 34–82 slots per layer**. Sixteen documents × 32 slots
  = **512 slot-writes landing in ~54 distinct slots ⇒ ~9 writes per slot.** The ranker is choosing
  *nearly the same slots for every document*, so later documents overwrite earlier ones **at the
  canonical operating point as well** — differing from the 512 case in degree, not in kind.
- **If true, it reframes the mission's central measurement:** ORACLE-WRITE's "perfect value write
  reaches MT 2.381" would be measuring *mostly the last few documents' values*, not 16 documents
  coexisting. The write ceiling would be an artefact of allocation, not a property of value-only writing.
- **Converging evidence (SCOUT-KEYS):** the written support (mean 54.7 slots/layer) carries **15.2%** of
  cartridge mass, while a **mass-ranked per-layer top-32 carries 70.4%** — a **4.6× bandwidth gap at a
  smaller budget**. Suspected cause: `query_accum.py:87-89` builds selection scores from a **GQA-group
  mean query taken before the softmax**. So the ranker may be picking both *low-mass* and *mutually
  colliding* slots.
- ✅ **CONFIRMED (DIAG-OVERWRITE, 2026-07-28) — and the survival number is catastrophic.**
  Measured on the canonical run itself (which reproduced EXP-007-top32 to all 16 digits; **no
  instrumentation needed** — the stock path already saves `am_doc_*.pt` selections and a
  `cache-after-*.pt` snapshot per document, `SAVE_AFTER_EACH_DOCUMENT=1`):
  - **Collision:** mean pairwise Jaccard between the 16 documents' per-layer top-32 sets = **0.713**.
    Union 54.4 slots/layer ⇒ **9.83 writes per slot** (max 16). **87.3% of all 18,432 slot-writes land in
    slots touched by ≥8 documents**; 18.6 slots per layer are written by **all 16**.
  - 🔥 **Survival: mean 4.7% across documents 1–15.** doc1 **4.9%**, doc8 **1.0%**, doc15 17.0%, doc16
    100%. Doc 1's values drift by a median **99% of their own magnitude**, cos(final, as-written) =
    **0.517**, each of its slots is re-written by **11.8** later documents, and the later documents' net
    change to doc 1's slots is **30× the change doc 1 itself made**.
  - **The gate is not gating:** mean Spearman between different documents' slot scores = **0.958**, and
    each document's top-32 overlaps a *document-independent* top-32 (built from the across-document mean
    score) by **87.4%**. Verified bitwise that `USE_IDF=0` ⇒ `tfidf == tf`, so this is the raw access
    score, not an IDF artefact.
- ❌ **The orchestrator's "16 documents are worth one" reading is REFUTED (DIAG-SEQUENCE, 2026-07-28).**
  Evaluating all 16 per-document snapshots: **MT after doc 1 is 3.7445 — only 0.038 below Phase-1** and
  **1.192 worse than the 16-document cartridge**. Doc 1 alone accounts for **3.1%** of the total MT drop;
  docs 2–16 for **96.9%**. **4.7% slot survival does not mean 4.7% information survival** — the trust
  region's partial traces carry real information. B-OVERWRITE is mechanically real but is **not** the
  acquisition bottleneck.
- ⭐ **What it is instead: a sharp capacity saturation at ≈12 documents.**

  | k | 0 | 1 | 2 | 4 | 8 | **12** | 13 | 16 |
  |---|---|---|---|---|---|---|---|---|
  | MT | 3.783 | 3.745 | 3.367 | 3.034 | 2.710 | **2.435 ← min** | 2.449 | 2.552 |
  | QA | 2.239 | 2.230 | 2.062 | 2.113 | 2.047 | 2.042 | **2.024 ← min** | 2.177 |

  MT falls **strictly monotonically at all 12 steps** to k=12, then rises **strictly monotonically** over
  the last four documents (+0.117). QA does the same one step later (min at k=13, then +0.153).
  **k=12 dominates the reported k=16 end state on BOTH axes** — the loop has been reporting the wrong
  point of the sequence. The tail rise is **not a coverage artefact**: docs 13–16 carry **27.5%** of the
  MT eval questions and writing them still makes MT worse.
- ⚠️ **Noise discipline:** MT n=69 / QA n=78. Single-step deltas below ~0.15 — **including the entire
  k=12→16 tail regression** — sit at or inside the band. Only the k=0→12 descent is far outside it.
  One seed; k=12 is not seed-varied. So "stop at 12" is a *candidate*, not a result.
- 🔴 **The caveat that opens the next question:** docs 1–3 hold only **18.8%** of the eval questions but
  deliver **56.7%** of the total MT improvement, and the curve sits *below* a linear-accumulation
  prediction at every k∈2..15. Per-document marginal ΔMT correlates only weakly with per-document eval
  share (**Spearman 0.51**). **Part of the "acquisition" may be a global distribution/format shift from
  perturbing the cartridge at all, not content being stored.** The same suspicion attaches to QA, which
  sits **below the untouched Phase-1 floor at every k∈1..16** even though the QA eval's 16 papers are
  **entirely disjoint** from the Phase-2 documents. → **DIAG-CONTENT dispatched** (write 16 *QA-topic*
  documents and eval MT: any MT gain is content-free by construction).
- **Eval coverage verified:** the MT eval spans all 16 training documents (69 examples, 16 unique
  `paper_id`s, matched 16/16 by title, 3–7 questions each).
- ⚠️ **Self-correction: the 4.6× bandwidth gap I put on this board is REFUTED.** DIAG-OVERWRITE reproduced
  SCOUT-KEYS' 70.4% exactly (0.70437) and ORACLE-WRITE/DIAG-ROUTING to 4–6 s.f., then showed the
  15.2%-vs-70.4% pair **mixes two aggregations of the same quantity** (the written union is 0.2037 in the
  aggregation the 70.4% uses), and that **0.5113 of the 0.7044 is the frozen sink slot 0**, unwritable by
  any AM config. Writable-only: mass-ranked per-layer top-32 = **0.1954 — 0.96× the tf-idf union, i.e.
  slightly WORSE at a smaller budget**; budget-matched = 0.2373 (1.17×); per-head = 0.2598 (1.28×).
  **There is no 4.6× writable-bandwidth lever.** Selection quality is not the problem; **allocation is.**
- ❌ **DISJOINT ALLOCATION IS DEAD BEFORE IT WAS BUILT (DIAG-PERDOC, 2026-07-29) — competition is not
  costing us, it is HELPING.** Writing a document **alone into 32 uncontested slots** is **+1.101 WORSE
  on that document's own questions** than writing it as one of 16 (sd 0.186, se 0.083, **same sign 5/5**;
  the θ=5e6 arm agrees at +0.753, 5/5). Pooled over 5 documents / 27 questions: Phase-1 3.9640 → **solo
  3.7488** → k=16 2.6186 → **k=12 2.5086**.
  A solo write buys only **15.5%** of what the full 16-document write buys on those same questions.
- 🔴 **And the benefit is not the document's own:** **71–93% of a solo write's full-MT gain lands on the
  OTHER 15 documents' questions** (doc-000: own −0.050 vs others −0.037). It is **not bandwidth or fit** —
  every solo write lands `ref_mass_on_S` 0.1477–0.1535 against the 16-doc run's 0.15851, with solve MSE
  0.0085–0.205. **A solo write is well-fitted and equally well-routed; it just doesn't carry the content.**
- **Uncertainty measured, not assumed** (subsets are 3–7 questions): a replicate solo write of doc-015
  differing only in which 32 of its 532 conversations were drawn gives spread **0.216**; leave-one-out
  recovers per-example losses exactly, giving paired solo−k16 = **+1.172 (se 0.226) with 0/7 questions
  favouring solo**. The gap is **5–14× the noise**; per-document *ordering* is not resolvable.
- **Provenance:** the filtered-parquet route reproduces the canonical `cache-after-doc-000`
  **bitwise (180/180 tensors)** and DIAG-SEQUENCE's k=1 loss to 16 digits; all session gates reproduce
  exactly.
- ⚠️ **Bonus correction to B-ROPE, forced into view by the gates:** the rotary base is **not a wash at
  small k**. One document alone: full MT **3.1146 at θ=5e6 vs 3.7445 at θ=1e4** — a first-document gain
  of 0.668 vs 0.038, **17.6×** — yet the arms converge by k=12 (2.4705 vs 2.4352) and k=16 (2.5296 vs
  2.5524). **DIAG-ROPE's "ΔMT −0.019, inside noise" was measured only at k=16.**

## 🔴 B-ROPE — the teacher targets are computed with the WRONG RoPE BASE (opened 2026-07-28)
- **Stage:** C BUILD (verified by the orchestrator; fix + A/B dispatched) · **Status:** **CONFIRMED bug,
  live in every AM run ever performed in this project**
- **The fact:** `cartridges/am/` hard-codes `rope_theta: float = 10000.0` as a default in **every**
  entry point (`core.py:24,51,83,122`, `teacher.py:142,164,187`, `key_select.py:59,110,236`), and
  **no caller anywhere passes it** — `grep rope_theta cartridges/am/finetune.py cartridges/am/continual.py`
  returns **nothing**. Meanwhile the actual model config:
  ```
  Qwen/Qwen3-4B-Instruct-2507 → rope_theta = 5000000, max_position_embeddings = 262144
  ```
  **500× wrong**, in the teacher-target path (`finetune.py:495` → `_apply_rope_offset_to_queries`) at
  document offsets T_doc = 3858–8900, where SCOUT-KEYS measured the two rotations as **decorrelated**
  (E[cos] 0.24–0.34 at 1–2k, **0.00 at 4k**).
- **Why this may reframe everything:** if the *target* is computed in the wrong rotary frame, then the
  solve has been fitting a **corrupted target** all along — and **B-OBJ's anti-correlation becomes
  exactly what you would predict**: fitting a wrong target better (18× lower MSE) *should* destroy the
  model (+13 loss). The same applies to `mass_on_S`, β, and every "AM cannot do X" conclusion.
- ✅ **RESOLVED (DIAG-ROPE, 2026-07-28). The frames do NOT cancel — and CE does not care.**
  **Trace:** cartridge slots sit at 0..511, reference queries at 512+i, document keys at 512+m; the model's
  own base is 5e6 (`modeling_qwen3.py:296-299`). `core.py:66-72` rotates the query forward by `T_doc` —
  correct in intent and magnitude — but `R_θ(a)∘R_θ(b) = R_θ(a+b)` **only for equal θ**. Composing the AM
  path's 1e4 on the model's 5e6 yields no single position's rotation: at T=4000 the per-pair angle error
  runs 0→749 rad and **76.6% of the 64 frequency pairs are off by more than π**; mean cos = 0.103 at
  T=3858, **−0.099 at T=8900**. `compute_teacher_log_mass` (β's own target) takes the identical rotation.

  | | QA | MT | mean `am/mean_mse` | \|v\|max | cartridge mass (MT) |
  |---|---|---|---|---|---|
  | A θ=1e4 (control, reproduced to 17 digits) | 2.17662 | 2.54836 | 0.11906 | 984.0 | 0.3285 |
  | B θ=5e6 (correct) | 2.15320 | 2.52961 | **0.01406** | **178.0** | **0.5930** |

  **ΔQA −0.023 / ΔMT −0.019 — an order of magnitude inside the noise floor.** Yet the target became
  **8.5× more fittable** (better on 15/16 docs), the write **5.5× gentler**, and the **B-CASCADE routing
  collapse fully repaired** (0.329 → 0.593 vs Phase-1's 0.588) — the same repair MECH-QUERIES could only
  buy by pumping `n`, and there it came with `|v|` *growing* to 7808.
- ⭐ **Refutes a sub-claim of B-OBJ:** the layer-34/35 residual DIAG-OBJ-c called the binding constraint
  (93–95% of residual, immovable under extra support and zero ridge) **was a rotary artefact** —
  L34 2.3067 → 0.0898 (26×), L35 1.0290 → 0.0075 (138×), share 77.8% → **19.2%**. 13/36 layers are
  individually *worse* under θ=5e6, so the gain is concentrated, not uniform.
- 🔑 **And it STRENGTHENS B-OBJ's headline:** a target now fitted **8.5× better**, with a gentler write and
  healthy routing, **still buys no acquisition**. The objective is not merely mis-specified — even
  correctly specified and well fitted, it does not move CE.
- **Blast radius (listed, not re-run):** CE numbers all stand, but every conclusion argued *from the
  internals* rested on a corrupted target — B-OBJ's "trust region was the binding constraint", the entire
  B-CASCADE/MECH-QUERIES `n`-sweep and its magnitude→routing story, MECH-QUERIES-B (inherits it), **the
  β/NNLS failures** (`compute_teacher_log_mass` shares the wrong frame, so "β is numerically broken" is
  confounded with "β was fitted to a corrupted mass target"), **ORACLE-WRITE's `assign="mass_ranked"`**
  (doc-row→slot pairing chosen on corrupted teacher attention, so **2.381 may UNDER-estimate the
  ceiling**), every post-write routing figure, and K-GATE/K-SUPPORT/K-RIDGE/EXP-001…009.
  **Unaffected:** DIAG-ROUTING and DIAG-WIRE.
- **Canonical config should move to θ=5e6** — strictly better on every internal metric, marginally better
  on both CE axes, and correct. MECH-003 is default-off; flipping the default is a separate decision.
- **Related hazards found in the same read (queued, not yet acted on):**
  H2 `key_select.py:283` installs a doc key into a cartridge slot **with no counter-rotation**, while
  `phase1.py::_rope_reposition` implements exactly that and `initial_am_compaction.py` already calls it.
  H4 🔴 `_should_fit_beta` (`finetune.py:264-269`) **silently enables the broken β/NNLS path whenever
  `KEY_MODE != freeze`** with `ENABLE_BETA` unset — so the *first* key experiment would crash for a
  B-SOLVE reason and look like "keys don't work". **`ENABLE_BETA=0` gives a clean keys-only arm.**

## 📏 D0-ICL (2026-07-29): the ruler was fine — and **the "ceiling" is not a ceiling**
- **The harness suspicion is refuted.** Re-measured on `eval_forgetting.py EVAL_MODE=icl`, all four cells
  match the board's prior numbers to **~5e-05** — four orders of magnitude below the noise band:
  `icl_QA|QA` **1.97341**, `icl_MT|MT` **1.89605**, `icl_QA|MT` 2.77610, `icl_MT|QA` 2.90249.
  The worker verified *why* rather than trusting the docstring: it counted denominators directly
  (`LossEvalDataset(packed_seq_length=2048)` yields **2619 QA / 2562 MT** scored entries, exactly the
  `num_target_tokens` the ICL path reports). Same tokens, same formula, same files; only the forward pass
  differs. **The ICL numbers were correctly rulered all along.**
- 🔴 **But the ordering the mission assumed is wrong.** On **QA**, ICL 1.97341 is **0.3565 worse** than
  the sparse-gradient reference (1.6169) — far outside noise. On **MT**, ICL 1.89605 is **0.0235 above
  (worse than) the dense@4ep bar** (1.87250) — inside the band, so honestly a tie, but the point estimate
  puts the "ceiling" **behind** the bar. **The mission has been reaching toward a line that two existing
  methods already sit at or past.** ICL is a *reference point*, not an upper bound.
- ⭐ **And it rehabilitates DIAG-CONTENT's finding rather than leaving it damning.** Off-diagonals bound
  the topic-general component: wrong-topic context costs +0.880/+0.929, so ICL's gain *is* largely
  content-specific — **but topic-mismatched papers still buy 1.00645 of the 1.88650 total MT drop, i.e.
  53.3% of ICL's own MT gain survives a deliberate topic mismatch.** Full-context ICL is not exempt from
  the content-free-gain effect. **Our 28–73% therefore sits inside the range the gold-standard method
  itself exhibits** — a property of this benchmark, not a defect unique to AM.
- ⚠️ **Repo gap worth fixing:** `eval_forgetting.py`'s ICL branch **never initialises wandb** (`_run_icl`
  only prints). The worker logged the cells via a parser under the no-edit rule; the losses are the
  harness's own, not recomputed. `EVAL_MODE=icl` also does **not** hang, unlike the cartridge path (§1).

## 🥇 NEW BEST GRADIENT-FREE POINT — MECH-KEYS (2026-07-29): **QA 2.0349 / MT 2.3305**
**The "keys collapse QA" folklore is FALSE in this setting. Every key arm beat the frozen control on
BOTH axes — there is no trade curve to report, because QA *improved*.**

| arm (θ=5e6, β off, top32) | QA | MT | `mass_on_S` MT | **MT/QA ratio** | doc keys/32 | solve_s |
|---|---|---|---|---|---|---|
| control `freeze` | 2.1597 | 2.5296 | 0.0711 | 1.0428 | — | 182 |
| keys, no reposition | 2.1201 | 2.4239 | 0.3756 | 1.0485 | 8.91 (27.8%) | 325 |
| **keys + reposition** | **2.0349** | **2.3305** | 0.2430 | 1.0510 | 12.91 (40.3%) | 272 |
| `omp` + reposition | 2.1169 | 2.3970 | 0.1918 | 1.0638 | 17.86 (55.8%) | **3857** |

- **MT 2.3305 is below ORACLE-WRITE's *perfect value write* ceiling (2.381) and below the k=12 value-only
  minimum (2.435)** — gradient-free. **QA clears its budget by 0.49.** Still **0.31 above** the 2.02 bar.
- 🔑 **It does NOT work through selectivity — DIAG-KEYSPACE's bound holds.** The MT/QA mass ratio moved
  only **1.0428 → 1.0510** (+0.0082, the same order as β's +0.008) and remains **below the untouched
  Phase-1 cartridge's own 1.0812**. What moved is **bandwidth whose content is the document**:
  `mass_on_S` 0.0711 → 0.2430.
- ⚠️ **This forces a correction to this board's "bandwidth doesn't predict MT".** β raised bandwidth to
  0.30 and made both axes *worse*; keys raised it to 0.243 and made both axes *better*. The distinction
  is **what the mass lands on**: β amplifies slots holding *solved values*, key installation puts mass on
  slots **keyed by the document itself**. Bandwidth-to-the-right-content is a real lever; generic
  bandwidth is not.
- **Not a support-size artefact:** the no-reposition arm writes a union of **51.6 slots/layer — smaller
  than the control's 53.4 — and still carries 5.3× the mass**.
- **The H2 fix was worth −0.093 MT / −0.085 QA** on top of the uncorrected key write, and it makes
  document keys win selection 40.3% vs 27.8% of the time. Unit check: repositioned logit error **4.8e-07**
  vs **1.054 uncorrected**; doing the same rotation at θ=1e4 gives errors of 4.4–5.1, **worse than not
  correcting at all** — which is why the driver refuses the flag without an explicit `AM_ROPE_THETA`.
- **OMP is dominated:** most document keys installed and the best ratio, but worse CE at **21× the solve
  cost** (a 200-iteration NNLS inside each of 32 greedy steps).
- ⚠️ **Noise:** ΔMT −0.199 sits at the **top edge** of the 0.1–0.2 band and ΔQA −0.125 sits **inside** it.
  **One seed, no variation.** This is the new best point *and* it needs verification before it is quoted.
- **Not yet done:** the 16 per-document snapshots exist for all four arms but were **not** evaluated
  (32 evals/arm). The value-only curve bottoms at k=12 and then *degrades by 0.117* — so the keys arm's
  own minimum is unmeasured and may be materially better than its k=16 endpoint. → DIAG-KEYCURVE.

## 🧭 SYNTHESIS (2026-07-29): the write is not performing content acquisition at all
Three routes were open. All three are now closed, and they converge on one account:

| route | verdict | decisive evidence |
|---|---|---|
| **values** | closed | perfect teacher values reach only MT 2.381; β's 4.23× bandwidth made **both axes worse**; bandwidth doesn't predict MT (13% spread vs 0.882 MT span, highest-bandwidth arm worst) |
| **keys / selectivity** | closed | ρ_key **at its held-out control floor everywhere**; QA/MT query separation **2.5× smaller than sampling noise**, 0/288 heads otherwise; optimal key buys 1.22 vs incumbent 1.083 |
| **allocation / capacity** | closed | a solo write into **uncontested** slots is **+1.101 worse** on its own document than writing it among 16, 5/5 documents; **71–93%** of a solo write's gain lands on *other* documents' questions |

**The account:** the closed-form write is not storing retrievable per-document content. It is producing a
**cumulative, largely document-agnostic adaptation** to the MT distribution. Everything on the board
follows from that single fact:
- why **28.3–73.2%** of the MT gain is reproduced by documents with **zero MT content** (DIAG-CONTENT);
- why **writing a paper in makes the model worse at that paper** (arm B degraded QA by +0.652 while
  rewriting the exact 16 papers the QA eval scores);
- why the curve **saturates at k≈12 and then degrades** — an adaptation saturates, a store would not;
- why **sequence alone is worth 0.340** and per-document survival of 4.7% costs almost nothing;
- why **better fitting is anti-correlated with CE**, and why a *perfect* value write closes only 25%.

**Consequence for the mission:** MT ≤ 2.02 is not reachable by improving *what* or *where* we write,
because the write's benefit is not localised in either. The remaining escape hatches are the two
DIAG-KEYSPACE explicitly did not bound — **mechanisms that change the query distribution** (the
cross-layer effect gradient descent exploits and a per-layer closed-form solve cannot: SCOUT-AM's
divergence #3, the paper's **on-policy layer-sequential re-extraction**, which we have never
implemented) — and **non-fixed-slot formulations** (out of scope; fixed size is the premise).

## ACTIVE BOTTLENECK — **B-ROUTE**, and the mission now turns on one number
**Status after cycle 1 (2026-07-28): three of six entries are closed and the search space has collapsed
onto a single axis.**

| entry | verdict |
|---|---|
| **B-WIRE** | ✅ CLOSED — confirmed bug; `target_mode` never reached the write path; HYP-T1 retracted |
| **B-OBJ** | ✅ CLOSED — **MSE and CE are anti-correlated**; no better fit of this objective can win |
| **B-CASCADE** | measured — the solve's own magnitude (\|v\| 984→21504) re-routes attention away from the cartridge |
| **B-ROUTE** | ⭐ **ACTIVE** — value-only writing is bandwidth-capped; a *perfect* write at 9% mass reaches only MT 2.381 |
| **B-CAP / B-SOLVE** | re-opened / root-caused, but subordinate to B-ROUTE |

**The decisive question, in flight (ORACLE-WRITE-512):** a perfect value write at **9%** bandwidth gave
MT 2.381. At `top_t=512` the bandwidth is **59%** (6.6×). If MT falls toward ~1.9, value-only writing was
bandwidth-limited and a gradient-free win is live. **If it stalls at ~2.3–2.4 with perfect values and
6.6× the bandwidth, then no value-only write can ever win** — and the mission's answer must be
**key-side**, which is the one axis no experiment in this loop has ever touched (`KEY_MODE=freeze` in
every row of `results.csv`).

**Why keys are still live despite DIAG-ROUTING's ρ→0 result:** that measured the **routing simplex**
(post-softmax, 512-dim). The **key/query space** is a different object — 128-dim per head, pre-softmax.
A new key placed orthogonal to the span of QA's queries would take ≈0 QA attention while remaining
reachable by MT queries. Nothing has measured that geometry yet. → SCOUT-KEYS dispatched.

---
## The gap under investigation
Best gradient-free point (AM top32): **QA 2.1766 / MT 2.5484**, 0 gradient steps.
The bar (dense @4ep): **QA 2.3721 / MT 1.8725**. Win needs **QA ≤ 2.52 AND MT ≤ 2.02**.
⇒ **Retention is already ahead of the bar. The entire deficit is acquisition: MT −0.53 to go.**
Eliminated as *knob-level* explanations: gating (HYP-G1), support (HYP-S1), ridge (HYP-R0),
`target_mode` (HYP-T1 — but see B-WIRE, the null is suspicious).

---
## B-OBJ — objective mismatch
- **Stage:** CLOSED · **Status:** ✅ **CONFIRMED — and stronger than the hypothesis: MSE and CE are
  ANTI-CORRELATED** (DIAG-OBJ-c, 2026-07-28)
- **Mechanistic sentence:** removing the trust region (`DELTA_WEIGHT=0`, single variable vs DIAG-OBJ-b)
  drove the solve's own objective 18.2× better — mean `am/mean_mse` 0.09155 → **0.005037**, improving on
  **all 16/16 documents**, the closest to MSE→0 this objective has ever come — and **destroyed the
  model**: MT 2.663 → **15.95**, QA 2.513 → **16.01** (ppl ≈ 8e6), ~70× the noise band. `|v|` went
  848 → **21504**, with **no NaN/Inf anywhere** and rc=0. The failure is **magnitude, not overflow**.
  ⇒ minimising value/attention-space reconstruction on the reference queries is not merely *decoupled*
  from token CE; over this range it is **anti-correlated**. **No better fit of this objective can win.**
- ⭐ **And it identified what was holding the residual:** layers 34/35 held 93–95% of the residual and
  refused to shrink under both extra support and zero ridge. Removing the trust region collapsed them
  (L34 3.8e-05×, L35 1.6e-05×; their residual share 95.44% → **0.43%**). **The trust region — not
  `top_t`, not ridge — was the binding constraint on the fit.** The knob the loop never swept was the
  one doing all the work.
- **Consequence for the mission:** the five queued *value-solve improvement* mechanisms are now
  deprioritised. What matters is the **metric/trust region** (→ MECH-METRIC: MEMIT's `C₀` from the old
  keys' second moment instead of `w·I`) and **routing/mass** (→ B-ROUTE, β), not fit quality.
- **Superseded intermediate finding (kept for provenance):**
- ✅ **DIAG-OBJ (2026-07-28):** 16× more support (`top_t` 32 → all 511) plus zero ridge bought a
  **23.1% MSE reduction** (0.11906 → 0.09155, uniform: per-doc ratio 0.62–0.85 across all 16 docs) and
  **MT moved the wrong way, +0.117** (2.5484 → 2.6652). QA +0.339 as coverage predicts.
  **Fitting the internal objective materially better bought zero CE.** (MT's +0.117 sits at the edge of
  the ±0.1–0.2 noise band, so read it as "MT did not improve", not a confirmed regression; the MSE drop
  and the QA change are outside the band.) Run (a) reproduced EXP-007-top32 **bit-identically**
  (QA 2.1766 / MT 2.5484), so the harness is sound.
- ⚠️ **Not yet a true MSE→0 test.** `DELTA_WEIGHT=1e-2` appends a 511×511 identity block to the stacked
  design in `guarded_sparse_am_value_update`, keeping the system over-determined and shrinking the
  solution toward the prior even with every slot free. **The decisive run is `DELTA_WEIGHT=0` +
  `TOP_T=511` + `RIDGE_LAMBDA=0`** — env-only, no code edit (dispatched as DIAG-OBJ-c).
- ⭐ **Structural signal worth its own board entry:** **93–95% of the residual sits in layers 34–35
  alone** (4.40 and 2.23 on the last doc), and those two layers *refuse to shrink* (0.93× / 0.77×) while
  the other 34 layers drop 44.5% and are already tiny (median ≈0.0027). **The solve is already at its
  own optimum almost everywhere.** So the sharp question is no longer "is MSE low?" but "**why can't
  layers 34–35 be fitted, and is that where CE lives?**" Per-layer support/query allocation is the
  obvious follow-up once `max_queries_per_head` is unpinned (MECH-QUERIES).
- **Prior claim (superseded):** "suspected"
- **Claim:** the solve minimizes value/attention-space MSE, but we are scored on token cross-entropy.
  A numerically excellent solve can leave CE almost untouched — the two objectives are only loosely
  coupled through the frozen LM.
- **Signature to measure:** achieved solve MSE (per layer) vs realized ΔCE on MT. Also: is
  `am/mean_mse` already small at the operating point?
- **Oracle:** drive MSE→~0 (unbounded support, `RIDGE_LAMBDA=0`) and read MT. **MSE→0 with flat MT is
  the strongest possible confirmation.**
- **If confirmed:** the fix is an objective change (output/logit-space weighting, Gauss-Newton-style
  preconditioning) — SCOUT direction, MISSION §5.
- **Evidence so far:** none. `am/mean_mse` is logged to wandb but has never been read against CE.

## 🧪 B-CONTENT — how much of "acquisition" is content at all? (DIAG-CONTENT, 2026-07-28)
- **Stage:** CLOSED · **Status:** **the acquisition claim SURVIVES, but damaged — and the answer is
  k-dependent**
- **Design:** arm B writes the **QA-topic corpus** (no MT content by construction; exactly 16 unique
  documents, so per-k curves are directly comparable) and evaluates MT. Arm C writes the canonical
  documents in **reversed order** (content held exactly constant, verified by slug).

  | k | A canonical | B content-free | C reversed | ΔB/ΔA | ΔC/ΔA |
  |---|---|---|---|---|---|
  | 4 | 3.0338 | 3.2659 | 3.1886 | 0.690 | 0.793 |
  | 8 | 2.7099 | **2.9978** ← B's min | 2.8801 | **0.732** | 0.841 |
  | 12 | **2.4352** | 3.0436 | 2.7166 | 0.548 | 0.791 |
  | 16 | 2.5524 | 3.4343 | 2.8920 | **0.283** | 0.724 |

- **The split, stated plainly:** of the canonical 1.230-loss MT improvement, **28.3% is content-free at
  the reported k=16 point**, rising to **73.2% at matched k=8**. All far outside the ±0.15 band.
  **But content is also real:** canonical beats content-free by 0.232 / 0.288 / 0.608 / 0.882 at
  k=4/8/12/16 — every one outside the band, and **widening with k**. Neither branch; the middle.
- ✅ **The confound was refuted by measurement, not argued away.** Writing the QA corpus back into the
  cartridge it built is **not** a no-op: arm B displaces **69.9%** of the Phase-1 value Frobenius norm
  (canonical 62.1%), changes **more** slots (2195 vs 1960), and is the harshest of the three writes.
- 🔴 **And it produces an outright anomaly:** arm B **degrades QA by +0.652** above the floor at k=16
  **while re-writing the exact 16 papers the QA eval scores** (16/16 title overlap, verified).
  **Writing a paper in makes the model worse at that paper.** That is not a retrieval story at all.
- **Sequence alone is worth 0.340** (arm C, same documents reversed) — comparable in size to arm B's
  entire content-free gain, and consistent with the recency/overwrite that B-OVERWRITE measured.
- ⚠️ **Strongest surviving objection (worker-stated):** arm B's documents are QASPER papers in the **same
  synthesis format**, differing only in topic — so "content-free" here means "**topic**-free within one
  corpus format" and does not separate a generic activation perturbation from transfer of shared
  scientific-paper structure. Arm A's per-k curve is reused from DIAG-SEQUENCE, not re-measured. Single
  seed; k=1 deltas are inside the band for all three arms.

## ⚠️ TWO INTERNALS FINDINGS THAT CUT AGAINST THIS BOARD'S OWN FRAMING (DIAG-CONTENT)
1. **Bandwidth does not predict MT — final nail.** All three arms sit within **0.079–0.091** eval-time
   `mass_on_S` (a **13% spread**) while MT spans **0.882**. The arm with the **highest** bandwidth (B,
   0.0906) has the **worst** MT. Together with MECH-BETA (4.23× bandwidth → both axes worse), the
   bandwidth hypothesis is dead in both directions.
2. **The B-CASCADE "routing collapse" is specific to the canonical arm, and is ANTI-ordered with MT.**
   Total cartridge mass: **0.329 (A, best MT)** vs 0.655 (B, worst MT) / 0.591 (C) against Phase-1's
   0.588. The arm with healthy routing has the worst acquisition. **Collapse is not the pathology this
   board took it for** — see the B-CASCADE entry, whose framing this contradicts.

## 🔥 B-CASCADE — the solve destroys the routing it depends on
- **Stage:** CLOSED (with one confound outstanding) · **Status:** ❌ **REFUTED as an acquisition
  explanation — but half of it was right** (MECH-QUERIES, 2026-07-28)
- **The prediction was:** raising `n ≫ t` shrinks `|v|`, restores cartridge mass, **and** improves MT —
  all three together. Result across `n ∈ {64, 256, 1024, 4096, 16384}` at `top_t=32`:

  | n | QA | MT | \|v\|max | cartridge mass (MT) |
  |---|---|---|---|---|
  | 64 (control) | **2.1772** | **2.5524** | 984 | 0.3285 |
  | 256 | 2.4097 | 2.7716 | 1328 | 0.3715 |
  | 1024 | 2.2575 | 2.5744 | 1968 | 0.6399 |
  | 4096 | 2.3412 | 2.7114 | 3344 | 0.6395 |
  | 16384 | 2.3325 | 2.6943 | 7808 | 0.6388 |

  **The three quantities decoupled.** ✅ The routing collapse **is repaired and overshoots** (0.329 →
  0.640, past Phase-1's 0.588). ❌ `|v|` **grows** 984 → 7808, *away* from the teacher's |63|.
  ❌ **MT never improves** — the best arm is the `n=64` control; n=1024's +0.022 is inside noise.
- **Mechanism for why `|v|` grew (worker-derived, reproduced at unit level):** the guarded solve stacks
  `[X_new (n×t); √w·I (t×t)]`, giving `(X_newᵀX_new + w·I)V = X_newᵀR + w·V_old`. The **data Gram scales
  with `n` while the trust region stays fixed at `t` rows**, so its relative pull decays like **1/n**.
  The spectral ridge (λ ∝ σ_max(X)²) is scale-invariant and does not compensate.
- ✅ **CONFOUND RESOLVED (MECH-QUERIES-B, 2026-07-28): query count is genuinely NOT an acquisition lever.**
  With `w(n) = 1e-2·n/64` holding the trust region's relative pull constant:

  | arm | QA | MT | \|v\|max | \|ΔV\| rel-Fro | cart mass (MT) | eval `mass_on_S` (MT) |
  |---|---|---|---|---|---|---|
  | n=64, w=1e-2 (control) | 2.1772 | **2.5524** | 984 | 0.621 | 0.3285 | 0.0810 |
  | n=1024, w=0.16 | **2.0401** | 2.5986 | 696 | 0.526 | 0.6384 | 0.0878 |
  | n=16384, w=2.56 | **2.0263** | 2.6084 | 656 | 0.503 | 0.6362 | 0.0886 |

  MT +0.046/+0.056 — inside the noise band **and on the wrong side of it**. `|v|` behaved exactly as the
  derivation predicted (7808 → 656, 12×, landing *below* the control). **The "near-trivial write"
  objection is closed, not assumed away:** at w=2.56 the write still displaces **50.3%** of the Phase-1
  value tensor's Frobenius norm and changes **1974 slots — more than the control's 1960**.
- 🔑 **Two findings that matter more than the null:**
  1. **QA is the only axis that moved, and it moved a lot: 2.1772 → 2.0263** — monotone in `n`, same sign
     on both arms, and **0.21 *below the untouched Phase-1 floor* (2.2388)**. We now hold **0.49 of QA
     slack** against the 2.52 budget. Caveat: at the upper edge of noise, not re-run or seed-varied.
  2. **Bandwidth rose and MT still did not move.** Eval `mass_on_S` climbed toward Phase-1 for the first
     time (0.0810 → 0.0886 vs 0.0896) — the written slots are read *more* than in any prior arm — with
     no MT response. Routing recovery is driven by `n` alone, cleanly decoupled from `|v|` (cartridge
     mass 0.636–0.638 at both 656 and 7808).
- **Also established:** the accumulator can supply **57,344–81,920 real queries per KV-head per
  document** — the hard-coded 64 was discarding **~99.9%** of them — and the cost is **flat**: 256× the
  queries for **1.23×** wall clock. This is a *quality*-dominated point, not a cost-dominated one.
- **Superseded framing below (kept for provenance):**
- **The observation:** our **solved** write reaches `|v|` up to **984** and **collapses total cartridge
  attention from 0.588 → 0.329 on MT** (0.570 → 0.320 on QA), pushing 10/36 layers under 5% mass on S.
  The **oracle's teacher values top out at |63|** — 15× smaller — and leave cartridge mass **at Phase-1
  level**. The closed-form solve is making the model route away from *the cartridge as a whole*.
- **Mechanism (why values can move attention at all):** the solve is **per-layer and independent**, but
  the model is **sequential**. Extreme values written at layer *l* perturb the residual stream, which
  shifts the **queries** at layers *l+1…35*, which re-routes attention away from the cartridge. We
  optimise each layer against queries collected from an activation distribution our own write destroys.
- **This converges with two independent findings:** (i) SCOUT-AM's divergence #3 — the paper does
  **on-policy, layer-sequential re-extraction** of reference queries and we do not; (ii) SCOUT-AM's
  divergence #2 — `max_queries_per_head=64` at `top_t=64` is an **exactly-determined** system, i.e.
  exact interpolation with no norm control, which is precisely how you get `|v| = 984`.
- **Prediction (sharp, falsifiable):** raising the reference-query count `n ≫ t` should shrink `|v|`
  toward the teacher's scale, restore cartridge mass toward 0.588, and improve MT — **all three
  together**. If `|v|` shrinks but MT does not move, B-CASCADE is refuted and the cap is elsewhere.
- **Also explains** why `DELTA_WEIGHT` (the trust region = MEMIT's `C₀ = w·I`) is load-bearing: it is
  the *only* thing currently restraining the write's magnitude, and it has never been swept.
- ✅ **CONFIRMED BY DIAG-OBJ-c (2026-07-28):** with the trust region removed, `|v|` reaches **21504** and
  both losses explode to ~16 (ppl 8e6) — with **no NaN/Inf**. Magnitude alone destroys the model. So the
  causal chain is established: *fewer reference rows than columns → exact interpolation → enormous
  values → the residual stream is perturbed → later layers' queries move → attention routes away from
  the cartridge → CE collapses.* `DELTA_WEIGHT` is the only brake, and it is a **crude** one (`C₀ = w·I`
  penalises every direction equally). **This is the strongest argument for MECH-METRIC:** replace the
  isotropic brake with MEMIT's `C₀` = second moment of the old routing vectors, so the write is
  restrained *in the directions the old content occupies* and free elsewhere.

## 🏁🏁 B-ROUTE (KEY SIDE) — CLOSED: **there is no task separation to exploit** (DIAG-KEYSPACE, 2026-07-29)
- **Stage:** CLOSED · **Status:** ✅ **confirmed + capped.** Selectivity is not geometrically available
  on the key side either. **Both sides of the write are now closed.**
- **ρ_key sits at its own control floor at every threshold** (control = eigenbasis from one half of the
  QA *documents*, energy from the disjoint other half — a document-level split, never token-level):

  | r | 4 | 8 | 16 | 32 | 64 |
  |---|---|---|---|---|---|
  | ρ_key (MT) | 0.4117 | 0.3398 | 0.2609 | 0.1711 | 0.0751 |
  | QA held-out control | 0.4033 | 0.3283 | 0.2514 | 0.1657 | 0.0741 |
  | ratio | 1.021 | 1.035 | 1.038 | **1.032** | 1.014 |

  At the GPM 99%-energy rule MT is **below** its floor (0.0113 vs 0.0118). **No head anywhere exceeds
  its control by >0.044.** The raw 0.171 lands inside LIT-020's live band [0.15, 0.6] **only because the
  floor is 0.166** — without the control this would have been read as a green light.
- ⭐ **The key-blindness premise was RIGHT; its consequence is REFUTED.** `Q₀` participation effective
  rank is **5.09 (QA) / 5.30 (MT)** — 106–107 of 128 dims to 99% energy — versus the routing Gram's
  **1.97**, so the incumbent keys really do resolve far less than the queries contain. But **QA and MT
  occupy the same subspace**: mean principal angle **4.3°** at r=1, 10–14° for r=4…32, traces agreeing
  to 0.3%. There was never a hidden separation for better keys to expose.
- **Mean-query separation is smaller than sampling noise:** ‖q̄_MT − q̄_QA‖/‖q̄_MT‖ = **0.0700** against a
  QA split-half control of **0.174** — the between-task gap is **2.5× smaller than the within-QA
  sampling gap**, and **0 of 288 heads** run the other way (cosine 0.99727). Prediction was ≥0.30.
- **Achievable selectivity bound (held-out documents):** an *optimally placed* extra key buys MT/QA
  **1.2245** at comparable bandwidth — **1.19× above the same-task floor** — against an incumbent 1.083.
  At β's bandwidth it is 1.080 vs a 1.034 control. A random key gives 1.011; the `q̄_MT − q̄_QA`
  direction gives **0.841, worse than random** (Δμ is noise-dominated).
- 🔬 **The worker caught its own overfitting, which is why this number is credible.** Free-form
  optimisation reached in-sample ratios of 2.2–166 but **held-out 0.885–1.12**, with the QA-vs-QA
  control reaching the *same* fit ratio (2.94 vs 2.82). Its first pass (superseded run `kx4hsp5t`)
  reported per-head ratios up to **1e26** before the held-out split was added. The reported bound comes
  from the **closed-form** construction that the optimiser fails to beat.
- **What this does NOT rule out (the escalation surface):** mechanisms that **change the query
  distribution**, and **non-fixed-512-slot** formulations. Also untouched: **allocation/capacity**
  mechanisms, which are not selectivity mechanisms and are therefore *not* bounded by this result.
- **Cross-validated:** independently reproduces DIAG-ROUTING's cartridge mass to 5–6 s.f. and the
  tf-idf-union ratio 1.083; ρ_key identical to 6 s.f. across two processes.

## 🏁 B-ROUTE (VALUE SIDE) — CLOSED: the write is NOT bandwidth-limited, and the value-only family is EXHAUSTED
- **Stage:** CLOSED · **Status:** ✅ **confirmed + capped** (MECH-BETA, 2026-07-28)
- **The decisive test.** β (AM's own mass-matching mechanism) **ran clean for the first time in this
  project** — 288 fits/doc × 16 docs, **0 non-finite warm starts, 0 raised**, every β inside [−3,3].
  It did exactly what it was supposed to do to bandwidth, and **both axes got worse**:

  | arm | QA | MT | eval `mass_on_S` (MT) | MT/QA ratio | `am/mean_mse` |
  |---|---|---|---|---|---|
  | control θ=1e4, β off | 2.1766 | 2.5484 | 0.0810 | 1.047 | 0.11906 |
  | rope θ=5e6, β off | 2.1532 | 2.5296 | 0.0711 | 1.043 | 0.01406 |
  | **β on, box 3** | **2.7364** | **2.8119** | **0.3010 (4.23×)** | **1.051** | 0.00782 |

  **`mass_on_S` rose 4.23× on all 36 layers, past Phase-1's 0.0896 — and MT regressed +0.282 while QA
  regressed +0.581, past the 2.52 budget.**
- **Mechanistic sentence:** β is **query-independent**, so raising mass on the written slots raises it for
  **QA queries too**. SCOUT-KEYS predicted the ratio would move only 1.03→1.05; measured **1.043 →
  1.051** — a 4.2× bandwidth gain bought **+0.008 of selectivity**. Bandwidth is not the constraint;
  **selectivity is, and no query-independent operator can supply it.**
- **What this closes, together with the rest of the board.** Every value-side lever has now been tried at
  the canonical point, and none reaches MT ≤ 2.02:
  | lever | result |
  |---|---|
  | perfect values (teacher's own KV) | MT 2.381 — closes 25% of the gap |
  | 4.2× bandwidth (β) | MT 2.812 — **worse** |
  | 256× reference queries | MT 2.552–2.608 — no effect |
  | better fit (ridge/trust region off) | MT 15.95 — catastrophically worse |
  | correct rotary base | ΔMT −0.019 — inside noise |
  | more support (`top_t` 64→511) | MT 2.55–2.67 — worse |
  | target mode | was a wiring no-op |
  ⇒ **A gradient-free win, if it exists, is not on the value side.** The best gradient-free point remains
  QA 2.042 / MT 2.435 (k=12 of the canonical run) against a target of QA ≤ 2.52 / **MT ≤ 2.02**.
- **Two build corrections worth keeping** (MECH-004): `torch.linalg.lstsq(driver='gelsd')` is **CPU-only
  and raises on CUDA** — the old bare `except RuntimeError` would have silently degraded to a uniform
  warm start; a CPU retry costs 1.31× `solve_s` on 64×32 matrices. And LIT-002's divergence #3
  (residual-target clamp) is a **non-issue at `KEY_MODE=freeze`** — the target is strictly positive there,
  measured clamp fraction 5.4e−5. β saturates upward: **59.7% of entries pinned at the +3 ceiling.**
- ✅ **H4 hazard fixed:** `_should_fit_beta` with `highest_attention` + `ENABLE_BETA` unset now returns
  **False** (it returned True). **A clean keys-only arm is now possible for the first time.**

## B-ROUTE (earlier stage, superseded above)
- **Stage:** E VERIFY (one control outstanding) · **Status:** ✅ **MEASURED — partially confirmed, and
  the value-only family is BOUNDED**
- ✅ **ORACLE-WRITE (2026-07-28), the decisive run — neither branch of the predicted dichotomy:**

  | | QA | MT |
  |---|---|---|
  | Phase-1 start | 2.2388 | 3.7825 |
  | control (solved values, flag off) | 2.1772 | 2.5524 |
  | **oracle (teacher's own doc values)** | **1.8955** | **2.3810** |
  | dense@4ep bar | 2.3721 | 1.8725 |

  A **perfect content transplant** into the tfidf-selected top-32 slots moves MT only 2.552 → 2.381
  (−0.171, at the edge of the noise band) — **closing just 25% of the gap, leaving 0.509 to the bar** —
  while QA improves −0.282 (outside noise).
- **Mechanistic sentence:** the rewritten slots are **readable through a narrow channel, not
  unreadable**. Under the full eval-time softmax they carry **~9% of total attention** (~15.5% of
  cartridge mass), and **MT queries route to them no more than QA queries do (ratio 1.04)**. So
  value-only writing at this support and selection is capped by **bandwidth**, not by zero routing.
- **What this bounds:** at `top_t=32` with frozen keys and tfidf selection, **no value-only write of any
  kind — however perfect — reaches MT ≤ 2.02.** Closing the gap requires *more mass on S* (β /
  mass-matching), *different keys*, or *more/better-selected support* — not a better value solve.
- **Bit-identical-when-off verified**, and the flag-off control reproduced EXP-007-top32 to all 16
  printed digits in a fresh process, which **also retires the sibling-import confound** on that number.
- ⚠️ **Outstanding control (W5, not yet run):** the oracle changes value **magnitude** (984 → 63) as
  well as content, so part of both gains could be a magnitude/entropy effect rather than the document
  content being read. A **norm-matched shuffled/random-value control** separates them. Until it runs,
  read the oracle as an upper bound whose *cause* is not yet attributed.
- **Superseded framing below (kept for provenance):**
- **Claim:** with `KEY_MODE=freeze`, eval-time MT queries may barely attend to the rewritten slots.
  Whatever we write is then unreadable, and no improvement to *what* we write can matter.
- **Signature to measure:** attention mass on rewritten slots at eval time, per layer, MT queries vs
  QA queries; compare against the mass those slots carried before the write.
- **Oracle (the decisive run):** **write-ceiling** — put the *teacher's own* KV for the new doc into
  the selected slots and eval. If MT stays ~2.5, **no value-only frozen-key write can ever win**, and
  the mission's answer must involve keys or a different write rule.
- **If confirmed:** key-side mechanisms (`KEY_MODE ∈ {highest_attention, omp}`, delta-rule /
  associative writes where the key is chosen so the write is retrievable, null-space key placement).
- **Evidence so far:** no direct measurement yet (ORACLE-WRITE in flight). **Every run in `results.csv`
  used `KEY_MODE=freeze`** — the prior "keys collapse QA" is pre-loop, Llama-era, and *untested here*.
- ✅ **THEORETICAL SUPPORT FROM THE PAPER ITSELF (SCOUT-AM, LIT-001).** App. A.2 *"Why biases matter"*:
  with subset keys and no bias, `Mass(q;Ck) ≤ Mass(q;K)` **for every q**, so the block "systematically
  receives too little global weight" once concatenated with future tokens. That is the paper's own
  account of a frozen-key routing deficit — **and β is its designed remedy.** Our canonical config has
  β off (forced by `KEY_MODE=freeze`) and keys frozen, i.e. the deficit with the remedy disabled.
- ⚠️ **AM IS NOT DESIGNED FOR OUR REGIME.** `Ck` is *always* a subset of the keys **of the block being
  compacted** (§3.3), so solved values sit on keys that already route for that content — which is why
  the paper's "no learned values" ablation still works. There is **no procedure in the paper** for
  writing into keys fitted to *different* content. §6 names our exact setting as future work ("move
  away from subset selection for `Ck`"; "architectures that explicitly operate over a **fixed set of
  keys and values**"), and its only multi-shot setting (App. F.3) **re-compacts the entire cache
  including previously compacted portions, explicitly rather than freezing them**.
  ⇒ **The mechanism we need is not in this paper.** It must be imported (→ SCOUT-EDIT: null-space
  editing, delta-rule associative writes, MEMIT-style multi-edit) or designed here. This is the
  novelty budget, and it is now a *measured* need rather than an aspiration.
- ✅ **THREE INDEPENDENT LITERATURES STATE B-ROUTE IDENTICALLY (SCOUT-EDIT, LIT-013/016).** Softmax
  attention as *nonparametric* regression (arXiv 2501.12352): a new association requires a new key.
  Modern Hopfield: retrievability is set by key **separation**; poorly separated keys return
  metastable averages. Fast weights: interference is key **overlap**. All three bound the write by
  `mass_on_S`.
- 🔥 **`mass_on_S` IS ALREADY COMPUTED IN OUR CODE AND NO BUNDLE HAS EVER RECORDED IT**
  (`value_solve.py:146`, `finetune.py:628`). The single number that bounds every value-only mechanism
  has been sitting in the solve the whole time. → now a standing bundle requirement (WORKERS.md).
- **Most promising imported operator (LIT-017):** null-space **key** placement — the only candidate
  that attacks routing and retention *together*, and the only one that escapes the nonparametric
  bound. Depends on β working (→ B-SOLVE's three named fixes).

## ⭐ CROSS-CUTTING FINDING (SCOUT-EDIT, 2026-07-28): our write is **already MEMIT — with the wrong metric**
`continual_am_sparse.py:96` defaults `DELTA_WEIGHT=1e-2`, **overriding the `0.0` dataclass default**
at `finetune.py:87`. So `finetune.py:580` has routed **every AM row in `results.csv`** through
`guarded_sparse_am_value_update`, which minimises `‖A_new·V_S − R_new‖² + w‖V_S − V_S^old‖²`. That is
**exactly MEMIT's** `Δ = R K₁ᵀ(C₀ + K₁K₁ᵀ)⁻¹` **with `C₀ = w·I`.**

The whole closed-form editing literature reduces to one statement: **`C₀` should be the second moment
of the OLD keys, not the identity.** Our "key" is the simplex routing vector `a_S(q) = alpha[:,S]` —
literally `X` at `value_solve.py:86` — so `C₀` is a `t×t` routing Gram obtainable from **one QA forward
pass** using hooks we already have. This is a small, principled change with a large literature behind
it, and it lands on a knob (`DELTA_WEIGHT`) that **has never been swept** — every row is 1e-2, and
HYP-R0 refuted `RIDGE_LAMBDA`, which is a *different* knob. `DELTA_WEIGHT=0` (DIAG-OBJ-c, in flight)
is therefore the free negative control for this entire family.

## ⚠️ CROSS-CUTTING FINDING (SCOUT-AM, 2026-07-28): we have been running AM's weakest ablation
The paper (`AM.pdf`, arXiv 2602.16284, ICML 2026) fits **two** quantities per layer per KV-head:
the locally-normalised attention **output**, *and* the unnormalised attention **mass**
`Σⱼ exp(q(Ck)ⱼᵀ/√d + βⱼ)`. App. A.2: attention over `[block ; K_fixed]` is a **mixture weighted by the
two blocks' masses**, so **mass IS the routing weight** of the compacted block against all future
tokens. Our canonical config runs **neither** component:
1. **β is silently disabled by `KEY_MODE=freeze`** — `finetune.py:256-261` `_should_fit_beta` returns
   `key_mode != "freeze"`. Every AM result in `results.csv` has β off *by construction*, not by choice.
2. **The solve is starved of reference queries.** `finetune.py:81` hard-codes
   `max_queries_per_head = 64` **with no env knob**. The paper uses **16k–50k queries per KV-head**.
   So at `top_t=64` we solve a **64×64 exactly-determined** system (zero generalisation headroom), and
   at `top_t=128` a **64×128 underdetermined** one that falls into the min-norm branch
   (`core.py:198-204`) — **this explains the board's unexplained EXP-007 top128 MT regression.**
3. **Queries never see the document**: `continual.py:172-182` draws them from the MT synthesis
   conversations against the cartridge; no on-policy re-extraction.
⇒ Several "AM cannot do X" conclusions were really "AM-with-both-mechanisms-off cannot do X".

## B-TARGET — target / reference distribution
- **Stage:** A MEASURE → **promoted; now has a measured mechanism** · **Status:** measured (see above)
- **Claim:** the reference queries the solve fits don't represent the MT eval distribution, so we fit
  the wrong thing well. (`MAX_REF_EXAMPLES_PER_DOC=32`, per-document execution.)
- **Signature to measure:** recon error on the reference queries vs held-out MT queries — the
  generalization gap of the solve.
- **Oracle:** **target-cheat** — build the reference set from the *eval* MT queries. A large MT drop
  ⇒ reference-limited (fixable, cheaply, by better reference construction).
- **If confirmed:** on-policy / self-generated references, coverage-driven selection, `MAX_REF…` sweep
  *with a cost ceiling* (NORTH_STAR: giant banks are drift).
- **Evidence so far:** none directly. HYP-T1's null is not evidence here — see B-WIRE.

## 🧭 DIAG-ROUTING (2026-07-28) — the routing geometry, and what it kills
Forward passes only, on the untouched Phase-1 cartridge, both eval splits, `top_t=512`.
Independent reimplementation of ORACLE-WRITE's convention — the two agree to **4–6 significant
figures** (0.08961 vs 0.08956; 0.58787 vs 0.58787), so both measurements are now cross-validated.

- **(a) The QA routing Gram HAS a large null space at full support.** λ₁/tr = 0.834 (pooled 0.892),
  participation effective rank **1.97**; effective rank of 512 = 6.5 at τ=1e-2, 72.7 at the 99%-energy
  rule ⇒ an approximate null space of **439–505 dimensions**. LIT-011's structural caveat ("P is
  identically zero at t=64") is answered: **the projection family is not vacuous at 512.**
- **(b) But MT routing energy does NOT live in it: ρ_MT ≈ 0.012.** (0.0423 at rank 32, 0.0095 at 128.)
  The in-sample QA leakage control at the same ranks is 0.0096 / 0.0361 / 0.0071 — **MT exceeds QA's own
  spectral tail by only 13–65%**. No layer exceeds ρ_MT = 0.0147; the best single head reaches 0.209.
  **LIT-011's own criterion:** ρ ∈ [0.3, 0.8] ⇒ separable; ρ → 0 ⇒ *"no value-space operator whatsoever
  can separate the two axes."* **We are squarely in the ρ → 0 branch.**
- **QA and MT want the same slots, almost exactly:** mean-routing cosine **0.99899** (min 0.9755 across
  288 heads), histogram intersection **0.963**, top-32 slot overlap **0.914**. LIT-013 named
  intersection < 0.2 as its falsifier — **we measure the opposite extreme.** Routing entropies are
  nearly equal too (QA 1.946 vs MT 2.009 nats ⇒ ~14.4 vs 15.1 effective slots of 512): MT is not more
  diffuse, just *not different*.

**What this rules out (an entire mechanism family, for ~86 s of GPU):**
- **Null-space VALUE projection (LIT-011/012) cannot buy acquisition here.** Writing inside QA's
  approximate null space discards ~99% of MT's routing leverage. It survives **only as a retention
  operator** — which the board suspected but had no number for. MECH-METRIC's `C₀` is re-scoped
  accordingly: a better brake on forgetting, not a source of acquisition.
- **The LIT-013 interference/disjoint-support family is falsified** at the selection level.
- ⚠️ **It also undercuts the project's gating premise:** at 0.914 top-32 overlap, "select the slots the
  new document attends to" ≈ "select the slots QA attends to". A discrete "prefer slots QA is blind to"
  selector has only ~9% of slots to work with. **Gating cannot separate what routing does not.**
- **Still alive:** LIT-017 (null-space **key** placement) is *not* tested by this — it concerns the
  128-dim query second moment `Q₀` and **synthetic** keys, whereas this measured routing vectors over
  frozen Phase-1 keys. Keys remain the one untouched axis.

**~~🔥 The lever this hands us~~ — ⚠️ CORRECTED 2026-07-28 by ORACLE-WRITE-512.** DIAG-ROUTING's
0.5697/0.5879 is **total cartridge mass**, not writable-slot mass (the two agree to 5–6 s.f., so it was
the *label* that was wrong, not the measurement). The **511 writable slots carry only 0.2190 QA /
0.2328 MT**. The difference is **frozen slot 0 — an attention sink holding 0.3506 QA / 0.3550 MT**
(0.624 in one layer), which **no AM config can write**. So the real writable-bandwidth gain of full
support over top-32 is **2.6×, not 6.6×**. The orchestrator propagated the 6.6× figure into two worker
briefs before it was checked; the corrected number stands.

**Caveats (worker-stated):** ρ_QA is in-sample by construction, so a held-out QA control would raise the
floor and *shrink* the MT/QA contrast, not widen it; all numbers are on the pre-write cartridge.

## B-CAP — capacity of the selected support
- **Stage:** A MEASURE · **Status:** weakly measured (indirect)
- **Claim:** the chosen slots cannot represent the new content.
- **Signature:** residual energy / effective rank of the solve system vs `top_t`.
- **Oracle:** `TOP_T=512`, read MT alone, ignore the QA cost.
- **Evidence so far:** HYP-S1 (EXP-007): MT flat across `top_t` 32/64/128 (2.548 / 2.543 / 2.686) while
  QA degrades monotonically.
- ✅ **THE top128 ANOMALY IS EXPLAINED (SCOUT-AM).** With `max_queries_per_head=64` hard-coded, the solve
  is `n=64` rows against `t=top_t` columns: at `top_t=64` it is **exactly determined**, at `top_t=128`
  it is **underdetermined** and falls into the min-norm branch (`core.py:198-204`). The top128
  regression is a *rank* artefact, not a property of support.
- ⚠️ **THIS CONFOUNDS HYP-S1's REFUTATION.** "More support doesn't help acquisition" was measured with
  the query count pinned at 64 — so adding columns could only *dilute* an already exactly-determined
  system. Support must be re-tested **jointly with query count** (`n ≫ t`) before B-CAP or K-SUPPORT
  can be considered closed. Re-opened.
- ✅ **A THIRD, INDEPENDENT REASON TO REVISIT `top_t` (SCOUT-EDIT, LIT-011/012):** null-space **value**
  editing is **identically zero at `top_t=64`** — `C₀ ∈ R^{64×64}` is full rank when `n_old ≫ 64`, so
  the projector `P = 0`. The retention-preserving projection operators only *exist* at large support.
  ⇒ `top_t` is entangled with three separate mechanisms (query count, min-norm rank, null-space
  existence) and cannot be swept as a scalar knob.
- **Note on what projection can and cannot do (LIT-009/011/012):** preconditioning and null-space
  projection are **retention** mechanisms — they cannot move MT on their own. Their value here is
  converting our **0.34 of QA slack** (2.177 vs the 2.52 budget) into usable support *without*
  EXP-007's monotone QA cost.

## B-SOLVE — numerics
- **Stage:** A MEASURE · **Status:** partially measured
- **Claim:** ill-conditioning / the broken β path means we don't reach our own optimum.
- **Signature:** condition number and spectrum of the query Gram; β finiteness.
- **Oracle:** solve the same objective iteratively (LSQR/CG) and compare to the closed form.
- **Evidence so far:** β (`ENABLE_BETA=1`) is **broken**: EXP-005 (cholesky not-PD), EXP-005b (lstsq
  NaN), EXP-006 (NaN produced *inside* `refit_beta_nnls`, so an output clamp cannot fix it). λ is
  irrelevant (HYP-R0 wash).
- ✅ **ROOT CAUSE IDENTIFIED (SCOUT-AM, LIT-002) — three concrete divergences from the paper's Alg. 3:**
  1. **`key_select.py:35` uses `torch.linalg.lstsq` with the default `gels` driver**, which on CUDA
     returns **NaN/Inf for rank-deficient input without raising** — so the `except RuntimeError` at
     `:36` never fires and the NaN survives every downstream clamp. **This is why EXP-006's output
     clamp could not work.** A 64×64 `Φ`, after the row-shift at `:216`, is very plausibly
     rank-deficient. Fix: `driver='gelsd'`.
  2. **No upper bound anywhere.** `:46` clamps the minimum only; the `1e-12` floor puts **β at −27.6**
     versus the paper's box **`w ∈ [e⁻³, e³]`** (β ∈ [−3,3], `iters=2`) for highest-attention keys, or
     `iters=0`, `w ≤ e⁷`, prune any key with `β < −7` for OMP. The paper's stated failure mode is
     verbatim ours: *"once β is very negative, the corresponding key cannot contribute to the
     attention output, regardless of Cv."*
  3. **`key_select.py:215`'s residual-target formulation `(target − fixed_mass).clamp_min(1e-12)` is
     absent from the paper** (which fits **all t** keys against the **full** mass). Whenever the 448
     untouched slots over-supply mass, this zeroes the entire NNLS target ⇒ w→0, β→−27.6, an all-zero
     value-solve design matrix, and the EXP-005b NaN at `core.py:230`.
- **Status upgrade:** B-SOLVE is no longer "a deep numerical rabbit hole" — it is three named,
  paper-specified fixes. Combined with `_should_fit_beta` (above), β has **never actually run**.

## B-WIRE — wiring bug in the target path
- **Stage:** CLOSED · **Status:** ✅ **CONFIRMED — it is a bug** (DIAG-WIRE, 2026-07-28)
- **Mechanistic sentence:** `TARGET_MODE` is parsed (`continual_am_sparse.py:78`), passed into the
  config (`:153`), echoed into the run name and the wandb tag — and then **never read** by the
  per-document path. `run_per_document_am_phase2` (`cartridges/am/continual.py:41`) contains **zero**
  occurrences of `target_mode`, and `apply_document_am_write_to_cache` (`finetune.py:337-595`)
  unconditionally builds the teacher KV (`:464`) and calls `compute_teacher_targets` (`:477-484`).
  The per-doc write is hard-wired to `cartridge_plus_doc` and **structurally cannot express
  `teacher_attention`** — that path takes no `target_accumulator`, and `continual.py:172` discards it
  as `_` with `collect_teacher_targets=False`. The only live reads of `config.target_mode` are in
  `run_decoupled_tfidf_am_update` (`finetune.py:630, :672-676`, the `legacy_decoupled` path) and the
  deprecated `train.py:591`.
- **Evidence:** three targets built explicitly differ (max-abs 1.22 / 3.44 / 3.56), yet the per-doc
  write on a fresh identical cache gives **max|dV| = 0.000e+00** and `mean_mse = 0.03795905038714409`
  to 17 digits for all three modes. Positive control: the legacy path *does* respond
  (max|dV| = 16.62; mse 1.35e-08 vs 0.845). The three EXP-008 caches are bitwise identical across all
  180 tensors, from three separate runs with distinct wall-clocks (181.4 / 169.3 / 165.2 s) —
  so it is not a stale cache or one checkpoint evaluated three times. Not the dual-`cartridges`
  foot-gun either: all eight `cartridges/am/*.py` are byte-identical between the two repos.
- **Consequence:** **EXP-008 is three replicas of EXP-001. HYP-T1's "NULL" rests on no evidence and is
  RETRACTED.** The target lever is **untested**, not dead → see B-TARGET, now promoted.
- **Latent hazards found nearby (queued, not yet acted on):** (i) `finetune.py:672-676` silently
  downgrades `cartridge_plus_doc` → `self` in the legacy path, mislabelling one of *its* three modes
  too; (ii) EXP-008 ran with `WANDB_DISABLED=1` (`launch_exp008.sh:67`) — no wandb run exists for
  either arm and no `result.json` was ever written. This is exactly the failure the §0b wandb mandate
  now prevents.
- **Fix:** sketched but **not applied** (bundle `results/DIAG-WIRE/result.json`): an opt-in target
  branch at `finetune.py:477`, the `target_accumulator` plumbing for `teacher_attention` (watch the
  `randperm` alignment trap at `finetune.py:425-428`), and a fail-loud guard in `continual.py` so a
  requested mode can never again be silently ignored. **Blocked on ORACLE-WRITE finishing** — that
  worker is currently editing `finetune.py` and two editors on one file is how this loop lost a batch
  before (MECH-000).

---
## Closed / refuted (keep — they constrain the search)
| id | verdict | the mechanistic sentence |
|---|---|---|
| K-GATE (HYP-G1) | refuted as an acquisition lever | pure-TF beats TF-IDF on **both** axes (QA 2.252 vs 2.635, MT 2.543 vs 3.007); IDF spends scarce support on slots the new doc barely attends to. Gating moves *forgetting*, not acquisition. |
| K-SUPPORT (HYP-S1) | ⚠️ **RE-OPENED 2026-07-28** | MT flat across `top_t` 32/64/128 while QA rises monotonically — but every arm ran with `max_queries_per_head=64` hard-coded, so `top_t=128` was an *underdetermined* solve (min-norm branch). Support and query count are confounded; re-test jointly at `n ≫ t`. |
| K-RIDGE (HYP-R0) | refuted | λ=0 vs 1e-4 is a wash (Δ ≪ noise); at 1e-4 the ridge is already negligible. |
