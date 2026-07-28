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
## ACTIVE BOTTLENECK
**none yet — the board is at its initial state (2026-07-28). First cycle must open with W1 MEASURE.**
Recommended opening batch: **B-WIRE** (free, code read) + **B-OBJ** and **B-ROUTE**'s write-ceiling
oracle on the two GPUs. B-ROUTE's oracle is the single most informative run available (MISSION §4).

---
## The gap under investigation
Best gradient-free point (AM top32): **QA 2.1766 / MT 2.5484**, 0 gradient steps.
The bar (dense @4ep): **QA 2.3721 / MT 1.8725**. Win needs **QA ≤ 2.52 AND MT ≤ 2.02**.
⇒ **Retention is already ahead of the bar. The entire deficit is acquisition: MT −0.53 to go.**
Eliminated as *knob-level* explanations: gating (HYP-G1), support (HYP-S1), ridge (HYP-R0),
`target_mode` (HYP-T1 — but see B-WIRE, the null is suspicious).

---
## B-OBJ — objective mismatch
- **Stage:** A MEASURE (sharpened, one decisive test left) · **Status:** **measured — leaning confirmed**
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

## B-ROUTE — routing / frozen keys
- **Stage:** A MEASURE · **Status:** suspected · **⭐ highest information value**
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
