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
- **Stage:** A MEASURE · **Status:** suspected
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
- **Evidence so far:** none. **Every run in `results.csv` used `KEY_MODE=freeze`** — the prior "keys
  collapse QA" is from a pre-loop, Llama-era session and is *untested here* (RUNBOOK §7).

## B-TARGET — target / reference distribution
- **Stage:** A MEASURE · **Status:** suspected
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
  QA degrades monotonically ⇒ **capacity is probably not binding**, but `top_t=128` also got *worse* on
  MT, which pure capacity cannot explain — that anomaly belongs to B-OBJ or B-ROUTE.

## B-SOLVE — numerics
- **Stage:** A MEASURE · **Status:** partially measured
- **Claim:** ill-conditioning / the broken β path means we don't reach our own optimum.
- **Signature:** condition number and spectrum of the query Gram; β finiteness.
- **Oracle:** solve the same objective iteratively (LSQR/CG) and compare to the closed form.
- **Evidence so far:** β (`ENABLE_BETA=1`) is **broken**: EXP-005 (cholesky not-PD), EXP-005b (lstsq
  NaN), EXP-006 (NaN produced *inside* `refit_beta_nnls`, so an output clamp cannot fix it). λ is
  irrelevant (HYP-R0 wash). β is AM's own explicit acquisition mechanism, so this is a real hole —
  but fix it *after* the cheap causes, and only if the board says mass-matching would matter.

## B-WIRE — wiring bug in the target path
- **Stage:** A MEASURE · **Status:** suspected · **cheap, do first**
- **Claim:** EXP-008 found all three `target_mode` values **bit-identical to 15 digits**. Three
  genuinely different targets producing identical solutions is a no-op, not a null result.
- **Signature:** read `cartridges/am/finetune.py` (`target_mode` handling ~L108-190, and the
  `per_document` path that may bypass it); assert the solved target tensor actually differs by mode.
- **If it is a bug:** fix it, re-run EXP-008, and **retract HYP-T1's "NULL"** in the hypothesis ledger —
  several downstream conclusions ("the target is not the lever") rest on it.
- **Cost:** a code read + one unit-level assertion. No GPU needed to start.

---
## Closed / refuted (keep — they constrain the search)
| id | verdict | the mechanistic sentence |
|---|---|---|
| K-GATE (HYP-G1) | refuted as an acquisition lever | pure-TF beats TF-IDF on **both** axes (QA 2.252 vs 2.635, MT 2.543 vs 3.007); IDF spends scarce support on slots the new doc barely attends to. Gating moves *forgetting*, not acquisition. |
| K-SUPPORT (HYP-S1) | refuted | MT flat across `top_t` 32/64/128 while QA rises monotonically — coverage buys forgetting, not acquisition. |
| K-RIDGE (HYP-R0) | refuted | λ=0 vs 1e-4 is a wash (Δ ≪ noise); at 1e-4 the ridge is already negligible. |
