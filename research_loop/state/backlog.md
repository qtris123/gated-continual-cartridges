# BACKLOG — work queued *behind the board*

> **This is not a sweep list.** The bottleneck board (`bottleneck_board.md`) drives the loop; this file
> holds work that is ready to dispatch **once a board entry reaches the stage that needs it**.
> Every item names the **board entry** it serves and what its outcome would mean.
> Anything here that cannot name one is drift — delete it.

Priority = whatever the active bottleneck needs. Do not work down this list top-to-bottom.

---
## Stage A — MEASURE (open now; every board entry is at stage A)

- [ ] **DIAG-WIRE → B-WIRE** *(no GPU, cheapest, do first)*. Read `cartridges/am/finetune.py`
      `target_mode` handling (~L108-190) + the `per_document` path; assert the solved target tensor
      actually differs across `{cartridge_plus_doc, self, teacher_attention}`.
      **Outcome:** identical ⇒ wiring bug ⇒ fix, re-run EXP-008, **retract HYP-T1's NULL**.
      Different ⇒ the null was real and B-TARGET must be attacked another way.
- [ ] **DIAG-ROUTE → B-ROUTE** *(GPU)*. Attention mass on rewritten slots at eval time, per layer,
      MT queries vs QA queries, before vs after the write.
      **Outcome:** near-zero mass on written slots ⇒ routing-limited ⇒ the answer involves keys.
- [ ] **ORACLE-WRITE → B-ROUTE** *(GPU)* ⭐ **the most informative run in the mission**. Put the
      teacher's own KV for the new doc into the selected slots; eval both splits.
      **Outcome:** MT ~2.5 ⇒ *no* value-only frozen-key write can win (family capped) — mission pivots
      to keys/write-rule. MT ≪ 2.5 ⇒ the ceiling is high and our *solve* is what's failing.
- [ ] **DIAG-OBJ → B-OBJ** *(GPU)*. Achieved solve MSE (per layer, `am/mean_mse`) against realized ΔCE.
      Then the MSE→0 oracle (unbounded support, `RIDGE_LAMBDA=0`).
      **Outcome:** MSE→0 with flat MT ⇒ objective mismatch confirmed ⇒ change what we fit.
- [ ] **DIAG-TARGET → B-TARGET** *(GPU)*. Recon error on reference queries vs held-out MT queries.
      Then the target-cheat oracle (reference set built from the eval MT queries).
      **Outcome:** large MT drop under the cheat ⇒ reference-limited ⇒ better reference construction.
- [ ] **DIAG-CAP → B-CAP** *(GPU)*. Residual energy / effective rank vs `top_t`; `TOP_T=512` oracle
      reading MT alone. **Outcome:** flat MT ⇒ capacity refuted (and the `top_t=128` MT *regression*
      belongs to another cause).
- [ ] **DIAG-SOLVE → B-SOLVE** *(GPU)*. Condition number / Gram spectrum; closed form vs an iterative
      (LSQR/CG) solve of the same objective. **Outcome:** a gap ⇒ we don't reach our own optimum.

## Stage B — SEARCH (dispatch as soon as the matching cause has a number)
- [ ] **SCOUT-AM → B-WIRE/B-TARGET/B-SOLVE.** `AM.pdf` in depth: what the target actually is, whether
      our `per_document` path is faithful, and the exact role/derivation of β.
- [ ] **SCOUT-ROUTE → B-ROUTE.** Delta rule / fast weights (DeltaNet, FWP), associative-memory capacity,
      closed-form key placement. The question: *how do you write so the write is retrievable?*
- [ ] **SCOUT-EDIT → B-CAP/forgetting.** ROME/MEMIT (covariance-preconditioned closed-form multi-edit),
      **AlphaEdit / null-space-projected editing**, Adam-NSCL. The question: *can we write in the null
      space of the old keys — acquisition without forgetting, in one operator?*
- [ ] **SCOUT-OBJ → B-OBJ.** Output/logit-space distillation objectives, Gauss-Newton / natural-gradient
      closed forms. The question: *what closed-form fit provably moves CE, not MSE?*
- [ ] **SCOUT-GATE.** H2O / SnapKV / PyramidKV / Scissorhands — how the KV-compression field decides
      which slot matters; what it implies for our gater.

## Stage C/D — BUILD & TEST (gated on a measured cause + a chosen LIT entry)
- [ ] **MECH-KEYS → B-ROUTE.** `KEY_MODE ∈ {highest_attention, omp}` already exists but has **never been
      run in this loop**. Frame as an acquisition↔forgetting **trade curve**, freeze as the reference
      point. (Not a sweep: it discriminates B-ROUTE.)
- [ ] **MECH-BETA → B-SOLVE.** Numerical guards *inside* `refit_beta_nnls` (input clamp, regularized /
      renormalized targets, NaN guards). Only after the board says mass-matching would matter — β is
      AM's own acquisition mechanism, but it has already burned three experiments (MECH-000).
- [ ] **MECH-ROUNDS.** Alternating closed-form re-solves (value-solve ↔ key/support re-select, ALS/EM
      style, K rounds). Still `gradient_steps = 0`. Directly attacks "one-shot linear projection is
      weaker than iterative fitting". **Cost it** — many re-solves is drift (NORTH_STAR).
- [ ] **MECH-NULLSPACE → B-ROUTE/forgetting.** If SCOUT-EDIT delivers: project the write onto the null
      space of Phase-1 keys/queries. This is the one candidate that could improve *both* axes at once.
- [ ] **MECH-SEED (infra) → confirmation standard.** Opt-in `SEED` env knob for
      `continual_am_sparse.py` (default = current, stock runs bit-identical). Needed before any PASS
      can be verified (RUNBOOK §9c).

## Stage E — VERIFY (mandatory before anything is believed)
- [ ] Any result that flips direction, closes a board entry, or would be reported as a win: fresh
      process + changed seed + mechanism-off ablation + Phase-1 floor control (WORKERS.md W5).

## Reference lines / bookkeeping
- [ ] **D0 ICL re-ruler** *(GPU, informational)*. REF-ICL was measured on a different harness; redo with
      `EVAL_MODE=icl` in `eval_forgetting.py` on `qasper_eval_{QA,MT}.parquet` so the ceiling shares the
      ruler. Gates nothing — do it when a GPU is otherwise idle.
- [ ] **T3 cost table.** AM Phase-2 vs cartridge Phase-2 train time + a synthesis-cost term (mined from
      logs/wandb or cited from the paper — **never re-synthesize**). Assemble the quality×cost Pareto plot.

---
## Retired (kept so they are not re-run)
Knob-level questions, all resolved and now part of the board's "closed" section:
`HYP-G1` no-IDF gating wins both axes · `HYP-S1` acquisition is not support-limited ·
`HYP-R0` ridge λ is a wash · `HYP-T1` target_mode bit-identical (**re-opened as B-WIRE**) ·
`HYP-SG1` sparse-gradient works but is disqualified by the gradient-free requirement.
Old sweep candidates with no board entry (HYP-R1/R2/R3, HYP-S2/S3, HYP-a5, HYP-PH1, GATE-N1/N2/N3):
**not queued.** Any of them may return the moment a measurement gives it a cause to serve.
