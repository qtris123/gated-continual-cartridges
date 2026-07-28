# OBSERVATION LOG

Append-only. One dated entry per observation. Observation = what was measured, raw.
Keep interpretation in the hypothesis ledger, not here (research-companion discipline).

Format: `## <YYYY-MM-DD HH:MM> [EXP-ID] observation` then 1-4 lines of raw fact + artifact link.

---
## 2026-07-25 06:16 [EXP-000-verify] HF Phase-1 cache = QA cache (provenance)
HF `qtris123/qwen_qasper-QA-task_8192_512_no-cartridge_10-epochs` cache_last.pt eval'd on both splits
(Qwen3-4B, this harness): QA loss 2.239 (ppl ~9.4), MT loss 3.783 (ppl ~44). QA≪MT → QA Phase-1 cache;
repo name correct, bundled config.yaml (MT-labelled) is stale. Cache = 511 trainable + 1 frozen slot.
Artifact: research_loop/results/EXP-000-verify/result.json, /tmp/exp000_{qa,mt}.log.

## 2026-07-25 06:16 [infra] eval_forgetting.py hangs post-completion; nvidia-smi flaky
eval_forgetting.py printed `Eval loss` then never exited (held GPU 1 ~78 min until killed). nvidia-smi
repeatedly >20s/hangs on this box. Guardrails added to RUNBOOK §1/§6. Harness `Eval loss` = ln(ppl).

## 2026-07-26 11:5x [REF-ICL] full-context ICL ceiling: QA 1.9734 / MT 1.8960
Full-context ICL (Qwen3-4B raw, no cartridge, topic papers in-context): QA 1.9734 (ppl 7.2, 103k ctx, 78 convos),
MT 1.8960 (ppl 6.7, 77k ctx, 69 convos). Both below all cartridge/AM/dense refs (proper ceiling). Off-diagonal
cross-topic 2.78/2.90 (confirms topic-matched context is doing the work). CAVEAT: measured via
qasper_loss_benchmark.evaluate_loss_chunked, a DIFFERENT harness than eval_forgetting.py (used for cartridge/AM/dense/
sparse-grad) — treat ICL as a ballpark ceiling, not a strictly-comparable number. Artifact: results/REF-ICL/result.json.

## 2026-07-26 11:20 [EXP-009] sparse-GRADIENT 62 steps = provisional WIN (QA 1.6169 / MT 1.9664)
Sparse gradient (continual_sparse.py, value-only/freeze_keys, tfidf top64 per_layer, USE_IDF=0, Adam LR2e-2, 1ep=62
steps, gloo, from Phase-1 cache on MT): QA-forgetting 1.6169 (n=6), MT-acquisition 1.9664 (n=5), 689s. MT closes the
AM→dense gap (AM 2.5426 → 1.9664, ~dense@4ep 1.87); QA BETTER than AM (2.2521) AND Phase-1 floor (2.239) — apparent
positive backward transfer. MT front-loads (3.78→2.11@15→~1.97@30, flat). ~1/10 dense's 624 steps. Dominates AM on
both axes. Meets +0.15 target vs dense. CAVEAT: single run, tiny eval, QA<floor surprising → EXP-009C confirming.
Ran torchrun DIRECTLY (the .sh wrapper force-sets BG_STATS→use_idf=True). Artifact: results/EXP-009/result.json.

## 2026-07-26 10:59 [EXP-008] target_mode is a NO-OP (HYP-T1 null); AM acquisition ceiling confirmed
target_mode ∈ {cartridge_plus_doc, self, teacher_attention} on the no-IDF canonical: all BIT-IDENTICAL
QA 2.2521 / MT 2.5426 (to 15 digits; solve 165-169s differ). target_mode does not change the written cache in
the per_document AM path. With HYP-S1 (support null for acquisition), this confirms AM's closed-form MT-acquisition
~2.54 is a hard ceiling. Partial REF-ICL: QA ICL ceiling 1.9734. Artifact: outputs/2026-07-26-10-53/10-56 dirs.

## 2026-07-26 10:5x [EXP-007] top_t sweep — acquisition NOT support-limited (HYP-S1 rejected)
AM-sparse no-IDF canonical at TOP_T∈{32,128} (vs EXP-001 top64 QA 2.2521/MT 2.5426): top32 QA 2.1766 / MT 2.5484
(solve 173.7s); top128 QA 2.4837 / MT 2.6860 (solve 185.4s). QA-forgetting monotonic in top_t (coverage∝forgetting,
t64→t128 +0.23>noise); MT-acquisition flat/worse (~2.54 across 32/64, worse at 128). ⇒ MT-acquisition gap vs
dense@4ep (1.87) is TARGET/solve-limited, not support-limited. top32 = marginally best AM point. Artifact: results/EXP-007/result.json.

## 2026-07-26 10:2x [EXP-006/β] β NNLS fit produces NaN — β deferred; import foot-gun found
EXP-006 (β + opt-in clamp[-3,3]) died `AssertionError: beta clamp failed: max|beta|=nan > 3.0` — the NNLS β-fit
itself yields NaN, so clamping the output can't help. β failed 3 ways (EXP-005 cholesky not-PD @66.6; EXP-005b
lstsq NaN; EXP-006 fit NaN). β DEFERRED (needs numerical guards in refit_beta_nnls). Separately: the β-clamp edit
made continual_am_sparse.py pass beta_clamp_abs UNCONDITIONALLY, which crashed ALL AM runs (incl EXP-007 top_t)
when `import cartridges` resolved to the SIBLING repo (no field). Edit REVERTED; foot-guns added RUNBOOK §6.10/6.11.

## 2026-07-26 09:2x [EXP-000] dense 10-epoch REF-CART bar = QA 2.6991 / MT 2.2137 (OVERFIT)
Final 10-epoch dense self-distillation checkpoint (cache-step624, ~101min train) eval'd both splits:
QA-forgetting 2.6991, MT-acquisition 2.2137. Both WORSE than dense@4ep (QA 2.3721 / MT 1.8725) → 10ep overfit
(QA-forgetting grows with epochs; MT regressed past ~4ep). Best dense point = @4ep. vs AM EXP-001 (QA 2.2521 /
MT 2.5426): AM retains QA better than both dense points; dense@4ep acquires MT better (gap 0.67). Artifact: results/EXP-000/result.json.

## 2026-07-26 08:51 [EXP-005b] β at RIDGE_LAMBDA=0 also failed (NaNs) — confirms unclamped β is root cause
ENABLE_BETA=1 with RIDGE_LAMBDA=0 (robust lstsq/gels path) crashed with "RuntimeError: NaNs in ridge lstsq
solution". So the failing solve branch is not the cause — the unclamped β log-weights themselves are. Both β
runs (λ=1e-4 cholesky, λ=0 lstsq) die. HYP-T2 needs the β-clamp (AM paper [-3,3]) → EXP-006 implements it.

## 2026-07-26 07:31 [EXP-005] ENABLE_BETA=1 crashed (β numerically unstable at λ=1e-4)
β engaged (max|β_logweight| 66.6, mean 4.23, ~19% nonzero, all 36 layers) but crashed on doc 3 with
linalg.cholesky not-positive-definite (am/core.py:225 _ridge_lstsq). Mechanism: unclamped β added to attention
logits pre-softmax → near-one-hot softmax → rank-deficient value-solve design matrix → XtX indefinite → cholesky
fails (primary + raised-floor fallback; λ=1e-4 too small). Docs 0-2 succeeded. No eval numbers (crash pre-eval).
AM paper clamps β∈[-3,3]; our impl doesn't. Retry EXP-005b (β at λ=0, robust lstsq). Artifact: results/EXP-005/result.json.

## 2026-07-26 01:18 [EXP-004] RIDGE_LAMBDA=0 vs 1e-4 on no-IDF canonical = WASH
Pure OLS value-solve (λ=0, ridge truly zeroed via core.py L144-145 gels branch), single var vs EXP-001:
QA-forgetting 2.2619 (ppl 9.6), MT-acquisition 2.5569 (ppl 12.9). vs EXP-001 (QA 2.2521 / MT 2.5426):
+0.010 / +0.014 — an order of magnitude under the ~0.1-0.2 noise band. solve 162.9s / e2e 186s / 0 grad.
value_global_max_abs 848. ⇒ λ=1e-4 ridge already negligible; keep canonical. Artifact: results/EXP-004/result.json.

## 2026-07-26 01:08 [infra][EXP-000] dense REF-CART run died mid-train + agent restarted from scratch
Attempt-1 dense baseline_continual (18:02 run dir) reached ~Epoch 5-6 (step ~329) then its process died; the
parked executor agent restarted training from Epoch 1 (new run dir 2026-07-26-01-07-07, gpu0) on a ~55-min
re-arm cycle. 10-epoch (~110min) dense may not finish in one window on this box; restarts lose progress.
Usable artifact: attempt-1 cache-step256.pt (~4 dense epochs). Dense MT bar already known (wandb scf175an
MT-acq 2.891). Plan: eval step256 for dense QA-forgetting rather than wait on a full run.

## 2026-07-25 18:50 [EXP-003] with-IDF canonical — IDF HURTS both axes (HYP-G1 supported)
AM-sparse USE_IDF=1 (bg_stats over Phase-1 QA corpus, tfidf, per_layer, top_t=64, freeze, ridge 1e-4 spectral),
single var vs EXP-001 (USE_IDF=0): QA-forgetting 2.6351 (ppl 13.9), MT-acquisition 3.0073 (ppl 20.2). vs EXP-001
(QA 2.2521 / MT 2.5426): QA +0.383, MT +0.465 — both >2× the tiny-eval noise band. value_global_max_abs 486
(vs 816) confirms IDF changed slot selection. solve 178.4s / e2e 201s / 0 grad. ⇒ canonical = no-IDF (EXP-001).
Artifacts: research_loop/results/EXP-003/result.json ; outputs/2026-07-25-18-47-23-continual_am_sparse/5989fef9.../.

## 2026-07-25 18:40 [SETUP-BG1] bg_stats collected over the Phase-1 self-distilled cache (infra)
Built examples/qasper2/train/collect_bg_stats.py; ran forward-only over the full Phase-1 QA corpus
(data/qasper/train/qwen_qasper_QA_task_8192.parquet, 1990 packed seqs) against
outputs/phase1_selfdistill_qwen512/cache_last.pt → outputs/phase1_selfdistill_qwen512/bg_stats.pt (292MB,
per_layer, access counts (1990,36,511) int64; derived IDF (36,511) finite). collect 265.3s. Verified loadable
by continual_am_sparse.py's bg_tracker.load()+CacheTFIDFRanker. Fix vs sketch: NUM_BG_BATCHES=full-corpus,
cache.to(local_rank) after from_pretrained. Artifact: research_loop/results/SETUP-BG1/result.json.

## 2026-07-25 18:20 [EXP-002] wandb cross-check + AM-paper facts (research)
Historical DENSE self-distillation Phase-2 (wandb SEACrowd scf175an; dense/no-sparse, adam lr2e-2, 10ep):
MT-acquisition loss 2.891 (ppl 18.01); QA-forgetting split NOT logged. Qwen Phase-1-start/no-P2 reference:
QA loss 2.051 (ppl 7.78) / MT loss 3.583 (ppl 36.0). Dense P2 train wall-clock 1805s (~30min); MT synth
10.3–11.7 ks. AM paper (2602.16284): L2 ridge on the VALUE-solve degrades ∀λ>0; β (NNLS mass-bias) is the
low-mass cure; per-head budget = top ablation. bg_stats background corpus = Phase-1 QA parquet, per_layer.
Artifact: research_loop/results/EXP-002/result.json.

## 2026-07-25 18:02 [EXP-001] AM-sparse no-IDF Phase-2 = first efficiency+quality point
Closed-form AM-sparse (USE_IDF=0, tfidf ranker, per_layer, top_t=64, freeze keys, ridge spectral) from
outputs/phase1_selfdistill_qwen512/cache_last.pt over 16 MT docs: QA forgetting loss 2.2521 (+0.013 vs 2.239
floor), MT acquisition loss 2.5426 (−1.240 vs 3.783 floor; ppl 44→12.7). solve_s 181.4 / phase2_e2e_s 217 /
gradient_steps 0. value_global_max_abs 816 (deep layers) despite intact QA. Artifacts:
research_loop/results/EXP-001/result.json ; outputs/2026-07-25-18-02-29-continual_am_sparse/efa3bc76.../cache_last.pt.

## 2026-07-28 — DIAG-WIRE (board B-WIRE): `target_mode` is a wiring no-op
- `TARGET_MODE` is parsed (`continual_am_sparse.py:78`), reaches the config (`:153`), and is echoed into
  the run name (`:87`) and the wandb tag `target-{TARGET_MODE}` (`:564`) — which is what made EXP-008
  look like a real sweep. It is then **never read** by the per-document path.
- `run_per_document_am_phase2` (`cartridges/am/continual.py:41`): zero occurrences of `target_mode`.
  `apply_document_am_write_to_cache` (`finetune.py:337-595`) unconditionally builds teacher KV (`:464`)
  and calls `compute_teacher_targets` (`:477-484`). It structurally cannot express `teacher_attention`
  (no `target_accumulator` in the signature; `continual.py:172` discards it, `collect_teacher_targets=False`).
- Unit check (CPU): explicitly-built targets differ (max-abs 1.22 / 3.44 / 3.56); the per-doc write gives
  **max|dV| = 0.000e+00** and `mean_mse = 0.03795905038714409` (17 digits) for all three modes. Positive
  control on the legacy path: max|dV| = 16.62, mse 1.35e-08 vs 0.845 — the target machinery works where wired.
- Ruled out: stale cache / one checkpoint evaluated 3× (three runs, distinct dirs, wall-clocks 181.4 /
  169.3 / 165.2 s, 16 docs each, three caches **bitwise identical** across 180 tensors); dual-`cartridges`
  (all eight `cartridges/am/*.py` byte-identical between repos); a dead write (EXP-001 vs Phase-1 gives
  max|dV| = 798.9, max|dK| = 0, exactly as `freeze_keys` predicts).
- Latent hazards: `finetune.py:672-676` silently downgrades `cartridge_plus_doc`→`self` in the legacy
  path; EXP-008 ran `WANDB_DISABLED=1` (`launch_exp008.sh:67`) so no wandb run and no `result.json` exist
  for it — precisely the failure mode the §0b wandb mandate now blocks.
- **Consequence: HYP-T1 RETRACTED; B-TARGET is untested, not dead.**

## 2026-07-28 — DIAG-OBJ (board B-OBJ): better fitting bought no CE
- (a) Canonical top32 reproduced **bit-identically** on a different GPU + fresh process:
  QA 2.1766157150268555 / MT 2.5483615398406982, `value_global_max_abs` 984.0,
  `mean_mse_last_doc` 0.198077 — all matching EXP-007-top32 to every printed digit.
  wandb https://wandb.ai/vqtri-purdue-university/SEACrowd/runs/suffgao1
- (b) Maximal-support oracle (`TOP_T=511`, `RIDGE_LAMBDA=0`, no OOM, 234 s): QA 2.5151 / MT 2.6652.
  wandb https://wandb.ai/vqtri-purdue-university/SEACrowd/runs/s2wb1tnk
- `am/mean_mse` mean over 16 docs: **0.11906 → 0.09155 (−23.1%)**, uniform (per-doc ratio 0.62–0.85 on
  all 16 docs). **MT went the wrong way (+0.117); QA +0.339.** Standalone re-evals of both
  `cache_last.pt` agree within 0.004. ⇒ a materially better fit of the solve's own objective bought
  **zero** CE improvement.
- Caveat (worker-flagged, correct): this is a *maximal-support* oracle, not MSE→0. `DELTA_WEIGHT=1e-2`
  appends a 511×511 identity block in `guarded_sparse_am_value_update`, keeping the system
  over-determined and shrinking the solution toward the prior. → DIAG-OBJ-c (`DELTA_WEIGHT=0`) dispatched.
- ⭐ **93–95% of the residual is in layers 34–35 alone** (4.40 / 2.23 on the last doc) and those two
  refuse to shrink (0.93× / 0.77×), while the other 34 layers drop 44.5% and are already tiny
  (median ≈0.0027). The solve is already near its own optimum nearly everywhere.
- Provenance: the concurrent ORACLE-WRITE worker edited `value_solve.py` (19:22:42), `finetune.py`
  (19:23:12) and `continual.py` (19:23:37) *during* run (a). This worker had PYTHONPATH-pinned imports
  to a frozen git-HEAD snapshot (verified byte-identical to HEAD before and after), so both runs used
  pristine solver code and the `AM_ORACLE_WRITE` block was inert (absent from both `config.yaml`s).
  **Without the pin, both runs would have silently used half-edited value-solve code.** → RUNBOOK §9c-bis.

## 2026-07-28 — SCOUT-EDIT (board B-ROUTE/B-CAP): LIT-009…LIT-018
- **Our canonical write is already MEMIT with `C₀ = w·I`.** `continual_am_sparse.py:96` sets
  `DELTA_WEIGHT=1e-2`, overriding the `0.0` dataclass default (`finetune.py:87`), so `finetune.py:580`
  routed every AM row through `guarded_sparse_am_value_update` =
  `min ‖A_new V_S − R_new‖² + w‖V_S − V_S^old‖²` = MEMIT's `Δ = R K₁ᵀ(C₀+K₁K₁ᵀ)⁻¹` at `C₀ = w·I`.
  The literature's single claim: `C₀` should be the old keys' second moment. Our key is the simplex
  routing vector `a_S(q) = alpha[:,S]` (`value_solve.py:86`) ⇒ `C₀` is a `t×t` routing Gram from one QA
  forward pass.
- **`DELTA_WEIGHT` has never been swept** — every row is 1e-2. HYP-R0 refuted `RIDGE_LAMBDA`, a
  different knob. DIAG-OBJ-c (`DELTA_WEIGHT=0`) is the family's free negative control.
- **`mass_on_S` is already computed** (`value_solve.py:146`, `finetune.py:628`) **and no bundle has ever
  recorded it** — the quantity three literatures (nonparametric-regression attention, modern Hopfield,
  fast weights) agree bounds every value-only write. Now a standing bundle requirement.
- **Null-space value editing is identically zero at `top_t=64`**: `C₀ ∈ R^{64×64}` is full rank when
  `n_old ≫ 64` ⇒ `P = 0`. Those projectors exist only at large support.
- Preconditioning/projection are **retention** mechanisms; they cannot move MT alone. Their value is
  converting our 0.34 of QA slack into support without EXP-007's monotone QA cost.
- Ranking (findings, not decisions): LIT-017 null-space key placement > LIT-011/012 null-space value
  projection at large `top_t` > LIT-009/010 `C₀`-metric solve (LIT-010 predicts a **sign flip** on the
  EXP-007 top128 anomaly) > LIT-013/015 disjoint-support selection.

## 2026-07-28 — ORACLE-WRITE (board B-ROUTE): the write ceiling is real but not flat
- Phase-1 2.2388/3.7825 · control (solved) 2.1772/2.5524 · **oracle (teacher's own doc values) 1.8955/2.3810**
  · dense@4ep bar 2.3721/1.8725. A *perfect* content transplant into the tfidf top-32 slots closes only
  **25% of the MT gap** (−0.171, edge of noise) and leaves **0.509 to the bar**; QA improves −0.282.
- Routing (full eval-time softmax over `[512 slots ‖ own-sequence causal prefix]`, mean over 36 layers,
  S = union of the 16 docs' selections, 34–82 slots/layer): mass on S = **MT 0.0927 / QA 0.0888**,
  **ratio 1.04** — MT queries do NOT preferentially route to what we wrote. S carries ~15.5% of cartridge
  mass, ~9% of total attention; only 4/36 layers below 5%. The write barely perturbs routing
  (cartridge total 0.588 → 0.596), as frozen keys predict.
  ⇒ the slots are **readable through a narrow channel, not unreadable**: value-only frozen-key writing
  is bandwidth-capped, and **no value-only write at this support/selection can reach MT ≤ 2.02.**
- Flag-off control reproduced EXP-007-top32 to all 16 printed digits in a fresh process (and the Phase-1
  floor exactly), which also **retires the sibling-import confound** on EXP-007's number.
- 🔥 **Unlooked-for (new board entry B-CASCADE):** the **solved** write reaches `|v|` up to **984** and
  **collapses total cartridge attention 0.588 → 0.329 on MT** (0.570 → 0.320 QA), 10/36 layers under 5%.
  The oracle's teacher values top out at **|63|** and leave cartridge mass at Phase-1 level. Our solve is
  pushing the model to route away from the cartridge *as a whole* — per-layer independent writes shifting
  the residual stream, hence the queries of every later layer. Converges with SCOUT-AM's divergence #3
  (no on-policy layer-sequential re-extraction) and #2 (n=64 ⇒ exactly-determined ⇒ exact interpolation
  with no norm control ⇒ |v|=984).
- ⚠️ Outstanding W5 control: the oracle changes value **magnitude** as well as content, so part of both
  gains may be a magnitude/entropy effect. Needs a **norm-matched shuffled-value control** before the
  oracle's cause is attributed. Queued.

## 2026-07-28 — DIAG-OBJ-c (board B-OBJ): MSE and CE are ANTI-correlated
- Single variable vs DIAG-OBJ-b: `DELTA_WEIGHT=1e-2 → 0` (with `TOP_T=511`, `RIDGE_LAMBDA=0`).
- mean `am/mean_mse` 0.09155 → **0.005037** (−94.5%, an 18.2× better fit, improved on **all 16/16
  documents**, per-doc ratio 0.0024–0.706 — the closest this objective has ever come to zero).
- **Both eval losses exploded: MT 2.6630 → 15.9504, QA 2.5133 → 16.0084** (ppl ≈ 8e6). Standalone
  re-eval of the run's own `cache_last.pt` agrees within 0.028. The delta is ~70× the noise band.
- `value_global_max_abs` 848 → **21504** (L32; also L20 16000, L31 8096). **No NaN/Inf anywhere**, rc=0,
  frozen keys unchanged at 302 — the failure is **magnitude**, not numerical overflow.
- **Layers 34/35 finally broke.** They held 93–95% of the residual and refused to shrink under both more
  support and zero ridge; removing the trust region collapsed them: L34 4.0764 → 0.000155 (3.8e-05×),
  L35 1.7208 → 0.0000273 (1.6e-05×); their share of per-layer residual 95.44% → **0.43%**. The other 34
  layers improved 0.152× (9 of 36 got slightly worse). ⇒ **the trust region — not `top_t`, not ridge —
  was what pinned them.**
- **B-OBJ CONFIRMED in the strongest available form:** an unregularised minimum-residual fit on the
  reference queries buys an 18× better reconstruction and pays +13.3 loss. The fitted distance is not
  merely decoupled from token CE — over this range it is **anti-correlated**.
- Provenance: ORACLE-WRITE's edits were **committed mid-run** (HEAD `0dbc6ed` → `c4e2e9f`) and the live
  `value_solve.py` no longer matches what DIAG-OBJ-b ran. The pin held — `import cartridges` resolved to
  the snapshot, manifest hash identical before/after, `diff -rq` against DIAG-OBJ's frozen tree shows no
  differences. `CARTRIDGES_DIR` was pinned too, so the driver also came from committed code.
- ⚠️ **RUNBOOK correction (worker-found):** `python -c "import cartridges"` run **from the repo root is a
  false negative** — cwd is `sys.path[0]` for `-c`, so it prints the repo path even when the pin is
  correct. Probe from `/tmp`. Also confirmed: **unpinned imports on this box resolve to the SIBLING
  repo**, not `_explore`.

## 2026-07-28 — DIAG-ROUTING (board B-ROUTE/B-CAP): no MT-specific routing subspace exists
- Forward passes only, Phase-1 cartridge, both splits, `top_t=512`, 86 s, 0 gradient steps.
  wandb https://wandb.ai/vqtri-purdue-university/SEACrowd/runs/lqj2ssic
- **QA Gram null space is large:** λ₁/tr 0.834 (pooled 0.892), participation eff. rank 1.97; eff. rank of
  512 = 6.5 @τ1e-2, 33 @1e-3, 110 @1e-4; GPM 90%/99% ⇒ r = 9.6 / 72.7 ⇒ null space **439–505 dims**.
  LIT-011's "P ≡ 0 at t=64" caveat is answered — the projection family is not vacuous at full support.
- **MT does not live there: ρ_MT = 0.0120** (99% rule), 0.0423 @rank32, 0.0095 @rank128, vs an in-sample
  QA leakage control of 0.0096 / 0.0361 / 0.0071 — MT exceeds QA's own spectral tail by **13–65% only**.
  No layer above 0.0147; best head 0.209. LIT-011's criterion ρ→0 ⇒ *no value-space operator can
  separate the axes*. **We are in that branch.**
- **QA/MT overlap is near-total:** mean-routing cosine **0.99899** (min 0.9755/288 heads), histogram
  intersection 0.963, top-32 slot overlap **0.914**. LIT-013's falsifier was intersection < 0.2.
  Entropies nearly equal (1.946 vs 2.009 nats ⇒ 14.4 vs 15.1 effective slots).
- **`mass_on_S` recorded for the first time in the loop's history:** `top_t=512` QA 0.5697 / MT 0.5879
  (ratio 1.032); tf-idf top-32 union QA 0.0827 / MT 0.0896 (ratio 1.083). **Full support is 6.6× the
  bandwidth of ORACLE-WRITE's operating point.**
- Convention cross-check against ORACLE-WRITE's independent implementation: 0.08961 vs 0.08956,
  0.58787 vs 0.58787, 0.56967 vs 0.56967 — **agreement to 4–6 s.f.**, so both are cross-validated.
- **Rules out:** null-space VALUE projection as an acquisition mechanism (retention only); the LIT-013
  disjoint-support/interference family. **Also undercuts the gating premise** — at 0.914 overlap,
  "slots the new doc attends to" ≈ "slots QA attends to"; gating cannot separate what routing does not.
- **Still alive:** LIT-017 null-space **key** placement (concerns the 128-dim query second moment `Q₀`
  and *synthetic* keys — untested by this diagnostic). Keys remain the one untouched axis.
- Caveat (worker-stated): ρ_QA is in-sample by construction, so a held-out control would *shrink* the
  MT/QA contrast, not widen it. All numbers are pre-write.

## 2026-07-28 — MECH-QUERIES (board B-CASCADE): refuted, with the mechanism for why
- BUILD: `MAX_QUERIES_PER_HEAD` env knob (MECH-002), default 64, **bit-identical when default** — driver
  `to_dict()` byte-identical to HEAD (empty diff) and the `n=64` GPU control reproduced EXP-007-top32 to
  all 16 digits, matching ORACLE-WRITE's independent control exactly. `randperm` untouched and
  cap-invariant, so the n=64 draw is an exact prefix of every larger draw (verified).
- **The accumulator can supply 57,344–81,920 real queries per KV-head per document.** The hard-coded 64
  was discarding **~99.9%** of them. n=16384 (the AM paper's own regime) ran fine.
- Sweep at `TOP_T=32` — the three quantities B-CASCADE bound together **decoupled**:
  | n | QA | MT | \|v\|max | cartridge mass MT |
  |---|---|---|---|---|
  | 64 | **2.1772** | **2.5524** | 984 | 0.3285 |
  | 256 | 2.4097 | 2.7716 | 1328 | 0.3715 |
  | 1024 | 2.2575 | 2.5744 | 1968 | 0.6399 |
  | 4096 | 2.3412 | 2.7114 | 3344 | 0.6395 |
  | 16384 | 2.3325 | 2.6943 | 7808 | 0.6388 |
  ✅ routing collapse repaired and overshooting (0.329 → 0.640 vs Phase-1's 0.588); ❌ `|v|` **grew** 8×
  away from the teacher's |63|; ❌ **MT never improved** (best = the n=64 control; n=1024's +0.022 is noise).
- **Why `|v|` grew** (worker-derived, reproduced at unit level 3.52 → 5.80): the guarded solve stacks
  `[X_new (n×t); √w·I (t×t)]` ⇒ `(X_newᵀX_new + w·I)V = X_newᵀR + w·V_old`. The data Gram scales with `n`
  while the trust region contributes a fixed `t` rows, so **its relative pull decays like 1/n**; the
  spectral ridge (λ ∝ σ_max(X)²) is scale-invariant and doesn't compensate.
- ⚠️ **Confound the worker raised against itself:** `n` was swept at fixed `DELTA_WEIGHT`, so "queries
  don't help" is not separated from "queries help but the 1/n decay cancels it". Env-only fix: scale
  `DELTA_WEIGHT ∝ n`. → MECH-QUERIES-B dispatched (n=1024@0.16, n=16384@2.56, + the n=64 control).
- **Cost is flat:** 256× the queries for **1.23×** wall clock (164 → 202 s). A quality-dominated point,
  not a cost-dominated one.
- Solve-time `mass_on_S` recorded for the first time from the `DELTA_WEIGHT` branch (previously
  unreachable in the canonical config): 0.1585 → 0.1452 over 36 layers, cross-checking ORACLE-WRITE's
  independent 0.1523.
