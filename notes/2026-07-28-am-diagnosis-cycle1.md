# AM diagnosis loop, cycle 1 — three prior conclusions retracted, one live bug found

**Date:** 2026-07-28 · **Status:** in-progress · **Branch:** `trivo-explore-research-work`
**State:** `research_loop/state/bottleneck_board.md` (live), `results.csv`, `literature_ledger.md` (LIT-001…027)

## What changed about how we work

The loop was rewritten from a **knob sweeper** into a **diagnosis engine**: measure the cause → read the
literature for a fix → build it → test one variable → verify adversarially → re-measure. Two hard gates:
**no mechanism may be built for a cause with no measured number**, and **no sweep without a stated
discriminand**. Stop conditions are only a verified PASS or a human stop — "out of knobs" triggers an
escalation ladder instead. Full spec: `research_loop/{PLAN_AM_MUST_WIN,DESIGN,ORCHESTRATOR,WORKERS}.md`.

## Headline: the prior loop's central claim did not survive contact with instrumentation

The 2026-07-26 write-up concluded that closed-form AM's acquisition is "a hard ceiling ~2.54 immovable by
gating/support/target/ridge". **Two of those four pillars were measurement artefacts, and a third is
confounded:**

| prior claim | what cycle 1 found |
|---|---|
| `target_mode` is a no-op (HYP-T1 NULL) | **A WIRING BUG.** `target_mode` never reaches the per-document write path (`continual.py:41` has zero occurrences; `finetune.py:337-595` is hard-wired to `cartridge_plus_doc`). Explicitly-built targets differ (max-abs 1.22/3.44/3.56) but the write gives **max\|dV\| = 0.0** for all three modes, and the three EXP-008 caches are **bitwise identical across 180 tensors**. EXP-008 was three replicas of EXP-001. **HYP-T1 retracted.** |
| support doesn't help (HYP-S1) | **Confounded.** `max_queries_per_head` was hard-coded at **64** with no knob, so `top_t=128` was an *underdetermined* solve (min-norm branch). That also explains the previously unexplained top128 regression. **Re-opened.** |
| β is a numerical rabbit hole | **Root-caused in three lines.** `key_select.py:35` uses `torch.linalg.lstsq` with the default **`gels` driver**, which returns NaN for rank-deficient input **without raising** — so the `except RuntimeError` never fires and EXP-006's output clamp could never have worked. Plus no upper bound (β floors at −27.6 vs the paper's `[−3,3]`), plus a residual-target clamp absent from the paper. |
| ridge is a wash (HYP-R0) | Stands; independently replicated at full support. |

**And the config was running AM's weakest ablation.** The paper fits attention **output** *and* attention
**mass**; our canonical config ran **neither** — `KEY_MODE=freeze` silently disables β via
`_should_fit_beta`, and 64 reference queries against the paper's 16k–50k left the solve exactly
determined. AM's `Ck` is also always a subset of *the block being compacted*, so the paper has **no
procedure for our regime** (writing into keys fitted to other content) and names it as future work.

## What is now established (measured, not argued)

- **B-OBJ — the fitted objective is anti-correlated with what we score.** Removing the trust region drove
  reconstruction MSE **18.2× better** (better on all 16/16 docs) and **destroyed the model**: both losses
  → ~16 (ppl 8e6), `|v|` 848 → 21504, **no NaN/Inf**. It also revealed that the trust region — not `top_t`,
  not ridge — was what pinned layers 34/35, which held 93–95% of the residual.
- **B-ROUTE — value-only writing is bandwidth-capped.** A *perfect* write (the teacher's own document
  values) into the tf-idf top-32 slots reaches only **MT 2.381** vs the 1.873 bar, closing 25% of the gap.
  Those slots carry ~9% of attention and **MT queries route to them no more than QA queries do (1.04)**.
- **No MT-specific routing subspace exists.** ρ_MT = **0.012** (vs an in-sample QA control of 0.0096);
  QA/MT mean-routing cosine **0.99899**, top-32 slot overlap **0.914**. By LIT-011's own criterion this is
  the "no value-space operator can separate the two axes" branch. Null-space *value* projection survives
  only as a retention operator. **This also undercuts the gating premise** — gating cannot separate what
  routing does not.
- **B-CASCADE — refuted as an acquisition explanation, half right as mechanics.** Raising `n` to 16384
  repairs the routing collapse (cartridge mass 0.329 → 0.640) but `|v|` *grows* 8× and **MT never
  improves**. Derived cause: the guarded solve stacks `[X_new (n×t); √w·I (t×t)]`, so the trust region's
  relative pull decays like **1/n**. Cost is flat (256× queries → 1.23× wall clock).
- **Our write is already MEMIT with the wrong metric.** `DELTA_WEIGHT=1e-2` makes
  `guarded_sparse_am_value_update` exactly `Δ = R K₁ᵀ(C₀ + K₁K₁ᵀ)⁻¹` with **`C₀ = w·I`**; the literature's
  single claim is that `C₀` should be the old keys' second moment.
- **β cannot fix acquisition, algebraically.** β is *query-independent*, so it maps MT and QA mass through
  the same monotone function: a +2.32-nat boost takes mass 0.090 → 0.500 but the MT/QA ratio **1.083 →
  1.046**. β buys bandwidth and provably cannot buy selectivity.

## Open, and the one that may reframe everything

🔴 **B-ROPE.** `cartridges/am/` hard-codes `rope_theta = 10000.0` in every entry point and **no caller
passes it**, while `Qwen3-4B-Instruct-2507` uses **5000000** — 500× off, live in the teacher-target path at
document offsets where the rotations are decorrelated. If the target was built in the wrong rotary frame,
**B-OBJ's anti-correlation is exactly what fitting a corrupted target would produce.** A single-variable
A/B is running; the competing reading (the frames cancel, the bug is cosmetic) is equally reportable.

Also open: whether keys are viable at all. DIAG-ROUTING measured the *post-softmax routing simplex*, whose
QA Gram has participation effective rank **1.97** — so the 0.914 overlap may be **key blindness, not query
similarity**. The 128-dim pre-softmax query geometry is unmeasured, and every row in `results.csv` is
`KEY_MODE=freeze`.

## Process lessons worth keeping

- **`mass_on_S` was already computed in the code** (`value_solve.py:146`) and no bundle recorded it for the
  loop's first nine experiments — the one number three literatures agree bounds any value-only write.
- **Frozen-snapshot import pins are mandatory** when a measuring worker overlaps an editing one (RUNBOOK
  §9c-bis). HEAD moved mid-run twice; the pins held. The verification probe is a **false negative from the
  repo root** (cwd is `sys.path[0]`), and unpinned imports on this box resolve to the *sibling* repo.
- **An orchestrator design error, recorded as such:** `ORACLE-WRITE-512` was specified at `TOP_T=512`
  without checking that sequential per-document writes at full support annihilate each other
  (`n_written=511` for all 16 docs ⇒ only doc 16 survives). It bounds "full support as implemented", not
  bandwidth. It also corrected a board number I had propagated: the writable-bandwidth gain is **2.6×, not
  6.6×**, because a **frozen slot-0 attention sink holds ~0.355** of all cartridge mass and no AM config
  can write it.
