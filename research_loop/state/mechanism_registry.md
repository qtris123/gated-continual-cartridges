# MECHANISM REGISTRY — what we built, and what it actually did

> **What this file is:** every code-level mechanism this loop adds, from the LIT entry that motivated
> it to the experiment that judged it. It is the answer to "what did you actually try, and why?"
>
> **Rules:**
> - Every mechanism is an **opt-in flag, default off** — stock configs must reproduce bit-identically.
> - A mechanism is only built for a cause with a **measured** number on the bottleneck board.
> - A verdict of `dead` requires a **mechanistic reason**, not "it didn't help".

**Status:** `built` (sanity-checked, untested) · `tested` (an EXP ran it) · `supported` ·
`dead` (+ reason) · `reverted` (removed from the tree, with why).

## Template
```
### MECH-XXX: <name>  →  flag `FLAG_NAME` (default: off)
- Status: built
- Implements: LIT-XXX          | Targets board entry: B-XXXX
- Files: cartridges/am/<file>.py::<function>  (+ driver plumbing in examples/qasper2/train/...)
- What it changes (2-3 sentences, at the level of the math):
- Sanity check: <the tiny run / unit call and the finite numbers it produced>
- Bit-identical with flag off? yes/no  (must be yes)
- Tested by: EXP-XXX → QA x.xxx / MT x.xxx vs baseline EXP-YYY (QA / MT), wandb: <url>
- Verdict + mechanistic reason:
- Kept in tree? yes/no
```

---
## Prior art in this tree (from earlier cycles — read before rebuilding anything)

### MECH-000: β / mass-matching clamp — `beta_clamp_abs` — **REVERTED**
- Status: reverted · Targets: B-SOLVE · Implements: the AM paper's β stability range [-3, 3]
- Files (at the time): `cartridges/am/key_select.py::refit_beta_nnls` + an unconditional kwarg in the driver
- What happened: EXP-005 (β at λ=1e-4) died with a non-PD Cholesky; EXP-005b (λ=0) died with NaNs in
  the lstsq solution; EXP-006 added an output clamp and died with `AssertionError max|beta|=NaN` —
  i.e. **the NNLS fit itself emits NaN**, so clamping its output cannot help.
- Verdict: **dead as implemented**. Any retry must add guards *inside* `refit_beta_nnls` (input
  clamping, regularized/renormalized targets, NaN guards), not at its boundary.
- Extra lesson (cost the loop a full batch): the unconditional new kwarg crashed **every** AM run via
  the sibling-`cartridges` import path (RUNBOOK §6.10). **New kwargs must be passed conditionally.**

## Entries

### MECH-001: write-ceiling oracle (teacher's own document values) → flag `AM_ORACLE_WRITE` (default: off)
- Status: tested
- Implements: MISSION §4 / RUNBOOK §9b oracle ladder row "write-ceiling" (no external LIT source — this is
  an oracle, not an imported mechanism) | Targets board entry: **B-ROUTE**
- Files:
  - `cartridges/am/value_solve.py::oracle_teacher_value_write` (new, +126 lines)
  - `cartridges/am/finetune.py` — config fields `oracle_write` / `oracle_write_assign`,
    `AMUpdateStats.extra`, and the leading `if oracle_write:` branch in
    `apply_document_am_write_to_cache`
  - `cartridges/am/continual.py` — carries `am_stats.extra` into the per-document record
  - `examples/qasper2/train/continual_am_sparse.py` — env knobs `AM_ORACLE_WRITE`,
    `AM_ORACLE_WRITE_ASSIGN`, and `_oracle_write_kwargs()` (conditional kwarg + loud failure)
- What it changes (at the level of the math): instead of solving
  `min_{V_S} ||A[:,S] V_S - (target - A[:,S^c] V_{S^c})||^2 + λ||V_S||^2` for the selected rows, the
  per-document write copies the teacher's own post-RoPE document value vectors `v_doc` (already
  available in this path from `prefill_document_kv_cache`, passed in as `doc_kv[layer]`) verbatim into
  the selected slots. Keys are untouched, so eval-time routing is exactly the routing the solve faces.
  `assign="mass_ranked"` (default) pairs the highest-teacher-attention document rows with the
  highest-student-attention selected slots; `assign="sequential"` is a positional copy.
- Sanity check: unit call on random tensors (T=512, d=128, n=96, T_doc=900, |S|=32) — `n_written=32`,
  exactly 32 rows changed, every changed row byte-identical to a teacher document row, none outside S,
  all finite; `mse=0.0033` vs the solve's `0.0012` on the same system; `mass_on_S=0.0616`,
  `teacher_doc_mass=0.6359`. Edge case |S| > T_doc wrote `min(|S|,T_doc)=5` rows, finite. Driver
  configs built flag-off vs flag-on differ in **exactly one field** (`oracle_write`).
- Bit-identical with flag off? **yes** — the flag-off control run reproduced EXP-007 top32 to all 16
  printed digits on both splits (QA 2.1766157150268555 / MT 2.5483615398406982) in a fresh process.
  Dual-`cartridges` guard verified: from a neutral cwd with no PYTHONPATH `import cartridges` resolves
  to the **sibling** repo — flag off constructs fine (stock runs safe), flag on raises the intended
  `RuntimeError`. A concurrent DIAG-OBJ run without the PYTHONPATH pin was unaffected.
- Tested by: **ORACLE-WRITE** → QA **1.8955** / MT **2.3810** vs the flag-off control QA 2.1772 /
  MT 2.5524 (Phase-1 floor QA 2.2388 / MT 3.7825; dense@4ep bar QA 2.3721 / MT 1.8725).
  wandb: https://wandb.ai/vqtri-purdue-university/SEACrowd/runs/t8n8ybqx (control: `.../7netm95e`)
- Verdict + mechanistic reason: **informative — the ceiling is real but not flat.** Perfect content in
  the tfidf-selected top-32 slots improves BOTH axes but closes only ~25% of the MT gap to the bar
  (2.548 → 2.381, bar 1.873). The paired routing measurement explains why: under the full eval-time
  softmax the rewritten slots carry only ~9% of total attention mass (~15.5% of cartridge mass), and MT
  queries route to them no more than QA queries do (ratio 1.04) — so the write is readable through a
  narrow channel, not unreadable. Verdict on the *mechanism as a method* is n/a (it is a deliberate
  oracle, not a candidate write rule). Orchestrator owns the board call.
- Kept in tree? yes — opt-in, default off, needed to reproduce/extend the ceiling measurement.
