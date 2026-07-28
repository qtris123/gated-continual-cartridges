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
_(none active yet — the first BUILD dispatch will populate this file)_
