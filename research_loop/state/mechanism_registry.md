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

### MECH-002: reference-query count per KV head → flag `MAX_QUERIES_PER_HEAD` (default: 64 = unchanged)
- Status: tested
- Implements: SCOUT-AM divergence #2 (`AM.pdf` uses **16k–50k** reference queries per KV-head; we
  hard-coded 64) | Targets board entry: **B-CASCADE** (also B-TARGET, B-CAP)
- Files:
  - `examples/qasper2/train/continual_am_sparse.py` — env knob `MAX_QUERIES_PER_HEAD` +
    `_max_queries_kwargs()` (conditional kwarg + loud failure, RUNBOOK §6.10); wandb tag `nq-<N>`
    added **only** when `N != 64`
  - `cartridges/am/value_solve.py::guarded_sparse_am_value_update` — stats now carry
    **`mass_on_S_mean`** (the WORKERS.md standing requirement, previously only in the un-guarded
    solve, which `DELTA_WEIGHT=1e-2` never reaches) + `v_selected_absmax_before/after`,
    `v_selected_absmean_after`, `v_delta_absmax`
  - `cartridges/am/finetune.py::apply_document_am_write_to_cache` — records into
    `AMUpdateStats.extra`: the **pre-subsample** query count per (layer,head)
    (`n_queries_available_*`, i.e. the real accumulator ceiling), the post-subsample count, and
    per-layer `ref_mass_on_S_per_layer` / `v_selected_absmax_after_per_layer`
- What it changes (at the level of the math): `AttentionMatchingFinetuningConfig.max_queries_per_head`
  was pinned at 64 with no env knob, so the per-document solve fitted `n=64` rows against `t=top_t`
  columns — exactly determined at `top_t=64`, underdetermined at `top_t=128`. The knob lets `n ≫ t`,
  making `X_new` in `min ‖A_new·V_S − R_new‖² + w‖V_S − V_S^old‖²` tall instead of square. **The
  `randperm` subsample was not touched**: it still draws from the full pool, so the RNG stream advances
  identically for any cap and the `n=64` draw is an exact *prefix* of every larger draw (verified);
  targets in the per-document path are computed *after* the subsample from the subsampled queries, so
  the DIAG-WIRE alignment trap does not arise there.
- **Accumulator ceiling (asked for explicitly): 57 344 – 81 920 real reference queries per KV head per
  document.** `continual.py:172-182` keeps every token of every packed reference batch
  (`queries_per_batch='all_tokens'`, 4 q-heads per KV head), so the hard-coded 64 was discarding
  ~99.9% of what was already collected. The paper's 16k–50k regime is reachable; `n=16384` ran in
  201.7 s. **Nothing was fabricated or padded in any arm.**
- Sanity check: (i) guarded solve on random `(T=512, d=128, |S|=32)` at n=64/256/1024 — shapes
  preserved, all finite, exactly the 32 selected rows changed, `mass_on_S` 0.0617/0.0635/0.0624;
  (ii) full `apply_document_am_write_to_cache` on a stub cache `(L=3,H=2,T=64,d=16,T_doc=40,|S|=8)` at
  n=64/256/1024 — finite, shapes preserved, 48/48 selected rows written, `extra` carries all 7 new
  keys, `avail=1200 used=64/256/1024`; (iii) `randperm` prefix property `prefix=True` for all pairs and
  post-draw RNG state identical.
- Bit-identical with flag off? **yes.** The driver's 289-line `config.to_dict()` at the default is
  **byte-identical to HEAD's**, and at `n=1024` differs in exactly two entries
  (`max_queries_per_head`, the `nq-1024` tag). The `n=64` GPU control reproduced EXP-007-top32 to all
  16 printed digits on both splits in-run (2.1766157150268555 / 2.5483615398406982), reproduced
  ORACLE-WRITE's standalone control exactly (2.1771795749664307 / 2.5524158477783203), and the Phase-1
  floor control reproduced exactly (2.23880672454834 / 3.7825491428375244). Dual-`cartridges` guard:
  the sibling repo *does* have `max_queries_per_head` but *not* the new instrumentation, so the
  launcher pins `PYTHONPATH` and hard-asserts both the resolved path and the presence of
  `mass_on_S_mean` in the guarded solve (`PREFLIGHT_OK` in every job log).
- Tested by: **MECH-QUERIES** (`WANDB_GROUP=B-CASCADE`), 5 arms at fixed `TOP_T=32`:

  | n | QA | MT | `value_global_max_abs` | cartridge mass MT | `mass_on_S` MT | e2e s |
  |---|---|---|---|---|---|---|
  | Phase-1 | 2.2388 | 3.7825 | — | 0.5879 | 0.0896 | — |
  | **64 (control)** | **2.1772** | **2.5524** | 984 | 0.3285 | 0.0810 | 164.2 |
  | 256 | 2.4097 | 2.7716 | 1328 | 0.3715 | 0.0784 | 153.8 |
  | 1024 | 2.2575 | 2.5744 | 1968 | 0.6399 | 0.0842 | 171.9 |
  | 4096 | 2.3412 | 2.7114 | 3344 | 0.6395 | 0.0836 | 159.0 |
  | 16384 | 2.3325 | 2.6943 | 7808 | 0.6388 | 0.0839 | 201.7 |

  wandb train: `g4vbe2nd` / `oy9at8n4` / `2oq9i3m2` / `5mbxe35f` / `71d29gbi` (12 eval runs in the bundle)
- Verdict + mechanistic reason: **the knob works and is nearly free; B-CASCADE's joint prediction does
  not hold.** `|v|` **grows** monotonically with `n` (984 → 7808, 8× *away* from the teacher's |63|),
  because `guarded_sparse_am_value_update` stacks `[X_new (n×t) ; √w·I (t×t)]` — the normal equations
  are `(X_newᵀX_new + w·I)V = X_newᵀR + w·V_old`, so `X_newᵀX_new` scales with `n` while the
  `DELTA_WEIGHT` trust region stays fixed at 32 rows and its relative pull decays like `1/n`; the
  spectral-scaled ridge (`λ ∝ σ_max(X)²`, itself growing with `n`) does not compensate. Reproduced at
  unit level (`|v|max` 3.52 → 5.80 over the same sweep). Meanwhile the routing collapse **is** repaired
  and overshoots — total eval-time cartridge attention on MT 0.329 → 0.640 vs Phase-1's 0.588 — and
  **MT still does not improve at any n** (best MT in the sweep is the n=64 control; n=1024's +0.022 is
  inside the ±0.1–0.2 noise band, n=256/4096/16384 are +0.219/+0.159/+0.142 and worse). So the three
  quantities B-CASCADE bound together are **decoupled**. Cost is flat (256× the queries for 1.23× wall
  clock), so this is a quality-dominated point, not a cost-dominated one. Confound not separated: `n`
  was swept at a **fixed** `DELTA_WEIGHT=1e-2`, so "more queries don't help" cannot be told apart from
  "more queries help but the 1/n decay of the trust region cancels it". Board call is the
  orchestrator's.
- Kept in tree? yes — opt-in, default 64, bit-identical when unset; the `mass_on_S` instrumentation on
  the guarded path is now the only place that number is recorded for the canonical `DELTA_WEIGHT>0`
  configuration.

### MECH-003: true rotary base in the AM teacher path → flag `AM_ROPE_THETA` (default: unset = 10000.0)
- Status: tested
- Implements: **B-ROPE** (a confirmed correctness bug found by the orchestrator; no external LIT source)
  | Targets board entry: **B-ROPE** (and, by consequence, B-OBJ)
- Files:
  - `cartridges/am/finetune.py` — new config field `rope_theta: float = 10000.0`; a local
    `rope_theta = float(getattr(config, "rope_theta", 10000.0))` in
    `apply_document_am_write_to_cache`; threaded into `teacher_rope_kwargs` (→
    `compute_teacher_targets` **and** `compute_teacher_log_mass`), into `rewrite_keys_on_support`,
    and into `oracle_teacher_value_write`; recorded in `AMUpdateStats.extra["rope_theta"]`
    **only when != 10000.0**
  - `cartridges/am/value_solve.py::oracle_teacher_value_write` — new `rope_theta` kwarg, forwarded to
    the teacher `compute_attention_weights` that drives `assign="mass_ranked"`
  - `examples/qasper2/train/continual_am_sparse.py` — env knob `AM_ROPE_THETA` (float, or
    `model`/`auto` to read `AutoConfig.from_pretrained(MODEL_NAME).rope_theta`), `_resolve_rope_theta()`
    + `_rope_theta_kwargs()` (conditional kwarg + loud failure, RUNBOOK §6.10); wandb tag
    `ropetheta-<v>` added **only** when the env var is set
- What it changes (at the level of the math): the teacher target is
  `Attn(q; [K_cart ‖ K_doc], [V_cart ‖ V_doc])`. The cartridge block is scored with the **unrotated**
  reference query (`core.py:65`), i.e. in exactly the frame the student will see at eval; the document
  block is scored with the query rotated forward by `doc_rope_offset = T_doc` (`finetune.py:501`,
  `core.py:66-72`) so the document sits immediately *before* the query. That composition is only a
  rotation to absolute position `512 + T_doc + i` when the extra rotation uses the **model's own**
  rotary base — `R_θ(a)∘R_θ(b) = R_θ(a+b)` requires equal θ. The AM package hard-coded θ = 10000.0
  while `Qwen/Qwen3-4B-Instruct-2507` uses **θ = 5e6**, so the composite per-frequency-pair angle was
  `(512+i)/5e6^{2k/d} + T_doc/1e4^{2k/d}` — not any single position's rotation. The flag threads the
  true base through; the default keeps the historical value.
- Sanity check (`research_loop/results/DIAG-ROPE/sanity_rope.py`, CPU, no GPU): (i) at offset
  `T_doc=4000`, `_apply_rope_offset_to_queries` with the default is **bit-identical** to explicit
  10000.0, and mean cos(θ=1e4-rotated, θ=5e6-rotated) = **0.1020**; teacher targets are bit-identical at
  the default and `‖T(5e6)−T(1e4)‖/‖T(1e4)‖ = 1.069`; (ii) full `apply_document_am_write_to_cache` on a
  stub cache (L=2, H=2, T=40, d=128, T_doc=4000, |S|=8): unset and explicit-10000.0 give
  `mean_mse = 0.0218288600` to 10 digits with identical per-layer MSE, `|v|max` and `mass_on_S`;
  θ=5e6 gives `mean_mse = 0.0219145315` (knob is live), all finite; `extra["rope_theta"]` present only
  in the 5e6 arm. Driver config constructs correctly for `AM_ROPE_THETA` unset / `5000000.0` / `model`
  (the last resolves to 5e6 off the HF config).
- Bit-identical with flag off? **yes.** Arm A (`AM_ROPE_THETA` unset) reproduced the canonical top32
  operating point to all 17 printed digits *in-run* — QA **2.1766157150268555** / MT
  **2.5483615398406982**, `mean_mse_last_doc` 0.19807696944925424, `value_global_max_abs` 984.0 — and
  its standalone `eval_forgetting.py` QA (2.1771795749664307) reproduced ORACLE-WRITE's standalone
  control exactly. Its `config.yaml` differs from the ORACLE-WRITE control's only by the new field at
  its default (`rope_theta: 10000.0`) plus the wandb group/notes; the A-vs-B configs differ in exactly
  one numeric field. Dual-`cartridges` guard verified: from `/tmp` with no PYTHONPATH,
  `import cartridges` resolves to the **sibling** repo, which has **no** `rope_theta` field — so the
  kwarg is passed conditionally and `AM_ROPE_THETA` fails loudly rather than silently no-op'ing.
- Tested by: **DIAG-ROPE** (`WANDB_GROUP=B-ROPE`), single variable at canonical top32:

  | arm | QA | MT | mean `am/mean_mse` (16 docs) | Σ per-layer MSE | `value_global_max_abs` | e2e s |
  |---|---|---|---|---|---|---|
  | A θ=1e4 (control) | 2.17662 | 2.54836 | 0.11906 | 4.2861 | 984.0 | 178.4 |
  | B θ=5e6 (model)   | 2.15320 | 2.52961 | **0.01406** | **0.5062** | **178.0** | 169.0 |

  wandb: A `https://wandb.ai/vqtri-purdue-university/SEACrowd/runs/pbx4ryku`,
  B `https://wandb.ai/vqtri-purdue-university/SEACrowd/runs/mxjf2jje`
- Verdict + mechanistic reason: **the bug is real and large in the solve's own units, and invisible in
  CE.** Correcting the rotary base makes the target **8.5× more fittable** (mean MSE 0.11906 → 0.01406,
  better on 15/16 documents) and the write **5.5× gentler** (|v| 984 → 178) — and moves CE by
  **−0.023 QA / −0.019 MT**, an order of magnitude inside the ±0.1–0.2 noise floor. Most strikingly it
  **dissolves the layer-34/35 residual** that DIAG-OBJ-c named as the binding constraint: L34
  2.3067 → 0.0898 (0.039×), L35 1.0290 → 0.0075 (0.0073×), their share of the residual 77.8% → **19.2%**
  — that "unfittable" structure was a rotary-frame artefact, not a capacity limit. So B-OBJ's
  conclusion is *strengthened*, not overturned: a target that is now 8.5× better fitted still buys no
  acquisition. Board call is the orchestrator's.
- Kept in tree? yes — opt-in, default 10000.0, bit-identical when unset.
