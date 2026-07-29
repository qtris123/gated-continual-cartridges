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

### MECH-004: box-constrained NNLS beta (mass matching) → flags `AM_BETA_BOX` / `AM_NNLS_ITERS` / `AM_NNLS_DRIVER` / `AM_BETA_TARGET` (default: all off)
- Status: tested
- Implements: **LIT-002** (AM paper App. C.2 "Stabilizing β" / Algorithm 3), enabling **LIT-001** (β mass
  matching) | Targets board entry: **B-SOLVE** (primary), **B-ROUTE**
- Files:
  - `cartridges/am/key_select.py::nnls_projected_gradient` — new `upper_bound`, `driver`, `info` kwargs;
    fail-loud finite check on `(Phi, target)` **and** on the returned `w`; NaN warm start replaced by the
    paper's uniform start and *recorded* instead of laundered; box projection `clip(w, lo, hi)` inside the
    PGD loop. Also `::refit_beta_nnls` — new `beta_box`, `nnls_driver`, `target_mode`, `info` kwargs and a
    `_record_beta_info` summarizer.
  - `cartridges/am/finetune.py` — config fields `beta_box: Optional[float] = None`, `nnls_iters: int = 200`,
    `nnls_driver: Optional[str] = None`, `beta_target: Literal["residual","full"] = "residual"`;
    `_should_fit_beta` **decoupled from `key_mode`**; a fail-loud `isfinite(beta_full)` assert before the
    write-back; per-layer β distributions + fit diagnostics into `AMUpdateStats.extra["beta"]` /
    `["beta_per_layer"]`, recorded **only when β actually ran**.
  - `examples/qasper2/train/continual_am_sparse.py` — env knobs `AM_BETA_BOX` (3.0), `AM_NNLS_ITERS` (2),
    `AM_NNLS_DRIVER` (gelsd), `AM_BETA_TARGET` (residual) + `_beta_fit_kwargs()` (conditional kwarg + loud
    failure, RUNBOOK §6.10); wandb tags added only when β is active.
- What it changes (at the level of the math): β is fitted by `min_w ‖Φw − r‖²` with `Φ = exp(qKᵀ/√d − shift)`
  on the selected slots and `r` the teacher's `[cartridge‖doc]` mass minus the untouched slots' mass, then
  `β = log w`. The three LIT-002 fixes are (1) a **rank-revealing** lstsq warm start instead of `gels`,
  (2) a **two-sided box** `w ∈ [e^-3, e^3] ⇔ β ∈ [−3,3]` instead of a `1e-12` floor alone (which put β at
  −27.6), (3) `AM_BETA_TARGET=full` to fit against the full teacher mass as the paper does. `AM_NNLS_ITERS`
  drops the PGD budget 200 → 2 (the paper's value).
- ⚠️ **Correction to LIT-002, found on GPU:** `torch.linalg.lstsq(..., driver='gelsd')` is **CPU-only**;
  on CUDA input it *raises*. The old `except RuntimeError` would then have silently fallen back to a
  uniform warm start. The implementation retries the warm start on the CPU with the requested driver
  (`info["warm_start_on_cpu"]`) — the matrices are 64×32, so the round trip costs ~1.3× `solve_s`.
- Sanity check (`research_loop/results/MECH-BETA/sanity_beta.py`, run on CPU **and** CUDA →
  `sanity_cuda.json`): (i) `nnls_projected_gradient` and `refit_beta_nnls` at default arguments are
  **bit-identical** to `git show HEAD:cartridges/am/key_select.py` (max|Δ| = 0.0; note the *first* lstsq
  call in a process differs from later ones by ~4e-8 on both old and new code, so the comparison burns one
  call first); (ii) on a rank-8 64×64 `Φ` the old path returns β ∈ [−27.63, +12.99] (CUDA) while the boxed
  `gelsd` path returns finite β with `lstsq_rank = 8` correctly reported; (iii) `refit_beta_nnls` on a
  rank-deficient synthetic head returns β ∈ [−3.0, +3.0], median 1.90, non-selected slots untouched;
  (iv) full `apply_document_am_write_to_cache` on a stub cache: β off → `mass_on_S` 0.2073/0.2078 with
  all-zero β and bias disabled; β on → finite β, bias enabled, `mass_on_S` 0.8252/0.8231 (**3.98×**),
  `resid_clamp_frac = 0`; (v) `_should_fit_beta`: freeze+unset **False**, freeze+ENABLE_BETA=1 **True**,
  highest_attention+unset **False** (was True — the H4 hazard).
- Bit-identical with flag off? **yes.** The MECH-BETA control arm reproduced EXP-007-top32 / DIAG-ROPE arm A
  to all 17 in-run digits (QA **2.1766157150268555** / MT **2.5483615398406982**, `mean_mse_last_doc`
  0.19807696944925424, `|v|max` 984.0), and its standalone `eval_forgetting.py` numbers
  (QA 2.1771795749664307 / MT 2.5524158477783203) match DIAG-ROPE's exactly. The rope-only arm likewise
  reproduced DIAG-ROPE arm B exactly (QA 2.15320086479187 / MT 2.5296061038970947, `|v|max` 178.0).
- Tested by: **MECH-BETA** (`WANDB_GROUP=B-SOLVE`), canonical top32, `KEY_MODE=freeze`:

  | arm | QA (in-run) | MT (in-run) | eval `mass_on_S` MT | MT/QA mass | cart mass MT | mean `am/mean_mse` | \|v\|max | solve_s |
  |---|---|---|---|---|---|---|---|---|
  | control θ=1e4, β off | 2.17662 | 2.54836 | 0.08098 | 1.047 | 0.3285 | 0.11906 | 984.0 | 177.4 |
  | rope θ=5e6, β off | 2.15320 | 2.52961 | 0.07111 | 1.043 | 0.5930 | 0.01406 | 178.0 | 184.3 |
  | **β on, θ=5e6, box 3** | **2.73637** | **2.81193** | **0.30101** | **1.051** | 0.6785 | **0.00782** | 107.5 | 241.4 |
  | Phase-1 reference | 2.2388 | 3.7825 | 0.08961 | 1.083 | 0.5879 | — | — | — |

  wandb: control `https://wandb.ai/vqtri-purdue-university/SEACrowd/runs/2a39ix5a`,
  rope `.../runs/zl1g5jtf`, β `.../runs/fk4n72af`.
- Verdict + mechanistic reason: **the mechanism works and the hypothesis it serves does not.** β is no
  longer numerically broken — 4608 fits, zero NaN, every β inside the paper's box — and it does exactly
  what LIT-001 says it should: eval-time `mass_on_S` on MT rises **0.0711 → 0.3010 (4.23×)**, past
  Phase-1's 0.0896, on **all 36 layers**. **Acquisition does not follow: MT regresses +0.282** (2.5296 →
  2.8112) and QA regresses **+0.581** (2.1597 → 2.7403, past the 2.52 retention budget). The reason is
  that β is **query-independent**: it maps MT and QA mass through the same monotone function, so the
  MT/QA `mass_on_S` ratio moved only 1.043 → 1.051 (Phase-1 1.083) — a 4.2× bandwidth gain bought +0.008
  of selectivity. β saturates at the ceiling (59.7% of fitted entries at +3, per-layer medians 2.22–3.00),
  so the box is the binding constraint and a larger box would buy more of the same mass with the same
  null selectivity. The write itself got *better* by its own metric while CE got worse (`am/mean_mse`
  0.01406 → 0.00782, `|v|max` 178 → 107.5) — the B-OBJ anti-correlation again.
- Kept in tree? yes — opt-in, all four flags default to the historical behaviour, bit-identical when off.

---
### MECH-005: RoPE counter-rotation on key install (B-ROPE hazard **H2**) → flag `AM_KEY_REPOSITION` (default: unset = off)
- Status: tested
- Implements: **LIT-026** (hazard 2: "`rewrite_keys_on_support` selects doc keys in one rotary frame and
  installs them in another with no counter-rotation"; `AM.pdf` App. C.3's uniform phase shift
  `R_Δ`, `Δ = p_target − p_source`) · negative control from **LIT-027**
  | Targets board entry: **B-ROUTE** (key side), **B-ROPE**
- Files:
  - `cartridges/am/key_select.py::rewrite_keys_on_support` — new `reposition: bool = False` and
    `info: Optional[dict] = None` kwargs. After selection, rows whose candidate index is
    `>= doc_key_start` (i.e. **document**-sourced) are re-based by `phase1._rope_reposition`
    (lazy import; `phase1` imports `key_select`, so a module-level import would be circular) from
    `from_pos = m` to `to_pos = m − doc_rope_offset`, in **float32**, then cast back. Fail-loud shape
    and `isfinite` assertions. Cartridge-sourced rows are untouched. `info` records `n_selected`,
    `n_from_doc`, `n_from_cartridge`, `n_changed`, `n_repositioned`, `rope_delta`.
  - `cartridges/am/finetune.py` — new config field `key_reposition: bool = False`; a local
    `key_reposition = bool(getattr(config, "key_reposition", False))` in
    `apply_document_am_write_to_cache`, threaded into `rewrite_keys_on_support` together with the
    MECH-003 `rope_theta`; per-(layer,head) `info` dicts aggregated into
    `AMUpdateStats.extra["key_rewrite"]` / `["key_rewrite_per_layer"]` **only when the rewrite ran**
    (`key_mode != "freeze"`), so every frozen-key run keeps a byte-identical `am_doc_*.pt` payload.
  - `examples/qasper2/train/continual_am_sparse.py` — env knob `AM_KEY_REPOSITION`,
    `_key_reposition_kwargs()` (conditional kwarg + loud failure, RUNBOOK §6.10) which additionally
    **refuses** `AM_KEY_REPOSITION=1` with `KEY_MODE=freeze` (no key is ever installed) and **refuses**
    it without an explicit `AM_ROPE_THETA` (the counter-rotation is only correct in the model's own
    rotary frame — at θ=1e4 it is *worse than doing nothing*, see the sanity numbers). wandb tag
    `keyrepos-<0|1>` added **only** when the env var is set.
- What it changes (at the level of the math): the candidate pool is `[K_cart[S] ‖ K_doc]` and
  `core._attention_scores` scores the two blocks in **different rotary frames** — the cartridge block
  against the raw reference query `q`, the document block against `R_Δ q` with `Δ = doc_rope_offset =
  T_doc`. A document key that wins selection was therefore chosen for the logit `⟨R_Δ q, k_doc⟩`, but
  once installed into a cartridge slot the student (and eval) scores it with the **raw** `q`, giving
  `⟨q, k_doc⟩` — a rotation by thousands of positions away from the logit it was picked for, and the
  value solve is then fitted against that corrupted routing. Because RoPE is orthogonal,
  `⟨R_Δ q, k⟩ = ⟨q, R_{−Δ} k⟩`, so installing `R_{−Δ} k_doc` restores the intended logit exactly. In
  absolute-position terms the document row `m` (prefilled at position `m`) behaves as a key at
  `m − T_doc` in the cartridge frame, and that is where it is re-based.
- Sanity check (`research_loop/results/MECH-KEYS/sanity_keys.py`, run on CPU → `sanity_cpu.json` and
  on CUDA → `sanity_cuda.json`; both agree):
  (i) **the headline numeric check** — for the 16 document-sourced installs at `T_doc=4000`, θ=5e6, the
  gap between the logit the installed key *delivers* (student frame, raw query) and the logit the
  selector *used* falls from **max 1.0540 / mean 0.2358** to **max 4.77e-07 / mean 4.99e-08** in float32,
  and to **max 5.52e-03 / mean 6.23e-04** in the cache's bf16, on a score scale `max|s| = 2.779`;
  cartridge-sourced rows are **exactly 0.0** in both arms;
  (ii) at the primitive level over this project's real document lengths `T_doc = 3858 / 4000 / 8900`,
  relative error **1.041 / 0.945 / 1.203 → 1.98e-07 / 2.03e-07 / 2.15e-07** and correlation with the
  intended logit **0.510 / 0.430 / 0.327 → 1.000**;
  (iii) **θ matters**: the same counter-rotation done at the AM package's historical 10000.0 gives max
  errors **4.43 / 5.02 / 5.11 — larger than not correcting at all**, which is why the driver refuses
  `AM_KEY_REPOSITION=1` without an explicit `AM_ROPE_THETA`;
  (iv) end-to-end on a stub cache (L=2, H=2, T=40, T_doc=4000, |S|=8): `freeze`, `highest_attention`
  ±reposition and `omp`+reposition all finite, with `extra["key_rewrite"]` populated only for the
  non-freeze arms;
  (v) driver config probe (`config_probe.json`) — all four arms construct against the `_explore`
  package with the expected fields; both guards fire; and **unpinned** (sibling `cartridges`) the run
  fails loudly instead of silently no-op'ing.
- Bit-identical with flag off? **yes.** `rewrite_keys_on_support` at default arguments is bit-identical
  to `git show HEAD:cartridges/am/key_select.py` for **both** `highest_attention` and `omp`
  (max|Δ| = 0.0, old-vs-old check passed first to burn the first-lstsq BLAS difference); the freeze
  end-to-end write is bit-identical to a `git archive HEAD` snapshot executed in a **separate
  interpreter** (mean_mse, per-layer MSE, key/value sums, |v|max, `mass_on_S` and the `extra` key set
  all equal); and the GPU control arm reproduced **DIAG-ROPE arm B to all 17 printed digits** in-run
  (QA **2.15320086479187** / MT **2.5296061038970947**), with mean `am/mean_mse`
  **0.014059890443260059** and `|v|max` **178.0** identical to MECH-BETA's rope arm, and standalone
  QA 2.159724712371826 / MT 2.529625177383423 identical to that run's standalone numbers.
- Tested by: **MECH-KEYS** (`WANDB_GROUP=B-ROUTE`), canonical top32 at θ=5e6, `ENABLE_BETA=0`:

  | arm | QA (standalone) | MT (standalone) | eval `mass_on_S` MT | **MT/QA mass** | cart mass MT | mean `am/mean_mse` | \|v\|max | doc keys installed /32 | solve_s |
  |---|---|---|---|---|---|---|---|---|---|
  | control `freeze` | 2.15972 | 2.52963 | 0.0711 | **1.0428** | 0.5930 | 0.01406 | 178.0 | — | 182 |
  | `highest_attention`, repos **0** | 2.12011 | 2.42385 | **0.3756** | **1.0485** | 0.7204 | 0.00384 | **63.25** | 8.91 (27.8%) | 325 |
  | **`highest_attention`, repos 1** | **2.03492** | **2.33050** | 0.2430 | **1.0510** | 0.6711 | 0.00394 | 74.0 | 12.91 (40.3%) | 272 |
  | `omp`, repos 1 | 2.11686 | 2.39698 | 0.1918 | **1.0638** | 0.6454 | 0.00484 | 141.0 | 17.86 (55.8%) | **3857** |
  | Phase-1 reference | 2.2388 | 3.7825 | 0.0878 | 1.0812 | 0.5879 | — | — | — | — |

  wandb: control `https://wandb.ai/vqtri-purdue-university/SEACrowd/runs/6cr40yfg`,
  keys-norepos `.../runs/ftsdkn1m`, keys-repos `.../runs/f28mqklc`, omp `.../runs/vdad238z`.
- Verdict + mechanistic reason: **the fix is real, it is worth −0.093 MT / −0.085 QA on top of the
  uncorrected key write, and the key axis as a whole is the first lever in this loop that improves BOTH
  axes at once.** The pre-loop folklore that key rewriting collapses QA (RUNBOOK §7) is **false in this
  setting**: every key arm beat the frozen-key control on retention as well as acquisition, and the best
  arm (`highest_attention` + reposition) reaches **MT 2.3305 / QA 2.0349** — below ORACLE-WRITE's
  *perfect value write* ceiling (2.381) and below the k=12 snapshot minimum (2.435), at 0 gradient steps.
  **But it does NOT work through selectivity.** The eval-time MT/QA `mass_on_S` ratio moved
  1.0428 → 1.0510 (+0.0082), the same order as β's +0.008 and still **below the untouched Phase-1
  cartridge's own 1.0812**. What moved is bandwidth (MT `mass_on_S` 0.0711 → 0.2430, and 0.3756 in the
  no-reposition arm) — and unlike β, which bought 4.2× bandwidth query-independently and made CE worse,
  the key write buys its bandwidth by **moving the keys themselves into the document's directions**, so
  the extra mass lands on slots whose *content* is the document. Part of the reposition/OMP bandwidth is
  a wider written support (87.6 / 105.2 union slots per layer vs the control's 53.4), but the
  no-reposition arm sits at **51.6 slots/layer — smaller than the control — and still carries 5.3× the
  mass**, so the gain is not a support-size artefact. OMP installs the most document keys (55.8% of
  147,456 slot-writes) and the highest ratio (1.0638) yet lands **worse** than `highest_attention` on
  both axes at **21× the solve cost** (3857 s vs 182 s), because our `key_select.py` runs a
  200-iteration NNLS inside each of its 32 greedy steps (LIT-002/LIT-027). ⚠️ Noise discipline:
  MT n=69 / QA n=78, so ΔMT −0.199 sits at the **top edge** of the 0.1–0.2 band and ΔQA −0.125 sits
  **inside** it; one seed, no seed variation was run, and MT 2.3305 is still 0.31 above the 2.02 bar.
- Kept in tree? yes — opt-in, default off, bit-identical when off.

---
### MECH-006: on-policy layer-sequential re-extraction → flags `AM_ONPOLICY_LAYERS` / `AM_ONPOLICY_DOCKV` (default: unset = off)
- Status: tested
- Implements: **LIT-006** (AM paper §3.1 "On-policy queries", App. C.4) = SCOUT-AM's **divergence #3**
  | Targets board entry: **B-CASCADE / query-distribution** — the one escape hatch DIAG-KEYSPACE
  explicitly did **not** bound, and the last unimplemented thing the paper does.
- Files:
  - `cartridges/am/finetune.py` — new config fields `onpolicy_layers: int = 0` and
    `onpolicy_refresh_doc_kv: bool = False`; new module-level `_onpolicy_refresh()` (validation +
    drift measurement + logging); `apply_document_am_write_to_cache` gains one optional kwarg
    `onpolicy_refresh_fn` and, at the top of its layer loop, rebinds `query_accumulator` (and
    optionally `doc_kv`) whenever `layer_idx > 0 and layer_idx % onpolicy_layers == 0`. Fails loudly
    if `onpolicy_layers > 0` arrives without a hook. Records `extra["onpolicy"]` /
    `extra["onpolicy_events"]` **only when the mechanism ran**, so every stock run keeps a
    byte-identical `am_doc_*.pt` payload.
  - `cartridges/am/continual.py` — inside the per-document loop, builds `_onpolicy_refresh_fn`
    (closure over `wrapped_model`, `cache`, `doc_loader`, and — for the doc-KV axis — `model`,
    `tokenizer`, `system_prompt`) and passes it **only when the knob is on** (RUNBOOK §6.10 /
    MECH-000: an unconditional new kwarg once crashed every AM run).
  - `examples/qasper2/train/continual_am_sparse.py` — env knobs `AM_ONPOLICY_LAYERS` (group size,
    0/unset = off) and `AM_ONPOLICY_DOCKV` (0/1), `_onpolicy_kwargs()` (conditional kwarg + loud
    failure; refuses `AM_ONPOLICY_DOCKV=1` without `AM_ONPOLICY_LAYERS>0`, refuses a negative group,
    refuses any `AM_EXECUTION_MODE != per_document`); wandb tag `onpolicy-<N>` added **only** when
    the env var is set.
- What it changes (at the level of the math): we collect `Q_ref` in **one** forward pass over the
  *pre-write* cartridge and then solve all 36 layers against it — but writing layer `l` perturbs the
  residual stream, so the queries the model actually emits at layers `l+1…35` are not the ones we
  fitted. With `onpolicy_layers = N` the write is split into `⌈36/N⌉` groups and, before each group
  after the first, the reference queries are re-extracted by replaying the same reference dataloader
  against the **partially-written** cache, so each layer is solved against the activations it will
  actually see. The teacher's cartridge block was already on-policy (layer `l` reads the current
  `k_param`/`v_param`); the *document* block is not, which is what the second flag addresses.
  Slot selection is deliberately **not** re-ranked, so the only variable is the query distribution.
- **Group size = 4**, i.e. refreshes before layers 4,8,…,32 (8 per document). Per-layer would be 35
  passes/doc; the brief's range was 4–6 and 4 is the finer end. Measured price: 2.6–2.9 s per
  refresh, 21–22 s per document, 342–353 s per 16-document run.
- Sanity check (`research_loop/results/MECH-SEQUENTIAL/sanity_seq.py`, run on CPU → `sanity_cpu.json`
  and on CUDA → `sanity_cuda.json`; both agree exactly):
  (i) **the decisive check** — with `onpolicy_layers=1` and a refresh that returns the *same*
  queries, every number is bit-identical to OFF (`mean_mse`, per-layer MSE, k/v sums, |v|max,
  `mass_on_S`, `n_queries`), and the only difference is the two new diagnostic keys; measured drift
  is cos = 1.0, rel-L2 = 0.0. So the restructuring changes **the queries and nothing else**;
  (ii) passing a refresh function that *raises* with the knob off is bit-identical to OFF — the hook
  cannot fire; (iii) with perturbed queries the write changes, stays finite, and `extra["onpolicy"]`
  records the expected refresh layers for group sizes 1 / 2 / 4; (iv) **fail-loud**: wrong query
  shape, empty layer, wrong batch count, non-finite queries, a non-tuple return, a wrong-shaped
  doc-KV refresh, and `onpolicy_layers>0` with **no** hook all raise (7/7);
  (v) driver config probe (`config_probe.json`) — all four arms construct against the `_explore`
  package, all three guards fire, and **unpinned** (sibling `cartridges`, which has neither field)
  the run fails loudly instead of silently no-op'ing.
- Bit-identical with flag off? **yes.** The stub write with the flag off is bit-identical to a
  `git archive HEAD` snapshot executed in a **separate interpreter**, and on GPU the control arm
  reproduced MECH-KEYS' control to all 17 printed digits in-run (QA **2.15320086479187** / MT
  **2.5296061038970947**, mean `am/mean_mse` **0.014059890443260059**, `|v|max` **178.0**) *and*
  standalone (QA **2.159724712371826** / MT **2.529625177383423**) — Δ = 0.0 on all four.
- Tested by: **MECH-SEQUENTIAL** (`WANDB_GROUP=B-CASCADE`), canonical top32 at θ=5e6, `ENABLE_BETA=0`:

  | arm | QA (standalone) | MT (standalone) | eval `mass_on_S` MT | **MT/QA mass** | **cart mass MT** | mean `am/mean_mse` | \|v\|max | solve_s | ×control |
  |---|---|---|---|---|---|---|---|---|---|
  | control (`freeze`, on-policy off) | 2.15972 | 2.52963 | 0.0711 | 1.0428 | 0.5930 | 0.014060 | 178.0 | 164.8 | 1.00 |
  | on-policy g=4, frozen keys | 2.13984 | 2.50850 | 0.0720 | 1.0413 | 0.5918 | 0.014377 | 169.0 | 586.2 | **3.56×** |
  | **on-policy g=4 + MECH-005** | 2.09319 | 2.38128 | 0.2388 | 1.0554 | 0.6714 | 0.004428 | 75.5 | 726.4 | **4.41×** |
  | MECH-005 alone (reference) | **2.03492** | **2.33050** | 0.2430 | 1.0510 | 0.6711 | 0.003940 | 74.0 | 272.0 | 1.65× |
  | Phase-1 reference | — | — | 0.0878 | 1.0812 | 0.5879 | — | — | — | — |

  wandb: control `https://wandb.ai/vqtri-purdue-university/SEACrowd/runs/5fk6z5oz`,
  on-policy `.../runs/05keqbxl`, on-policy+keys `.../runs/3sag81hz` (+14 standalone eval runs, see
  `results/MECH-SEQUENTIAL/curve.tsv`).
- Verdict + mechanistic reason: ❌ **the mechanism works, is measured, and buys nothing — and it
  identifies why.** (1) **The cross-layer activation shift is real but tiny.** Per group boundary the
  on-policy queries differ from the stale ones by mean cosine **0.99817** / mean relative L2
  **4.98%**, and the drift **does not widen with depth** — rel-L2 by boundary layer is
  9.20/2.55/3.61/7.95/6.57/5.44/2.47/2.05% at L4…L32, *largest at the shallowest boundary*, directly
  contradicting LIT-006's prediction. (2) **The pathology this entry was opened for no longer
  exists.** B-CASCADE was opened on ORACLE-WRITE's "the write collapses total cartridge attention
  0.588 → 0.329". At the model's true rotary base (MECH-003) the control already sits at
  **0.5930 vs Phase-1's 0.5879** — the collapse was a θ=1e4 artefact. On-policy moves it by
  **−0.0012**. There was nothing left to repair *by construction*. (3) **Both axes move by ~1/5 of
  the noise band and the composition is on the wrong side:** on-policy alone −0.020 QA / −0.021 MT;
  composed with the best point it is **+0.058 QA / +0.051 MT worse** than MECH-005 alone. Selectivity
  is untouched (MT/QA ratio 1.0428 → 1.0413 frozen-key, still below Phase-1's 1.0812) and the solve's
  own objective gets *slightly worse* (0.014060 → 0.014377). (4) **Cost is 3.6–4.4× `solve_s`** for
  that null. ⚠️ Noise discipline: MT n=69 / QA n=78; every delta reported here is **inside** the
  0.1–0.2 band, so the honest reading is "no effect", not "a small effect". One seed, no seed
  variation. k-curve (k ∈ {8,12,16}): the frozen-key on-policy arm is **monotonically worse in k**
  (MT 2.4733/2.4938/2.5085) and the composed arm is non-monotone with its minimum at k=8 (MT
  2.3247/2.4483/2.3813) — on-policy does **not** remove the late-document degradation, and only three
  k were measured. The `AM_ONPOLICY_DOCKV` axis is implemented and unit-tested but **not run on GPU**.
- 🔑 **Matched-k comparison against DIAG-KEYCURVE's `keys_repos` curve (same harness, same splits) —
  the only delta in this experiment that escapes the noise floor, and it points the wrong way.**
  Adding on-policy to MECH-005 gives MT **2.3289 → 2.3247 (−0.004)** at k=8, **2.2720 → 2.4483
  (+0.176)** at k=12, **2.3305 → 2.3813 (+0.051)** at k=16, and QA +0.001 / **+0.188** / +0.058.
  At **k=12 — the k at which MECH-005 is actually best (QA 1.9560 / MT 2.2720)** — on-policy costs
  **+0.176 MT and +0.188 QA**, at or above the top of the 0.1–0.2 band. Everywhere else it is inside
  the band. There is no k at which on-policy helps.
- Kept in tree? yes — opt-in, both flags default off, bit-identical when off.

### MECH-007: opt-in per-document reference-draw seed offset → flag `AM_SEED_OFFSET` (default: 0 = off)
- Status: **tested** (MECH-SEED, 2026-07-29) · Targets board entry: **the confirmation standard itself**
  (PLAN_AM_MUST_WIN.md §3) — and, through it, **BEST GRADIENT-FREE POINT** / B-ROUTE (key side).
- Implements: no external LIT entry. This is infrastructure the mission's own PASS criterion requires and
  that DIAG-KEYCURVE proved by direct test did not exist (`seed_probe.json`): `continual_am_sparse.py`
  reaches `pydrantic.main` **only** in `AM_EXECUTION_MODE=train_loop`, so a `seed=<n>` CLI override never
  reaches `config.seed` in the canonical `per_document` path (probe: it stayed 42); there is no `SEED` env
  knob; and `cartridges/am/continual.py:150` seeded the per-document reference draw with `seed=doc_idx`,
  a constant. **Every number in this project was one seed.**
- Files:
  - `cartridges/am/finetune.py` — `AttentionMatchingFinetuningConfig.seed_offset: int = 0` (+13 lines)
  - `cartridges/am/continual.py::run_per_document_am_phase2` — reads it via
    `getattr(config, "seed_offset", 0)` (so a sibling-package config still works), raises on a negative
    value, logs once when non-zero, and derives `draw_seed = doc_idx + seed_offset` — used for **both**
    `limit_conversations(...)` (which conversations are drawn) and `build_reference_dataloader(...)`
    (packing/ordering of the drawn subset). Records `aggregate["seed_offset"]` **only** when non-zero, so
    a stock `per_document_am_stats.pt` keeps exactly its historical keys. (+34 lines)
  - `examples/qasper2/train/continual_am_sparse.py` — `AM_SEED_OFFSET` env var and a
    `_seed_offset_kwargs()` helper wired into `_build_am_config()`, following the repo's
    conditional-kwarg discipline (RUNBOOK §6.10 / MECH-000): the kwarg is omitted entirely when the env
    var is unset, and if it IS set while the imported `cartridges` lacks the field the driver **raises**
    rather than silently running the historical seed. (+45 lines)
- What it changes (at the level of the math): nothing in the solve. It changes only **which 32 of a
  document's ~500 synthesised conversations** form the reference set `Q_ref` that the closed-form
  value/key write is fitted to — the single stochastic choice the per-document AM path contains. DIAG-PERDOC
  had already measured that this choice is a genuine source of variance (0.216 spread on a per-document
  subset loss), which is why it is the right thing to vary.
- Sanity check (CPU, no GPU): 6-case config-construction matrix — flag unset → `seed_offset=0`; `=0` →
  0 + the `MECH-007:` log line; `=1000` → 1000; **HEAD-snapshot package + `=1000` → RuntimeError**
  (fail-loud, as designed); HEAD-snapshot package + unset → constructs fine with the field absent (the
  sibling-import path is not broken); `=-5` → ValueError. Plus a draw probe on the real MT parquet:
  offset 0 vs 1000 shares **2.19 of 32** conversations per document (Jaccard 0.036), 0 vs 2000 shares
  **2.63 of 32** (Jaccard 0.043), and **0 of 16 documents** keep the same draw. The knob is not cosmetic.
- **Bit-identical with flag off? YES — proven two ways.** Three runs of the identical keys+reposition
  configuration: A1 = `git archive HEAD` snapshot (no `seed_offset` field at all), A2 = edited tree with
  `AM_SEED_OFFSET` unset, A3 = edited tree with `AM_SEED_OFFSET=0` explicit. **All 17 cartridge artefacts
  (16 `cache-after-doc-*.pt` + `cache_last.pt`) are sha256-identical across all three**, as are
  `am/mean_mse` (0.0039414096599943195), `|v|max` (78.5) and the in-run evals (QA 2.0349531173706055 /
  MT 2.3290133476257324). The only difference anywhere in the tree is **one line of config.yaml**,
  `seed_offset: 0` — unavoidable when adding a pydantic field, and numerically inert.
- Tested by: **MECH-SEED** (this build's own W5 half) → the mission's best point (MECH-005 keys+reposition
  at k=12) run at offsets {0, 1000, 2000}, both splits, k ∈ {12, 16}, against the mechanism-off control
  (`KEY_MODE=freeze`) at the same three offsets.

  | arm | k | MT @off0 | MT @off1000 | MT @off2000 | mean | range |
  |---|---|---|---|---|---|---|
  | keys+repos | 12 | 2.27202 | 2.26954 | 2.26263 | **2.26806** | **0.0094** |
  | keys+repos | 16 | 2.33050 | 2.30624 | 2.29011 | 2.30895 | 0.0404 |
  | control `freeze` | 12 | 2.47046 | 2.44861 | 2.51633 | 2.47846 | 0.0677 |
  | control `freeze` | 16 | 2.52963 | 2.48238 | 2.61136 | 2.54112 | 0.1290 |

  **ΔMT (keys − control) at matched k:** k=12 → −0.1984 / −0.1791 / −0.2537 (mean **−0.2104**);
  k=16 → −0.1991 / −0.1761 / −0.3212 (mean **−0.2322**). **All six same sign, all six outside the ±0.15
  band.** Adversarially paired (keys' worst seed vs control's best seed) it is still −0.1766 at k=12 and
  −0.1519 at k=16. **ΔQA** at k=12 is −0.1181 / −0.1125 / −0.1262 (mean −0.1189) — consistently negative,
  consistently **inside** the band at every seed; at k=16 the mean is −0.1658 but only offset 2000 clears.
  wandb (group `VERIFY`): A1 `790bm7pf`, A2 `i0s58v1k`, A3 `xn2qekrp`, keys@1000 `qts8j7kw`,
  keys@2000 `t1p85sr0`, control@1000 `nmratchj`, control@2000 `buqjv0fy`, + 26 eval runs (`curve.tsv`).
- Verdict + mechanistic reason: ✅ **supported — and it retires the blocker.** The knob does exactly the
  one thing that was missing, at zero cost to every existing configuration, and the first thing it was
  used for produced a clean answer: **the seed-to-seed spread of the point under test (0.0094 MT at k=12)
  is 22× smaller than the keys-vs-control effect it was meant to test (−0.2104), so the MT advantage of
  key installation is NOT a seed artefact.** The mechanism's *internal* signature is seed-stable too —
  solve-time `ref_mass_on_S` 0.3136/0.3615/0.3587 (keys) vs 0.1452/0.1460 (control), a 2.4–2.5× ratio at
  every seed; `am/mean_mse` 0.00394/0.00353/0.00348 vs 0.01424/0.01353; `|v|max` 78.5/77.0/88.0 vs
  139/162; solve cost 296–311 s vs 157–165 s (~1.9×, seed-independent). ⚠️ Two honest limits. (a) The
  **QA** half of the old "beats frozen keys on BOTH axes" headline is now refuted at three seeds rather
  than one — ΔQA never leaves the noise band at k=12. (b) The point is **still not a mission PASS**:
  averaged over seeds it is QA 1.9468 / MT 2.2681, clearing the QA budget by 0.573 but **0.248 short of
  MT ≤ 2.02 at every seed, best seed included**. (c) This varies the *only* stochastic choice the
  per-document path has; it does not vary document order or eval sampling, so it is a **lower bound** on
  total run-to-run variance, and three seeds on an n=69/78 eval are three samples of a coarse instrument.
- Kept in tree? **yes** — opt-in, default 0, bit-identical when default, fail-loud when requested against
  a package that lacks it. Every future result in this project can and should be seed-varied with it.
- ⚠️ **Scope note added after two concurrent workers landed (2026-07-29).** **DIAG-NOISE** measured the
  harness resolution (MT paired 95% = **0.0494**, composite 0.1055), so the matched-k ΔMT above is
  **3.6–6.5× the applicable yardstick** and this job's own seed spread (0.0094) is **5× smaller than the
  resolution** — the seed is not the dominant noise term. But **DIAG-CONTROLCURVE** mapped all 16 control
  k and relocated the control's true MT minimum to **k=10 (2.4083)**, shrinking the **own-optimum** ΔMT to
  **−0.1363** and ΔQA to −0.0115. MECH-SEED seed-varied the **matched-k** comparison (k=12, k=16), **not**
  the own-optimum one. So the seed-confirmed statement is *"keys beat frozen keys at matched k"*; the
  own-optimum claim is weaker and remains un-seed-varied.
