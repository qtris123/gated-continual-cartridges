# LITERATURE LEDGER — mechanisms imported from outside this repo

> **What this file is:** every source a SCOUT worker reads, turned into something testable *here*.
> The loop's premise is that a measured bottleneck usually has a known fix in the literature; this is
> where those fixes enter the project.
>
> **Rule: an entry with no PREDICTION about our measured signature is unusable.** Mark it `unusable`
> and say why, rather than leaving an interesting-sounding paper on the board.

**Status vocabulary:** `candidate` (read, mapped, predicts something) · `queued` (chosen for BUILD) ·
`implemented` (MECH-XXX exists) · `tested` (EXP-XXX ran) · `supported` · `dead` (+ the reason) ·
`unusable` (no mapping or no prediction).

## Template
```
### LIT-XXX: <mechanism name> — <source short title> (arXiv id / link)
- Status: candidate
- Board entry: B-XXXX          ← which measured cause it attacks
- Mechanism (1 paragraph, equations if short):
- Maps to our code: <file>::<function>, proposed opt-in flag `FLAG_NAME`
- Prediction (about OUR signature, not the paper's benchmark):
- Cost: gradient-free? extra solves? extra memory?
- Why it might NOT transfer: (compressed KV inside attention, 512 slots, frozen LM, 2-stage only)
- Related: MECH-XXX / EXP-XXX
```

---
## Priority reading queue (seeded 2026-07-28 — SCOUT is expected to go beyond this)

**Local PDFs, read the algorithm not the abstract:**
- `AM.pdf` — *Fast KV Compaction via Attention Matching* (arXiv 2602.16284). The exact value solve we
  are trying to make win, **including the β / mass-matching step we have never got to run** (B-SOLVE).
  Question for the scout: what does the paper's target actually consist of, and does our
  `per_document` path implement it faithfully? (→ B-WIRE, B-TARGET)
- `TF-IDF.pdf` — *Continual Learning via Sparse Memory Finetuning* (arXiv 2510.15103v1). The gating
  ancestor. Question: what does it rely on that a compressed KV cache inside attention does *not*
  provide?

**External families, by the cause they attack:**
| cause | family | why it's relevant here |
|---|---|---|
| B-ROUTE | **delta rule / fast weights** (DeltaNet, delta-rule linear attention, Fast Weight Programmers) | closed-form associative writes where the *key* is chosen so the write is retrievable — exactly the frozen-key problem |
| B-ROUTE | Hopfield / associative memory capacity | how many writes a fixed-size associative store can hold before interference |
| B-CAP / B-SOLVE | **ROME / MEMIT** | covariance-preconditioned closed-form least-squares editing, many facts at once, no gradients |
| B-CAP / forgetting | **AlphaEdit / null-space-projected editing**, Adam-NSCL | write in the null space of *old* keys ⇒ acquisition without forgetting: our two axes, in one operator |
| B-SOLVE | recursive least squares / Kalman / Sherman-Morrison updates | exact sequential closed-form updates — the natural N-stage extension |
| B-CAP | OMP / matching pursuit | already a `KEY_MODE`; the literature says when it beats top-k selection |
| B-OBJ | output-space distillation, Gauss-Newton / natural-gradient closed forms | fitting what we're *scored* on instead of an internal activation |
| B-TARGET | test-time training, on-policy reference construction | how to build a reference set that generalizes to the eval distribution |
| gating | H2O, SnapKV, PyramidKV, Scissorhands | what "which KV slot matters" means, and how the field decides it |

## Entries

> **SCOUT-AM (2026-07-28)** read `AM.pdf` = Zweiger, Fu, Guo, Kim, *Fast KV Compaction via Attention
> Matching*, arXiv **2602.16284v2**, ICML 2026, code `https://github.com/adamzweiger/compaction`.
> LIT-001…LIT-008 below all come from that one paper (§2, §3, App. A, App. C, App. F). They are split
> by **mechanism**, because each one maps to a different function in `cartridges/am/` and predicts a
> different signature.
>
> **One-line summary of the read:** the paper's object is `(C_k, β, C_v)` fitted *jointly* so a
> compacted block reproduces **both** the attention output **and the attention mass** of *the very
> block it was derived from*, over **10⁴–5·10⁴ reference queries per KV-head** drawn from prefilling
> the model **on that block's own text**. Our canonical config (`KEY_MODE=freeze`, `ENABLE_BETA` unset,
> `max_queries_per_head=64`) runs **none of `C_k` selection, none of β/mass-matching, and 1/250th of
> the reference queries** — i.e. it is the paper's *weakest ablation*, applied outside the paper's
> problem statement.

---
### LIT-001: Mass matching via per-key bias β (Eq. 2 + the mixture identity, Eq. 5) — *Fast KV Compaction via Attention Matching* (arXiv 2602.16284, §2, App. A.2)
- Status: candidate
- Board entry: **B-ROUTE** (primary), B-SOLVE (secondary)
- Mechanism (1 paragraph): AM matches **two** things, not one. Eq. (1) matches the *locally
  normalised* attention output `Attn(q;C_k,β,C_v) ≈ Attn(q;K,V)`; Eq. (2) matches the *unnormalised
  mass* `Mass(q;C_k,β) = Σ_j exp(q(C_k)_j^T/√d + β_j) ≈ Mass(q;K) = Σ_{k=1..T} exp(qK_k^T/√d)`. The
  reason both are needed is App. A.2's mixture identity: when the compacted block is concatenated with
  *any* later block (chat template, the eval question, decoded tokens),
  `Attn(q;[C;K_fix],[V;V_fix]) = w·Attn(q;C_k,β,C_v) + (1−w)·Attn(q;K_fix,V_fix)` with
  `w = Mass(q;C_k,β)/(Mass(q;C_k,β)+Mass(q;K_fix))`. **The mass *is* the routing weight of the whole
  compacted block against the future tokens.** The paper states flatly ("Why biases matter"): if `C_k`
  is a subset of `K` and β is absent then `Mass(q;C_k) ≤ Mass(q;K)` for every q, so the compacted
  block "systematically receives too little global weight". β is fitted by NNLS on
  `min_{w≥0} ‖Aw − m‖²` with `A_ij = exp(q_i(C_k)_j^T/√d)`, then `β = log w`; the paper's own
  leave-one-out ablation (Fig. 5, "no biases") shows removing β costs real quality.
- Maps to our code: `cartridges/am/key_select.py::refit_beta_nnls` (fit),
  `cartridges/am/finetune.py::apply_document_am_write_to_cache` L485 (`target_log_mass`) / L511-543
  (β write-back), `cartridges/am/teacher.py::compute_teacher_log_mass`,
  `cartridges/cache.py::get_cartridge_beta` (β is already plumbed into eval-time attention at
  `models/qwen/modeling_qwen3.py:195`). Existing flag `ENABLE_BETA` / `BETA_FIT_SCOPE`; the fix this
  entry motivates is a **working** fit, see LIT-002, proposed flag `AM_BETA_BOX=3.0`.
- **Prediction (about OUR signature):** the B-ROUTE diagnostic (attention mass on rewritten slots at
  eval time, MT queries) is currently set *entirely* by frozen Phase-1 keys and must be **strictly
  below** the mass the same queries put on those slots in the teacher `[cartridge‖doc]` (student has
  512 keys, teacher has 512+T_doc). Turning on a *finite, box-constrained* β and fitting it on the
  **selected** slots must raise that measured mass on the rewritten slots and lower the mass on the
  448 untouched QA slots. Two-way discriminator: if MT stays ≈2.54 **after** the rewritten-slot mass
  reaches teacher level, B-ROUTE is **refuted** and the cause is B-OBJ/B-CAP; if MT moves toward the
  1.87 bar, B-ROUTE is confirmed and β is the missing mechanism, not a knob. Also predicts a *cost* on
  QA: raising β on 32–64 slots demotes the other 448, so QA should rise from 2.18 toward/past 2.37.
- Cost: fully gradient-free. One extra NNLS per (layer, head) per document — paper measures β fitting
  at **2.2 s** for a 60k-token context on 64 heads (Table 3), i.e. negligible next to our 165–185 s
  `solve_s`. Extra memory `(2d+1)/2d`, i.e. ~0.2% of the cache.
- Why it might NOT transfer: in the paper the *whole* compacted block is one homogeneous object
  derived from one text, so raising its mass is unambiguously right. Here the 512 slots are a
  **mixture** of Phase-1 QA content and rewritten MT content; raising β on the MT slots is not
  "restoring lost mass", it is **re-allocating mass away from the retained QA content** — the two axes
  are directly traded, which the paper never has to face.
- Related: EXP-005 / EXP-005b / EXP-006 (all three failed *inside* this mechanism), B-SOLVE.

---
### LIT-002: Box-constrained NNLS with warm-started projected gradient (Algorithm 3 + "Stabilizing β") — *Fast KV Compaction via Attention Matching* (arXiv 2602.16284, App. C.2)
- Status: candidate
- Board entry: **B-SOLVE**
- Mechanism (1 paragraph): the paper's NNLS is 19 lines and every line is a safeguard.
  `B⁽⁰⁾ = argmin_B‖MB − y‖²` (unconstrained lstsq) → `B⁽⁰⁾ ← max(B⁽⁰⁾, ε)` → **`B⁽⁰⁾ ← min(B⁽⁰⁾, u)`**
  → if `iters = 0` **return** → estimate `L ≈ ‖M‖₂²` by power iteration, `η = 1/L` → `iters` projected
  gradient steps `B ← clip(B − ηMᵀ(MB − y), ε, u)`. The **bounds are the point**: for
  Highest-Attention keys the paper *replaces NNLS with a bounded least-squares* enforcing
  `e⁻³ ≤ w_j ≤ e³` (⇔ `β_j ∈ [−3, 3]`) and uses **`iters = 2`**; for OMP keys it uses **`iters = 0`**,
  caps `w_j ≤ e⁷`, and **prunes any selected key with `β < −7`, continuing selection until all
  retained keys satisfy `β ≥ −7`**. The paper's stated failure mode is exactly ours: "mass matching
  can assign extremely small weights to some selected keys (i.e. very negative β, or effectively
  β = −∞). Such keys may have little effect on the mass objective yet still be useful for reducing
  attention-output error; however, once β is very negative, the corresponding key **cannot contribute
  to the attention output, regardless of C_v**." The paper also computes β and C_v in FP32 (we do) and
  reports that ℓ2 regularisation on the β/C_v solves **degraded performance for every positive λ**.
- Maps to our code: `cartridges/am/key_select.py::nnls_projected_gradient` (L13-48) and
  `::refit_beta_nnls` (L165-224). Proposed opt-in flags `AM_BETA_BOX` (upper/lower box on `w`, default
  3.0 ⇒ `β ∈ [−3,3]`), `AM_NNLS_ITERS` (default 2), `AM_NNLS_DRIVER=gelsd`.
  **Concrete divergences, line by line:**
  `key_select.py:35` warm start is `torch.linalg.lstsq(Phi, target)` with the **default `gels`
  driver**, which on CUDA requires `Phi` to be full-rank and returns *undefined values (NaN/Inf)
  without raising* otherwise — and the `except RuntimeError` on `:36` therefore never fires. With
  `max_queries_per_head=64` and `top_t=64`, `Phi` is **64×64** and, after the row-shift on
  `key_select.py:216`, most of its entries underflow toward 0 ⇒ badly rank-deficient ⇒ this is the
  most likely birthplace of the EXP-006 "NaN inside the fit".
  `key_select.py:46` clamps **only from below** (`torch.clamp(w, min=1e-12)`) — **there is no upper
  bound anywhere in our code**, so nothing prevented the finite `max|β| ≈ 66–68` seen in EXP-005
  (`e⁶⁶` ≫ the paper's `e³` ceiling), and nothing removes a NaN (`torch.clamp` passes NaN through,
  which is precisely why EXP-006's magnitude clamp failed).
  `key_select.py:219` / `:97` / `:160` set `β = log(w.clamp(min=1e-12))` ⇒ a floor of
  **β ≈ −27.6**, i.e. exactly the "effectively β = −∞, key can no longer contribute regardless of C_v"
  state the paper writes a special rule to avoid (paper floor: −3, or prune at −7).
  `key_select.py:215` `residual_target = (target − fixed_mass).clamp_min(1e-12)`: when the 448
  non-selected slots already supply more mass than the teacher target, **every** row of the NNLS
  target becomes 1e-12, so `w → 0` and all selected slots get β ≈ −27.6 — a *silent* deletion of the
  rewritten slots, which then makes the value-solve design matrix `X = alpha[:, S]` (`value_solve.py:86`)
  numerically zero and produces the EXP-005b "NaNs in ridge lstsq solution" at `core.py:230`. The
  paper has no residual-target variant at all; it fits β over **all** t compacted keys against the
  full mass `m` (our `BETA_FIT_SCOPE=selected`, `finetune.py:71`, is our own invention).
  `key_select.py:96/145/159/172` use `n_iters = 200`; the paper uses **2** (highest-attention) or
  **0** (OMP).
- **Prediction (about OUR signature):** the doc-3 NaN is a *rank-deficiency* NaN, not an overflow NaN.
  Therefore (a) replacing the `gels` warm start with a rank-revealing driver (`gelsd`/`pinv`) or a
  ridge-stabilised normal-equation init, **plus** an upper box `w ≤ e³`, should let `ENABLE_BETA=1`
  run all documents to completion with `max|β| ≤ 3` and finite `am/mean_mse`; (b) purely clamping the
  *output* cannot help — already shown by EXP-006, which is consistent with this diagnosis and
  inconsistent with "β merely blew up"; (c) if instead the NaN is generated at
  `key_select.py:215`'s all-1e-12 residual, the fix is to fit β over **all** trainable slots
  (`BETA_FIT_SCOPE=all`) rather than the selected subset, and the tell will be that the pre-clamp
  `w` vector is ≈0 **for every** selected slot, not just a few.
- Cost: gradient-free, no extra solves; **cheaper** than today (200 → 2 PGD iterations per fit).
- Why it might NOT transfer: the paper's NNLS always sees a *tall* `Φ` (n ≈ 16k–50k rows ≫ t columns),
  where an unconstrained lstsq warm start is well posed. Our `Φ` is 64×64 or 64×128. The safeguards
  transfer; the **regime** does not (see LIT-003) — box constraints alone may just replace a NaN with a
  well-behaved but meaningless β fitted on 64 points.
- Related: EXP-005, EXP-005b, EXP-006, LIT-001, LIT-003.

---
### LIT-003: Reference-query construction and scale — repeat-prefill / context-prefill / self-study, ≤50k queries per KV-head (arXiv 2602.16284, §3.1, §4, App. C.4, App. D)
- Status: candidate
- Board entry: **B-TARGET** (primary), B-CAP / B-SOLVE (secondary)
- Mechanism (1 paragraph): `Q_ref ∈ R^{n×d}` is "a proxy for the queries the model is likely to
  produce when attending to the context", and every one of the paper's constructions extracts those
  queries **while the model is reading the very text being compacted**: *repeat-prefill* prefills
  "`{C}` Repeat the previous context. `{C}`" and keeps the queries emitted while reconstructing `C`;
  *context-prefill* prefills `C` alone; *self-study* samples LM(·|C, prompt) with 4 fixed prompts and
  records every query vector during prefill+decode. Scale is a first-class design parameter: **"we use
  at most 50,000 reference queries per KV-head for each context/chunk"**, reservoir-sampled, averaging
  ~16k/head on QuALITY (App. C.4); App. D shows even `ss-5k`/`ss-plus-repeat-5k` subsamples (5,000
  per head) preserve performance, and that *random Gaussian* queries "work, though [they] lag". The
  ordinary least-squares fit `C_v* = argmin‖X C_v − Y‖_F² = (XᵀX)⁻¹XᵀY` (Eqs. 3–4) is therefore always
  **massively overdetermined**: `X ∈ R^{n×t}` with `n ~ 10⁴` and `t ~ 10²`.
- Maps to our code: `cartridges/am/finetune.py:81` (`max_queries_per_head: int = 64`) and the
  subsample at `finetune.py:425-427`; query source is
  `cartridges/am/continual.py:172-182` → `finetune.py::_collect_reference_queries` (L291-334), which
  prefills the **MT synthesis conversations** against the **current cartridge** — the document text is
  *not* in context. Proposed opt-in flags `AM_MAX_QUERIES_PER_HEAD` (there is currently **no env knob
  at all**; `examples/qasper2/train/continual_am_sparse.py` never sets it, so every experiment in
  `results.csv` ran at exactly 64) and `AM_REF_QUERY_SOURCE ∈ {conversations, doc_prefill,
  repeat_prefill}`.
- **Prediction (about OUR signature):** this is the sharpest quantitative divergence found —
  **n = 64 vs the paper's 16,000–50,000, and `t = top_t = 64`**, so our per-head system `X` is
  **64×64, exactly determined**: the value solve *interpolates* its reference queries, `am/mean_mse`
  is small for trivial reasons, and the generalisation gap to held-out MT queries should be enormous.
  Concretely I predict: (i) reference-query recon MSE ≪ held-out-MT-query recon MSE, by orders of
  magnitude, at `top_t = 64`; (ii) **this explains the board's unexplained EXP-007 anomaly** — at
  `top_t = 128` the system becomes 64 rows × **128 unknowns**, i.e. *underdetermined*, and
  `core.py:198-204` silently switches to the `n < k` min-norm branch, so `top_t=128` fits 64 points
  with 128 free parameters and MT got *worse* (2.686) exactly as pure overfitting predicts, while
  `top_t = 32` (64×32, the only mildly overdetermined point) gave the best MT/QA pair; (iii) raising
  `max_queries_per_head` from 64 to ≥1024 with everything else canonical should *not* change
  `am/mean_mse` in the same direction as MT — MSE should rise while MT falls, which is the signature
  that separates B-TARGET from B-OBJ. (iv) Additionally, our queries are extracted from a model that
  **has never read the new document** (only the cartridge is in context), whereas the paper's queries
  always come from a model that has read the source text; predicts a further, smaller gap that a
  `doc_prefill` reference source would close.
- Cost: gradient-free. Query *collection* is already done at full scale (thousands per head) and
  discarded at `finetune.py:425`, so raising the cap costs **only** the solve: `X` grows from 64×t to
  n×t, and `XᵀX` is t×t regardless — the ridge/Cholesky path in `core.py:206-211` is O(n t²), so 16×
  more queries ≈ 16× the matmul, on a 165 s `solve_s` budget. Cheap.
- Why it might NOT transfer: the paper's queries are cheap because there is one context and one
  compaction; we do **one write per document** over many documents, so per-document query banks
  multiply. Also, more reference queries make the fit *harder*, not better, when only 64 of 512 slots
  are free to move — overdetermining a system whose support is too small may simply convert
  interpolation into a large, honest residual without moving CE.
- Related: B-CAP (EXP-007 top32/64/128), B-SOLVE.

---
### LIT-004: `C_k` is always a subset of the block being compacted — AM has **no** procedure for writing into foreign, frozen keys (arXiv 2602.16284, §3.3, §6 "Future work")
- Status: candidate
- Board entry: **B-ROUTE**
- Mechanism (1 paragraph): "We do not have a closed-form solution for constructing `C_k` in general.
  However we found it empirically effective (and efficient) to restrict `C_k` to be **a subset of the
  original keys**, i.e. `C_k = K_{S,:}` for some index set `S ⊂ {1,…,T}`." Both selection rules
  (highest-attention-RMS, OMP) pick `S` **from the same `K` whose behaviour is being reproduced**, and
  the pipeline order is fixed: choose `C_k` → fit β → fit `C_v`. Consequently the values being solved
  for sit on keys that already route correctly for this content — which is why the Fig. 5 ablation
  "no learned values" (keep the *original* `V_S` for the selected keys) still "yields a reasonable
  approximation". The paper's Discussion names our regime as an **open problem**, twice: "a promising
  direction is to move away from subset selection for `C_k` (e.g. directly optimizing compact keys)"
  and "architectures or training procedures that … **explicitly operate over a fixed set of keys and
  values**". App. F.3 (online compaction) is the closest the paper comes to a continual setting, and
  it **re-compacts the entire cache including previously compacted portions** every time, explicitly
  noting it does *not* freeze previously compacted portions. Chunked compaction (§3.5, App. C.3)
  likewise *concatenates* per-chunk compacted segments; it never overwrites chunk *i*'s slots with
  chunk *j*'s content.
- Maps to our code: `cartridges/am/finetune.py:493-509` (`key_mode != "freeze"` → 
  `key_select.py::rewrite_keys_on_support` over `candidate_keys = cat([K_cartridge[S], K_doc])`) —
  this is already the paper-faithful path and is **untested**; the default
  `key_mode="freeze"` (`finetune.py:69`) has no counterpart anywhere in the paper. Existing flag
  `KEY_MODE ∈ {highest_attention, omp}`. Note `finetune.py:256-261`: `_should_fit_beta` returns
  `key_mode != "freeze"`, so `KEY_MODE=freeze` **also silently disables β** — the canonical config runs
  AM with *neither* of its two `C_k`/β components.
- **Prediction (about OUR signature):** the B-ROUTE write-ceiling oracle should come out **capped**
  under `KEY_MODE=freeze`. Specifically: with frozen keys the eval-time MT attention mass on the 32–64
  rewritten slots is fixed by Phase-1 and cannot exceed what the *pre-write* cartridge already gave
  them; the solve's only recourse is to inflate `‖V_S‖`, which is already visible in our logs
  (`value_max_abs 816` in EXP-001, `486` in EXP-003 — an order of magnitude above normal KV scale) and
  is the classic signature of "route is too narrow, amplitude is compensating". I predict the
  teacher-KV write-ceiling oracle leaves MT ≈2.4–2.55 (i.e. ≲0.15 movement) under `KEY_MODE=freeze`,
  and that `KEY_MODE=highest_attention`/`omp` is the *only* configuration in our repo that can break
  that ceiling — at a QA cost, because the rewritten keys stop matching the QA content they were fit
  to. If `KEY_MODE=highest_attention` moves MT below ~2.3 while QA degrades, B-ROUTE is confirmed and
  the mission becomes a key-allocation problem.
- Cost: gradient-free. `highest_attention` is the paper's cheapest selector (**3 s** for a 60k context
  on 64 heads, Table 3) — negligible. OMP is 565 s in the paper's own profile (see LIT-005).
- Why it might NOT transfer: the paper never *re-selects* keys for a cache that must keep serving old
  content. Every key we take for MT is a key removed from QA; the paper's `S` has no incumbent. Our
  512 slots are also **not** original model keys (they are Phase-1 self-distilled parameters), so
  "subset of the original keys" has no literal meaning here — the candidate pool at
  `finetune.py:496-499` mixes learned cartridge keys with genuine post-RoPE doc keys, two different
  distributions.
- Related: B-ROUTE oracle, LIT-001, LIT-005.

---
### LIT-005: OMP key selection with periodic NNLS refit (Algorithms 1 & 2) (arXiv 2602.16284, §3.3, App. C.1)
- Status: candidate
- Board entry: **B-ROUTE / B-CAP**
- Mechanism (1 paragraph): build the mass-feature matrix `Φ_ij = exp(q_i K_jᵀ/√d)` and target
  `m_i = Σ_j Φ_ij`; greedily grow `S` by `j* = argmax_{j∉S} (rᵀ Φ_{:,j})`, refit `w = argmin_{w≥0}
  ‖Φ_{:,S} w − m‖²` by NNLS, update `r = m − Φ_{:,S} w`; return `S` and `β = log w`; then fit `C_v` by
  ordinary least squares. `C_k` and β are thus fit **jointly** (selection is driven by the mass
  residual), which is the paper's best method (AM-OMP). Alg. 2 makes it affordable: select **k = 4**
  keys per greedy iteration and refit NNLS only every **τ = 2** iterations, worth a 4–8× speedup "with
  little degradation"; NNLS runs with `iters = 0` for OMP.
- Maps to our code: `cartridges/am/key_select.py::select_keys_omp` (L102-162), reached via
  `KEY_MODE=omp`. **Divergences:** L145/L159 refit NNLS at **every one of the t greedy steps** with
  `n_iters=200` PGD (paper: k=4, τ=2, `iters=0`) — that is `t × 200` inner iterations per head per
  document instead of `t/8` outer refits with a single lstsq each; L142/L147 use
  `(Φᵀr).abs()` where Alg. 1 line 5 uses the **signed** `rᵀΦ_{:,j}` (abs admits columns that can only
  help with a negative weight, which NNLS then zeroes); there is **no `β < −7` pruning loop and no
  `w ≤ e⁷` cap** (App. C.2 "Stabilizing β"); and `rewrite_keys_on_support` (L227-284) **discards the
  β returned by the selector** (`_, sel_idx` at L259/L270) and refits it separately at
  `finetune.py:514`, breaking the joint `(C_k, β)` fit that makes OMP the paper's best variant.
- **Prediction (about OUR signature):** if `KEY_MODE=omp` is run as-is, `solve_s` explodes from
  ~170 s to **hours** (36 layers × 8 KV-heads × 64 greedy steps × a 200-iteration PGD each, per
  document) — I predict the run does not finish in a comparable wall-clock budget, which would make it
  a NORTH_STAR cost failure even if MT improved. With the paper's k=4/τ=2/`iters=0` settings the same
  selection costs ~1/100th as much. Quality-wise, OMP selects for **mass residual**, so its selected
  slots are precisely those that can carry mass for the new document — if B-ROUTE is the cause, OMP
  should move MT further than `highest_attention` and further than any value-only change; if MT is
  still ≈2.5 with OMP-selected keys *and* a working β, B-ROUTE is refuted outright.
- Cost: gradient-free but the dominant term — paper measures OMP at **565 s** vs highest-attention at
  **3 s** for one 60k context (Table 3), and we would pay it **once per document**.
- Why it might NOT transfer: the paper's OMP chooses among `T ≈ 10⁴` genuine context keys; ours
  chooses among `t + T_doc` candidates of which the first `t` are learned cartridge keys. Also the
  mass target `m` here is the teacher `[cartridge‖doc]` mass, which includes the *old* content's mass
  — greedy mass-residual selection may simply re-select cartridge keys and change nothing.
- Related: LIT-004, LIT-002.

---
### LIT-006: On-policy, layer-sequential reference queries (arXiv 2602.16284, §3.1 "On-policy queries", App. C.4)
- Status: candidate *(low expected value — the paper itself ranks it least important)*
- Board entry: **B-TARGET**
- Mechanism (1 paragraph): compacting layer `ℓ` changes the residual stream that layers `>ℓ` see, so
  queries extracted from the *unmodified* model are off-distribution for later layers. The paper
  compacts layers **sequentially** and, for each layer `ℓ`, re-extracts `Q_ref^ℓ` by running the model
  with layers `< ℓ` already compacted, "then optimize compaction at layer `ℓ` using these on-policy
  queries. This yields slight but consistent improvements." Fig. 5's leave-one-out puts "No On-Policy"
  among the two least damaging ablations.
- Maps to our code: `cartridges/am/continual.py:172-182` collects `query_acc` **once**, before any
  layer is written, and `finetune.py::apply_document_am_write_to_cache` L359-579 then writes layers
  0…n−1 in one pass using those stale queries. Proposed opt-in flag `AM_ONPOLICY_LAYERS=1` (re-run the
  prefill after every `k` layers).
- **Prediction (about OUR signature):** the per-layer solve MSE reported in `am_stats.mse_per_layer`
  should be *monotonically worse in depth* relative to a fresh on-policy re-extraction, and the gap
  should widen with layer index; fixing it should move MT by **< 0.1** (inside our noise floor), i.e.
  I predict this is **not** the acquisition bottleneck and should not be built before LIT-001/003.
- Cost: gradient-free, but multiplies reference prefill passes by the number of re-extraction points
  (n_layers in the limit) — the single most expensive stage in the paper's own profile is query
  generation (Table 3: 139 s for self-study vs 1.8 s for the value fit).
- Why it might NOT transfer: our per-layer edit touches only 64/512 slots, so the induced residual
  drift is far smaller than the paper's 50× compaction of every layer.
- Related: LIT-003.

---
### LIT-007: Repeated / online compaction re-compacts the **whole** cache, never a frozen sub-block (arXiv 2602.16284, §5, App. F.3)
- Status: candidate
- Board entry: **B-ROUTE**
- Mechanism (1 paragraph): the paper's only multi-shot setting. Mid-trajectory, whenever the physical
  KV budget is hit, it "compact[s] the entire context except for the most recent 20 tokens" —
  **including previously compacted portions** — halving the whole cache each time; up to 6 consecutive
  compactions preserve AIME accuracy at 4× the physical budget. The paper explicitly flags the
  alternative we are running as *not what it did*: "Our study uses a simple uniform compaction scheme
  and always compacts the entire context, rather than, for example, **freezing previously compacted
  portions** or selectively compacting recent tokens." §5 frames this as the paper's answer to
  fixed-size state: "one-shot compaction can be applied repeatedly to maintain a fixed maximum state
  size."
- Maps to our code: a new execution mode alongside
  `cartridges/am/finetune.py:64` `execution_mode`, proposed `AM_EXECUTION_MODE=recompact_all`:
  per document, form the teacher `[current 512-slot cartridge ‖ doc KV]`, then run **full** AM
  (`C_k` re-selected over all 512 + T_doc candidates, β and `C_v` refit over **all 512 slots**, not a
  64-slot subset) back down to 512. Touches `cartridges/am/continual.py::run_per_document_am_phase2`
  and `finetune.py::apply_document_am_write_to_cache`.
- **Prediction (about OUR signature):** this is the paper-faithful instantiation of our task, and it
  predicts the *opposite* trade to everything in `results.csv`: MT should fall substantially (the
  write is no longer routed through 64 incumbent keys) while QA should degrade toward or past the
  dense bar (old content is re-compacted, not preserved). Quantitatively I predict MT ≤ 2.2 and QA
  ≥ 2.4 — i.e. it should *move both axes*, whereas every `KEY_MODE=freeze` run so far has MT pinned in
  [2.54, 2.69] regardless of gating, support, target or ridge. A run that moves neither axis would be
  strong evidence that the cause is B-OBJ (attention-space fit ↛ CE), not routing.
- Cost: gradient-free, but the solve is now `t = 512` wide and `T = 512 + T_doc` deep per head instead
  of `t = 64`, per document — expect `solve_s` up by ~(512/64)² ≈ 64× in the `XᵀX` term unless the
  query count is raised in step (LIT-003); this is the entry most at risk on the NORTH_STAR cost axis.
- Why it might NOT transfer: the paper's repeated compaction operates on a *homogeneous, ordered*
  trajectory where the old content is genuinely stale; ours must **retain** the old content at a fixed
  quality bar. Re-compacting the QA cartridge every document also compounds compaction error across
  documents, which the paper only tested to 6 rounds on a single reasoning trace.
- Related: LIT-004, B-ROUTE oracle.

---
### LIT-008: Reconstruction loss as a proxy for downstream quality — and where it breaks (arXiv 2602.16284, App. F.2, Fig. 5, Fig. 11)
- Status: candidate
- Board entry: **B-OBJ**
- Mechanism (1 paragraph): the paper measures "loss (log-perplexity) on the original-cache
  generations": sample responses from the *full* cache, then score their NLL under the *compacted*
  cache — a forward-KL-to-the-teacher objective up to a constant. Its conclusions are stated as two
  bullets: "(i) reconstruction loss is a useful proxy for comparing compaction methods **within
  families that aim to preserve attention behavior**, and (ii) it should **not** be treated as a
  universal predictor of downstream performance when methods fundamentally change the objective."
  Their evidence: Cartridges (end-to-end token-level distillation) achieves *lower* reconstruction
  loss than attention-matching methods at equal downstream accuracy, while summarization achieves
  *higher* reconstruction loss at equal accuracy. In other words, AM's own authors report that the
  attention-space objective and the token-CE objective come apart precisely when comparing AM against
  a gradient cartridge — which is exactly our comparison.
- Maps to our code: the target itself, `cartridges/am/teacher.py::compute_teacher_targets` (attention
  output per KV-head, pre-`o_proj`) consumed at `finetune.py:477`. Proposed opt-in flag
  `AM_TARGET_SPACE ∈ {attn_out, attn_out_owhitened}` — e.g. weight the per-head least-squares by the
  head's `o_proj` block (`min ‖(X C_v − Y) W_oᵀ‖²`, still a closed form: right-multiply `Y` and solve
  in the whitened metric), so the ℓ2 we minimise is the ℓ2 the residual stream actually sees.
- **Prediction (about OUR signature):** the paper predicts a *bounded* payoff for any purely
  attention-space fix. If B-OBJ's oracle (drive `am/mean_mse` → ~0, unbounded support, λ=0) leaves MT
  flat at ≈2.5, that is fully consistent with F.2 and would mean the remaining 0.67 to the dense bar
  is not reachable by improving an attention-space MSE at all. Conversely, F.2's "within families the
  proxy works" says that among *our* AM variants, `am/mean_mse` measured **on held-out MT queries**
  (not on the 64 reference queries — LIT-003) should rank-order MT loss; if it does not, our MSE
  metric is measuring interpolation, not reconstruction. Concrete falsifiable claim: held-out-query
  recon MSE and MT CE should be **positively rank-correlated across EXP-001/003/007-top32/top128**,
  while reference-query MSE should be **uncorrelated or inverted**.
- Cost: gradient-free; `o_proj`-whitening adds one `d×d` (per head) matmul to `Y` and one Cholesky in
  the whitened metric — negligible against `solve_s`.
- Why it might NOT transfer: the paper's proxy claim is made at 50–100× compaction of a *single*
  homogeneous context with 10⁴ reference queries; our regime (64 queries, mixed old/new content,
  64/512 slots free) sits outside the family the correlation was measured in. Whitening by `W_o`
  ignores the MLP and every downstream layer, so it is a first-order correction at best.
- Related: B-OBJ oracle, LIT-003.
