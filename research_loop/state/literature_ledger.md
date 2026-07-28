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

---

> **SCOUT-EDIT (2026-07-28) — external import block, LIT-009…LIT-018.** Dispatched after SCOUT-AM
> established (LIT-004) that `AM.pdf` contains **no** procedure for writing new content into keys
> fitted to *different* content, and that §6 names our regime as future work. Question answered here:
> **how does the literature perform a closed-form, gradient-free write of NEW content into a FIXED,
> ALREADY-OCCUPIED associative store without destroying what is already there?** Families covered:
> closed-form model editing (ROME / MEMIT / EMMET), null-space & orthogonal-projection continual
> learning (AlphaEdit, Adam-NSCL, OWM, GPM), delta-rule / fast-weight associative writes (Schlag et
> al., DeltaNet, modern Hopfield, test-time regression), recursive least squares / Sherman-Morrison /
> Kalman sequential updates (Larimar, RLS layers), plus one synthesis (LIT-017) and one falsifier
> (LIT-018).
>
> **NOTATION — read once; every entry below uses it.** Per (layer, KV-head) of Qwen3-4B-Instruct-2507
> (36 layers × 8 KV-heads, d = 128): the cartridge is `K ∈ R^{512×d}` (frozen under `KEY_MODE=freeze`)
> and `V ∈ R^{512×d}` (`cache.trainable_values`). A head's output is `o(q) = a(q)ᵀV` with
> `a(q) = softmax(qKᵀ/√d + β) ∈ Δ^{511}`. Only rows `S` (|S| = `top_t`) move, so
> `Δo(q) = a_S(q)ᵀ ΔV_S`. **Therefore, in every editing paper's notation, our "weight matrix" is
> `W = V_Sᵀ ∈ R^{d×t}` and our "key" is the *routing vector* `a_S(q) ∈ R^t_{≥0}` — NOT the model's key
> vector.** Write `A_new = [a_S(q)]_{q ∈ MT-ref} ∈ R^{n×t}` (this is literally `X` at
> `value_solve.py:86`), `A_old = [a_S(q)]_{q ∈ QA}`, and the old-key second moment
> `C₀ = A_oldᵀA_old ∈ R^{t×t}`. Two structural consequences used repeatedly: (i) our keys are
> **non-negative and sum-to-one**, so `⟨a_old, a_new⟩ ≥ 0` always and two routing vectors can only be
> orthogonal by having **disjoint support**; (ii) `a_S(q)` is fully determined by `K` and `q` — with
> frozen keys **we do not get to choose the key we write on**, which is the one freedom every
> mechanism below assumes it has.
>
> **CODE FINDING THAT FRAMES THIS ENTIRE BLOCK (verified by reading, no GPU).** Our canonical write is
> *already* a MEMIT-shaped update — with the **wrong metric**. `continual_am_sparse.py:96` defaults
> `DELTA_WEIGHT=1e-2` (the dataclass default at `finetune.py:87` is `0.0`, so the launcher overrides
> it), and `finetune.py:580` branches on `use_old_guard or config.delta_weight > 0`. **Therefore every
> AM row in `results.csv` — EXP-001/003/004/005/005b/007/008, all with `delta_w=1e-2` — ran
> `guarded_sparse_am_value_update`, never the plain `sparse_am_value_update`.** That function appends
> `√w·I` rows with target `√w·V_S^old` (`value_solve.py:365-369`), i.e. it minimises
> `‖A_new V_S − R_new‖² + w‖V_S − V_S^old‖²` — which is exactly MEMIT's
> `Δ = R K₁ᵀ(C₀ + K₁K₁ᵀ)⁻¹` **with `C₀ = w·I`, an isotropic metric.** Every entry below is a
> statement about what `C₀` should be instead of `1e-2·I`. Note also that `DELTA_WEIGHT` itself has
> **never been swept** (every AM row is 1e-2); HYP-R0 refuted `RIDGE_LAMBDA`, a different knob.

---
### LIT-009: Covariance-preconditioned closed-form multi-edit — *Mass-Editing Memory in a Transformer* (MEMIT, arXiv 2210.07229, ICLR 2023)
- Status: candidate
- Board entry: **B-CAP** (primary), forgetting (secondary)
- Mechanism (1 paragraph): treat a weight matrix as a linear associative memory whose pre-existing
  content satisfies the normal equation `W₀K₀K₀ᵀ = M₀K₀ᵀ`. To add new pairs `(K₁, M₁)` while keeping
  the old ones, expand to `W₁[K₀|K₁][K₀|K₁]ᵀ = [M₀|M₁][K₀|K₁]ᵀ` and subtract, giving the **entire
  method in one line**: `Δ = R K₁ᵀ(C₀ + K₁K₁ᵀ)⁻¹` with residual `R ≜ M₁ − W₀K₁` and
  **`C₀ = λ·E_k[kkᵀ]`**, the *uncentered second moment of the pre-existing keys*, estimated once from
  100,000 Wikitext samples (GPT-J; 50,000 for GPT-NeoX) with `λ ≈ 1.5·10⁴–2·10⁴` "balancing the
  weighting of new vs. old associations". The edit key `kᵢ` is the MLP input activation averaged over
  `P = 10` random prefixes (`kᵢ = (1/P)Σⱼ k(xⱼ + sᵢ)`) so the key generalises across contexts; the
  target `zᵢ` is found by 20–25 optimisation steps and then **spread over a range of layers**,
  `rᵢˡ = (zᵢ − hᵢᴸ)/(L − l + 1)`. Scales to 10,000 simultaneous edits. **Note the split: the only
  gradient in MEMIT is in computing the target `z`; the weight update itself is closed-form.** We are
  strictly more gradient-free than MEMIT — our target is `compute_teacher_targets` (`teacher.py`), a
  closed-form teacher attention output, so `gradient_steps` stays 0.
- Maps to our code: `cartridges/am/core.py::_ridge_lstsq` (add an optional `metric: Tensor` argument
  and replace `XtX.diagonal().add_(lam)` at `:208` with `XtX += C0_SS`), and
  `cartridges/am/value_solve.py::guarded_sparse_am_value_update` (`:365-369`: replace the
  `√w·I` block with a Cholesky factor of `C₀_SS + λI`, keeping today's behaviour when `C₀ = wI`).
  A new collector — `G = Σ_{q∈QA} a(q)a(q)ᵀ ∈ R^{512×512}` per (layer, head) — reuses the machinery
  already present: `install_query_capture_hooks` (`sparse_cache_finetuning`), `AMQueryAccumulator`,
  and the `OLD_REF_DATA_PATH` QA-parquet loader at `continual_am_sparse.py:251`. Proposed opt-in flags
  `AM_PRECOND ∈ {none, memit_c0}` (default `none` ⇒ bit-identical), `AM_OLD_GRAM_PATH`,
  `AM_C0_LAMBDA` (MEMIT's λ, the new-vs-old dial).
- **Prediction (about OUR signature):** the preconditioner acts on **forgetting**, not acquisition, so
  at fixed `top_t = 64` I predict **MT essentially unchanged (Δ ≤ 0.1, inside noise)** and QA improved
  from 2.252 toward the Phase-1 floor 2.239. The *discriminating* prediction is about the `top_t`
  curve: EXP-007's monotone QA degradation (2.1766 → 2.2521 → 2.4837 at t = 32/64/128) is the
  signature of an isotropic penalty that is blind to *which slot combinations QA reads*. Under `C₀`
  that curve should **flatten** — QA at `top_t=128` should come back under ~2.30. If QA still
  degrades monotonically with the true `C₀` in place, the forgetting is not entering through the value
  write at all and this whole family is refuted for us.
- Cost: **gradient-free, `gradient_steps = 0`.** One extra forward pass over the QA reference set with
  query capture (the same pass `ENABLE_OLD_REFERENCE_GUARD=1` already does), done **once** after
  Phase 1 and cached to disk. Memory: `36 × 8 × 512² × 4 B ≈ 302 MB` fp32 for the full Gram (slice
  `G[S,S]` per document); or store the raw QA queries and recompute `a` per document (less memory,
  more flops). Solve cost unchanged — still one `t×t` Cholesky. No extra solves.
- Why it might NOT transfer: MEMIT's `C₀` is a covariance of *model activations* in a `d₀`-dim space
  estimated from 10⁵ samples; ours is a covariance of **simplex-valued routing vectors** in a
  `t ≤ 512`-dim space. Non-negative keys give `C₀` a dominant near-rank-one component (the mean
  routing vector), so the preconditioner may act as global shrinkage rather than directional
  whitening — i.e. degenerate to a `DELTA_WEIGHT` rescale (see LIT-018). Also MEMIT preserves
  *pretraining* knowledge under a frozen `W₀`; our `V` has itself been *fitted* to QA in Phase 1, so
  the thing being preserved is a fit, not a prior, and `C₀` built from a QA *reference* set is only as
  representative as `OLD_REF_DATA_PATH`.
- Related: LIT-010, LIT-011, LIT-018, EXP-007, B-CAP.

---
### LIT-010: Preservation-memorization with an **equality** constraint — ROME + EMMET, *A Unified Framework for Model Editing* (arXiv 2403.14236, EMNLP Findings 2024)
- Status: candidate
- Board entry: **B-CAP** (primary), **B-SOLVE** (secondary)
- Mechanism (1 paragraph): ROME and MEMIT optimise the *same* objective —
  `argmin_Ŵ ‖ŴK₀ − W₀K₀‖²_F  s.t.  Ŵk_e = v_e` — differing only in whether the new content enters as
  an **equality constraint** (ROME, one edit) or a **least-squares term** (MEMIT, batched). ROME's
  Lagrangian gives a rank-one update
  `Δ = (v_e − W₀k_e)(k_eᵀC₀⁻¹)/(k_eᵀC₀⁻¹k_e)` (a Sherman-Morrison form); EMMET generalises the
  equality constraint to a batch:
  **`Δ = (V_E − W₀K_E)(K_EᵀC₀⁻¹K_E)⁻¹K_EᵀC₀⁻¹`**, with `C₀ = K₀K₀ᵀ = Σᵢ k⁰ᵢ(k⁰ᵢ)ᵀ` requiring ≥ ~4d
  independent vectors to be invertible, and `D = K_EᵀC₀⁻¹K_E` "often ill-conditioned" hence stabilised
  as `D ← D + αI, α = 0.1`. Batches to 10,000 edits with MEMIT-level quality. **The transferable
  point for us is the tradeoff direction:** the equality constraint makes *acquisition* a hard
  guarantee and puts all the slack into preservation — the opposite of MEMIT's least-squares
  compromise, and the right way round given that our entire deficit is acquisition.
- Maps to our code: in our notation EMMET reads
  `ΔV_S = C₀⁻¹A_newᵀ(A_newC₀⁻¹A_newᵀ + αI)⁻¹R_new` — an exact fit of the `n` reference queries
  whenever `t ≥ n`, selecting the **minimum-`C₀`-norm** solution among all exact fits. **Our code
  already takes exactly this branch, with the identity metric:** `core.py:198-204` computes
  `W = Xᵀ(XXᵀ + λI)⁻¹Y` whenever `n < k`, i.e. whenever `top_t > max_queries_per_head = 64`. So the
  change is one branch: `core.py::_ridge_lstsq`, `n < k` path — replace `XXᵀ` with `A C₀⁻¹ Aᵀ` and
  post-multiply by `C₀⁻¹`. Proposed opt-in flag `AM_PRECOND=emmet_c0` (shares `AM_OLD_GRAM_PATH` with
  LIT-009); `α` maps onto the existing `RIDGE_LAMBDA_MIN`.
- **Prediction (about OUR signature):** the board's unexplained EXP-007 `top_t=128` anomaly
  (QA 2.4837 / MT 2.6860, *both worse*) is the **identity-metric min-norm interpolant**: at n=64 < t=128
  the solve interpolates its 64 reference queries and spends its remaining 64 degrees of freedom
  isotropically — including on slot combinations QA reads. Under the `C₀⁻¹` metric with everything
  else identical I predict a **sign flip on the QA axis**: QA at `top_t=128` back to ≤ 2.30 (the free
  directions are now spent where QA is blind) with MT no worse than the t=64 value (~2.54), because
  both metrics fit the reference queries exactly. **Falsifier:** if `top_t=128` under `C₀` still gives
  QA > 2.40, the min-norm metric is not the mechanism, and B-CAP's re-opening (SCOUT-AM) should be
  resolved by query count alone (LIT-003) rather than by the metric.
- Cost: **gradient-free, `gradient_steps = 0`**, identical flop count (an `n×n` Cholesky either way),
  no extra solves. Needs the same `G` statistic as LIT-009 — collect once, use for both.
- Why it might NOT transfer: EMMET's equality constraint is meaningful because its `k_e` are a handful
  of well-separated activation vectors; our `A_new` rows are up to 64 **near-parallel simplex
  vectors**, so `A_newC₀⁻¹A_newᵀ` will be far worse conditioned than EMMET's `D` and may need an `α`
  large enough to erase the distinction from ridge. And "exact fit of the reference queries" is an
  interpolation guarantee about our *reference* set — with `max_queries_per_head = 64` (LIT-003) that
  is precisely the quantity we have most reason to distrust.
- Related: LIT-009, LIT-003, EXP-007, B-CAP, B-SOLVE.

---
### LIT-011: Null-space-projected editing — *AlphaEdit: Null-Space Constrained Knowledge Editing* (arXiv 2410.02355, ICLR 2025) + its reproducibility study (arXiv 2606.26783)
- Status: candidate
- Board entry: **forgetting** (primary), **B-CAP** (secondary)
- Mechanism (1 paragraph): instead of *penalising* damage to preserved knowledge, **forbid** it. Take
  `{U, Λ, Uᵀ} = SVD(K₀K₀ᵀ)` over the preserved-knowledge keys `K₀` (in practice ~100,000 Wikipedia
  triplets), delete the eigenvectors whose eigenvalue exceeds a threshold (**10⁻²**), keep the rest as
  `Û`, and form the projector **`P = ÛÛᵀ`**. The edit is then computed *inside* that subspace:
  **`Δ = R K₁ᵀP(K_pK_pᵀP + K₁K₁ᵀP + I)⁻¹`**, where `K_p` are the keys of *previous* edits in a
  sequential chain (the term `‖Δ̃PK_p‖²` protects them, using the fact that `W K_p = V_p` already
  holds after those edits). Because `ΔP K₀ ≈ 0`, "the output of post-edited LLMs remains unchanged
  when queried about the preserved knowledge" **by construction, with no preserved targets required**.
  `P` is independent of the edit and is computed **once**; the paper advertises the whole thing as
  "a single line of additional code" giving +36.7% average over locate-then-edit baselines.
- Maps to our code: in our notation `ΔV_S = P·Z`, `Z = (P A_newᵀA_new P + I)⁻¹ P A_newᵀ R_new`, with
  `P` the projector onto the approximate null space of `C₀ = A_oldᵀA_old`. Same insertion point as
  LIT-009: `cartridges/am/core.py::_ridge_lstsq` (accept a projector and solve in the projected basis)
  and `value_solve.py::guarded_sparse_am_value_update`. `P` is a per-(layer, head) tensor built once
  after Phase 1 from the same `G` Gram, cached next to the cartridge (mirror the `trainable_beta`
  persistence pattern in `cartridges/cache.py::save`/`from_pretrained`, `:243`/`:308`). Proposed
  opt-in flag `AM_PRECOND=alphaedit`, `AM_NULLSPACE_TAU` (AlphaEdit's 1e-2).
- **Structural caveat, stated before the prediction:** with `top_t = 64` and `n_old ≫ 64` QA reference
  queries, `C₀ ∈ R^{64×64}` is generically **full rank ⇒ P = 0 ⇒ the update vanishes entirely.**
  Null-space editing has room only if (a) the QA routing vectors are effectively low-rank, or (b) `t`
  is much larger than that effective rank. **This makes null-space editing a reason to run
  `top_t = 512` — the one support setting HYP-S1 never tested — rather than a drop-in at t=64.**
- **Prediction (about OUR signature):** three, in increasing order of cost. (i) The eigenspectrum of
  `C₀` at `top_t = 512` decides the entire family: with `r_eff(τ) = #{λᵢ > τ·λ_max}`, the writable
  dimension is `512 − r_eff`. I predict `r_eff` is **small per head** (attention from one task's
  queries onto a 512-slot cartridge is spiky and query-clustered), so a 512-slot null-space write has
  real room while a 64-slot one has none. (ii) The decisive scalar is
  `ρ = E_{q∈MT}‖P a(q)‖²/‖a(q)‖²`: `ρ → 0` means MT reads exactly where QA reads, so **no value-space
  operator whatsoever can separate the two axes** — that would confirm B-ROUTE as *capped*
  independently of the write-ceiling oracle; `ρ ∈ [0.3, 0.8]` means the axes are separable in value
  space and this is the operator that separates them. (iii) If run at `top_t=512` with
  `AM_PRECOND=alphaedit`, QA should be **pinned within noise of the Phase-1 floor 2.2388** — better
  than any AM point on the board — with MT improving only in proportion to `ρ`.
- Cost: **gradient-free, `gradient_steps = 0`.** One symmetric 512×512 eigendecomposition per
  (layer, KV-head) = 288 tiny EVDs, milliseconds each, computed **once** after Phase 1 (P does not
  depend on the document — the paper's own efficiency argument). Same 302 MB `G` as LIT-009, or store
  `P` directly. No extra solves per document.
- Why it might NOT transfer (with *measured* failure modes from the reproducibility study, arXiv
  2606.26783): that study reproduces efficacy/generalisation/specificity but **not** fluency and
  consistency (8–15% and 3–10× worse), finds degradation past ~5,000 sequential edits, and — the part
  that matters most to us — reports that AlphaEdit **fails entirely on Gemma2** because its
  post-feedforward layer normalisation "violates the linear contribution assumption, making the
  null-space projection's protection against catastrophic forgetting **bounded rather than
  unconditional**", and on Phi3 because fused projections break the localisation heuristic. **Our
  setting violates that linear assumption more severely than Gemma2's:** the path from `V` to CE runs
  through a softmax we may also be perturbing (if β or keys move, `a_old` itself changes and `P` goes
  stale), then `o_proj`, RMSNorm and 30+ downstream layers. Exact preservation of a *head output* is
  not exact preservation of token CE. Chain length is not our risk (we do ~dozens of writes, not
  5,000), but the `C₀` sample is: it comes from `OLD_REF_DATA_PATH`, not the QA eval split.
- Related: LIT-009, LIT-012, LIT-017, LIT-018, B-CAP, B-ROUTE.

---
### LIT-012: How the field actually *builds* the approximate null space — Adam-NSCL (CVPR 2021), OWM (Nature MI 2019), GPM (ICLR 2021)
- Status: candidate
- Board entry: **forgetting** (primary), B-CAP (secondary)
- Mechanism (1 paragraph): three constructions of the same projector, differing in the statistic and,
  crucially, in the **threshold rule** — which is what AlphaEdit leaves as a bare `10⁻²`.
  **Adam-NSCL** (*Training Networks in Null Space of Feature Covariance for Continual Learning*, Wang,
  Li, Sun, Xu, CVPR 2021 oral) states the stability condition as `X̄ˡ_{t−1} Δwˡ_t = 0`, where `X̄` is
  the **uncentered feature covariance of all previous tasks' layer inputs**, accumulated across tasks;
  since the exact null space is empty in practice, it takes the SVD of `X̄` and keeps the subspace
  spanned by the singular vectors of the **smallest** singular values, then projects the candidate
  (Adam) update into it. **OWM** (Zeng et al., *Continual learning of context-dependent processing*,
  Nature Machine Intelligence 2019) writes the projector in closed form and maintains it
  **recursively, RLS-style**: `P_n = I − A_n(A_nᵀA_n + αI)⁻¹A_nᵀ` with `A_n` holding all previously
  seen inputs as columns and `α` a noise floor; the RLS recursion means old inputs are **never
  stored**. **GPM** (Saha et al., ICLR 2021) makes the threshold explicit and *energy-based*: build
  `Rˡ = [xˡ₁ … xˡ_{n_s}]` from `n_s` random samples, SVD it, and keep the smallest `k` satisfying
  **`‖(Rˡ)_k‖²_F ≥ ε_th‖Rˡ‖²_F`**; project `∇W ← ∇W − (∇W)Mˡ(Mˡ)ᵀ`; append each task's new directions
  to the basis. GPM also names the failure mode that matters to us: "the size of GPM is determined by
  the network architecture", so on dissimilar task sequences the basis **saturates** and plasticity
  dies (they report 45–78% of the maximum basis used on ResNet18).
- Maps to our code: same insertion point as LIT-011 (`core.py::_ridge_lstsq`, plus a new
  `cartridges/am/nullspace.py` that builds and caches `P`), but this entry supplies the knobs
  AlphaEdit hides: proposed `AM_NULLSPACE_RULE ∈ {eig_abs, eig_rel, energy}` (`eig_abs` = AlphaEdit's
  absolute 10⁻²; `eig_rel` = Adam-NSCL's smallest-singular-value rule; `energy` = GPM's `ε_th`),
  `AM_NULLSPACE_TAU`, and OWM's recursion as `AM_NULLSPACE_RECURSIVE=1` for the QA→MT→SA chain —
  after MT is written, fold `A_MT` into the accumulated Gram and re-derive `P` in closed form, with no
  re-solve of anything else and no stored data.
- **Prediction (about OUR signature):** the threshold **is** the plasticity/stability dial, and I
  predict a single τ sweep traces our **entire Pareto front with one mechanism**: `ε_th → 1` (τ → 0)
  ⇒ writable subspace ≈ ∅ ⇒ QA ≈ the Phase-1 floor **2.2388** and MT ≈ the untrained ceiling **3.7826**;
  `ε_th → 0` (τ → ∞) ⇒ `P = I` ⇒ exactly today's numbers (QA 2.18–2.25, MT ≈ 2.54). The win condition
  (QA ≤ 2.52 **and** MT ≤ 2.02) therefore requires an **interior** τ that beats *both* endpoints on
  MT — which can only exist if MT's routing energy outside the QA subspace is substantial
  (LIT-011's `ρ`). **Falsifiable in one sweep:** if τ produces a monotone trade with MT never below
  ~2.5 anywhere on the curve, projection-based methods are refuted for us and our deficit is not an
  interference problem at all.
- Cost: **gradient-free, `gradient_steps = 0`.** GPM's `n_s` is only a few hundred samples per task,
  so the QA statistic need not be large (this bounds the LIT-009 collection cost from above). OWM's
  recursion makes the N-stage version `O(t²)` per stage with **zero** stored old data — the cheapest
  path to a 3-stage QA→MT→SA chain we have found. One EVD/SVD per (layer, head), once per stage.
- Why it might NOT transfer: all three project **gradients**, and we have none — we apply the
  projector to a *closed-form solution* instead. That is mathematically fine (solve inside the
  P-subspace) but it removes the assumption these methods lean on: an SGD step is *small*, so
  "old outputs unchanged" is a well-justified linearisation. Our updates are **not** small — the
  logged `value_max_abs` of 816 (EXP-001) and 486 (EXP-003) is an order of magnitude above normal KV
  scale — so a projector guarantees `A_old ΔV_S ≈ 0` only to the extent that `A_old` itself does not
  move, which β and key rewrites would violate. GPM's saturation warning also applies directly: with
  only 512 slots, each stage consumes null-space dimensions permanently.
- Related: LIT-011, LIT-014, LIT-015, LIT-017.

---
### LIT-013: Delta rule / fast weights — the *key* is the thing that must be chosen, and non-negative keys can only be orthogonal by disjoint support (arXiv 2102.11174; DeltaNet arXiv 2406.06484)
- Status: candidate
- Board entry: **B-ROUTE**
- Mechanism (1 paragraph): in *Linear Transformers Are Secretly Fast Weight Programmers* (Schlag,
  Irie, Schmidhuber, ICML 2021) the purely additive (Hebbian) write
  `W⁽ⁱ⁾ = W⁽ⁱ⁻¹⁾ + v⁽ⁱ⁾ ⊗ φ(k⁽ⁱ⁾)` interferes as soon as keys are non-orthogonal: "to prevent
  associations from interfering with each other upon retrieval, the respective keys need to be
  orthogonal. Otherwise the dot product will attend to more than one key and return a linear
  combination of values", and "with keys embedded in a `d_dot` space, there cannot be more than
  `d_dot` orthogonal vectors" — empirically, `d_key = d_dot = 64` "begins to accumulate errors with 60
  or more associations". The **delta rule** fixes the *write*, not the key: read out what is currently
  stored at that key, `v̄⁽ⁱ⁾ = W⁽ⁱ⁻¹⁾φ(k⁽ⁱ⁾)`, then write only the correction,
  `W⁽ⁱ⁾ = W⁽ⁱ⁻¹⁾ + β⁽ⁱ⁾(v⁽ⁱ⁾ − v̄⁽ⁱ⁾) ⊗ φ(k⁽ⁱ⁾)`, with `β⁽ⁱ⁾` the **write strength** ("to which extent
  the new value will replace the previous value"). Everything else in that literature is about making
  keys retrievable: DPFP `φ_{iν}(k) = ReLU([k;−k])_i · ReLU([k;−k])_{i+ν}` to raise effective
  orthogonality, sum-normalisation `φ'(q) = φ(q)/Σ_j φ(q)_j` so the positive and negative terms of the
  delta write stay balanced in the overcapacity regime, and in DeltaNet (Yang et al., NeurIPS 2024)
  ℓ2-normalised keys with `β ∈ (0,2)`, giving the generalised-Householder recurrence `(I − βkkᵀ)`.
- Maps to our code: **the delta-rule *write* is already what we do** — the residual formulation at
  `value_solve.py:79-83` (`residual = targets − α_{¬S}V_{¬S}`) is exactly "subtract what is currently
  stored, then solve for the correction", and our batch least-squares is the converged limit of
  repeated delta steps. So the transferable content is **not** the update rule; it is the **key
  condition**, which has two concrete code consequences. (i) `cartridges/am/ranking.py::rank_am_slots`
  should score slots by a **margin**, not by new-content mass alone:
  `score_j = MT_mass_j − μ·QA_mass_j` — proposed `SLOT_SELECTION=mass_margin` with
  `AM_SLOT_MARGIN_MU` (μ = 0 recovers today's pure-TF selector **bit-identically**). This is the
  correct specialisation of orthogonality to our store: because routing vectors are non-negative,
  `⟨a_old, a_new⟩ ≥ 0` always, **with equality iff the slots QA reads and the slots MT reads are
  disjoint** — so "orthogonal keys" here literally means "disjoint support". (ii) The candidate pool
  at `finetune.py:496-499` for `KEY_MODE ≠ freeze` should be scored the same way (→ LIT-017).
- **Prediction (about OUR signature):** the capacity/orthogonality statement gives a checkable
  quantity. Per (layer, head), define the **histogram intersection**
  `overlap = Σ_{j=1..512} min(ā_QA,j, ā_MT,j)` between the mean QA and mean MT routing vectors. I
  predict `overlap` is **large (> 0.5)** — the two tasks read the same slots — and, worse, that the
  current TF selector *increases* it, because TF picks the slots receiving the largest MT mass and
  those are also the highest-mass QA slots (attention sinks / high-norm slots). That reframes the
  K-GATE result (pure TF beat TF-IDF on **both** axes, QA 2.252 vs 2.635 and MT 2.543 vs 3.007) as
  "IDF is the *wrong* orthogonality proxy, but some orthogonality term is still needed": IDF is a
  corpus-level document-frequency statistic, whereas the quantity the theory names is the **actual QA
  attention mass on that slot**. `mass_margin` at small μ should therefore beat both pure TF and
  TF-IDF at identical cost. **Falsifier for a whole family:** if `overlap` is small (< 0.2), then
  interference is not our problem and LIT-011/012/013 are all refuted at once.
- Cost: **gradient-free, `gradient_steps = 0`, no extra solves.** The margin selector needs one extra
  512-vector per (layer, head) — the mean QA routing vector `ā_QA` — which is **the diagonal of the
  same `G` matrix LIT-009 already collects**, so it is free if `G` exists. No extra forward passes
  beyond that one QA pass, no extra memory beyond 512 floats × 288 heads (≈ 0.6 MB).
- Why it might NOT transfer: the fast-weight literature writes into a **linear** store whose keys the
  model itself produces and can adapt end-to-end; ours is a softmax store with keys frozen from a
  different task, so we cannot *make* a key orthogonal — only choose among 512 pre-existing ones,
  which upper-bounds how disjoint the supports can be. DeltaNet's `β` is a learned per-step scalar
  gate; our nearest analogue (AM's per-key `β` logit bias) is currently broken (B-SOLVE, LIT-002).
  And capacity results stated for `d_dot` orthogonal directions do not directly bound a softmax store
  where retrieval is soft and every slot contributes.
- Related: LIT-016, LIT-017, K-GATE (HYP-G1), B-ROUTE.

---
### LIT-014: Test-time regression — softmax attention is *nonparametric*, so new content needs new keys; and RLS is the exact sequential write (arXiv 2501.12352)
- Status: candidate
- Board entry: **B-ROUTE** (primary, as a theoretical cap), **B-SOLVE** (secondary)
- Mechanism (1 paragraph): Wang & Li formalise associative recall as memorise-then-retrieve and cast
  memorisation as **weighted least squares**, `min_{m∈ℳ} ½Σ_{i≤t} γᵢ⁽ᵗ⁾‖vᵢ − m(kᵢ)‖²`, whose linear
  solution is `M_t = V_tᵀK_t(K_tᵀK_t)⁻¹`. Three design choices (regression weights, function class,
  test-time optimiser) then generate the whole zoo: **linear attention** = one Hebbian step with
  `K_tᵀK_t ≈ I`; the **delta rule** = one gradient step,
  `M_t ← M_{t−1} + α(v_t − M_{t−1}k_t)k_tᵀ`; **recursive least squares (RLS)** = the *exact* solution
  maintained online by Sherman-Morrison,
  `P_t = P_{t−1} − (P_{t−1}k_tk_tᵀP_{t−1})/(1 + k_tᵀP_{t−1}k_t)` with `P_t = (K_tᵀK_t)⁻¹`, and
  `M_t = M_{t−1} + (v_t − M_{t−1}k_t)k_tᵀP_t`; and **softmax attention = nonparametric
  (Nadaraya-Watson) kernel regression**. Their diagnosis of linear attention is ours verbatim: it
  "ignores the covariance between the dimensions of the key vectors" by approximating `K_tᵀK_t ≈ I`,
  and whitening by `(K_tᵀK_t)⁻¹` "accounts for inter-key correlations, preventing information loss
  when keys occupy correlated directions". **The load-bearing consequence for B-ROUTE:** because
  softmax attention is *nonparametric*, the model **is** its stored key-value pairs — a new
  association requires a **new key**, since there is no parametric weight into which it can be folded.
  A value-only write into frozen keys is therefore a request for a nonparametric regressor to
  represent a new function **without new sample points**; it can only re-weight what the existing
  kernel already places mass on.
- Maps to our code: (a) the whitening statement is the same one-line change as LIT-009/010
  (`core.py::_ridge_lstsq`: `λI → C₀`); (b) the RLS recursion is the **N-stage extension** — after MT
  is written, fold its routing into the Gram (`C ← C + A_MTᵀA_MT`) so that a Phase-3 SA write
  automatically preserves *both* QA and MT with **no stored data and no re-solve**. Proposed opt-in
  flag `AM_SEQUENTIAL_GRAM=1`, with the running Gram persisted beside the cache in
  `cartridges/cache.py::save` / `::from_pretrained` (mirror the `trainable_beta` pattern at `:243`
  and `:308`).
- **Prediction (about OUR signature):** this entry predicts the **sign and the bound** of the
  ORACLE-WRITE result. If softmax attention is nonparametric in the strict sense above, substituting
  *any* values into frozen slots — the teacher's own document values included — changes only the
  regressor's values at existing sample points, so the achievable change in the head output is bounded
  by the attention mass those slots receive from MT queries:
  **`‖Δo(q)‖ ≤ mass_on_S(q)·(value range)`**. That quantity is **already computed** in our code
  (`value_solve.py:146`, `X.sum(-1).mean()`, and aggregated per layer at `finetune.py:628` as
  `oracle_ref_mass_on_S_per_layer`) but **has never been recorded in any bundle in
  `research_loop/results/`**. Concretely I predict `mass_on_S_mean` for MT reference queries at
  `top_t = 64` is **well under 0.5**, and that post-oracle MT tracks `2.548 − c·mass_on_S` rather than
  approaching the teacher's 1.87. Reading this number is a **zero-GPU** action wherever an
  ORACLE-WRITE log exists.
- Cost: **gradient-free, `gradient_steps = 0`.** The RLS form is `O(t²)` per stage and **removes** the
  need for `OLD_REF_DATA_PATH`/`ENABLE_OLD_REFERENCE_GUARD` entirely after stage 1 — strictly cheaper
  than what we run today, which re-collects 64 old queries per document.
- Why it might NOT transfer: the framework's exact-RLS results are for **linear** memories; softmax
  attention is the *nonparametric* member of the family, and the paper gives **no closed-form write**
  for it — which is precisely the gap AM.pdf §6 also names, so this entry corroborates LIT-004 rather
  than escaping it. Our `(KᵀK)⁻¹` analogue whitens the **routing** Gram, not the model's key Gram, so
  the transfer is by analogy. And "new keys required" is an argument about the *ideal* regressor; a
  frozen 30-layer network downstream may still recover usable signal from a metastable mixture.
- Related: LIT-009, LIT-013, LIT-015, LIT-016, LIT-004, B-ROUTE oracle.

---
### LIT-015: A closed-form, sequential, **erasable** write into a FIXED 512-slot memory — *Larimar: LLMs with Episodic Memory Control* (arXiv 2403.11901, ICML 2024)
- Status: candidate
- Board entry: **B-CAP** (primary), forgetting / N-stage chain (secondary)
- Mechanism (1 paragraph): a Kanerva-style episodic memory `M ∈ R^{K×C}` with **K = 512 fixed slots**
  sitting beside a **frozen** LLM. Writing an episode with encodings `Z` and addressing matrix `W₀` is
  the pseudo-inverse / least-squares solution **`M = W₀†Z`** — Bayesian memory update reformulated as
  "finding least-square solutions to linear systems". Sequential writes maintain a key covariance and
  update in **exact RLS form**:
  **`Cᵢ = Cᵢ₋₁ + αᵢWᵢᵀWᵢ`**, **`Mᵢ = Mᵢ₋₁ + αᵢCᵢ⁻¹Wᵢᵀ(Zᵢ − WᵢMᵢ₋₁)`**, with `αᵢ = +1` to write, so
  that `M` remains the exact least-squares solution for the growing data; setting **`αᵢ = −1` erases**
  an episode, leaving `M` the exact least-squares solution with that episode removed. One-shot, no
  gradients, 8–10× faster than editing baselines, competitive in the **sequential** editing setting.
  Reported capacity at K = 512: rewrite accuracy ~100% up to **512 edits**, dropping to **82% at
  1024**.
- Maps to our code: this is the closest published system to what we are building — same memory size,
  same frozen LM, same one-shot closed-form write, same sequential requirement. Term for term,
  `Mᵢ − Mᵢ₋₁ = Cᵢ⁻¹Wᵢᵀ(Zᵢ − WᵢMᵢ₋₁)` **is** our
  `ΔV_S = (C₀ + A_newᵀA_new)⁻¹A_newᵀ(targets − αV)` with `Wᵢ ↔ A_new` and `Zᵢ ↔ targets` — the only
  difference being that Larimar's `Cᵢ` accumulates **every previous episode's addresses**, whereas
  ours is `1e-2·I`. Proposed opt-in flag `AM_PRECOND=rls` (shares the accumulator with LIT-014's
  `AM_SEQUENTIAL_GRAM`), the running Gram persisted in `cartridges/cache.py`, and `AM_ERASE_ALPHA=-1`
  as a Phase-3 unlearning knob (`value_solve.py`).
- **Prediction (about OUR signature):** two, both checkable against data we already produce.
  (i) **The capacity number is directly comparable and we are far past it.** 512 slots hold ~512
  one-shot writes at full fidelity — about **one write per slot**. We ask `top_t = 64` slots to absorb
  an entire document's associations, for each of many documents, on a *shared* support: 1–2 orders of
  magnitude past Larimar's per-slot budget. I therefore predict per-document MT quality **degrades
  with document index** within a single Phase-2 run (later documents overwrite earlier ones on the
  same slots), which is measurable from the existing per-document `am_stats` / `mean_mse` series
  **with no new GPU run**; if true, the reported MT 2.548 is the average of a decaying curve, not a
  stable operating point, and per-document slot *disjointness* (a `ranking.py` change) becomes a
  first-class lever. (ii) Replacing `1e-2·I` with the accumulated `Cᵢ` should show up first as a
  **reduction in the variance of per-document `mean_mse` across documents**, before it shows up in CE.
- Cost: **gradient-free, `gradient_steps = 0`**, `O(t²)` per document, and it **removes** the
  `OLD_REF_DATA_PATH` forward pass after the first stage (the Gram absorbs it). Memory: one `t×t`
  (or `512×512`) matrix per (layer, head), the same 302 MB object as LIT-009.
- Why it might NOT transfer: Larimar's addressing matrix `W` is (pseudo-)random and its encoder is
  **trained** to keep the address space well-conditioned; our addresses are softmax routing vectors we
  neither choose nor condition, and the pseudo-inverse write assumes episodes share the address space
  benignly. Larimar also has a trained encoder producing `Z`; our `Z` is the frozen LM's teacher
  attention output, fixed. And its 512 slots serve *only* the memory, whereas our 512 slots must also
  keep serving Phase-1 QA.
- Related: LIT-009, LIT-012, LIT-014, B-CAP.

---
### LIT-016: Capacity is a property of the **keys** — modern Hopfield separation `Δᵢ` and metastable averages (*Hopfield Networks is All You Need*, arXiv 2008.02217, ICLR 2021)
- Status: candidate
- Board entry: **B-ROUTE**
- Mechanism (1 paragraph): the continuous modern Hopfield energy with interaction `F(x) = exp(x)` has
  **exponential** storage capacity `~2^{d/2}` in the pattern dimension, converges in **one update**,
  and that update **is** transformer attention `softmax(βQKᵀ)V`. Whether a stored pattern `xᵢ` is
  actually retrievable is governed by its **separation**
  **`Δᵢ = min_{j≠i}(xᵢᵀxᵢ − xᵢᵀx_j)`**: with large `Δᵢ` the query converges to a fixed point
  exponentially close to `xᵢ`; with small `Δᵢ` the fixed point is a **metastable average** of the
  poorly separated patterns. Capacity is thus entirely a property of the **keys**; the values are
  merely what gets read out once routing has been decided.
- Maps to our code: diagnostic for the value path, prescriptive for the key path. Compute per
  (layer, head): the separation `Δ_j` of each of the 512 cartridge keys, and the entropy `H(a)` of the
  routing distribution for MT vs QA queries. `cartridges/am_stability_probe.py` already computes
  `mass_on_S_mean/min/p10` (`:146-148`, `:243-251`) and is the natural home — proposed opt-in flag
  `AM_PROBE_SEPARATION=1`. Prescriptive corollary for `cartridges/am/key_select.py`: a rewritten key
  is only *retrievable* if the rewrite **increases** its separation against the other 511 with respect
  to MT queries — a criterion neither `select_keys_highest_attention` nor `select_keys_omp` optimises.
- **Prediction (about OUR signature):** for MT queries against a cartridge whose keys were fitted to
  QA, I predict `H(a_MT)` sits close to the uniform limit `log 512 ≈ 6.24` nats and is materially
  higher than `H(a_QA)` — i.e. **no cartridge slot is a well-separated fixed point for MT content**,
  so MT queries land on a metastable average of many slots. This is the precise, non-hand-wavy version
  of B-ROUTE, and it makes a **quantitative** claim the write-ceiling oracle can be scored against:
  the fraction of the head output any value-only write can control is exactly `mass_on_S`, and under
  near-uniform MT routing `mass_on_S ≈ t/512 = 0.125` at `top_t = 64` (0.0625 at t=32, 0.25 at t=128).
  **If the oracle write moves MT by much less than a ~12.5% output share implies, the cap is
  confirmed and it is a key/separation cap, not a value cap** — and note this predicts the observed
  *flatness* of MT across `top_t` 32/64/128 (2.548 / 2.543 / 2.686) only if the extra mass is being
  spent on QA-relevant slots, which is exactly what the monotone QA degradation shows.
- Cost: **gradient-free, `gradient_steps = 0`, no extra solves, no extra forward passes** — two extra
  reductions over the `alpha` matrix that `compute_attention_weights` already materialises.
- Why it might NOT transfer: Hopfield capacity results assume patterns are (near-)random, norm-matched
  and well-separated on a sphere; our 512 cartridge keys are the output of Phase-1 self-distillation
  and are none of those. The "one update" retrieval story concerns a recurrent map, whereas our
  attention is applied once inside a much larger network — 30+ downstream layers may recover
  information that a single metastable average appears to lose. And the `t/512` uniform-routing
  estimate is an upper bound on *flatness*, not a measurement.
- Related: LIT-013, LIT-014, LIT-017, B-ROUTE oracle.

---
### LIT-017: **SYNTHESIS** — null-space **key** placement (AlphaEdit applied to `C_k` instead of `C_v`)
- Status: candidate *(synthesis, not a single published method: derived from LIT-011 + LIT-012 +
  LIT-013 + LIT-014 + LIT-016; it is also precisely the open problem AM.pdf §6 names — "move away
  from subset selection for `C_k`", "architectures that explicitly operate over a fixed set of keys
  and values")*
- Board entry: **B-ROUTE** (primary), forgetting (secondary)
- Mechanism (1 paragraph): every mechanism above operates on **values** and therefore inherits the
  routing cap LIT-014/LIT-016 describe. Turn the null-space idea 90°: instead of projecting the
  **value** update into the null space of the old **routing vectors**, choose the **keys** of the
  rewritten slots to lie in the approximate null space of the old **query** distribution while
  aligned with the new one. Let `Q₀ = Σ_{q∈QA} qqᵀ ∈ R^{d×d}` (d = 128) with `Q₀ = UΛUᵀ`, let `U_r` be
  its top-`r` eigenvectors (`r` chosen by GPM's energy rule, LIT-012), and set `P⊥ = I − U_rU_rᵀ`.
  Place the new key as **`k_new = c · P⊥q̄_MT / ‖P⊥q̄_MT‖`**, where `q̄_MT` is the mean (or top
  principal direction) of that head's MT reference queries and `c` is chosen so the resulting MT logit
  `q_MTᵀk_new/√d` matches the logit the *teacher's own* document keys achieve. Then solve values on
  those keys exactly as now. Result: MT queries route to the rewritten slots (**acquisition**), QA
  queries produce a logit ≈ 0 there (**retention**), and the value solve finally operates on keys that
  route for the content being written — the property LIT-004 showed AM always has and we always lack.
  A **discrete, safer variant** needs no synthetic keys: keep the existing candidate pool at
  `finetune.py:496-499` (`[K_cartridge[S] ; K_doc]`) but rank candidates by the *whitened* score
  `(q̄_MTᵀk)·‖P⊥k‖` instead of attention RMS — prefer document keys that MT queries like **and** QA
  queries are blind to.
- Maps to our code: `cartridges/am/key_select.py` — a new `select_keys_nullspace(...)` and a new mode
  in `rewrite_keys_on_support` (`:227-284`); consumed at `cartridges/am/finetune.py:493-509`, the
  `key_mode != "freeze"` branch that is **already wired and completely untested** (every row in
  `results.csv` is `key=freeze`). Proposed opt-in flags `KEY_MODE=nullspace` (continuous synthesis) /
  `KEY_MODE=highest_attention_orth` (discrete selection), plus `AM_KEY_NULLSPACE_TAU`. Requires the
  QA **query** second moment `Q₀`, collected by the same hooks the `OLD_REF_DATA_PATH` path uses.
- **Prediction (about OUR signature):** this is the only mechanism in this block that predicts
  movement on **both** axes in the good direction. (i) Eval-time MT attention mass on the rewritten
  slots should rise **above** the frozen-key level — which is currently pinned by Phase-1 and, per
  LIT-016, may be near `t/512 ≈ 0.125` — toward the mass the teacher's document keys receive.
  (ii) QA attention mass on those same slots should **fall relative to `KEY_MODE=highest_attention`**,
  which has no orthogonality term and should show the classic key-theft signature (MT improves, QA
  collapses). (iii) Quantitatively: `KEY_MODE=highest_attention` → MT ≈ 2.2–2.4 with QA > 2.7 (a
  fail on the QA axis); `KEY_MODE=nullspace` → MT below ~2.3 with QA held under ~2.4.
  **Discriminator between them:** if both inflict the *same* QA damage, then the damage comes from
  *removing* the incumbent key, not from the new key's overlap with QA queries — in which case the
  lever is slot budget/disjointness (LIT-013, LIT-015), not key geometry, and this entry is refuted.
- Cost: **strictly gradient-free, `gradient_steps = 0`.** One `128×128` eigendecomposition per
  (layer, KV-head) — 288 tiny EVDs computed **once**; `Q₀` storage `36 × 8 × 128² × 4 B ≈ 19 MB`
  (16× cheaper than the routing Gram of LIT-009). No extra solves per document;
  `highest_attention` is the AM paper's cheapest selector (3 s for a 60k context, Table 3) versus
  OMP's 565 s.
- Why it might NOT transfer: (i) **off-manifold keys** — `k_new` is synthetic and post-RoPE, the
  frozen LM has never seen a key in that direction, and nothing constrains its norm distribution, so
  attention may behave pathologically (the discrete variant exists precisely to dodge this).
  (ii) Orthogonality yields a QA logit of **0, not −∞**; softmax is shift-invariant per row, so a zero
  logit can still capture mass when the other 511 logits are negative — the *level* must be set by β,
  which is currently broken (B-SOLVE, LIT-002), so this entry is **coupled to fixing β**.
  (iii) Every key taken for MT is a key removed from QA: `rewrite_keys_on_support` overwrites
  `keys[selected_indices]`, destroying whatever QA content those slots served, independent of where
  the new key points. (iv) **RoPE frame mismatch** — cartridge keys and document keys sit at different
  absolute positions (`doc_rope_offset`, handled at `core.py:20-42`), so `q̄_MT` and the candidates
  are not in a common frame without explicit care. (v) `Q₀` is a *second moment over a QA reference
  set*, not the QA eval distribution.
- Related: LIT-004, LIT-011, LIT-012, LIT-013, LIT-016, LIT-001 (β), B-ROUTE.

---
### LIT-018: The counter-evidence — "the covariance trap" (arXiv 2603.15518)
- Status: candidate *(falsifier for LIT-009 / LIT-010 / LIT-011 / LIT-012)*
- Board entry: **B-CAP / forgetting** (as a negative control)
- Mechanism (1 paragraph): a 2026 critique of covariance-based editing argues that `C₀⁻¹`
  preconditioning (MEMIT) and null-space projection (AlphaEdit) are **geometrically equivalent** —
  "different manifestations of the same geometric misconception" — and that both "impose unnecessary
  constraints that offer marginal gains in locality while compromising robust generalization", with
  the projection matrix additionally introducing "numerical noise and geometric distortion". Its
  positive claim is that the covariance of the *pre-existing* distribution is the wrong object when
  the new and old content share subjects/contexts, and that relational structure must be modelled
  instead.
- Maps to our code: **no new code.** This entry is a prediction about the *outcome* of LIT-009 vs
  LIT-011 vs a plain `DELTA_WEIGHT` sweep, i.e. it defines the negative control those entries need.
- **Prediction (about OUR signature):** if the critique holds here, then a `AM_PRECOND=memit_c0` arm
  and an `AM_PRECOND=alphaedit` arm (identical `top_t`, queries, seed) land **within the noise floor
  (≤ 0.15) of each other on both axes**, and both sit on the **same QA/MT trade curve traced by simply
  sweeping `DELTA_WEIGHT`** — which is `C₀ = w·I`. That is a cheap three-arm test of whether the
  *directional* structure of `C₀` matters at all in our store, or only its **scale**. Note the control
  arm is nearly free and has never been run: **`DELTA_WEIGHT` has never been swept — every AM row in
  `results.csv` is `delta_w=1e-2`** (HYP-R0/K-RIDGE refuted `RIDGE_LAMBDA`, a different knob). If a
  `DELTA_WEIGHT` sweep alone reproduces whatever `C₀` achieves, the entire preconditioning family
  (LIT-009/010/011/012) collapses to an already-available scalar knob.
- Cost: none beyond the arms it evaluates; the `DELTA_WEIGHT` control needs **zero** new code and
  **zero** new statistics.
- Why it might NOT transfer: it is a *same-subject generalization* critique aimed at FFN memories on a
  factual-editing benchmark (CounterFact/zsRE-style); our failure axis is acquisition of an entire new
  task in a compressed KV cache, and our "keys" are simplex routing vectors rather than MLP
  activations, so its geometric argument may simply not be about our object.
- Related: LIT-009, LIT-010, LIT-011, LIT-012, HYP-R0 (K-RIDGE).

---

> **SCOUT-KEYS (2026-07-28) — the key axis, LIT-019…LIT-027.** Dispatched after DIAG-ROUTING showed
> ρ_MT = 0.012, QA/MT mean-routing cosine 0.99899 and top-32 slot overlap 0.914 — i.e. **no value-space
> operator can separate the two axes** — while leaving the **128-dim per-head key/query geometry**
> completely unmeasured. Question answered here: **what closed-form, gradient-free procedures exist for
> choosing, synthesising or moving KEYS in a fixed-size associative store so a new write is retrievable
> by new queries without destroying retrieval of the old content?** Families: closed-form key
> *synthesis* (ROME/r-ROME/SUIT), null-space **key** placement in the old **query** second moment,
> associative-memory capacity & interference (Hopfield / Personnaz / modern Hopfield / scaling laws /
> fast weights), writable-key memory architectures (NTM, DNC, Kaiser-LRU, product-key memory, SDM), and
> RoPE-specific hazards. Plus three code-level hazards found by reading, and one free re-analysis of
> `DIAG-ROUTING.npz`.
>
> **NOTATION (differs from SCOUT-EDIT's on purpose — that is the whole point).** SCOUT-EDIT's "key" was
> the post-softmax routing vector `a_S(q) ∈ Δ^{511}`. **Here "key" means the actual 128-dim per-head
> vector `k_j ∈ R^d`, d = 128, stored in `cache.trainable_keys[l][0,h,j]`.** The head's logit is
> `ℓ_j(q) = q·k_j/√d + β_j`, `a(q) = softmax(ℓ(q))`. Write `Q₀^QA = Σ_{q∈QA} qqᵀ ∈ R^{128×128}` and
> `Q₀^MT` likewise; `q̄_QA`, `q̄_MT` are the mean reference queries. Qwen3-4B-Instruct-2507: 36 layers ×
> 8 KV heads (32 Q heads, GQA group = 4), d = 128, **rope_theta = 5,000,000**.
>
> **THE ONE ALGEBRAIC FACT THIS BLOCK RESTS ON (derive it once, used by every entry below).**
> Adding a constant `Δ` to the logits of the written slots `S` maps the measured masses as
> `m(Δ) = m₀e^Δ/(m₀e^Δ + 1 − m₀)`. With DIAG-ROUTING's measured `m₀`(MT) = 0.08961 and
> `m₀`(QA) = 0.08272 this gives, for the **same** Δ applied to both:
> | Δ (nats) | 0 | 1 | 2 | 2.32 | 3 | 5 |
> |---|---|---|---|---|---|---|
> | mass_on_S MT | .0896 | .211 | .421 | **.500** | .664 | .936 |
> | mass_on_S QA | .0827 | .197 | .400 | .478 | .644 | .930 |
> | **MT/QA ratio** | **1.083** | 1.071 | 1.052 | 1.046 | **1.031** | 1.006 |
> ⇒ **β (a per-key, query-independent logit bias) provably cannot raise the MT/QA mass ratio — it drives
> the ratio monotonically toward 1.** It buys bandwidth for both tasks at once. The *only* term that can
> be asymmetric is a **key change**, because its logit contribution `q·Δk/√d` is query-**dependent**:
> `Δ_MT − Δ_QA = (q̄_MT − q̄_QA)·Δk/√d ≤ ‖Δk‖·‖q̄_MT − q̄_QA‖/√d`.
> **The entire key-side lever is therefore proportional to `‖q̄_MT − q̄_QA‖` in the 128-dim query space —
> a quantity this loop has never measured.** DIAG-ROUTING measured `‖ā_MT − ā_QA‖/‖ā_MT‖ = 0.053` in the
> 512-dim *routing* space (recomputed here from `DIAG-ROUTING.npz`, reproducing cosine 0.99899 and
> overlap 0.9142 exactly) — that is a statement about what the **current 512 keys can resolve**, not
> about the queries. With a QA routing-Gram participation rank of **1.97**, the incumbent key set
> projects every query onto ≈2 effective directions; two query populations that are *orthogonal in
> R^128* would still produce near-identical routing through such a key set. **The measured 0.914 overlap
> is evidence of key blindness, not of query similarity.**

---
### LIT-019: Closed-form key **synthesis** by prefix-averaged activations — ROME (arXiv 2202.05262, NeurIPS 2022) and r-ROME (arXiv 2403.07175, EMNLP 2024)
- Status: candidate
- Board entry: **B-ROUTE** (primary), B-TARGET (secondary)
- Mechanism (1 paragraph): ROME never *selects* a key — it **constructs** one. The lookup key for a new
  fact is the MLP post-nonlinearity activation at the subject's last token,
  **`k* = (1/N) Σ_{j=1..N} k(x_j ⊕ s)`**, averaged over N texts formed by prepending **random prefixes
  of length 2–10 tokens** to the subject `s`. The averaging is not cosmetic: it is what makes the key a
  *direction the model reliably produces for that content across contexts* rather than one sample of it,
  and it is what makes the edit generalise to paraphrases. The edit is then the rank-one closed form
  `Δ = (v* − W₀k*)(k*ᵀC₀⁻¹)/(k*ᵀC₀⁻¹k*)` — an equality constraint at `k*`, least-squares preservation
  everywhere else. r-ROME's contribution is a **negative** result we should heed: the original
  implementation used the *prefix-averaged* `k*` in one place and an *un-prefixed* `k` in another, and
  that asymmetry alone causes **disabling edits and model collapse under sequential editing**; making
  the two consistent removes the instability. I.e. *the key you optimise against and the key the model
  will actually emit must be the same object.*
- Maps to our code: a new `select_keys_doc_centroid(...)` in `cartridges/am/key_select.py` plus a mode in
  `rewrite_keys_on_support` (`:227-284`), consumed at `finetune.py:516-532`. Instead of picking one
  literal document key from `candidate_keys = cat([original_keys[S], k_doc])` (`:519-522`), synthesise
  `k*_m = Σ_i w_{mi} R_{Δ_i} k_i^doc` where `w_{mi}` is the teacher's own attention mass from the
  reference queries onto doc token `i` (already computed inside `compute_teacher_targets`) and `R_{Δ}`
  is the RoPE re-basing `phase1.py::_rope_reposition` **already implements** (→ LIT-026). Proposed
  opt-in flag `KEY_MODE=doc_centroid`, with `AM_KEY_CENTROID_CLUSTERS=t` (t centroids by weighted
  k-means or by the top-t eigenvectors of `Σ_i w_i k_i k_iᵀ`, so the t written slots are not t copies of
  the same vector). r-ROME's lesson maps to a **required assertion**: the key used for selection/β and
  the key written into `k_param` at `finetune.py:634` must be byte-identical — today the selection frame
  and the storage frame differ (LIT-026).
- **Prediction (about OUR signature):** the AM paper's `C_k ⊂ K` restriction (LIT-004) exists because
  the paper has no principled synthesis; ROME says the principled synthesis is *mass-weighted averaging
  of the content's own activations*. Concretely: a doc-centroid key should raise `mass_on_S` on MT
  queries **above** the single-doc-key selector's, because it points at the doc's attention-weighted
  centroid rather than at one token, and it should raise it **asymmetrically** (MT/QA ratio > 1.04)
  exactly to the extent that `q̄_MT ≠ q̄_QA`. Quantitatively I predict `mass_on_S`(MT) ≥ 0.25 and
  MT/QA ratio in **[1.2, 2.5]** if `‖q̄_MT − q̄_QA‖/‖q̄_MT‖ > 0.3`, and ratio ≈ 1.0–1.1 (i.e. the whole
  key family is dead) if that norm is < 0.05. **Falsifier:** if `KEY_MODE=doc_centroid` moves
  `mass_on_S` but the ratio stays ≤ 1.1, then keys buy *bandwidth* only, exactly like β, and the
  acquisition/retention trade cannot be separated at all — B-ROUTE closes as `confirmed+capped`.
- Cost: **gradient-free, `gradient_steps = 0`.** ROME's only gradient is in computing `v*`; our target
  is `compute_teacher_targets`, already closed form. One weighted mean (or one t-cluster k-means, ~10
  Lloyd iterations on ≤ 8900 × 128 per layer/head) per document — cheaper than `highest_attention`'s
  512-way softmax. No extra forward passes; `w` is a by-product of the teacher softmax we already
  compute. Memory: negligible.
- Why it might NOT transfer: ROME's `k*` lives in an MLP's input space where the "association" is a
  learned linear map; ours must be a **post-RoPE attention key**, so the average of keys at different
  positions is an average of *differently rotated* vectors and is not the key of any token (→ LIT-026 —
  this is the entry's single biggest risk). ROME also has one incumbent-free slot per edit; every
  cartridge key we overwrite was serving QA. And ROME's N ≈ 50 prefixes probe *paraphrase* variation;
  our `w`-weighted average probes *token* variation, which is not the same invariance.
- Related: LIT-004, LIT-017, LIT-020, LIT-026, B-ROUTE.

---
### LIT-020: **SYNTHESIS** — null-space key placement in the old **QUERY** second moment `Q₀` (128-dim), and the theorem that separates it from β and from LIT-011 (AlphaEdit arXiv 2410.02355 · Adam-NSCL CVPR 2021 · GPM ICLR 2021 · SUIT arXiv 2509.24502)
- Status: candidate *(this is LIT-017's real content, now with the statistic, the sample size and the falsifier)*
- Board entry: **B-ROUTE** (primary), forgetting (secondary)
- Mechanism (1 paragraph): AlphaEdit/Adam-NSCL/GPM all build the same object — the projector off the
  dominant subspace of an old-data second moment — and all three apply it to a **weight update**.
  DIAG-ROUTING applied it to the **routing** Gram and got ρ_MT = 0.012 (dead). The untried 90° turn is
  to apply it in the **query/key space**: take `Q₀^QA = Σ_{q∈QA} qqᵀ ∈ R^{128×128}` per (layer, KV-head),
  `Q₀^QA = UΛUᵀ`, keep the top-r by GPM's energy rule `Σ_{i≤r}λ_i ≥ ε_th·tr(Q₀)`, set
  `P⊥ = I − U_rU_rᵀ`, and place the new key as **`k* = c·P⊥q̄_MT/‖P⊥q̄_MT‖`**. Then for every QA query
  `q ∈ span(U_r)` the logit `q·k*/√d ≈ 0` **by construction**, while MT queries with energy outside
  `span(U_r)` see `c‖P⊥q̄_MT‖/√d > 0`. `c` is set so the MT logit matches what the teacher's own doc keys
  achieve (that target is already computed — `compute_teacher_log_mass`, `finetune.py:508`).
  SUIT (arXiv 2509.24502, 2025) is the closest published instance of the same idea in editing: it
  reports that constructing key/value vectors **without** a subspace constraint causes large hidden-state
  perturbation, and confines both the key and the value to an "edit-critical subspace" identified by SVD
  — closed-form at edit time, with preserved knowledge protected by orthogonality rather than by penalty.
  **The decisive scalar is the key-space analogue of DIAG-ROUTING's ρ:**
  **`ρ_key(r) = tr(P⊥_r Q₀^MT)/tr(Q₀^MT)`**, the fraction of MT *query* energy outside QA's dominant
  query subspace.
- Maps to our code: new `cartridges/am/key_select.py::select_keys_nullspace(...)`, a `nullspace` branch in
  `rewrite_keys_on_support` (`:227-284`), consumed at the **already-wired, never-run**
  `finetune.py:516-532`. Needs one new statistic, `Q₀^QA` per (layer, KV-head), collected by the *same*
  hooks `_collect_reference_queries` (`finetune.py:272-334`) already uses on `OLD_REF_DATA_PATH` /
  `ENABLE_OLD_REFERENCE_GUARD=1`, cached beside the cartridge (mirror the `trainable_beta` persistence in
  `cartridges/cache.py::save` `:237` / `::from_pretrained` `:255`). Proposed opt-in flags
  `KEY_MODE=nullspace`, `AM_KEY_NULLSPACE_RULE ∈ {eig_rel, energy}`, `AM_KEY_NULLSPACE_TAU`,
  `AM_QA_QUERY_GRAM_PATH`. A discrete, synthesis-free variant needs no new key at all: rescore the
  existing candidate pool at `finetune.py:519-522` by `(q̄_MT·k)·‖P⊥k‖` instead of attention RMS
  (`KEY_MODE=highest_attention_orth`).
- **Prediction (about OUR signature):** three, in order of cost, and the first one decides the family.
  (i) **`ρ_key` ≫ `ρ_route` = 0.012.** Reason: `ρ_route` is measured *through* a key set whose routing
  Gram has participation rank **1.97**, so it is upper-bounded by what those keys can resolve; `ρ_key`
  is measured in the raw 128-dim query space where nothing has been projected away. I predict
  **ρ_key ∈ [0.15, 0.6]** at the GPM 99%-energy rule, i.e. **10–50× ρ_route** — and equivalently
  `‖q̄_MT − q̄_QA‖/‖q̄_MT‖ ≥ 0.3`, versus the 0.053 measured in routing space. (ii) If ρ_key lands in
  that band, `KEY_MODE=nullspace` should push the **MT/QA mass ratio from 1.04 to ≥ 1.5** at
  `mass_on_S`(MT) ≥ 0.3 — the first mechanism on the board that moves the *ratio* rather than the level
  — and MT toward 2.1–2.3 with QA held under ~2.4 (Phase-1 floor 2.2388; budget 2.52). (iii) If instead
  **ρ_key < 0.05**, then MT and QA queries are genuinely the same direction in R^128, no key placement
  can be asymmetric, and **B-ROUTE closes as `confirmed+capped` for keys as well as values** — the whole
  mission would have to move to β/bandwidth (which raises both axes together) or off the fixed-512-slot
  formulation. **This single number is the cheapest decisive measurement left on the board** (below).
- Cost: **gradient-free, `gradient_steps = 0`.** `Q₀` collection = **one** forward pass over the QA
  reference set with the existing query hooks (DIAG-ROUTING did QA+MT collection in 5.7 s total for
  49k/40k query rows per head). 288 symmetric `128×128` EVDs (milliseconds). Storage
  `36×8×128²×4 B ≈ 19 MB` — **16× cheaper than LIT-009's routing Gram**. Zero extra solves per document.
- Why it might NOT transfer: (i) **orthogonality gives logit 0, not −∞**; softmax is shift-invariant per
  row, so whether a zero logit is "invisible" depends on where the other 511 logits sit — and our
  measured routing says one incumbent slot already owns ~52% of cartridge mass (→ LIT-021), so a
  logit-0 key is likely *below* the incumbents rather than tied with them, which is favourable for
  retention but means `c` must be large for acquisition. (ii) `Q₀^QA` comes from `OLD_REF_DATA_PATH`,
  not the QA eval split, exactly as AlphaEdit's `C₀` comes from Wikipedia rather than the eval set; the
  reproducibility study (arXiv 2606.26783) already showed the protection is *bounded, not
  unconditional*, once a nonlinearity intervenes — and between our `V` and CE sit a softmax we are also
  perturbing, `o_proj`, RMSNorm and 30+ layers. (iii) The synthesised key is **post-RoPE**, so `P⊥q̄_MT`
  is only meaningful at a stated position (→ LIT-026). (iv) Overwriting a key destroys whatever QA
  content that slot served regardless of where the new key points (`rewrite_keys_on_support:283`).
- Related: LIT-011, LIT-012, LIT-017 (this entry supersedes its prediction section), LIT-019, LIT-021,
  LIT-026, DIAG-ROUTING.

---
### LIT-021: Attention sinks and massive activations — key **norm** is a mass lever, and one incumbent slot already owns half the cartridge (*The Spike, the Sparse and the Sink*, arXiv 2603.05498 · StreamingLLM, arXiv 2309.17453 · *Massive Activations in LLMs*, arXiv 2402.17762)
- Status: candidate
- Board entry: **B-ROUTE** (primary), B-CASCADE (secondary)
- Mechanism (1 paragraph): a small number of tokens attract disproportionate attention **regardless of
  semantic relevance**. 2603.05498 gives the geometry: diverse "spike" tokens collapse to nearly the
  same vector, so their keys are **low-dimensional and near-invariant across prompts**, and the learned
  key projection maps spike keys and non-spike keys into **distinct subspaces**; a head becomes a sink
  head exactly when *its query subspace aligns more with the fixed sink-key subspace than with the
  non-sink-key subspace*. StreamingLLM's operational finding is the same object from the other side —
  the first few keys act as a bias term that absorbs unwanted mass, and removing them collapses the
  model. Two consequences for a key-side write. **(a) The lever:** because the logit is `q·k/√d`, key
  *norm* is a free parameter — a synthesised key can be scaled until it captures any desired mass. This
  is the one thing values cannot do (LIT-014's nonparametric bound `‖Δo(q)‖ ≤ mass_on_S·range` binds
  values, not keys). **(b) The hazard:** a sink key captures mass from *every* query, so scaling a key
  is exactly how you manufacture a new sink and destroy retention.
- Maps to our code: `cartridges/am/key_select.py::rewrite_keys_on_support` — a norm policy on the written
  key (`AM_KEY_NORM_MODE ∈ {as_is, match_incumbent, logit_target}` + `AM_KEY_LOGIT_TARGET`), and a
  protection list so the incumbent sink slot is never overwritten (`AM_KEY_PROTECT_TOPK`, computed from
  the same access scores `ranking.py::rank_am_slots` already has). Diagnostics belong in
  `cartridges/am_stability_probe.py` alongside `mass_on_S`.
- **Prediction (about OUR signature) — with the free measurement that motivates it.** Recomputing
  `DIAG-ROUTING.npz` (no GPU): **the top-1 slot carries 52.5% of cartridge routing mass on average
  (median 58.1%), and in 29 of 36 layers it is the SAME slot index for all 8 KV heads.** Routing entropy
  is 2.81 (QA) / 2.89 (MT) nats = **40–43 effective slots of 512**. So: (i) I predict
  `KEY_MODE=highest_attention` **will select and overwrite that sink slot** (its selector is attention
  RMS, `key_select.py:79`, which the sink maximises by construction), and that this — not the new key's
  direction — will produce the folkloric "keys collapse QA": QA should jump **past 2.52 and plausibly
  past 2.7**, with total cartridge attention falling below the 0.588 baseline the same way
  B-CASCADE measured for values. (ii) The same run with the top-1 slot per layer **protected** should
  lose far less QA at nearly the same MT, and the difference between the two arms is a clean, one-bit
  attribution of the collapse to sink destruction rather than to key geometry. (iii) Scaling a
  synthesised key to `logit_target` reproduces the β table above **plus** an asymmetry term, so
  `mass_on_S` should be tunable continuously from 0.09 to > 0.6 — but any arm whose MT/QA ratio stays
  at ≈ 1.04 while `mass_on_S` rises is behaving as a **sink**, not as a selective key, and is therefore
  β in disguise.
- Cost: **gradient-free, `gradient_steps = 0`, zero extra forward passes** — the sink slot is the argmax
  of an access-score vector the pipeline already accumulates (`query_accum.py:114`), and the norm policy
  is one scalar per written key.
- Why it might NOT transfer: sink analyses are done on *pretrained* caches of real tokens; our 512 slots
  are **self-distilled parameters** (`outputs/phase1_selfdistill_qwen512`), so the "sink" we measure is
  a learned artefact of Phase-1 and need not obey the spike-token geometry. Also, 2603.05498's
  sink-key/non-sink-key subspace split is a statement about `W_K` applied to token activations; our
  keys bypass `W_K` entirely.
- Related: LIT-016, LIT-020, LIT-024, B-CASCADE, B-ROUTE.

---
### LIT-022: What the capacity bounds actually say for 512 slots, 16 documents and d = 128 (Hopfield 1982/Amit-Gutfreund-Sompolinsky · Personnaz et al. 1985 projection rule · modern Hopfield arXiv 2008.02217 · *Scaling Laws for Associative Memories* arXiv 2310.02984 · *Birth of a Transformer* arXiv 2306.00802 · fast weights arXiv 2102.11174)
- Status: candidate
- Board entry: **B-CAP** (primary), **B-ROUTE** (secondary)
- Mechanism (1 paragraph): the field gives four bounds, and they disagree about *what* is scarce.
  (1) **Hebbian outer-product store:** capacity `≈ 0.138 N` patterns for N units; beyond it the crosstalk
  term wins and retrieval fails catastrophically. (2) **Projection / pseudo-inverse rule**
  (Personnaz et al. 1985), `J = ξ(ξᵀξ)⁻¹ξᵀ` — *this is our least-squares solve* — stores up to **N
  linearly independent patterns exactly**, but basins of attraction vanish past **N/2**, so exact
  storage ≠ usable retrieval. Its advantage over Hebb is precisely that it **degrades far less as
  patterns become correlated**, which matters here because our routing vectors have cosine 0.999.
  (3) **Modern Hopfield** (2008.02217): exponential capacity `~2^{d/2}` in the *pattern* dimension, one-step
  retrieval, and retrievability governed by the **separation** `Δ_i = min_{j≠i}(x_iᵀx_i − x_iᵀx_j)`;
  poorly separated patterns return a **metastable average**. (4) **Random-embedding capacity**
  (2310.02984, 2306.00802): with embeddings sampled on the sphere in `R^d`, an outer-product memory
  stores a number of associations **proportional to its parameter count**, strictly more than the `d`
  you get from insisting on exact orthogonality; and (2102.11174) measures the practical limit at
  `d_dot = 64` as errors accumulating from **~60 associations**.
- Maps to our code: no code change — this entry **sizes** the others. Our store per (layer, KV-head) is
  `N = 512` slots of dimension `d = 128`; we write `t = 32` slots per document for **16 documents**, all
  drawn from a support union of only **~55 slots per layer** (measured, below). Bound (2) says the
  *value* solve can in principle satisfy up to 512 independent constraints; bound (4) says the *key*
  space supports O(d) = O(128) near-orthogonal directions per head, i.e. **~8 mutually distinguishable
  keys per document at 16 documents**, or ~128 if documents share directions gracefully.
- **Prediction (about OUR signature):** **capacity is not our binding constraint; separation is.**
  16 documents × 32 slots = 512 write-slots nominally, but the measured support union is 55 slots/layer,
  so each slot is written **~9 times** (16 docs × 32 / 55) — that is inside Personnaz's N/2 usable
  regime only if the write directions are near-independent, and they are not (routing cosine 0.999).
  Concretely I predict: (a) per-document MT quality **decays with document index** within a Phase-2 run
  (later documents overwrite earlier ones on the same ~55 slots) — measurable **with zero GPU** from the
  per-document `am_stats.mean_mse` series already in every run's `phase2_summary.json`, and already
  partially visible (DIAG-OBJ-a: `am/mean_mse` first doc 0.00851 → last doc 0.19808, a **23× rise**);
  (b) forcing per-document slot **disjointness** (16 docs × 32 = 512 distinct slots, exactly one write
  per slot — Larimar's measured full-fidelity regime, LIT-015) should collapse that 23× rise; (c) if
  disjointness does **not** improve MT, capacity/interference is refuted and the cap is separation, i.e.
  keys. **Falsifier for this entry:** if `mean_mse` is flat across document index in some run, the 23×
  is a per-document difficulty effect, not interference.
- Cost: **zero.** The document-index series is already logged; the disjointness variant is a change to
  `ranking.py::rank_am_slots` (exclude already-written slots), no extra solves, no extra passes.
- Why it might NOT transfer: every bound above is for a memory whose retrieval is the *identity* of a
  stored pattern; ours is a soft mixture read by 30+ downstream layers that may recover information a
  "metastable average" appears to lose. Hopfield/Personnaz capacities also assume patterns are the
  *unknowns*; in our store the 512 keys are fixed and only 55 values move.
- Related: LIT-013, LIT-015, LIT-016, LIT-023, B-CAP.

---
### LIT-023: Sparse Distributed Memory — the formal condition under which value-only writing into **fixed random** keys works (Kanerva 1988 · *Attention Approximates SDM*, arXiv 2111.05498, NeurIPS 2021 · *SDM is a Continual Learner*, arXiv 2303.11934, ICLR 2023)
- Status: candidate
- Board entry: **B-ROUTE** (primary), forgetting (secondary)
- Mechanism (1 paragraph): SDM fixes `N` "hard locations" — **random addresses that are never moved** —
  and writes a datum by *adding* its value into **every** hard location within Hamming distance `d` of
  the datum's address; reading pools the same neighbourhood. Bricken & Pehlevan show that under
  conditions they verify in pretrained GPT-2, **transformer attention is SDM's read operation**, with
  the softmax temperature playing the role of the Hamming radius. The 2023 follow-up shows SDM's *fixed,
  sparse, random* addressing is **itself** a continual-learning mechanism: because addresses are random
  and coverage is uniform, two unrelated data write into nearly disjoint neighbourhoods without any
  coordination. **This is the exact theoretical statement of when our frozen-key regime is legitimate:**
  value-only writing into fixed keys works iff the fixed address set **covers** the query space, so that
  every new query has its own well-populated neighbourhood. Kanerva's usable capacity is only
  `≈ 0.1 N` data per N hard locations.
- Maps to our code: diagnostic and prescriptive. The coverage statistic is the *effective number of
  distinct neighbourhoods*, i.e. the routing entropy already computable from
  `core.py::compute_attention_weights`; the prescription is that if coverage fails, you do not fix it by
  writing values — you fix it by **adding an address**. Proposed opt-in probe flag
  `AM_PROBE_COVERAGE=1` in `cartridges/am_stability_probe.py`.
- **Prediction (about OUR signature):** SDM's coverage condition is **violated in our cartridge, and I
  can already quantify by how much.** Recomputed from `DIAG-ROUTING.npz`: routing entropy is 2.81 nats
  (QA) / 2.89 (MT) against `log 512 = 6.24`, i.e. **40–43 effective addresses out of 512 (8%)**, and the
  QA routing Gram has participation effective rank **1.97**. A store with ~40 effective addresses and a
  single address holding 52% of the mass is the *opposite* of Kanerva's uniform random coverage — it is
  a store with one hard location. Prediction: **any mechanism that raises effective coverage should move
  MT more per unit of QA cost than any mechanism that improves the fit at fixed coverage**, and the
  measurable proxy is routing entropy: I predict a monotone relation between post-write routing entropy
  and ΔMT across arms, and **no** relation between `am/mean_mse` and ΔMT (which DIAG-OBJ-c already
  confirmed to be *anti*-correlated). Falsifier: an arm that raises entropy from 2.9 to > 4.0 nats and
  still leaves MT ≥ 2.5.
- Cost: **gradient-free, `gradient_steps = 0`, no extra forward passes** — two extra reductions over the
  `alpha` matrix `compute_attention_weights` already materialises.
- Why it might NOT transfer: SDM's guarantees assume **binary** addresses and uniformly random hard
  locations; our keys are learned, real-valued, RoPE-phased and highly non-uniform. Kanerva's `0.1N`
  capacity is for autoassociative recall of the stored datum, not for a language model's CE.
- Related: LIT-016, LIT-022, LIT-024.

---
### LIT-024: The only published **closed-form, gradient-free key-write rules** — usage/LRU allocation (Kaiser et al., *Learning to Remember Rare Events*, arXiv 1703.03129, ICLR 2017 · NTM arXiv 1410.5401 · DNC, Graves et al., Nature 538:471–476, 2016)
- Status: candidate ⭐ *(the single most directly transferable mechanism in this block)*
- Board entry: **B-ROUTE**
- Mechanism (1 paragraph): Kaiser et al. maintain a memory `(K, V, A)` — keys, values, **ages** — beside
  a network, with all keys ℓ2-normalised and queries normalised. Given a query `q` with true value `v`
  and nearest neighbour `n₁`: **if `V[n₁] = v`, move the key toward the query and reset its age,
  `K[n₁] ← (q + K[n₁])/‖q + K[n₁]‖`, `A[n₁] ← 0`; otherwise pick the OLDEST slot,
  `n' = argmax_i (A[i] + r_i)` with `r_i` a tiny tie-break noise, and write `K[n'] ← q`, `V[n'] ← v`,
  `A[n'] ← 0`**; all other ages increment. Nothing here is learned by backprop — it is a rule.
  NTM/DNC give the same idea a differentiable form: DNC maintains a usage vector `u_t` and a free list
  `φ_t` (indices sorted by ascending usage) and writes with the **allocation weighting**
  `a_t[φ_t[j]] = (1 − u_t[φ_t[j]]) ∏_{i<j} u_t[φ_t[i]]`, which concentrates the write on the
  **least-used** location; NTM's write is `M_t(i) = M_{t−1}(i)[1 − w_t(i)e_t] + w_t(i)a_t`, an explicit
  erase-then-add. **The invariant across all three: you write the new key AT the query, and you place it
  where the old content is NOT being read.**
- Maps to our code: two independent, small changes, both of which invert the sign of what we do today.
  (i) **Where:** `cartridges/am/ranking.py::rank_am_slots` currently selects the slots with the
  **highest** new-content access score (TF). The LRU/DNC rule selects the slots with the **lowest old**
  usage. Proposed `SLOT_SELECTION=qa_least_used` using the QA access scores that
  `ENABLE_OLD_REFERENCE_GUARD=1` already collects (`continual.py:61-88`) — and note this is the
  *sharpened* form of the IDF idea that K-GATE refuted: IDF used corpus document-frequency, DNC uses the
  **actual old usage of that slot**, which is the quantity the theory names (LIT-013 made the same point
  for the margin selector; this entry supplies the published rule and the exact formula).
  (ii) **What:** `key_select.py` gains `KEY_MODE=query_centroid`: `k* = c·q̄_MT/‖q̄_MT‖` where `q̄_MT` is
  the mean (or top principal direction) of that head's MT reference queries — Kaiser's `K[n'] ← q`
  verbatim. Combined with LIT-020's `P⊥`, this is `k* = c·P⊥q̄_MT/‖P⊥q̄_MT‖`.
- **Prediction (about OUR signature):** this is the **only** mechanism reviewed that raises the MT/QA
  mass ratio *by construction on both factors at once* — the new key points at MT's query centroid
  (raising the MT logit) and sits on slots QA barely reads (limiting the QA logit). Concretely:
  `SLOT_SELECTION=qa_least_used` alone, values only, no key change, should push the MT/QA `mass_on_S`
  ratio **above 1.04** — I predict **1.3–3.0** — while *lowering* the absolute `mass_on_S` (least-used
  slots carry less mass), so MT may not improve and **QA should improve toward or below the 2.2388
  floor**. Adding `KEY_MODE=query_centroid` on those slots should then raise `mass_on_S` back to ≥ 0.3
  **at the elevated ratio**, which is the combination the win condition needs (MT ≤ 2.02, QA ≤ 2.52).
  **Sharp falsifier:** if `qa_least_used` leaves the ratio at ≈ 1.04, then no slot subset of this
  cartridge is QA-blind — consistent with the 0.914 top-32 overlap — and the discrete lever is dead,
  leaving only synthesis (LIT-019/020).
- Cost: **gradient-free, `gradient_steps = 0`, no extra solves.** `qa_least_used` reuses an access-score
  vector already accumulated; `query_centroid` is one mean per (layer, head) per document over queries
  already in memory. Strictly cheaper than `highest_attention` (which runs a 512-way softmax plus a
  200-iteration NNLS, `key_select.py:96`).
- Why it might NOT transfer: Kaiser's memory sits **beside** a network that is trained end-to-end to
  produce queries the memory can serve; ours must work with a frozen LM whose queries we cannot shape,
  and whose keys we cannot ℓ2-normalise without changing the logit scale the model expects. DNC's usage
  is a *within-episode* counter with a free gate; our "usage" is a cross-task statistic estimated from a
  reference set. And "write at the query" assumes the query is stationary — ours are RoPE-phased
  (LIT-026) and drift as soon as earlier layers are written (B-CASCADE).
- Related: LIT-013, LIT-019, LIT-020, LIT-021, K-GATE (HYP-G1), B-ROUTE.

---
### LIT-025: Key-usage imbalance is the *known* failure mode of writable-key memories, and the published fix is **query-side** (Lample et al., *Large Memory Layers with Product Keys*, arXiv 1907.05242 · arXiv 2010.03881 · *Memory Layers at Scale*, arXiv 2412.09764 · `TF-IDF.pdf` = arXiv 2510.15103)
- Status: candidate
- Board entry: **B-ROUTE** (primary), K-GATE (secondary)
- Mechanism (1 paragraph): product-key memory is a fixed table of `N` learned key–value pairs read by
  `I = TopK(Kq(x), k)`, `s = softmax(K_I q(x))`, `y = sV_I` — architecturally our object with `N = 512`,
  `k = t`. Its documented pathology is **catastrophic key-usage imbalance**: without intervention only a
  minority of slots are ever accessed. The fix that works is **not** on the keys — it is **batch
  normalisation on the query vectors**, which raises memory usage from **25.8% to 80.3%** at `N = 1M`
  and drops perplexity 19.8 → 18.0; dead keys are additionally re-initialised with noise. The field's
  standard diagnostic is the **KL divergence between the slot-access distribution and uniform**.
  `TF-IDF.pdf` (our gating ancestor) sits on exactly this architecture and updates **values only** —
  its keys are frozen throughout — selecting slots by TF-IDF of access counts against a background
  corpus.
- Maps to our code: (a) the diagnostic — log `KL(access_distribution ‖ uniform)` and top-1 share
  alongside `mass_on_S` in `ranking.py::rank_am_slots` / `am_stability_probe.py`; (b) the query-side fix
  has **no gradient-free analogue for a frozen LM** (we cannot insert a BatchNorm), which is itself the
  finding: *the one intervention that reliably fixes usage imbalance in this architecture is unavailable
  to us*, so imbalance must be attacked on the key side instead (LIT-020/LIT-024) or accepted.
- **Prediction (about OUR signature) — and the quantitative reason TF-IDF.pdf's premise does not hold
  here.** Their store is `N = 10⁶–10⁸` with `k = 32`, a selection density of **3×10⁻⁵**; ours is
  `N = 512` with `k = 32`, density **6×10⁻²** — **2,000× denser**. Two random 32-subsets of 10⁶ slots
  overlap with probability ~10⁻³; two random 32-subsets of 512 overlap in ~2 slots (6.25%) by chance
  alone. **We measure a top-32 QA/MT overlap of 0.914 — 15× the chance level and 3 orders of magnitude
  above the regime TF-IDF.pdf was designed for.** Prediction: any selection-only mechanism (TF, TF-IDF,
  margin, least-used) can move the MT/QA mass ratio by at most the amount the ~9% of non-overlapping
  slots allows, i.e. **ratio ≤ ~1.2 from selection alone**; getting past that requires changing what
  the slots *are* (keys), not which ones are picked. Falsifier: a selection-only arm that reaches
  ratio > 1.3.
- Cost: **zero** for the diagnostic; the transfer analysis is free.
- Why it might NOT transfer: product-key memories are trained end-to-end with the keys in the loss, so
  "usage" is something the optimiser can fix; our keys are a fixed post-hoc artefact. And their memory
  replaces an FFN (position-independent), while ours lives inside attention with RoPE (LIT-026).
- Related: LIT-013, LIT-024, K-GATE (HYP-G1), TF-IDF.pdf.

---
### LIT-026: RoPE hazards for key synthesis and key relocation — and **three concrete, unreported hazards in our own code** (RoFormer arXiv 2104.09864 · StreamingLLM arXiv 2309.17453 · `AM.pdf` App. C.3 · DapQ arXiv 2603.11564 · *Round and Round We Go* arXiv 2410.06205 · KVMerger arXiv 2407.08454)
- Status: candidate ⭐ *(blocking: it gates every other key-side entry)*
- Board entry: **B-ROUTE** (primary), **B-TARGET** (secondary)
- Mechanism (1 paragraph): **a synthesised key is well-defined only as a (direction, position) pair.**
  RoPE stores `R_p k`; the logit against a query at position `n` is `qᵀR_{n−p}k`, so a stored key means
  nothing until you say where it lives. The literature's answer to relocation is always the same
  operator: a **uniform phase shift** `R_Δ`, `Δ = p_target − p_source`. `AM.pdf` App. C.3 does exactly
  this for text-based chunking ("applies a uniform RoPE phase shift to the compacted keys to align them
  to the chunk's original global offset, i.e. a rotation by `Δ = p_global − p_local`"), and §2 notes the
  compacted cache **retains a logical length T** so appended tokens keep the position IDs they would
  have had. StreamingLLM's central operational rule is the same: assign positions by **cache index, not
  original text index**. DapQ (2603.11564) supplies the quantitative penalty for getting it wrong:
  synthetic "pseudo queries" must be given the **future** positions they will actually occupy, and
  query similarity to real queries **decays monotonically with the absolute position offset** — the
  semantic content of the pseudo token barely matters, the position does. *Round and Round We Go*
  (2410.06205) explains the structure: the low-index (high-frequency) RoPE bands carry positional
  selectivity, the high-index (low-frequency) bands carry semantic content — so a phase error is not a
  uniform degradation, it corrupts the positional bands first. KVMerger (2407.08454) merges keys only
  within *adjacent* token sets, where the phase mismatch is small, precisely to dodge this.
- Maps to our code — **three hazards, found by reading, none previously reported, all no-GPU checkable:**
  1. **`rope_theta` is hard-coded to `10000.0` in every AM function and no caller ever overrides it**
     (`core.py:24/51/83/122`, `teacher.py:142/164/187`, `key_select.py:59/110/236`, `phase1.py:47/264`;
     `grep 'rope_theta='` finds no non-default call site). **Qwen3-4B-Instruct-2507 uses
     `rope_theta = 5,000,000`** (verified in the HF snapshot `config.json`). This is live in the
     **teacher-target construction of every AM run**: `finetune.py:495` sets
     `doc_rope_offset = k_doc.shape[0]`, which `_attention_scores` (`core.py:66`) turns into
     `_apply_rope_offset_to_queries(..., rope_theta=10000.0)`. Quantifying it: for an isotropic
     128-dim query, `E[cos(rot_{θ=1e4}(q), rot_{θ=5e6}(q))]` = **0.29 / 0.34 / 0.27 / 0.24 / 0.00** at
     offsets 256 / 512 / 1024 / 2048 / 4096, and at offset 1024 **75% of dimension pairs are rotated by
     more than 1 radian in error**. The measured document lengths in this project are
     **T_doc = 3858–8900 tokens** (`logs/phase1_*.log`, `doc_token_counts`), i.e. **the offsets where the
     two thetas are fully decorrelated.**
  2. **`rewrite_keys_on_support` selects doc keys in one rotary frame and installs them in another with
     no counter-rotation.** `key_select.py:258-278` scores candidates with `doc_key_start=t`,
     `doc_rope_offset=T_doc` (so the query is rotated forward by T_doc for the doc block), then
     `:283` does `out[selected_indices] = new_k` — the **raw** doc key, unrotated, into a cartridge slot.
     `phase1.py::_rope_reposition` (`:42-68`) implements exactly the missing `R_Δ`, and
     `initial_am_compaction.py:150` already uses it via `rebake_key_positions`. **So the fix exists in
     this repo and the `KEY_MODE ≠ freeze` path does not call it.**
  3. **Frame bookkeeping is inconsistent end-to-end.** Cartridge keys are baked at positions 0…511
     (`initialization/text.py:45`, `arange`); document keys are baked at positions 0…T_doc−1
     (`teacher.py::prefill_document_kv_cache` is called at `continual.py:162` with the default
     `position_offset=0` — the Phase-1 logs' own `rope_note` says "Per-doc prefill at position 0; docs
     concatenated so absolute RoPE positions **overlap**"); and eval/reference queries are at
     `arange(seq_len)` from 0 (`utils/chat.py:70`, `generation.py:228`). In that convention doc keys and
     cartridge keys are **already in a common frame**, yet `finetune.py:495` applies a T_doc query
     rotation anyway. Proposed opt-in flags: `AM_ROPE_THETA` (default 10000.0 ⇒ bit-identical),
     `AM_DOC_ROPE_OFFSET ∈ {t_doc, zero}`, `AM_KEY_REBAKE=1` (call `_rope_reposition` inside
     `rewrite_keys_on_support`).
- **Prediction (about OUR signature):** (i) Hazard 2 means **the first-ever `KEY_MODE≠freeze` run is
  measuring a phase-corrupted key, not a key-side mechanism** — I predict `KEY_MODE=highest_attention`
  as it stands produces `mass_on_S` on MT **at or below** the frozen-key 0.0896 with the ratio still
  ≈1.0, because a key rotated by thousands of positions is near-orthogonal to the query that selected
  it; the run would look like "keys don't help" for a reason that has nothing to do with keys.
  (ii) Hazards 1+3 predict a *target* defect independent of keys: the doc block's contribution to
  `compute_teacher_targets` (`finetune.py:500`) is computed at a scrambled relative phase, so the
  teacher output we fit is **not** the output the model would produce for `[cartridge ‖ doc]`. That is a
  candidate mechanistic account for why ORACLE-WRITE's *perfect content transplant* (teacher's own doc
  **values**, which are RoPE-free and therefore unaffected) beat the solve by 0.17 while the solve —
  which fits RoPE-corrupted **targets** — has been pinned at 2.54 through every knob. Falsifiable and
  cheap: rerun the canonical config with `AM_ROPE_THETA=5e6` and, separately, `AM_DOC_ROPE_OFFSET=zero`,
  one variable each; if either moves MT by more than the 0.1–0.2 noise band, a substantial fraction of
  this loop's "AM cannot acquire" evidence was measured against a mis-specified target.
- Cost: **gradient-free, `gradient_steps = 0`, zero extra passes.** `_rope_reposition` is one elementwise
  rotation of a `t×128` block; passing the true theta is a constant.
- Why it might NOT transfer: the position convention in this repo is deliberately non-standard (the
  cartridge overlaps the query positions), so "the correct frame" is a design choice, not a fact — it is
  possible the current offset is intentional and only θ is wrong, or vice versa. Also, θ affects only
  the *offset* paths; with `doc_rope_offset=0` (frozen-key runs on cartridge-only keys, e.g.
  `refit_beta_nnls:189`, which uses a raw `queries @ keys.T`) it is a no-op — so this hazard bounds the
  **teacher target** and the **key-rewrite** paths, not the value solve's design matrix.
- Related: LIT-004, LIT-019, LIT-020, LIT-024, B-TARGET, B-ROUTE, ORACLE-WRITE.

---
### LIT-027: The counter-case — why a key write may be *strictly worse*, and what "keys collapse QA" is actually made of
- Status: candidate *(falsifier for LIT-019/020/021/024; also an operational blocker report)*
- Board entry: **B-ROUTE** (as the negative control)
- Mechanism (1 paragraph): three named reasons a key-side write can lose where a value-side write does
  not, each with a code location. **(a) Eviction, not addition.** `rewrite_keys_on_support:283` does
  `out[selected_indices] = new_k` — every key we take is a key **removed**, and the value that slot held
  is simultaneously re-solved, so the slot's entire QA contribution is destroyed regardless of where the
  new key points. Values-only writes destroy the value but keep the routing; key writes destroy both.
  **(b) The candidate pool guarantees eviction of the best slots.** The pool is
  `cat([original_keys[S], k_doc])` (`finetune.py:519-522`) and `S` is chosen by highest MT access mass —
  which, at 0.914 QA/MT top-32 overlap, is also the highest **QA** mass. So the slots offered up for key
  replacement are precisely the ones QA reads most, including (LIT-021) the layer's single dominant slot
  in the 29/36 layers where all 8 heads share it. **(c) The write moves the softmax the value solve was
  fitted against.** `finetune.py:500` computes the teacher targets, `:516` rewrites the keys, `:591`
  solves values on the *new* keys — internally consistent — but B-CASCADE showed that perturbing layer
  `ℓ` moves the queries at layers `> ℓ`, and a key rewrite perturbs the *routing* at layer `ℓ`, which is
  a strictly larger perturbation than a value rewrite of the same magnitude.
  **The verdict on the folklore.** The pre-loop claim "moving keys collapses QA" is **untested in this
  setting**: every row in `results.csv` is `KEY_MODE=freeze`, the claim comes from a different model and
  a pre-loop session, and — per (a)/(b)/LIT-026 — the mechanism it would have measured is *eviction of
  the highest-mass slots plus a rotary phase error*, not key geometry. It is folklore **about a
  confound**, not evidence about keys.
- Maps to our code: **no new code — this entry is a set of controls.** Arm 1:
  `KEY_MODE=highest_attention` unchanged (the folklore reproduction). Arm 2: same, with the top-1 mass
  slot per layer excluded from `S` (LIT-021's `AM_KEY_PROTECT_TOPK=1`). Arm 3: same, with
  `AM_KEY_REBAKE=1` (LIT-026). The three arms attribute any QA collapse to eviction / sink destruction /
  phase error respectively.
- **🔴 OPERATIONAL BLOCKER (verified by reading, affects the cheapest experiment on the board):**
  `finetune.py:264-269` `_should_fit_beta` returns `config.key_mode != "freeze"` when `enable_beta` is
  `None`, and `continual_am_sparse.py:104-109` maps an **unset** `ENABLE_BETA` to `None`. **Therefore
  `KEY_MODE=highest_attention` with `ENABLE_BETA` unset silently switches on the β/NNLS path that
  crashed EXP-005 (cholesky not-PD), EXP-005b (NaN in ridge lstsq) and EXP-006 (NaN generated *inside*
  `refit_beta_nnls`, so the output clamp could not fix it).** A key-only run is nevertheless available
  **with no code change**: `ENABLE_BETA=0` forces `_should_fit_beta → False` at `:267-268` while leaving
  `key_mode` free. Without that env var, the first key experiment will fail for a B-SOLVE reason and be
  mis-attributed to keys.
- **Prediction (about OUR signature):** if (a)+(b) dominate, all three arms lose QA by a similar amount
  (QA > 2.7, past the 2.52 budget) and the MT/QA `mass_on_S` ratio stays ≈ 1.04 — keys bought bandwidth
  and paid for it symmetrically, exactly like β, and **B-ROUTE closes `confirmed+capped`**. If arm 2 or
  arm 3 recovers QA to ≤ 2.5 at the same MT, the collapse was a confound and the key axis is live. If
  arm 1 alone already gives MT ≤ 2.3 at QA ≤ 2.52, the folklore is simply false.
- Cost: **gradient-free, `gradient_steps = 0`**; `highest_attention` is AM's cheapest selector (3 s for a
  60k context on 64 heads, Table 3) — but note our `key_select.py:96` runs a **200-iteration** NNLS per
  head per document versus the paper's 2 (LIT-002), so budget the arm's `solve_s` against the 165–185 s
  baseline rather than assuming parity.
- Why it might NOT transfer: this entry is about *our* code, so the transfer risk is inverted — the risk
  is that (a)/(b)/(c) are individually real but jointly small, and the true cap is the one DIAG-ROUTING
  measured (queries genuinely inseparable), in which case none of the controls matter.
- Related: LIT-002 (β is broken), LIT-004, LIT-020, LIT-021, LIT-026, B-SOLVE, B-CASCADE, B-ROUTE.
