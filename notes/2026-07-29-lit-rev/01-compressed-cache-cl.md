# LITREV-01 — Continual learning on COMPRESSED CACHES / fixed-size KV memory

**Worker:** W2 SCOUT (literature) · **ID:** LITREV-01 · **Date:** 2026-07-29 · **No GPU, no source edits.**
**Grounding:** `../2026-07-29-am-investigation-synthesis.md`, `../../research_loop/GLOSSARY.md`,
`../../research_loop/state/bottleneck_board.md`, `../../research_loop/state/literature_ledger.md` (LIT-001…027).
**Papers read at method level: 19** (each entry states which sections/equations were read).

**Our facts, restated so every prediction below is anchored to them.** Best point **QA 1.9468 / MT 2.2681**;
target QA ≤ 2.52 / MT ≤ 2.02; retention **0.573 ahead**, acquisition **0.248 short**; paired resolution
**±0.049 MT / ±0.054 QA**; `log(MT routing mass)` → MT loss Pearson **−0.877 / −0.9775 / −0.986**;
perfect value write ceiling **MT 2.381**; QA/MT routing cos **0.99899**, top-32 overlap **0.914**;
k-curve minimum at **k = 12 of 16**, then strictly worse; solo write into uncontested slots **+1.101 worse**;
content-free controls reproduce **28–73%** of the MT gain (full-context ICL: **53.3%**).

**One number I computed for this review, because every paper below is priced in it.**
Measured from `data/qasper/train/qwen_qasper_{QA,MT}_task_8192.parquet` (16 unique `<title>` per split,
longest chunk per title, chars ÷ 3.8):

| corpus | documents | ≈ tokens | slots | compaction |
|---|---|---|---|---|
| Phase-1 (QA papers) | 16 | **123,094** | 512 | **240×** |
| Phase-2 (MT papers) | 16 | **96,334** | +0 | — |
| **both, after Phase 2** | **32** | **≈219,400** | **512** | **≈428×** |
| per-document budget after Phase 2 | | | **16 slots/doc** | |

Hold that 428× and that 16 slots/document. They are the single most load-bearing facts in this review.

---

## Executive summary — what the field does that we do not (10 bullets)

1. **Nobody in this literature writes a second corpus into a fixed compressed cache. Composition is by
   CONCATENATION, universally.** Cartridges (§5.4), ICAE (Tab. 8), AutoCompressor (σ_{<i} = Concat(σ_1…σ_{i−1})),
   Activation Beacon ("condensed activations are accumulated"), C²KV, KBLaM, InfLLM, InduceKV — every single
   one adds capacity when it adds content. The one operation we perform (overwrite slot j's value with new-domain
   content while keeping the slot count fixed) **has no published instance anywhere I could find**.
2. **The field's own answer to "more documents into fixed memory" is a measured failure.** Cartridges at Scale
   (2606.04557) reports a monolithic cartridge collapsing where per-document cartridges do not: LongHealth
   **72.6 → 80.1**, QuALITY **65.2 → 78.6**, FinQA **14.1 → 62.7** at matched token budget, and
   **73.6% oracle → 26.0%** when 20 independently-trained cartridges are loaded together. Their per-document
   budget is **≈1.2K tokens/document**. Ours is **16 slots/document** — **75× less**.
3. **The AM paper itself states our result before we measured it.** §6: AM beats Cartridges at **50×**
   compaction but Cartridges wins at **100×** "on some benchmarks", attributed to gradient optimization
   "not [being] restricted to optimizing intermediate attention outputs or selecting keys from the original
   cache … which becomes increasingly important as the compaction budget shrinks." **We operate at 428×.**
   Our 0.248 acquisition shortfall is the AM authors' own predicted regime, in their own words.
4. **The one controlled gradient-free-vs-gradient comparison in the literature says the write is the problem,
   not the selection.** GradMem (2603.13875), same architecture, same **8 memory vectors**: forward-only write
   **19.3–45.5%**, one gradient step **58.6–96.3%**, five steps **99.1–100%**. And explicitly: *repeating the
   forward-only write yields weak or inconsistent improvements* — which is our k=12→16 tail, verbatim.
5. **Fixed-capacity memories that DO accept gradient-free writes report ≈1 item per slot, not ≈400 tokens per
   slot.** Larimar (512×768 memory, closed-form pseudo-inverse write): ~100% rewrite accuracy to **512 edits**,
   **82% at 1024**. MemoryLLM (7,680 slots/layer, gradient-free random-drop self-update): retention horizon
   **<20k tokens** ⇒ **2.6 tokens/slot**. We are asking **428 tokens/slot**, i.e. **~165× past MemoryLLM's
   own measured frontier** for a gradient-free write.
6. **Streaming KV compression never writes — it evicts, and what survives is *original* KV.** H2O, SnapKV,
   PyramidKV, Scissorhands, StreamingLLM, InfLLM all keep a subset of unmodified cache rows. InfLLM's memory
   is *never modified*, only appended and retrieved (offloaded to CPU, LRU on GPU). Our "gating" family and
   theirs are **not the same problem**: they choose what to *drop*, we choose what to *destroy and replace*.
7. **The field's mass control goes the opposite way from ours.** KBLaM adds `log C − log M` to every knowledge-
   token logit specifically to *cap* the memory block's share of the softmax as M grows. Our β raised
   `mass_on_S` **4.23×** and made **both** axes worse. Two independent architectures agree that an unbounded
   query-independent mass lever on a memory block is harmful; the AM paper needs β only because its compacted
   block is the *whole* context, which ours is not.
8. **Where "amortize the write" has been made to work, it is a trained encoder, not a solve.** xRAG (178× to one
   token, frozen LLM, gradient-free at write time), Still (2606.07878, per-layer Perceiver, RoPE stripped
   before compaction and re-applied after — independently confirming our MECH-005), ICAE, Activation Beacon.
   All pay a **one-time training cost on the compaction operator**, never a per-document optimization. This is
   the only family that is simultaneously gradient-free-per-document and competitive at high compaction.
9. **"Selection methods are subset-bound" is now the field's stated diagnosis of exactly our operator.**
   Still: *"a single Still slot can blend information from arbitrarily many positions via cross-attention"*,
   which selection cannot. We are **more** constrained than the methods this critique targets: our `C_k` is not
   even a subset of the document being written, it is a **foreign frozen Phase-1 key**.
10. **Our k≈12/16 saturation is not anomalous — it is early only relative to token budget, and it is exactly
    what the two published repeated-compression curves look like.** Still's iterative compaction collapses from
    **51.0% @32k to 1.5% @128k — below the no-context floor** once past its training horizon; L2P/DualPrompt's
    fixed prompt pool degrades to **<50% selection accuracy by task 10**, with *"prompts repeatedly selected
    across different tasks … prone to being overwritten."* A non-monotone curve with a mid-sequence optimum is
    the **normal** signature of a fixed-capacity memory written repeatedly. What would be anomalous is its
    absence.

**Directions I call dead on arrival, with the reason.** (a) *Any reallocation of the 512 slots* — §L01-2 and
§L01-12 price the deficit at 75–165× budget, and no permutation of a 16-slot-per-document budget closes a 75×
gap. (b) *Any mass-shaping mechanism* — three independent literatures (β/AM, KBLaM, our own dose-response at
Pearson −0.9775) agree mass is a scalar the incumbent already maximises. (c) *Any importance/redundancy gate* —
the streaming-eviction family it is borrowed from operates on *unmodified original KV* and therefore has no
analogue of our destruction cost; the transplant was already refuted at 2.2–8.1× resolution.

---

## Entries

### L01-1: Trainable fixed-size KV cache + context distillation, composed by CONCATENATION — *Cartridges: Lightweight and general-purpose long context representations via self-study* (arXiv 2506.06266, NeurIPS 2025 · https://arxiv.org/abs/2506.06266)
*Read: §3.1 desiderata, §3.2 parameterization, Alg. 1, §4.2 Eq. (3), §5.1–5.4, Fig. 5–7, App. D.4.*
- **Claim / mechanism.** `Z ∈ R^{L×p×d×2}` is a simplified prefix-tuning KV cache of `p` trainable
  (k, v) pairs, substituted for the corpus' KV; the LM is frozen and gradients flow only into `Z`. Training
  is context distillation over self-generated conversations:
  `argmin_Z Σ_(x,c̃)∈D Σ_i D_KL( F(·| c̃ ⊕ x[:i]) ‖ F_Z(·| x[:i]) )`, with `c̃` a 512–4096-token chunk and 5
  generic seed prompt types. Composition is an explicit *design desideratum* (§3.1: "Given `Z1` and `Z2` …
  ideally `F_[Z1,Z2](q) ≈ F(·|C1 ⊕ C2 ⊕ q)`") and is realised by **concatenating the two caches**, never by
  merging them into one. §5.4: pairs of independently trained 10-K cartridges at sizes {512, 1024, 2048, 4096},
  concatenated with **no joint training**, beat both a single cartridge and truncated ICL on multi-document
  questions.
- **Maps to our code.** This *is* our Phase-1 object: `cartridges/am/phase1.py`, `cartridges/cache.py`
  (`TrainableCache`), `examples/qasper2/train/*`. The composition path does not exist in our code; it would be
  a new eval-time flag `AM_COMPOSE_CACHES=<path1>,<path2>` reading two caches into
  `examples/qasper2/train/eval_forgetting.py` and concatenating along the slot axis before attention.
- **Prediction about OUR signal.** App. D.4 is **our exact corpus**: "we concatenate 16 papers all related to
  QA NLP models … 78 questions" — the same 16 papers and the same 78 QA examples as
  `data/qasper/eval/qasper_eval_QA.parquet`. Their reported QASPER frontier is **up to 100× memory saving vs
  ICL at parity**. Our Phase-1 512 slots over 123k tokens is already **240×**, i.e. past their measured parity
  point; after Phase 2 we demand **428×**. Concretely I predict a **512-slot Phase-1 + 512-slot MT-only
  cartridge, concatenated (1024 slots total, no joint training)** lands **MT ≤ 2.05 and QA ≤ 2.30** — i.e. it
  passes the mission's numeric target on both axes while our monolithic write misses MT by 0.248. If that
  composition *fails* to reach MT ≤ 2.15, the bottleneck is not capacity and this whole review's thesis is wrong.
- **Cost.** Composition itself is **free and gradient-free** at eval. Building the second cartridge is
  gradient-based unless AM writes it (see L01-3). Extra state: **2× KV memory**, +512 slots of eval-time attention.
- **Why it might NOT transfer.** It violates the mission's fixed-512-slot constraint by construction — this is a
  **capacity oracle**, not a submission. Also, their composition pairs were *topically disjoint* 10-Ks; our QA/MT
  routing cosine is **0.99899**, so the two caches will compete for the same queries far more than AMD-vs-Pepsi do.
- **Verdict:** **candidate — as the decisive capacity diagnostic (Experiment 2).** Unusable as a compliant mechanism.

---

### L01-2: Per-document cartridges + distractor mixing + retrieval — *Cartridges at Scale: Training Modular KV Caches over Large Document Collections* (arXiv 2606.04557 · https://arxiv.org/abs/2606.04557)
*Read: abstract, Fig. 1, Eq. (1) mixed-visibility training, budget manager, Tab. 3–4.*
- **Claim / mechanism.** The direct descendant of L01-1, and it exists **because monolithic cartridges fail**.
  Two failure modes measured: (i) *naive* mixing — 20 cartridges each trained in isolation, loaded together,
  drop from **73.6% oracle to 26.0%** (near chance); (ii) *monolithic* — a single 28.7K-token cartridge over
  the whole collection scores **72.6 / 65.2 / 14.1** (LongHealth / QuALITY / FinQA) vs per-document cartridges
  at matched budget **80.1 / 78.6 / 62.7**. CAS fixes (i) with **probabilistic distractor mixing** (each
  cartridge is trained while irrelevant neighbours are in the prefix, so the frozen model learns selective
  attention), and fixes memory with a **budget manager** rotating `B` active cartridges GPU↔disk every `R`
  steps with fraction `φ` replaced. Inference = retrieve document IDs, **concatenate** the selected cartridges.
- **Maps to our code.** `cartridges/am/continual.py::run_per_document_am_phase2` already loops per document but
  writes into **one shared cache**; CAS's structure is the same loop emitting **one cache per document**. Flag:
  `AM_PER_DOC_CACHES=1` (write to `cache-doc-XXX.pt` instead of mutating the live cache) plus the
  `AM_COMPOSE_CACHES` eval path of L01-1. Distractor mixing maps onto
  `finetune.py::_collect_reference_queries` — draw reference queries with *other* documents' cartridges present.
- **Prediction about OUR signal.** The **75× budget gap** is the headline: CAS spends ≈1.2K tokens/document,
  we spend **16 slots/document** post-Phase-2 (32 pre-). Their FinQA row (14.1 vs 62.7) is a monolithic cache at
  *near-full* token budget still collapsing — so budget alone does not save monolithic packing; **contention
  does the damage**. But note this **contradicts DIAG-PERDOC**, which measured a solo write into uncontested
  slots as **+1.101 WORSE** on its own document. Reconciliation and its prediction: CAS's per-document
  cartridges are *gradient-trained on their own document*, so they carry content; our solo AM write carries
  almost none (`ref_mass_on_S` 0.1477–0.1535, solve MSE 0.0085–0.205, yet only 15.5% of the 16-doc gain). I
  therefore predict **per-document AM caches composed at eval will NOT reproduce CAS's +7.5 to +48.6 points** —
  concretely, composed 16 solo AM caches will land **MT ≥ 3.4** (vs solo pooled 3.7488, k=16 2.6186), because
  16 caches that each carry 15.5% of a gain do not sum. **This is a discriminator between "capacity" and
  "the write carries nothing": L01-1's oracle uses a gradient-trained second cartridge, L01-2's uses AM ones.**
- **Cost.** Gradient-based end to end ("all updates are gradient-based through the frozen model's KL loss");
  the *composition* is free. Extra state: 16× caches.
- **Why it might NOT transfer.** Their retrieval step picks 1–few cartridges per query; we must serve QA and MT
  from one prefix with **top-32 overlap 0.914**, so retrieval has almost nothing to separate on
  (`ρ_key` at its held-out control floor, query separation 2.5× below sampling noise).
- **Verdict:** **candidate — as the negative control for Experiment 2**, and as the strongest external evidence
  that the mission's monolithic 512-slot framing is the binding constraint rather than the ranker.

---

### L01-3: Repeated/online compaction re-compacts the WHOLE cache, never a frozen sub-block — *Fast KV Compaction via Attention Matching* (arXiv 2602.16284v2, ICML 2026 · local `AM.pdf`)
*Read: §3.4 nonuniform compaction, §4.2 Tab. 1–2, §4.4 Tab. 3, §6 Discussion + Future work, App. F.3 Tab. 9, App. F.4.*
- **Claim / mechanism.** The paper's only continual story is **App. F.3 online compaction**: mid-trajectory,
  when the physical cache hits budget, it **compacts the entire context except the last 20 tokens** — including
  *previously compacted portions* — by 50%, up to 6 times. AIME 2025 with Qwen3-4B: physical 2048 / effective
  8192 scores **13/30**, matching uncompacted decoding at 8192 (13/30). The paper is explicit that it does
  **not** freeze previously compacted portions: *"Our study … always compacts the entire context, rather than,
  for example, freezing previously compacted portions or selectively compacting recent tokens."* §6 Future work
  lists our problem as **open**: *"architectures or training procedures that support compaction as a simple
  primitive, or explicitly operate over a fixed set of keys and values."*
- **Maps to our code.** `cartridges/am/continual.py::run_per_document_am_phase2` (the per-document loop),
  `cartridges/am/finetune.py::apply_document_am_write_to_cache`, `cartridges/am/compaction.py::compact_kv_head`.
  The paper's actual continual primitive would be a new mode
  `AM_CONTINUAL_MODE=recompact_all` — at document *d*, form `[cartridge ‖ doc_d]` (512 + T_doc keys) and
  **re-compact the union back to 512**, refitting `C_k`, `β`, `C_v` over *all* 512 slots, rather than solving
  values on a top-32 subset with frozen Phase-1 keys.
- **Prediction about OUR signal.** Two numbers. (a) The paper's own regime statement (§6) — AM > Cartridges at
  50×, Cartridges > AM at 100× — combined with our measured **428×** predicts a gradient-free AM shortfall of the
  order we see; at 428× I predict **no** AM configuration reaches MT ≤ 2.02, which matches "every in-scope family
  closed". (b) `recompact_all` should raise `mass_on_S` toward 1.0 (the whole cache is the written support, vs
  our 0.2799 realised MT routing mass); by the mass law (Pearson −0.877/−0.9775/−0.986, slope from the
  MECH-CONSTRAINED dose-response: log-mass 0.2799→0.1000 ⇒ MT 2.2720→2.4388, i.e. **ΔMT ≈ −0.166 per unit
  Δlog(mass)**) a move 0.2799 → ~0.90 is Δlog ≈ +1.168 ⇒ predicted **ΔMT ≈ −0.19**, landing MT ≈ **2.08**.
  That is the single largest predicted acquisition move in this review, and it is the paper's own procedure.
  **Falsifier:** if `recompact_all` moves MT by < 0.049 despite `mass_on_S` > 0.8, the mass law is not causal and
  the ceiling is the value write (already bounded at MT 2.381 by ORACLE-WRITE).
- **Cost.** Gradient-free. Cost is the *teacher prefill of the union* per document plus a **full 512-column**
  ridge solve per (layer, head) instead of a 32-column one — the design matrix goes 64×32 → 64×512, i.e.
  massively **underdetermined** at `MAX_QUERIES_PER_HEAD=64`, so this **requires** raising the query cap
  (LIT-003's finding; the paper uses 16k–50k queries/head). Realistically ~10–50× current `solve_s`.
- **Why it might NOT transfer.** Re-compacting all 512 slots destroys the Phase-1 keys that currently *are* the
  retention mechanism — QA would be re-derived from the compacted union rather than preserved. Given QA is
  **0.573 ahead of budget**, that is affordable; given `ORACLE-WRITE-512` already showed that full-support
  writes make 16 documents **mutually annihilate**, it is also the known failure mode. The difference is that
  `recompact_all` re-derives keys *and* β *and* values jointly from a real teacher, which ORACLE-WRITE-512 did not.
- **Verdict:** **candidate — the highest-value in-scope mechanism left**, and the only one the source paper
  actually endorses for the continual case. Note explicitly: this is *not* the same as `ORACLE-WRITE-512`.

---

### L01-4: Amortized compaction by a trained per-layer Perceiver — *Still: Amortized KV Cache Compaction in a Single Forward Pass* (arXiv 2606.07878 · https://arxiv.org/abs/2606.07878)
*Read: abstract, architecture (latent queries, RoPE handling), objective, iterative-compaction results, comparison table vs AM.*
- **Claim / mechanism.** Per-layer latent bank `Z ∈ R^{H×t×d_ℓ}` cross-attends the **full** per-layer KV cache
  (**keys inverse-rotated to strip RoPE before compaction, re-rotated at chosen output positions after**),
  refines with latent self-attention + FFN, and emits compact `(K̃, Ṽ)` in **one forward pass**. Trained once per
  model with **forward KL from a full-context teacher to the compact-cache student, masked to answer tokens**;
  base model frozen, compactor is the only trainable object. Their framing of the field is the sharpest one
  available: *"selection methods are lightweight but subset-bound, while synthesis methods are expressive but
  rely on per-context optimization"*; Still *"breaks the selection constraint — a single Still slot can blend
  information from arbitrarily many positions via cross-attention."* Iterative chunked compaction is supported
  ("compaction is a forward pass, Still can be applied iteratively"), and its failure is documented: the
  **8k-trained compactor reaches 51.0% @32k and collapses to 1.5% @128k — below the no-context floor**;
  the 32k-trained one holds 39.2% @128k. *"The binding constraint is the training horizon."*
- **Maps to our code.** Replaces `cartridges/am/value_solve.py::sparse_am_value_update` and
  `key_select.py::select_keys_highest_attention` wholesale with a learned module. Flag: `AM_WRITE_OP=perceiver`
  with weights at `AM_COMPACTOR_PATH`. The RoPE strip/re-apply is **already ours**: MECH-005
  (`AM_KEY_REPOSITION=1` + `AM_ROPE_THETA=5000000`, `key_select.py`), which buys ΔMT **−0.1545**.
- **Prediction about OUR signal.** (a) **Independent confirmation of MECH-005.** Still built the same
  inverse-rotate/re-rotate step from first principles; our unit check (repositioned logit error **4.8e-07** vs
  **1.054** uncorrected) is the same phenomenon. Expect MECH-005 to hold under any further perturbation —
  it is not a fluke. (b) **The subset-bound critique prices our ceiling.** ORACLE-WRITE gives **MT 2.381** with
  the teacher's *own* document KV in the selected slots — that is the best a subset/value-only operator can do.
  Still's claim is that a synthesis operator exceeds it because one slot can mix many positions. Predicted:
  a trained compactor over our `[cartridge ‖ doc]` union reaches **MT < 2.38** (i.e. beats our value-write
  ceiling), which no in-scope AM configuration can. (c) **Their iterative collapse is our k-curve.** 51.0% →
  1.5% past the training horizon is qualitatively our 2.435 (k=12) → 2.552 (k=16); this makes our tail regression
  **expected**, not a bug.
- **Cost.** **Gradient-free per document** (one forward pass), but requires a **one-time training run** of the
  compactor against the frozen model — outside the mission's `gradient_steps = 0` only if you count steps on the
  *cartridge*, which the GLOSSARY does (`gradient_steps` = optimizer steps **on the cartridge**). Extra state:
  compactor weights (small, per-layer).
- **Why it might NOT transfer.** Their compactor is trained on the *same distribution* it compacts; ours would be
  trained on QA and asked to compact MT, and the training-horizon result says extrapolation is exactly where it
  collapses. Also: 36 layers × 8 KV heads of Perceiver is real engineering, not a flag.
- **Verdict:** **candidate — the strongest long-horizon direction**, but it is a build, not an experiment, and it
  moves the project out of "closed-form" into "amortized". Flag this to the orchestrator as a scope question.

---

### L01-5: Recursive summary-vector accumulation across segments — *Adapting Language Models to Compress Contexts* (AutoCompressor, arXiv 2305.14788, EMNLP 2023 · https://arxiv.org/abs/2305.14788)
*Read: §2 method, summary accumulation equation, §3 training (randomized segmenting, BPTT), §4 ablations Fig. 2, §5 30k-token results.*
- **Claim / mechanism.** κ = **50** summary tokens `<Sum>_i` appended per 2,048-token segment; the model emits
  summary vectors through its own LM head. The continual primitive is **summary accumulation**:
  `σ_{<i} = Concat(σ_1, …, σ_{i−1})` prepended to segment `S_i` — *"summary accumulation allows for a direct
  information pathway between each segment and all segments preceding it"*, contrasted explicitly with RMT which
  passes only `σ_{i−1}`. Trained with BPTT over up to **20 segments** (30,720 tokens). Ablation (Fig. 2):
  accumulation strictly beats single-segment compression; perplexity still *improves* from 14 → 20 segments
  (12.49 → 12.47). Inference is gradient-free (summaries are precomputed and cached).
- **Maps to our code.** Our continual loop is the RMT variant, not the AutoCompressor one: each document's write
  **overwrites** the state (`finetune.py::apply_document_am_write_to_cache`, survival **4.7%**). The
  AutoCompressor analogue is `AM_SLOTS_PER_DOC=32` with a **disjoint, monotone allocator** in
  `cartridges/am/ranking.py::rank_am_slots` — document *d* may only write slots `[32d, 32d+32)`, so 16 documents
  tile 512 slots with **zero overwriting**.
- **Prediction about OUR signal.** This is the cleanest "just don't overwrite" proposal, and **our data already
  refutes it.** DIAG-PERDOC: a document written **alone into 32 uncontested slots** is **+1.101 worse on its own
  questions** (sd 0.186, 5/5 same sign; θ=5e6 arm +0.753, 5/5), buying **15.5%** of what the shared write buys,
  with **71–93%** of even that landing on *other* documents' questions. So a disjoint tiling would give
  16 × 15.5% of a non-additive gain: I predict **MT ≥ 3.2** (vs 2.2681), i.e. **worse by ≥0.9**, ~18× the paired
  resolution. Second prediction: `mass_on_S` under a disjoint tiling stays near the incumbent
  (each solo write measured **0.1477–0.1535** vs 0.15851 shared), so the loss is **not** routing — confirming
  the write carries no per-document content, which is the real finding.
- **Cost.** Gradient-free, no extra solves; allocator is 3 lines.
- **Why it might NOT transfer.** AutoCompressor's summaries are produced by a **model trained to produce them**
  through 20 segments of BPTT; ours are produced by a least-squares solve against 64 reference queries. Their
  additivity is trained in. Ours is not, and DIAG-PERDOC says it is absent.
- **Verdict:** **unusable — dead on arrival, and already measured dead.** Recorded here so nobody re-proposes
  "give each document its own slots"; DIAG-PERDOC is the refutation and its effect size is 22× the resolution.

---

### L01-6: Streaming compression where compressed activations ACCUMULATE and raw ones are discarded — *Long Context Compression with Activation Beacon* (arXiv 2401.03462, ICLR 2025 · https://arxiv.org/abs/2401.03462)
*Read: §3 beacon attention (stepwise expansion), interval processing, §4 condensing-ratio sampling, App. F limitation.*
- **Claim / mechanism.** For an interval of `l` raw tokens, `k` beacon tokens with **dedicated projections**
  `{W_Q^b, W_K^b, W_V^b, W_O^b}` (the LM's own weights frozen) query the raw keys/values to emit condensed
  activations at ratio `α = l/k`. Streaming rule, quoted: *"when dealing with the next interval, the raw
  activations of the previous interval are discarded while its condensed activations are accumulated."*
  Condensing ratios are **randomly sampled per interval from {2, 4, …, 128}** during training, which is what
  makes one model serve all compression levels. Documented limitation (App. F): tokens at the start of an
  interval see only condensed context and degrade.
- **Maps to our code.** The beacon projections are the amortized alternative to
  `value_solve.py::sparse_am_value_update`; the interval loop is
  `continual.py::run_per_document_am_phase2`. Flag if built: `AM_WRITE_OP=beacon`.
- **Prediction about OUR signal.** Beacon **accumulates** — its state grows by `k` per interval, so it makes no
  prediction about overwriting. Its transferable claim is the **ratio-randomization** one: a compaction operator
  trained across α ∈ {2…128} generalizes across compaction levels. Mapped to us: our operator is fitted at exactly
  one budget (`TOP_T=32`, 64 reference queries) and MECH-BUDGET-B found `top_t` 64 vs 128 **not even sign-
  consistent** across k. Prediction: **`top_t` will remain a non-lever** for any fixed-slot AM variant
  (|ΔMT| < 0.049 at all k), because the binding constraint is the 428× ratio, not the per-document budget.
  This is a *negative* prediction and it is already 90% confirmed by MECH-BUDGET-B.
- **Cost.** One-time training of the beacon module; gradient-free per interval thereafter. Extra state: growing.
- **Why it might NOT transfer.** It never faces our problem. There is no fixed-slot version of Beacon.
- **Verdict:** **unusable as a mechanism** (its state grows), **useful as evidence for bullet 1**: the streaming
  compression family's answer to "new content" is *accumulate the compressed form and discard the raw*, which is
  the opposite of a fixed-slot overwrite.

---

### L01-7: Gist tokens — compression amortized into the LM by an attention mask — *Learning to Compress Prompts with Gist Tokens* (arXiv 2304.08467, NeurIPS 2023 · https://arxiv.org/abs/2304.08467)
*Read: §2 gist masking, §3 training, §4 results, §5 failure analysis.*
- **Claim / mechanism.** Insert `k ∈ {1,2,5,10}` gist tokens between prompt and input, `(t, g_1…g_k, x)`, and
  **modify the attention mask** so nothing after the gists may attend to anything before them. Compression is
  then learned *for free* during ordinary instruction tuning — no extra objective, no extra cost. Up to **26×**
  prompt compression, 92–104% of ChatGPT parity on seen tasks, degrading to 84.9% / 63.2% out of distribution.
  Failure analysis is the useful part: gists lose **verbatim detail and specific names**; exact-match identity
  drops from ~50% (seen) to ~10% (OOD). k=1 is nearly as good as k=10 — **capacity per gist token is not the
  binding constraint; the trained mask is**.
- **Maps to our code.** No direct map (requires training the LM). The transferable observation maps to
  `cartridges/am/teacher.py::compute_teacher_targets` — what the write is *asked* to preserve.
- **Prediction about OUR signal.** The "k=1 ≈ k=10" result predicts that **slot count is not the axis that carries
  document-specific content in a learned compressor**; content-specificity comes from the *training*. Transferred:
  our **content-free control (28–73% of the MT gain from zero-MT-content documents, ICL itself 53.3%)** is the
  same phenomenon — most of what a compressed prefix delivers on this benchmark is *format/distribution*, not
  retrievable content. Quantitatively I predict: any AM variant will keep the content-free fraction in the
  **25–75%** band regardless of `TOP_T`, `SLOT_SELECTION`, or `KEY_MODE`, because the fraction is set by the
  benchmark and the frozen LM, not the write. **Cheap check:** re-run DIAG-CONTENT at `TOP_T=64` and at
  `SLOT_SELECTION=redundancy`; if the content-free fraction moves outside 25–75%, the write *is* selective and
  the "not a store" conclusion needs revising.
- **Cost.** Requires LM finetuning — out of scope. The check above is 2 GPU-runs, gradient-free.
- **Why it might NOT transfer.** Gisting compresses *instructions* (~20 tokens), not 6k-token papers.
- **Verdict:** **candidate, weak** — as a falsifiable prediction about the content-free fraction, not as a mechanism.

---

### L01-8: In-context autoencoding — the measured limit of gradient-free-at-inference compression is ~4× — *In-context Autoencoder (ICAE)* (arXiv 2307.06945, ICLR 2024 · https://arxiv.org/abs/2307.06945)
*Read: §3 architecture, §3.1 pretraining (autoencoding + LM), §3.2 instruction FT, §4.2 compression limits, Tab. 8.*
- **Claim / mechanism.** A LoRA-adapted copy of the LM encodes context + `k` learnable memory tokens into `k`
  memory slots; the **frozen** decoder conditions on them. Default: **128 slots for 512 tokens = 4×**. Median
  BLEU > 0.98 at 4×; *"as the context length increases beyond 400, both BLEU and EM start to decline."*
  A pretrained ICAE at **8×** (k=64) only matches a non-pretrained one at 4×. **Composition is by
  concatenation** (Tab. 8: segment a long context into N chunks, compress each, concatenate — 1024 memory slots
  ≈ 4096 raw tokens) and **requires concatenation examples in training** to work.
- **Maps to our code.** Concatenation composition → the `AM_COMPOSE_CACHES` eval path of L01-1.
- **Prediction about OUR signal.** This is the sharpest capacity anchor in the *lossless-reconstruction* regime:
  **4–8× is where a purpose-trained encoder into a frozen decoder stops reconstructing.** We are at **428×**,
  i.e. **50–100× beyond ICAE's reconstruction frontier**, which is a first-principles reason our write cannot be
  a store — and matches the measurement that it isn't. Quantitative prediction: the *reconstructible* fraction of
  a document at 16 slots/document is ≈0; therefore **any diagnostic that scores per-document content recovery
  will read at floor**, which is what DIAG-CONTENT (28–73% content-free), DIAG-PERDOC ("writing a paper in makes
  the model worse at that paper") and the 4.7% survival number already say. Second, ICAE's "concatenation needs
  training examples" predicts our L01-1 composition oracle will be **imperfect but positive**: expect a composed
  512+512 cache to land between the two single-cartridge losses, closer to the better one — my point estimate
  **MT 2.00–2.10, QA 2.25–2.35**.
- **Cost.** Gradient-free at inference; encoder training out of scope.
- **Why it might NOT transfer.** ICAE optimizes reconstruction; we optimize attention-output matching, and
  App. F.2/Fig. 11 of `AM.pdf` already show reconstruction loss and downstream quality decouple.
- **Verdict:** **candidate** — for the 428×-vs-4× capacity argument and the composition point estimate.

---

### L01-9: One-token document memory with an amortized, gradient-free write — *xRAG: Extreme Context Compression for Retrieval-augmented Generation* (arXiv 2405.13792, NeurIPS 2024 · https://arxiv.org/abs/2405.13792)
*Read: §3 modality bridge, §3.1 paraphrase pretraining, §3.2 context-aware instruction tuning (NLL + self-distillation KL), §4 results, §6 limitations.*
- **Claim / mechanism.** A document becomes **one token**: a frozen retriever embedding `E`, projected by a
  two-layer MLP `W` (0.46% of Mistral-7B params) into the LM's representation space. Retriever and LM both
  frozen; only `W` trains, with (i) paraphrase pretraining (reconstruct `D` from `W(E)`) and (ii) instruction
  tuning with NLL + **self-distillation KL between text-conditioned and embedding-conditioned predictions**.
  Compression **175.1 → 1 ≈ 178×**, FLOPs ÷3.53. Crucially: **new documents need no gradient step** — encode
  offline, plug in. Breaks on reasoning-heavy multi-hop (HotpotQA, FactKG); §6 admits **only top-1 is evaluated,
  multi-document composition is untested**.
- **Maps to our code.** The closest thing to an *amortized* replacement for our per-document solve. Would
  require a projector trained once, then `finetune.py::apply_document_am_write_to_cache` replaced by
  `v_new = W(embed(doc))` scattered into the selected slots. Flag: `AM_WRITE_OP=projector`.
- **Prediction about OUR signal.** 178× is the only published gradient-free-at-write compression near our 428×,
  and it works **for recall and fails for reasoning** — which is precisely our benchmark's split:
  full-context ICL on MT is **53.3% content-free**, i.e. our MT eval is largely a distribution/format task, and
  the residual 47% is the part that behaves like reasoning/recall. Prediction: an xRAG-style projector write
  would recover roughly the **content-free portion and little else**, landing MT ≈ the level our zero-MT-content
  controls already reach: **MT 2.7–3.1**, i.e. **worse than the incumbent 2.2681**. Also predicts that
  `mass_on_S` is irrelevant to this variant (the write is 1 slot/document = 16 slots total, mass ≪ 0.28).
- **Cost.** Gradient-free per document; one-time projector training; extra state 1 slot/document.
- **Why it might NOT transfer.** 178× on a 175-token passage ≠ 428× on a 6,000-token paper; and their
  single-token memory is *appended*, so the fixed-slot problem never arises.
- **Verdict:** **candidate, low priority.** The prediction is a *loss*, and it is worth stating so nobody
  proposes "just embed the document into a slot".

---

### L01-10: Eviction, not writing — the streaming KV family — *H2O* (arXiv 2306.14048, NeurIPS 2023) + *SnapKV* (2404.14469) + *PyramidKV* (2406.02069) + *Scissorhands* (2305.17118) + *StreamingLLM* (2309.17453)
*Read: H2O §3–4 (Alg. 1, Thm 4.4), and the family's shared protocol as reported in `AM.pdf` §4.1/App. B.3 and the KVPress benchmark discussion.*
- **Claim / mechanism.** H2O: "heavy hitters" are tokens whose **accumulated attention score** is large;
  attention scores follow a power law; greedy eviction with `|S_i| = k` and `|S_i \ S_{i−1}| ≤ 1`, and under a
  submodularity assumption `f(S̃_i) ≥ (1−α)(1−1/e)·max_{|S|=k} f(S) − β`. Budgets swept at **4 / 10 / 20 / 60%**;
  20% is near-lossless. SnapKV votes with an observation window at prefill and pools; PyramidKV allocates a
  **non-uniform per-layer budget**; Scissorhands relies on "persistence of importance"; StreamingLLM keeps
  attention sinks + a rolling window. **The decisive shared property: every retained entry is an ORIGINAL,
  UNMODIFIED KV row.** Nothing is ever written. Once evicted, it is gone.
- **Maps to our code.** `cartridges/am/ranking.py::_rank_attention_mass_per_layer` **is** the H2O score
  (`tf`, and we verified `USE_IDF=0 ⇒ tfidf == tf`). PyramidKV's non-uniform budget maps to a per-layer
  `TOP_T` schedule — AM.pdf §3.4 independently finds **nonuniform head budgets** to be the single most important
  ingredient in its own ablation ("the most important being nonuniform head budgets").
- **Prediction about OUR signal.** (a) **The transplant of this family is what MECH-INFOGATE/MECH-CONSTRAINED
  already tested and it failed**, and now there is a reason: eviction methods rank *what to keep among genuine
  content*; we rank *what to destroy*. The two rankings have opposite cost structure and there is no reason a
  keep-criterion is a destroy-criterion. Our own data agrees: `log(MT routing mass)` → MT at Pearson −0.9775, so
  H2O's score **is** our objective, and every deviation from it loses. **Prediction: no member of this family
  ranked by anything other than raw accumulated attention mass can beat the incumbent — |ΔMT| ≥ +0.05 for all
  of them.** Already 3-for-3 (redundancy, fisher, mass×redundancy) plus a monotone dose-response.
  (b) **PyramidKV's non-uniform budget is the ONE untested member.** We hold `GRANULARITY=per_layer` with a
  *uniform* `TOP_T=32` across all 36 layers; AM.pdf calls the non-uniform version its most important ingredient.
  Prediction: a per-layer budget proportional to each layer's share of writable MT routing mass raises realised
  MT bandwidth from **0.2799** toward **0.33–0.36** at the same total 36×32 = 1152 slot-writes ⇒ by the measured
  slope (−0.166 per unit Δlog mass) **ΔMT ≈ −0.03 to −0.04** — i.e. **just inside the ±0.049 resolution and
  therefore probably not worth a run**. State that plainly.
- **Cost.** Gradient-free, free (a reallocation of an existing budget).
- **Why it might NOT transfer.** Fundamentally: their fixed budget holds *original* KV; ours holds *solved*
  values fitted to 64 reference queries, and a slot we keep is not a slot that holds content.
- **Verdict:** **unusable for retention/selection (closed by measurement); one weak candidate** (non-uniform
  per-layer budget), predicted **sub-resolution** — I do not recommend spending a run on it.

---

### L01-11: A compressed memory that is appended and retrieved, never overwritten — *InfLLM: Training-Free Long-Context Extrapolation with Efficient Context Memory* (arXiv 2402.04617, NeurIPS 2024 · https://arxiv.org/abs/2402.04617)
*Read: §3 block memory units, representative-token scoring, §3.3 relevance lookup, §4 offloading/LRU, §5 results.*
- **Claim / mechanism.** Evicted tokens are grouped into blocks of `l_bs = 128`; each block keeps `r_k = 4`
  **representative tokens** scored by `r_m = (1/l_L) Σ_j q_{m+j}·k_m` (how much the local window attends to it).
  At decode, `sim(X, B) = Σ_i Σ_j q_{i+l_P}·k^B_{b_j}` selects the **top `k_m = 16` units**, concatenated with
  initial + local tokens. **Fully training-free.** Blocks are **never modified** — only appended, offloaded to
  CPU, and LRU-cached on GPU. Scales to **1,024K tokens**, 50× the training length, with minimal degradation.
- **Maps to our code.** No map: our memory has no retrieval step and cannot grow. The nearest analogue would be
  an eval-time gate that *selects which 512-slot cartridge to load per query*, i.e. L01-2's retrieval.
- **Prediction about OUR signal.** InfLLM's "capacity is unbounded because memory is external and retrieved"
  answer is unavailable to us by construction. Its transferable number is the **representative-token ratio:
  4 of 128 = 3.1% of tokens retained per block as *addressing keys*, with the rest offloaded, not destroyed.**
  Our analogue keeps **512 slots for 219k tokens = 0.23%**, and destroys the rest. Prediction: **any purely
  in-slot mechanism is ~13× below even InfLLM's addressing-key density**, so no in-slot ranking can recover
  document-level addressability — which is exactly the measured `ρ_key` result (at its held-out control floor
  everywhere, 0/288 heads with MT-specific query directions). **These two numbers agree and should be reported
  together**: the field's cheapest retrieval memory still spends 13× more slots on addressing alone.
- **Cost.** Training-free; requires external storage and a retrieval step at every decode.
- **Why it might NOT transfer.** Requires unbounded storage — violates the mission.
- **Verdict:** **unusable as a mechanism; load-bearing as a bound.** The 3.1%-vs-0.23% comparison is the cleanest
  external justification for closing the key/selectivity family.

---

### L01-12: A fixed-size, self-updating, GRADIENT-FREE memory pool — and its measured retention horizon — *MemoryLLM* (arXiv 2402.04624, ICML 2024) + *M+* (arXiv 2502.00592, ICML 2025)
*Read: MemoryLLM §3 memory pool + self-update, §4 knowledge-retention experiments; M+ §1 limitation, §3 long-term pool + retriever, §4 retention results, Fig. 7.*
- **Claim / mechanism.** This is the **closest published architecture to our problem**: a fixed pool of
  `N = 7,680` hidden vectors **per layer** (32 layers, d = 4096 ⇒ 1.066B params of pure memory), updated at
  inference **without any gradient**: take the last `K = 256` memory tokens, concatenate with the new context's
  hidden states, push through the layer, **randomly drop `K` old memory tokens**, shift left, and append the
  `K` new ones. Steady-state retention `(1 − K/N)^t`, with `(1−K/N)^{N/K} ≈ 1/e`. They run **650,000+ sequential
  updates with no collapse**. But M+ then measures the thing we actually care about: **MemoryLLM "struggles to
  retain knowledge beyond 20k tokens."** M+ fixes it by **not throwing the dropped tokens away** — they go to a
  layer-wise **long-term pool of up to 150k tokens** with an age variable and a contrastively-trained retriever
  (two-layer MLPs, output dim d/20), extending retention **<20k → >160k tokens** at equal GPU memory.
- **Maps to our code.** The random-drop self-update is a *gradient-free fixed-slot write* and maps onto
  `ranking.py::rank_am_slots` + `finetune.py::apply_document_am_write_to_cache`. Flag: `SLOT_SELECTION=random`
  — which, remarkably, **we have never run** as a control despite it being MemoryLLM's actual policy.
  M+'s long-term pool maps to nothing we have (it grows).
- **Prediction about OUR signal.** Two hard numbers.
  (a) **Capacity per slot.** MemoryLLM: 7,680 slots/layer retaining ≈20k tokens ⇒ **2.6 tokens/slot** before
  decay, for a *purpose-built, jointly-trained* gradient-free write. We ask **428 tokens/slot** — **165× beyond
  the only published gradient-free frontier.** Even M+ with a 150k-token external pool reaches only
  ≈20 tokens/slot-equivalent. **This is the review's central quantitative answer to "is our k≈12 saturation
  anomalous": it is not anomalous, it is 1–2 orders of magnitude *more generous* than the literature would
  predict.** Our curve saturating at 12 of 16 documents (≈72k tokens absorbed) at 512 slots = **141 tokens/slot
  at the optimum** is, if anything, an unexplainedly *good* number — and the content-free result (28–73%)
  explains why: it is not storing 141 tokens/slot, it is applying a distribution shift.
  (b) **Random selection is the literature's own baseline and we lack it.** Prediction: `SLOT_SELECTION=random`
  at `TOP_T=32` will land MT **2.40–2.50** (between the pure-safety selectors at 2.4388 and the incumbent at
  2.2720), because random selection captures MT routing mass ≈ 32/511 = 6.3% of writable mass vs the incumbent's
  ~28%, i.e. Δlog(mass) ≈ −1.49 ⇒ ΔMT ≈ **+0.25**. **Falsifier for the whole mass law:** if random lands
  within 0.049 of the incumbent, mass does not cause MT and three regressions are confounded.
- **Cost.** Gradient-free. `SLOT_SELECTION=random` is ~10 lines in `ranking.py` and one GPU run.
- **Why it might NOT transfer.** MemoryLLM's model is *trained* with the self-update in the loop (50% of
  training steps use it), so the frozen LM has learned to read a decayed memory. Our Qwen3-4B never has.
  That difference is probably worth more than any selector.
- **Verdict:** **candidate — highest-value cheap control in this review** (`SLOT_SELECTION=random`), plus the
  single best external anchor for the capacity question.

---

### L01-13: Closed-form, sequential, erasable write into a FIXED 512-slot memory — *Larimar: LLMs with Episodic Memory Control* (arXiv 2403.11901, ICML 2024 · https://arxiv.org/abs/2403.11901)
*Read: §3 memory model, write/read equations, sequential-update rule, §4.1 capacity/rewrite results, §4.3 forgetting, Tab. 4.*
- **Claim / mechanism.** Memory `M ∈ R^{K×C}` with **K = 512** (K = 1000 for the sequential-editing study),
  C = 768. Write is a **one-shot pseudo-inverse**: `M = W_0^† Z_ζ`, i.e. `min_M ‖Z_ζ − W_0 M‖_F²`. Read:
  `W̄ = Z M^†`, sample `W ~ N(W̄, σ_W² I)`, `Z_readout = W M`. Sequential updates are **exactly recursive least
  squares / Sherman–Morrison**:
  `C_i = C_{i−1} + α_i W_i^T W_i`, `M_i = M_{i−1} + α_i C_i^{−1} W_i^T (Z_i − W_i M_{i−1})`, with **α_i = +1 to
  write and α_i = −1 to forget** — a single sign flip gives exact erasure (~0% recall on forgotten facts, ~99%
  retained). Capacity: **~100% rewrite accuracy up to 512 edits (= K), dropping to 82% at 1024 (= 2K)** — a
  **cliff at N = K**, not a gradual saturation. 1000 sequential edits: retention rate **0.97**.
- **Maps to our code.** `cartridges/am/value_solve.py::sparse_am_value_update` is a ridge least-squares solve;
  the **RLS/Sherman–Morrison form is the exact sequential extension we do not have** — currently every document
  re-solves from scratch on the live cache, discarding the previous documents' normal equations. Proposed:
  `AM_SEQUENTIAL_SOLVE=rls`, carrying `C_i = C_{i−1} + X_i^T X_i` across documents inside
  `continual.py::run_per_document_am_phase2` and updating `M_i` with the residual form above.
- **Prediction about OUR signal.** (a) **Capacity.** Larimar is the only published fixed-slot gradient-free
  memory with a clean capacity law, and it is **≈1 item per slot with a cliff at N = K**. Our items are 6,000-token
  documents, not sentences; if a document is worth ~40 "items" then 12 documents ≈ 480 items ≈ **K = 512** —
  **our k≈12 saturation lands on Larimar's capacity cliff to within 7%.** That is the most quantitatively
  satisfying external explanation of the k-curve I found, and it predicts the saturation point should scale
  **linearly with slot count**: at 1024 slots the minimum should move to **k ≈ 24 (i.e. past 16, so monotone
  through k=16)**; at 256 slots to **k ≈ 6**. **This is directly testable with existing code** (`n_tokens` is a
  Phase-1 config) and is a strong falsifier: if the k-curve minimum does *not* move with slot count, capacity is
  not what saturates.
  (b) **RLS.** Carrying the Gram matrix across documents should reduce the mutual-annihilation that gives 4.7%
  survival. But note DIAG-SEQUENCE already showed 4.7% survival ≠ 4.7% information survival, and DIAG-PERDOC
  showed contention *helps*. So I predict RLS buys **|ΔMT| < 0.10** and possibly negative — it preserves
  *earlier documents'* fit, which is retention, and **retention is already 0.573 ahead**. Mark it accordingly.
- **Cost.** Gradient-free. One extra `K×K` (here `t×t`, t = 32–128) inverse carried per (layer, head, document);
  cheap. Extra state: one Gram matrix per (layer, head) ≈ 36 × 8 × 128² floats ≈ 4.7M — negligible.
- **Why it might NOT transfer.** Larimar's addressing `W` is *derived from the item* and the memory is
  **fully writable**; ours has 511 writable slots with **frozen foreign keys** and an addressing vector we do
  not control (QA/MT routing cosine 0.99899 — every item addresses the same slots). Larimar's capacity law
  assumes near-orthogonal addressing; ours is maximally collinear, so the effective capacity should be
  **far below** K, not at it. That cuts against the (a) coincidence above and is worth saying honestly.
- **Verdict:** **candidate** for the slot-count/capacity-scaling test (a); **RLS marked low-value** because it
  buys the axis that is already ahead.

---

### L01-14: Adding memory with zero gradient steps — and CAPPING its attention mass as it grows — *KBLaM: Knowledge Base augmented Language Model* (arXiv 2410.10450, ICLR 2025 · https://arxiv.org/abs/2410.10450)
*Read: §3 knowledge tokens + rectangular attention equations, §3.3 the log C − log M scaling, §4 scaling to 10K triples, §5 over-refusal analysis.*
- **Claim / mechanism.** Each KB triple `(name, property, value)` is encoded by a **frozen** sentence encoder and
  projected by **learned per-layer linear adapters** into one `(k̃_m, ṽ_m)` **knowledge token**, injected into
  every attention layer. "Rectangular attention": prompt tokens attend to all prior prompt tokens **and** all `M`
  knowledge tokens; knowledge tokens attend to nothing.
  `ỹ_n = [Σ_m exp(w̃_{n,m}) ṽ_m + Σ_i exp(w_{n,i}) v_i] / [Σ_m exp(w̃_{n,m}) + Σ_i exp(w_{n,i})]`.
  **Updating the KB touches only the affected token — no gradient pass, no cache recomputation.** Scales past
  **10K triples** with `O(M)` memory. And the mechanism that matters most to us: to stop `M` knowledge tokens
  from swamping `N` prompt tokens as `M` grows, they add a **query-independent scalar bias**
  `w̃_{n,m} ← log C − log M + ⟨q̃_n, k̃_m⟩/√D`, which multiplies the KB's contribution by `C/M` and **holds the
  memory block's share of the softmax constant** as the KB scales.
- **Maps to our code.** `log C − log M` is **exactly a β**, and β is already fully plumbed:
  `cartridges/am/key_select.py::refit_beta_nnls`, `cartridges/am/teacher.py::compute_teacher_log_mass`,
  `cartridges/cache.py::get_cartridge_beta`, eval-time at `models/qwen/modeling_qwen3.py:195`. Flag: a fixed
  scalar offset `AM_BETA_CONST=<value>` applied to the written support (no NNLS at all).
- **Prediction about OUR signal.** This is the review's **cleanest independent corroboration of MECH-BETA's
  negative**. KBLaM adds a query-independent scalar because raising a memory block's mass **hurts**; our MECH-BETA
  raised `mass_on_S` **4.23×** and made **both axes worse**, and moved selectivity only 1.043 → 1.051 (β is
  query-independent, so it *provably* cannot move selectivity — same algebra as KBLaM's `C/M`). Prediction:
  a constant `AM_BETA_CONST > 0` reproduces MECH-BETA (both axes worse); a constant `AM_BETA_CONST < 0`
  (KBLaM's *direction* — capping written-slot mass) will **hurt MT by the mass law** (Δlog mass < 0 ⇒ ΔMT > 0)
  and **help QA by less than 0.054**. So: **the β family is closed in both directions**, and KBLaM explains why
  — mass control is a *stability* device for a growing memory, not an acquisition device for a fixed one.
  One residual, genuinely untested question: **does the mass law survive past the k=12 optimum?** MT *rises*
  +0.117 over k=13…16; if `mass_on_S` also rises there, the law breaks at exactly the saturation point.
  That is Experiment 1.
- **Cost.** Gradient-free, one scalar. Free.
- **Why it might NOT transfer.** KBLaM's memory is **appended** (M grows, N fixed); ours is **substituted** into
  a fixed 512. Their scalar defends the prompt from the memory; ours would have to defend Phase-1 from Phase-2
  *inside the same block*, where a single scalar per slot cannot distinguish QA queries from MT queries
  (separation 2.5× below sampling noise).
- **Verdict:** **candidate (as a closure argument, not a new run).** The β family should be recorded as closed
  with a *mechanistic* reason imported from KBLaM, not merely "it didn't help".

---

### L01-15: Test-time gradient descent as the WRITE operator — the controlled comparison we need — *GradMem: Learning to Write Context into Memory with Test-Time Gradient Descent* (arXiv 2603.13875 · https://arxiv.org/abs/2603.13875)
*Read: §3 memory parameterization, §3.2 write objective, §4 forward-only vs gradient comparison tables, §4.2 capacity scaling, §6 discussion.*
- **Claim / mechanism.** Memory `M ∈ R^{m×d}` (m = **8** vectors, prefix-prepended, **independent of context
  length**). The write is literal test-time training: `K ≤ 5` steps of
  `L_write(M; C) = −Σ_i log f_θ(t_i | [M; t_{<i}])`, `M_{k+1} = M_k − α∇L_write`. Then the head-to-head, **same
  architecture, same 8 vectors**:

  | task | forward-only (RMT) | GradMem ×1 | GradMem ×5 |
  |---|---|---|---|
  | 16 KV-pairs | 45.5% | **96.3%** | 100.0% |
  | 32 KV-pairs | 44.3% | **86.9%** | 99.9% |
  | 64 KV-pairs | 19.3% | **58.6%** | 99.1% |
  | 96 KV-pairs | — | 32.6% | 88.4% |

  And, quoted: *"repeating the forward-only WRITE by re-reading the same context segment (×2–×5) yields weak or
  inconsistent improvements"*, whereas extra gradient steps monotonically help. Their explanation: *"once M is
  emitted, the write operation cannot verify whether the context was encoded well enough, nor correct mistakes."*
- **Maps to our code.** `examples/qasper2/train/continual_am_sparse.py` + a new
  `AM_POST_SOLVE_STEPS=<0|1|2|5>` that runs `n` optimizer steps **restricted to the AM-selected slots**
  (a masked gradient on `cache.values[:, S]`), initialised from the AM solution, using the same
  self-distillation loss as Phase 1. This **does** move `gradient_steps` off 0 and must be reported as such.
- **Prediction about OUR signal.** GradMem's table is the field's answer to "why is our gradient-free write
  0.248 short": **a forward-only write recovers 19–46% of what one gradient step recovers, at the same memory
  size**. Transferred to our axis with the Phase-1 floor (MT 3.7825) and the dense bar (MT 1.8725) as endpoints,
  our best gradient-free point 2.2681 already recovers **79%** of the available range
  ((3.7825−2.2681)/(3.7825−1.8725)) — *better* than GradMem's forward-only writes do. So my prediction is
  **`AM_POST_SOLVE_STEPS=1` reaches MT 2.02–2.10** (i.e. clears or nearly clears the target) and
  `=5` reaches **MT ≤ 1.95**, while QA rises by **< 0.15** because the gradient is masked to the 32 written slots
  per layer and the QA-Fisher exposure of that set is **29.6%**. **Falsifier:** if 1 step buys < 0.049 MT, then
  the shortfall is *not* "forward-only-ness" and the capacity argument (L01-1/2/12) owns the whole 0.248.
  Either outcome is decisive and neither is currently known.
- **Cost.** **Not gradient-free** — this is the point. 1–5 optimizer steps on ≤32 slots/layer.
  ~seconds. Extra state: none.
- **Why it might NOT transfer.** GradMem's tasks are synthetic KV-pair retrieval, where "capacity" is countable;
  our MT loss is 53%-content-free, so a gradient step may just re-learn the format faster rather than store more.
  Also, m = 8 vectors for one context ≠ 512 slots for 32 documents.
- **Verdict:** **candidate — the single most informative experiment in this review** (Experiment 3), precisely
  *because* it breaks the `gradient_steps = 0` rule: it converts the mission's constraint from an assumption into
  a measured price.

---

### L01-16: Fixed-footprint continual adaptation by INDUCING and RETRIEVING KV entries — *InduceKV: Fixed-Footprint Continual Adaptation of Multimodal LLMs via Inducing KV Memories* (arXiv 2607.02010 · https://arxiv.org/abs/2607.02010)
*Read: abstract, §1, §3.1–3.4 (bilevel inducing-set selection, Alg. 1), §4 theory (online budgeted subset selection).*
- **Claim / mechanism.** The only 2026 paper I found that names our problem: *"fixed-footprint continual
  adaptation: the deployed adaptation state is kept under a fixed memory budget, while the backbone model is
  left unchanged."* Their answer is **not** to overwrite. Each selected training prefix is stored as an
  **attention-ready memory entry** = a frozen retrieval key + compact layerwise KV payloads **appended** to the
  self-attention cache. Under budget `B`, a **bilevel** optimizer builds a compact "inducing set": inner level
  fits a lightweight retrieval calibration `φ`; outer level selects a budgeted subset `w` (on the simplex
  `Σ w_i = B`) trading **current-task likelihood, anchor-based retention, and coverage in the frozen retrieval
  space**, solved by projected gradient. They state explicitly: *"we … perform no gradient updates to θ at any
  time"* — the backbone is frozen, but the memory construction is **not** gradient-free.
- **Maps to our code.** The three-term outer objective is the exact generalization of what
  `cartridges/am/ranking.py::_rank_slot_prior_per_layer` does with one term. `constrained_mass` (MECH-009) is
  their *likelihood + retention* pair with the retention term as a hard constraint; **the coverage/diversity
  term is the piece we have never implemented** (a log-det/DPP-style repulsion between the 16 documents'
  selections). Flag: `SLOT_SELECTION=coverage_mass`, `AM_COVERAGE_LAMBDA`.
- **Prediction about OUR signal.** The coverage term is designed to fix exactly the pathology DIAG-OVERWRITE
  measured — **mean pairwise Jaccard 0.713 between documents' top-32 sets, 9.83 writes/slot, mean Spearman 0.958
  between documents' scores, 87.4% overlap with a document-independent top-32**. On the literature's logic it
  should help a lot. **On our data it must lose**, and I will state the number: forcing document *d*'s selection
  away from documents `<d` moves it down the mass ranking; MECH-CONSTRAINED's dose-response gives
  MT ≈ 2.2720 + 0.166·(−Δlog mass), and a Jaccard-0.713 → Jaccard-0.2 repulsion at `TOP_T=32` costs roughly the
  same mass as `q = 0.5` did (0.2799 → 0.1327, Δlog = −0.746) ⇒ **predicted MT ≈ 2.39, i.e. +0.12 worse**.
  Plus DIAG-PERDOC's independent measurement that decontention costs **+1.101**. **This direction is dead, and
  it is dead for a reason that is *specific to us*: the field assumes overlap is interference; we measured that
  overlap is the mechanism.** That is a genuinely novel negative worth writing up.
- **Cost.** Not gradient-free (projected gradient on `w`, inner steps on `φ`). Memory grows with `B` entries.
- **Why it might NOT transfer.** Their entries are *appended payloads with their own retrieval keys*; our slots
  have frozen foreign keys and no retrieval step. Their budget is entries; ours is slots inside one block.
- **Verdict:** **unusable — dead on arrival, with a measured cause** (DIAG-PERDOC +1.101, MECH-CONSTRAINED
  dose-response). Recorded because it is the most likely thing a reader of the 2026 literature would propose next.

---

### L01-17: A fixed-size prompt pool, selected per input — and its documented selection collapse — *Learning to Prompt for Continual Learning (L2P)* (arXiv 2112.08654, CVPR 2022) + *DualPrompt* (2204.04799, ECCV 2022) + *CODA-Prompt* (2211.13218, CVPR 2023)
*Read: L2P §4 prompt pool, key-query matching, Eq. (4)–(5), Fig. 3–4; DualPrompt G/E-prompt split; the selection-accuracy analysis reported in the follow-up literature.*
- **Claim / mechanism.** A pool `P = {P_1…P_M}` of `M ≈ 10–20` prompts, each `P_j ∈ R^{L_p×D}` with `L_p = 5`,
  each paired with a **learnable key** `k_j`. The query is the frozen model's own `[class]` feature
  `q(x) = f(x)[0,:]`; selection is top-N cosine, `K_x = argmin_{s_i} Σ γ(q(x), k_{s_i})`, and the loss adds
  `λ Σ γ(q(x), k_{s_i})` (λ = 0.5) to **pull the selected keys toward the query**. Crucially there is an
  explicit **anti-collapse device**: a frequency table `H_t = [h_1…h_M]` of normalized selection counts, and the
  lookup score is multiplied by `h_{s_i}` so over-used prompts are penalized. Fig. 3: on Split-CIFAR-100 "tasks
  largely share all prompts"; on 5-datasets they share less. The documented failure, from the follow-up
  literature: DualPrompt's E-prompt selection is perfect after task 1 and drops **below 50% by task 10**, and
  *"prompts repeatedly selected across different tasks are prone to being overwritten, aggravating catastrophic
  forgetting."*
- **Maps to our code.** L2P's key-query matching **is** our slot-ranking problem with the roles reversed
  (they *learn* the keys; ours are frozen). The frequency penalty maps directly onto
  `ranking.py::_rank_slot_prior_per_layer` as a new prior: `score_l[j] = tf_l[j] · h_l[j]^{-γ}` with
  `h_l[j]` = the running count of documents that have already written slot *j*. Flag:
  `SLOT_SELECTION=freq_penalized_mass`, `AM_FREQ_PENALTY_GAMMA`.
- **Prediction about OUR signal.** This is the exact mechanism the human's gating hypothesis wants, from the
  literature that invented it, and **we can predict its number without running it**: it is a monotone
  down-weighting of the incumbent's own score, so by the dose-response law it must **lose MT**. At γ = 1 and
  9.83 mean writes/slot, the penalty on the most-contested slots is ~10×, which is a larger perturbation than
  MECH-CONSTRAINED's `q = 0.25` (Δlog mass −1.029 ⇒ ΔMT +0.167). **Predicted MT ≈ 2.44, QA ≈ 1.87** — i.e.
  the same corner MECH-INFOGATE's `fisher` already found (QA 1.8641, +0.397 MT). **The prompt-pool literature's
  anti-collapse device is our `fisher` selector under another name, and it buys the axis we do not need.**
  Second, sharper prediction: L2P's *own* diagnostic — selection overlap across tasks — is our **0.914 top-32
  overlap / 0.958 Spearman**; L2P calls that pathological, but DIAG-PERDOC measured it as **load-bearing**
  (+1.101 penalty for removing it). **If a frequency penalty at any γ > 0 improves MT, DIAG-PERDOC is wrong.**
- **Cost.** Gradient-free, free (a counter array in `ranking.py`).
- **Why it might NOT transfer.** L2P's prompts are **learned**, so a fresh prompt can be *trained* to be useful;
  our slots are written by a solve against 64 reference queries with a frozen foreign key, so an unused slot is
  not a blank one — it is a low-mass one, and low mass is causally bad (Pearson −0.877/−0.9775/−0.986).
- **Verdict:** **unusable — predicted to lose, with the number.** Worth one run *only* as the falsifier of
  DIAG-PERDOC; otherwise skip. Register it so the gating family is closed against its own home literature.

---

### L01-18: Test-time memorization with an explicit forget gate over a bounded state — *Titans: Learning to Memorize at Test Time* (arXiv 2501.00663 · https://arxiv.org/abs/2501.00663) + *Learning to (Learn at Test Time)* (TTT layers, arXiv 2407.04620, ICML 2024)
*Read: Titans §3 neural memory, surprise + momentum + decay equations, §4 MAC/MAG/MAL, §5.5 depth ablation; TTT §2 inner-loop formulation.*
- **Claim / mechanism.** The memory `M` (a deep MLP, `L_M ≥ 1`) is updated **at test time** by descending an
  associative loss `ℓ(M_{t−1}; x_t) = ‖M_{t−1}(k_t) − v_t‖²` with momentum and decay:
  `S_t = η_t S_{t−1} − θ_t ∇ℓ(M_{t−1}; x_t)`, `M_t = (1 − α_t) M_{t−1} + S_t`. `α_t` is a **data-dependent
  forgetting gate** ("closely related to the gating mechanism in modern RNNs"): `α_t → 0` preserves, `α_t → 1`
  clears. `η_t → 0` resets surprise at context boundaries. Three integration variants (MAC/MAG/MAL). Depth
  helps monotonically (`L_M` = 1…4), and replacing deep memory with linear memory costs 27.01 → 28.49 ppl.
  **It is not gradient-free** — `∇ℓ` is a backward pass through the memory at inference.
- **Maps to our code.** The `(1 − α_t)` decay maps onto `DELTA_WEIGHT` (our trust region, currently 1e-2) in
  `finetune.py::apply_document_am_write_to_cache`; the surprise term maps onto the residual
  `(Z_i − W_i M_{i−1})` we do not carry (see L01-13). A data-dependent `α_t` would be
  `AM_DELTA_WEIGHT_SCHEDULE=surprise` — scale the per-document trust region by the document's own solve residual.
- **Prediction about OUR signal.** The one transferable, cheap idea is a **document-dependent trust region**.
  Our `DELTA_WEIGHT` is a constant 1e-2 for all 16 documents, yet the k-curve says documents 1–3 deliver
  **56.7%** of the MT improvement while holding **18.8%** of eval questions, and documents 13–16 make MT
  **worse by +0.117**. A surprise-scaled `α_t` that shrinks the write as the residual falls should suppress
  precisely the k=13…16 regression. **Prediction: a decaying schedule `DELTA_WEIGHT_k = 1e-2 · (12/k)` for k > 12
  recovers most of the +0.117 tail, giving MT ≈ 2.44 at k=16 vs 2.552 — but does NOT beat the k=12 point
  (2.435/2.4352), so it buys reporting hygiene, not acquisition.** Because the mission reads the argmin over k
  anyway, **the value of this is ≈0.** State that plainly rather than dressing it up.
- **Cost.** The Titans update itself is **not** gradient-free (disqualifying). The trust-region schedule is free.
- **Why it might NOT transfer.** Titans' memory is a network trained end-to-end *with* its test-time update in
  the loop; our frozen Qwen3-4B has never seen a written cartridge slot during pretraining.
- **Verdict:** **unusable as proposed** (needs gradients, and the free part is predicted worth ~0). Recorded
  because "surprise-gated writing" is the obvious next suggestion and it is priced here at zero.

---

### L01-19: Composable compressed KV with position-agnostic extraction — *C²KV: Compressed and Composable KV Cache Reuse for Efficient LLM Inference* (arXiv 2607.17715 · https://arxiv.org/abs/2607.17715)
*Read: structured information flow, C²Token asymmetry, positional handling, composition results, training setup.*
- **Claim / mechanism.** Learnable **C²Tokens** attend to local document blocks under an **asymmetric** mask —
  original tokens feed into C²Tokens, but C²Tokens **never modify original token representations** — which makes
  the extracted KV **position-agnostic and context-independent**. Positional encodings are applied *after*
  extraction: *"the retrieved KVs are applied new positional embeddings according to their placement in the final
  input sequence, where positional offsets are accumulated across concatenated segments."* Composition is
  **direct concatenation** with accumulated offsets, stable on RULER from 4k→64k. Training-based
  (compression–concatenation co-training; base model frozen, only C²Token embeddings + new QKV heads trained).
- **Maps to our code.** The "store before RoPE, apply RoPE at placement time" discipline is **exactly MECH-005**
  (`cartridges/am/key_select.py`, `AM_KEY_REPOSITION=1` + `AM_ROPE_THETA=5000000`). Our documented hazard —
  rotating at θ = 1e4 gives logit error **1.054**, at the correct θ **4.8e-07**, at the wrong θ **4.4–5.1**
  (worse than no correction) — is the same failure C²KV designs around.
- **Prediction about OUR signal.** Third independent confirmation (with Still and the AM appendix) that
  **position-free storage + re-rotation at placement is the correct primitive**. Prediction: MECH-005's
  **ΔMT −0.1545** should be **reproducible under any further perturbation of the write** (selector, budget,
  seed), because it is fixing a representation error, not exploiting a configuration. Corollary prediction:
  the **QA** side of MECH-005 — currently "consistently negative, consistently inside the ±0.054 band" — should
  become significant if measured with more paired examples, since a correctly-positioned key cannot hurt QA.
  Concretely: at n = 78 QA examples the observed effect is inside the band; at n ≈ 300 the same effect size
  would clear it. **That is the cheapest available path to confirming the project's own positive result on both
  axes, and it needs no new mechanism — only more eval examples.**
- **Cost.** Gradient-free (MECH-005 already exists and is on). More QA eval examples: cheap, no training.
- **Why it might NOT transfer.** C²KV composes by concatenation and never overwrites; only the RoPE discipline
  transfers.
- **Verdict:** **candidate** — not for a new mechanism, but for the observation that **the project's only
  confirmed win is the one the 2026 literature independently converged on**, and that its unconfirmed QA axis is
  an *n* problem, not a mechanism problem.

---

## Direct answers to the two questions I was asked

**"Does anyone report a fixed-capacity compressed memory that accepts new domains without a gradient pass?"**

**One architecture, and it is 165× more generous than ours: MemoryLLM** (L01-12). Fixed pool of 7,680 slots per
layer, updated at inference with **no gradient** (random-drop-K, append-K), 650k+ sequential updates without
collapse — but with a measured **retention horizon under 20k tokens**, i.e. **2.6 tokens/slot**. Its successor M+
does not fix this in-place; it fixes it by **adding a 150k-token external pool with a trained retriever**.
Larimar (L01-13) is the only *closed-form* gradient-free fixed-slot write, and its capacity is **≈1 item per
slot with a cliff at N = K**. KBLaM (L01-14) accepts new knowledge with zero gradient steps, but its memory
**grows** (`O(M)`) and it must add a `log C − log M` bias just to stop the growing memory from swamping the
prompt. Everything else — Cartridges, CAS, ICAE, AutoCompressor, Beacon, C²KV, InfLLM, InduceKV, Still —
**either grows the memory or pays a gradient pass, or both.**
**Nobody overwrites a fixed compressed cache with a new domain.** Our operator has no published precedent.

**"Is our k≈12-of-16 saturation consistent with, or anomalous against, reported capacities?"**

**Consistent — and if anything our capacity is unexpectedly high.** Four independent anchors:

| source | memory | capacity reported | per-slot |
|---|---|---|---|
| MemoryLLM (grad-free write) | 7,680 slots/layer | retention < 20k tokens | **2.6 tok/slot** |
| M+ (grad-free + retriever) | 10,240 + 150k external | > 160k tokens | ≈ 20 tok/slot-equiv |
| Larimar (closed-form write) | K = 512 | ~100% to 512 edits, 82% at 1024 | **≈1 item/slot** |
| ICAE (trained encoder) | 128 slots | 4× (8× with pretraining) | **4–8 tok/slot** |
| Cartridges at Scale | per-document | ≈1.2K tokens/document | — |
| **ours at the k-curve minimum** | **512 slots** | **12 docs ≈ 72k tokens** | **141 tok/slot** |
| **ours as configured (k = 16, 2 phases)** | **512 slots** | **32 docs ≈ 219k tokens** | **428 tok/slot** |

Our 141 tokens/slot at the optimum sits **54× above** MemoryLLM's gradient-free frontier and **35×** above ICAE's
trained-encoder frontier. The resolution is DIAG-CONTENT: **28–73% of the "gain" is content-free** (full-context
ICL is itself 53.3% content-free), so we are not storing 141 tokens/slot — we are applying a distribution shift
that costs almost no capacity, plus a small content residue that saturates exactly where a fixed memory should.
Two independent published curves have the same non-monotone shape: **Still's iterative compaction (51.0% @32k →
1.5% @128k, below the no-context floor)** and **DualPrompt's prompt-selection accuracy (100% → <50% by task 10)**.
A mid-sequence optimum under repeated writing into fixed capacity is the **normal** signature. **What would be
anomalous is a monotone curve.**

---

## Top 3 experiments I would run next

### E1 — `K-CURVE-MASS`: does the mass law survive past the capacity knee?
- **Single variable:** `k` (documents written so far), 1…16. **Nothing else changes.**
- **What to measure:** `mass_on_S` (MT queries) at every one of the 16 `cache-after-doc-*.pt` snapshots that
  `SAVE_AFTER_EACH_DOCUMENT=1` already wrote, using the code that already computes it
  (`cartridges/am/value_solve.py:146`, `cartridges/am/finetune.py:628`), paired with the MT losses DIAG-SEQUENCE
  already has. **No training run at all** — this is an eval sweep over existing checkpoints.
- **Baseline:** our own k-curve (MT 3.783 / 3.745 / 3.367 / 3.034 / 2.710 / **2.435 @k=12** / 2.449 / 2.552)
  and the mass law's three fits (Pearson −0.877 / −0.9775 / −0.986).
- **What it decides:** the mass law was fitted **across arms at fixed k**, never **along k**. If `mass_on_S`
  rises monotonically through k=16 while MT rises after k=12, the law is **not causal past the knee** and the
  tail is a capacity effect — which promotes E2 and demotes every routing mechanism. If instead `mass_on_S`
  *peaks at k≈12 and falls*, the law is complete, the knee is a routing collapse, and the fix is allocation.
- **Falsified by:** `mass_on_S` monotone in k **and** MT monotone in k — i.e. the tail regression is noise.
  (DIAG-SEQUENCE already flagged the tail as ≤ the old ±0.15 band; with the true paired ±0.049 it is 2.4×
  resolution, so this is answerable.)
- **Cost:** ~16 eval runs, no GPU training. Cheapest decisive bit available.

### E2 — `COMPOSE-512+512`: is the 0.248 shortfall capacity, and does the field's own answer close it?
- **Single variable:** *where the MT content lives* — **written into the Phase-1 cartridge's 512 slots**
  (incumbent) vs **held in a second, independent 512-slot cartridge concatenated at eval** (Cartridges §5.4 /
  CAS). Slot budget per document goes 16 → 32; everything else (selector, `TOP_T`, RoPE, seeds) is held.
- **Arms:** (a) incumbent, best point **QA 1.9468 / MT 2.2681**. (b) Phase-1 cache ‖ *AM-written* MT-only
  512-slot cache (tests whether **the AM write** is additive). (c) Phase-1 cache ‖ *self-distilled* MT-only
  cartridge (tests whether **capacity** is the constraint, with the write held at its best-known quality).
- **Predictions:** (c) **MT ≤ 2.05, QA ≤ 2.30** — passes on both axes. (b) **MT ≥ 3.4** — fails badly, because
  a solo AM write buys only 15.5% of the shared write (DIAG-PERDOC) and 71–93% of even that lands on other
  documents' questions.
- **What it decides:** (c) passing and (b) failing means **capacity is the constraint and the AM write is not
  additive** — i.e. the mission's 512-slot framing, not the ranker, owns the 0.248. (c) failing means capacity is
  *not* the constraint and this whole review's thesis is wrong.
- **Falsified by:** (c) landing MT > 2.15. Then 1024 slots do not help and the deficit is mechanism, not budget.
- **Caveat to state up front:** this **violates the fixed-512-slot constraint**. It is a **capacity oracle**,
  reported as a diagnostic, never as a submission. The orchestrator should treat a pass as a scope question.
- **Cost:** one Phase-1-style self-distillation run on the MT corpus (arm c), one AM run (arm b), 6 evals.

### E3 — `GRAD-LADDER`: price the `gradient_steps = 0` constraint in MT loss
- **Single variable:** `n ∈ {0, 1, 2, 5}` optimizer steps applied **only to the AM-selected slots**
  (`cache.values[:, S]`, masked), initialised from the AM solution, same self-distillation loss as Phase 1,
  same documents, same order, same seed. `n = 0` is bit-identical to the incumbent.
- **Baseline:** **QA 1.9468 / MT 2.2681** at `n = 0`; the dense bar QA 2.3721 / MT 1.8725 as the far endpoint.
- **Prediction (from GradMem's 8-vector table, L01-15):** `n = 1` → **MT 2.02–2.10** and QA ≤ 2.10;
  `n = 5` → **MT ≤ 1.95**. QA should rise by < 0.15 because the masked set carries only **29.6%** of that
  layer's QA Fisher mass.
- **What it decides:** it converts the mission's hardest constraint from an assumption into a **measured price
  in MT loss per gradient step**. If one step buys ≥ 0.20 MT, the honest report is "gradient-free AM is 0.248
  short and one masked step closes it" — which is a *result*, not a failure. If one step buys < 0.049, then
  forward-only-ness is **not** the cause, capacity is (→ E2), and that is equally decisive.
- **Falsified by:** `n = 1` moving MT by less than the paired resolution ±0.049.
- **Cost:** 4 short runs. Explicitly **breaks `gradient_steps = 0`** and must be reported as
  `gradient_steps = n`, not laundered as a diagnostic.

---

## Ledger deltas proposed (for `research_loop/state/literature_ledger.md`)

New entries LIT-028…LIT-034 (I did not write them — no source edits): **LIT-028** Cartridges composition
(2506.06266 §5.4) · **LIT-029** Cartridges-at-Scale monolithic collapse (2606.04557 Tab. 4, Fig. 1) ·
**LIT-030** GradMem forward-only vs gradient write (2603.13875) · **LIT-031** MemoryLLM/M+ gradient-free fixed
pool + 20k-token retention horizon (2402.04624 / 2502.00592) · **LIT-032** Still amortized compaction + the
subset-bound critique + iterative collapse (2606.07878) · **LIT-033** KBLaM `log C − log M` mass cap
(2410.10450) — closes the β family with an imported mechanism · **LIT-034** L2P/DualPrompt frequency penalty
and selection collapse (2112.08654 / 2204.04799) — closes the gating family against its home literature.
Existing **LIT-015** (Larimar) should gain the capacity numbers (100% @512 edits, 82% @1024) and the
slot-count-scaling prediction in L01-13.
