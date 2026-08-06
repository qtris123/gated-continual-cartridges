# AM Phase-2 experiments — techniques, commands, and what each one found

Closed-form Attention Matching writes a Phase-1 cartridge forward onto new documents with
**zero gradient steps**. This file lists every technique that is implemented, the exact command that
exercises it, and the result it produced.

- Field-by-field config reference: [`research_loop/AM_CONFIG.md`](../../../research_loop/AM_CONFIG.md)
- Full mechanism write-ups (math, sanity checks, caveats): [`research_loop/state/mechanism_registry.md`](../../../research_loop/state/mechanism_registry.md)
- Automated verification of every number below: [`research_loop/REPRODUCE_ALL.sh`](../../../research_loop/REPRODUCE_ALL.sh)

This file is the human-readable map. `REPRODUCE_ALL.sh` is the executable version — it pins a git
snapshot, claims a GPU, and asserts each number. Use that if you want a pass/fail; use this if you
want to understand or modify an arm.

Every command block below was checked against the post-restructure driver: each arm constructs a valid
`AMContinualConfig` with the field values the text claims, and each documented guard raises. That
check is config-construction only — the loss numbers are quoted from the original runs, not re-run.

---

## 1. Setup

### Environment

Nothing in the repo imports without these. There are sibling checkouts on this box, and an unpinned
`import cartridges` has historically resolved to the wrong one.

```bash
cd /localhome/local-triv/gated-continual-cartridges_explore
export CARTRIDGES_DIR=$PWD
export CARTRIDGES_OUTPUT_DIR=$PWD/outputs
export PYTHONPATH="$CARTRIDGES_DIR:${PYTHONPATH:-}"
PY="$CARTRIDGES_DIR/.venv/bin/python"

# Verify from a neutral cwd -- probing from the repo root is a false negative,
# because for `python -c` the cwd is sys.path[0].
(cd /tmp && "$PY" -c "import cartridges, os; print(os.path.dirname(cartridges.__file__))")
```

That must print the `_explore` path.

### Inputs

| Variable | Value |
|---|---|
| `MODEL_NAME` | `Qwen/Qwen3-4B-Instruct-2507` (rotary base 5e6 — see MECH-003) |
| `PHASE1_CACHE_PATH` | `outputs/phase1_selfdistill_qwen512/cache_last.pt` (512-slot dense self-distill) |
| `SYNTH_DATA_PATH` | `data/qasper/train/qwen_qasper_MT_task_8192.parquet` (16 unique documents) |
| `EVAL_QA_PATH` | `data/qasper/eval/qasper_eval_QA.parquet` (retention, n=78) |
| `EVAL_MT_PATH` | `data/qasper/eval/qasper_eval_MT.parquet` (acquisition, n=69) |

Rebuilding Phase 1 from scratch is `core/train_initial.sh` (dense self-distill) or
`core/train_initial_am.sh` / `core/train_initial_am_compaction.sh` (AM-based). Background TF-IDF
statistics for `USE_IDF=1` come from `infra/collect_bg_stats.py`. None of the experiments below need
either rebuilt — they all start from the committed Phase-1 checkpoint.

### The canonical operating point

Every arm below is this, plus one changed variable.

```bash
canon() {
  export PHASE1_CACHE_PATH=outputs/phase1_selfdistill_qwen512/cache_last.pt
  export SYNTH_DATA_PATH=data/qasper/train/qwen_qasper_MT_task_8192.parquet
  export EVAL_QA_PATH=data/qasper/eval/qasper_eval_QA.parquet
  export EVAL_MT_PATH=data/qasper/eval/qasper_eval_MT.parquet
  export TOP_T=32 GRANULARITY=per_layer SLOT_SELECTION=tfidf USE_IDF=0
  export KEY_MODE=freeze ENABLE_BETA=0 MAX_QUERIES_PER_HEAD=64
  export RIDGE_LAMBDA=1e-4 RIDGE_SCALE=spectral DELTA_WEIGHT=1e-2
  unset AM_ROPE_THETA AM_KEY_REPOSITION AM_SEED_OFFSET AM_SAFE_FRACTION \
        AM_SAFE_METRIC AM_ORACLE_WRITE AM_ONPOLICY_LAYERS AM_ONPOLICY_DOCKV \
        AM_SLOT_FISHER_PATH ENABLE_OLD_REFERENCE_GUARD OLD_REF_DATA_PATH
}
```

Note `TOP_T=32` is set explicitly: the driver's own default is 64. The `unset` list matters when you
run arms in a loop from one shell — a leftover export from a previous arm is otherwise silently
inherited.

Run an arm with:

```bash
canon
export RUN_NAME=my_arm
bash examples/qasper2/scripts/core/train_continual_am_sparse.sh
```

The wrapper passes a fixed set of variables explicitly and inherits the rest, so any knob it does not
name still needs to be `export`ed rather than prefixed.

---

## 2. Reference points

Every claim is relative to these. Mixing them up is the fastest way to misread a result.

| Reference | QA (retention) | MT (acquisition) |
|---|---|---|
| ICL full-context ceiling | 1.9734 | 1.8960 |
| Dense finetune @4 epochs — **the bar** | 2.3721 | 1.8725 |
| Dense finetune @10 epochs (overfit) | 2.6991 | 2.2137 |
| Phase-1 cartridge, untouched — **the floor** | 2.2388 | 3.7825 |
| Canonical AM, θ=1e4 (historical default) | 2.1766 | 2.5484 |
| **Best AM point** (MECH-005 keys+reposition, k=12) | **1.9560** | **2.2720** |

**Win condition:** QA ≤ 2.52 **and** MT ≤ 2.02. The best point clears QA by 0.57 and misses MT by
0.25, at every seed tested.

### Two eval conventions — do not mix

- **in-run** — the loss the training job prints itself, in `phase2_summary.json`.
- **standalone** — `eval_forgetting.py` re-run on a saved checkpoint.

These differ by up to 0.0065 on a byte-identical cartridge. `results.csv` contains both.

### Noise floor

DIAG-NOISE measured the **paired** 95% resolution for comparing two arms at **±0.049 MT / ±0.054 QA**
(conservative composite ±0.1055). A delta smaller than that is not a result.

This is *not* a reproduction tolerance. The closed-form solve is deterministic — six nominally
identical runs produced byte-identical caches — so a reproduction should match to ~1e-6.

### The k-curve

With `SAVE_AFTER_EACH_DOCUMENT=1` (the default), each document leaves a snapshot
`cache-after-doc-<k-1>-*.pt` after `k` documents are written. **The best point of a run is usually not
the k=16 endpoint.** The best AM arm bottoms at k=12; the frozen-key control bottoms at k=10. Always
evaluate the curve, not just `cache_last.pt`:

```bash
snap_k() { ls "$1"/cache-after-doc-$(printf '%03d' $(( $2 - 1 )))-*.pt 2>/dev/null | head -1; }
```

### Evaluating a checkpoint

`eval_forgetting.py` prints `Eval loss` and then **hangs forever**. Parse the line, then kill the PID.

```bash
CHECKPOINT_PATH=<ckpt> EVAL_DATA_PATH=<parquet> RUN_NAME=eval_tag \
  "$PY" examples/qasper2/train/eval_forgetting.py > eval.log 2>&1 &
pid=$!
for _ in $(seq 1 600); do grep -qE 'Eval loss' eval.log && break; sleep 1; done
sleep 1
grep -oE 'Eval loss[^0-9]*[0-9.]+' eval.log | tail -1 | grep -oE '[0-9.]+$'
kill -TERM "$pid"; sleep 2; kill -KILL "$pid" 2>/dev/null
```

`REPRODUCE_ALL.sh::eval_ckpt` wraps this.

---

## 3. Techniques implemented

All nine are **opt-in, default off**, and bit-identical to the historical behaviour when unset.

| ID | Technique | Flag(s) | Verdict |
|---|---|---|---|
| MECH-001 | Write-ceiling oracle (copy teacher's own doc values) | `AM_ORACLE_WRITE` | Diagnostic. Ceiling is MT 2.381 — real but not flat |
| MECH-002 | Reference queries per KV head | `MAX_QUERIES_PER_HEAD` | Not a lever. Best MT is the n=64 control |
| MECH-003 | True rotary base in the teacher | `AM_ROPE_THETA` | Real bug, 8.5× more fittable target, CE unmoved |
| MECH-004 | Box-constrained NNLS β (mass matching) | `ENABLE_BETA`, `AM_BETA_BOX`, `AM_NNLS_ITERS`, `AM_NNLS_DRIVER`, `AM_BETA_TARGET` | Mechanism works, hypothesis fails. Both axes worse |
| MECH-005 | RoPE counter-rotation on key install | `KEY_MODE`, `AM_KEY_REPOSITION` | ✅ **The one confirmed positive** |
| MECH-006 | On-policy layer-sequential re-extraction | `AM_ONPOLICY_LAYERS`, `AM_ONPOLICY_DOCKV` | Null, at 3.6–4.4× the cost |
| MECH-007 | Per-document reference-draw seed offset | `AM_SEED_OFFSET` | ✅ Infrastructure. Made seed variation possible at all |
| MECH-008 | Information-theoretic slot selection | `SLOT_SELECTION=redundancy\|fisher\|mass_x_redundancy` | ❌ Dead for acquisition, with a measured cause |
| MECH-009 | Mass-ranked within a safety constraint | `SLOT_SELECTION=constrained_mass`, `AM_SAFE_FRACTION`, `AM_SAFE_METRIC` | ❌ Dead. Controlled dose-response closes the family |

### Guards the driver enforces

These raise at config-build time rather than running something that does not mean what it says:

- `AM_KEY_REPOSITION=1` requires `KEY_MODE != freeze` (no key is ever installed otherwise).
- `AM_KEY_REPOSITION=1` requires an explicit `AM_ROPE_THETA` — counter-rotating at the historical
  θ=1e4 is measurably **worse than not correcting at all**.
- `AM_ONPOLICY_DOCKV=1` requires `AM_ONPOLICY_LAYERS > 0`.
- `SLOT_SELECTION=fisher` (or `constrained_mass` with `AM_SAFE_METRIC=fisher`) requires
  `AM_SLOT_FISHER_PATH`.
- `ENABLE_OLD_REFERENCE_GUARD=1` requires `OLD_REF_DATA_PATH`.

---

## 4. Experiment groups

### Group A — the write ceiling

**ORACLE-WRITE (MECH-001).** Bounds the entire value-only family in one run: skip the solve and copy
the teacher's own document value vectors into the selected slots. If a *perfect* value write cannot
reach the bar, no better solve can.

```bash
canon
export AM_ORACLE_WRITE=1 RUN_NAME=ORACLE-WRITE
bash examples/qasper2/scripts/core/train_continual_am_sparse.sh
```

Expected [standalone]: **QA 1.8955 / MT 2.3810** vs the flag-off control QA 2.1772 / MT 2.5524.

**Finding.** Perfect content in the tfidf-selected top-32 closes only ~25% of the MT gap to the 1.873
bar. The paired routing measurement says why: the rewritten slots carry ~9% of total eval-time
attention mass, and MT queries route to them no more than QA queries do (ratio 1.04). The write is
readable through a narrow channel, not unreadable.

Also relevant: **DIAG-OVERWRITE** measured that the 16 documents collide badly — mean pairwise Jaccard
0.713 between their per-layer top-32 sets, ~9.8 writes per slot, and **mean 4.7% survival for
documents 1–15**. The ceiling above is therefore mostly measuring the last few documents.

---

### Group B — the three bugs

**DIAG-ROPE (MECH-003).** The AM package hard-coded rotary base 10000.0 while Qwen3-4B-Instruct-2507
uses 5e6. Composing two RoPE rotations only lands on an absolute position when both use the same θ.

```bash
canon; RUN_NAME=ROPE-A                                  # arm A: historical θ=1e4 (still the default)
bash examples/qasper2/scripts/core/train_continual_am_sparse.sh
canon; export AM_ROPE_THETA=5000000; RUN_NAME=ROPE-B    # arm B: the model's own base
bash examples/qasper2/scripts/core/train_continual_am_sparse.sh
```

| arm | QA | MT | mean `am/mean_mse` | max abs value |
|---|---|---|---|---|
| A θ=1e4 | 2.17662 | 2.54836 | 0.11906 | 984.0 |
| B θ=5e6 | 2.15320 | 2.52961 | **0.01406** | **178.0** |

**Finding.** The bug is large in the solve's own units and invisible in CE. The target becomes 8.5×
more fittable and the write 5.5× gentler, while CE moves an order of magnitude inside the noise floor.
Most striking: it dissolves the layer-34/35 residual that an earlier diagnostic had named the binding
constraint (their share of residual 77.8% → 19.2%) — that "unfittable structure" was a rotary-frame
artefact, not a capacity limit.

**MECH-QUERIES (MECH-002).** `max_queries_per_head` was pinned at 64 against the paper's 16k–50k. The
accumulator can supply 57,344–81,920 per KV head per document, so ~99.9% was being discarded.

```bash
for n in 64 256 1024 4096 16384; do
  canon; export MAX_QUERIES_PER_HEAD=$n RUN_NAME="MECH-QUERIES_n$n"
  bash examples/qasper2/scripts/core/train_continual_am_sparse.sh
done
```

Expected MT [standalone]: 2.5524 / 2.7716 / 2.5744 / 2.7114 / 2.6943 — **the best is the n=64
control**, and `|v|max` grows 984 → 7808, away from the teacher's |63|.

**The confound, found by the worker against its own result:** the guarded solve stacks
`[X_new (n×t) ; √w·I (t×t)]`, so the trust region's relative pull decays like `1/n`. Sweeping `n` at
fixed `DELTA_WEIGHT` moves two things. MECH-QUERIES-B corrects it with `w = 1e-2 · n/64`:

```bash
canon; export MAX_QUERIES_PER_HEAD=16384 DELTA_WEIGHT=2.56 RUN_NAME=MECH-QUERIES-B_n16384
bash examples/qasper2/scripts/core/train_continual_am_sparse.sh
```

Expected: QA 2.0263 / MT 2.6084. MT still on the wrong side, but **QA falls 0.21 below the Phase-1
floor** — the only axis that moved.

**MECH-BETA (MECH-004).** β is AM's own mass-matching mechanism and had failed three times before
(non-PD Cholesky, lstsq NaN, NaN inside the fit). Root causes: `torch.linalg.lstsq` with the default
`gels` driver returns NaN on rank-deficient CUDA input **without raising**; no upper bound existed, so
β reached −27.6 against the paper's box [−3, 3]; and `_should_fit_beta` silently disabled β whenever
`KEY_MODE=freeze`, so it had never actually run.

```bash
canon
export AM_ROPE_THETA=5000000 ENABLE_BETA=1 \
       AM_BETA_BOX=3.0 AM_NNLS_ITERS=2 AM_NNLS_DRIVER=gelsd
export RUN_NAME=MECH-BETA_on
bash examples/qasper2/scripts/core/train_continual_am_sparse.sh
```

Expected: **QA 2.73637 / MT 2.81193** vs the β-off rope arm QA 2.15320 / MT 2.52961.

**Finding.** The mechanism works and the hypothesis it serves does not. 4608 fits, zero NaN, every β
inside the box; eval-time `mass_on_S` on MT rises **0.0711 → 0.3010 (4.23×)** on all 36 layers. But
both axes regress. β is **query-independent** — a per-slot constant inside the softmax shifts every
query's logit equally, so it maps MT and QA mass through the same monotone function. The MT/QA ratio
moved only 1.043 → 1.051. It buys bandwidth, never selectivity. 59.7% of entries pin at the ceiling,
so a larger box buys more of the same.

> `AM_NNLS_DRIVER=gelsd` is **CPU-only**; on CUDA input it raises. The implementation retries the warm
> start on CPU (matrices are 64×32, ~1.3× solve cost).

---

### Group C — keys, the one confirmed positive

**MECH-KEYS (MECH-005).** Every prior row in `results.csv` was `KEY_MODE=freeze`; "key rewriting
collapses QA" was pre-loop, Llama-era folklore that had never been tested here. The hazard: a document
key prefilled at position `512+m` gets installed into a cartridge slot with no counter-rotation, so
the logit that *selected* it is not the logit it *delivers*. Since RoPE is orthogonal, installing
`R_{−Δ}k_doc` restores it exactly.

```bash
canon
export AM_ROPE_THETA=5000000 KEY_MODE=highest_attention AM_KEY_REPOSITION=1 ENABLE_BETA=0
export RUN_NAME=MECH-KEYS_repos
bash examples/qasper2/scripts/core/train_continual_am_sparse.sh
```

| arm | QA | MT | `mass_on_S` MT | doc keys installed /32 | solve s |
|---|---|---|---|---|---|
| control `freeze` | 2.15972 | 2.52963 | 0.0711 | — | 182 |
| `highest_attention`, repos 0 | 2.12011 | 2.42385 | 0.3756 | 8.91 | 325 |
| **`highest_attention`, repos 1** | **2.03492** | **2.33050** | 0.2430 | 12.91 | 272 |
| `omp`, repos 1 | 2.11686 | 2.39698 | 0.1918 | 17.86 | **3857** |

At **k=12** (the run's actual optimum, not the k=16 endpoint): **QA 1.9560 / MT 2.2720**.

**Finding.** The folklore is false here — every key arm beat the frozen control on *both* axes. The
counter-rotation is worth −0.093 MT / −0.085 QA on top of the uncorrected key write, and unit-level it
drops the installed-vs-intended logit error from 1.054 to 4.8e-07.

**But it does not work through selectivity.** The MT/QA mass ratio moved 1.0428 → 1.0510, still below
the untouched Phase-1 cartridge's own 1.0812. What moved is bandwidth — and unlike β, the key write
buys its bandwidth by moving the keys *into the document's directions*, so the extra mass lands on
slots whose content is the document.

OMP installs the most document keys and gets the highest ratio, yet lands worse on both axes at **21×
the solve cost**, because each of its 32 greedy steps runs a 200-iteration NNLS inside.

**MECH-SEED (MECH-007).** Before this flag, seed variation was *impossible*: the per-document draw was
seeded with the constant `doc_idx`, and `config.seed` never reached it. Every number in the project was
one draw. The offset replaces ~30 of each document's 32 reference conversations (Jaccard 0.036–0.043;
0 of 16 documents keep their draw).

```bash
for off in 0 1000 2000; do
  canon
  export AM_ROPE_THETA=5000000 KEY_MODE=highest_attention AM_KEY_REPOSITION=1
  export AM_SEED_OFFSET=$off RUN_NAME="MECH-SEED_off$off"
  bash examples/qasper2/scripts/core/train_continual_am_sparse.sh
done
```

Expected k=12 MT: 2.27202 / 2.26954 / 2.26263 — across-seed range **0.0094**.

**Finding.** The seed spread is 22× smaller than the keys-vs-control effect it was meant to test, so
the MT advantage is not a seed artefact: ΔMT is −0.198 / −0.179 / −0.254 at k=12, all six cells
(k=12 and k=16) the same sign and outside the band.

⚠️ **The QA half of the headline is refuted.** ΔQA never leaves the noise band at k=12. And measured
against the control's *own* optimum (k=10, MT 2.4083) rather than matched-k, ΔMT shrinks to −0.1363.
The seed-confirmed claim is "keys beat frozen keys at matched k"; the own-optimum claim is weaker and
was never seed-varied.

---

### Group D — the gating hypothesis (three ways of asking, all negative)

**MECH-INFOGATE (MECH-008).** Attention mass is not importance — Spearman(`tf_mass_qa`, `fisher`) =
0.579 — so ranking slots by an information-theoretic criterion was not bounded a priori.

```bash
for arm in "tfidf 32" "redundancy 32" "redundancy 64" "redundancy 128" \
           "mass_x_redundancy 32" "fisher 32"; do
  set -- $arm
  canon
  export AM_ROPE_THETA=5000000 KEY_MODE=highest_attention AM_KEY_REPOSITION=1
  export SLOT_SELECTION=$1 TOP_T=$2 RUN_NAME="MECH-INFOGATE_$1_t$2"
  bash examples/qasper2/scripts/core/train_continual_am_sparse.sh
done
```

| arm | best MT | ΔMT | realised MT routing mass |
|---|---|---|---|
| control `tfidf` t32 | **2.2720** | — | **0.2799** |
| `redundancy` t32 | 2.4386 | +0.167 | 0.0662 |
| `redundancy` t128 | 2.4582 | +0.186 | 0.2146 |
| `mass_x_redundancy` t32 | 2.3789 | +0.107 | 0.1251 |
| `fisher` t32 | 2.6692 | +0.397 | 0.0104 |

**Finding.** Every gate is worse on MT by 2.2×–8.1× the resolution, and **4× the budget does not fix
it** — redundancy at t=128 still captures less MT routing mass than the incumbent at t=32. Across the
arms, `log(realised MT routing mass)` predicts best MT at Pearson **−0.877**. The incumbent maximises
MT routing mass by construction, so any importance-aware selector captures less of it and necessarily
loses. `fisher` is the anti-alignment end to end: best retention ever measured here (QA 1.8641) for
+0.397 MT.

`kl_loo` was deliberately **not** implemented: the exact leave-one-out KL is `−log(1−w_j)`, monotone
in the slot's own attention weight, so ranking by it *is* ranking by attention mass (ρ = 0.968).

> ⚠️ Later audit: `redundancy` uses the wrong Gram. The correct damage-of-overwriting is
> `‖v_j‖²/(H⁻¹)_jj` with `H = E_q[aaᵀ]` (the routing second moment). Ours used `G = VVᵀ` in value
> space, which is query-independent and therefore cannot be a QA-importance metric at all. Fixing it
> deepens the anti-alignment. See `notes/2026-07-29-lit-rev/04-*`.

**MECH-BUDGET.** Isolates "bigger budget" from "smarter selection", which MECH-INFOGATE confounded.

```bash
for pair in "128 512 1e-2" "64 512 1e-2" "256 1024 1e-2"; do
  set -- $pair
  canon
  export AM_ROPE_THETA=5000000 KEY_MODE=highest_attention AM_KEY_REPOSITION=1
  export TOP_T=$1 MAX_QUERIES_PER_HEAD=$2 DELTA_WEIGHT=$3
  export RUN_NAME="MECH-BUDGET-B_t$1_q$2"
  bash examples/qasper2/scripts/core/train_continual_am_sparse.sh
done
```

**Finding.** At matched `n=512` and matched `w`, t128 vs t64 gives ΔMT +0.035 / +0.028 / −0.023 /
−0.019 at k=8/10/12/16 — all inside the ±0.049 resolution and **not sign-consistent**. Support is not
a lever. The control fires: at t64, already determined at n=64, raising `n` alone makes both axes
worse at every k, so only matched-n contrasts isolate support.

**MECH-CONSTRAINED (MECH-009).** The strongest form of the hypothesis — mass-ranked *within* a safety
constraint, which is the selector the original projection actually described. `q=1.0` is provably
identical to `attention_mass`.

```bash
for pair in "0.75 32" "0.50 32" "0.25 32"; do
  set -- $pair
  canon
  export AM_ROPE_THETA=5000000 KEY_MODE=highest_attention AM_KEY_REPOSITION=1
  export SLOT_SELECTION=constrained_mass AM_SAFE_FRACTION=$1 AM_SAFE_METRIC=redundancy TOP_T=$2
  export RUN_NAME="MECH-CONSTRAINED_q$1_t$2"
  bash examples/qasper2/scripts/core/train_continual_am_sparse.sh
done
```

| q | best MT | realised MT routing mass |
|---|---|---|
| 1.00 (= incumbent) | **2.2720** | **0.2799** |
| 0.75 | 2.3288 | 0.1661 |
| 0.50 | 2.3693 | 0.1327 |
| 0.25 | 2.4388 | 0.1000 |

**Finding.** With `TOP_T` fixed and only the constraint strength moving, MT degrades **monotonically**
and tracks the realised routing mass monotonically. Pearson −0.9775 within the sweep, −0.8681 pooled
over ten arms. The curve's optimum is `q=1.0`, which *is* the incumbent — there is no untried `q`
where this might have worked. Even the lightest touch is expensive: excluding only the most
QA-critical quarter already costs 41% of the incumbent's MT bandwidth, because MT-wanted slots
concentrate in exactly the least-redundant part of the cartridge.

**Second finding worth carrying forward:** offline doc-0 projections are systematically ~0.6×
optimistic (realised/predicted 0.749 / 0.598 / 0.567 / 0.585 / 0.612), because redundancy is recomputed
on the drifting live cache while the written union grows.

**VERIFY-GATE.** Seed-varies the negative to the same standard as the positive.

```bash
for off in 0 1000 2000; do
  for sel in redundancy tfidf; do
    canon
    export AM_ROPE_THETA=5000000 KEY_MODE=highest_attention AM_KEY_REPOSITION=1
    export SLOT_SELECTION=$sel AM_SEED_OFFSET=$off RUN_NAME="VERIFY-GATE_${sel}_off$off"
    bash examples/qasper2/scripts/core/train_continual_am_sparse.sh
  done
done
```

ΔMT at each arm's own per-seed argmin: +0.167 / +0.163 / +0.200. **All 12 matched-k cells clear even
the conservative ±0.1055 composite**, as does the adversarial pairing. Note the asymmetry: this
negative is better established than the project's own positive, whose QA axis never clears.

---

### Group E — on-policy queries

**MECH-SEQUENTIAL (MECH-006).** Queries are collected in one pass over the pre-write cartridge, but
writing layer `l` perturbs the residual stream, so layers `l+1…35` are solved against stale queries.
This re-extracts every N layers.

```bash
canon
export AM_ROPE_THETA=5000000 KEY_MODE=highest_attention AM_KEY_REPOSITION=1
export AM_ONPOLICY_LAYERS=4 RUN_NAME=MECH-SEQUENTIAL_g4
bash examples/qasper2/scripts/core/train_continual_am_sparse.sh
```

**Finding.** The mechanism works, is measured, and buys nothing — and it identifies why. The
cross-layer drift is real but tiny (mean cosine 0.998, rel-L2 4.98%) and **does not widen with depth**,
directly contradicting the paper's prediction — it is largest at the *shallowest* boundary. The
pathology the entry was opened for no longer exists: the "attention collapse" it targeted was a θ=1e4
artefact that MECH-003 already removed. Composed with the best point it is **+0.058 QA / +0.051 MT
worse**, at 3.6–4.4× the solve cost. At k=12, where MECH-005 is actually best, on-policy costs +0.176
MT. There is no k at which it helps.

---

## 5. Hazards

Each of these cost the original investigation real time.

1. **`eval_forgetting.py` hangs after printing.** Parse the line, then kill the PID. Never `wait`.
2. **Unpinned `import cartridges` resolves to a sibling repo** that lacks the newer flags. Pin
   `PYTHONPATH` and verify from `/tmp`.
3. **The import probe is a false negative from the repo root** — for `python -c`, cwd is `sys.path[0]`.
4. **Never `env VAR=val bash ...`** — a `~/.local/bin/env` PATH shim swallows it silently (rc=0,
   no-op). Use `VAR=val bash ...` or `export`.
5. **One job per GPU**, via `flock`. Two training jobs on one device means OOM and garbage timings.
6. **Evaluate the k-curve, not just `cache_last.pt`.** Arms bottom at different k, and comparing one
   arm's optimum against another's endpoint is not a comparison.
7. **`DELTA_WEIGHT` and `MAX_QUERIES_PER_HEAD` are coupled.** Raising `n` at fixed `w` dilutes the
   trust region like `1/n`. Only matched-n contrasts isolate anything else.

---

## 6. What is not reproducible here

- **`target_mode` sweeps.** The field was deleted in the AM package restructure; the per-document path
  always builds the `[cartridge ‖ doc]` teacher, which is what it pretended to select. An unknown env
  var is now simply ignored, so the old 3-arm sweep would agree for the wrong reason. Re-run at a
  pre-restructure commit if you need the original artefact. See `AM_CONFIG.md` "Deleted".
- **The original β failures** (non-PD Cholesky, lstsq NaN, NaN inside the fit). Reproducing them
  requires reverting MECH-004 on a scratch branch. `AM_NNLS_DRIVER=gels` may reproduce the third
  without an edit — try that first.
- **`execution_mode` variants** (`train_loop`, `legacy_decoupled`). Deleted; they were dead in the live
  tree before the restructure.
