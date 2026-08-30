# Phase-1 RoPE fix, then 5×5 continual AM on QASPER and QuALITY

- **Date:** 2026-08-27
- **Status:** done
- **Related:** [am-diagnosis-cycle1](./2026-07-28-am-diagnosis-cycle1.md) (MECH-003 rotary-base bug), [am-investigation-synthesis](./2026-07-29-am-investigation-synthesis.md), `notes/experiments/qasper_am_experiments.md`

## Goal

Confirm that QASPER Phase-1 AM compaction prefills every document from position 0 and then rebakes keys at the wrong rotary base (`rope_theta=1e4` vs the model's `5e6`). Measure whether correcting that improves log-perplexity and generation. Then chain the best Phase-1 cartridge through p02–p05 on QASPER and QuALITY and read the 5×5 stage-by-eval grid.

## Setup

- **Branch / commit:** `b7dc1be` on `trivo-explore-research-work`
- **Model:** Qwen3-4B-Instruct-2507, 512-slot cartridge
- **Hardware:** NVIDIA GH200 480GB (MIG disabled; ~95 GiB)
- **Phase-1 sweep:** `examples/qasper/pipelines/sweep_initial_am_rope.sh`
- **Continual chain:** `examples/shared/am/continual_chain.py` (recipe read from `outputs/caches/qasper/p03/delta_ha_b0_idf0/run/config.yaml`: `highest_attention` keys, `key_reposition=true`, beta off, `top_t=32`, `delta_weight=0.01`, `rope_theta=5e6`)
- **Metric:** teacher-forced mean token loss (natural log) on `data/<dataset>/phases/phase{k}_eval.parquet`

## What I tried

- Six-arm Phase-1 positional sweep on QASPER QA, holding everything else fixed. Arms: shipped (`rebake@1e4`, beta on); same without beta; no rebake; rebake at `5e6`; rebake at `5e6` plus unique global teacher positions; rebake at `5e6` with beta on.
- Continual AM p02–p05 from the best Phase-1 cache (`D_b0_rebake5e6_globalpos`) on QASPER.
- Same recipe on QuALITY: a fresh arm-D Phase-1 compaction of quality p1, then continual p02–p05.
- Slot-coverage and task-specificity analyses on both 5×5 grids.

## Key findings / insights

### 1. Shipped Phase-1 rebake is worse than doing nothing

QASPER Phase-1, 16 papers, 99,792 teacher tokens. Loss is teacher-forced nats on the QA / MT eval parquets (byte-identical to `phases/phase{1,2}_eval.parquet`).

| arm | rebake | theta | global pos | beta | QA loss | MT loss | QA ppl |
|---|---|---|---|---|---|---|---|
| **S shipped** | yes | 1e4 | no | on | **14.65** | **14.62** | 2.3e6 |
| A | yes | 1e4 | no | off | 14.54 | 14.52 | 2.1e6 |
| B no rebake | no | — | no | off | 8.99 | 9.16 | 8.0e3 |
| E | yes | 5e6 | no | on | 6.25 | 6.58 | 520 |
| C | yes | 5e6 | no | off | 5.46 | 5.58 | 236 |
| **D (best)** | yes | 5e6 | **yes** | off | **4.07** | **4.10** | **58.8** |

The shipped default is ~3.6 nats worse than leaving keys at their teacher positions, and ~10.6 nats worse than rebaking at the model's own base with unique positions. `recon_mse` does not rank the arms (shipped 0.0063 vs D 0.0088) and is not a useful diagnostic. Beta hurts at both bases (S vs A, E vs C).

Generation on the shipped arm was degenerate looping (`the the the…`); arm D produced fluent, question-conditioned answers. Qualitative dump: `outputs/p1_rope_sweep/generations.json`.

Historical self-distilled QASPER p01 is still far ahead of even the fixed AM p01 (QA ~2.07 vs 4.07). The RoPE fix is necessary, not sufficient.

### 2. Fixed p01 → continual 5×5 on QASPER

Lineage `delta_ha_b0_idf0_ropefix_p01D`. Rows are the cartridge after that stage; columns are held-out evals.

```
stage        QA       MT       SA      ASR       KG
p01       4.075    4.095    4.182    3.738    3.941
p02       4.221    4.034    4.204    3.923    3.990
p03       3.870    3.715    3.848    3.639    3.631
p04       3.895    3.761    3.900    3.601    3.712
p05       3.888    3.848    3.936    3.677    3.706
```

Mean forgetting at p05: **−0.052**. Against the historical self-distilled p01 + same continual recipe, p03 QA/MT/SA is still ~1.4–1.8 nats worse (3.870 / 3.715 / 3.848 vs 2.065 / 2.331 / 2.317).

### 3. The same recipe on QuALITY, with a fresh arm-D p01

Phase-1 compaction of 7 stories, 55,090 teacher tokens, `rope_theta_matches_model=True`, global positions. p01 eval (from the compaction summary; the chain's p01 row agrees to ~0.002):

| p1 | p2 | p3 | p4 | p5 |
|---|---|---|---|---|
| 2.323 | 2.926 | 2.576 | 2.153 | 3.162 |

Then the 5×5:

```
stage         P1       P2       P3       P4       P5
p01        2.321    2.925    2.576    2.153    3.161
p02        2.122    2.678    2.296    1.989    2.863
p03        1.843    2.431    2.044    1.771    2.596
p04        1.691    2.289    1.895    1.649    2.417
p05        1.660    2.224    1.848    1.617    2.344
```

Every column descends at every stage. Mean forgetting: **−0.336**. P5's loss falls 3.161 → 2.344 *before* P5 documents are written.

### 4. The writes are not task-specific; TF-IDF lands on ~32 slots

Per-layer unique slots touched (`top_t=32`, 512 slots):

| dataset | p02 unique | p03 | p04 | p05 | cumulative coverage |
|---|---|---|---|---|---|
| QASPER (16 docs) | 53.5 | 34.8 | 33.8 | 34.2 | 56.8 / 512 (**11.1%**) |
| QuALITY (7 docs) | 49.9 | 33.7 | 32.5 | 32.8 | 52.7 / 512 (**10.3%**) |

`max rewrites == n_docs` on both: some slots are written by every document. Mean specificity (target-task Δ minus mean non-target Δ): QASPER **−0.083**, QuALITY **−0.002**. The descending grid is a generic cartridge improvement, not acquisition of the stage's documents. Negative forgetting is mostly "we never wrote 89% of the slots."

## Gotchas / surprises

- Phase-1 rebake at the wrong theta is **actively harmful**, not a no-op. Continual AM already used `rope_theta=5e6`; only initial compaction was broken.
- `initial_compaction.py` used `float(AM_ROPE_THETA)` while `continual_write.py` accepted `"model"`. Passing `AM_ROPE_THETA=model` to Phase 1 would crash; now both accept it. Unset still defaults to `10000.0` for reproducibility of old runs.
- `examples/quality/pipelines/run_5phase_am.sh` as shipped used the *broken* Phase-1 defaults (`ENABLE_BETA=1`, no `AM_ROPE_THETA`) and pointed at a non-existent `.venv/bin/python`.
- `outputs/caches/index.json` lists five QuALITY `am-canonical-512` caches that **are not on this disk**. That index was produced by `examples/maintenance/cache/group_compacted_caches.py` from `outputs/quality_5phase_state/p{1..5}.json` on some other checkout. Those state files are also absent here.
- Dynamo cache-size and compiled block-mask were required to run Phase-1 compaction without OOM / eager fallback. Generation decode needed eager FlexAttention.

## Artifacts

- Phase-1 sweep: `outputs/p1_rope_sweep/` (`summary.json` per arm; collector `examples/shared/evaluate/collect_p1_rope_sweep.py`)
- QASPER 5×5: `outputs/evaluations/qasper/delta_ha_b0_idf0_ropefix_p01D/teacher-forced-logppl-v1/`
- QuALITY 5×5: `outputs/evaluations/quality/delta_ha_b0_idf0_ropefix_p01D/teacher-forced-logppl-v1/`
- QuALITY p01 cache: `outputs/quality_p1_ropefix/2026-08-27-10-22-35-initial_am_compaction/8b6a58e2-452b-440d-9160-fbb13b9e1dbd/`
- Drivers: `examples/shared/am/continual_chain.py`, `examples/shared/am/continual_env.py`, `examples/qasper/pipelines/run_qasper_5x5_continual.sh`, `examples/quality/pipelines/run_quality_5x5_continual.sh`
- Analysis: `examples/shared/evaluate/am_slot_coverage.py`, `examples/shared/evaluate/continual_specificity.py`

## Open questions / next steps

- [ ] Run the *shipped* Phase-1 recipe on QuALITY (rebake@1e4, beta on) so the S-vs-D comparison exists off QASPER. **Not on disk today.**
- [ ] Make slot selection document-dependent, or raise `top_t`, and re-measure specificity.
- [ ] Compare fixed AM p01 to a QuALITY self-distilled p01 (`examples/quality/pipelines/train_initial_selfdistill.sh`); that run dir is also absent here.
