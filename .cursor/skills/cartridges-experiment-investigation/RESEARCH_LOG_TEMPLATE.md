# Investigation: <dataset> continual sparse — <one-line concern>

Single-source-of-truth research log. Update after every probe with findings,
evidence pointers, and the next priority. Keep `Headline` current so the next
agent (or future-you) can resume quickly.

---

## Headline (latest)

<one-paragraph TL;DR — dominant cause + a single number from a credible probe + the
secondary effects that look real but subordinate>

---

## Constraints & user direction (from initial Q&A)

- **Scope**: <which surprises to characterize>
- **Metrics**: <ppl / MCQ / both>
- **Dense baseline**: <available / missing / TBD>
- `NUM_TOKENS`: <value, intentional?>
- **Compute**: <analysis-only / few short runs / full sweep>

---

## Key facts about the pipeline (verified)

- Phase-1 KV cache initialized from `<context-file>` (size, contents).
- Phase 1 trains on `<train-parquet>` (rows, granularity).
- Phase 2 starts from Phase-1 `cache_last.pt` via `KVFromLocal`, trains on
  `<phase-2-parquet>`.
- Sparse fine-tuning settings: `top_t=<>`, `granularity=<>`,
  `momentum_masking=<>`, `freeze_keys=<>`. Note: `FREEZE_KEYS=0` in the shell
  means keys ARE updated (see SKILL.md pitfall #2).
- LR scheduler: `CosWithWarmup(max_steps=<>, warmup_steps=<>, alpha_f=<>)`,
  peak LR `<>`. Floor = `peak * alpha_f`. Compare `max_steps` to actual
  `train/optimizer_step` — mismatch = LR floor for tail steps.
- Eval set sizes: `<rows>` rows × `<packed_seq_length>` packed → `<batches>`
  batches.

---

## Hypotheses (ranked by explanatory power × cheap-to-test)

(Start from SKILL.md's H-menu; prune to the user's specific concern.)

| H | Hypothesis | Cheap probe | Status |
|---|---|---|---|
| H1 | Phase-1 undertrained | mine wandb history | pending |
| H1' | Phase-1 overtrained | look for U-shape in eval-ppl | pending |
| H2 | Eval too easy / not patient-specific | tokenize eval, structural-vs-unique | pending |
| H3 | Two phases too similar | reference dataset comparison | pending |
| H4 | Cache init pre-encodes target | tokenize cache-init source | pending |
| H5 | Eval set too small | bootstrap CI / token count | pending |
| H7 | Granularity ranking = budget | re-read mask code | pending |
| H9 | LR scheduler horizon mismatch | config + final step | pending |
| H10 | Saved ckpt is last not best | check intermediate ckpts | pending |

---

## Open questions for the user

1. <decision points that block more probes>
2. <ambiguous intent on metrics or recipe>
3. <missing baseline that would unblock interpretation>

---

## Probes log

Format per entry:

```
### YYYY-MM-DD HH:MM  P{n}  <one-line probe name>
- Hypothesis: H{n}
- Setup: <files read / sub-agent dispatched / commands run>
- Evidence: <numbers, file:line, screenshots>
- Conclusion: confirms / refutes / partial / inconclusive
- Next: <follow-up>
```

---

## Final synthesis

(Promote into Headline once sufficiently confident.)

### Dominant cause

### Secondary effects

### What you can do about it

1. **Eval-only** (no training): …
2. **Training adjustment**: …
3. **Dataset-level change**: …

---

## Document map

- `notes/00_research_workflow.md` — this file
- `notes/01_trajectories.md` — per-run trajectories from wandb
- `notes/02_data_probe.md` — train+eval parquet characterization
- `notes/03_sparse_dynamics.md` — slot churn / IDF / cache-delta
- `notes/04_<reference>_vs_<target>.md` — reference comparison
- `results/trajectories.csv` — wandb-sourced per-step
- `results/wandb_*_summaries.json`, `wandb_*_histories.json` — raw wandb dumps
- `results/<reference>_vs_<target>.png` — the closing image
- `scripts/fetch_wandb_data.py`, `build_trajectories_csv.py`, `compare_recipes.py`
