# NORTH STAR — the direction this autoresearch must not drift from

> Orchestrator: re-read this every cycle in STEP 4. Any experiment or design decision that does not
> serve the two success axes below — or wins one axis by badly violating the other — is drift.
> Reject it and say why.

## The story / spirit
A **cartridge** is a corpus compressed into a numerical state (a fixed-size KV cache) that should
behave like **ICL** over that corpus, but cheap at inference. This project extends cartridges to
**continual learning**: when a new domain arrives, update the cartridge in place.

**The thesis (reframed 2026-07-28): the closed-form, backprop-free update must be the winning
mechanism.** Attention Matching is not merely a fast substrate we tolerate — it is the claim. A
**gradient-free** in-place write, with a purpose-built **gating** mechanism deciding which slots get
written, should match a gradient-trained self-distilled cartridge on continual-learning quality.
Sparse gradient steps stay on the plot as a **costed reference point**; they are not the answer.

## Two success axes (BOTH matter)
1. **Training efficiency** — match cartridge-level quality at a fraction of its train + synthesis cost.
   **`gradient_steps = 0`** is the headline version of this axis.
2. **Continual-learning quality** — forgetting (QA) and acquisition (MT) at or below the dense
   cartridge, reaching toward the ICL ceiling.

Everything is judged as a **Pareto trade**: (training cost) × (forgetting, acquisition). A config
that improves loss but blows up runtime (giant reference banks, dozens of re-solves, gradient epochs)
is a **dominated point** — record it, don't chase it.

## How we are allowed to get there (the method constraint)
This loop **diagnoses, then imports** — it does not tune. See `PLAN_AM_MUST_WIN.md` and `DESIGN.md`.
- A cause must be **measured** before a fix is built for it.
- A fix should come from **outside this repo** when the literature already solves that cause; cite it.
- A knob sweep is legitimate **only** as a diagnostic that discriminates named causes.
- "We ran out of knobs" is never an ending; it is the start of the escalation ladder (MISSION §6).

**Gating remains where the novelty budget goes** — but "gating" now includes the *write rule itself*
(what target, which keys, which projection, how many closed-form rounds), not just slot selection.
Designing and implementing new mechanisms on-branch is expected, not exceptional.

Inherit cartridge's substrate choices (self-study data, text init, self-distillation objective) —
those are the ground, not variables to overturn.

## Comparators / reference lines (measured, on the `eval_forgetting.py` ruler)
- **Cartridge dense @4ep = THE BAR**: QA 2.3721 / MT 1.8725 (256 steps). (dense @10ep overfits: 2.699 / 2.214.)
- **ICL / full-context = the aspirational CEILING**: QA 1.9734 / MT 1.8960 — measured on a *different*
  harness and context source, so re-ruler it once on the eval parquets before quoting it as the ceiling.
- **Sparse-gradient reference**: QA 1.6169 / MT 1.9664 at 62 steps — the cost point we must beat *at zero gradients*.
- **Phase-1 start**: QA 2.2388 (retention floor) / MT 3.7826 (untrained acquisition).
- **Best gradient-free so far**: AM top32, QA 2.1766 / MT 2.5484 — the line to move.

## Efficiency measurement (3 tiers; synthesis relaxed)
- **T1 solve:** closed-form value-solve wall-clock.
- **T2 end-to-end Phase-2:** ref-query collection + bg_stats + solve + eval (GPU-seconds + wall-clock).
- **T3 vs cartridge full pipeline:** T2 vs the cartridge's Phase-2 gradient train time **plus** a
  synthesis-cost term. **Do NOT re-synthesize to fill blanks** — use logs/wandb, else cite the paper's
  reported cost and mark it estimated. The headline story is T3, honestly caveated.
Every GPU worker records T1 + T2 into `results.csv`; T3 is assembled by the orchestrator.

## Design constraints (from the human)
- **Gradient-free is the requirement, not a preference.** `gradient_steps = 0` for any claimed winner.
- **GPUs are free to use** (2× GH200). Keep them busy; one job per GPU.
- **Every GPU run is logged to wandb**, cleanly named and grouped (RUNBOOK §0b). No wandb URL ⇒ the
  result is invalid.
- **The investigation is a deliverable**: the bottleneck board, the literature ledger, and the
  mechanism registry are maintained as carefully as the numbers.
- Full autonomy on branch `trivo-explore-research-work`; may commit there; **never** touch other branches.

## Scope
- This session: **QA → MT (2-stage)**, Qwen3-4B, 512-slot cartridge, perplexity/mean-CE signal.
- Future (out of scope unless a winning recipe emerges): **QA → MT → SA** longer chain. Design
  mechanisms so they *could* extend to N stages; validate on 2.

## Literature (read these first, then go further)
- **`AM.pdf`** (repo root) — *Fast KV Compaction via Attention Matching*, arXiv 2602.16284: the exact
  teacher/target and closed-form value solve we are trying to make win. Read the algorithm, including β.
- **`TF-IDF.pdf`** (repo root) — *Continual Learning via Sparse Memory Finetuning*, arXiv 2510.15103v1:
  the gating idea we build on, and what it must improve on in *our* setting (a compressed embedding
  inside attention, not a sparse FFN memory).
- **Cartridges / self-study long-context KV** (arXiv 2506.06266) — the quality bar + synthesis costs.
- **Adjacent families the loop is expected to mine** (MISSION §5): closed-form model editing
  (ROME/MEMIT), null-space-projected editing (AlphaEdit, Adam-NSCL), delta-rule / fast-weight
  associative writes (DeltaNet, FWP), recursive least squares / Kalman updates, OMP / matching pursuit,
  KV-cache salience (H2O, SnapKV, PyramidKV, Scissorhands).
