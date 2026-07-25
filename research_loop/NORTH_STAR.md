# NORTH STAR — the direction this autoresearch must not drift from

> Orchestrator: re-read this every cycle in STEP 4. Any experiment or design decision that does not
> serve one of the two success axes below — or that wins one axis by badly violating the other —
> is drift. Reject it and say why.

## The story / spirit
A **cartridge** is a corpus compressed into a numerical state (a fixed-size KV cache) that should
behave like **ICL** over that corpus, but cheap at inference. This project extends cartridges to
**continual learning**: when a new domain arrives, update the cartridge in place. The **core novelty
is the GATING mechanism** — which slots to update — purpose-built for our setting (a compressed
embedding living inside the attention module), inspired by TF-IDF slot selection from *Continual
Learning via Sparse Memory Finetuning* but not bound to it.

We now also adopt a **fast / (near) training-free** update via **Attention Matching** (*Fast KV
Compaction via Attention Matching*) to kill cartridge's biggest cost: long self-study training +
data synthesis. AM is the **fast substrate**; the gater is the **novelty**; cartridge is the **quality
bar**; ICL is the **ceiling**.

## Two success axes (BOTH matter — this is the deliberate "greedy" bet)
1. **Training efficiency** (AM contribution): match cartridge-level quality at a fraction of its
   train + synthesis cost.
2. **Continual-learning quality** (gating contribution): better forgetting/acquisition than the
   naive TF-IDF baseline, approaching cartridge and ideally ICL.

Everything is judged as a **Pareto trade**: (training cost) vs (forgetting-loss, acquisition-loss).
Target region = at/below the cartridge point on quality, far to the cheap side on cost, reaching
toward ICL. **A config that improves ppl but blows up runtime (giant reference banks, many
re-solves, gradient epochs) is drift, not progress** — record it as a dominated point, don't chase it.

## Comparators / reference lines
- **Cartridge (self-distillation)** = the quality BAR to match. (dense `baseline_continual.py`, the
  human-provided HF Phase-1 cache → dense Phase-2.)
- **ICL / full-context** = the aspirational CEILING. Measure once (icl_eval / fullctx benchmark) on
  QA+MT. Don't re-measure every cycle.
- **Naive TF-IDF sparse** = the prior-gating baseline we aim to BEAT on quality at equal/lower cost.

## Efficiency measurement (3 tiers; synthesis relaxed)
- **T1 solve:** the closed-form value-solve wall-clock.
- **T2 end-to-end Phase-2:** ref-query collection + bg_stats + solve + eval (GPU-seconds + wall-clock).
- **T3 vs cartridge full pipeline:** T2 vs cartridge's Phase-2 (10-epoch gradient) train time, PLUS a
  synthesis-cost term. **Do NOT re-synthesize to fill blanks.** The synth parquets already exist; if
  synthesis time wasn't recorded, use whatever is in logs/wandb, else cite the cartridge paper's
  reported synthesis cost or mark it estimated/qualitative. The headline story is T3, honestly caveated.
Every training executor records T1 + T2 into `results.csv`; T3 is assembled by the orchestrator.

## Design constraints (from the human)
- **Fast is the requirement; gradient-free is strongly preferred.** Closed-form AM is the default.
  A few *sparse* gradient steps are allowed ONLY if they stay far cheaper than cartridge and clearly
  help — logged as a distinct, more-expensive point on the Pareto curve, never as the default.
- **Gating is where the novelty budget goes.** After exhausting existing-knob sweeps
  (`SLOT_SELECTION` etc.), the loop MAY design & implement NEW gating mechanisms on the research
  branch (opt-in flags, sanity-checked, swept, with the design written to notes/). Draw from the two
  cited papers + observed slot/attention-mass evidence. This is expected, not exceptional.
- Inherit cartridge's design choices (self-study data, text init, self-distillation objective) —
  those are the substrate, not variables to overturn.

## Scope
- This session: **QA → MT (2-stage)**, Qwen3-4B, 512-slot cartridge, perplexity signal.
- Future (out of scope unless a winning recipe emerges): **QA → MT → SA** longer chain — the harder,
  more compelling continual story. Design gaters so they *could* extend to N stages, but validate on 2.

## Literature to mine (research executors)
- *Continual Learning via Sparse Memory Finetuning* (the TF-IDF gating inspiration).
- *Fast KV Compaction via Attention Matching* (the fast-update substrate).
- Cartridges: self-study long-context KV (arXiv 2506.06266) — for the quality bar + synthesis-cost figures.
- Adjacent: closed-form model/KV editing (ROME/MEMIT), attention-salience routing, OMP/least-squares.
