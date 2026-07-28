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
_(none yet — first SCOUT dispatch will populate this file)_
