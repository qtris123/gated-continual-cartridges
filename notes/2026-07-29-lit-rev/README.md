# Literature review — 2026-07-29 — toward a better gating mechanism

Five parallel scout threads, ≥7 papers each, commissioned after the AM investigation closed every
in-scope mechanism family with a measured cause (see `../2026-07-29-am-investigation-synthesis.md`).

| file | topic |
|---|---|
| `01-compressed-cache-cl.md` | continual learning on compressed caches / fixed-size KV memory |
| `02-sparse-finetuning-cl.md` | sparse finetuning & continual update (parameter/slot isolation) |
| `03-gating-mechanisms.md` | gating & routing mechanisms for CL on compressed KV |
| `04-measuring-interference.md` | algorithms & maths for *measuring* overwriting/interference |
| `05-non-overwriting-intuition.md` | the non-overwriting intuition itself — is it right? |

**Shared grounding:** every entry must map to our code and predict one of *our* measured signals.
The binding constraint is **acquisition** (MT 0.248 short), not retention (0.573 ahead of budget).
