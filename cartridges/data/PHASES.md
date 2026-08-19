# Continual-learning phase plans

Five datasets, each split into five phases, for the ICLR continual-cartridge
experiments. A *phase* is one training stage: the cartridge is trained on phase 1,
then continued onto phase 2, and so on, and forgetting is measured by re-scoring
earlier phases' questions.

Each dataset defines its split in code, with a `build_phases()` that regenerates
the committed assignment from the raw data:

| dataset | module | phase key |
|---|---|---|
| LongHealth | `longhealth/phases.py` | `PHASE_TO_PATIENT_IDS` |
| QASPER | `qasper/resources.py` | `TOPIC_TO_IDS` |
| QuALITY | `quality/resources.py` | `PHASE_TO_ARTICLE_IDS` |
| FinQA | `finqa/resources.py` | `PHASE_TO_COMPANIES` |
| TechQA | `techqa/resources.py` | `PHASE_TO_FILENAMES` (frozen in `phases.json`) |

## Summary

| dataset | unit of a phase | tokens/phase | questions/phase | total tokens | total questions |
|---|---|---|---|---|---|
| LongHealth | 4 patients | 42.6k – 49.5k | 80 | 236,124 | 400 |
| FinQA | 13–15 companies | 46.8k – 47.3k | 116 – 166 | 235,635 | 737 |
| QuALITY | 7 articles | ~54.9k | 122 – 128 | 274,675 | 622 |
| QASPER | 16 papers | 76.8k – 108.9k | 47 – 78 | 460,595 | 304 |
| TechQA | 98–100 technotes | ~151.9k | 116 – 129 | 759,611 | 610 |

Against a 1024-slot cartridge that is a compression ratio of roughly 45x
(LongHealth, FinQA), 54x (QuALITY), 75–106x (QASPER) and 148x (TechQA).

## Evaluation formats

The five datasets do not share an answer format, and the differences matter to
the scorer. Each dataset's `### Sample` section below shows a real question in
its native form.

| dataset | eval format | answer key | grading |
|---|---|---|---|
| LongHealth | 5-way multiple choice | the **option text**, e.g. `"Vincristine"` | exact match |
| QuALITY | 4-way multiple choice | `gold_label`, **1-indexed** into `options` | exact match |
| FinQA | short numeric | `answer` plus an executable `program` | exactly checkable |
| QASPER | free-form, rewritten into `<answer>` tags | prose | LLM judge or log-perplexity |
| TechQA | free-form, from real support tickets | prose | LLM judge or log-perplexity |

Three consequences:

- **The two multiple-choice datasets disagree on what an answer is** — LongHealth
  expects the option's text, QuALITY a 1-based index — so the scorer needs a
  per-dataset adapter rather than one shared path.
- **FinQA is the only one gradable without a judge.** It ships the arithmetic
  program alongside the answer, e.g. `subtract(179, 129) -> 50`.
- **QASPER's eval questions are not the ones QASPER ships.** They are rewritten by
  `qasper/rewrite.py` so each names its paper, which is necessary because the
  model sees a panel of 16 at once. The rewritten parquets exist for QA, MT and
  SA only; ASR and KG still need generating.

QuALITY also carries a `difficult` flag marking questions that speed-readers
answered wrong during its validation pass — a ready-made harder eval subset.

## How the token counts were produced

All counts use the **Qwen3-4B-Instruct-2507 tokenizer** with
`add_special_tokens=False`, measured on each resource's `to_string()` — that is,
the string that actually gets prefilled, including the XML wrappers each
resource adds. Counts measured on raw document text run a few percent lower.

```python
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
len(tok(resource.to_string(), add_special_tokens=False)["input_ids"])
```

Context windows the numbers are compared against, read from the model configs:

| model | `max_position_embeddings` | notes |
|---|---|---|
| Qwen3-4B-Instruct-2507 | **262,144** | `rope_scaling=None` — native, not YaRN-extended |
| Llama-3.2-3B-Instruct | **131,072** | per `experiments/scripts/baselines_common.sh` |

Do not use `tokenizer.model_max_length` for this — Qwen3's tokenizer reports
1,010,000, which is a tokenizer-file artifact, not a model capability.

Two caveats:

- **Llama-3.2-3B numbers are not measured.** That repo is gated and this
  environment gets a 401, so every comparison against Llama's 131,072-token
  window below uses Qwen3 counts as a proxy. Both are ~128k-vocab BPE and
  typically land within a few percent on English prose.
- Documents are **deduplicated before counting**. LongHealth, QuALITY, FinQA and
  TechQA all repeat the document text once per question in their raw form.

## How phases are balanced: LPT

QuALITY, FinQA and TechQA all assign items to phases by **LPT —
longest-processing-time first**, a greedy bin-packing heuristic (Graham, 1969).
Two rules:

1. Sort items **largest first**.
2. Put each item into whichever phase has the **smallest total so far**.

```python
for item in sorted(items, key=size, reverse=True):
    phase = min(phases, key=current_total)
    phase.append(item)
```

Phases need to be even in size or a forgetting curve is unreadable — a phase that
simply had less to learn is not evidence about learning. Items cannot be split
(a 44k-token technote goes wholly into one phase; a company's pages all stay
together), so this is bin packing, which is NP-hard, hence a heuristic.

**Why not just deal them round-robin.** Round-robin is positional — 1, 2, 3, 4,
5, 1, 2, … — and ignores how full each phase already is. Over a size-sorted list
that hands phase 1 the largest remaining item on every pass, and the skew
compounds. Worked example, ten documents (thousands of tokens) into three phases:

    A=44  B=20  C=12  D=9  E=8  F=6  G=5  H=4  I=3  J=2      total 113

    round-robin   phase1 = A+D+G+J = 60    phase2 = B+E+H   = 32    phase3 = C+F+I    = 21   spread 39
    LPT           phase1 = A       = 44    phase2 = B+E+H+I = 35    phase3 = C+D+F+G+J = 34  spread 10

Under LPT, once phase 1 takes A it is never the emptiest again, so D goes to
phase 3 instead of back to phase 1. Sorting largest-first is what makes this
work: the lumpy items get placed while a long tail of small items remains to act
as filler and level the totals.

**Spread** below means the gap between the largest and smallest phase as a
fraction of the average phase, `(max − min) / (total / 5)`. It matters because an
uneven phase confounds the measurement: if phase 1 held 5% more text than phase
5, "phase 5 scored better" could just mean phase 5 had less to memorise into the
same number of slots.

On the real data:

| | item-size ratio | round-robin spread | LPT spread |
|---|---|---|---|
| TechQA (496 documents) | 608x (73 → 44,383) | 26.3% (39,396 tokens) | **0.05%** |
| FinQA (71 companies) | 52.7x (355 → 18,698) | 36.8% (17,306 tokens) | **0.96%** (454 tokens) |
| QuALITY (35 articles) | 1.13x (7,442 → 8,445) | 1.68% (920 tokens) | **0.03%** (19 tokens) |

The pattern is monotone in item-size ratio: the more skewed the items, the worse
round-robin does and the more LPT buys. FinQA's smallest company (ORLY, 355
tokens — a single page) against its largest (ETR, 18,698) is a 52.7x range, and
round-robin leaves phase 1 at 57,119 tokens against phase 5's 39,813.

The gain is largest where item sizes are most skewed, but LPT is never worse, so
it is used everywhere. QuALITY's articles are near-uniform and round-robin
already balanced them to 1.68%; LPT still tightened that 48x, for free.

LPT does far better on the real corpora than on the toy above because TechQA has
496 documents at a median of 737 tokens — abundant fine-grained filler — whereas
ten items give one 44k outlier nothing to balance against.

**Determinism.** The assignments are committed to disk, so both tie-breaks are
pinned: equal-sized items order by filename/ticker ascending, and equal-loaded
phases resolve to the lower phase number. `build_phases()` therefore regenerates
the committed table exactly, which is asserted in the module checks.

**Two limits.** LPT is a heuristic, not optimal — the guarantee is that the
largest bin is within `4/3 − 1/(3m)` of optimal, i.e. ≤1.267× for five phases,
though at 0.05% actual spread there is nothing left to gain. And it balances only
the quantity it is given: both use token count, so question counts even out only
incidentally (FinQA phase 2 has 116 questions against phase 5's 166). Packing on
question count instead is a one-line change to the `lengths` map.

---

## LongHealth — 4 patients per phase

**Source:** `benchmark_v5.json` from the [LongHealth repo](https://github.com/kbressem/LongHealth), fetched by `longhealth/utils.py`.
**Document:** one patient's full record (2–13 clinical notes).
**Split rule:** patients in ID order, four per phase.

| phase | patients | notes | tokens | questions |
|---|---|---|---|---|
| 1 | patient_01–04 | 30 | 49,476 | 80 |
| 2 | patient_05–08 | 27 | 42,609 | 80 |
| 3 | patient_09–12 | 24 | 47,873 | 80 |
| 4 | patient_13–16 | 23 | 46,788 | 80 |
| 5 | patient_17–20 | 29 | 49,378 | 80 |
| **total** | **20** | **133** | **236,124** | **400** |

There are exactly 20 patients, so five phases of four use every patient once
with nothing left over and no selection judgement to make. Four rather than one
per phase came from review: one patient is only ~11k tokens, too small for
compaction to be the point.

This is the only dataset with an **identical question count in every phase**,
which makes it the cleanest place to read forgetting curves.

Diagnoses are heterogeneous *within* each phase — phase 1 alone spans DLBCL,
melanoma, multiple myeloma and pancreatic cancer — so phases are separated by
patient identity, not clinical domain. The question being asked is whether the
cartridge still knows patient_02's record after training through patient_20's.

### Sample

Document — `patient_01` (Anna Sample, DLBCL), note `text_0` of 2:

```
**Dear colleague,** We wish to provide an update regarding Mrs. Anna Sample,
born on 01.01.1970. She was admitted to our clinic from 01/01/2017 to
01/02/2017. **Diagnosis:** Diffuse large B-cell lymphoma of germinal center
type; ID 01/2017 - Ann-Arbor: Stage IV - R-IPI: 2 (LDH, stage) - CNS-IPI: 2 -
Histology: Aggressive B-NHL (DLBCL, NOS); no evidence of t(14;18)
translocation. Ki-67 at 40%. Positive reaction to MUM1, numerous CD68-positive
macrophages. Negative reaction to ALK1 and TdT. …
```

Eval — 5-way multiple choice, answer given as the option text rather than a letter:

```
Q: Which of the following is NOT part of the current treatment regimen for
   Mrs. Sample's DLBCL?
   (a) Polatuzumab vedotin   (b) Rituximab      (c) Cyclophosphamide
   (d) Vincristine           (e) Doxorubicin
correct: Vincristine
```

---

## QASPER — 5 task topics

**Source:** `allenai/qasper`, `train` split (888 papers).
**Document:** one NLP paper (title, abstract, full text by section).
**Split rule:** 16 papers per topic. QA/MT/SA were hand-curated previously; ASR
and KG were selected by `qasper/select_topic_papers.py`.

| phase | topic | papers | tokens | answerable QAs |
|---|---|---|---|---|
| 1 | QA — question answering | 16 | 103,280 | 78 |
| 2 | MT — machine translation | 16 | 76,820 | 69 |
| 3 | SA — sentiment analysis | 16 | 82,159 | 62 |
| 4 | ASR — speech recognition | 16 | 89,441 | 48 |
| 5 | KG — knowledge graphs | 16 | 108,895 | 47 |
| **total** | | **80** | **460,595** | **304** |

Selection for the two new topics is title-weighted: a paper whose *title* names
the task is almost certainly about it, whereas an abstract mention is often
incidental ("we also report ROUGE", "extractive question answering"). Exclude
patterns kill the systematic false-positive families. Among eligible papers,
ranking is by answerable-QA count, since those questions become the eval set via
`qasper/rewrite.py`.

**Phase 5 is knowledge graphs, not summarization as originally planned.**
Embedding each candidate panel and comparing centroids, a summarization panel
sits closer to the other four phases (mean cosine 0.65) than those phases sit to
each other (0.51–0.67), and the curated QA panel already contains a
multi-document summarisation paper (`1909.00542`). Knowledge graphs measure 0.60
with no such overlap. The cost is the smallest eval set of any phase, 47.

Two inherited quirks, both left in place deliberately:

- The `SA` topic is **sentiment** analysis. Its comment previously said
  "Semantic Analysis", but every paper in it is sentiment. Fixed the comment.
- The `MT` panel contains `1810.03459`, a multilingual seq2seq **speech
  recognition** paper, so a little ASR content sits in the MT phase. Changing it
  would require regenerating `qasper_eval_MT.parquet`.

Speech *translation* papers are excluded from ASR — they are half MT papers, and
MT is its own phase.

### Sample

Document — paper `1911.09419` (phase 5, KG), abstract and a section:

```
Learning Hierarchy-Aware Knowledge Graph Embeddings for Link Prediction

<abstract> Knowledge graph embedding, which aims to represent entities and
relations as low dimensional vectors (or matrices, tensors, etc.), has been
shown to be a powerful technique for predicting missing links in knowledge
graphs. Existing knowledge graph embedding models mainly focus on modeling
relation patterns such as symmetry/antisymmetry, inversion, and composition.
However, many existing approaches fail to model semantic hierarchies … </abstract>

<section><section-title>Related Work</section-title>
In this section, we will describe the related work and the key differences
between them and our work in two aspects—the model category and the way to
model hierarchy structures in knowledge graphs. …
```

QASPER ships its own questions with extractive spans or free-form answers:

```
Q: What benchmark datasets are used for the link prediction task?
A: WN18RR | FB15k-237 | YAGO3-10
```

But the **eval set is not those raw questions**. `qasper/rewrite.py` rewrites each
one with GPT-4.1 to be answerable closed-book — naming the paper, since the model
sees a panel of 16 — and this is the format in `examples/qasper2/qasper_eval_*.parquet`:

```
user:      Please write a succinct answer to the following question. You do not
           need to restate the paper name or answer in complete sentences.
           <question>
           In the paper "Question Answering based Clinical Text Structuring Using
           Pre-trained Language Model," what dataset is used to pretrain the
           language model?
           </question>
           Provide your answer in the following format (output nothing else):
           <answer>{your answer here}</answer>
assistant: <answer>
           Chinese general corpus
           </answer>
```

Those parquets exist for QA (78 rows), MT and SA. **ASR and KG do not have them
yet** — see [Still outstanding](#still-outstanding).

---

## QuALITY — 7 articles per phase

**Source:** `tasksource/QuALITY`, `train` + `validation` (265 unique articles).
**Document:** one narrative — mostly 1950s–60s pulp science fiction from Project
Gutenberg, plus a tail of late-1990s Slate journalism.
**Split rule:** articles ≥ 6,000 tokens, the 35 longest, assigned by
[LPT](#how-phases-are-balanced-lpt).

| phase | articles | tokens | questions |
|---|---|---|---|
| 1 | 7 | 54,929 | 124 |
| 2 | 7 | 54,935 | 128 |
| 3 | 7 | 54,933 | 122 |
| 4 | 7 | 54,947 | 126 |
| 5 | 7 | 54,931 | 122 |
| **total** | **35** | **274,675** | **622** |

**One row of the HF dataset is one question**, with the full article repeated on
every row — `article_id` is the document key, and there are only 265 documents
behind 4,609 questions.

**Grouped by length, not by writer.** The plan doc proposed grouping by
`writer_id`, but that column is the crowdworker who wrote the *questions*, not
the author of the story (`author` is a separate column, and every article has
exactly two question writers). Holding `writer_id` fixed mixes pulp science
fiction and Slate film criticism inside one phase while holding only question
phrasing constant. Grouping by `author` is coherent but only six authors have
five or more articles and their totals range 21k–74k tokens, a 4x spread with no
room to rebalance. Filtering on length matches the review comment ("docs being
from the same author doesn't matter too much ... we can filter for docs over 5k
or 6k tokens only") and balances the phases to within 2%.

Each article is self-contained fiction, so forgetting stays measurable however
the buckets are drawn. The 6k filter excludes Slate entirely — those top out
around 4.3k tokens — so all 35 are Gutenberg. 167 articles clear 6k in total, so
there are 132 in reserve if larger phases are wanted.

A few authors recur across phases (Keith Laumer in 1, 3, 4 and 5). That is
inherent to a length-first selection and harmless for an accumulation
experiment, but is a constraint that could be added if disjoint authorship
matters.

### Sample

Document — article `51651`, "Conditionally Human" by Walter M. Miller, Jr.
(phase 1, 8,445 tokens):

```
Conditionally Human
By WALTER M. MILLER, JR.  Illustrated by DAVID STONE

[Transcriber's Note: This etext was produced from Galaxy Science Fiction
February 1952. Extensive research did not uncover any evidence that the U.S.
copyright on this publication was renewed.]

They were such cute synthetic creatures, it was impossible not to love them.
Of course, that was precisely why they were dangerous!

There was no use hanging around after breakfast. His wife was in a hurt mood,
and he could neither endure the hurt nor remedy it. …
```

Eval — 4-way multiple choice, `gold_label` is 1-indexed into `options`. Note the
question needs the whole plot, not a lookup:

```
Q: What will happen if Anne becomes pregnant?
   (1) Anne and Terry will be arrested and sterilized.
   (2) Anne and Terry will be executed.
   (3) Anne and Terry will be arrested, and Anne will be forced to abort.
   (4) Anne and Terry will be forced to divorce, and they will be given a
       hysterectomy and a vasectomy, respectively.
gold_label: 4    difficult: 1
```

The `difficult` flag marks questions that speed-readers answered wrong in
QuALITY's validation pass — useful as a harder eval subset.

---

## FinQA — 13–15 companies per phase, one fiscal year

**Source:** `train`/`dev`/`test` JSON from the [FinQA repo](https://github.com/czyssrs/FinQA) (script-based HF loading no longer works). Cached under `$CARTRIDGES_DIR/data/finqa/`.
**Document:** one **page** of a filing (~1k tokens) — `pre_text` + `table` +
`post_text`, keyed `COMPANY/YEAR/page_N.pdf`.
**Split rule:** fiscal year 2017 fixed; all 71 companies packed into five phases
by longest-processing-time bin packing, a company never splitting across phases.

| phase | companies | pages | tokens | questions |
|---|---|---|---|---|
| 1 | 13 | 52 | 47,275 | 155 |
| 2 | 13 | 44 | 46,821 | 116 |
| 3 | 15 | 46 | 47,239 | 150 |
| 4 | 15 | 48 | 47,157 | 150 |
| 5 | 15 | 51 | 47,143 | 166 |
| **total** | **71** | **241** | **235,635** | **737** |

**Year fixed, company varied** — the alternative raised in review ("we had some
empirical trouble in the past with this, lots of hallucination; why not same
year different companies?"). The plan doc proposed the opposite, one company
across five years, and the data does not condemn it: 163 of 172 Lockheed
questions (94.8%) name their year explicitly, and after masking years only two
question texts recur across years. It was dropped because the residual nine
year-less questions were not worth the risk, and because a cartridge trained on
five near-identical filings is exactly where cross-year figures blur. Varying
the company makes FinQA an accumulation stream like LongHealth and QuALITY.

FY2017 has the largest inventory of any year — 71 companies, 241 pages, 737
questions — which is what lets each phase reach ~47k tokens. An earlier draft
used only the five *largest* companies of FY2012, one per phase, and landed at
9–13k tokens per phase: four to fifteen times smaller than the other datasets.

Companies are the atomic unit — every page of a company stays in one phase — and
are assigned by [LPT](#how-phases-are-balanced-lpt), balancing tokens to within
0.96%. Packing is measured on `FinQAPage.text`, i.e. including the per-page
`<page><source>` wrapper, so the balance holds for the string that is actually
prefilled. Question counts even out only as a side effect, so phase 2 is light at
116 against 166 for phase 5.

### Sample

Document — `AAPL/2017/page_23.pdf` (phase 1). Every page is `pre_text`, a table,
then `post_text`; the tables are the point, and text is lowercased in the source:

```
pre_text: apple inc. | 2017 form 10-k | 20 company stock performance the
          following graph shows a comparison of cumulative total shareholder
          return, calculated on a dividend reinvested basis, for the company,
          the s&p 500 index, the s&p information technology index and the dow
          jones u.s. technology supersector index for the five years …

table:                  | sep2012 | sep2013 | sep2014 | sep2015 | sep2016 | sep2017
          apple inc.    |   $ 100 |    $ 74 |   $ 111 |   $ 128 |   $ 129 |   $ 179
          s&p 500 index |   $ 100 |   $ 119 |   $ 143 |   $ 142 |   $ 164 |   $ 194
```

Eval — arithmetic over the table. FinQA also ships the executable `program`,
so answers are checkable without a judge:

```
Q:       what was the change in the apple stock return between 2016 and 2017?
A:       50
program: subtract(179, 129)
```

This is what the "it is all calculation, there is only one right answer" argument
rests on: the answer is derived, not recalled, and gradable exactly.

---

## TechQA — ~99 technotes per phase

**Source:** `nvidia/TechQA-RAG-Eval` (910 rows).
**Document:** one IBM Technote, median 737 tokens.
**Split rule:** 496 technotes packed into five disjoint sets by LPT.

| phase | docs | tokens | questions |
|---|---|---|---|
| 1 | 99 | 151,966 | 116 |
| 2 | 98 | 151,895 | 123 |
| 3 | 99 | 151,905 | 129 |
| 4 | 100 | 151,918 | 123 |
| 5 | 100 | 151,927 | 119 |
| **total** | **496** | **759,611** | **610** |

**Sets of filenames, not one filename per phase.** The plan doc said "different
filename for each of the phase", but technotes have a median of 737 tokens and
p75 of 1,222 — a single one per phase would be a few hundred tokens.

**[LPT](#how-phases-are-balanced-lpt) rather than round-robin**, because the size
distribution is heavy-tailed: one technote is 44,383 tokens and three more clear
20k. Round-robin over the sorted list leaves a 26.3% spread between largest and
smallest phase; LPT brings it to 0.05% while keeping every document, outliers
included.

**300 of the 910 questions are excluded**, and they are not excludable-by-choice
— they are unscoreable. Every `is_impossible` row has an empty `contexts` list
*and* an `answer` field of `'-'` (one distinct value across all 300), while all
610 answerable rows have exactly one context and real answer text. So there is
no document to assign them to a phase, and no gold answer to score against:
log-perplexity would be measured against the literal string `-`.

They are by-design unanswerable questions, included in TechQA so a system can be
tested on whether it *abstains* rather than fabricates. That is worth running for
cartridges — a cartridge has no retrieval step to fail loudly — but it needs its
own rubric (a judge on "did the model refuse"), not the `'-'` as a target, so it
belongs in a separate hallucination eval rather than in the per-phase counts.

A product-based split was considered and rejected as unbalanceable: WebSphere
alone is 145 documents and 293k tokens, Tivoli 28 and 125k, then MQ 21 and 20k,
DB2 18 and 12k. One phase would be 15x another.

### Sample

Document — technote `swg21996508.txt` (phase 1). Every technote opens with a
`Title:` line and then a support-article body:

```
Title: IBM STREAMS 4.1.1.1 and 4.1.1.2 JOBS DO NOT INHERIT THE ENVIRONMENT
VARIABLES SET IN .BASHRC, WHEN STREAMS IS RUN AS A SYSTEM SERVICE - United States

Text:
FLASH (ALERT)
ABSTRACT
  In Streams 4.1.1.1 and 4.1.1.2 Streams jobs may not pick up the user
  environment from the streams user's .bashrc. This behavior is different from
  earlier releases. With these versions, when Streams is run as a system
  service, application environment variables must be set with streamtool.
CONTENT
Problem Description …
```

Eval — free-form. Questions are real user posts, so they are long and messy,
and answers are prose rather than a span:

```
Q: User environment variables no longer getting picked up after upgrade to
   4.1.1.1 or 4.1.1.2? Have you found that after upgrade to Streams 4.1.1.1 or
   4.1.1.2, that environment variables set in your .bashrc are no longer being
   set? For example ODBCINI is not set for the database toolkit and you get …
A: To work around the issue, set environment variables that are needed by the
   application directly in the instance with:
   streamtool setproperty -d <domain> -i <instance>
       --application-ev <VARIABLE NAME>=<VARIABLE VALUE>
```

Free-form answers mean TechQA needs an LLM judge or log-perplexity scoring —
unlike LongHealth and QuALITY, which are multiple choice, and FinQA, which is
exactly checkable.

---

## Consequences for the in-context baseline

Single phases fit both models' windows everywhere **except TechQA**, whose ~152k
phases exceed Llama-3.2-3B's 131,072. The cumulative stream is where it bites —
the phase at which the concatenated corpus stops fitting:

| dataset | overflows Llama-3.2-3B (131k) | overflows Qwen3-4B (262k) |
|---|---|---|
| LongHealth | phase 3 (139,958) | never — 236,124 total |
| FinQA | phase 3 (141,335) | never — 235,635 total |
| QuALITY | phase 3 (164,797) | phase 5 (274,675) |
| QASPER | phase 2 (180,100) | phase 3 (262,259) |
| TechQA | phase 1 (151,966) | phase 2 (303,861) |

QASPER is the knife-edge case: phases 1–3 total 262,259 tokens against Qwen3's
262,144 — over by 115 tokens, before any chat template or question is added.

So the summarization branch of the ICL baseline is required for TechQA on both
models and for QASPER beyond phase 2; on Llama it is required for every dataset
from phase 3 onward.

---

## Running the baselines on these phases

The phase corpora are materialised for the baseline suite by

```bash
python experiments/baselines/export_phase_corpus.py --dataset longhealth
```

which writes `data/phases/<dataset>/phase<k>.txt` — the phase resource's
`to_string()`, i.e. exactly the string the token counts above were measured on —
and `phase<k>_eval.parquet` holding that phase's questions. The exporter prints
the per-phase token and question counts, and they reproduce the tables above
exactly; that is the check that the baselines consume the documented corpus.

The baseline scripts then take `DOC_SET=<dataset>`, with `DOC_SEQ` defaulting to
`p1 p2 p3 p4 p5`:

```bash
DOC_SET=longhealth ICL_CUMULATIVE=1 sbatch experiments/scripts/baseline_icl.sh
```

Supported for export: LongHealth, FinQA, QuALITY, TechQA. QASPER is not, pending
its rewritten eval sets (below). Only `EVAL_MODE=logppl` is wired end to end —
answers are normalised to text so cross-entropy is comparable across the five
answer formats, but accuracy scoring still needs the per-dataset adapters (the
built-in MCQ scorer is hardcoded to four choices A–D, whereas LongHealth is
5-way). See `experiments/baselines/README.md`.

## Still outstanding

- **Eval sets for the new QASPER topics.** `qasper_eval_{MT,QA,SA}.parquet`
  exist; ASR and KG need `qasper/rewrite.py` run against them, which needs
  `OPENAI_API_KEY`. Until then QASPER has no phase export.
- **Self-study synthesis** for every phase of all five datasets. This blocks
  baselines 1 (dense sequential) and 2a (concatenated trained cartridges) on
  these streams; the in-context baselines need none, and AM (2b) runs today with
  `QUERY_SOURCE=repeat-prefill`.
- **Per-dataset accuracy adapters** in `experiments/evaluate/cartridge.py`, so
  the multiple-choice datasets can be scored on accuracy rather than only
  log-perplexity.
- **Llama-3.2-3B token counts**, blocked on gated-repo access.
