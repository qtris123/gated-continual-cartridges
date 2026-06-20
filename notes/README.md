# Lab notebook — `notes/`

Durable, portable record of what's been tried in this repo, why, and what was
learned. **Survives server migrations** (Cursor's chat history doesn't).

## Layout

```
notes/
├── README.md          # this file — how to use the folder
├── JOURNAL.md         # chronological index, one line per entry
├── TEMPLATE.md        # copy this when starting a new entry
├── <YYYY-MM-DD>-<slug>.md   # one file per chat / problem / experiment
└── _raw_transcripts/  # (optional) verbatim Cursor chat dumps, for archive
```

## When to write an entry

- Finishing (or pausing) a chat tab on a distinct problem.
- Wrapping up an experiment, even if the result was negative.
- Hitting a non-obvious gotcha you'd want future-you (or future-agent) to know.

One entry ≈ one "thread of investigation". If a chat covered three unrelated
things, split it into three entries.

## How to write an entry

1. `cp notes/TEMPLATE.md notes/$(date +%F)-<short-slug>.md`
2. Fill it in. Be concrete: configs, numbers, paths, run IDs, commit SHAs.
3. Append a one-line entry to `JOURNAL.md` (most-recent at top).

## Prompt to paste into each open Cursor tab before migrating

> Summarize this entire chat as a new file under `notes/` in this repo, using
> `notes/TEMPLATE.md` as the structure. Filename: `notes/<YYYY-MM-DD>-<short-slug>.md`
> with today's date and a 2–4 word slug describing the topic.
>
> Be concrete: include exact configs, hyperparameters, numbers, file paths,
> commit SHAs, run IDs / wandb links, and any non-obvious gotchas. Capture
> *conclusions and decisions*, not a transcript of the conversation. Note
> anything that surprised me, anything I'd warn a colleague about, and what
> I'd try next if I came back tomorrow.
>
> Then prepend a one-line entry to `notes/JOURNAL.md` linking to the new file
> (date — slug — one-sentence summary — status).

## House rules

- **Past tense, conclusions first.** Notes are for future readers, not a diary.
- **Link to artifacts, don't paste them.** Reference plot paths, wandb runs,
  commit SHAs. Embed only small key numbers / tables inline.
- **Status field is honest.** `done` / `in-progress` / `abandoned` /
  `superseded-by:<file>`. An abandoned experiment with a clear "why" is
  more valuable than a silent gap.
- **No secrets.** No API keys, tokens, or private data — these notes get
  committed.
