"""DIAG-CONTENT arm C: build a DOCUMENT-ORDER-REVERSED copy of the MT synth corpus.

This is a DATA artefact only -- no source file is touched. The Phase-2 per-document
loop takes its write order from `group_conversations_by_document`, which is a plain
dict keyed by <title>, so the write order is the order of FIRST APPEARANCE of each
title in the parquet's row order. Reversing that first-appearance order (while keeping
each document's own rows in their original relative order) yields a corpus that is
row-for-row identical in CONTENT and differs only in the SEQUENCE in which the 16
documents are written.

Env:
  SRC_PARQUET  input parquet
  DST_PARQUET  output parquet
"""

from __future__ import annotations

import os
import re

import pandas as pd

SRC = os.environ["SRC_PARQUET"]
DST = os.environ["DST_PARQUET"]

_TITLE = re.compile(r"<title>(.*?)</title>", re.DOTALL | re.IGNORECASE)


def key(prompt) -> str:
    p = prompt or ""
    m = _TITLE.search(p)
    return m.group(1).strip() if m else p


df = pd.read_parquet(SRC)
assert "system_prompt" in df.columns, df.columns.tolist()

keys = df["system_prompt"].map(key)
first_seen: list[str] = []
seen = set()
for k in keys:
    if k not in seen:
        seen.add(k)
        first_seen.append(k)

order = {k: i for i, k in enumerate(reversed(first_seen))}
print(f"[rev] {len(df)} rows, {len(first_seen)} documents")
for i, k in enumerate(first_seen):
    print(f"[rev] orig #{i + 1:2d} -> new #{order[k] + 1:2d}  {k[:70]!r}")

df = df.assign(_docrank=keys.map(order))
df = df.sort_values("_docrank", kind="stable").drop(columns=["_docrank"]).reset_index(drop=True)

# verify: first-appearance order is exactly the reverse of the original
new_first: list[str] = []
seen = set()
for k in df["system_prompt"].map(key):
    if k not in seen:
        seen.add(k)
        new_first.append(k)
assert new_first == list(reversed(first_seen)), "reversal failed"

df.to_parquet(DST, compression="snappy", index=False)
print(f"[rev] wrote {DST} ({os.path.getsize(DST) / 1e6:.1f} MB); order reversed OK")
