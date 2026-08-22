#!/usr/bin/env python3
"""Sanity + diversity checks for a self-study parquet, compared to QASPER ASR norms."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path


def _titles(system_prompt: str, kind: str) -> str:
    if kind == "longhealth":
        m = re.search(r"\(ID:\s*([^)]+)\)", system_prompt or "", re.I)
        return m.group(1).strip() if m else "NONE"
    if kind == "quality":
        m = re.search(r"<title>(.*?)</title>", system_prompt or "", re.S | re.I)
        return (m.group(1).strip() if m else "NONE")[:120]
    if kind == "finqa":
        m = re.search(r"<source>([^/<]+)/", system_prompt or "")
        if m:
            return m.group(1)
        m = re.search(r"<source>(.*?)</source>", system_prompt or "", re.S)
        return (m.group(1).strip() if m else "NONE")[:80]
    if kind == "techqa":
        m = re.search(r"<source>(.*?)</source>", system_prompt or "", re.S)
        return (m.group(1).strip() if m else "NONE")[:120]
    m = re.search(r"<title>(.*?)</title>", system_prompt or "", re.S | re.I)
    return (m.group(1).strip() if m else "NONE")[:120]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("quality", "finqa", "qasper", "techqa", "longhealth"),
    )
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--min-rows", type=int, default=32)
    parser.add_argument("--expect-docs", type=int, default=None)
    args = parser.parse_args()

    from cartridges.structs import read_conversations

    path = Path(args.parquet)
    cs = read_conversations(str(path))
    n = len(cs)
    print(f"path={path}")
    print(f"rows={n}")
    if n < args.min_rows:
        print(f"FAIL: expected >= {args.min_rows} rows")
        return 1

    msg_n = Counter(len(c.messages) for c in cs)
    print(f"n_messages={dict(msg_n)}")
    if any(k < 2 for k in msg_n):
        print("FAIL: some rows have <2 messages")
        return 1

    empty_sys = sum(1 for c in cs if not (c.system_prompt or "").strip())
    empty_user = sum(1 for c in cs if not c.messages or not (c.messages[0].content or "").strip())
    empty_asst = sum(1 for c in cs if not c.messages or not (c.messages[-1].content or "").strip())
    print(f"empty_sys={empty_sys} empty_user={empty_user} empty_asst={empty_asst}")
    if empty_sys or empty_user or empty_asst:
        print("FAIL: empty fields")
        return 1

    docs = [_titles(c.system_prompt or "", args.dataset) for c in cs]
    counts = Counter(docs)
    print(f"unique_docs={len(counts)}")
    for k, v in counts.most_common():
        print(f"  {v:5d}  {k}")

    expect = args.expect_docs
    if expect is None:
        expect = {
            "quality": 7,
            "finqa": 13,
            "qasper": 16,
            "techqa": 99,
            "longhealth": 4,
        }.get(args.dataset)
    if expect and n >= 48 and len(counts) < max(3, expect // 2):
        print(f"FAIL: too few unique docs ({len(counts)}) for {args.dataset} expect~{expect}")
        return 1
    if "NONE" in counts:
        print("FAIL: some rows missing document id tag")
        return 1

    mean_sys = sum(len(c.system_prompt or "") for c in cs) / n
    mean_user = sum(len(c.messages[0].content) for c in cs) / n
    mean_asst = sum(len(c.messages[-1].content) for c in cs) / n
    uniq_user = len({c.messages[0].content[:80] for c in cs})
    print(f"mean_sys_chars={mean_sys:.0f} mean_user_chars={mean_user:.0f} mean_asst_chars={mean_asst:.0f}")
    print(f"unique_user_prefixes={uniq_user}")
    # Large phase runs intentionally reuse a finite prompt bank. Require broad
    # coverage without making unique-prefix count scale unboundedly with rows.
    if uniq_user < max(8, min(256, n // 8)):
        print("FAIL: user turns not diverse enough")
        return 1

    seeds = [(c.metadata or {}).get("seed_prompt", "")[:60] for c in cs]
    print(f"unique_seed_heads={len(set(seeds))}")
    if len(set(seeds)) < 3:
        print("FAIL: seed prompts collapsed")
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
