#!/usr/bin/env python3
"""Quantify within-generation degeneracy from a cartridge_generations dump.

distinct-across-questions (the existing panel) misses *within-answer* looping,
which is the actual failure here. Per arm we report:
  rep4    = mean fraction of repeated 4-grams (1 - unique4/total4); ->1 = loops
  uniqw   = mean unique-word ratio inside each answer; ->0 = loops
  toprun  = mean longest run of one repeated token
"""
from __future__ import annotations
import argparse, json


def rep4(words):
    grams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def uniqw(words):
    return len(set(words)) / len(words) if words else 0.0


def toprun(words):
    best = cur = 0
    prev = None
    for w in words:
        cur = cur + 1 if w == prev else 1
        best = max(best, cur)
        prev = w
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    args = ap.parse_args()
    data = json.loads(open(args.dump).read())
    results = data["results"]
    print(f"{args.dump}")
    print(f"{'arm':<6} {'rep4':>7} {'uniqw':>7} {'toprun':>7} {'meanlen':>8}")
    for arm, rows in results.items():
        answers = [r["answer"].split() for r in rows]
        r4 = sum(rep4(a) for a in answers) / len(answers)
        uw = sum(uniqw(a) for a in answers) / len(answers)
        tr = sum(toprun(a) for a in answers) / len(answers)
        ml = sum(len(a) for a in answers) / len(answers)
        print(f"{arm:<6} {r4:>7.3f} {uw:>7.3f} {tr:>7.1f} {ml:>8.1f}")


if __name__ == "__main__":
    main()
