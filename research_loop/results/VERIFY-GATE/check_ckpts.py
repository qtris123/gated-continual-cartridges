#!/usr/bin/env python
"""VERIFY-GATE: prove every eval loaded its intended, DISTINCT checkpoint.

Reads curve.tsv + the per-eval logs and checks, for each (arm, k, split):
  * the checkpoint path recorded is the one the arm's run dir holds for that k;
  * the eval log actually printed that path (the wrapper's `ckpt_in_log` grep);
  * across arms and k, the checkpoint paths are pairwise distinct AND the files
    are pairwise distinct by sha256 (so no two arms silently read one cartridge).
"""

from __future__ import annotations

import collections
import glob
import hashlib
import json
import os

REPO = "/localhome/local-triv/gated-continual-cartridges_explore"
RESDIR = os.path.join(REPO, "research_loop/results/VERIFY-GATE")


def sha(path, blocks=None):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 22)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    rows = []
    for line in open(os.path.join(RESDIR, "curve.tsv")).read().splitlines()[1:]:
        f = line.split("\t")
        rows.append({"arm": f[0], "selector": f[1], "offset": int(f[2]), "k": int(f[3]),
                     "split": f[4], "loss": f[5], "ckpt": f[6], "url": f[7]})

    rundirs = {}
    for line in open(os.path.join(RESDIR, "rundirs.tsv")).read().splitlines()[1:]:
        f = line.split("\t")
        rundirs[f[0]] = f[3]

    problems = []
    seen_in_log = {}
    for r in rows:
        log = os.path.join(RESDIR, "evals", f"eval_{r['arm']}_k{r['k']}_{r['split']}.log")
        txt = open(log).read() if os.path.exists(log) else ""
        r["ckpt_printed_in_log"] = r["ckpt"] in txt
        r["eval_loss_line_present"] = "Eval loss - " in txt
        want_dir = rundirs.get(r["arm"], "")
        r["ckpt_in_arm_rundir"] = bool(want_dir) and os.path.dirname(r["ckpt"]) == want_dir
        r["ckpt_is_doc_index"] = os.path.basename(r["ckpt"]).startswith(
            f"cache-after-doc-{r['k'] - 1:03d}-")
        if not (r["ckpt_printed_in_log"] and r["ckpt_in_arm_rundir"]
                and r["ckpt_is_doc_index"] and r["eval_loss_line_present"]):
            problems.append(r)
        seen_in_log[(r["arm"], r["k"], r["split"])] = r["ckpt"]

    # distinctness across (arm, k)
    unique_pairs = {(r["arm"], r["k"]): r["ckpt"] for r in rows}
    by_ckpt = collections.defaultdict(list)
    for (arm, k), c in unique_pairs.items():
        by_ckpt[c].append((arm, k))
    dup_paths = {c: v for c, v in by_ckpt.items() if len(v) > 1}

    hashes = {}
    for c in sorted(set(unique_pairs.values())):
        hashes[c] = sha(c)
    by_hash = collections.defaultdict(list)
    for c, h in hashes.items():
        by_hash[h].append(c)
    dup_hashes = {h: v for h, v in by_hash.items() if len(v) > 1}

    out = {
        "n_evals": len(rows),
        "n_distinct_checkpoint_paths": len(set(unique_pairs.values())),
        "n_arm_k_cells": len(unique_pairs),
        "all_evals_printed_their_checkpoint": not problems,
        "problems": problems,
        "duplicate_checkpoint_paths_across_arm_k": {k: v for k, v in dup_paths.items()},
        "duplicate_checkpoint_CONTENT_sha256": {h: v for h, v in dup_hashes.items()},
        "checkpoint_sha256": {os.path.relpath(c, REPO): h for c, h in sorted(hashes.items())},
    }
    dest = os.path.join(RESDIR, "ckpt_check.json")
    json.dump(out, open(dest, "w"), indent=2)
    print(json.dumps({k: out[k] for k in
                      ("n_evals", "n_distinct_checkpoint_paths", "n_arm_k_cells",
                       "all_evals_printed_their_checkpoint",
                       "duplicate_checkpoint_paths_across_arm_k",
                       "duplicate_checkpoint_CONTENT_sha256")}, indent=2)[:4000])
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
