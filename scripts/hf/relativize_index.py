#!/usr/bin/env python3
"""Rewrite outputs/caches/index.json into a machine-independent index.

The on-disk index embeds absolute cache paths (and an absolute ``root``). For
the HuggingFace dataset we store paths relative to the caches root so the index
is portable; ``prepare_artifacts.sh`` rewrites ``root`` back to the local path
after download.

Usage:
    relativize_index.py <src_index.json> <dst_index.json>
"""
import json
import os
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    with open(src) as fh:
        idx = json.load(fh)

    root = idx.get("root", "")
    caches = idx.get("caches", [])
    out_caches = []
    for c in caches:
        cp = c.get("cache_path", "") or ""
        if root and cp.startswith(root):
            cp = os.path.relpath(cp, root)
        c = dict(c)
        c["cache_path"] = cp  # now relative to caches root
        out_caches.append(c)

    out = {
        "root": "outputs/caches",  # placeholder; rewritten by prepare_artifacts.sh
        "layout": idx.get("layout", "dataset/stage/technique/cache.pt"),
        "n_caches": len(out_caches),
        "note": (
            "Paths are relative to the caches root. finqa/quality/techqa 5-phase "
            "caches are shipped as run-tree shards (caches-<ds>-5phase-runs.tar.zst); "
            "their final compacted caches are the cache-step<N>.pt files referenced "
            "by results/state/<ds>/<method>/p<k>.json."
        ),
        "caches": out_caches,
    }
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {dst} ({len(out_caches)} caches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
