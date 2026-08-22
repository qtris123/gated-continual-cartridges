#!/usr/bin/env python3
"""Rebuild rope-corrected stage-2 inventories for the SA continual sweep.

Scans outputs/*/*/cache_last.pt for cartridges with:
  key_reposition=true, rope_theta=5e6, key in {omp, highest_attention}

Writes:
  outputs/sa_stage3_from_rope_inventory.json           # all ladder cells
  outputs/sa_stage3_from_rope_inventory_default_dw.json # one per (obj,key,β,idf)
      both/delta @ dw=1e-2, ridge @ dw=0

Exit code 0 always when write succeeds. Prints counts; use --require-default N
to fail if the default-dw inventory is short of N cells.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from examples.shared.paths import ROOT

OUT = ROOT / "outputs"
DEFAULT_DW = {"both": 0.01, "delta": 0.01, "ridge": 0.0}


def parse_cfg(txt: str) -> dict:
    d: dict = {}
    lines = txt.splitlines()
    for line in lines:
        s = line.strip()
        for k in (
            "rope_theta",
            "key_reposition",
            "key_mode",
            "delta_weight",
            "ridge_lambda",
            "use_idf",
            "name",
        ):
            if s.startswith(k + ":"):
                d[k] = s.split(":", 1)[1].strip()
    for i, line in enumerate(lines):
        if line.startswith("beta:"):
            for j in range(i + 1, min(i + 20, len(lines))):
                if "enabled:" in lines[j]:
                    d["beta"] = lines[j].split(":", 1)[1].strip()
                    break
            break
    return d


def obj_of(dw: float, ridge: float) -> str:
    if ridge > 0 and dw > 0:
        return "both"
    if ridge == 0 and dw > 0:
        return "delta"
    if ridge > 0 and dw == 0:
        return "ridge"
    return "none"


def run_cache_paths(out: Path):
    for root in (out, out / "ablations"):
        yield from root.glob("*/*/cache_last.pt")


def collect() -> list[dict]:
    rows: list[dict] = []
    for cache in sorted(run_cache_paths(OUT)):
        folder = cache.parent.parent.name
        # Stage-2 only. SA3_* names contain KEY_/etc and must be skipped.
        if "-SA3_" in folder:
            continue
        if not any(x in folder for x in ("-ROPE_", "-KEY_", "-BK_", "-MEET_")):
            continue
        cfgp = cache.parent / "config.yaml"
        if not cfgp.exists():
            continue
        cfg = parse_cfg(cfgp.read_text())
        if cfg.get("key_reposition") != "true":
            continue
        if float(cfg.get("rope_theta", 0)) != 5_000_000.0:
            continue
        if cfg.get("key_mode") not in ("omp", "highest_attention"):
            continue
        dw = float(cfg["delta_weight"])
        ridge = float(cfg["ridge_lambda"])
        rows.append(
            {
                "folder": folder,
                "name": cfg.get("name") or folder,
                "cache_rel": str(cache.relative_to(ROOT)),
                "key": cfg["key_mode"],
                "beta": 1 if cfg.get("beta") == "true" else 0,
                "idf": 1 if cfg.get("use_idf") == "true" else 0,
                "dw": dw,
                "ridge": ridge,
                "obj": obj_of(dw, ridge),
            }
        )
    return rows


def default_dw_subset(rows: list[dict]) -> list[dict]:
    """One row per (obj, key, beta, idf) at the screenshot default dw."""
    by_cell: dict[tuple, dict] = {}
    for r in rows:
        o = r["obj"]
        if o == "none":
            continue
        target = DEFAULT_DW[o]
        if abs(r["dw"] - target) > 1e-12:
            continue
        key = (o, r["key"], r["beta"], r["idf"])
        # Prefer ROPE_ names when duplicates somehow appear
        prev = by_cell.get(key)
        if prev is None or ("ROPE_" in r["folder"] and "ROPE_" not in prev["folder"]):
            by_cell[key] = r
    return sorted(
        by_cell.values(),
        key=lambda r: (r["obj"], r["key"], r["beta"], r["idf"]),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-default", type=int, default=0)
    args = ap.parse_args()

    rows = collect()
    default = default_dw_subset(rows)

    full_path = OUT / "sa_stage3_from_rope_inventory.json"
    def_path = OUT / "sa_stage3_from_rope_inventory_default_dw.json"
    json.dump(rows, open(full_path, "w"), indent=2)
    json.dump(default, open(def_path, "w"), indent=2)

    print(f"full ladder:   {len(rows):2d} -> {full_path}")
    print(f"default dw:    {len(default):2d} -> {def_path}")
    for r in default:
        print(
            f"  ({r['obj']}, {r['key']}, {r['beta']}, {r['idf']})  "
            f"dw={r['dw']:g}  ← {r['name']}"
        )

    if args.require_default and len(default) < args.require_default:
        print(
            f"ERROR: default-dw inventory has {len(default)} cells, "
            f"need {args.require_default}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
