#!/usr/bin/env python3
"""Build (and optionally run) QA/MT/SA evals on Phase-1 and Phase-2 cartridges.

Phase-3 SA-write runs already logged sa_acquisition. Phase-1 and Phase-2 did not.
This maps each eval to the cache sitting next to that run's config.yaml so the
tuple (obj, key, beta, idf) cannot drift from the .pt.

Discover jobs:
    python examples/shared/am/sweeps/eval_p1_p2_sa_refs.py --write-jobs JOBS.json

Run one job (called by the GPU queue):
    python examples/shared/am/sweeps/eval_p1_p2_sa_refs.py --run-tag TAG --jobs JOBS.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

from examples.shared.paths import ROOT

OUT = ROOT / "outputs"
DEFAULT_DW = {"both": 0.01, "delta": 0.01, "ridge": 0.0}
P1_CACHE = OUT / "phase1_selfdistill_qwen512" / "cache_last.pt"
P1_CFG = OUT / "phase1_selfdistill_qwen512" / "config.yaml"


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
            "qasper_topic",
            "document_data_path",
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


def tag_for(stage: str, obj: str, key: str, beta: int, idf: int) -> str:
    key_short = "ha" if key == "highest_attention" else key
    return f"{stage}_{obj}_{key_short}_b{beta}_idf{idf}"


def discover_p2() -> dict[tuple, dict]:
    """One completed P2 (MT-write) cartridge per (obj, key, beta, idf) at default dw.

    Mapping rule: cache_last.pt lives in the same uuid dir as config.yaml.
    Incomplete runs (no phase2_summary.json) are skipped. If two complete
    folders share a tuple, keep the later launch-id folder.
    """
    by: dict[tuple, dict] = {}
    for cache in sorted(
        list(OUT.glob("*/*/cache_last.pt"))
        + list((OUT / "ablations").glob("*/*/cache_last.pt"))
    ):
        uuid_dir = cache.parent
        folder = uuid_dir.parent.name
        if "-SA3_" in folder:
            continue
        if not any(x in folder for x in ("-ROPE_", "-KEY_", "-BK_", "-MEET_")):
            continue
        cfgp = uuid_dir / "config.yaml"
        sm = uuid_dir / "phase2_summary.json"
        if not cfgp.exists() or not sm.exists():
            continue
        cfg = parse_cfg(cfgp.read_text())
        if cfg.get("key_reposition") != "true":
            continue
        if float(cfg.get("rope_theta", 0)) != 5_000_000.0:
            continue
        if cfg.get("key_mode") not in ("omp", "highest_attention"):
            continue
        topic = cfg.get("qasper_topic", "MT")
        if topic == "SA":
            continue
        dw = float(cfg["delta_weight"])
        ridge = float(cfg["ridge_lambda"])
        o = obj_of(dw, ridge)
        if o == "none":
            continue
        if abs(dw - DEFAULT_DW[o]) > 1e-12:
            continue
        beta = 1 if cfg.get("beta") == "true" else 0
        idf = 1 if cfg.get("use_idf") == "true" else 0
        key = cfg["key_mode"]
        cell = (o, key, beta, idf)
        rec = {
            "tag": tag_for("P2", o, key, beta, idf),
            "stage": "P2",
            "obj": o,
            "key": key,
            "beta": beta,
            "idf": idf,
            "dw": dw,
            "ridge": ridge,
            "qasper_topic": topic,
            "name": cfg.get("name") or folder,
            "folder": folder,
            "run_id": uuid_dir.name,
            "cache_rel": str(cache.relative_to(ROOT)),
            "config_rel": str(cfgp.relative_to(ROOT)),
            "summary_rel": str(sm.relative_to(ROOT)),
            "in_run_evals": list(json.load(open(sm)).get("eval_metrics", {})),
        }
        prev = by.get(cell)
        if prev is None or folder > prev["folder"]:
            by[cell] = rec
    return by


def build_jobs() -> list[dict]:
    jobs: list[dict] = []
    if not P1_CACHE.exists():
        raise SystemExit(f"missing Phase-1 cache: {P1_CACHE}")
    jobs.append(
        {
            "tag": "P1_phase1_selfdistill_qwen512",
            "stage": "P1",
            "obj": None,
            "key": None,
            "beta": None,
            "idf": None,
            "dw": None,
            "ridge": None,
            "qasper_topic": "QA-selfdistill",
            "name": "phase1_selfdistill_qwen512",
            "folder": "phase1_selfdistill_qwen512",
            "run_id": None,
            "cache_rel": str(P1_CACHE.relative_to(ROOT)),
            "config_rel": str(P1_CFG.relative_to(ROOT)) if P1_CFG.exists() else None,
            "summary_rel": None,
            "in_run_evals": ["qasper_perplexity"],
        }
    )
    p2 = discover_p2()
    if len(p2) != 24:
        raise SystemExit(f"expected 24 P2 default-dw cells, found {len(p2)}")
    jobs.extend(sorted(p2.values(), key=lambda r: (r["obj"], r["key"], r["beta"], r["idf"])))
    return jobs


def run_one(job: dict, out_json: Path) -> dict:
    """Eval QA/MT/SA for one cache; write JSON that includes the mapping fields."""
    os.environ["CACHE_PATH"] = str(ROOT / job["cache_rel"])
    os.environ["EVAL_QA_PATH"] = str(ROOT / "data/qasper/eval/qasper_eval_QA.parquet")
    os.environ["EVAL_MT_PATH"] = str(ROOT / "data/qasper/eval/qasper_eval_MT.parquet")
    os.environ["EVAL_SA_PATH"] = str(ROOT / "data/qasper/eval/qasper_eval_SA.parquet")
    os.environ["OUT_JSON"] = str(out_json)
    os.environ.setdefault("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    # Import after env is set: eval_cartridge reads CACHE_PATH at import time.
    spec = importlib.util.spec_from_file_location(
        "eval_cartridge",
        ROOT / "examples/shared/evaluate/cartridge_perplexity.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.CACHE_PATH = os.environ["CACHE_PATH"]
    mod.OUT_JSON = str(out_json)
    mod.main()

    payload = json.load(open(out_json))
    payload["mapping"] = {k: job[k] for k in job}
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-jobs", type=Path)
    ap.add_argument("--jobs", type=Path)
    ap.add_argument("--run-tag")
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()

    if args.write_jobs:
        jobs = build_jobs()
        args.write_jobs.parent.mkdir(parents=True, exist_ok=True)
        json.dump(jobs, open(args.write_jobs, "w"), indent=2)
        print(f"wrote {len(jobs)} jobs -> {args.write_jobs}")
        for j in jobs:
            loc = f"({j['obj']}, {j['key']}, {j['beta']}, {j['idf']})" if j["stage"] == "P2" else "P1"
            print(f"  {j['tag']:40s}  {loc:42s}  {j['cache_rel']}")
            if j["stage"] == "P2" and "sa_acquisition" in (j.get("in_run_evals") or []):
                print("    WARN: this P2 summary already has SA (unexpected)", file=sys.stderr)
        return 0

    if args.run_tag and args.jobs and args.out_dir:
        jobs = {j["tag"]: j for j in json.load(open(args.jobs))}
        job = jobs[args.run_tag]
        cache = ROOT / job["cache_rel"]
        cfg = ROOT / job["config_rel"] if job.get("config_rel") else None
        if not cache.exists():
            raise SystemExit(f"missing cache {cache}")
        if cfg and not cfg.exists():
            raise SystemExit(f"missing config {cfg}")
        # Same-directory mapping check: cache and config share a parent.
        if cfg and cache.parent != cfg.parent and job["stage"] == "P2":
            raise SystemExit(f"cache/config parent mismatch: {cache.parent} vs {cfg.parent}")
        out_json = args.out_dir / f"{job['tag']}.json"
        if out_json.exists():
            prev = json.load(open(out_json))
            if prev.get("eval_metrics", {}).get("sa_acquisition"):
                print(f"skip {job['tag']}: already has SA in {out_json}")
                return 0
        print(f"eval {job['tag']} cache={job['cache_rel']}")
        run_one(job, out_json)
        return 0

    ap.error("use --write-jobs PATH  or  --run-tag TAG --jobs PATH --out-dir DIR")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
