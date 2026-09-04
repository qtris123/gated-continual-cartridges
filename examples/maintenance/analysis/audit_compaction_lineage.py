"""Audit the restored AM cache tree: lineage roots and per-phase eval metrics.

Read-only. Walks outputs/caches/index.json, pairs each cache with its run
config.yaml / phase2_summary.json, and reconstructs the phase chain by following
kv_cache_initializer.path so that runs seeded from a self-distilled Phase 1 can
be separated from those seeded from a compacted (attention-matched) Phase 1.
"""

import json
import os
import re
from collections import defaultdict

import yaml

REPO = os.environ.get("CARTRIDGES_DIR", os.getcwd())
CACHES = os.path.join(REPO, "outputs", "caches")
TASKS = ["qa", "mt", "sa", "asr", "kg"]


def load(path, loader):
    try:
        with open(path) as fh:
            return loader(fh)
    except Exception:
        return None


def main():
    index = load(os.path.join(CACHES, "index.json"), json.load)
    rows = []
    for entry in index["caches"]:
        cache_path = entry["cache_path"]
        run = os.path.join(os.path.dirname(cache_path), "run")
        cfg = load(os.path.join(run, "config.yaml"), yaml.safe_load)
        summ = load(os.path.join(run, "phase2_summary.json"), json.load)
        init = None
        topic = None
        if cfg:
            init = (cfg.get("kv_cache_initializer") or {}).get("path")
            topic = (cfg.get("teacher") or {}).get("qasper_topic")
        rows.append(
            dict(
                stage=entry["stage"],
                technique=entry["technique"],
                cache_path=cache_path,
                has_cfg=cfg is not None,
                has_summary=summ is not None,
                init=init,
                topic=topic,
                run_name=(summ or {}).get("run_name"),
                mse=(summ or {}).get("mean_mse_last_doc"),
                vmax=((summ or {}).get("value_norms") or {}).get("global_max_abs"),
                metrics={
                    t: ((summ or {}).get("eval_metrics") or {}).get(t, {}).get("loss")
                    for t in TASKS
                },
            )
        )

    print(f"caches indexed: {len(rows)}")
    print(f"  with run/config.yaml       : {sum(r['has_cfg'] for r in rows)}")
    print(f"  with phase2_summary.json   : {sum(r['has_summary'] for r in rows)}")
    print(f"  cache file present on disk : {sum(os.path.exists(r['cache_path']) for r in rows)}")

    by_stage = defaultdict(int)
    for r in rows:
        by_stage[r["stage"]] += 1
    print("\nper stage:", dict(sorted(by_stage.items())))

    print("\n=== distinct Phase-1 seeds referenced by init paths ===")
    seeds = defaultdict(int)
    for r in rows:
        if r["init"]:
            seeds[re.sub(r"/[0-9a-f-]{36}/.*$", "", r["init"]).split("/")[-1]] += 1
        else:
            seeds["<none: this run IS a phase-1 root>"] += 1
    for k, v in sorted(seeds.items(), key=lambda x: -x[1]):
        print(f"  {v:4d}  {k}")

    print("\n=== stage x whether init parent cache still exists on disk ===")
    miss = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["init"]:
            miss[r["stage"]][os.path.exists(r["init"])] += 1
    for s in sorted(miss):
        absent, present = miss[s][0], miss[s][1]
        print(f"  {s}: parent present {present}, parent MISSING {absent}")

    print("\n=== eval loss by stage/technique (lower is better) ===")
    hdr = "  " + "stage technique".ljust(38) + "".join(t.upper().rjust(8) for t in TASKS) + "     mse   vmax"
    print(hdr)
    for r in sorted(rows, key=lambda r: (r["stage"], r["technique"])):
        if not r["has_summary"]:
            continue
        name = f"{r['stage']} {r['technique']}"[:37].ljust(38)
        vals = "".join(
            (f"{r['metrics'][t]:8.3f}" if r["metrics"][t] is not None else "     n/a")
            for t in TASKS
        )
        mse = f"{r['mse']:8.4f}" if r["mse"] is not None else "     n/a"
        vmax = f"{r['vmax']:7.1f}" if r["vmax"] is not None else "    n/a"
        print("  " + name + vals + mse + vmax)

    print("\n=== caches with NO summary (metrics unavailable) ===")
    for r in rows:
        if not r["has_summary"]:
            print(f"  {r['stage']:4s} {r['technique']}")


if __name__ == "__main__":
    main()
