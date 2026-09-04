#!/usr/bin/env python3
"""Sweep manifest: resolve arms, derive stable tags, materialize recipes.

The manifest (see outputs/experiments/soft_locality/sweeps.yaml) is the single
source of truth shared by the launcher and every analysis/plot script, so adding
an arm or changing a technique's values is a one-file edit. Each arm resolves to
an `Arm(label, tag, recipe, dataset, axis, value)`; the `tag` is either an
explicit alias to an existing on-disk lineage or a deterministic structured tag
derived from the selector + axis value (identity that cannot drift from config).

CLI:
    python -m examples.shared.am.sweep arms   --manifest M [--stream S]
    python -m examples.shared.am.sweep render --manifest M [--stream S]   # write recipes for alias-less arms
    python -m examples.shared.am.sweep launch --manifest M --stream S --dataset D --gpus 0,1,2,3 [--dry-run] [--force]
    python -m examples.shared.am.sweep eval   --manifest M --stream S --dataset D --gpus 0,1,2,3 --kind accuracy|generations
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import yaml

from cartridges.am import (
    BetaFitter,
    KeyWriter,
    ReferenceQueries,
    SlotSelector,
    ValueObjective,
)
from examples.shared.am.continual_env import DATASETS, spec
from examples.shared.paths import ROOT

# Recipes-as-config: a valid AM axis is any field that exists in the component
# config schema (no slot-knob registry). Non-AM axes (e.g. `p01.*`) are validated
# by the endpoint that consumes them (build_p01).
_AM_SECTION_CONFIG = {
    "slots": SlotSelector.Config,
    "queries": ReferenceQueries.Config,
    "keys": KeyWriter.Config,
    "beta": BetaFitter.Config,
    "objective": ValueObjective.Config,
}


def _schema_has_axis(dotted: str) -> bool:
    """Whether a dotted axis names a real field of the AMContinualConfig tree."""
    head, *rest = dotted.split(".")
    section = _AM_SECTION_CONFIG.get(head)
    if section is None or not rest:
        return True  # not an AM-section axis (e.g. p01.*): endpoint validates it
    return rest[-1] in section.model_fields

LOSS_MATRIX = "outputs/evaluations/{ds}/{tag}/teacher-forced-logppl-v1/matrix.json"
RUN_CHAIN = ROOT / "examples/shared/am/run_chain.py"
BUILD_P01 = ROOT / "examples/shared/am/build_p01.py"
PY_BIN = os.environ.get("CARTRIDGES_PYTHON", ".venv/bin/python")


@dataclass(frozen=True)
class Arm:
    stream: str
    label: str
    tag: str
    recipe: str          # recipe path (existing alias recipe, or generated)
    axis: str            # dotted recipe path, e.g. "slots.usage_penalty_lambda"
    value: object
    is_alias: bool


def _num_slug(value) -> str:
    """Filesystem-safe number: 0.5 -> 0p5, 16.0 -> 16, 1e-06 -> 1e-06."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).replace(".", "p").replace("-", "m")


def derive_tag(axis: str, value, fixed: dict | None) -> str:
    """Deterministic structured tag from the axis field + value (+ pinned fields).

    e.g. axis=slots.usage_penalty_lambda value=2.0 fixed={slots.top_t:32}
         -> "usage_penalty_lambda2.topt32". Same config always yields the same
         tag, so a run's directory name encodes its identity.
    """
    field = axis.split(".")[-1]
    parts = [f"{field}{_num_slug(value)}"]
    for k, v in sorted((fixed or {}).items()):
        parts.append(f"{k.split('.')[-1]}{_num_slug(v)}")
    return ".".join(parts)


def _set_dotted(cfg: dict, dotted: str, value) -> None:
    node = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value


def load_manifest(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text())
    for name, stream in data.get("streams", {}).items():
        axis = stream["axis"]
        if not _schema_has_axis(axis):
            raise ValueError(
                f"stream {name!r}: axis {axis!r} is not a field of the "
                "AMContinualConfig component tree; check the dotted path."
            )
    return data


def resolve(manifest: dict, stream: str | None = None) -> list[Arm]:
    streams = manifest["streams"]
    names = [stream] if stream else list(streams)
    arms: list[Arm] = []
    for name in names:
        s = streams[name]
        axis, fixed = s["axis"], s.get("fixed")
        fmt = s.get("label_fmt", "{value}")
        for a in s["arms"]:
            value = a["value"]
            alias = a.get("alias")
            tag = alias or derive_tag(axis, value, fixed)
            recipe = (
                s["base_recipe"]
                if alias
                else f"outputs/recipes/{tag}.yaml"
            )
            arms.append(
                Arm(name, fmt.format(value=value), tag, recipe, axis, value, bool(alias))
            )
    return arms


def render(manifest: dict, stream: str | None = None) -> list[Arm]:
    """Materialize recipes for arms without an alias; return all resolved arms."""
    made = 0
    for arm in resolve(manifest, stream):
        if arm.is_alias:
            continue
        s = manifest["streams"][arm.stream]
        base = yaml.safe_load((ROOT / s["base_recipe"]).read_text())
        _set_dotted(base, arm.axis, arm.value)
        for k, v in (s.get("fixed") or {}).items():
            _set_dotted(base, k, v)
        out = ROOT / arm.recipe
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(base, sort_keys=True))
        print(f"  wrote {arm.recipe}  (tag={arm.tag})")
        made += 1
    if made == 0:
        print("  (all arms use aliases; no recipes generated)")
    return resolve(manifest, stream)


def _loss_done(ds: str, tag: str) -> bool:
    return (ROOT / LOSS_MATRIX.format(ds=ds, tag=tag)).exists()


def _build_p01_root(manifest_name: str, tag: str) -> Path:
    return ROOT / f"outputs/experiments/{manifest_name}/{tag}"


def _p01_cache_done(manifest_name: str, tag: str) -> bool:
    return bool(sorted(_build_p01_root(manifest_name, tag).glob("*/*/cache_last.pt")))


def _resolve_p01_cache(dataset: str, launch_cfg: dict, *, allow_missing: bool) -> str:
    """The p01 cartridge run_chain consumes -- an explicit pointer or auto-discovery.

    Sweeps reference p01 by *pointer*; build it beforehand with
    `python -m examples.shared.am.build_p01 --dataset D`.
    """
    explicit = launch_cfg.get("p01_cache")
    if explicit:
        path = ROOT / explicit
        if path.exists() or allow_missing:
            return str(path)
        raise SystemExit(f"p01_cache {path} not found; run build_p01 --dataset {dataset}")
    root = ROOT / launch_cfg.get("p01_root", f"outputs/{dataset}_p1_ropefix")
    caches = sorted(root.glob("*/*/cache_last.pt"))
    if caches:
        return str(caches[0])
    if allow_missing:
        return f"<p01: run build_p01 --dataset {dataset}>"
    raise SystemExit(
        f"no p01 cache for {dataset} under {root}; "
        f"run `python -m examples.shared.am.build_p01 --dataset {dataset}` first"
    )


def launch(manifest: dict, stream: str, dataset: str, gpus: list[str],
           dry_run: bool = False, force: bool = False) -> None:
    """Run the manifest's endpoint for every arm, one job per GPU.

    Replaces the hand-batched shell orchestrators AND the per-dataset
    `run_*_5x5_continual.sh` wrappers: the launcher invokes the shared endpoints
    directly (`run_chain` by default, or `build_p01` for p01-axis sweeps). A GPU
    pool backfills arms as they finish; idempotent -- arms whose output already
    exists are skipped unless --force.
    """
    manifest_name = manifest.get("name", "sweep")
    endpoint = manifest.get("endpoint", "run_chain")
    method = manifest.get("method", "am")
    launch_cfg = (manifest.get("launch") or {}).get(dataset) or {}

    p01_cache = None
    if endpoint == "run_chain":
        p01_cache = _resolve_p01_cache(dataset, launch_cfg, allow_missing=dry_run)

    arms = resolve(manifest, stream)
    if endpoint == "build_p01":
        done = lambda a: _p01_cache_done(manifest_name, a.tag)  # noqa: E731
    else:
        done = lambda a: _loss_done(dataset, a.tag)  # noqa: E731
    pending = [a for a in arms if force or not done(a)]
    skipped = [a for a in arms if a not in pending]
    print(f"launch endpoint={endpoint} stream={stream} dataset={dataset} gpus={gpus}")
    for a in skipped:
        print(f"  [done]   {a.label:<10} {a.tag}")
    if not pending:
        print("  nothing to launch (all arms complete).")
        return
    (ROOT / "logs").mkdir(exist_ok=True)
    gpu_q: "queue.Queue[str]" = queue.Queue()
    for g in gpus:
        gpu_q.put(g)

    def _cmd(arm: Arm, g: str) -> list[str]:
        recipe = str(ROOT / arm.recipe)
        if endpoint == "build_p01":
            return [PY_BIN, str(BUILD_P01), "--dataset", dataset,
                    "--recipe-config", recipe,
                    "--p1-root", str(_build_p01_root(manifest_name, arm.tag)),
                    "--gpu", g] + (["--force"] if force else [])
        return [PY_BIN, str(RUN_CHAIN), "--dataset", dataset, "--method", method,
                "--p01-cache", str(p01_cache), "--tag", arm.tag,
                "--recipe-config", recipe, "--gpu", g, "--eval-gpus", g]

    def run_arm(arm: Arm) -> tuple[str, int]:
        g = gpu_q.get()
        try:
            env = os.environ.copy()
            env.update(CARTRIDGES_PYTHON=env.get("CARTRIDGES_PYTHON", PY_BIN),
                       PYTHONPATH=f"{ROOT}:{env.get('PYTHONPATH', '')}",
                       GPU=g, EVAL_GPUS=g, TAG=arm.tag)
            cmd = _cmd(arm, g)
            log = ROOT / f"logs/{stream}_{dataset}_{arm.tag}.log"
            if dry_run:
                print(f"  [dry] gpu{g}: {' '.join(cmd)}  > {log}")
                return arm.tag, 0
            print(f"  [run] gpu{g}: {arm.label} ({arm.tag}) -> {log}")
            with open(log, "w") as fh:
                rc = subprocess.run(cmd, env=env, cwd=ROOT, stdout=fh, stderr=fh).returncode
            print(f"  [{'ok' if rc == 0 else 'FAIL'}] {arm.tag} (gpu{g}, rc={rc})")
            return arm.tag, rc
        finally:
            gpu_q.put(g)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        results = list(pool.map(run_arm, pending))
    failed = [t for t, rc in results if rc != 0]
    print(f"launch complete: {len(results) - len(failed)} ok, {len(failed)} failed"
          + (f" ({failed})" if failed else ""))


def check(manifest: dict) -> None:
    """Completeness by construction: every dataset in the manifest must have a
    `DatasetSpec` + synth data for phases 1-5 + eval data + a scorer.

    This makes the matrix provably dense: adding a dataset to an experiment is
    just adding it to `datasets:` (given its facts/synth/scorer exist).
    """
    datasets = manifest.get("datasets", [])
    problems: list[str] = []
    for ds in datasets:
        if ds not in DATASETS:
            print(f"[FAIL] {ds}: no DatasetSpec in continual_env.DATASETS")
            problems.append(ds)
            continue
        s = spec(ds)
        miss_synth = [p for p in range(1, 6) if not s.synth_path(p).exists()]
        miss_eval = [p for p in range(1, 6) if not s.eval_path(p).exists()]
        scorer_ok = bool(getattr(s, "scorer", None))
        status = "OK" if not miss_synth and not miss_eval and scorer_ok else "FAIL"
        print(f"[{status}] {ds}: p01_method={s.p01_method} scorer={s.scorer}")
        if miss_synth:
            print(f"    missing synth phases {miss_synth}: e.g. {s.synth_path(miss_synth[0])}")
        if miss_eval:
            print(f"    missing eval phases {miss_eval}: e.g. {s.eval_path(miss_eval[0])}")
        if not scorer_ok:
            print("    no scorer defined")
        if status == "FAIL":
            problems.append(ds)
    if problems:
        raise SystemExit(f"check FAILED for: {problems}")
    print(f"check OK: {len(datasets)} datasets have DatasetSpec + synth + eval + scorer")


def eval_stream(manifest: dict, stream: str, dataset: str, gpus: list[str],
                kind: str) -> None:
    """Loop a per-arm eval wrapper (accuracy/generations) over the stream.

    Each wrapper fans its 5 stages across all `gpus`, so arms run sequentially to
    avoid GPU contention (safe, no OOM).
    """
    wrapper = {
        "accuracy": ROOT / "examples/shared/evaluate/run_accuracy.sh",
        "generations": ROOT / "examples/shared/evaluate/record_generations.sh",
    }[kind]
    gpu_csv = ",".join(gpus)
    for arm in resolve(manifest, stream):
        print(f"  [{kind}] {arm.label} ({arm.tag}) on gpus {gpu_csv}")
        subprocess.run(["bash", str(wrapper), dataset, arm.tag, gpu_csv],
                       cwd=ROOT, check=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["arms", "render", "launch", "eval", "check"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--stream", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--kind", default="accuracy", choices=["accuracy", "generations"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    manifest = load_manifest(args.manifest)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]

    if args.cmd == "check":
        check(manifest)
        return
    if args.cmd == "render":
        render(manifest, args.stream)
        return
    if args.cmd == "launch":
        if not (args.stream and args.dataset):
            raise SystemExit("launch requires --stream and --dataset")
        launch(manifest, args.stream, args.dataset, gpus, args.dry_run, args.force)
        return
    if args.cmd == "eval":
        if not (args.stream and args.dataset):
            raise SystemExit("eval requires --stream and --dataset")
        eval_stream(manifest, args.stream, args.dataset, gpus, args.kind)
        return
    print(f"{'stream':<8}{'label':<10}{'value':>7}  {'alias?':<6} tag")
    for arm in resolve(manifest, args.stream):
        print(f"{arm.stream:<8}{arm.label:<10}{str(arm.value):>7}  "
              f"{'yes' if arm.is_alias else 'no':<6} {arm.tag}")


if __name__ == "__main__":
    main()
