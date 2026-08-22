"""Canonical compacted-KV layout: outputs/caches/<dataset>/<stage>/<technique>/.

Training may still write a wandb-style run directory first. ``publish_cache``
then adopts the checkpoint as ``cache.pt`` and moves the rest of the run into
``run/`` so the bytes live in one place.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from examples.shared.paths import ROOT as REPO_ROOT

CACHES_ROOT = REPO_ROOT / "outputs" / "caches"


def stage_id(phase: int | str) -> str:
    text = str(phase).strip().lower()
    if text.startswith("p"):
        text = text[1:]
    return f"p{int(text):02d}"


def cache_dir(dataset: str, stage: int | str, technique: str) -> Path:
    return CACHES_ROOT / dataset / stage_id(stage) / technique


def resolve_cache_file(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.exists() and right.exists() and left.samefile(right)
    except OSError:
        return False


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        if _same_file(dest, src) or dest.resolve() == src.resolve():
            return
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def _rmdir_parents_until(path: Path, stop: Path) -> None:
    current = path
    stop = stop.resolve()
    while current != current.parent:
        if current.resolve() == stop:
            return
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _move_run_contents(src_dir: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for item in list(src_dir.iterdir()):
        target = dest_dir / item.name
        if target.exists() or target.is_symlink():
            if item.is_symlink() or not item.is_dir():
                item.unlink()
                continue
            continue
        shutil.move(str(item), str(target))
    return dest_dir


def relocate_originals(
    dest_dir: Path,
    original_cache_path: Path,
    original_run_dir: Path | None = None,
    *,
    already_moved: dict[str, str] | None = None,
) -> dict[str, str]:
    """Make ``dest_dir`` the only copy of the published cache and run files.

    Duplicate hardlink names under the old run dir are unlinked. Remaining run
    artifacts are moved to ``dest_dir/run``. Shared originals (one P1 used by
    many techniques) are moved once.
    """
    dest_dir = dest_dir.resolve()
    dest_cache = dest_dir / "cache.pt"
    if not dest_cache.is_file():
        raise FileNotFoundError(dest_cache)

    original_cache_path = Path(original_cache_path)
    run_dir = Path(original_run_dir) if original_run_dir else original_cache_path.parent
    run_key = str(run_dir)
    moved_to = None
    if already_moved is not None and run_key in already_moved:
        moved_to = Path(already_moved[run_key])

    for name in ("cache.pt", "bg_stats.pt"):
        dest_file = dest_dir / name
        orig_file = run_dir / name
        if dest_file.is_file() and orig_file.is_file() and _same_file(dest_file, orig_file):
            orig_file.unlink()

    if original_cache_path.exists() and _same_file(original_cache_path, dest_cache):
        if original_cache_path.resolve() != dest_cache.resolve():
            original_cache_path.unlink()

    last = run_dir / "cache_last.pt"
    if last.exists() or last.is_symlink():
        try:
            if last.is_symlink() or _same_file(last, dest_cache):
                last.unlink()
        except OSError:
            last.unlink()

    dest_run = dest_dir / "run"
    if (
        run_dir.is_dir()
        and run_dir.resolve() != dest_dir
        and run_dir.resolve() != dest_run.resolve()
    ):
        if moved_to is None:
            _move_run_contents(run_dir, dest_run)
            parent = run_dir.parent
            _rmdir_parents_until(run_dir, CACHES_ROOT.parent)
            _rmdir_parents_until(parent, CACHES_ROOT.parent)
            moved_to = dest_run
            if already_moved is not None:
                already_moved[run_key] = str(dest_run)

    if (
        moved_to is not None
        and not dest_run.exists()
        and dest_run.resolve() != Path(moved_to).resolve()
    ):
        dest_run.symlink_to(moved_to, target_is_directory=True)

    last = dest_dir / "cache_last.pt"
    if last.exists() or last.is_symlink():
        last.unlink()
    last.symlink_to("cache.pt")
    return {
        "cache_path": str(dest_cache),
        "run_dir": str(moved_to or (dest_run if dest_run.exists() else dest_dir)),
    }


def publish_cache(
    src: str | Path,
    dataset: str,
    stage: int | str,
    technique: str,
    *,
    extra_files: dict[str, str | Path] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Hardlink ``src`` to ``.../cache.pt`` and write ``source.json``."""
    src_path = resolve_cache_file(src)
    dest_dir = cache_dir(dataset, stage, technique)
    dest = dest_dir / "cache.pt"
    _link_or_copy(src_path, dest)

    extras = extra_files or {}
    sibling_bg = src_path.parent / "bg_stats.pt"
    if "bg_stats.pt" not in extras and sibling_bg.is_file():
        extras = {**extras, "bg_stats.pt": sibling_bg}
    published_extras = {}
    for name, extra_src in extras.items():
        extra_path = Path(extra_src)
        if extra_path.is_file():
            _link_or_copy(extra_path.resolve(), dest_dir / name)
            published_extras[name] = str((dest_dir / name).resolve())

    source = {
        "dataset": dataset,
        "stage": stage_id(stage),
        "technique": technique,
        "cache_path": str(dest.resolve()),
        "original_cache_path": str(src_path),
        "original_run_dir": str(src_path.parent),
        "extras": published_extras,
    }
    if metadata:
        source["metadata"] = metadata
    (dest_dir / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    relocated = relocate_originals(dest_dir, src_path, src_path.parent)
    source["run_dir"] = relocated["run_dir"]
    source["original_cache_path"] = str(src_path)
    (dest_dir / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    return dest.resolve()


def rewrite_state_cache_path(state_path: str | Path, cache_path: str | Path) -> None:
    path = Path(state_path)
    data = json.loads(path.read_text())
    original = data.get("cache_path")
    data["original_cache_path"] = original or data.get("original_cache_path")
    data["cache_path"] = str(Path(cache_path).resolve())
    path.write_text(json.dumps(data, indent=2) + "\n")
