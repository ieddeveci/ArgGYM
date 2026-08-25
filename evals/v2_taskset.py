"""Freeze the ArgGYM_v2 grid into a reusable, provenance-stamped taskset.

ArgGYM_v2 generates and scores itself; this module only freezes what it
produces, so the eval roster sees byte-identical prompts and every published
number names the code that produced it.

Kept separate from `evals.taskset` (which freezes the v1 generator) because the
two benchmarks are not comparable: v2 has no content mode, its own task list,
its own scorers, and its own notion of a minimal answer. Sharing a builder would
invite averaging across them, which is not a quantity.

    python -m evals.v2_taskset --name v2-pilot --seeds 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "ArgGYM_v2"
sys.path.insert(0, str(ROOT))

from evals.taskset import git_dirty, git_sha  # noqa: E402


def _v2_export():
    """Import ArgGYM_v2's exporter with v2's own package directory winning.

    Both trees define a top-level `tasks` package. With the repo root ahead of
    ArgGYM_v2 on sys.path, v1's `tasks/` shadows v2's and the v2 exporter
    imports the wrong modules -- so put v2 first, and drop any `tasks`/`core`
    already bound to the other tree before and after the import. Deferred into a
    function so merely importing this module has no effect on sys.path.
    """
    for k in [k for k in sys.modules
              if k.split(".")[0] in ("tasks", "core", "aspic", "structures")]:
        del sys.modules[k]
    # v2 stays ahead of the repo root for the rest of the process: the exporter
    # imports `tasks.*` lazily inside _export_row, so a later re-import must
    # still resolve to v2. This module therefore commits the process to v2 --
    # which is why it is a separate entry point from the v1 builder.
    sys.path.insert(0, str(V2))
    from core import export  # ArgGYM_v2/core/export.py
    return export

# Every exportable v2 task. attack_defense contributes three separate tasks --
# attacking, defending and doing both at once are different moves, and a single
# blended score would hide which one a model can actually make.
TASKS = (
    "status_query", "formalization", "defeat_diagnosis", "claim_chain",
    "perturbation", "attack", "defence", "attack_defense",
    "preference_construction", "counter_argument", "counter_argument_strict",
    # Added with origin/main. Every other task asks about grounded semantics
    # only; this one also asks about preferred, stable and eager, so it is the
    # first thing here that tests whether a model tracks which semantics it is
    # being asked about. Roughly 35 minutes of build time, dominated by L15 at
    # ~43s an item.
    "semantics_query",
)

LEVELS = (3, 6, 9, 12, 15)
ORDERINGS = ("last_link_elitist", "weakest_link_elitist")


def sample_id(task: str, level: int, ordering: str, seed: int) -> str:
    """Stable per-item id. The ordering is abbreviated to keep ids readable;
    `ll`/`wl` map to last_link_elitist / weakest_link_elitist."""
    o = "ll" if ordering.startswith("last") else "wl"
    return f"{task}__{o}__L{level:02d}__{seed:03d}"


def build_rows(tasks: Iterable[str] = TASKS, levels: Iterable[int] = LEVELS,
               seeds: int = 20, progress=None) -> List[dict]:
    """Generate every (task, level, ordering, seed) item via the v2 exporter.

    Gold is verified at generation time: v2 scores each item's own reference
    answer and we refuse anything below 1.0, so a scorer that cannot read its
    own gold is caught here rather than being reported as model failure.
    """
    export = _v2_export()

    rows: List[dict] = []
    rejected: dict = {}
    gold_failures: List[tuple] = []

    for task in tasks:
        if task not in export._EXPORTABLE:
            raise SystemExit(f"unknown v2 task {task!r}; "
                             f"known: {sorted(export._EXPORTABLE)}")
        n_task = 0
        for level in levels:
            for ordering in ORDERINGS:
                for seed in range(seeds):
                    got = export._export_row(task, level, ordering, seed)
                    if got is None:
                        # The generator declined this (level, ordering, seed).
                        # Recorded rather than silently dropped: a cell that
                        # rejects often is a cell whose n is smaller than the
                        # nominal grid says, which changes its error bars.
                        key = f"{task}|L{level:02d}|{ordering}"
                        rejected[key] = rejected.get(key, 0) + 1
                        continue
                    item, refscore, goals, mind = got
                    if refscore < 0.999:
                        gold_failures.append((task, level, ordering, seed, refscore))
                    rows.append({
                        "sample_id": sample_id(task, level, ordering, seed),
                        "task": task, "level": level, "ordering": ordering,
                        "seed": seed, "prompt": item.prompt,
                        "reference": item.reference, "goals": goals,
                        "min_directives": mind, "metadata": item.metadata,
                        "reference_score": refscore,
                    })
                    n_task += 1
        if progress:
            progress(task, n_task, len(rows))

    if gold_failures:
        raise RuntimeError(
            f"gold self-check failed for {len(gold_failures)} item(s): the v2 "
            f"scorers do not accept their own reference answers, so no score "
            f"from this taskset would be trustworthy. "
            f"First few: {gold_failures[:5]}")
    return rows, rejected


def taskset_hash(rows: List[dict]) -> str:
    """Content hash over ids, prompts and gold.

    Gold is included as well as the prompt: an edit that leaves a question
    identical but changes its answer produces a different benchmark, and must
    not be able to reuse the old identity.
    """
    h = hashlib.sha256()
    for r in rows:
        for part in (r["sample_id"], r["prompt"], r["reference"]):
            h.update(part.encode("utf-8"))
            h.update(b"\0")
    return h.hexdigest()


def write_taskset(root: Path, name: str, rows: List[dict], rejected: dict,
                  seeds: int, allow_dirty: bool = False) -> Path:
    """Freeze `rows`, refusing a dirty working tree by default.

    A taskset's identity is the code that produced it. If that code was never
    committed, the recorded git_sha names a tree that does not regenerate it and
    nobody -- including its author later -- can say what was measured.
    `allow_dirty` stamps `dirty` into the directory name so a throwaway build
    can never be mistaken for a canonical one.
    """
    dirty = git_dirty()
    if dirty and not allow_dirty:
        raise RuntimeError(
            "refusing to build a taskset from a dirty working tree: the "
            "recorded git_sha would not reproduce it. Commit (or stash) first, "
            "or pass --allow-dirty for a throwaway local build.")

    full = taskset_hash(rows)
    built = datetime.now(timezone.utc)
    stamp = built.strftime("%Y%m%dT%H%M%SZ")
    tag = "-dirty" if dirty else ""
    out = Path(root) / f"{name}{tag}-{stamp}-{full[:8]}"
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "taskset.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    cells: dict = {}
    for r in rows:
        cells[f"{r['task']}|L{r['level']:02d}|{r['ordering']}"] = \
            cells.get(f"{r['task']}|L{r['level']:02d}|{r['ordering']}", 0) + 1

    manifest = {
        "name": name,
        "benchmark": "ArgGYM_v2",
        "taskset_hash": full,
        "taskset_id": out.name,
        "built_at": built.isoformat().replace("+00:00", "Z"),
        "git_sha": git_sha(),
        "git_dirty": dirty,
        "python": sys.version.split()[0],
        # Recorded because v1 shipped a taskset whose generation depended on
        # set-iteration order and therefore on this value.
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "<unset>"),
        "n_samples": len(rows),
        "n_cells": len(cells),
        "tasks": list(dict.fromkeys(r["task"] for r in rows)),
        "levels": sorted({r["level"] for r in rows}),
        "orderings": list(ORDERINGS),
        "seeds_per_cell": seeds,
        "cells": cells,
        "generator_rejections": rejected,
        "total_generator_rejections": sum(rejected.values()),
        "gold_self_check": "passed (every item's own reference scores 1.0)",
        "minimum_caveat": (
            "min_directives is minimal among the candidate directives the "
            "generator produced, not proven globally minimal. Efficiency is "
            "therefore an upper bound on the moves required: a model that finds "
            "a cheaper legal answer than gold is clamped to 1.0, never rewarded "
            "beyond it."),
        "no_content_mode": (
            "ArgGYM_v2 is symbolic only. There is no content rendering, so no "
            "score here is comparable with a v1 content-mode number."),
    }
    with open(out / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    return out


def load_taskset(taskset_dir: Path) -> List[dict]:
    import gzip
    d = Path(taskset_dir)
    plain, gz = d / "taskset.jsonl", d / "taskset.jsonl.gz"
    if plain.exists():
        opener, path = open, plain
    elif gz.exists():
        opener, path = (lambda p: gzip.open(p, "rt")), gz
    else:
        raise FileNotFoundError(f"no taskset.jsonl(.gz) in {d}")
    with opener(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="v2-pilot")
    ap.add_argument("--seeds", type=int, default=20,
                    help="items per (task, level, ordering) cell")
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--taskset-root", default="data/tasksets")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()

    def progress(task, n_task, total):
        print(f"  {task:26} {n_task:4d} items  (total {total})", flush=True)

    print(f"building v2 taskset '{a.name}': {len(a.tasks)} tasks x "
          f"{len(LEVELS)} levels x {len(ORDERINGS)} orderings x {a.seeds} seeds",
          flush=True)
    rows, rejected = build_rows(a.tasks, LEVELS, a.seeds, progress=progress)
    out = write_taskset(ROOT / a.taskset_root, a.name, rows, rejected, a.seeds,
                        allow_dirty=a.allow_dirty)
    print(f"\nwrote {len(rows)} samples to {out}")
    if rejected:
        print(f"generator declined {sum(rejected.values())} (level, ordering, "
              f"seed) combinations across {len(rejected)} cells")
    print(f"taskset_id: {out.name}")


if __name__ == "__main__":
    main()
