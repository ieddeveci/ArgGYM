"""Freeze a taskset from a spec.

The generator is pure: `ds[k]` is the item at seed `start + k`, and a seed that
builds nothing raises. Densifying is this module's job, because this is where it
can be recorded.

So a cell is filled by scanning seeds until it has what the spec asked for, and
the manifest records both halves: which seeds produced the items, and which were
skipped. That pair is a better determinism check than comparing hashes. Two
exports with the same skips and different hashes mean a renderer or a scorer
moved; different skips mean a generator moved. A hash alone cannot tell you
which.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from arggym.core.dataset import BuildFailed, TaskDataset
from arggym.core.rows import score as score_row
from arggym.core.spec import TasksetSpec, check_versions

#: Bumped when the rendered question changes for any reason. A prompt change is
#: a different taskset even when the theory behind it is identical.
PROMPT_VERSION = 1
#: Bumped when a scoring policy constant or weight changes. Those move a score
#: without moving a prompt, so the hash over prompts cannot see them.
SCORING_VERSION = 1


@dataclass
class CellReport:
    task: str
    level: int
    ordering: str
    seeds_used: List[int] = field(default_factory=list)
    seeds_skipped: List[Dict[str, Any]] = field(default_factory=list)
    scan_end: int = 0

    @property
    def acceptance(self) -> float:
        tried = len(self.seeds_used) + len(self.seeds_skipped)
        return len(self.seeds_used) / tried if tried else 0.0

    def key(self) -> str:
        return f"{self.task}|L{self.level}|{self.ordering}"

    def to_dict(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for s in self.seeds_skipped:
            counts[s["reason"]] = counts.get(s["reason"], 0) + 1
        return {"n": len(self.seeds_used), "seeds_used": self.seeds_used,
                "seeds_skipped": self.seeds_skipped, "reason_counts": counts,
                "scan_end": self.scan_end, "acceptance_rate": round(self.acceptance, 4)}


class CellUnfilled(SystemExit):
    """A cell could not produce what the spec asked for."""


class ReferenceNotVerified(SystemExit):
    """A row's own reference answer does not score 1.0 against it."""


def reference_score(entry: Dict[str, Any]) -> float:
    """What the row's reference answer scores against the row.

    Scored through the row rather than through the item, so this is also the
    check that the row carries everything its scorer reads: a row that lost a
    field either refuses or scores its own gold below 1.0, and either way the
    freeze stops. The old export computed the same number and counted how many
    reached 1.0 (`core/export.py`); a count nobody reads is a check that has
    already been lost, so this refuses instead.
    """
    return score_row(entry["reference_answer"], entry).score


def fill_cell(task: str, level: int, ordering: str, spec: TasksetSpec
              ) -> Tuple[List[Dict[str, Any]], CellReport]:
    """Scan seeds until the cell has `take` items, or fail naming the cell."""
    report = CellReport(task, level, ordering)
    rows: List[Dict[str, Any]] = []
    ds = TaskDataset(task, level, ordering, size=spec.seeds.scan_limit,
                     seed=spec.seeds.start, profile=spec.profile)
    for k in range(spec.seeds.scan_limit):
        if len(rows) == spec.seeds.take:
            break
        seed = spec.seeds.start + k
        report.scan_end = seed
        try:
            entry = ds[k]
        except BuildFailed:
            # The generator discards the reason today: `build` returns a bare
            # None and `make_item` loops over it. Recording the reason is #72,
            # and this dict is where it goes once `build` reports one.
            report.seeds_skipped.append({"seed": seed, "reason": "build_returned_none"})
            continue
        entry["metadata"]["reference_score"] = reference_score(entry)
        entry["metadata"]["source_index"] = len(rows)
        rows.append(entry)
        report.seeds_used.append(seed)

    if len(rows) < spec.seeds.take:
        raise CellUnfilled(
            f"{report.key()} yielded {len(rows)} of {spec.seeds.take} items within "
            f"{spec.seeds.scan_limit} seeds (acceptance {report.acceptance:.2f}). "
            f"Raise seeds.scan_limit, lower seeds.take, or find out why the cell "
            f"rejects -- a cell this thin is a finding, not a setting.")
    if report.acceptance < spec.min_acceptance:
        raise CellUnfilled(
            f"{report.key()} filled, but took {report.scan_end - spec.seeds.start + 1} "
            f"seeds for {spec.seeds.take} items (acceptance {report.acceptance:.2f}, "
            f"spec asks for {spec.min_acceptance}). A degraded cell should be a "
            f"decision: raise min_acceptance deliberately or fix the generator.")
    return rows, report


def taskset_hash(rows: List[Dict[str, Any]]) -> str:
    """A digest over everything a model sees and everything it is graded on.

    Metadata is included, so an edit that changes gold produces a new hash
    rather than silently overwriting a frozen taskset under the same name.
    """
    h = hashlib.blake2b(digest_size=16)
    for r in rows:
        h.update(json.dumps({"id": r["id"], "question": r["question"],
                             "reference_answer": r["reference_answer"],
                             "metadata": r["metadata"]},
                            sort_keys=True, default=str).encode())
    return h.hexdigest()


def _versions() -> Dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    import arggym
    from arggym.core.serialize import THEORY_SCHEMA

    try:
        pyarg = _pkg_version("python-argumentation")
    except PackageNotFoundError:  # pragma: no cover - depends on the install
        pyarg = None
    return {"arggym": arggym.__version__, "pyarg": pyarg,
            "prompt_version": PROMPT_VERSION, "scoring_version": SCORING_VERSION,
            "theory_schema": THEORY_SCHEMA,
            "python": sys.version.split()[0],
            "pythonhashseed": os.environ.get("PYTHONHASHSEED", "<unset>")}


def freeze(spec: TasksetSpec, path: str, verbose: bool = True) -> Dict[str, Any]:
    """Write one JSONL holding a manifest line and every row."""
    check_versions(spec)
    rows: List[Dict[str, Any]] = []
    cells: Dict[str, Any] = {}
    for task, level, ordering in spec.cells:
        got, report = fill_cell(task, level, ordering, spec)
        # source_index is the row's position in the whole file, not in its cell.
        for r in got:
            r["metadata"]["source_index"] = len(rows)
            rows.append(r)
        unverified = [r["id"] for r in got
                      if r["metadata"]["reference_score"] < 0.999]
        if unverified:
            raise ReferenceNotVerified(
                f"{report.key()}: the reference answer does not score 1.0 on "
                f"{len(unverified)} of {len(got)} rows ({', '.join(unverified[:3])}). "
                f"Either the row is missing something its scorer reads or the gold "
                f"is wrong; a taskset whose own answers fail is not publishable.")
        cells[report.key()] = report.to_dict()
        if verbose:
            thin = "" if report.acceptance == 1.0 else f"  ({report.acceptance:.0%} accepted)"
            print(f"  {report.key():54s} {len(got):3d} items{thin}")

    manifest = {
        "spec": spec.to_dict(),
        "versions": _versions(),
        "taskset_hash": taskset_hash(rows),
        "n_items": len(rows),
        # Every reference scores 1.0 or the freeze has already raised, so this
        # equals n_items and is written for the reader who wants to see it said.
        "n_verified": sum(1 for r in rows
                          if r["metadata"]["reference_score"] >= 0.999),
        "cells": cells,
        "n_skipped": sum(len(c["seeds_skipped"]) for c in cells.values()),
        # Travels with the rows because rows get separated from manifests the
        # moment anyone loads the JSONL.
        "minimum_caveat": ("min_directives is minimal among the candidate directives "
                           "the generator produced, not proven globally minimal"),
    }
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps({"__manifest__": manifest}) + "\n")
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    if verbose:
        print(f"wrote {len(rows)} items, {manifest['n_skipped']} seeds skipped -> {path}")
        print(f"taskset_hash {manifest['taskset_hash']}")
    return manifest


def read_manifest(path: str) -> Optional[Dict[str, Any]]:
    with open(path) as f:
        first = json.loads(f.readline())
    return first.get("__manifest__")
