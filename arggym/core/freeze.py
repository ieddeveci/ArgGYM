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

Under the skip list sits the retry loop, and it is where the interesting number
lives. On the standard grid no seed is skipped at all, while `semantics_query` at
level 12 reaches its two items by discarding candidates six times in seven. So
the cell also records how many candidates `build` was asked for and why it
refused them (#72), counted over every seed rather than only the failed ones.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from arggym.core.build import reasons_text, top_reason
from arggym.core.dataset import TaskDataset
from arggym.core.prompting import MINIMALITY
from arggym.core.rows import score as score_row
from arggym.core.spec import TasksetSpec, check_versions

#: Bumped when the row a harness reads changes shape or wording: the rendered
#: question for any reason, and the metadata schema too. Either half moves it on
#: its own. Version 5 removed `answer_template`, which moved `taskset_hash`
#: without moving a single question; versions 6 and 7 each state rules the scorer
#: was already enforcing, and so move questions and no metadata at all -- the 280
#: engine-scored and `formalization` questions at 6, the remaining 200 at 7.
#: Version 8 is the narrowest kind: #121 stopped `formalization` writing a second
#: route to a single queried claim, which moves three of its 40 rows and leaves the
#: other 37 exactly as they were. "The rendered question for any reason" covers a
#: generator that writes a different theory as much as a renderer that words the
#: same one differently.
PROMPT_VERSION = 8
#: Bumped when a scoring policy constant or weight changes. Those move a score
#: without moving a prompt, so the hash over prompts cannot see them.
#:
#: A legality rule that the prompt states and the scorer enforces bumps both, the
#: way the bloat factor does (`docs/dataset-contract.md`): the prompt hash records
#: that the question changed, and this records that an unchanged question is now
#: scored differently. #89's name-collision rule is such a rule.
SCORING_VERSION = 4


@dataclass
class CellReport:
    task: str
    level: int
    ordering: str
    seeds_used: List[int] = field(default_factory=list)
    seeds_skipped: List[Dict[str, Any]] = field(default_factory=list)
    scan_end: int = 0
    #: Candidates `build` was asked for across every seed scanned, the accepted
    #: ones included, and why it discarded the ones it did.
    build_calls: int = 0
    build_rejections: Dict[str, int] = field(default_factory=dict)

    @property
    def acceptance(self) -> float:
        tried = len(self.seeds_used) + len(self.seeds_skipped)
        return len(self.seeds_used) / tried if tried else 0.0

    @property
    def build_acceptance(self) -> float:
        """Items per candidate built, which is the rate a thin cell shows up in.

        Seed acceptance hides the retry loop: every cell of the grid fills, and
        `semantics_query` at level 12 under `last_link_democratic` still reaches
        it by discarding 26 candidates out of 30.
        """
        return len(self.seeds_used) / self.build_calls if self.build_calls else 0.0

    def key(self) -> str:
        return f"{self.task}|L{self.level}|{self.ordering}"

    def to_dict(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for s in self.seeds_skipped:
            counts[s["reason"]] = counts.get(s["reason"], 0) + 1
        return {"n": len(self.seeds_used), "seeds_used": self.seeds_used,
                "seeds_skipped": self.seeds_skipped, "reason_counts": counts,
                "scan_end": self.scan_end, "acceptance_rate": round(self.acceptance, 4),
                "build_calls": self.build_calls,
                "build_rejections": dict(sorted(self.build_rejections.items())),
                "build_acceptance_rate": round(self.build_acceptance, 4)}


class CellUnfilled(SystemExit):
    """A cell could not produce what the spec asked for."""


class PromptClaimUnsupported(SystemExit):
    """A question states a rule the row gives the scorer no way to apply."""


class ReferenceNotVerified(SystemExit):
    """A row's own reference answer does not score 1.0 against it."""


def reference_score(entry: Dict[str, Any]) -> float:
    """What the row's reference answer scores against the row.

    Scored through the row rather than through the item, so this is also the
    check that the row carries everything its scorer reads: a row that lost a
    field either refuses or scores its own gold below 1.0, and either way the
    freeze stops. The pre-contract export computed the same number and counted
    how many reached 1.0; a count nobody reads is a check that has already been
    lost, so this refuses instead.
    """
    return score_row(entry["reference_answer"], entry).score


def minimum_unproven(entry: Dict[str, Any]) -> bool:
    """Whether the row's stated minimum is a bound the search gave up on.

    The six construction generators write `minimality_proven` into the item's
    statistics, which the row carries under `metadata.gold.metadata`. Only an
    explicit `False` counts: a task with no minimum search writes nothing there,
    and nothing is not a failed proof.
    """
    stats = (entry.get("metadata", {}).get("gold", {}) or {}).get("metadata") or {}
    return stats.get("minimality_proven") is False


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
        # `build_at` rather than `ds[k]`: the reasons behind a seed that
        # succeeded on its seventh candidate are the ones with a signal in them,
        # and an exception can only carry the reasons from a seed that failed
        # outright.
        entry, built = ds.build_at(k)
        report.build_calls += built.calls
        for reason, n in built.reasons.items():
            report.build_rejections[reason] = report.build_rejections.get(reason, 0) + n
        if entry is None:
            # `reason` names what this seed hit most often, which is what a skip
            # list is read for; the full count is in `build_rejections`.
            report.seeds_skipped.append({"seed": seed, "tries": built.calls,
                                         "reason": top_reason(built.reasons),
                                         "reasons": dict(sorted(built.reasons.items()))})
            continue
        if minimum_unproven(entry):
            # The minimum search ran out of budget and fell back to every
            # candidate it had, so `min_directives` is an upper bound rather than
            # a minimum: on the one such row of the standard grid it said 55
            # where 10 suffice, and the bloat gate admitted 110 directives where
            # it should admit 20 (#124). A reference that scores below 1.0 stops
            # the freeze; a stated minimum 5.5x the real one is the same class of
            # claim, and the flag that says so was read by nothing. The seed is
            # skipped and named, the way a seed that built nothing is.
            reasons = {**built.reasons, "minimality_unproven": 1}
            report.build_rejections["minimality_unproven"] = (
                report.build_rejections.get("minimality_unproven", 0) + 1)
            report.seeds_skipped.append({"seed": seed, "tries": built.calls,
                                         "reason": "minimality_unproven",
                                         "reasons": dict(sorted(reasons.items()))})
            continue
        entry["metadata"]["reference_score"] = reference_score(entry)
        entry["metadata"]["source_index"] = len(rows)
        rows.append(entry)
        report.seeds_used.append(seed)

    if len(rows) < spec.seeds.take:
        raise CellUnfilled(
            f"{report.key()} yielded {len(rows)} of {spec.seeds.take} items within "
            f"{spec.seeds.scan_limit} seeds (acceptance {report.acceptance:.2f}). "
            f"It discarded {report.build_calls - len(rows)} candidates: "
            f"{reasons_text(report.build_rejections)}. Raise seeds.scan_limit, lower "
            f"seeds.take, or fix what the reasons name -- a cell this thin is a "
            f"finding, not a setting.")
    if report.acceptance < spec.min_acceptance:
        raise CellUnfilled(
            f"{report.key()} filled, but took {report.scan_end - spec.seeds.start + 1} "
            f"seeds for {spec.seeds.take} items (acceptance {report.acceptance:.2f}, "
            f"spec asks for {spec.min_acceptance}). A degraded cell should be a "
            f"decision: raise min_acceptance deliberately or fix the generator.")
    if report.build_acceptance < spec.min_build_acceptance:
        # The guard above never fires on the standard grid, where every seed
        # builds and the retry loop hides how many candidates it threw away
        # (#114). This one reads the number the retry loop reports.
        raise CellUnfilled(
            f"{report.key()} filled, but kept {len(rows)} of {report.build_calls} "
            f"candidates (build acceptance {report.build_acceptance:.2f}, spec asks for "
            f"{spec.min_build_acceptance}). It discarded: "
            f"{reasons_text(report.build_rejections)}. A cell this thin is a finding, "
            f"not a setting: lower min_build_acceptance deliberately or fix what the "
            f"reasons name.")
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
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

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
        # The bloat rule is gated on `min_directives` in the scorer and asserted flatly
        # by every prompt that carries it. All six construction generators do set one,
        # but that was a property of six generators rather than a checked invariant, so
        # a prompt could come to promise a rule the scorer would skip.
        unsupported = [r["id"] for r in got
                       if MINIMALITY in r["question"]
                       and not r["metadata"].get("gold", {}).get("min_directives")]
        if unsupported:
            raise PromptClaimUnsupported(
                f"{report.key()}: {len(unsupported)} of {len(got)} rows state the "
                f"minimality rule and carry no min_directives, so the scorer would not "
                f"apply it ({', '.join(unsupported[:3])}). The question would be making "
                f"a promise the row cannot keep.")
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
            note = "" if report.acceptance == 1.0 else f"  ({report.acceptance:.0%} of seeds accepted)"
            # Printed whenever the cell built more candidates than it kept, which
            # seed acceptance never shows: every grid cell fills, and some fill by
            # discarding six candidates in seven.
            if report.build_calls > len(got):
                note += (f"  [{report.build_calls} candidates, "
                         f"{report.build_acceptance:.0%} kept: "
                         f"{reasons_text(report.build_rejections)}]")
            print(f"  {report.key():54s} {len(got):3d} items{note}")

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
        # Candidates `build` discarded, over every seed including the ones that
        # produced an item. `n_skipped` counts only seeds that ran out of tries,
        # and on the standard grid that number is zero while this one is not.
        "n_rejected": sum(sum(c["build_rejections"].values()) for c in cells.values()),
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
        print(f"wrote {len(rows)} items, {manifest['n_skipped']} seeds skipped, "
              f"{manifest['n_rejected']} candidates rejected -> {path}")
        print(f"taskset_hash {manifest['taskset_hash']}")
    return manifest


def read_manifest(path: str) -> Optional[Dict[str, Any]]:
    with open(path) as f:
        first = json.loads(f.readline())
    return first.get("__manifest__")
