"""Reading a frozen taskset, and slicing one without rebuilding it.

A taskset is a JSONL whose first line is a manifest and whose remaining lines are
rows (`arggym/core/freeze.py:217`). Everything a scorer needs is on the row, so
nothing here imports a generator.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import arggym


class TasksetCorrupt(SystemExit):
    """A taskset file does not hold what its manifest says it holds."""


def load(path: str, verify: bool = True) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """The manifest and the rows.

    The hash is recomputed from the rows rather than read out of the manifest.
    Comparing the manifest against a copy of itself answers "do these two
    strings match", not "are these the questions that hash names": a file
    truncated to eight lines loads seven rows, passes a string comparison, and
    gets a full taskset's hash printed beside numbers measured on seven items.
    Blake2b over 1440 rows costs milliseconds.

    A file with no manifest line is refused, because a taskset with no hash
    disables every check downstream at once and does it quietly: `score.py`
    compares the run's `None` against the file's `None`, finds them equal, and
    scores completions against whatever gold the file now holds. Row ids are
    stable across regenerations, so the unknown-id check cannot catch it either.
    `verify=False` reads such a file deliberately.
    """
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        first = json.loads(f.readline())
        manifest = first.get("__manifest__")
        if manifest is None:
            manifest, _ = {}, rows.append(first)
        rows.extend(json.loads(line) for line in f if line.strip())
    if verify:
        if not manifest:
            raise TasksetCorrupt(
                f"{path} has no manifest line, so it names no taskset_hash and "
                f"nothing downstream can tell it from the taskset a run was "
                f"actually generated against. Freeze it with `arggym freeze`, "
                f"or load it with verify=False and accept that its scores name "
                f"no questions.")
        check(path, manifest, rows)
    return manifest, rows


def check(path: str, manifest: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    """Refuse a file whose contents do not match its own manifest."""
    want_n = manifest.get("n_items")
    if want_n is not None and want_n != len(rows):
        raise TasksetCorrupt(
            f"{path} holds {len(rows)} rows and its manifest says {want_n}. A "
            f"short taskset scored against a full manifest publishes the whole "
            f"grid's identity beside a subset's numbers.")
    want = manifest.get("taskset_hash")
    if want is None:
        return
    got = arggym.taskset_hash(rows)
    if got != want:
        raise TasksetCorrupt(
            f"{path} hashes to {got} and its manifest claims {want}. The rows "
            f"have been edited, reordered or truncated since the freeze, so the "
            f"hash a report would cite names questions this file does not hold.")


def select(rows: Sequence[Dict[str, Any]], tasks: Optional[Sequence[str]] = None,
           levels: Optional[Sequence[int]] = None,
           orderings: Optional[Sequence[str]] = None,
           limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """A slice of a frozen taskset, by coordinate.

    Slicing rather than freezing a second taskset: two exports of "the same"
    grid are only the same if nothing moved in between, and a run that cites one
    hash for questions it did not ask is worse than one that cites a filter.

    A filter naming a value the taskset does not carry raises. Silently matching
    nothing would report a whole missing sweep as a clean run of zero items.
    """
    out = list(rows)
    for name, wanted, key in (("tasks", tasks, lambda r: r["task"]),
                              ("levels", levels, lambda r: r["metadata"]["level"]),
                              ("orderings", orderings,
                               lambda r: r["metadata"]["ordering"])):
        if wanted is None:
            continue
        wanted = list(wanted)
        present = {key(r) for r in out}
        missing = [w for w in wanted if w not in present]
        if missing:
            raise ValueError(
                f"{name}={missing} match no row here (available: "
                f"{sorted(present)}). A filter that matches nothing would report "
                f"a missing sweep as a clean run of zero items.")
        out = [r for r in out if key(r) in wanted]
    if limit is None:
        return out
    if limit <= 0:
        # Zero is not "no limit", it is a filter that matches nothing -- and it
        # would generate nothing, then be written down as a completed run with
        # an error rate of 0.0. Same refusal as a task name that is in no row,
        # for the same reason.
        raise ValueError(
            f"limit={limit} selects no rows. A run of zero items would be "
            f"recorded as a clean completed run; leave limit unset for all of "
            f"them.")
    return out[:limit]
