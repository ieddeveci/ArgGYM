"""Reading a frozen taskset, and slicing one without rebuilding it.

A taskset is a JSONL whose first line is a manifest and whose remaining lines are
rows (`arggym/core/freeze.py:217`). Everything a scorer needs is on the row, so
nothing here imports a generator.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple


def load(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """The manifest and the rows. A file with no manifest line yields `{}`."""
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        first = json.loads(f.readline())
        manifest = first.get("__manifest__")
        if manifest is None:
            manifest, _ = {}, rows.append(first)
        rows.extend(json.loads(line) for line in f if line.strip())
    return manifest, rows


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
    return out[:limit] if limit else out
