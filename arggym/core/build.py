"""Why a build candidate was discarded, and how many it took.

`build` used to answer `None`, and a `None` says nothing. `claim_chain.build`
alone rejects at five distinct sites and the export recorded every one of them
as `build_returned_none`, so a cell that rejects most of what it builds reads
exactly like a healthy one. That is #72. `build` now answers `Rejected(reason)`
and the retry loop keeps the histogram.

**The histogram is per build call, not per failed seed.** A freeze over 96 grid
cells skipped no seed at all, so a record of fully-failed seeds would ship and
print nothing. The signal lives inside the retry loop: `semantics_query` at level
12 under `last_link_democratic` spends 30 candidates on its first four items, and
what it spends them on -- a theory over `MAX_DIRECTIVES`, 23 times out of 26 --
is the finding the issue asks for.

`make_item` keeps returning `Optional[Item]`. A sentinel is truthy and passes
`is not None`, so widening its return would turn some fifty `assert item is not
None` call sites in `tests/` and `arggym/inspector.py` vacuously green rather
than failing them. Callers that want the reasons ask for the report instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class Rejected:
    """One discarded build candidate, and the site that discarded it.

    `reason` is a short stable key rather than a sentence: it is a histogram
    bucket in the export manifest, so it names the check that failed and not the
    values that reached it. Two builds rejected for the same cause must produce
    the same string or the histogram stops counting.
    """

    reason: str


@dataclass(frozen=True)
class BuildReport:
    """One seed's trip through the retry loop.

    `calls` counts candidates asked for, including the accepted one, so a cell
    can report build-level acceptance beside seed-level acceptance.
    """

    item: Optional[Any] = None
    calls: int = 0
    reasons: Dict[str, int] = field(default_factory=dict)


def retry(candidate: Callable[[int], Any], tries: int) -> BuildReport:
    """Ask `candidate(k)` for an item until one comes back, counting refusals.

    The nine task modules differ only in the seed mixer they pass, so the loop
    itself lives here: a per-module copy would be nine places for the reason
    counting to drift.
    """
    reasons: Dict[str, int] = {}
    for k in range(tries):
        out = candidate(k)
        if isinstance(out, Rejected):
            reasons[out.reason] = reasons.get(out.reason, 0) + 1
            continue
        if out is None:
            # A rejection with no reason is what #72 removed, so this is a
            # `return None` that survived the edit rather than an empty cell.
            raise TypeError(
                f"build answered None on try {k}; a rejected candidate must say "
                f"why by returning Rejected(...)")
        return BuildReport(out, k + 1, reasons)
    return BuildReport(None, tries, reasons)


def reasons_text(counts: Dict[str, int]) -> str:
    """A reason histogram as one line, commonest first."""
    if not counts:
        return "no rejections"
    return ", ".join(f"{r}x{n}" for r, n in
                     sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def top_reason(counts: Dict[str, int]) -> str:
    """The reason a seed hit most often, ties broken by name so it is stable.

    Called only where a seed produced no item, so every try recorded a reason and
    `counts` is never empty.
    """
    return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
