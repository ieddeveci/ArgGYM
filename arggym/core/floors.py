"""What a model scores by not reasoning.

A benchmark number means nothing without the number an uninformed answer gets.
Measured on the v1 pilot, one constant string scored 0.490 on a task whose
metric took three values, so any result near 0.5 there carried almost no
information and anything below it was worse than a fixed reply.

A floor is a property of the scorer, not of the tasks, so it has to be
re-measured whenever a scoring policy changes -- which is what `scoring_version`
is for.

The strategies are deliberately dumb. They are not attacks on the scorer; they
are what a model that read the answer format and nothing else can produce.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Sequence, Tuple

from arggym.core import registry
from arggym.core.rows import row_task, score

#: Statuses a label-map task can answer with.
_STATUSES = ("justified", "overruled", "undecided")


def _asked(row: Dict[str, Any]) -> List[str]:
    """The keys a label-map question asks about, read from the question.

    From the question rather than from `metadata.gold`, because a strategy that
    read the gold would not be uninformed. Three tasks phrase the ask three
    ways, and a strategy that matched only one of them would report a floor of
    zero for the other two -- an unmeasured floor being worse than none.
    """
    q = row["question"]
    m = re.search(r"State the status of each of the following claims: (.+?)\.\n", q)
    if m:  # status_query
        return [c.strip() for c in m.group(1).split(",")]
    m = re.search(r"NAMED BESIDE IT:\n((?:   .+\n)+)", q)
    if m:  # semantics_query: "<claim> under <semantics>"
        return [line.strip() for line in m.group(1).splitlines() if line.strip()]
    # perturbation asks which claims changed, and does not list them. The
    # uninformed answer is every literal the theory mentions.
    lits = re.findall(r"\[(?:premise|axiom):\s*(-?\w+)\]", q)
    return sorted(set(lits))


def _constant_labels(status: str) -> Callable[[Dict[str, Any]], str]:
    def make(row: Dict[str, Any]) -> str:
        return "\n".join(f"{c}: {status}" for c in _asked(row))
    return make


def _copy_theory(row: Dict[str, Any]) -> str:
    """Hand back the theory as the answer.

    The laziest non-empty answer available, and the one a substring-matching
    scorer would reward.
    """
    q = row["question"]
    return "\n".join(re.findall(r"^\[[^\]]*\]$", q, re.M))


STRATEGIES: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "empty": lambda row: "",
    "copy_theory": _copy_theory,
    "none": lambda row: "none",
    **{f"all_{s}": _constant_labels(s) for s in _STATUSES},
}


def floor_for(rows: Sequence[Dict[str, Any]]) -> Tuple[float, str, Dict[str, float]]:
    """The best a constant strategy does on these rows.

    The floor is the best of them, not the average: a reader comparing a model
    against chance is asking whether the model beat the easiest thing that
    works, and the easiest thing is whichever of these happens to fit.
    """
    means: Dict[str, float] = {}
    for name, make in STRATEGIES.items():
        total = 0.0
        for row in rows:
            try:
                total += score(make(row), row).score
            except Exception:
                # A strategy that cannot even be scored contributes nothing,
                # which is the honest reading of "this answer is worthless".
                pass
        means[name] = round(total / len(rows), 4) if rows else 0.0
    best = max(means, key=lambda k: means[k])
    return means[best], best, means


def floors(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per task, because a floor is not a property of the benchmark."""
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(row_task(row), []).append(row)

    out: Dict[str, Dict[str, Any]] = {}
    for task in sorted(by_task):
        value, best, means = floor_for(by_task[task])
        out[task] = {"floor": value, "strategy": best, "n": len(by_task[task]),
                     "answer_shape": registry.get(task).answer_shape,
                     "by_strategy": means}
    return out


def corrected(score_value: float, floor: float) -> float:
    """A score rescaled so chance is zero and perfect is one.

    The only defensible way to put twelve metrics on one axis, and still not a
    reason to average them.
    """
    if floor >= 1.0:
        return 0.0
    return round((score_value - floor) / (1.0 - floor), 4)
