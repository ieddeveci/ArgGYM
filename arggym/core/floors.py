"""What a model scores by not reasoning.

A benchmark number means nothing without the number an uninformed answer gets.
Measured on the v1 pilot, one constant string scored 0.490 on a task whose
metric took three values, so any result near 0.5 there carried almost no
information and anything below it was worse than a fixed reply.

A floor is a property of the scorer, not of the tasks, so it has to be
re-measured whenever a scoring policy changes -- which is what `scoring_version`
is for.

The strategies are deliberately dumb, and the line they stay behind is stated
rather than left to whatever the search happens to try: a strategy fixes, once
per task, the answer it gives to each coordinate the answer format exposes, and
reads the question only to learn which coordinates are asked. It never inspects
the theory to decide what to answer (`docs/dataset-contract.md` section 10). A
search that crossed that line would report a solver's score as the floor, and
past it there is no stopping point short of a full solver.

One constant for the whole item is not the widest fixed map that line allows.
Where the answer key carries a component the task holds fixed -- `semantics_query`
asks its claims under five semantics at once -- the answer a reader would reach
for is one constant per semantics, and searching one constant per item put that
floor at 0.459 where it is 0.671 (#95).
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from arggym.core import registry
from arggym.core.rows import row_task, score

#: Statuses a label-map task can answer with.
_STATUSES = ("justified", "overruled", "undecided")

#: What one key group of a fitted map may answer. `no stable extension` is the
#: answer `semantics_query` asks for when a theory has no stable extension
#: (`tasks/semantics_query.py:_render_prompt`), and leaving it out of the search
#: was the same defect as searching one constant per item: an answer the format
#: offers that nothing ever tries. Every group is offered it rather than only
#: the stable one, so the search needs no table of which vocabulary belongs to
#: which group -- a group whose gold never says it cannot fit it.
_CANDIDATES = _STATUSES + ("no stable extension",)


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
    # uninformed answer names every literal the theory mentions -- which is
    # premises and axioms *and* rule consequents, since a rule's conclusion is
    # a claim that can change status. Reading only the premises understates
    # this floor.
    lits = re.findall(r"\[(?:premise|axiom):\s*(-?\w+)\]", q)
    lits += re.findall(r"(?:=>|->)\s*(-?\w+)\]", q)
    return sorted(set(lits))


def _key_parts(coord: str) -> Tuple[str, ...]:
    """One asked coordinate, split into the components the scorer keys on.

    `semantics_query` keys its gold by (claim, semantics) and both the question
    and the answer write that pair as `<claim> under <semantics>`
    (`tasks/semantics_query.py:_PAIR`). Every other ask is a bare claim, so one
    component.
    """
    m = re.fullmatch(r"(.+?)\s+under\s+(.+)", coord)
    return (m.group(1), m.group(2)) if m else (coord,)


def _key_groups(rows: Sequence[Dict[str, Any]]) -> Tuple[int, ...]:
    """Which components of the answer key the task fixes rather than the item.

    A key group is a component drawn from a vocabulary the task holds fixed:
    `semantics_query`'s semantics is five names scheduled by level
    (`tasks/semantics_query.py:SEMANTICS_BY_LEVEL`), while its claim is sampled
    per item from a pool of two thousand. One constant per group is a map an
    uninformed answerer can write down before seeing an item; one constant per
    sampled claim would be the gold, spelled differently.

    Derived from the asked coordinates, so no per-task table decides it. The
    test is that a single item shows the whole vocabulary the task uses: a name
    the item drew makes the pooled vocabulary grow with the rows, and a name the
    task gave it does not. Two items are the minimum -- one cannot tell those
    apart -- and where no component passes, the search stays on whole-item
    constants.

    On the shipped grid that leaves ten tasks untouched and one group worth
    something, `semantics_query`'s semantics. `formalization` also passes, on
    the two literals `_asked` reads out of the answer-format example rather than
    out of a question that lists none; they are the same two on every item, so
    the rule is right about the vocabulary and wrong about what it is reading.
    Nothing moves either way -- that task scores an operation list, and a label
    line scores zero whichever status it carries.
    """
    keys = [[_key_parts(c) for c in _asked(row)] for row in rows]
    keys = [k for k in keys if k]
    arity = {len(k) for row in keys for k in row}
    if len(keys) < 2 or len(arity) != 1:
        return ()
    out = []
    for i in range(arity.pop()):
        per_row = [{k[i] for k in row} for row in keys]
        if len(set().union(*per_row)) <= max(len(s) for s in per_row):
            out.append(i)
    return tuple(out)


def _group_of(coord: str, idx: Sequence[int]) -> Tuple[str, ...]:
    return tuple(_key_parts(coord)[i] for i in idx)


def _by_group(row: Dict[str, Any], idx: Sequence[int]) -> Dict[Tuple[str, ...], List[str]]:
    """The row's asked coordinates, bucketed by the key group they belong to."""
    out: Dict[Tuple[str, ...], List[str]] = {}
    for coord in _asked(row):
        out.setdefault(_group_of(coord, idx), []).append(coord)
    return out


def _lines(coords: Sequence[str], status: str) -> str:
    return "\n".join(f"{c}: {status}" for c in coords)


def _constant_labels(status: str) -> Callable[[Dict[str, Any]], str]:
    def make(row: Dict[str, Any]) -> str:
        return _lines(_asked(row), status)
    return make


def _copy_theory(row: Dict[str, Any]) -> str:
    """Hand back the theory as the answer.

    The laziest non-empty answer available, and the one a substring-matching
    scorer would reward.
    """
    q = row["question"]
    return "\n".join(re.findall(r"^\[[^\]]*\]$", q, re.M))


def _fit_per_key_group(rows: Sequence[Dict[str, Any]]
                       ) -> Optional[Tuple[Callable[[Dict[str, Any]], str], Dict[str, str]]]:
    """One constant per key group, fitted on these rows, confirmed by the scorer.

    Greedy: each group takes the candidate that scores best on the coordinates
    that group owns, and the groups are fitted one at a time. That is the exact
    argmax while the metric is linear in per-line correctness, which is where
    the only task with more than one group sits -- a constant map answers every
    asked coordinate exactly once, so `|pred| = |gold|` and `pair_f1` reduces to
    `tp / n` (`core/pairs.py`). Any cross-line term breaks the decomposition:
    the bloat gate and the all-or-nothing `success` in `core/scoring.py`,
    `defeat_diagnosis`'s `0.85*f1 + 0.15*status_ok`, or an item that asked one
    coordinate twice.

    So greedy only proposes. The number that comes back is the fitted map put
    through the real `score()`, on the same text a model would submit, which
    leaves the correctness of the floor independent of the argument above and
    only its optimality resting on it. A scorer that later grows a cross-line
    term makes this floor merely not the best fixed map, which is the failure
    worth having.

    The fit reads scores and never gold: a group's candidates are ranked by
    submitting that group's lines alone, and every candidate then answers the
    same coordinates, so ranking those scores ranks the number of correct lines.

    The map is fitted on the rows it is then reported over, which is the same
    transduction the `max` over the fixed strategies already does, with a few
    more parameters. It costs something on a small group: five groups fitted
    over the eight rows of one level is optimistic where the same five over
    forty are not.
    """
    idx = _key_groups(rows)
    if not idx:
        # One group is the whole-item constant the fixed strategies already try,
        # so there is nothing here they do not measure.
        return None

    plan = [_by_group(row, idx) for row in rows]
    fitted: Dict[Tuple[str, ...], str] = {}
    for group in sorted({g for row in plan for g in row}):
        asked = [(row, groups[group]) for row, groups in zip(rows, plan) if group in groups]
        fitted[group] = max(_CANDIDATES, key=lambda c: sum(
            score(_lines(coords, c), row).score for row, coords in asked))

    def make(row: Dict[str, Any]) -> str:
        # In the order the question asks, not grouped: the answer a model
        # submits follows the ask list, and the fit is what differs from a
        # constant, not the shape of the text.
        return "\n".join(f"{coord}: {fitted[_group_of(coord, idx)]}"
                         for coord in _asked(row))

    return make, {" ".join(g): s for g, s in fitted.items()}


STRATEGIES: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "empty": lambda row: "",
    "copy_theory": _copy_theory,
    "none": lambda row: "none",
    **{f"all_{s}": _constant_labels(s) for s in _STATUSES},
}

#: Strategies whose map is fitted on the rows being measured rather than named
#: in advance. Each returns the answerer and the map it fitted, or `None` where
#: the task offers it nothing the fixed strategies do not already cover.
FITTED: Dict[str, Callable[[Sequence[Dict[str, Any]]],
                           Optional[Tuple[Callable[[Dict[str, Any]], str],
                                          Dict[str, str]]]]] = {
    "per_key_group": _fit_per_key_group,
}


def _mean(make: Callable[[Dict[str, Any]], str], rows: Sequence[Dict[str, Any]]) -> float:
    total = 0.0
    for row in rows:
        # Deliberately not guarded. A row this build cannot score -- a
        # different engine, a missing field, an unknown schema -- must stop
        # the measurement, not contribute a zero. Swallowing those made
        # every task report a floor of 0.000 and exit successfully, which
        # reads exactly like a task where guessing does not pay: the
        # failure this module exists to prevent.
        total += score(make(row), row).score
    return round(total / len(rows), 4) if rows else 0.0


def floor_for(rows: Sequence[Dict[str, Any]]
              ) -> Tuple[float, str, Dict[str, float], Optional[Dict[str, str]]]:
    """The best a fixed-map strategy does on these rows, and the map it fixed.

    The floor is the best of them, not the average: a reader comparing a model
    against chance is asking whether the model beat the easiest thing that
    works, and the easiest thing is whichever of these happens to fit.

    A tie goes to the fixed strategies, which come first: where a fitted map
    reaches nothing a named constant does not, the named one is the better
    description of the same number.
    """
    means: Dict[str, float] = {name: _mean(make, rows)
                               for name, make in STRATEGIES.items()}
    details: Dict[str, Dict[str, str]] = {}
    for name, fit in FITTED.items():
        got = fit(rows)
        if got is None:
            continue
        make, details[name] = got
        means[name] = _mean(make, rows)
    best = max(means, key=lambda k: means[k])
    return means[best], best, means, details.get(best)


def floors(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per task, because a floor is not a property of the benchmark."""
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(row_task(row), []).append(row)

    out: Dict[str, Dict[str, Any]] = {}
    for task in sorted(by_task):
        value, best, means, detail = floor_for(by_task[task])
        out[task] = {"floor": value, "strategy": best, "detail": detail,
                     "n": len(by_task[task]),
                     "answer_shape": registry.get(task).answer_shape,
                     "by_strategy": means}
    return out


def floor_strategy(entry: Dict[str, Any]) -> str:
    """How a floor was reached, as one string to print beside the number.

    The number alone does not say whether it is plausible as *the* uninformed
    baseline: `0.0000 empty` means the search found nothing that fits the answer
    format, which is a different sentence from "guessing does not pay here".
    A fitted strategy's name does not say what it fitted either, so the map
    comes with it.
    """
    detail = entry.get("detail")
    if not detail:
        return entry["strategy"]
    body = ", ".join(f"{k}={v}" for k, v in sorted(detail.items()))
    return f"{entry['strategy']}({body})"


def corrected(score_value: float, floor: float) -> float:
    """A score rescaled so chance is zero and perfect is one.

    The only defensible way to put twelve metrics on one axis, and still not a
    reason to average them.
    """
    if floor >= 1.0:
        return 0.0
    return round((score_value - floor) / (1.0 - floor), 4)
