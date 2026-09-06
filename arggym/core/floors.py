"""What a model scores by not reasoning.

A benchmark number means nothing without the number an uninformed answer gets.
Measured on the v1 pilot, one constant string scored 0.490 on a task whose
metric took three values, so any result near 0.5 there carried almost no
information and anything below it was worse than a fixed reply.

A floor is a property of the scorer and of this search, not of the tasks, so it
has to be re-measured whenever either moves: `scoring_version` covers the
scorer, `FLOORS_VERSION` covers what is searched here.

The strategies are deliberately dumb, and the line they stay behind is stated
rather than left to whatever the search happens to try: a strategy fixes, once
per task, the answer it gives to each coordinate the answer format exposes, and
reads the question only to learn which coordinates are asked. It never inspects
the theory to decide what to answer (`docs/dataset-contract.md` section 10). A
search that crossed that line would report a solver's score as the floor, and
past it there is no stopping point short of a full solver.

One constant for the whole item is not the widest fixed map that line allows.
Where the answer key carries a component the task holds fixed -- `semantics_query`
asks its claims under as many as five semantics at once -- the answer a reader
would reach for is one constant per semantics, and searching one constant per
item put that floor at 0.459 where it is 0.671 (#95).
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

from arggym.core import registry
from arggym.core.rows import row_task, score

#: Bumped when the search changes: a strategy added or dropped, a candidate
#: added, the key-group rule moved. A floor moves with the scorer and with the
#: search, and `scoring_version` sees only the first, so two scoring artifacts
#: carrying different floors for the same rows are otherwise identical.
FLOORS_VERSION = 1

#: Statuses a label-map task can answer with.
_STATUSES = ("justified", "overruled", "undecided")

#: What one key group of a fitted map may answer. `no stable extension` is the
#: answer `semantics_query` asks for when a theory has no stable extension
#: (`tasks/semantics_query.py:_render_prompt`), and leaving it out of the search
#: was the same defect as searching one constant per item: an answer the format
#: offers that nothing ever tries. Which groups the format extends it to is a
#: fact about one prompt's wording, and a second parser for that wording is a
#: thing to keep in step with the first, so every group is offered every
#: candidate and the fit drops the ones that do not pay.
_CANDIDATES = _STATUSES + ("no stable extension",)

#: Answered coordinates a fitted map must have per entry. It prices a map that
#: is really the gold rather than forbidding one -- enough rows repeating an ask
#: list buy any number of entries -- so what forbids it is `FittedFloorIsGold`
#: below, and what makes it hard to reach is the cap. Measured on
#: `semantics_query`: five entries over the 274 coordinates of the whole task
#: (55 each) reproduce themselves on a fit that never saw the rows, five over
#: the 56 of one level (11 each) buy up to 0.040.
MIN_COORDS_PER_ENTRY = 20

#: Entries a fitted map may have at all, whatever the row count. The vocabularies
#: a task fixes are small -- five semantics on `semantics_query`, two literals on
#: `formalization` -- while a map keyed by anything an item samples has as many
#: entries as that item has coordinates, dozens. A count is a harder invariant to
#: slip past than a ratio: 24 rows sharing one ask list clear the budget and
#: report the majority gold per coordinate as a floor, and this refuses them.
MAX_MAP_ENTRIES = 8


class Fit(NamedTuple):
    """What a fitted strategy came back with.

    Three cases, and a reader of a floor has to be able to tell them apart: a
    map, no key group at all, and a key group found and then refused. The last
    one leaves the floor at a constant, which is the same number a task with no
    group would report, so the reason travels with the result and
    `floor_strategy` prints it.
    """

    make: Optional[Callable[[Dict[str, Any]], str]] = None
    detail: Optional[Dict[str, str]] = None
    refused: Optional[str] = None


class FittedFloorIsGold(RuntimeError):
    """A fitted map reproduced the gold, so it is not a floor.

    Raised rather than returned. A floor of 1.000 makes `corrected` zero for
    every score in the group (`corrected`, below), which reads as a task no
    model can beat rather than as a measurement that went wrong.
    """


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


def _key_groups(rows: Sequence[Dict[str, Any]]
                ) -> Tuple[Tuple[int, ...], Optional[str]]:
    """Which components of the answer key the task fixes rather than the item.

    Returns the components, and where a set of them was found and then refused
    for the size of the map it implies, why.

    A key group is a component drawn from a vocabulary the task holds fixed:
    `semantics_query`'s semantics is five names scheduled by level
    (`tasks/semantics_query.py:SEMANTICS_BY_LEVEL`), while its claim is sampled
    per item from a pool of two thousand. One constant per group is a map an
    uninformed answerer can write down before seeing an item; one constant per
    sampled claim would be the gold, spelled differently.

    Two tests decide it, both read off the asked coordinates, neither a per-task
    table.

    **The vocabulary does not grow with the rows.** One item shows the whole of
    it: a name the item drew makes the pooled set larger than any single row's,
    a name the task gave it does not. Two items are the minimum, since one
    cannot tell those apart.

    **The map stays small**: at most `MAX_MAP_ENTRIES` entries, and at least
    `MIN_COORDS_PER_ENTRY` answered coordinates for each of them. The first test
    alone accepts the *claim* as soon as a set of rows repeats an ask list --
    which a filtered or partial run produces, since a report measures per
    reporting group over the rows a run actually scored. The map over (claim,
    semantics) is then the gold in another notation: it reports a floor of
    1.000, and `corrected` turns every score in the group into 0.0 without
    raising.

    Both numbers, because they fail differently. The ratio prices a gold fit;
    enough repeated rows pay for it. The cap says what a task-fixed vocabulary
    looks like -- a handful of names -- and no row count buys past it.

    Refusal comes back as a reason rather than as silence. A floor that fell
    back to a constant because the map was priced out reads exactly like a floor
    on a task with no key group at all, which is the ambiguity `floor_strategy`
    exists to remove.

    On the shipped grid this leaves ten tasks on whole-item constants and one
    group worth something, `semantics_query`'s semantics. `formalization` also
    passes, on the two literals `_asked` reads out of the answer-format example
    rather than out of a question that lists none; they are the same two on
    every item, so the rule is right about the vocabulary and wrong about what
    it is reading. Nothing moves either way -- that task scores an operation
    list, and a label line is worth zero whichever status it carries.
    """
    keys = [[_key_parts(c) for c in _asked(row)] for row in rows]
    keys = [k for k in keys if k]
    arity = {len(k) for row in keys for k in row}
    if len(keys) < 2 or len(arity) != 1:
        return (), None
    out = []
    for i in range(arity.pop()):
        per_row = [{k[i] for k in row} for row in keys]
        if len(set().union(*per_row)) <= max(len(s) for s in per_row):
            out.append(i)
    if not out:
        return (), None
    entries = {tuple(k[i] for i in out) for row in keys for k in row}
    coords = sum(len(row) for row in keys)
    if len(entries) > MAX_MAP_ENTRIES:
        return (), f"{len(entries)} entries, over the cap of {MAX_MAP_ENTRIES}"
    if len(entries) * MIN_COORDS_PER_ENTRY > coords:
        return (), (f"{len(entries)} entries over {coords} coordinates, under one "
                    f"per {MIN_COORDS_PER_ENTRY}")
    return tuple(out), None


def _group_of(coord: str, idx: Sequence[int]) -> Tuple[str, ...]:
    return tuple(_key_parts(coord)[i] for i in idx)


def _by_group(row: Dict[str, Any], idx: Sequence[int]) -> Dict[Tuple[str, ...], List[str]]:
    """The row's asked coordinates, bucketed by the key group they belong to."""
    out: Dict[Tuple[str, ...], List[str]] = {}
    for coord in _asked(row):
        out.setdefault(_group_of(coord, idx), []).append(coord)
    return out


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


def _fit_per_key_group(rows: Sequence[Dict[str, Any]]) -> Fit:
    """One constant per key group, fitted on these rows, confirmed by the scorer.

    Each group takes the candidate that scores best with the other groups
    pinned, and one pass over the groups is the exact argmax while two things
    hold: every asked coordinate is answered exactly once, so `|pred|` is the
    same whichever candidates the map carries; and the row's score is then an
    affine function of its correct lines with a weight that does not move with
    them, which `pair_f1` gives as `2*tp / (|pred| + |gold|)`
    (`core/pairs.py`). Under those two the objective separates across groups and
    one pass reaches the maximum. What breaks it is a cross-line term -- the
    bloat gate and the all-or-nothing `success` in `core/scoring.py`, or
    `defeat_diagnosis`'s `0.85*f1 + 0.15*status_ok`. Rounding survives as a
    nuisance rather than a hole: over 70 buckets of `semantics_query` rows this
    search found an exhaustive one's argmax every time, and on one bucket the
    winning map's mean landed on 0.71875, where the fourth decimal goes either
    way with the order the rows are added. Ranking a group by its own lines
    alone does NOT hold: a partial answer changes `|pred|`, which reweights the
    rows, and that search misses maps an exhaustive one finds.

    So greedy proposes and the real scorer disposes. Every ranking here is a
    whole answer put through `score()`, the same text a model would submit, and
    so is the number that comes back. A scorer that later grows a cross-line
    term leaves this floor true and merely stops it being the best fixed map.

    The fit reads scores and never gold. It is still fitted, so it belongs to
    the rows it saw: fit it once over a whole task and hand the map down
    (`floors`), rather than refitting on the eight rows of a level and then
    correcting those same eight by it.
    """
    idx, refused = _key_groups(rows)
    if not idx:
        # One group is the whole-item constant the fixed strategies already try,
        # so there is nothing here they do not measure -- but say so where a
        # group set was found and priced out, since that floor is understated.
        return Fit(refused=refused)

    groups = sorted({g for row in rows for g in _by_group(row, idx)})

    def answer(mapping: Dict[Tuple[str, ...], str], row: Dict[str, Any]) -> str:
        out = []
        for coord in _asked(row):
            group = _group_of(coord, idx)
            if group not in mapping:
                raise KeyError(
                    f"the fitted map has no status for the key group "
                    f"{' '.join(group)!r}, so it was fitted on rows that never "
                    f"ask it and these rows are not the ones it belongs to. "
                    f"Filling the gap with a default would report a floor for a "
                    f"strategy nobody searched. It has: "
                    f"{sorted(' '.join(g) for g in mapping)}")
            out.append(f"{coord}: {mapping[group]}")
        return "\n".join(out)

    def total(mapping: Dict[Tuple[str, ...], str]) -> float:
        return sum(score(answer(mapping, row), row).score for row in rows)

    fitted = {g: _CANDIDATES[0] for g in groups}
    for group in groups:
        fitted[group] = max(_CANDIDATES, key=lambda c: total({**fitted, group: c}))

    if total(fitted) / len(rows) >= 1.0:
        raise FittedFloorIsGold(
            f"a map of {len(fitted)} entries scored 1.000 on the {len(rows)} rows "
            f"it was fitted over, so it is the gold rather than a floor: "
            f"{ {' '.join(g): s for g, s in fitted.items()} }")

    def make(row: Dict[str, Any]) -> str:
        return answer(fitted, row)

    return Fit(make, {" ".join(g): s for g, s in fitted.items()})


STRATEGIES: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "empty": lambda row: "",
    "copy_theory": _copy_theory,
    "none": lambda row: "none",
    **{f"all_{s}": _constant_labels(s) for s in _STATUSES},
}

#: Strategies whose map is fitted on rows rather than named in advance. Each
#: returns a `Fit`: the answerer and the map, or an empty one where the task
#: offers nothing the fixed strategies do not already cover.
FITTED: Dict[str, Callable[[Sequence[Dict[str, Any]]], Fit]] = {
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


def floor_for(rows: Sequence[Dict[str, Any]],
              fit_rows: Optional[Sequence[Dict[str, Any]]] = None
              ) -> Tuple[float, str, Dict[str, float], Optional[Dict[str, str]],
                         Dict[str, str]]:
    """The best a fixed-map strategy does on these rows, and the map it fixed.

    The floor is the best of them, not the average: a reader comparing a model
    against chance is asking whether the model beat the easiest thing that
    works, and the easiest thing is whichever of these happens to fit.

    `fit_rows` is where a fitted strategy's map comes from, defaulting to the
    rows being measured. The two differ wherever a report corrects a subgroup:
    the map is a property of the task, the number is a measurement of the rows.

    A tie goes to the fixed strategies, which come first: where a fitted map
    reaches nothing a named constant does not, the named one is the better
    description of the same number.

    The last value is what was refused and why, which is not the same as never
    having been tried: a floor that fell back to a constant because its map was
    priced out is understated, and only that string says so.
    """
    means: Dict[str, float] = {name: _mean(make, rows)
                               for name, make in STRATEGIES.items()}
    details: Dict[str, Dict[str, str]] = {}
    refused: Dict[str, str] = {}
    for name, fit in FITTED.items():
        got = fit(rows if fit_rows is None else fit_rows)
        if got.refused:
            refused[name] = got.refused
        if got.make is None:
            continue
        details[name] = got.detail
        means[name] = _mean(got.make, rows)
    best = max(means, key=lambda k: means[k])
    return means[best], best, means, details.get(best), refused


def floors(rows: Sequence[Dict[str, Any]],
           fit_rows: Optional[Sequence[Dict[str, Any]]] = None
           ) -> Dict[str, Dict[str, Any]]:
    """Per task, because a floor is not a property of the benchmark.

    `fit_rows` is the taskset a fitted strategy's map is fitted over, grouped by
    task the same way. Pass it wherever `rows` is a subgroup: an ordering's mean
    is corrected by that ordering's floor, and fitting five constants on those
    ten rows bought up to 0.040 of hindsight over fitting them on the task's
    forty. A task absent from `fit_rows` is fitted on the rows in hand.
    """
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(row_task(row), []).append(row)
    fit_by_task: Dict[str, List[Dict[str, Any]]] = {}
    for row in fit_rows or ():
        fit_by_task.setdefault(row_task(row), []).append(row)

    out: Dict[str, Dict[str, Any]] = {}
    for task in sorted(by_task):
        value, best, means, detail, refused = floor_for(by_task[task],
                                                        fit_by_task.get(task))
        out[task] = {"floor": value, "strategy": best, "detail": detail,
                     "refused": refused, "n": len(by_task[task]),
                     "answer_shape": registry.get(task).answer_shape,
                     "by_strategy": means}
    return out


def floor_strategy(entry: Dict[str, Any]) -> str:
    """How a floor was reached, as one string to print beside the number.

    The number alone does not say whether it is plausible as *the* uninformed
    baseline: `0.0000 empty` means the search found nothing that fits the answer
    format, which is a different sentence from "guessing does not pay here".
    A fitted strategy's name does not say what it fitted either, so the map
    comes with it, and a strategy that was refused says why -- a constant that
    won because the map above it was priced out is a floor with a known
    understatement, and the name alone hides that.
    """
    detail = entry.get("detail")
    out = entry["strategy"]
    if detail:
        out += "(" + ", ".join(f"{k}={v}" for k, v in sorted(detail.items())) + ")"
    for name, why in sorted((entry.get("refused") or {}).items()):
        out += f"; {name} refused ({why})"
    return out


def corrected(score_value: float, floor: float) -> float:
    """A score rescaled so chance is zero and perfect is one.

    The only defensible way to put twelve metrics on one axis, and still not a
    reason to average them.

    A floor of 1.0 collapses the scale, and the two ways of reaching one are not
    the same event. A fixed strategy that scores 1.000 is a finding about the
    scorer -- one constant answers the task -- and the clamp reports it as a task
    with no room above chance, which it is. A fitted map that scores 1.000 has
    reproduced the gold, a fitting artifact rather than a finding, so it raises
    (`FittedFloorIsGold`) and never reaches here.
    """
    if floor >= 1.0:
        return 0.0
    return round((score_value - floor) / (1.0 - floor), 4)
