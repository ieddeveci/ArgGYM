"""status_query has to ask as many questions as its level says (issue #43).

`n_query` scales with the level, from 8 up to 40. The queried set may not let one status
take more than `MAX_STATUS_SHARE` of it, so with `U` undecided literals in the theory the
cap allows at most `10 * U` questions: balancing at J = O = k needs k / (2k + U) <= 0.45,
so k <= 4.5 * U.

Undecided literals were the scarce status. The junction branch takes a whole group and
plants one OVERRULED and one JUSTIFIED regardless of what that group was meant to be, and
it spends its budget on a prefix of the groups, so it consumed the undecided third from the
front. Twenty-one of the forty grid cells then asked fewer questions than their level
called for, and when supply fell to two the item was rejected outright, since 10 / 22 is
0.4545.

Nothing rejected a short build either, so the query count held by luck of the rng stream:
`make_item` retries only when `build` rejects the candidate, and `build` returned whatever
length it happened to reach.
"""
from __future__ import annotations

import pytest

from arggym.core.curriculum import JUNCTION_START
from arggym.core.spec import ALL_ORDERINGS, SEEDS
from arggym.tasks import status_query as sq

#: The curriculum range, which the grid now equals. Every assertion below is a
#: property of the item the cell built, and `TasksetSpec` accepts any level the
#: curriculum spans, so the invariant is written against the range rather than the
#: export and would still hold if the grid narrowed. `status_query` is cheap on every
#: ordering: all 120 cells build in under two seconds, so the cache pays for the
#: widening and the file is faster than it was on the five levels it swept before.
GRID = tuple(range(1, 16))
CELLS = [(lv, o, s) for lv in GRID for o in ALL_ORDERINGS for s in SEEDS]

SHAPES = ("justified", "overruled", "undecided")

#: Three tests walk every cell and two more walk a level's eight, so one build serves
#: five. Under xdist they scatter across workers and each rebuilds what it is handed;
#: the cache is what keeps a serial run of this file cheap.
_CACHE: dict = {}


def item_at(level: int, ordering: str, seed: int):
    if (level, ordering, seed) not in _CACHE:
        it = sq.make_item(level, seed, ordering)
        assert it is not None, f"L{level}/{ordering}/s{seed} generated nothing"
        _CACHE[level, ordering, seed] = it
    return _CACHE[level, ordering, seed]


@pytest.mark.parametrize("level,ordering,seed", CELLS)
def test_every_cell_asks_as_many_questions_as_its_level_says(level, ordering, seed):
    m = item_at(level, ordering, seed).metadata
    assert m["n_queried"] == m["target_n_query"], (
        f"L{level}/{ordering}/s{seed} asked {m['n_queried']} of {m['target_n_query']}: "
        f"{m['status_counts']}")


@pytest.mark.parametrize("level,ordering,seed", CELLS)
def test_the_theory_supplies_enough_undecided_literals(level, ordering, seed):
    """Counted in the theory, not in the query set.

    `status_counts` counts the queried literals, so with the two tests either side of this
    one it is arithmetic -- J and O each below 0.45N forces U above 0.1N -- and it cannot
    fail while they pass. The supply is a property of what the generator built, and it is
    what separates a generator that stopped producing undecided literals from a selection
    loop that stopped using them.
    """
    item = item_at(level, ordering, seed)
    sm = sq.full_status_map(item.base_ops, item.ordering)
    assert sm, "the theory has no status map"
    supply = sum(1 for lit, st in sm.items()
                 if st == "UNDECIDED" and not lit.startswith("-"))
    target = item.metadata["target_n_query"]
    assert 10 * supply >= target, (
        f"L{level}/{ordering}/s{seed}: {supply} undecided literals in the theory cap the "
        f"query set at {10 * supply}, below the {target} this level asks for")


@pytest.mark.parametrize("level,ordering,seed", CELLS)
def test_no_status_exceeds_the_share_cap(level, ordering, seed):
    counts = item_at(level, ordering, seed).metadata["status_counts"]
    assert len(counts) == 3, f"a status is missing: {counts}"
    assert max(counts.values()) / sum(counts.values()) <= sq.MAX_STATUS_SHARE


@pytest.mark.parametrize("level", GRID)
def test_every_group_shape_still_reaches_every_level(level):
    """The junction takes the whole group, so an uncapped budget erases the other shapes.

    With the budget above the number of groups it may land on, every justified and every
    overruled group became a junction, and three constructions left the exported grid:
    justified-with-tower, overruled-by-undermined-premise and overruled-by-rebuttal.
    """
    seen: set = set()
    for o in ALL_ORDERINGS:
        for s in SEEDS:
            it = item_at(level, o, s)
            seen |= {k for k, v in it.metadata["group_shapes"].items() if v}
    missing = [k for k in SHAPES if k not in seen]
    assert not missing, f"L{level} builds no group of shape {missing}; saw {sorted(seen)}"
    if level >= JUNCTION_START:
        assert "junction" in seen, f"L{level} builds no junction group"


@pytest.mark.parametrize("level", [lv for lv in GRID if lv >= JUNCTION_START])
def test_the_junction_leaves_some_groups_to_their_own_shape(level):
    for o in ALL_ORDERINGS:
        for s in SEEDS:
            shapes = item_at(level, o, s).metadata["group_shapes"]
            own = shapes.get("justified", 0) + shapes.get("overruled", 0)
            assert own > 0 and shapes.get("junction", 0) > 0, (
                f"L{level}/{o}/s{s}: {shapes}")
