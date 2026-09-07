"""No two exported levels may build the same defence item shape, and no step may move
everything at once.

Four of the structural knobs were `max(2, min(5, 2 + level // 4))`, which saturates at
level 12, so levels 12 and 15 built identical shapes and the top of the curriculum had no
step in it. The level 6 to 9 step moved five knobs together, so a difficulty jump there
could not be attributed to any of them (#31).
"""
from __future__ import annotations

import pytest

from arggym.core.curriculum import DEFENCE
from arggym.core.spec import ALL_ORDERINGS, LEVELS, SEEDS
from arggym.tasks import attack_defense as ad

GRID = LEVELS
# `_defence_shape` is arithmetic on the level, so the properties that hold between
# neighbouring levels can be checked at every level for nothing. Distinctness cannot:
# the shape is flat inside each band of levels the grid steps over, so it is a
# property of the exported levels and stays on GRID.
SCHEDULE = tuple(range(1, 16))

KNOBS = ("n_attackers", "support_depth", "attacker_depth",
         "n_strict_attackers", "n_decoys", "n_decoy_strict")


def _shape(level):
    return ad._defence_shape(level)


def test_no_two_exported_levels_share_a_shape():
    shapes = {lv: _shape(lv) for lv in GRID}
    dupes = [(a, b) for i, a in enumerate(GRID) for b in GRID[i + 1:]
             if shapes[a] == shapes[b]]
    assert not dupes, f"levels build the same item: {dupes} from {shapes}"


def test_no_step_moves_every_knob():
    """A step that changes everything cannot be attributed to any of it."""
    for a, b in zip(SCHEDULE, SCHEDULE[1:]):
        moved = [i for i, (x, y) in enumerate(zip(_shape(a), _shape(b))) if x != y]
        assert len(moved) < len(KNOBS), f"L{a} to L{b} moves all {len(KNOBS)} knobs"


def test_the_shape_never_goes_backwards():
    for a, b in zip(SCHEDULE, SCHEDULE[1:]):
        assert all(y >= x for x, y in zip(_shape(a), _shape(b))), (a, b)


@pytest.mark.parametrize("level", GRID)
def test_every_cell_generates_and_records_its_shape(level):
    """Both halves matter, and only the first is a real assertion.

    The metadata is unpacked from `_defence_shape(level)`, so comparing it back against
    that tuple cannot fail on a shape mismatch. What it does establish is that every cell
    builds at the shape its level asks for -- including the second strict attacker at
    level 15, which the builder had never been asked for before.
    """
    want = dict(zip(KNOBS, _shape(level)))
    seen = 0
    for o in ALL_ORDERINGS:
        for s in SEEDS:
            it = ad.make_item(level, s, o, mode=DEFENCE)
            assert it is not None, f"L{level} {o} seed {s} produced no item"
            seen += 1
            assert {k: it.metadata[k] for k in KNOBS} == want
    assert seen == len(ALL_ORDERINGS) * len(SEEDS)


@pytest.mark.parametrize("level", GRID)
def test_the_theory_carries_the_strict_attackers_the_level_asks_for(level):
    """Counted off the built theory, which the shape comparison above cannot do."""
    want = dict(zip(KNOBS, _shape(level)))["n_strict_attackers"]
    for o in ALL_ORDERINGS:
        for s in SEEDS:
            it = ad.make_item(level, s, o, mode=DEFENCE)
            assert it is not None
            strict_rules = sum(1 for op in it.base_ops if op.kind == "strict")
            floor = want + dict(zip(KNOBS, _shape(level)))["n_decoy_strict"]
            assert strict_rules >= floor, (
                f"L{level} {o} seed {s}: {strict_rules} strict rules in the theory, below "
                f"the {floor} the schedule asks for")


def test_the_top_of_the_grid_still_costs_more_than_the_middle():
    def cost(level):
        return max(ad.make_item(level, s, o, mode=DEFENCE).min_directives
                   for o in ALL_ORDERINGS for s in SEEDS)
    assert cost(GRID[-1]) > cost(GRID[-2]), "levels 12 and 15 asked for the same work"
