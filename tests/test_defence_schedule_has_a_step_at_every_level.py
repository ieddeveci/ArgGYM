"""The defence curriculum has to step, and no step may move everything at once.

Four of the structural knobs were `max(2, min(5, 2 + level // 4))`, which saturates at
level 12, so every level from 12 up built one shape and the top of the curriculum had no
step in it. The level 6 to 9 step moved five knobs together, so a difficulty jump there
could not be attributed to any of them (#31).

`_defence_shape` is flat inside a band of levels and changes between bands: levels 1-5,
6-8, 9-11, 12-14 and 15 each build one shape. Two exported levels sharing a shape is
therefore normal now and says nothing -- the grid exports all fifteen, so the interesting
levels are the ones where the shape is supposed to change and does not. Saturation is
still the failure to look for, and it shows up as the top of the grid building what the
level below it builds, which is #31 read on today's grid: the same four knobs saturating
would now collapse levels 14 and 15.
"""
from __future__ import annotations

import pytest

from arggym.core.curriculum import DEFENCE
from arggym.core.spec import ALL_ORDERINGS, LEVELS, SEEDS
from arggym.tasks import attack_defense as ad

GRID = LEVELS
# `_defence_shape` is a pure function of the level and builds nothing, so the properties
# that hold between neighbours can be checked at every level of the curriculum, whether or
# not the grid exports it. Today the two coincide; they are written apart because the
# shape properties below are about the schedule and the band properties are about what
# gets exported, and a grid that narrowed again would have to re-check only the second.
SCHEDULE = tuple(range(1, 16))

KNOBS = ("n_attackers", "support_depth", "attacker_depth",
         "n_strict_attackers", "n_decoys", "n_decoy_strict")


def _shape(level):
    return ad._defence_shape(level)


def _bands():
    """The maximal runs of levels that build one shape, read off the shape function."""
    bands = []
    for lv in SCHEDULE:
        if not bands or _shape(lv) != _shape(bands[-1][0]):
            bands.append([lv])
        else:
            bands[-1].append(lv)
    return bands


def test_every_band_is_exported_and_none_of_them_repeats():
    """A shape belongs to one stretch of the curriculum, and the grid reaches all of them.

    Two claims, and the first is the one that used to be written as "no two exported levels
    share a shape". That phrasing held only because the old grid picked one level out of
    each of the five bands; it says nothing about a grid that exports every level. What it
    was really asserting is that no band is skipped -- a declared step the grid never
    samples is a difficulty knob nothing measures, which is #30 one file over.

    The second claim is what distinctness bought on top: a shape may not come back after
    the schedule has left it. A recurring shape means a level further up the curriculum
    rebuilds a rung from further down, which no amount of sampling would make visible.
    """
    bands = _bands()
    unsampled = [b for b in bands if not set(b) & set(GRID)]
    assert not unsampled, (
        f"the grid exports no level from {unsampled}, so the step into that band is a "
        f"knob nothing measures")
    shapes = [_shape(b[0]) for b in bands]
    assert len(set(shapes)) == len(shapes), (
        f"a shape occurs in two separate bands, so the curriculum returns to a rung it "
        f"had left: {[(b, _shape(b[0])) for b in bands]}")


def test_the_top_of_the_grid_is_not_the_level_below_it_again():
    """#31 itself, read on today's grid.

    Four knobs were `max(2, min(5, 2 + level // 4))` and saturated at level 12, so the
    schedule's last step changed nothing and the hardest cells asked for what the level
    below asked for. The old grid caught it as "12 and 15 build the same item"; the same
    saturation today would read as 14 and 15, because the grid exports every level and the
    top band is level 15 alone.
    """
    assert _shape(GRID[-1]) != _shape(GRID[-2]), (
        f"L{GRID[-2]} and L{GRID[-1]} build the same shape {_shape(GRID[-1])}, so the top "
        f"of the curriculum has no step in it")


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
    assert cost(GRID[-1]) > cost(GRID[-2]), (
        f"levels {GRID[-2]} and {GRID[-1]} asked for the same work")
