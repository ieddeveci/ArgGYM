"""The shared-conflict schedule has to land on the levels the grid evaluates (issue #30).

`preference_construction` can point two claims at one conflict, so a single preference
directive settles both goals. The schedule that turns it on used `level % 3 == 2`, and
the export grid steps by 3, so `level % 3` is the constant 0 on that grid and the branch
never ran. These tests pin the property rather than the arithmetic: whatever the schedule
is, it must vary across the evaluated levels, and both shapes must actually generate.
"""
from __future__ import annotations

import pytest

from arggym.core.export import ALL_ORDERINGS
from arggym.tasks import preference_construction as pc

GRID = (3, 6, 9, 12, 15)


def shared_at(level: int) -> bool:
    """Read the schedule off a generated item rather than re-implementing it."""
    item = pc.make_item(level, 0, ALL_ORDERINGS[0])
    assert item is not None, f"L{level} generated nothing"
    return bool(item.metadata["shared_conflict"])


@pytest.mark.parametrize("level", GRID)
def test_every_evaluated_level_generates(level):
    assert pc.make_item(level, 0, ALL_ORDERINGS[0]) is not None


def test_schedule_is_not_constant_on_the_grid():
    seen = {level: shared_at(level) for level in GRID}
    assert True in seen.values(), f"shared conflict never occurs on the grid: {seen}"
    assert False in seen.values(), f"shared conflict occurs at every grid level: {seen}"


@pytest.mark.slow
def test_shared_levels_share_a_conflict_across_orderings_and_seeds():
    """Slow: preference_construction takes minutes under weakest_link_democratic at L12."""
    shared_levels = [lv for lv in GRID if shared_at(lv)]
    assert shared_levels, "no level schedules a shared conflict"
    for level in shared_levels:
        for ordering in ALL_ORDERINGS:
            for seed in (0, 1):
                item = pc.make_item(level, seed, ordering)
                assert item is not None, f"L{level}/{ordering}/s{seed} generated nothing"
                assert item.metadata["shared_conflict"] is True, \
                    f"L{level}/{ordering}/s{seed} did not share"


def test_a_shared_item_has_two_goals_on_one_conflict():
    """The point of the branch: one preference can settle more than one goal."""
    level = next(lv for lv in GRID if shared_at(lv))
    item = pc.make_item(level, 0, ALL_ORDERINGS[0])
    claims = [g["claim"] for g in item.goals]
    assert len(claims) == len(set(claims)), "goals are not distinct"
    assert len(claims) >= 2
    assert item.min_directives < len(claims), \
        (f"L{level}: {len(claims)} goals need {item.min_directives} directives; "
         "a shared conflict should let one directive settle two")
