"""The shared-conflict schedule has to land on the levels the grid evaluates (issue #30).

`preference_construction` can point two claims at one conflict, so a single preference
directive settles both goals. The schedule that turned it on read `level % 3 == 2`. The
export grid steps by 3, so `level % 3` is the constant 0 on that grid: the branch ran at
levels 5, 8, 11 and 14, which the grid never evaluates, and so never reached an exported
item. These tests pin the property rather than the arithmetic. Whatever the schedule is,
it has to vary across the evaluated levels, and both shapes have to generate.

The grid comes from `arggym.core.export`. Restating it here would reintroduce exactly the
disagreement between curriculum and grid that #30 was.
"""
from __future__ import annotations

import pytest

from arggym.core.export import LEVELS, SEEDS
from arggym.tasks import preference_construction as pc

# Named rather than taken from ALL_ORDERINGS by index: the economy property below holds
# under the last-link orderings only, and reordering that tuple would otherwise change
# both what these tests assert and how long they take.
LAST_LINK = "last_link_elitist"


def shared_at(level: int) -> bool:
    """Read the schedule off a generated item rather than re-implementing it."""
    item = pc.make_item(level, 0, LAST_LINK)
    assert item is not None, f"L{level} generated nothing"
    return bool(item.metadata["shared_conflict"])


@pytest.mark.parametrize("level", LEVELS)
def test_every_evaluated_level_generates(level):
    assert pc.make_item(level, 0, LAST_LINK) is not None


def test_schedule_is_not_constant_on_the_grid():
    seen = {level: shared_at(level) for level in LEVELS}
    assert True in seen.values(), f"shared conflict never occurs on the grid: {seen}"
    assert False in seen.values(), f"shared conflict occurs at every grid level: {seen}"


def test_goal_count_does_not_fall_across_the_grid():
    """The shared branch adds a claim, so it must not land where the base ramp steps up."""
    goals = {}
    for level in LEVELS:
        item = pc.make_item(level, 0, LAST_LINK)
        assert item is not None, f"L{level} generated nothing"
        goals[level] = len(item.goals)
    counts = [goals[l] for l in LEVELS]
    assert counts == sorted(counts), f"goal count is not monotonic across levels: {goals}"
    assert len(set(counts)) == len(counts), f"two levels have the same goal count: {goals}"


def test_a_shared_item_settles_two_goals_with_one_directive():
    """The point of the branch: one preference can settle more than one goal.

    Stated under the last-link ordering. Under weakest-link the minimum directive count
    is driven by the strength ordering rather than by how the claims share a conflict,
    and exceeds the goal count even when the item does share.
    """
    level = next(l for l in LEVELS if shared_at(l))
    item = pc.make_item(level, 0, LAST_LINK)
    claims = [g["claim"] for g in item.goals]
    assert len(claims) == len(set(claims)), "goals are not distinct"
    assert len(claims) >= 2
    assert item.min_directives < len(claims), \
        (f"L{level}: {len(claims)} goals need {item.min_directives} directives; "
         "a shared conflict should let one directive settle two")


@pytest.mark.slow
def test_shared_levels_share_under_every_ordering_and_seed():
    """Slow: some cells take minutes under weakest_link_democratic."""
    from arggym.core.export import ALL_ORDERINGS
    shared_levels = [l for l in LEVELS if shared_at(l)]
    assert shared_levels, "no level schedules a shared conflict"
    for level in shared_levels:
        for ordering in ALL_ORDERINGS:
            for seed in SEEDS:
                item = pc.make_item(level, seed, ordering)
                assert item is not None, f"L{level}/{ordering}/s{seed} generated nothing"
                assert item.metadata["shared_conflict"] is True, \
                    f"L{level}/{ordering}/s{seed} did not share"
