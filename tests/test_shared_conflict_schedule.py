"""The shared-conflict schedule has to land on the levels the grid evaluates (issue #30).

`preference_construction` can point two claims at one conflict, so a single preference
directive settles both goals. The schedule that turned it on read `level % 3 == 2`. The
export grid stepped by 3 then, so `level % 3` was the constant 0 on it: the branch ran at
levels 5, 8, 11 and 14, which the grid never evaluated, and so never reached an exported
item. These tests pin the property rather than the arithmetic. Whatever the schedule is,
it has to vary across the evaluated levels, and both shapes have to generate.

The grid exports all fifteen levels now, which retires the miss itself -- a branch keyed
on the level fires somewhere the grid looks or it fires nowhere at all. What is left is
the damage the branch does when it fires in the wrong place, and that is what the goal
count below is about: the extra claim has to land where the base ramp is flat, or two
neighbouring levels ask for the same width and the rung between them is lost.

The grid comes from `arggym.core.spec`. Restating it here would reintroduce exactly the
disagreement between curriculum and grid that #30 was.
"""
from __future__ import annotations

import pytest

from arggym.core.spec import LEVELS, SEEDS
from arggym.tasks import preference_construction as pc

# Named rather than taken from ALL_ORDERINGS by index: the economy property below holds
# under the last-link orderings only, and reordering that tuple would otherwise change
# both what these tests assert and how long they take.
LAST_LINK = "last_link_elitist"

# The curriculum range, which the grid currently equals. Written apart from LEVELS
# because these are two different claims -- what the schedule does, and what the export
# samples of it -- and only the second moves if the grid is ever narrowed again.
SCHEDULE = tuple(range(1, 16))


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


def test_goal_count_climbs_one_rung_at_a_time():
    """The shared branch adds a claim, so it must not land where the base ramp steps up.

    `n_claims` is `1 + (level * 8) // 15` capped at 8, so it steps at every even level and
    is flat at every odd one, and the shared branch adds one claim on top. Land the branch
    on a level where the base ramp also steps and that level jumps two claims at once,
    which means the width between them is a rung the curriculum defines and no level
    builds. `preference_construction.py:131` says exactly this, and it is why the branch
    sits at 9 rather than at 6 or 12.

    So the property is monotone with unit steps: no level asks for fewer goals than the
    level below it, and none asks for more than one extra. Over levels 1-15 the count runs
    1 2 2 3 3 4 4 5 6 7 7 7 7 8 8, and every step in it is a step of one.

    This used to read "no two exported levels have the same goal count", which held only
    because the grid sampled one level in three -- far enough apart that the ramp had
    always moved between them. On a grid that exports every level neighbours share a count
    by construction, and asking for distinctness would ask the ramp to climb 15 rungs it
    does not have. The unit-step form is what that distinctness was standing in for, and
    it is the sharper of the two: it catches the branch landing on a stepping level, which
    distinctness on a sparse grid could miss entirely.
    """
    goals = {}
    for level in SCHEDULE:
        item = pc.make_item(level, 0, LAST_LINK)
        assert item is not None, f"L{level} generated nothing"
        goals[level] = len(item.goals)
    steps = [(a, b, goals[b] - goals[a]) for a, b in zip(SCHEDULE, SCHEDULE[1:])]
    fell = [f"L{a} to L{b}" for a, b, d in steps if d < 0]
    assert not fell, f"goal count falls at {', '.join(fell)}: {goals}"
    jumped = [f"L{a} to L{b} by {d}" for a, b, d in steps if d > 1]
    assert not jumped, (
        f"goal count jumps more than one claim at {', '.join(jumped)}, so a branch lands "
        f"where the base ramp already steps and the width in between is a rung no level "
        f"builds: {goals}")
    assert goals[SCHEDULE[-1]] > goals[SCHEDULE[0]], (
        f"the goal count never rises across the curriculum: {goals}")


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
    from arggym.core.spec import ALL_ORDERINGS
    shared_levels = [l for l in LEVELS if shared_at(l)]
    assert shared_levels, "no level schedules a shared conflict"
    for level in shared_levels:
        for ordering in ALL_ORDERINGS:
            for seed in SEEDS:
                item = pc.make_item(level, seed, ordering)
                assert item is not None, f"L{level}/{ordering}/s{seed} generated nothing"
                assert item.metadata["shared_conflict"] is True, \
                    f"L{level}/{ordering}/s{seed} did not share"
