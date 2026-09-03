"""status_query has to ask as many questions as its level says (issue #43).

`n_query` scales with the level, from 8 up to 40. The queried set may not let one status
take more than `MAX_STATUS_SHARE` of it, so with `U` undecided literals available the cap
allows at most `10 * U` questions: balancing at J = O = k needs k / (2k + U) <= 0.45, so
k <= 4.5 * U.

Undecided literals were the scarce status. The junction branch takes a whole group and
plants one OVERRULED and one JUSTIFIED regardless of what that group was meant to be, and
it spends its budget on a prefix of the groups, so it consumed the undecided third from the
front. At level 12 six groups ask for UNDECIDED and one survived. Twenty-one of the forty
grid cells then asked fewer questions than their level called for, and when supply fell to
two the item was rejected outright, since 10 / 22 is 0.4545.

These tests pin the outcome. `target_n_query` and `n_queried` are both already recorded and
were never compared, which is how this stayed quiet.
"""
from __future__ import annotations

import pytest

from arggym.core.export import ALL_ORDERINGS
from arggym.tasks import status_query as sq

# Level 3 is left out. There n_group is 5, and the want cycle over three statuses deals
# 2 JUSTIFIED, 2 OVERRULED and 1 UNDECIDED, so supply is one by construction rather than by
# anything the junction did. That is a curriculum question and not this fix.
LEVELS = (6, 9, 12, 15)
SEEDS = (0, 1)
CELLS = [(lv, o, s) for lv in LEVELS for o in ALL_ORDERINGS for s in SEEDS]


@pytest.mark.parametrize("level,ordering,seed", CELLS)
def test_every_cell_asks_as_many_questions_as_its_level_says(level, ordering, seed):
    item = sq.make_item(level, seed, ordering)
    assert item is not None, f"L{level}/{ordering}/s{seed} generated nothing"
    m = item.metadata
    assert m["n_queried"] == m["target_n_query"], (
        f"L{level}/{ordering}/s{seed} asked {m['n_queried']} of {m['target_n_query']}: "
        f"{m['status_counts']}")


@pytest.mark.parametrize("level,ordering,seed", CELLS)
def test_the_scarce_status_can_carry_the_query_count(level, ordering, seed):
    """The supply relation, stated directly, so a regression names its own cause.

    A cell can only be short because the cap bound it, and the cap can only bind because
    undecided literals ran out. Asserting the supply separates the two: a failure here is a
    generator that stopped producing them, not a selection loop that stopped using them.
    """
    item = sq.make_item(level, seed, ordering)
    assert item is not None
    n_undecided = item.metadata["status_counts"].get("UNDECIDED", 0)
    target = item.metadata["target_n_query"]
    assert 10 * n_undecided >= target, (
        f"L{level}/{ordering}/s{seed}: {n_undecided} undecided literals cap the query set "
        f"at {10 * n_undecided}, below the {target} this level asks for")


@pytest.mark.parametrize("level,ordering,seed", CELLS)
def test_no_status_exceeds_the_share_cap(level, ordering, seed):
    item = sq.make_item(level, seed, ordering)
    assert item is not None
    counts = item.metadata["status_counts"]
    assert len(counts) == 3, f"a status is missing: {counts}"
    assert max(counts.values()) / sum(counts.values()) <= sq.MAX_STATUS_SHARE
