"""The strict ablation needs a cheap answer to exist, at its own entry level too.

`counter_argument_strict` asks whether the model finds the cheapest answer when a strict
rule is permitted. The one-directive strict counter-argument only wins when no chain
reaches the target through a strict rule -- otherwise it contradicts that chain instead
of defeating it, and the framework is inconsistent.

Below level 6 there are two or three chains and `n_strict` floored at one, so a
strict-final chain always reached the target and the shortcut never existed. The cheapest
level-3 item cost three directives where level 6 cost one, which is the inversion #32
reports; levels 4 and 5 had it too, off the exported grid. The draw at levels 3 to 5
zeroes `n_strict` for the strict arm alone on about half those items, which is where the
one-directive answer comes from now and the only place it comes from.

Level 6 used to have it too, by a second route: `mid_target` put the target before the
chain's strict rule, so no chain reached the target strictly whatever `n_strict` said.
That route was the defect in #93 and it is gone, so the levels are split below -- the
draw levels must offer the one-directive answer, level 6 must not.

The one-directive answer is not the only strict answer. Where a chain does reach the
target strictly, the generator breaks it first and the strict answer costs 1 + k for the
k chains that block (#37), which is why the cheapest-answer question these tests ask
stays "is there a one-directive answer here" while the ablation's own question, "does the
strict arm answer more cheaply than the plain one", is asked next door in
tests/test_the_strict_ablation_asks_its_own_question.py.
"""
from __future__ import annotations

import pytest

from arggym.core.spec import ALL_ORDERINGS, LEVELS, SEEDS
from arggym.tasks import counter_argument as ca

GRID = LEVELS
CHEAP_LEVELS = (3, 6)
#: Where the entry-level draw may zero `n_strict` for the strict arm alone (#32).
DREW_LEVELS = (3, 4, 5)
#: The entry level above the draw, and the level `mid_target` enters at.
ABOVE_THE_DRAW = (6,)


def _minimums(level, allow_strict):
    out = []
    for o in ALL_ORDERINGS:
        for s in SEEDS:
            it = ca.make_item(level, s, o, allow_strict=allow_strict)
            assert it is not None, f"L{level} {o} seed {s} allow_strict={allow_strict}"
            out.append(it.min_directives)
    return out


@pytest.mark.parametrize("level", DREW_LEVELS)
def test_some_item_at_this_level_admits_the_one_directive_answer(level):
    mins = _minimums(level, allow_strict=True)
    assert 1 in mins, (f"no level-{level} item has a cheap strict answer, so the ablation "
                       f"has nothing to measure there: {sorted(mins)}")


@pytest.mark.parametrize("level", DREW_LEVELS)
def test_not_every_item_at_this_level_admits_it(level):
    """The question is whether a shortcut exists here, so it must not always exist."""
    mins = _minimums(level, allow_strict=True)
    assert any(m > 1 for m in mins), (
        f"every level-{level} item takes the same one-directive answer, so the variant is "
        f"answerable without reading the theory: {sorted(mins)}")


@pytest.mark.parametrize("level", ABOVE_THE_DRAW)
def test_no_item_above_the_draw_admits_the_one_directive_answer(level):
    """The other half of the split: the draw is the only source of the cheap answer.

    Level 6 is the entry level above the draw and the level `mid_target` enters at, so it
    is where the second route showed up. The rest of the grid is checked in
    tests/test_neither_arm_answers_from_the_question_line_and_one_grep.py, which scores the
    fixed line rather than reading its cost.
    """
    mins = _minimums(level, allow_strict=True)
    assert min(mins) > 1, (
        f"a level-{level} item answers with the fixed line "
        f"[strict cs: <premise> -> -<target>] and nothing else: {sorted(mins)}")


def test_the_entry_level_is_not_dearer_than_the_one_above_it():
    lo, hi = _minimums(3, allow_strict=True), _minimums(6, allow_strict=True)
    assert min(lo) <= min(hi), f"L3 floor {min(lo)} above L6 floor {min(hi)}"


@pytest.mark.parametrize("level", CHEAP_LEVELS)
def test_the_plain_variant_is_untouched_by_the_draw(level):
    """The draw answers a question only the ablation asks, so it may not move the other."""
    for o in ALL_ORDERINGS:
        for s in SEEDS:
            it = ca.make_item(level, s, o, allow_strict=False)
            assert it is not None
            assert it.metadata["n_strict_final"] >= 1, (
                f"L{level} {o} seed {s} has no strict-final chain, so the draw leaked into "
                f"the plain variant")


@pytest.mark.parametrize("level", CHEAP_LEVELS)
def test_permitting_a_strict_rule_never_costs_more(level):
    """A permitted move can only widen the search, so the minimum cannot rise."""
    for o in ALL_ORDERINGS:
        for s in SEEDS:
            plain = ca.make_item(level, s, o, allow_strict=False)
            strict = ca.make_item(level, s, o, allow_strict=True)
            assert plain is not None and strict is not None
            assert strict.min_directives <= plain.min_directives, (
                f"L{level} {o} seed {s}: strict {strict.min_directives} > "
                f"plain {plain.min_directives}")


@pytest.mark.slow
@pytest.mark.parametrize("level", GRID)
@pytest.mark.parametrize("allow_strict", [False, True])
def test_every_cell_still_generates(level, allow_strict):
    assert len(_minimums(level, allow_strict)) == len(ALL_ORDERINGS) * len(SEEDS)
