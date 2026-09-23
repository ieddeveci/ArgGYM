"""What the strict ablation costs, and why one directive is never enough.

`counter_argument_strict` asks whether the model finds the cheapest answer when a strict
rule is permitted. With `n` chains reaching the target and `k` of them ending strictly,
the strict answer costs `1 + k`: the strict counter-argument itself, plus one undercut for
each chain that would otherwise derive the target strictly and make the framework
inconsistent. The plain answer also has to beat the `n - k` defeasible chains, which the
strict rule rebuts for free.

So `k` decides both properties this task needs. At `k = 0` nothing has to be broken first
and the whole strict answer is the fixed line `[strict cs: <premise> -> -<target>]`, which
anyone can write from the question line. At `k = n` the two arms cost the same and ship
the same reference, which is #37. The generator holds `1 <= k <= n - 1` at every level
(`n - 2` where the decoy is in play), and `n >= 2` is what makes that interval non-empty.

This file asserts the cost law and the floor it implies. Whether the fixed line actually
scores is asked next door, in
tests/test_neither_arm_answers_from_the_question_line_and_one_grep.py, which runs the
template through the scorer rather than reading its cost.

Levels 1 and 2 used to build one chain, so the interval was empty there and they took
`k = 0`: 16 of their 16 strict cells answered to the fixed line, and 8 of them answered
the plain arm to a rebuttal plus a grep as well. A draw at levels 3 to 5 zeroed `k` for
the strict arm alone on 14 more, which this file used to assert as a feature -- it was
added for #32, when the entry level cost three directives where level 6 cost one. #117
raised the strict minimum to `1 + k` and took level 6's floor to two, so the inversion the
draw was closed on had already stopped existing; what the draw still did was make the two
arms build different theories, which is the one thing an ablation may not do (#167).
"""
from __future__ import annotations

import pytest

from arggym.core.spec import ALL_ORDERINGS, LEVELS, SEEDS
from arggym.tasks import counter_argument as ca

GRID = LEVELS
#: The two cheapest levels to build, and where the laws below are asserted by default.
#: They are properties of every cell, but a dozen weakest-link-democratic cells at the top
#: levels take 170 to 540 seconds each to build (#111), so the whole grid rides the slow
#: marker at the foot of the file rather than the default run.
CHEAP_LEVELS = (3, 6)


def _items(level, allow_strict):
    out = []
    for o in ALL_ORDERINGS:
        for s in SEEDS:
            it = ca.make_item(level, s, o, allow_strict=allow_strict)
            assert it is not None, f"L{level} {o} seed {s} allow_strict={allow_strict}"
            out.append(((o, s), it))
    return out


@pytest.mark.parametrize("level", CHEAP_LEVELS)
def test_the_strict_answer_costs_one_per_blocked_chain(level):
    """`1 + k`, on every cell. The law the floor below follows from."""
    for (o, s), it in _items(level, allow_strict=True):
        k = it.metadata["n_strict_final"]
        assert it.min_directives == 1 + k, (
            f"L{level} {o} seed {s}: {it.min_directives} directives against {k} chains "
            f"reaching the target strictly, so the answer is no longer the strict "
            f"counter-argument plus one undercut each:\n{it.reference}")


@pytest.mark.parametrize("level", CHEAP_LEVELS)
def test_at_least_one_chain_reaches_the_target_strictly(level):
    """`k >= 1`, which is what stops the fixed line being the whole answer.

    A level that draws `k` to zero ships items answerable from the question line. That is
    what levels 1 and 2 did on every cell, and the draw did on 14 more at levels 3 to 5.
    """
    for (o, s), it in _items(level, allow_strict=True):
        assert it.metadata["n_strict_final"] >= 1, (
            f"L{level} {o} seed {s} has no strict-final chain, so one directive answers "
            f"it and the arm is template-answerable here")
        assert it.min_directives > 1, f"L{level} {o} seed {s}: min_directives 1"


@pytest.mark.parametrize("level", CHEAP_LEVELS)
def test_some_chain_stays_defeasible(level):
    """`k <= n - 1`. Without it the two arms cost the same and ship one answer (#37)."""
    for (o, s), it in _items(level, allow_strict=True):
        assert it.metadata["n_strict_final"] <= it.metadata["n_chains"] - 1, (
            f"L{level} {o} seed {s}: every chain ends strictly, so none is left for the "
            f"plain arm to pay an undercut on")
        plain = ca.make_item(level, s, o, allow_strict=False)
        assert it.min_directives < plain.min_directives, (
            f"L{level} {o} seed {s}: strict {it.min_directives} against plain "
            f"{plain.min_directives}, so permitting a strict rule bought nothing")
        assert it.reference != plain.reference, (
            f"L{level} {o} seed {s}: the two arms ship one answer between them")


def test_the_entry_level_is_not_dearer_than_the_one_above_it():
    """#32's inversion, asserted where it was measured rather than where it was fixed."""
    lo = min(it.min_directives for _, it in _items(3, allow_strict=True))
    hi = min(it.min_directives for _, it in _items(6, allow_strict=True))
    assert lo <= hi, f"L3 floor {lo} above L6 floor {hi}"


@pytest.mark.parametrize("level", CHEAP_LEVELS)
def test_permitting_a_strict_rule_never_costs_more(level):
    """A permitted move can only widen the search, so the minimum cannot rise."""
    for (o, s), strict in _items(level, allow_strict=True):
        plain = ca.make_item(level, s, o, allow_strict=False)
        assert plain is not None
        assert strict.min_directives <= plain.min_directives, (
            f"L{level} {o} seed {s}: strict {strict.min_directives} > "
            f"plain {plain.min_directives}")


@pytest.mark.slow
@pytest.mark.parametrize("level", GRID)
def test_the_three_laws_hold_at_every_exported_cell(level):
    """The same three laws, over the whole grid rather than the two cheap levels."""
    for (o, s), it in _items(level, allow_strict=True):
        k = it.metadata["n_strict_final"]
        plain = ca.make_item(level, s, o, allow_strict=False)
        where = f"L{level} {o} seed {s}"
        assert it.min_directives == 1 + k, f"{where}: {it.min_directives} against k={k}"
        assert k >= 1, f"{where}: no strict-final chain, so one directive answers it"
        assert it.min_directives < plain.min_directives, (
            f"{where}: strict {it.min_directives} against plain {plain.min_directives}")
        assert it.reference != plain.reference, f"{where}: the arms ship one answer"


@pytest.mark.slow
@pytest.mark.parametrize("level", GRID)
@pytest.mark.parametrize("allow_strict", [False, True])
def test_every_cell_still_generates(level, allow_strict):
    assert len(_items(level, allow_strict)) == len(ALL_ORDERINGS) * len(SEEDS)
