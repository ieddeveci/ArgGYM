"""The strict ablation needs a cheap answer to exist, at its own entry level too.

`counter_argument_strict` asks whether the model finds the cheapest answer when a strict
rule is permitted. The one-directive strict counter-argument only wins when no chain
reaching the target ends in a strict rule -- otherwise it contradicts that chain instead
of defeating it, and the framework is inconsistent.

Below level 6 there are two or three chains and `n_strict` floored at one, so a
strict-final chain always reached the target and the shortcut never existed. The cheapest
level-3 item cost three directives where level 6 cost one, which is the inversion #32
reports; levels 4 and 5 had it too, off the exported grid.

From level 6 the shortcut arrives by a different route, `mid_target`, which is gated on
`seed % 2 == 1`. So the level-6 assertions below hold only while `SEEDS` contains an odd
number, and that is asserted rather than assumed.
"""
from __future__ import annotations

import inspect

import pytest

from arggym.core.export import ALL_ORDERINGS, export_task
from arggym.tasks import counter_argument as ca

GRID = inspect.signature(export_task).parameters["levels"].default
SEEDS = inspect.signature(export_task).parameters["seeds"].default
CHEAP_LEVELS = (3, 6)
DREW_LEVELS = (3, 4, 5)


def _minimums(level, allow_strict):
    out = []
    for o in ALL_ORDERINGS:
        for s in SEEDS:
            it = ca.make_item(level, s, o, allow_strict=allow_strict)
            assert it is not None, f"L{level} {o} seed {s} allow_strict={allow_strict}"
            out.append(it.min_directives)
    return out


def test_the_level_six_route_is_reachable_from_the_exported_seeds():
    """`mid_target` needs an odd seed, so the level-6 assertions below rest on `SEEDS`."""
    assert any(s % 2 == 1 for s in SEEDS), (
        f"no odd seed in {SEEDS}, so no level-6 item takes the mid-chain target and the "
        f"tests below would fail for a reason that has nothing to do with #32")


@pytest.mark.parametrize("level", sorted(set(CHEAP_LEVELS) | set(DREW_LEVELS)))
def test_some_item_at_this_level_admits_the_one_directive_answer(level):
    mins = _minimums(level, allow_strict=True)
    assert 1 in mins, (f"no level-{level} item has a cheap strict answer, so the ablation "
                       f"has nothing to measure there: {sorted(mins)}")


@pytest.mark.parametrize("level", sorted(set(CHEAP_LEVELS) | set(DREW_LEVELS)))
def test_not_every_item_at_this_level_admits_it(level):
    """The question is whether a shortcut exists here, so it must not always exist."""
    mins = _minimums(level, allow_strict=True)
    assert any(m > 1 for m in mins), (
        f"every level-{level} item takes the same one-directive answer, so the variant is "
        f"answerable without reading the theory: {sorted(mins)}")


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
