"""Under last-link, elitist and democratic defeat the same arguments.

py_arg implements the textbook distinction -- elitist is `any(all(...))`,
democratic is `all(any(...))` -- and the two agree on a singleton. Last-link
compares an argument's last defeasible rules, and ArgGYM's theories give an
argument exactly one of those nearly everywhere, so half the ordering axis is a
duplicate of the other half. `status_query` is the exception: at levels 6 and 12
it builds an argument with two last defeasible rules, and the relation splits.

`docs/dataset-card.md` used to claim the collapse held "on eight of the twelve
tasks", which was wrong on the count and on the dimension -- the split is by
family, not by task. This test is what keeps the corrected claim honest: a
generator change that gives arguments two last defeasible rules turns
`last_link_democratic` into a cell worth evaluating, and this fails and says so.

Measured on the theory rather than by diffing items. Every generator mixes the
ordering into its RNG seed, so two orderings at one (task, level, seed) build two
different theories and everything differs for a reason that says nothing about
the axis.
"""
from __future__ import annotations

import pytest

from arggym.aspic.api import ASPICVerifier
from arggym.core import registry

LAST_LINK = ("last_link_elitist", "last_link_democratic")
#: The two levels where `status_query` splits, of the grid's five. The second
#: last defeasible rule comes from a junction its curriculum turns on here and
#: not at 3, 9 or 15.
SPLIT_LEVELS = (6, 12)
LIVE = "status_query"
INERT = tuple(t for t in registry.task_names() if t != LIVE)


def theory_of(task: str, level: int, seed: int, ordering: str):
    """The operations the item was built from.

    `formalization` publishes no theory -- the operations are its gold answer --
    so it is read from `reference_ops` rather than from the registry's table.
    """
    spec = registry.get(task)
    item = spec.make_item(level, seed, ordering)
    assert item is not None, f"{task} L{level} s{seed} {ordering} built no item"
    fields = spec.theory_fields or ("reference_ops",)
    ops = [op for f in fields for op in getattr(item, f)]
    assert ops, f"{task} has no theory to evaluate"
    return ops


def defeat_relation(ops, ordering: str):
    v = ASPICVerifier.from_operations(list(ops), ordering=ordering)
    return {(str(d.from_argument), str(d.to_argument)) for d in v.fw.af.defeats}


@pytest.mark.parametrize("built_with", LAST_LINK)
@pytest.mark.parametrize("level", SPLIT_LEVELS)
@pytest.mark.parametrize("task", INERT)
def test_both_last_link_orderings_defeat_the_same_arguments(task, level, built_with):
    ops = theory_of(task, level, 0, built_with)
    elitist, democratic = (defeat_relation(ops, o) for o in LAST_LINK)
    extra = sorted(elitist ^ democratic)
    assert not extra, (
        f"{task} L{level} built under {built_with} defeats differently under the two "
        f"last-link orderings ({len(extra)} pairs). If that is intended, the axis "
        f"is live on one more task and docs/dataset-card.md has to say so.")


@pytest.mark.parametrize("built_with", LAST_LINK)
@pytest.mark.parametrize("level", SPLIT_LEVELS)
def test_status_query_is_the_one_task_the_axis_reaches(level, built_with):
    ops = theory_of(LIVE, level, 0, built_with)
    elitist, democratic = (defeat_relation(ops, o) for o in LAST_LINK)
    assert elitist != democratic, (
        f"{LIVE} L{level} no longer separates the last-link orderings, so "
        f"last_link_democratic is now a duplicate cell on all twelve tasks and "
        f"the card's claim needs re-measuring.")
