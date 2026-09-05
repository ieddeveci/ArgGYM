"""Where a preference is declared both ways, the prompt has to say what that means.

A tie is legal ASPIC+ -- the ordering is a preorder, so mutual preference means "equally
preferred" and behaves exactly like no preference -- and gold stays correct, because it is
re-derived from the engine either way. What it does is grade a model on a convention no
prompt states, which is what #6 reports.

The generators reach a tie in two places, and neither shows up in the theory alone:

* `perturbation` adds a preference the theory already declares the other way. The DSL has
  no removal, so that is the only way to write "this preference no longer decides the
  conflict". 13 of the 40 exported items do it.
* `preference_construction`'s gold answer does the same to undo a resolved conflict.
  8 of the 40 exported items do it.

So the check reads the theory, the theory plus whatever the prompt adds to it, and the
theory plus the gold answer -- and where a tie is reachable, it requires the prompt to
explain the convention rather than forbidding the tie.
"""
from __future__ import annotations

import inspect
from typing import List

import pytest

from arggym.aspic.engine import Operation
from arggym.core.export import _EXPORTABLE, ALL_ORDERINGS, _export_row, export_task
from arggym.core.scoring import parse_answer

GRID = inspect.signature(export_task).parameters["levels"].default
SEEDS = inspect.signature(export_task).parameters["seeds"].default
TASKS = sorted(set(_EXPORTABLE))
CHEAP_LEVELS = (3, 6)

# The two tasks whose generators reach a tie. `perturbation` ties from level 3, so the
# sweep below catches it cheaply. `preference_construction` ties only at levels 12 and 15,
# where generating the eight cells costs eleven minutes -- too slow to sit in the default
# run, and the sweep alone would have passed with the sentence deleted. So the sentence is
# asserted directly on both, which is the stronger statement anyway: it is unconditional
# in the prompt, not something an item earns by tying.
EXPLAINS_TIES = ("perturbation", "preference_construction")

PREFERENCE = ("prefer_rule", "prefer_premise")
TIE_NOTE = ("Preference is a preorder, so a pair declared stronger in both directions is "
            "equally preferred and settles nothing between them.")


def operation_lists(item) -> List[List[Operation]]:
    """Every list of operations the item carries, not the first one that happens to fill.

    An earlier version walked `("base_ops", "reference_ops", "ops")` and returned the
    first truthy one. `PerturbItem` has both `base_ops` and `pert_ops`, so it stopped at
    the theory and never saw the perturbation, where every one of its ties lives.
    """
    out = []
    for name in dir(item):
        if name.startswith("__"):
            continue
        value = getattr(item, name, None)
        if isinstance(value, list) and value and all(isinstance(o, Operation) for o in value):
            out.append(value)
    assert out, f"{type(item).__name__} exposes no operation list"
    return out


def tied_pairs(*op_lists) -> list:
    prefs = {(o.stronger, o.weaker)
             for ops in op_lists for o in ops if o.kind in PREFERENCE}
    return sorted({tuple(sorted((a, b))) for a, b in prefs if (b, a) in prefs})


def _reference_ops(item) -> List[Operation]:
    ref = item.reference() if callable(item.reference) else item.reference
    return parse_answer(ref).ops if isinstance(ref, str) else []


def _items(task, levels):
    for lv in levels:
        for o in ALL_ORDERINGS:
            for s in SEEDS:
                got = _export_row(task, lv, o, s)
                if got is not None:
                    yield lv, o, s, got[0]


def _all_ops(item):
    merged: List[Operation] = []
    for ops in operation_lists(item):
        merged.extend(ops)
    return merged


def _check(task, levels):
    seen = 0
    unexplained = []
    for lv, o, s, item in _items(task, levels):
        seen += 1
        pairs = tied_pairs(_all_ops(item), _reference_ops(item))
        if pairs and TIE_NOTE not in item.prompt:
            unexplained.append(f"L{lv} {o} seed {s}: {pairs}")
    assert seen, f"{task} produced no item at levels {levels}"
    assert not unexplained, (
        f"{task} declares a preference in both directions and its prompt does not say a "
        f"tie settles nothing:\n  " + "\n  ".join(unexplained))


@pytest.mark.parametrize("task", EXPLAINS_TIES)
def test_the_tasks_that_tie_always_carry_the_sentence(task):
    """Unconditional, so deleting the sentence fails here rather than only in the sweep.

    `preference_construction` ties only at levels 12 and 15, so a sweep over cheap levels
    cannot see its half at all -- it passed green with the sentence removed.
    """
    for lv, o, s, item in _items(task, (3,)):
        assert TIE_NOTE in item.prompt, f"{task} L{lv} {o} seed {s}"


@pytest.mark.parametrize("task", TASKS)
def test_no_cheap_item_ties_without_saying_what_a_tie_means(task):
    _check(task, CHEAP_LEVELS)


@pytest.mark.slow
@pytest.mark.parametrize("task", TASKS)
def test_no_item_on_the_whole_grid_ties_without_saying_what_a_tie_means(task):
    """The whole grid, which is a many-minute run: counter_argument at 12 and 15 dominates."""
    _check(task, GRID)


@pytest.mark.slow
@pytest.mark.parametrize("task,levels", [("perturbation", CHEAP_LEVELS),
                                         ("preference_construction", (12,))])
def test_the_tasks_that_reach_a_tie_still_reach_one(task, levels):
    """Without this the check above passes on a task that has stopped tying, and the
    sentence in its prompt becomes dead weight that nothing would remove."""
    tied = [f"L{lv} {o} s{s}" for lv, o, s, it in _items(task, levels)
            if tied_pairs(_all_ops(it), _reference_ops(it))]
    assert tied, (f"{task} no longer declares a preference in both directions at levels "
                  f"{levels}; the sentence in its prompt is now unused")


def test_the_check_sees_a_tie_outside_base_ops():
    """The bug this file had: the perturbation list was never read."""
    class Fake:
        prompt = ""
        reference = ""
        base_ops = [Operation(kind="prefer_rule", stronger="r1", weaker="r2")]
        pert_ops = [Operation(kind="prefer_rule", stronger="r2", weaker="r1")]

    assert tied_pairs(_all_ops(Fake())) == [("r1", "r2")]
    assert tied_pairs(Fake.base_ops) == [], "the theory alone is clean, which is the trap"


def test_the_check_sees_a_tie_introduced_by_the_gold_answer():
    class Fake:
        prompt = ""
        reference = "<answer>\n[prefer_rule: r2 > r1]\n</answer>"
        base_ops = [Operation(kind="prefer_rule", stronger="r1", weaker="r2")]

    assert tied_pairs(_all_ops(Fake()), _reference_ops(Fake())) == [("r1", "r2")]
