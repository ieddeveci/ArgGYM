"""`claim_chain` asks for the line in order, so the score has to read the order.

The prompt requires the directives "in order from the premise to the claim". The scorer
compared sets, so reversing the complete gold answer scored 1.000 (#25). It already
computed the order and reported it beside the score without ever consulting it.

Order here is a constraint, not a sequence. From level 6 the line is a tree, not a chain
-- six premises feeding sixteen rules at level 9 -- and its branches interleave in
billions of orders that all read premise to claim. Scoring against the one order the
generator happened to build in would grade the branch order instead of the reasoning.
"""
from __future__ import annotations

import inspect
import random

import pytest

from arggym.core.export import ALL_ORDERINGS, export_task
from arggym.tasks import claim_chain as cc
from arggym.tasks.claim_chain import render_op

GRID = inspect.signature(export_task).parameters["levels"].default
SEEDS = inspect.signature(export_task).parameters["seeds"].default
CELLS = [(lv, o, s) for lv in GRID for o in ALL_ORDERINGS for s in SEEDS]


def _item(level, seed, ordering):
    it = cc.make_item(level, seed, ordering)
    assert it is not None, f"L{level} {ordering} seed {seed} produced no item"
    return it


def _gold(item):
    return item.reference.split("[answer]\n")[1].split("\n[/answer]")[0].splitlines()


def _answer(lines):
    return "[answer]\n" + "\n".join(lines) + "\n[/answer]"


def _split(item, gold):
    by = {render_op(o): o for o in item.base_ops}
    prem = [l for l in gold if by[l].kind in ("premise", "axiom")]
    rules = [l for l in gold if by[l].kind not in ("premise", "axiom")]
    return prem, rules


@pytest.mark.parametrize("level,ordering,seed", CELLS)
def test_the_reference_still_scores_itself(level, ordering, seed):
    it = _item(level, seed, ordering)
    assert cc.score(it.reference, it)["score"] == pytest.approx(1.0)


@pytest.mark.parametrize("level,ordering,seed", CELLS)
def test_any_premise_to_claim_order_scores_full(level, ordering, seed):
    """The point of the constraint. This order is not the generator's and is correct."""
    it = _item(level, seed, ordering)
    prem, rules = _split(it, _gold(it))
    r = cc.score(_answer(prem + rules), it)
    assert r["score"] == pytest.approx(1.0), r
    assert r["correct_order"] is True


@pytest.mark.parametrize("level,ordering,seed", CELLS)
def test_the_reversed_line_no_longer_scores_full(level, ordering, seed):
    it = _item(level, seed, ordering)
    gold = _gold(it)
    r = cc.score(_answer(list(reversed(gold))), it)
    assert r["exact_match"] is True, "the content is still all and only the gold lines"
    assert r["correct_order"] is False
    assert r["score"] < 1.0
    assert r["diagnostics"]["n_rules_after_their_antecedents"] == 0


@pytest.mark.parametrize("level,ordering,seed", CELLS)
def test_a_shuffled_line_scores_below_the_ordered_one(level, ordering, seed):
    it = _item(level, seed, ordering)
    gold = _gold(it)
    shuffled = random.Random(f"{level}{ordering}{seed}").sample(gold, len(gold))
    if cc.score(_answer(shuffled), it)["correct_order"]:
        pytest.skip("the shuffle happened to land on a valid premise-to-claim order")
    assert cc.score(_answer(shuffled), it)["score"] < 1.0


def test_one_misplaced_rule_costs_less_than_a_reversal():
    """Graded, not a cliff: a near miss and a wrong answer are different answers."""
    it = _item(9, 0, "last_link_elitist")
    gold = _gold(it)
    swapped = gold[:-2] + [gold[-1], gold[-2]]
    s_swap = cc.score(_answer(swapped), it)["score"]
    s_rev = cc.score(_answer(list(reversed(gold))), it)["score"]
    assert s_rev < s_swap < 1.0, (s_rev, s_swap)


def test_repeating_a_correct_line_is_not_free():
    it = _item(9, 0, "last_link_elitist")
    gold = _gold(it)
    assert cc.score(_answer(gold + [gold[0]]), it)["score"] < 1.0
