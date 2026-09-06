"""A level that turns junctions off must get none, on either side of the argument.

`build_defence` places junctions on the support chain and hands the leftovers to the
attackers, so a support chain too short to hold the budget still spends it: the leftover
goes to `min(leftover, n_attackers)` of them, one each. The leftover read
`n_junctions - len(j_points)`, and an empty `j_points` is what both "none wanted" and
"none fitted" look like, so the reader could not tell them apart and `junction=False`
sent the whole budget to the attacker side (#31). Level 5 was where the defence schedule
hit that: it asked for no junctions and got two.
"""
from __future__ import annotations

import pytest

from arggym.core.curriculum import DEFENCE, junctions_for
from arggym.core.spec import ALL_ORDERINGS, SEEDS
from arggym.structures.defence import build_defence
from arggym.tasks import attack_defense as ad

LAST_LINK = "last_link_elitist"
ALL_LEVELS = range(1, 16)


def _multi_antecedent(ops):
    return [o.name for o in ops
            if o.kind in ("defeasible", "strict") and len(o.antecedents) > 1]


def _schedule_budgets():
    """Every (support depth, junction budget) pair the defence schedule can ask for."""
    out = set()
    for level in ALL_LEVELS:
        n, sup, atk_d, _ns, _nd, _nds = ad._defence_shape(level)
        out.add((n, sup, atk_d, junctions_for(level, max(1, n * 4))))
    return sorted(out)


@pytest.mark.parametrize("n,sup,atk_d,budget", _schedule_budgets())
def test_no_junction_survives_the_flag_being_off(n, sup, atk_d, budget):
    item = build_defence(n, LAST_LINK, iter(ad._names(1234, ad._POOL)),
                         support_depth=sup, attacker_depth=atk_d,
                         junction=False, n_junctions=budget)
    assert item is not None
    assert _multi_antecedent(item.all_ops()) == [], (
        f"{n} attackers, support depth {sup}, budget {budget}: junctions with the flag off")


@pytest.mark.parametrize("sup", [2, 3, 4, 5])
@pytest.mark.parametrize("budget", [0, 1, 2, 3, 4, 5])
def test_the_flag_is_off_at_every_budget_and_depth(sup, budget):
    """Wider than the schedule, so a new rung cannot reintroduce the defect quietly."""
    item = build_defence(3, LAST_LINK, iter(ad._names(99, ad._POOL)),
                         support_depth=sup, attacker_depth=3,
                         junction=False, n_junctions=budget)
    assert item is not None
    assert _multi_antecedent(item.all_ops()) == []


def test_the_attackers_still_take_the_junctions_the_support_chain_cannot_hold():
    """The other half of the rule, which the fix must not turn off.

    A support chain of depth 3 has one usable junction position, so a budget of 3 leaves
    two for the attackers. That is deliberate: the budget is what the level asks the item
    to cost, and dropping the remainder would make the shorter chain cheaper than asked.
    """
    item = build_defence(3, LAST_LINK, iter(ad._names(1234, ad._POOL)),
                         support_depth=3, attacker_depth=3,
                         junction=True, n_junctions=3)
    assert item is not None
    names = _multi_antecedent(item.all_ops())
    assert sum(1 for x in names if x.startswith("s")) == 1, names
    assert sum(1 for x in names if x.startswith("x")) == 2, names


@pytest.mark.parametrize("level", [lv for lv in ALL_LEVELS if lv < ad.JUNCTION_LEVEL])
def test_a_defence_item_below_the_junction_level_has_no_junctions(level):
    """Through the builder, where the flag and the budget are both set by the level.

    Level 5 is the case that used to fail. It is also the only one that could: below it
    `junctions_for` returns nothing to misspend, and from level 6 the flag is on.
    """
    for ordering in ALL_ORDERINGS:
        for seed in SEEDS:
            it = ad.make_item(level, seed, ordering, mode=DEFENCE)
            assert it is not None, f"L{level} {ordering} seed {seed} produced no item"
            assert it.metadata["n_junctions"] == 0, (
                f"L{level} {ordering} seed {seed}: {it.metadata['n_junctions']} junctions "
                f"below the level junctions start at")
