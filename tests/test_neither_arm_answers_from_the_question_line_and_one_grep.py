"""Neither arm answers from the question line plus one grep of the theory.

Two templates, one per arm. The strict one reads nothing but the question line, which
prints the target and the premises:

    [strict cs: <premise> -> -<target>]

The plain one is that move made legal, plus a single grep of the printed rules:

    [defeasible w: <premise> => -<target>]
    [prefer_rule: w > <r>]        for each defeasible rule r concluding <target>

Neither follows a chain, a preference or an ordering. Scored through `score_item` with
legality on, the strict one took 1.0 on 21 of the 40 exported strict cells and the plain
one reached both goals on 8 of the 40 plain cells, 2 of them at 1.0 (#93).

One geometry produced both. `mid_target` puts the target on a junction literal, every
chain reached that literal through its rule at index 0, and the strict flag named the
chain's *last* rule -- past the target and irrelevant to it. So no chain reached the
target strictly: nothing contradicted the strict counter-argument, and on the plain side
rebutting the target left only the rules concluding it to outrank. The junction sits at
index 1 now and the flag names the rule that reaches the target, so a mid-target chain
blocks the shortcut exactly as a last-rule strict chain always did.

**The title is a bound, not a claim about every template.** A walk of two hops still
answers the strict arm on every cell -- rebut the target strictly, then undercut each rule
feeding a strict rule that concludes it -- and it did so on `main` too, so it is a defect
this change neither causes nor closes. That is #116; do not read this file as saying the
arm cannot be answered without reading an argument, because it can.

`not success` rather than `score < 1.0`. An answer that reaches both goals and loses only
economy is the same defect wearing a smaller number: on `main` the plain sibling scored
0.90 to 0.9286 on six of the eight cells it succeeded, so a `< 1.0` assertion would have
passed on six of the cells this file exists to name.

No cell takes the one-liner now, so there is no exemption list. `DRAWN_EXEMPT` named five
level-3 cells while the #32 draw zeroed `n_strict` for the strict arm alone, and this file
asserted the exemption rather than hiding it so that the day the draw went those five
assertions would fail and say so. They did. Every level builds at least two
target-reaching chains, at least one strict and at least one defeasible (#167), so the strict minimum is `1 + k` with
`k >= 1` everywhere and #93's first acceptance criterion -- the fixed line below 1.0 on
*every* strict cell -- is met for the first time. The sweep asserts it flat, on all 120
cells, with nothing carved out.

One test per cell rather than one per template, so a cell that starts answering is named
once with both templates' verdicts in hand.
"""
from __future__ import annotations

from typing import List

import pytest

from arggym.core.scoring import score_item
from arggym.core.spec import ALL_ORDERINGS, LEVELS, SEEDS
from arggym.tasks import counter_argument as ca

CHEAP = [(lv, o, s) for lv in (3, 6) for o in ALL_ORDERINGS for s in SEEDS]
GRID = [(lv, o, s) for lv in LEVELS for o in ALL_ORDERINGS for s in SEEDS]
DEAR = [c for c in GRID if c not in CHEAP]


def _plain_sibling(item) -> List[str]:
    """Rebut the target from a premise, then outrank the rebuttal over what concludes it.

    Read off `base_ops`, which is the theory the prompt prints: an answerer parses the
    rendered lines back and gets the same list.
    """
    killers = [o.name for o in item.base_ops
               if o.kind == "defeasible" and o.consequent == item.target]
    return ([f"[defeasible w: {item.seed_lit} => -{item.target}]"]
            + [f"[prefer_rule: w > {name}]" for name in killers])


def _no_template_answers_this_cell(level: int, ordering: str, seed: int) -> None:
    strict = ca.make_item(level, seed, ordering, allow_strict=True)
    plain = ca.make_item(level, seed, ordering, allow_strict=False)
    where = f"L{level} {ordering} seed {seed}"
    assert strict is not None and plain is not None, where

    one_liner = f"[strict cs: {strict.seed_lit} -> -{strict.target}]"
    got = score_item(one_liner, ca.as_score_input(strict))

    # The bank offers the one-liner and keeps it only where it holds, so "the template
    # answers" and "the minimum is one directive" are one fact. Asserting both makes a
    # cell that starts taking the shortcut fail here rather than only in the cost law.
    assert strict.min_directives > 1, (
        f"{where}: min_directives {strict.min_directives}, so one directive is the whole "
        f"answer and no chain reaches the target strictly. Every level builds at least "
        f"one strict chain (#167); this cell lost that.")
    assert not got.success, (
        f"{where}: {one_liner} reaches both goals reading nothing but the question "
        f"line, scoring {got.score}")

    sibling = _plain_sibling(plain)
    got = score_item("\n".join(sibling), ca.as_score_input(plain))
    assert not got.success, (
        f"{where}: the plain arm is answered by a rebuttal plus a grep, scoring "
        f"{got.score}:\n" + "\n".join(sibling))


@pytest.mark.parametrize("level,ordering,seed", CHEAP)
def test_neither_arm_answers_from_the_question_line_and_one_grep(level, ordering, seed):
    _no_template_answers_this_cell(level, ordering, seed)


@pytest.mark.slow
@pytest.mark.parametrize("level,ordering,seed", DEAR)
def test_neither_arm_answers_that_way_at_every_exported_cell(level, ordering, seed):
    _no_template_answers_this_cell(level, ordering, seed)
