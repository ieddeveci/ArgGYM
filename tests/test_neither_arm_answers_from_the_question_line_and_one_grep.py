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

Five level-3 cells still take the one-liner. They are named in `DRAWN_EXEMPT` rather than
skipped, because they are the #32 draw and not this geometry: below level 6 the draw
zeroes `n_strict` for the strict arm alone so that the entry level has a one-directive
item at all, and every cell it reaches is a cell where the fixed line is the whole answer.
While they stand, #93's first acceptance criterion -- below 1.0 on *every* strict cell --
is not met. Deleting the draw meets it and reopens #32's inversion, which is why this file
asserts the exemption instead of hiding it: the day the draw goes, these five assertions
fail and say so.

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

#: The strict cells where the entry-level draw leaves the one-liner winning (#32, #93).
DRAWN_EXEMPT = {
    (3, "last_link_elitist", 0),
    (3, "last_link_elitist", 1),
    (3, "last_link_democratic", 1),
    (3, "weakest_link_elitist", 1),
    (3, "weakest_link_democratic", 1),
}


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

    exempt = (level, ordering, seed) in DRAWN_EXEMPT
    one_liner = f"[strict cs: {strict.seed_lit} -> -{strict.target}]"
    got = score_item(one_liner, ca.as_score_input(strict))

    # The bank offers the one-liner and keeps it only where it holds, so "the template
    # answers" and "the minimum is one directive" are one fact. Asserting both makes a
    # cell that starts taking the shortcut fail here rather than only in the cost law.
    assert (strict.min_directives == 1) is exempt, (
        f"{where}: min_directives {strict.min_directives} against exempt={exempt}; "
        f"the exemption list has gone stale")

    if exempt:
        assert got.success and got.score == pytest.approx(1.0), (
            f"{where}: listed as an entry-level draw cell, but the one-liner no longer "
            f"answers it -- {got.score}, {got.reason}. If the draw at "
            f"counter_argument.py went, drop this cell from DRAWN_EXEMPT and close #93's "
            f"first criterion.")
        assert strict.metadata["n_strict_final"] == 0, (
            f"{where}: exempt without the draw -- {strict.metadata['n_strict_final']} "
            f"chains reach the target strictly, so the one-liner should not answer")
    else:
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
