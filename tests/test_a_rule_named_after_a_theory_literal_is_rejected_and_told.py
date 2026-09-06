"""A new rule named after a literal already in the theory scores zero and says so (#89).

`-<name>` for a rule name switches that rule off (NOTATION.md, undercutting); the same
`-<name>` for a literal is its negation. `check_legality` built `rule_names` from the
base theory's own rules and checked a new name against only that set, so a rule named
after a premise, an axiom, or another rule's antecedent or consequent built and applied
fine -- the model's new rule silently took over the literal's contrary slot instead of
adding a rule, and whatever preference the theory already carried on that literal decided
whether the takeover mattered. The reference at `counter_argument_strict` level 9 goes
from score 1.0 to 0.0 this way by renaming one rule, with `diagnostics["illegal"] == []`:
a zero indistinguishable from reasoning badly. Renaming the rule fixes the theory only for
the rule that clashes with it; the two namespaces stay shared, so the check has to reject
the collision rather than resolve it.
"""
from __future__ import annotations

import arggym
from arggym.aspic.engine import Operation
from arggym.core.scoring import check_legality

BASE = [
    Operation(kind="premise", content="p"),
    Operation(kind="premise", content="q"),
    Operation(kind="axiom", content="ax"),
    Operation(kind="defeasible", name="r1", antecedents=("p",), consequent="m"),
]


def _legal(ops):
    kept, reasons = check_legality(ops, BASE)
    return [o.name or o.content for o in kept], reasons


def test_a_rule_named_after_a_premise_is_rejected():
    kept, reasons = _legal([Operation(kind="defeasible", name="q", antecedents=("p",),
                                      consequent="z")])
    assert kept == []
    assert reasons == ["illegal_name_collides_with_literal:q"]


def test_a_rule_named_after_an_axiom_is_rejected():
    kept, reasons = _legal([Operation(kind="defeasible", name="ax", antecedents=("p",),
                                      consequent="z")])
    assert kept == []
    assert reasons == ["illegal_name_collides_with_literal:ax"]


def test_a_rule_named_after_another_rules_consequent_is_rejected():
    kept, reasons = _legal([Operation(kind="defeasible", name="m", antecedents=("p",),
                                      consequent="z")])
    assert kept == []
    assert reasons == ["illegal_name_collides_with_literal:m"]


def test_a_fresh_literal_shaped_name_absent_from_the_theory_is_kept():
    """The rejection is a namespace check, not a shape check -- a name that merely
    looks like a literal (`zz9` in #89) and is not one is a legal, ordinary rule name."""
    kept, reasons = _legal([Operation(kind="defeasible", name="zz9", antecedents=("p",),
                                      consequent="z")])
    assert reasons == []
    assert kept and kept[0] == "zz9"


def test_the_reference_answer_collides_with_its_own_theory_and_is_told_so():
    """The measured case from #89: renaming the reference's rule to a premise that
    carries a preference flips the verdict from 1.0 to 0.0 with `illegal: []` on `main`.
    A wrong score with no flag is the defect -- both halves are asserted, so a fix that
    rejects the answer but forgets to name why still fails this test."""
    row = arggym.TaskDataset("counter_argument_strict", 9, "last_link_elitist",
                             size=1, seed=0)[0]
    ref = row["reference_answer"]
    assert " z0:" in ref, "the reference no longer names its rule z0; update the rename"
    colliding = ref.replace(" z0:", " as6:")

    honest = arggym.score_row(ref, row)
    assert honest.score == 1.0

    result = arggym.score_row(colliding, row)
    assert result.score == 0.0
    assert "illegal_name_collides_with_literal:as6" in result.diagnostics["illegal"], (
        "a zero with an empty `illegal` list is indistinguishable from reasoning badly")
