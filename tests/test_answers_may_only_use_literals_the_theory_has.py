"""A rule the answer adds may only rest on literals the theory already carries.

A contrary the answer legally introduces counts too, wherever in the answer it is written:
the check collects those first, so the two lines may be given in either order.

docs/notation.md states it -- "rule antecedents must be literals ALREADY present in the
theory" -- and `check_legality` never checked it, so a rule over two invented literals
was accepted and an answer could route its chain through an intermediate the theory does
not mention (#35). No prompt said it either, which is fixed in the same branch.
"""
from __future__ import annotations

from arggym.aspic.engine import Operation
from arggym.core.prompting import permitted_block
from arggym.core.scoring import check_legality, score_item
from arggym.tasks import counter_argument as ca

BASE = [
    Operation(kind="premise", content="p"),
    Operation(kind="premise", content="q"),
    Operation(kind="axiom", content="ax"),
    Operation(kind="defeasible", name="r1", antecedents=("p",), consequent="m"),
]


def _legal(ops):
    kept, reasons = check_legality(ops, BASE)
    return [o.name or o.content for o in kept], reasons


def test_a_rule_over_a_literal_the_theory_has_is_kept():
    for ant in ("p", "q", "ax", "m"):
        kept, reasons = _legal([Operation(kind="defeasible", name="n", antecedents=(ant,),
                                          consequent="z")])
        assert kept and not reasons, (ant, reasons)


def test_a_rule_over_an_invented_literal_is_rejected():
    kept, reasons = _legal([Operation(kind="defeasible", name="n", antecedents=("zz9",),
                                      consequent="qq1")])
    assert kept == []
    assert reasons == ["illegal_unknown_antecedent:zz9"]


def test_one_invented_antecedent_among_several_is_enough_to_reject():
    kept, reasons = _legal([Operation(kind="defeasible", name="n", antecedents=("p", "zz9"),
                                      consequent="z")])
    assert kept == []
    assert reasons == ["illegal_unknown_antecedent:zz9"]


def test_a_chain_through_an_invented_intermediate_no_longer_reaches_the_goal():
    it = ca.make_item(3, 1, "last_link_elitist")
    assert it is not None
    answer = (f"[defeasible n1: {it.seed_lit} => nw1]\n"
              f"[defeasible n2: nw1 => -{it.target}]")
    r = score_item(answer, ca.as_score_input(it))
    assert r.score == 0.0
    assert "illegal_unknown_antecedent:nw1" in r.diagnostics["illegal"]


def test_the_prompt_states_the_rule():
    for allow_strict in (False, True):
        assert "already present in the theory" in permitted_block(allow_strict)


def test_the_order_of_the_two_lines_does_not_matter():
    """`build_framework` sorts operations by kind before applying them, so nothing else in
    the pipeline is order-sensitive and no prompt asks for an order. Checked in one pass,
    a rule over a contrary was legal above the premise that introduces it and illegal
    below, which only ever costs a model."""
    add = Operation(kind="premise", content="-q")
    use = Operation(kind="defeasible", name="n", antecedents=("-q",), consequent="z")
    for order in ([add, use], [use, add]):
        kept, reasons = check_legality(order, BASE)
        assert reasons == [], f"{[o.name or o.content for o in order]}: {reasons}"
        assert {o.name or o.content for o in kept} == {"-q", "n"}


def test_a_contrary_the_answer_may_not_add_does_not_license_a_rule():
    """The pre-pass collects only contraries `check_legality` would accept anyway."""
    add = Operation(kind="premise", content="-ax")        # undermines an axiom
    use = Operation(kind="defeasible", name="n", antecedents=("-ax",), consequent="z")
    kept, reasons = check_legality([use, add], BASE)
    assert kept == []
    assert reasons == ["illegal_unknown_antecedent:-ax", "illegal_undermine_axiom:-ax"]
