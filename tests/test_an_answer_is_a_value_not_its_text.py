"""`claim_chain` and `defeat_diagnosis` score a value; text is one way to submit it.

`docs/dataset-contract.md` section 4 splits each scorer in two. `parse` reads a value
out of text and `score_value` grades the value, so a solver with constrained decoding,
a JSON schema or a tool call submits the value and never imitates our serialization.
`score` is the two composed, which is what makes the two routes score alike: there is
one scorer, and text reaches it through the parser.

The value is plain lists and dicts, so a schema can describe it and JSON can carry it.
"""
from __future__ import annotations

import json

import pytest

from arggym.core.answers import UnparseableAnswer
from arggym.tasks import claim_chain as cc
from arggym.tasks import defeat_diagnosis as dd

ORDERING = "last_link_elitist"


@pytest.fixture(scope="module")
def cc_item():
    item = cc.make_item(6, 0, ORDERING)
    assert item is not None
    return item


@pytest.fixture(scope="module")
def dd_item():
    """Level 9, so the routes carry a defeated defeater and `survives_because` is asked for."""
    item = dd.make_item(9, 0, ORDERING)
    assert item is not None
    assert any(d["survives_because"] for d in item.diagnoses)
    return item


def cc_answers(item):
    lines = [cc.render_op(o) for o in item.line_ops]
    spare = next(cc.render_op(o) for o in item.base_ops if cc.render_op(o) not in set(lines))
    return {
        "the reference": "\n".join(lines),
        "the reference reversed": "\n".join(reversed(lines)),
        "one directive short": "\n".join(lines[:-1]),
        "one directive too many": "\n".join(lines + [spare]),
        "a repeated directive": "\n".join(lines + [lines[0]]),
        "nothing": "",
        "prose": "The line runs from the premise to the claim.",
    }


def dd_answers(item):
    lines = item.reference.splitlines()
    return {
        "the reference": item.reference,
        "the reference reversed": "\n".join(reversed(lines)),
        "one failure point short": "\n".join(lines[:-1]),
        "one failure point too many": "\n".join(
            lines + ["defeated_at: zz9; defeater: zz8; kind: rebut"]),
        "the status alone": lines[0],
        "the failure points alone": "\n".join(lines[1:]),
        "nothing": "",
    }


@pytest.mark.parametrize("case", [
    "the reference", "the reference reversed", "one directive short",
    "one directive too many", "a repeated directive", "nothing", "prose"])
def test_a_claim_chain_value_scores_exactly_as_the_text_that_parses_to_it(cc_item, case):
    text = cc_answers(cc_item)[case]
    assert cc.score(text, cc_item) == cc.score_value(cc.parse(text, cc_item), cc_item)


@pytest.mark.parametrize("case", [
    "the reference", "the reference reversed", "one failure point short",
    "one failure point too many", "the status alone", "the failure points alone",
    "nothing"])
def test_a_defeat_diagnosis_value_scores_exactly_as_the_text_that_parses_to_it(dd_item, case):
    text = dd_answers(dd_item)[case]
    assert dd.score(text, dd_item) == dd.score_value(dd.parse(text, dd_item), dd_item)


def test_a_claim_chain_value_a_schema_could_emit_scores_one(cc_item):
    """The value is the quoted directives in order, and nothing about it is text."""
    value = [cc.render_op(o) for o in cc_item.line_ops]
    assert json.loads(json.dumps(value)) == value
    assert cc.score_value(value, cc_item) == cc.score(cc_item.reference, cc_item)
    assert cc.score_value(value, cc_item).score == 1.0


def test_a_defeat_diagnosis_value_a_schema_could_emit_scores_one(dd_item):
    """The status and one record per failure point, each carrying the prompt's fields."""
    value = {"status": [dd_item.claim_status.lower()],
             "records": [{k: d[k] for k in ("defeated_at", "defeater", "kind")}
                         | ({"survives_because": d["survives_because"]}
                            if d["survives_because"] else {})
                         for d in dd_item.diagnoses]}
    assert json.loads(json.dumps(value)) == value
    assert dd.score_value(value, dd_item) == dd.score(dd_item.reference, dd_item)
    assert dd.score_value(value, dd_item).score == 1.0


def test_the_order_a_claim_chain_value_lists_its_directives_in_is_scored(cc_item):
    """The prompt asks for the line premise to claim, so the value is a sequence."""
    value = [cc.render_op(o) for o in cc_item.line_ops]
    assert cc.score_value(value, cc_item).score == 1.0
    assert cc.score_value(list(reversed(value)), cc_item).score == 0.0


def test_the_order_a_defeat_diagnosis_value_lists_its_records_in_is_not_scored(dd_item):
    """Failure points are a set: the prompt asks which they are, not which comes first."""
    value = {"status": [dd_item.claim_status.lower()],
             "records": [{k: d[k] for k in ("defeated_at", "defeater", "kind")}
                         | ({"survives_because": d["survives_because"]}
                            if d["survives_because"] else {})
                         for d in dd_item.diagnoses]}
    reordered = {"status": value["status"], "records": list(reversed(value["records"]))}
    assert dd.score_value(reordered, dd_item) == dd.score_value(value, dd_item)


def test_an_empty_answer_parses_to_an_empty_value_rather_than_failing_to_parse(cc_item, dd_item):
    """It is an answer with nothing in it, not an answer that failed to arrive."""
    assert cc.parse("", cc_item) == []
    assert cc.score_value([], cc_item).reason == "empty_answer"
    assert dd.parse("", dd_item) == {"status": [], "records": []}
    assert dd.score_value({"status": [], "records": []}, dd_item).reason == "empty_answer"


@pytest.mark.parametrize("text, reason", [
    ("[premise: aa0] and then it follows", "unparseable_tokens:4"),
])
def test_claim_chain_refuses_text_that_spells_out_no_answer(cc_item, text, reason):
    with pytest.raises(UnparseableAnswer) as caught:
        cc.parse(text, cc_item)
    assert caught.value.reason == reason


@pytest.mark.parametrize("text, reason", [
    ("The claim is not justified because a rule is undercut.", "unparseable_tokens:10"),
    ("status: overruled\ndefeated_at: aa0; defeater: bb1; kind: sideways", "invalid_kind:sideways"),
])
def test_defeat_diagnosis_refuses_text_that_spells_out_no_answer(dd_item, text, reason):
    with pytest.raises(UnparseableAnswer) as caught:
        dd.parse(text, dd_item)
    assert caught.value.reason == reason


@pytest.mark.parametrize("module, text", [
    (cc, "[premise: aa0] and then it follows"),
    (dd, "The claim is not justified because a rule is undercut."),
    (dd, "status: overruled\ndefeated_at: aa0; defeater: bb1; kind: sideways"),
])
def test_text_that_does_not_parse_scores_zero_and_says_why(module, text, cc_item, dd_item):
    """A harness scoring a taskset wants a row for every item, so `score` catches it."""
    item = cc_item if module is cc else dd_item
    with pytest.raises(UnparseableAnswer) as caught:
        module.parse(text, item)
    result = module.score(text, item)
    assert result.score == 0.0 and result.success is False
    assert result.reason == caught.value.reason
    assert caught.value.diagnostics.items() <= result.diagnostics.items()


@pytest.mark.parametrize("value", [
    [], ["not a directive"], ["[premise: nowhere]", "[premise: nowhere]"]])
def test_score_value_grades_a_claim_chain_value_it_cannot_use_rather_than_raising(cc_item, value):
    """A solver that submits a value has done its own parsing; its failures are its own."""
    assert cc.score_value(value, cc_item).score == 0.0


@pytest.mark.parametrize("value", [
    {"status": [], "records": []},
    {"status": ["justified"], "records": [{"defeated_at": "aa0"}]},
    {"status": ["overruled"], "records": [{"defeated_at": "aa0", "defeater": "bb1",
                                           "kind": "sideways"}]},
])
def test_score_value_grades_a_defeat_diagnosis_value_it_cannot_use_rather_than_raising(
        dd_item, value):
    """A solver that submits a value has done its own parsing; its failures are its own."""
    assert dd.score_value(value, dd_item).score < 1.0


def test_a_record_naming_no_defeater_is_not_a_failure_point_in_either_route(dd_item):
    """`survives_because` under such a record is a reason about it, not about the one above.

    The text rule and the value rule are the same rule: a record without a defeater
    names nothing the answer can be scored against, and the reason written under it
    goes with it.
    """
    gold = dd_item.diagnoses[0]
    text = "\n".join([f"status: {dd_item.claim_status.lower()}",
                      f"defeated_at: {gold['defeated_at']}; defeater: {gold['defeater']}; "
                      f"kind: {gold['kind']}",
                      f"defeated_at: {gold['defeated_at']}",
                      f"survives_because: {gold['survives_because']}"])
    named, unnamed = dd.parse(text, dd_item)["records"]
    assert "defeater" not in unnamed
    assert unnamed["survives_because"] == gold["survives_because"]
    assert "survives_because" not in named
    assert dd.score(text, dd_item).diagnostics["survives_because_correct"] == 0
