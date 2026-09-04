"""The answer body is what the evaluator hands over, wrapped or not.

`docs/dataset-contract.md` section 1: delimiters are delivery, so ArgGYM accepts
them and never requires them. These tests pin that both directions, and pin the
one behaviour that is not obvious -- which region wins when there are several.
"""
from arggym.core.answers import ScoreResult, extract_answer


def test_a_bare_answer_is_its_own_body():
    assert extract_answer("[prefer_rule: r1 > r2]") == "[prefer_rule: r1 > r2]"


def test_a_wrapped_answer_gives_up_its_body():
    assert extract_answer("thinking\n[answer]\n[prefer_rule: r1 > r2]\n[/answer]").strip() \
        == "[prefer_rule: r1 > r2]"


def test_the_last_region_wins_because_the_first_is_the_draft():
    # A reasoning model drafts a candidate mid-thought and then revises it. The
    # first region is the draft, so taking it scores work the model discarded.
    text = "[answer]\n[prefer_rule: a > b]\n[/answer]\nwait, that undermines c.\n" \
           "[answer]\n[prefer_rule: b > a]\n[/answer]"
    assert extract_answer(text).strip() == "[prefer_rule: b > a]"


def test_nothing_at_all_is_an_empty_body_not_an_error():
    assert extract_answer(None) == ""
    assert extract_answer("") == ""


def test_an_unclosed_region_is_not_a_region():
    # Half a wrapper is a truncated generation, not a delimiter contract. The
    # text is returned whole so the parser sees what the model actually wrote.
    assert extract_answer("[answer]\n[prefer_rule: a > b]") \
        == "[answer]\n[prefer_rule: a > b]"


def test_success_is_carried_separately_from_the_score():
    # The contract's reason for a result object: a construction answer that
    # meets every goal is a success at 0.5 when the minimum is unknown.
    r = ScoreResult(score=0.5, success=True, reason="success_but_minimum_unknown")
    assert r.success and r.score < 1.0
    assert r.as_dict()["success"] is True
