"""The answer body is what the evaluator hands over, fenced or not.

`docs/dataset-contract.md` section 1: the fence is delivery. ArgGYM's prompts ask
for one -- `<answer>` by default, matching reasoning-gym -- because a strict
parser needs to know where the answer starts, but which fence is a render-time
choice and none of them is required at scoring time. These tests pin that in
every direction, and pin the two behaviours that are not obvious: which region
wins when there are several, and which fence wins when both are present.
"""
from arggym.core.answers import (
    DEFAULT_TEMPLATE,
    XML_TAGS,
    AnswerTemplate,
    ScoreResult,
    extract_answer,
)


def test_a_bare_answer_is_its_own_body():
    assert extract_answer("[prefer_rule: r1 > r2]") == "[prefer_rule: r1 > r2]"


def test_a_fenced_answer_gives_up_its_body():
    assert extract_answer("thinking\n<answer>\n[prefer_rule: r1 > r2]\n</answer>").strip() \
        == "[prefer_rule: r1 > r2]"


def test_only_the_fence_the_question_asks_for_is_read():
    # A second accepted convention would mean the scorer honours something no
    # prompt requests -- the stated-versus-enforced mismatch this work keeps
    # closing -- and would need a precedence rule for answers carrying both.
    # Square brackets are directive syntax here, so the text below is an answer
    # whose first and last lines are unreadable, not a fenced one.
    text = "[answer]\n[prefer_rule: a > b]\n[/answer]"
    assert extract_answer(text) == text


def test_the_default_template_is_the_reasoning_gym_one():
    # reasoning-gym's system prompts ask for <answer>answer here</answer> and its
    # `utils.extract_answer` reads the region back with the same tag name.
    assert DEFAULT_TEMPLATE is XML_TAGS
    assert (XML_TAGS.open, XML_TAGS.close) == ("<answer>", "</answer>")
    assert XML_TAGS.instruction == "Give your final answer between <answer> and </answer>."


def test_a_template_is_data_so_a_harness_can_bring_its_own():
    mine = AnswerTemplate("boxed", "\\boxed{", "}")
    assert mine.instruction == "Give your final answer between \\boxed{ and }."
    assert mine.wrap("x") == "\\boxed{\nx\n}"


def test_the_last_region_wins_because_the_first_is_the_draft():
    # A reasoning model drafts a candidate mid-thought and then revises it. The
    # first region is the draft, so taking it scores work the model discarded.
    text = "<answer>\n[prefer_rule: a > b]\n</answer>\nwait, that undermines c.\n" \
           "<answer>\n[prefer_rule: b > a]\n</answer>"
    assert extract_answer(text).strip() == "[prefer_rule: b > a]"


def test_nothing_at_all_is_an_empty_body_not_an_error():
    assert extract_answer(None) == ""
    assert extract_answer("") == ""


def test_an_unclosed_region_is_not_a_region():
    # Half a wrapper is a truncated generation, not a delimiter contract. The
    # text is returned whole so the parser sees what the model actually wrote.
    assert extract_answer("<answer>\n[prefer_rule: a > b]") \
        == "<answer>\n[prefer_rule: a > b]"


def test_success_is_carried_separately_from_the_score():
    # The contract's reason for a result object: a construction answer that
    # meets every goal is a success at 0.5 when the minimum is unknown.
    r = ScoreResult(score=0.5, success=True, reason="success_but_minimum_unknown")
    assert r.success and r.score < 1.0
    assert r.as_dict()["success"] is True
