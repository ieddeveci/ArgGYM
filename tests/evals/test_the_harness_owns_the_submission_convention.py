"""The question names no fence. The harness names one, exactly once.

`arggym/core/prompting.py:answer_format` renders no submission instruction
unless a caller asks for one, and a frozen row records `answer_template: null`.
So the convention lives here, and the sentence that asks for it and the
extraction that reads it back come from the same `AnswerTemplate` object.
"""
from __future__ import annotations

import pytest

import arggym
from evals.prompt import TEMPLATES, Elicitation, compose, region, template


def test_the_frozen_question_names_no_fence(rows):
    for row in rows:
        assert row["metadata"]["answer_template"] is None
        for fence in ("<answer>", "[answer]", r"\boxed"):
            assert fence not in row["question"], row["id"]


def test_the_instruction_and_the_extraction_come_from_one_object(rows):
    """Change the template and both ends move together, or neither does."""
    for name, tmpl in TEMPLATES.items():
        _, user = compose(rows[0], name, Elicitation())
        assert tmpl.instruction in user
        assert tmpl.open in user and tmpl.close in user
        body, found = region(tmpl.wrap("the answer"), name)
        assert found and body == "the answer", name


def test_extraction_agrees_with_the_package_on_the_shared_convention():
    """`arggym.extract_answer` is the tested one; this must not drift from it.

    The harness has its own extractor because it needs two facts the package's
    cannot give: the region under a template other than `<answer>`, and whether
    a region was found at all. On the case they share they must agree.
    """
    cases = ["<answer>one</answer>",
             "draft <answer>first</answer> revised <answer>second</answer>",
             "<answer>\n  spaced  \n</answer>",
             "<ANSWER>upper</ANSWER>",
             "no fence at all",
             ""]
    for text in cases:
        assert region(text, "xml_tags")[0] == arggym.extract_answer(text), repr(text)


def test_the_last_region_wins_because_a_reasoning_model_revises():
    body, found = region("<answer>draft</answer> ... <answer>final</answer>", "xml_tags")
    assert found and body == "final"


def test_no_template_means_the_whole_completion_is_the_answer(rows):
    _, user = compose(rows[0], None, Elicitation())
    assert user == rows[0]["question"]
    assert region("just the answer", None) == ("just the answer", True)


def test_an_unknown_template_is_refused_by_name():
    with pytest.raises(KeyError) as e:
        template("curly_braces")
    assert "xml_tags" in str(e.value)


def test_elicitation_wraps_without_touching_the_question(rows):
    e = Elicitation(name="cot", system="think first", prefix="Before:", suffix="After:")
    system, user = compose(rows[0], "xml_tags", e)
    assert system == "think first"
    assert user.startswith("Before:")
    assert user.endswith("After:")
    assert rows[0]["question"] in user
    assert "prefix" in e.summary() and "system" in e.summary()
