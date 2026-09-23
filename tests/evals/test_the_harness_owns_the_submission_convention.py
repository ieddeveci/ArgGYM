"""The question names no fence. The harness names one, exactly once.

`arggym/core/prompting.py:answer_format` renders no submission instruction at
all, and generation takes no argument that would render one. So the convention
lives here, and the sentence that asks for it and the extraction that reads it
back come from the same `AnswerTemplate` object.
"""
from __future__ import annotations

import pytest

import arggym
from evals.prompt import TEMPLATES, Elicitation, compose, region, template


def test_the_frozen_question_names_no_fence(rows):
    for row in rows:
        assert "answer_template" not in row["metadata"]
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
             "I'll use <answer> tags.\n<answer>\nreal\n</answer>",
             "<answer>X</answer> revising: <answer>Y",
             "<answer>a<answer>b</answer>c</answer>",
             "<answer>unclosed",
             "no fence at all",
             ""]
    for text in cases:
        assert region(text, "xml_tags")[0] == arggym.extract_answer(text), repr(text)


def test_the_last_region_wins_because_a_reasoning_model_revises():
    body, found = region("<answer>draft</answer> ... <answer>final</answer>", "xml_tags")
    assert found and body == "final"


def test_a_tag_named_in_the_reasoning_does_not_start_the_region():
    """#183, under every template: the region starts at the last opening delimiter."""
    for name, tmpl in TEMPLATES.items():
        text = f"I'll put it in {tmpl.open} as asked.\n{tmpl.wrap('final')}"
        assert region(text, name) == ("final", True), name


def test_the_reasoning_fallback_ignores_a_tag_named_in_the_reasoning():
    from evals.score import answer_of

    found = answer_of({"completion": "",
                       "reasoning": "I'll answer in <answer> tags.\n<answer>\nX\n</answer>"},
                      "xml_tags")
    assert found["answer"] == "X" and found["answer_in_cot"] is True


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


def test_an_answer_only_in_the_reasoning_is_found_and_labelled(rows):
    """A model that reached an answer but never submitted it did something else.

    Crediting it is a scoring-policy choice, so it has to be visible: the score
    counts, and `answer_in_cot` says the answer came out of the thinking rather
    than out of a submission. Deleting this branch left the whole suite green
    before, which meant the policy could change by accident.
    """
    from evals.score import answer_of

    row = rows[0]
    ref = row["reference_answer"]
    found = answer_of({"completion": "I could not decide.",
                       "reasoning": f"maybe <answer>{ref}</answer>"}, "xml_tags")
    assert found["answer"] == ref
    assert found["answer_in_cot"] is True
    assert found["no_answer_region"] is False


def test_a_submitted_answer_is_preferred_over_one_in_the_reasoning(rows):
    """The fallback rescues an unsubmitted answer; it never overrides a submitted one."""
    from evals.score import answer_of

    found = answer_of({"completion": "<answer>submitted</answer>",
                       "reasoning": "<answer>draft</answer>"}, "xml_tags")
    assert found["answer"] == "submitted"
    assert found["answer_in_cot"] is False


def test_nothing_anywhere_is_reported_as_no_region(rows):
    from evals.score import answer_of

    found = answer_of({"completion": "no idea", "reasoning": "still no idea"},
                      "xml_tags")
    assert found["no_answer_region"] is True
    assert found["answer_in_cot"] is False
