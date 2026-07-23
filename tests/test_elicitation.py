"""Elicitation is the evaluator's choice and must stay out of the taskset."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals import elicitation, template  # noqa: E402
from evals.client import build_payload  # noqa: E402
from prompting import _format_block  # noqa: E402


def test_none_is_a_true_noop():
    """The default must not alter the taskset bytes the model sees."""
    prompt = "Theory: [premise: p]\n\nAnswer format: ..."
    for cfg in (None, {}, {"name": "none", "system": None,
                           "prefix": None, "suffix": None}):
        out, system = elicitation.apply(prompt, cfg)
        assert out == prompt
        assert system is None


def test_suffix_and_prefix_wrap_the_item_prompt():
    out, system = elicitation.apply(
        "ITEM", {"name": "x", "prefix": "BEFORE", "suffix": "AFTER"})
    assert out == "BEFORE\n\nITEM\n\nAFTER"
    assert system is None
    assert "ITEM" in out


def test_system_is_returned_separately_not_concatenated():
    out, system = elicitation.apply("ITEM", {"name": "x", "system": "SYS"})
    assert out == "ITEM"
    assert system == "SYS"


def test_system_becomes_a_system_message():
    body = build_payload("m", "ITEM", {}, 16, system="SYS")
    assert body["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "ITEM"},
    ]


def test_no_system_message_when_none():
    body = build_payload("m", "ITEM", {}, 16)
    assert body["messages"] == [{"role": "user", "content": "ITEM"}]


def test_item_prompts_carry_no_reasoning_scaffold():
    """The taskset states WHAT to answer -- not how to think, nor how to submit.

    `[reasoning]` / `[/reasoning]` were never parsed by anything -- see
    _answer_region, which matches only [answer] -- while instructing a thinking
    model to open a reasoning section pushed its answer into `reasoning_content`.
    The submission contract ([answer] tags) is likewise a runtime template, not
    baked into the item prompt.
    """
    block = _format_block("name the attackers")
    assert "[reasoning]" not in block and "[/reasoning]" not in block
    assert "step by step" not in block.lower()
    # neither the reasoning scaffold nor the submission contract is in the taskset
    assert "[answer]" not in block and "[/answer]" not in block
    # the submission contract is supplied at run time by the template
    assert "[answer]" in template.apply(block) and "[/answer]" in template.apply(block)


def test_describe_records_what_was_applied():
    assert elicitation.describe(None)["name"] == "none"
    d = elicitation.describe({"name": "cot", "suffix": "think first"})
    assert d == {"name": "cot", "system": False, "prefix": False, "suffix": True}
