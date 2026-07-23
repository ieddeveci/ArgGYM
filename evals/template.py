"""The answer submission contract, applied at run time.

A taskset carries the task and its answer spec -- WHAT to answer. The delimiters
that wrap the submitted answer (HOW to submit) are an evaluator's choice, kept
out of the frozen taskset so a template can change without regenerating it and
without invalidating results, exactly like the reasoning method in
evals/elicitation.py.

The default reproduces the ``[answer]...[/answer]`` contract that the scorer's
``aspic_gym._answer_region`` reads, so applying it to a taskset prompt yields the
same bytes the prompt used to carry inline.
"""
from __future__ import annotations

from typing import Optional

# Keep in lockstep with aspic_gym._answer_region, which extracts what the model
# submits. The leading blank line separates the contract from the task's answer
# spec, matching the historical prompt layout so runs stay byte-comparable.
DEFAULT = {
    "name": "answer_tags",
    "instruction": "\n\nAnswer format: give your final answer between [answer] and [/answer].\n",
    "open": "[answer]",
    "close": "[/answer]",
}


def resolve(template: Optional[dict]) -> dict:
    return template or DEFAULT


def apply(prompt: str, template: Optional[dict] = None) -> str:
    """Append the submission contract to an item prompt."""
    return prompt + resolve(template).get("instruction", "")


def wrap(content: str, template: Optional[dict] = None) -> str:
    """Wrap raw answer content in the template's delimiters.

    Used to render a gold answer (stored raw) the way a compliant model would
    submit it -- e.g. for display or to score the gold through the full
    ``score_answer`` path rather than ``score_content``.
    """
    t = resolve(template)
    return f'{t["open"]}\n{content}\n{t["close"]}'


def describe(template: Optional[dict]) -> str:
    return resolve(template).get("name", "answer_tags")
