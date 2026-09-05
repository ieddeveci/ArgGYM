"""How an answer arrives, and what comes back when it is scored.

Three things live here because every task needs them and each was previously
decided per module: how a prompt asks for its answer to be fenced, how that
fence is read back off a completion, and the shape of a score result.

The rule this module implements is that ArgGYM owns what a legal answer *is* and
the evaluator owns how it is *delivered* (`docs/dataset-contract.md`). The fence
is delivery. That is why it is a render-time choice rather than a property of
the benchmark: `AnswerTemplate` says which delimiters a prompt asks for, the
default matches reasoning-gym's `<answer>` tags, and a harness with another
convention re-renders with its own template instead of editing prompt strings.

What the fence is *for* is the other half. Strict parsing means every line
inside the fence has to be an answer line, so a model that reasons before or
after its answer is not charged for the reasoning -- the fence is what separates
the two, and without one the thinking would be read as a malformed answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class AnswerTemplate:
    """The delimiters a prompt asks for, and the sentence that asks for them.

    `name` is what a frozen row records, so a taskset says what its questions
    asked for rather than leaving a reader to infer it from the prompt text.
    """

    name: str
    open: str
    close: str

    @property
    def instruction(self) -> str:
        return f"Give your final answer between {self.open} and {self.close}."

    def wrap(self, body: str) -> str:
        return f"{self.open}\n{body}\n{self.close}"


#: The default, matching reasoning-gym: its system prompts ask for
#: `<answer>answer here</answer>` and `reasoning_gym/utils.py:25` reads the
#: region back with `<{tag}>\s?(.*?)\s?</{tag}>`.
XML_TAGS = AnswerTemplate("xml_tags", "<answer>", "</answer>")
#: The convention ArgGYM asked for before this one. Kept so a harness can
#: reproduce an older prompt without editing strings.
SQUARE_TAGS = AnswerTemplate("square_tags", "[answer]", "[/answer]")
DEFAULT_TEMPLATE = XML_TAGS

_XML = re.compile(r"<answer>\s?(.*?)\s?</answer>", re.S | re.I)
# Still read, so generations recorded against the older prompts keep scoring.
_SQUARE = re.compile(r"\[answer\](.*?)\[/answer\]", re.S | re.I)


def extract_answer(text: Optional[str]) -> str:
    """The part of a completion that holds the answer.

    A fenced answer yields its last region, `<answer>` first and `[answer]`
    after it; an unfenced completion is returned whole. The last case is a
    fallback for an evaluator that already cut the answer out, not the path the
    prompts describe.

    The last region rather than the first: a reasoning model drafts a candidate
    mid-thought and then revises it, so the first region is the draft. The v1
    harness settled this and said so in `evals/extract.py`; the v2 scorers used
    `.search`, which took the draft.
    """
    if not text:
        return ""
    for pattern in (_XML, _SQUARE):
        found = pattern.findall(text)
        if found:
            return found[-1]
    return text


@dataclass(frozen=True)
class ScoreResult:
    """One score, with the same fields whatever the task.

    `success` is the task's own definition of a fully correct answer and is
    always present. It is not `score == 1.0`: a construction answer that meets
    every goal is a success at 0.5 when the minimal directive count is unknown,
    because goal satisfaction is the task and economy is a second measurement on
    the same answer.

    Anything task-specific goes in `diagnostics`, so a harness can aggregate
    across tasks without knowing twelve return shapes.
    """

    score: float
    success: bool
    reason: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"score": self.score, "success": self.success,
                "reason": self.reason, "diagnostics": self.diagnostics}
