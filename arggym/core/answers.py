"""How an answer arrives, and what comes back when it is scored.

ArgGYM owns what a legal answer *is*. The harness owns how it gets one: it
composes the prompt from the question, calls whatever solver it likes, and
extracts the answer before handing it over. Nothing here is called by a scorer.

So `AnswerTemplate` and `extract_answer` are conveniences for a harness that
wants the common convention -- one names a fence in a prompt, the other reads it
back -- and a harness with a JSON schema, constrained decoding or a symbolic
solver uses neither.

What the dataset does define is the shape of what comes back: `ScoreResult` for
a score, and `UnparseableAnswer` for text that does not spell out an answer at
all (`docs/dataset-contract.md`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class AnswerTemplate:
    """The delimiters a harness asks for, and the sentence that asks for them.

    `name` is what a run records (`evals/run.py`), so a set of completions says
    which convention produced it. No frozen row records it: a question names no
    fence, so there is nothing about delivery for the dataset to write down.
    """

    name: str
    open: str
    close: str

    @property
    def instruction(self) -> str:
        return f"First, state your reasoning between <thought> and </thought>. Then, give your final answer between {self.open} and {self.close}."

    def wrap(self, body: str) -> str:
        return f"{self.open}\n{body}\n{self.close}"


#: The default, matching reasoning-gym: its system prompts ask for
#: `<answer>answer here</answer>` and `reasoning_gym/utils.py:25` reads the
#: region back with `<{tag}>\s?(.*?)\s?</{tag}>`.
XML_TAGS = AnswerTemplate("xml_tags", "<answer>", "</answer>")
DEFAULT_TEMPLATE = XML_TAGS

_FENCE = re.compile(r"<answer>\s?(.*?)\s?</answer>", re.S | re.I)


def extract_answer(text: Optional[str]) -> str:
    """The part of a completion that holds the answer, for a harness that wants it.

    Nothing in ArgGYM calls this. Pulling the answer out of a completion is the
    harness's job, and this is offered because the convention below is a common
    one, not because the dataset has an opinion. A harness using another fence
    reads its own; one with a structured-output solver needs no extraction at all.

    A fenced answer yields its last region; unfenced text is returned whole.

    Only one fence is read. Accepting several would need a precedence rule for a
    completion carrying two of them, and there is no reason to prefer either.

    The last region rather than the first: a reasoning model drafts a candidate
    mid-thought and then revises it, so the first region is the draft. The v1
    harness settled this and said so in `evals/extract.py`; the v2 scorers used
    `.search`, which took the draft.
    """
    if not text:
        return ""
    found = _FENCE.findall(text)
    return found[-1] if found else text


class UnparseableAnswer(ValueError):
    """Text that does not spell out an answer of the shape the task expects.

    Raised by `parse`, never by `score_value`: a solver that hands over a value
    has already done its own parsing, and its failures are its own. `score` turns
    this back into a zero `ScoreResult` so a harness scoring a whole taskset gets
    a row rather than an exception.
    """

    def __init__(self, reason: str, diagnostics: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.diagnostics: Dict[str, Any] = dict(diagnostics or {})


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
