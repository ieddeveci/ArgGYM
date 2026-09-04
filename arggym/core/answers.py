"""How an answer arrives, and what comes back when it is scored.

Two things live here because every task needs both and each was previously
copied per module: the `[answer]` region regex was defined seven times, and the
shape of a score result was decided independently by each scorer.

The rule this module implements is that ArgGYM owns what a legal answer *is* and
the evaluator owns how it is *delivered* (`docs/dataset-contract.md`). Delimiters
are delivery, so they are accepted and never required.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Kept for evaluators that still wrap answers the old way, and for scoring
# generations recorded before the delimiters were dropped from the prompts.
_TAGGED = re.compile(r"\[answer\](.*?)\[/answer\]", re.S | re.I)


def extract_answer(text: Optional[str]) -> str:
    """The part of a completion that holds the answer.

    A wrapped answer yields its last region; anything else is returned whole.

    The last region rather than the first: a reasoning model drafts a candidate
    mid-thought and then revises it, so the first region is the draft. The v1
    harness settled this and said so in `evals/extract.py`; the v2 scorers used
    `.search`, which took the draft.
    """
    if not text:
        return ""
    found = _TAGGED.findall(text)
    return found[-1] if found else text


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
