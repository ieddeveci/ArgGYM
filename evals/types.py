"""What a solver is, and what it hands back.

The contract's abstraction is that a solver turns a question into an answer, and
how it does that is its own business: a bare model, an agent with tools, a
symbolic procedure with no network at all. This module is the whole of that
seam. Nothing else in the harness names the OpenAI SDK.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@dataclass
class Attempt:
    """One solver's work on one row.

    It carries no score, and no extracted answer. Extraction is cheap and
    belongs beside scoring, where changing the convention costs nothing but a
    rerun of `score.py`; deciding it here would make every re-extraction cost a
    generation instead.

    `error`, `truncated` and a wrong answer are three different events. The
    first two are recorded here so the scorer can keep them apart from the
    third, because reporting infrastructure trouble as a reasoning result is the
    mistake this field makes routinely.
    """

    #: Everything the solver wrote. The answer is somewhere in here, or is not.
    completion: str = ""
    #: The reasoning trace, when the provider returns one separately.
    reasoning: str = ""
    #: Set only by a solver that produced the answer as a value rather than as
    #: text -- constrained decoding, a tool call, a schema. Scoring then routes
    #: to `arggym.score_row_value` and no extraction happens at all.
    #:
    #: It must be JSON, because it travels to `score.py` as a line in a file.
    #: `evals.values.encode` puts the two shapes that need help into JSON, and
    #: `decode` takes them back out.
    value: Any = None
    #: Non-None means the attempt did not happen. Never a score of zero.
    error: Optional[str] = None
    #: The generation hit its token cap. On the August sweep this removed 74-89%
    #: of items for three of seven models, so it is a contamination flag, not a
    #: footnote.
    truncated: bool = False
    usage: Dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0
    attempts: int = 0
    #: Exactly what was sent. Every provider on our list silently ignores
    #: parameters it does not recognise, so what a run actually asked for is
    #: only knowable from what we wrote down.
    request: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"completion": self.completion, "reasoning": self.reasoning,
                "value": self.value, "error": self.error,
                "truncated": self.truncated, "usage": self.usage,
                "latency_s": round(self.latency_s, 3), "attempts": self.attempts,
                "request": self.request}


@runtime_checkable
class Solver(Protocol):
    """Anything that turns a row into an `Attempt`.

    The row rather than the bare question, so a solver may read `task` or
    `metadata.answer_shape` when it wants to ask for a structured answer. A
    solver that only needs the text reads `row["question"]` and ignores the
    rest.
    """

    def __call__(self, row: Dict[str, Any]) -> Attempt: ...
