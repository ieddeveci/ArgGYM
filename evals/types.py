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
    #: Which kind of failure `error` was, decided where the failure happened.
    #: `error` is prose for a person -- a provider's own message, an exception's
    #: `repr` -- and grouping counts by it would make the taxonomy a property of
    #: how a provider phrases things. A timeout and a 401 have to be countable
    #: apart, because only one of them moves when the token cap moves, and by
    #: the time the record reaches `score.py` the exception is gone.
    #: `evals/client.py` names the values it uses.
    error_kind: Optional[str] = None
    #: The generation hit its token cap. On the August sweep this removed 74-89%
    #: of items for three of seven models, so it is a contamination flag, not a
    #: footnote.
    truncated: bool = False
    #: Why the provider stopped. Recorded rather than reduced to `truncated`,
    #: because `content_filter` is a refusal and reducing it away leaves an
    #: empty completion that scores a hard zero on every task -- a safety
    #: refusal published as a reasoning result. Scoring cannot recover this
    #: later: it is the one thing the completion does not contain.
    finish_reason: str = ""
    #: A provider-side refusal, which OpenAI returns beside a null content.
    refusal: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    #: The whole item, retries and their backoff included. What one row cost.
    latency_s: float = 0.0
    #: The last attempt alone, which is the quantity `timeout_s` bounds: the
    #: timeout is applied per request, so a generation that failed once and
    #: succeeded on the retry has a `latency_s` well over the timeout while no
    #: single request came near it. Measuring headroom from `latency_s` would
    #: read that run as out of budget when it is not.
    attempt_latency_s: float = 0.0
    attempts: int = 0
    #: Exactly what was sent. Every provider on our list silently ignores
    #: parameters it does not recognise, so what a run actually asked for is
    #: only knowable from what we wrote down.
    request: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"completion": self.completion, "reasoning": self.reasoning,
                "value": self.value, "error": self.error,
                "error_kind": self.error_kind,
                "truncated": self.truncated, "finish_reason": self.finish_reason,
                "refusal": self.refusal, "usage": self.usage,
                "latency_s": round(self.latency_s, 3),
                "attempt_latency_s": round(self.attempt_latency_s, 3),
                "attempts": self.attempts,
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
