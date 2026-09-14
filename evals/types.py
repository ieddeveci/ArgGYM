"""What a solver is, and what it hands back.

The contract's abstraction is that a solver turns a question into an answer, and
how it does that is its own business: a bare model, an agent with tools, a
symbolic procedure with no network at all. This module is the whole of that
seam. Nothing else in the harness names the OpenAI SDK.

It also holds the vocabulary of `Attempt.error_kind`, because that vocabulary is
part of the seam: `client.py` writes it, `run.py` adds the two failures that
happen outside a client, and `score.py` groups by it. Three modules agreeing on
a set of strings is a set of constants, not three sets of literals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable

#: The closed part of the `error_kind` vocabulary: one label per failure we can
#: tell apart at the moment it happens. `timeout` is the one that has to stand
#: alone, because it is the failure that moves with the token cap, and inside a
#: single `n_api_error` a rising timeout rate and a rising 429 rate are the same
#: number.
TIMEOUT = "timeout"
CONNECTION = "connection"
MALFORMED = "malformed_response"
REFUSAL = "refusal"
#: Neither a status nor a transport failure: a `TypeError` from a bad sampling
#: value, an `ImportError` from a missing dependency. Ours, not the provider's.
CALLER = "caller_error"
#: The two that happen outside any client, in `run.py`: a solver that raised,
#: and a solver whose `value` would not serialise.
SOLVER_RAISED = "solver_raised"
SOLVER_VALUE_NOT_JSON = "solver_value_not_json"

#: And the open part. A status code names itself (`http_401`) and a provider's
#: stop reason names itself (`finish_reason_content_filter`), rather than being
#: mapped onto a closed set. Deliberate, and for the reason `extra_body` exists:
#: normalising them would mean this harness knowing every provider's vocabulary,
#: and a provider we have not met yet would be filed under whichever of our
#: buckets was least wrong. The cost is that `http_429` from two providers is
#: one row and `finish_reason_length_cap` from vLLM is its own row even where
#: another provider calls the same thing something else. A reader comparing two
#: providers' error tables has to read the labels; a reader watching one
#: provider over time, which is what this is for, does not.
HTTP_PREFIX = "http_"
FINISH_REASON_PREFIX = "finish_reason_"

#: Every label this harness mints without a provider in the loop. `http_<n>` and
#: `finish_reason_<r>` are excluded by construction -- that is what open means.
CLOSED_ERROR_KINDS = frozenset({
    TIMEOUT, CONNECTION, MALFORMED, REFUSAL, CALLER, SOLVER_RAISED,
    SOLVER_VALUE_NOT_JSON,
})


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
    #: the time the record reaches `score.py` the exception is gone. The
    #: vocabulary is at the top of this module.
    #:
    #: `None` beside a `None` error means nothing failed. `None` beside a real
    #: error means the cause was not recorded, which is true only of generations
    #: made before this field existed; `score.py` calls that `unclassified` and
    #: refuses to guess.
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
    #: The whole item, every attempt and every backoff included. What one row
    #: cost in wall time.
    #:
    #: `None`, not `0.0`, when nothing timed it. A solver is free to be a
    #: symbolic procedure with no clock in it, and a run of 480 such rows
    #: publishing a 95th-percentile latency of zero seconds would report 480
    #: measurements that were never taken as the healthiest endpoint on record.
    latency_s: Optional[float] = None
    #: The slowest single request made for this row, which is the quantity
    #: `timeout_s` bounds -- the timeout applies per request, not per row.
    #:
    #: The slowest rather than the last, because the last is the one that
    #: succeeded and the interesting one is the one that nearly did not. An item
    #: whose first request spent 99% of the budget and whose second returned at
    #: once is an item at the wall, and reporting the second hides that. Reading
    #: the whole of `latency_s` instead would overshoot the other way: it adds
    #: the attempts together and exceeds the timeout after a single retry,
    #: reading a run as out of budget while every request had room.
    attempt_latency_s: Optional[float] = None
    #: How many of this row's requests hit the wall. Not derivable from
    #: `error_kind`, which describes the row's final outcome: with the shipped
    #: `retries: 2`, a row is labelled `timeout` only if all three requests
    #: expired, so a request that timed out and then succeeded leaves no trace
    #: anywhere else. That row still cost two full generations of server time,
    #: which is the waste `conf/config.yaml` sized the timeout to avoid.
    requests_timed_out: int = 0
    attempts: int = 0
    #: Exactly what was sent, plus the two things about the request that are not
    #: in its body: how many messages it carried, and the deadline it was sent
    #: under. Every provider on our list silently ignores parameters it does not
    #: recognise, so what a run actually asked for is only knowable from what we
    #: wrote down -- and the deadline is what its latency has to be read against.
    request: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"completion": self.completion, "reasoning": self.reasoning,
                "value": self.value, "error": self.error,
                "error_kind": self.error_kind,
                "truncated": self.truncated, "finish_reason": self.finish_reason,
                "refusal": self.refusal, "usage": self.usage,
                "latency_s": _round(self.latency_s),
                "attempt_latency_s": _round(self.attempt_latency_s),
                "requests_timed_out": self.requests_timed_out,
                "attempts": self.attempts,
                "request": self.request}


def _round(x: Optional[float]) -> Optional[float]:
    """An unmeasured duration stays unmeasured. `round(None, 3)` raises, and the
    obvious guard -- `round(x or 0.0, 3)` -- publishes the zero this field
    exists to avoid."""
    return None if x is None else round(x, 3)


@runtime_checkable
class Solver(Protocol):
    """Anything that turns a row into an `Attempt`.

    The row rather than the bare question, so a solver may read `task` or
    `metadata.answer_shape` when it wants to ask for a structured answer. A
    solver that only needs the text reads `row["question"]` and ignores the
    rest.
    """

    def __call__(self, row: Dict[str, Any]) -> Attempt: ...
