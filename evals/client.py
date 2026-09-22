"""The one place that speaks HTTP.

An OpenAI-compatible `/chat/completions` call, which every provider we care
about serves: OpenRouter, Gemini's compatibility endpoint, Vertex's, Azure's,
Anthropic's shim, and any local vLLM. Switching between them is a `base_url`, an
API key and a model id -- never a change of code.

Two of those layers are known to be lossy. Anthropic says of its own shim that
it is "not considered a long-term or production-ready solution", and both it and
Gemini's silently ignore parameters they do not recognise rather than refusing
them. So this module records the exact body it sent (`Attempt.request`): what a
run asked for is only knowable afterwards from what we wrote down.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from evals.types import (
    CALLER,
    CONNECTION,
    FINISH_REASON_PREFIX,
    HTTP_PREFIX,
    MALFORMED,
    REFUSAL,
    TIMEOUT,
    Attempt,
)

#: Sampling keys we will forward. An unknown key raises rather than being
#: dropped, because a run that silently ignores one looks configured, records
#: the key in its manifest, and generates as if it were never set. A whole sweep
#: on the previous harness was scored under a `repetition_penalty` that never
#: reached the server.
SAMPLING_KEYS = frozenset({
    "temperature", "top_p", "max_tokens", "max_completion_tokens", "seed",
    "stop", "presence_penalty", "frequency_penalty", "logprobs", "top_logprobs",
    "n", "reasoning_effort",
})

#: Errors worth trying again. A 4xx that is not a rate limit will not succeed on
#: the third attempt either, and retrying it burns the budget that a real
#: transient failure needs.
_RETRYABLE = (408, 409, 429, 500, 502, 503, 504, 529)

#: A provider that stopped for one of these produced an answer, or ran out of
#: room while producing one. Anything else -- `content_filter` most of all --
#: stopped for its own reasons, and the empty completion it leaves behind would
#: otherwise be scored as a wrong answer on every task.
_ANSWERED = frozenset({"stop", "length", "eos", ""})


def kind_of(exc: Exception) -> str:
    """Which failure an exception from the SDK was, labelled here because here is
    the only place that still has the exception. Downstream there is a sentence
    of prose and nothing else.

    Two separate reasons for the shape of this, and only one of them is about
    order.

    `APITimeoutError` subclasses `APIConnectionError`, so the narrower test has
    to run first or every timeout is filed as a dropped connection. That is a
    real ordering constraint and the only one here: `APIStatusError` and
    `APIResponseValidationError` are siblings under `APIError` in openai 3.8.0,
    so swapping those two branches changes nothing.

    What matters about `APIResponseValidationError` is that it carries a
    `status_code` while being a malformed *body* rather than an HTTP failure.
    Asking `getattr(exc, "status_code")`, as this did, filed a 200 whose JSON
    did not parse as `http_200`. The fix is `isinstance` against
    `APIStatusError` -- what raised, not what it happens to carry -- which is
    why the attribute is never read except off a class that guarantees it.
    """
    try:
        from openai import (
            APIConnectionError,
            APIResponseValidationError,
            APIStatusError,
            APITimeoutError,
        )
    except ImportError:  # pragma: no cover - the group is installed in dev
        return CALLER
    if isinstance(exc, APIResponseValidationError):
        return MALFORMED
    if isinstance(exc, APIStatusError):
        # The status names itself, so `http_401` and `http_429` stay apart
        # without this module listing every status a provider might answer with.
        return f"{HTTP_PREFIX}{exc.status_code}"
    if isinstance(exc, APITimeoutError):
        return TIMEOUT
    if isinstance(exc, APIConnectionError):
        return CONNECTION
    return CALLER


@dataclass
class Endpoint:
    """Where to send a request and what to ask for.

    `api_key_env` names the variable rather than carrying the key, so a run
    manifest can record the whole of this object without recording a secret.
    """

    model: str
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    sampling: Dict[str, Any] = field(default_factory=dict)
    #: Provider-specific knobs passed through untouched: OpenRouter's
    #: `reasoning` and `provider`, vLLM's `chat_template_kwargs`, Gemini's
    #: `thinking_config`. Nothing here is validated, because validating it would
    #: mean this module knowing every provider -- which is what we are avoiding.
    extra_body: Dict[str, Any] = field(default_factory=dict)
    #: The same value `conf/config.yaml` sets, and for the same reason: on the
    #: August sweep 1800s expired on 9.8%, 14.8% and 21.8% of `qwen3.6-27b`'s
    #: requests at levels 3, 6 and 9, and each retry made the server generate
    #: the whole completion again. The rate rising with level is the part that
    #: sizes a timeout, since a harder item is a longer generation. A solver
    #: built outside Hydra gets this default, so leaving it at 1800 shipped the
    #: failure to exactly the caller who never saw the config that explains it.
    #: Whether 5400 is enough is measured rather than assumed: `score.py` reports
    #: the latency percentile, how many rows came near this value, and how many
    #: requests went past it -- the last being the one that moves first, since
    #: `retries` hides an expired request behind a successful retry.
    timeout_s: float = 5400.0
    retries: int = 2

    def check(self) -> None:
        unknown = set(self.sampling) - SAMPLING_KEYS
        if unknown:
            raise ValueError(
                f"unknown sampling key(s) {sorted(unknown)}. Forwarding a key this "
                f"module does not know would leave the provider free to ignore it "
                f"while the run manifest claims it was set. Add it to "
                f"SAMPLING_KEYS if the provider takes it, or pass it in "
                f"extra_body if it is provider-specific.")

    def redacted(self) -> Dict[str, Any]:
        """Everything about the endpoint except the secret."""
        return {"model": self.model, "base_url": self.base_url,
                "api_key_env": self.api_key_env, "sampling": dict(self.sampling),
                "extra_body": dict(self.extra_body), "timeout_s": self.timeout_s,
                "retries": self.retries}


def reasoning_of(message: Any) -> str:
    """The reasoning trace, under whichever of the two names it arrived.

    vLLM renamed `reasoning_content` to `reasoning` and warns that a client can
    read an empty one while the other is populated; OpenRouter uses `reasoning`.
    Neither is in the OpenAI schema, so both arrive in `model_extra` -- the SDK's
    response models allow extra fields, which is what makes this ten lines
    instead of a dependency.
    """
    extra = getattr(message, "model_extra", None) or {}
    if not extra and isinstance(message, dict):
        extra = message
    return extra.get("reasoning") or extra.get("reasoning_content") or ""


class ChatClient:
    """One client per run. Thread-safe; the OpenAI SDK's is."""

    def __init__(self, endpoint: Endpoint) -> None:
        endpoint.check()
        self.endpoint = endpoint
        self._client = None

    def _lazy(self):
        if self._client is None:
            import os

            from openai import OpenAI

            # A local vLLM needs no key and rejects none, so an unset variable
            # is normal rather than an error. A hosted provider answers 401 and
            # that surfaces as an error on the item, which is the honest place
            # for it.
            self._client = OpenAI(
                base_url=self.endpoint.base_url,
                api_key=os.environ.get(self.endpoint.api_key_env) or "none",
                timeout=self.endpoint.timeout_s, max_retries=0)
        return self._client

    def body(self, system: Optional[str], user: str) -> Dict[str, Any]:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}]
        body: Dict[str, Any] = {"model": self.endpoint.model, "messages": messages,
                                **self.endpoint.sampling}
        if self.endpoint.extra_body:
            # A real keyword argument of the SDK, not a merged top-level key:
            # `create()` refuses an unknown kwarg, and refusing OpenRouter's
            # `reasoning` here would defeat the point of having the field.
            body["extra_body"] = dict(self.endpoint.extra_body)
        return body

    def complete(self, system: Optional[str], user: str) -> Attempt:
        """One completion, with retries, returning an `Attempt` and never raising.

        A failure here is infrastructure. It comes back as `error` so the scorer
        can keep it out of the scores rather than counting it as a model that
        answered wrongly.

        Three numbers come back about time, because one will not do. `latency_s`
        is what the row cost. `attempt_latency_s` is the slowest single request,
        which is what `timeout_s` bounds. `requests_timed_out` is how many
        requests hit the wall -- and with `retries: 2` that is the only place a
        row which timed out twice and then answered is recorded at all.
        """
        body = self.body(system, user)
        # The prompt is the bulk of the body and is already stored per row; what
        # is worth recording is everything else, which is what silently varies.
        recorded = {k: v for k, v in body.items() if k != "messages"}
        recorded["n_messages"] = len(body["messages"])
        # Not in the body -- it is a client-side deadline -- but it is the number
        # this request's latency has to be read against, and a resumed run can
        # hold generations made under two different ones. Recorded per request so
        # scoring reads the deadline each generation actually ran under rather
        # than whichever value the last invocation left in the manifest.
        recorded["timeout_s"] = self.endpoint.timeout_s
        started = time.monotonic()
        slowest = 0.0
        timed_out = 0
        last = "no attempt made"
        last_kind: Optional[str] = None
        attempt = 0
        for attempt in range(1, max(self.endpoint.retries, 0) + 2):
            attempt_started = time.monotonic()
            try:
                resp = self._lazy().chat.completions.create(**body)
            except Exception as e:  # noqa: BLE001 - the taxonomy is below
                last = f"{type(e).__name__}: {e}"
                last_kind = kind_of(e)
                # Counted per request, not per row. A row is labelled `timeout`
                # only when every attempt expired, so without this a request
                # that hit the wall and then succeeded leaves no trace: no
                # error, no kind, and a latency taken from the attempt that
                # worked. The whole completion was generated twice either way.
                timed_out += last_kind == TIMEOUT
                slowest = max(slowest, time.monotonic() - attempt_started)
                if not _retryable(e) or attempt > self.endpoint.retries:
                    break
                time.sleep(min(5 * attempt, 30))
                continue
            slowest = max(slowest, time.monotonic() - attempt_started)
            choice = resp.choices[0] if resp.choices else None
            if choice is None:
                # A 200 whose body is not the shape we expect is still
                # infrastructure, not reasoning. Retried, because a provider
                # under load can return this intermittently.
                last = "malformed response: no choices"
                last_kind = MALFORMED
                if attempt > self.endpoint.retries:
                    break
                time.sleep(min(5 * attempt, 30))
                continue
            reason = choice.finish_reason or ""
            refusal = getattr(choice.message, "refusal", None) or ""
            if reason not in _ANSWERED or refusal:
                # Not an answer, so not a score. A refusal recorded as an empty
                # completion is a zero on every task, which reports the
                # provider's policy as the model's reasoning.
                return Attempt(
                    error=f"provider did not answer: finish_reason={reason!r}"
                          + (f" refusal={refusal!r}" if refusal else ""),
                    # The stop reason when the provider gave one that is not an
                    # answer, because `content_filter` is a policy decision and
                    # an unrecognised reason is a protocol surprise. `refusal`
                    # only when the stop reason itself says nothing -- a
                    # `content_filter` arriving with a refusal string is still a
                    # content filter, and labelling that pair `refusal` throws
                    # away the one field that says which.
                    error_kind=(f"{FINISH_REASON_PREFIX}{reason}"
                                if reason not in _ANSWERED else REFUSAL),
                    completion=choice.message.content or "", refusal=refusal,
                    finish_reason=reason,
                    usage=(resp.usage.model_dump() if resp.usage else {}),
                    latency_s=time.monotonic() - started,
                    attempt_latency_s=slowest, requests_timed_out=timed_out,
                    attempts=attempt, request=recorded)
            return Attempt(
                completion=choice.message.content or "",
                reasoning=reasoning_of(choice.message),
                truncated=reason == "length", finish_reason=reason,
                usage=(resp.usage.model_dump() if resp.usage else {}),
                latency_s=time.monotonic() - started,
                attempt_latency_s=slowest, requests_timed_out=timed_out,
                attempts=attempt, request=recorded)
        return Attempt(error=last, error_kind=last_kind,
                       latency_s=time.monotonic() - started,
                       attempt_latency_s=slowest, requests_timed_out=timed_out,
                       attempts=attempt, request=recorded)


def _retryable(exc: Exception) -> bool:
    """Whether trying again could plausibly work.

    Narrow on purpose. Anything without a status code used to be retried, which
    swept in `ImportError` from a missing dependency and `TypeError` from a bad
    sampling value: every row then burned three attempts and fifteen seconds of
    backoff on a fault no retry can fix, and 480 of those look like a provider
    outage rather than a typo.
    """
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status in _RETRYABLE
    try:
        from openai import APIConnectionError, APITimeoutError
    except ImportError:  # pragma: no cover - the group is installed in dev
        return False
    # A dropped connection or a timeout carries no status and is exactly the
    # case retrying exists for. A bug in our own call path is not.
    return isinstance(exc, (APIConnectionError, APITimeoutError))
