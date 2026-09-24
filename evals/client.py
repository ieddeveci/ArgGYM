"""The one place that speaks HTTP.

An OpenAI-compatible `/chat/completions` call, which every provider we care
about serves: OpenRouter, Gemini's compatibility endpoint, Vertex's, Azure's,
Anthropic's shim, Evren, and any local vLLM. Switching between them is a
`base_url`, an API key and a model id -- never a change of code.

Two of those layers are known to be lossy. Anthropic says of its own shim that
it is "not considered a long-term or production-ready solution", and both it and
Gemini's silently ignore parameters they do not recognise rather than refusing
them. So this module records the exact body it sent (`Attempt.request`): what a
run asked for is only knowable afterwards from what we wrote down.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
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
    QuotaExhausted,
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

#: Codes a provider puts on a 429 when the quota is gone until a reset hours
#: away, so no retry inside the run can succeed: Evren's daily token limit (per
#: account, reset at 00:00 UTC) and OpenAI's exhausted billing quota. A 429
#: naming no such code is caught by its reset time instead; see `_told_to_wait`.
QUOTA_CODES = frozenset({"daily_token_limit_exceeded", "insufficient_quota"})

#: Codes that say a 429 is only a rate limit, whose window ends within minutes.
#: A `resets_at` beside one is read as a wait capped at `max_retry_after_s` and
#: never as a quota, so a skewed clock or a misread field cannot stop a run.
RATE_LIMIT_CODES = frozenset({"rate_limit_exceeded"})

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
    #: The same value `conf/config.yaml` sets, and for the same reason: it is a
    #: safety net for a hung connection, sized so the largest token cap in
    #: `conf/model/` (65,536) finishes at 10 tokens/s, because the cap and not
    #: the clock should end a long generation. A timeout turns a row into an API
    #: error, which leaves the score's denominator; a truncation does not. A
    #: solver built outside Hydra gets this default, so it must not lag the
    #: config. `score.py` reports the latency percentile, how many rows came
    #: near this value, and how many requests went past it -- the last being
    #: the one that moves first, since `retries` hides an expired request
    #: behind a successful retry.
    timeout_s: float = 10800.0
    retries: int = 2
    #: The longest single `Retry-After` a 429 is waited out for. A provider
    #: asking for longer has run out of quota rather than rate, and the run
    #: stops on `QuotaExhausted` instead of sleeping through it. Ten minutes,
    #: above the longest rate window we know of: Evren's token limit slides
    #: over 5 minutes (`GET /v1/quota`), so its waits run to about 300 s.
    max_retry_after_s: float = 600.0

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
                "retries": self.retries,
                "max_retry_after_s": self.max_retry_after_s}


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
        """One completion, with retries, returning an `Attempt`.

        A failure here is infrastructure. It comes back as `error` so the scorer
        can keep it out of the scores rather than counting it as a model that
        answered wrongly. The one exception is an exhausted quota, which raises
        `QuotaExhausted`: that is not this row failing but every row after it,
        and recording it as an error would write one junk row per item left.

        A 429 carrying `Retry-After` is waited out and does not spend
        `retries`, so a provider that limits requests per minute slows a run
        down rather than failing its rows. A row stops honouring those waits
        once they add up to `timeout_s`, so a provider that never lets a
        request through still ends the row.

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
        # `attempt` counts requests and `retried` the ones that spent `retries`.
        # They differ by the 429s waited out on the provider's word.
        attempt = retried = 0
        waited = 0.0
        while True:
            attempt += 1
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
                wait = _told_to_wait(e, self.endpoint.max_retry_after_s)
                if wait is not None and waited + wait <= self.endpoint.timeout_s:
                    # The jitter spreads workers told the same second, which
                    # would otherwise all wake and collide again. Counted in
                    # `waited`, so `Retry-After: 0` forever still ends.
                    pause = wait + random.uniform(0, 1)
                    waited += pause
                    time.sleep(pause)
                    continue
                if not _retryable(e) or retried >= self.endpoint.retries:
                    break
                retried += 1
                time.sleep(min(5 * retried, 30))
                continue
            slowest = max(slowest, time.monotonic() - attempt_started)
            choice = resp.choices[0] if resp.choices else None
            if choice is None:
                # A 200 whose body is not the shape we expect is still
                # infrastructure, not reasoning. Retried, because a provider
                # under load can return this intermittently.
                last = "malformed response: no choices"
                last_kind = MALFORMED
                if retried >= self.endpoint.retries:
                    break
                retried += 1
                time.sleep(min(5 * retried, 30))
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
    backoff on a fault no retry can fix, and a taskset's worth of those look like a provider
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


def _told_to_wait(exc: Exception, cap_s: float) -> Optional[float]:
    """How long a 429 asked us to wait, or `None` to fall back on the backoff.

    The wait comes from `Retry-After`, else from a `resets_at` in the error
    body. Raises `QuotaExhausted` when the answer is "not within this run": the
    body names a code in `QUOTA_CODES`, or the wait is longer than `cap_s`. A
    body under a code in `RATE_LIMIT_CODES` never raises on its `resets_at`.

    Every wait is at least `_MIN_WAIT_S`. A `Retry-After` that is zero, in the
    past or not a finite number says nothing usable, and falls back on the
    backoff, which spends `retries` and so ends.
    """
    if getattr(exc, "status_code", None) != 429:
        return None
    # The SDK has already unwrapped `{"error": {...}}` into `body`.
    body = getattr(exc, "body", None)
    body = body if isinstance(body, dict) else {}
    code = body.get("code")
    response = getattr(exc, "response", None)
    header = _retry_after(response.headers.get("retry-after")
                          if response is not None else None)
    reset = _reset_time(body)
    now = time.time()

    def stop(when: Optional[float]) -> QuotaExhausted:
        return QuotaExhausted(
            " ".join(str(x) for x in ("HTTP 429", code, body.get("message") or exc)
                     if x),
            resets_at=_utc(when))

    if code in QUOTA_CODES:
        raise stop(reset)
    if header is not None:
        if header > cap_s:
            raise stop(reset if reset is not None else now + header)
        return max(header, _MIN_WAIT_S)
    if reset is None:
        return None
    wait = reset - now
    if code in RATE_LIMIT_CODES:
        # A clock skewed either way still gets a wait of at most the cap.
        return max(min(wait, cap_s), _MIN_WAIT_S)
    if wait > cap_s:
        raise stop(reset)
    return max(wait, _MIN_WAIT_S) if wait > 0 else None


#: The shortest wait a 429 gets, so a provider's `0.2` is not a busy loop.
_MIN_WAIT_S = 1.0


def _utc(when: Optional[float]) -> Optional[str]:
    """A Unix time as ISO 8601 UTC, or `None` for one no calendar holds."""
    try:
        return None if when is None else datetime.fromtimestamp(
            when, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OverflowError, OSError, ValueError):
        return None


def _retry_after(value: Optional[str]) -> Optional[float]:
    """`Retry-After` in seconds, or `None` when it asks for no positive wait.

    RFC 9110 allows a number or an HTTP date.
    """
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return None
    return seconds if math.isfinite(seconds) and seconds > 0 else None


def _reset_time(body: Dict[str, Any]) -> Optional[float]:
    """A `resets_at` from anywhere in an error body, as a Unix time.

    Searched rather than addressed, because each provider nests it under a name
    of its own (Evren: `error.evren.resets_at`). An ISO 8601 string or a Unix
    time in seconds or milliseconds; anything else is ignored.
    """
    for key, value in body.items():
        if isinstance(value, dict):
            found = _reset_time(value)
            if found is not None:
                return found
        elif key == "resets_at":
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                if not math.isfinite(value):
                    continue
                # Seconds reach 1e11 in the year 5138; milliseconds passed it
                # in 1973.
                return value / 1000 if value > 1e11 else float(value)
            try:
                when = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                continue
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return when.timestamp()
    return None
