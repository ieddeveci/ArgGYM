"""A rate limit is a wait, an exhausted quota is a stop, and neither is an error row.

Measured on Evren (#220): one request per minute per key, and 10M tokens a day
per account. The harness ignored `Retry-After`, backed off 5 s and 10 s, and
recorded the 429 as an error; when the daily quota ran out it kept pulling rows
and wrote every remaining one as an error. Here the stub plays both limits.
"""
from __future__ import annotations

import json
import os
import threading
import time

import pytest
from conftest import answering
from test_a_reference_answer_scores_one_end_to_end import a_run
from test_a_run_is_reachable_from_the_command_line import cfg_for

from evals import artifacts
from evals.client import Endpoint
from evals.run import StoppedOnQuota, execute, generate
from evals.score import score_run
from evals.solver import ChatSolver
from evals.types import QuotaExhausted

RESETS_AT = "2026-09-25T00:00:00Z"


def _gens(run_dir):
    return list(artifacts.read_jsonl(os.path.join(run_dir, artifacts.GENERATIONS)))


def _daily_limit(body):
    """Evren's reply once the account's daily tokens are spent, in its shape."""
    return {"__status__": 429, "error": {
        "code": "daily_token_limit_exceeded",
        "message": "Günlük token limitiniz aşıldı (tüm API anahtarlarınızın toplamı)",
        "evren": {"limit_tokens": 10000000, "resets_at": RESETS_AT}}}


def _recorded_sleeps(monkeypatch):
    """Record the client's backoff instead of sleeping it, and hand back the real
    `sleep`. `evals.client.time` is the `time` module, so this patches every
    caller's; a handler that is slow on purpose has to keep the real one."""
    import evals.client as client_module

    real_sleep = time.sleep
    slept = []
    monkeypatch.setattr(client_module.time, "sleep", slept.append)
    return slept, real_sleep


# --- (a) a rate limit is waited out ------------------------------------------


def test_a_429_with_retry_after_is_waited_out_and_leaves_no_error(tmp_path, rows,
                                                                   taskset_file,
                                                                   provider):
    """With `retries: 0` the 429 would have been an error row on the spot."""
    answer, seen = answering(rows), set()

    def once_limited(body):
        user = body["messages"][-1]["content"]
        if user not in seen:
            seen.add(user)
            return {"__status__": 429, "__headers__": {"Retry-After": "2"},
                    "error": {"code": "rate_limit_exceeded",
                              "message": "İstek limiti aşıldı."}}
        return answer(body)

    p = provider(once_limited)
    run_dir = a_run(tmp_path, p.url, rows[:2], taskset_file, retries=0)
    gens = _gens(run_dir)
    assert len(gens) == 2 and not any(g["error"] for g in gens)
    for g in gens:
        assert g["attempts"] == 2
        assert g["latency_s"] >= 2.0, g["latency_s"]
    assert len(p.requests) == 4


def test_one_request_per_window_finishes_every_row(tmp_path, rows, taskset_file,
                                                   provider):
    """Evren's limit, with a one-second window standing in for its minute.

    It counts requests started in a window, not requests in flight, and every
    worker beyond the first collects a 429 each window. They must all get
    through, however many times each is told to wait.
    """
    answer, lock, last = answering(rows), threading.Lock(), {"window": None}

    def one_per_second(body):
        with lock:
            now = time.time()
            if last["window"] == int(now):
                return {"__status__": 429, "__headers__": {"Retry-After": "1"},
                        "error": {"code": "rate_limit_exceeded", "message": "wait"}}
            last["window"] = int(now)
        return answer(body)

    p = provider(one_per_second)
    run_dir = a_run(tmp_path, p.url, rows[:4], taskset_file, retries=0,
                    concurrency=4, timeout_s=60)
    gens = _gens(run_dir)
    assert len(gens) == 4 and not any(g["error"] for g in gens), [
        g["error"] for g in gens]
    assert score_run(run_dir)["coverage"]["n_api_error"] == 0


# --- (b) an exhausted quota stops the run --------------------------------------


def test_an_exhausted_quota_stops_the_run_and_a_rerun_completes_it(tmp_path, rows,
                                                                   taskset_file,
                                                                   provider):
    answer, state = answering(rows), {"answered": 0, "quota": True}

    def three_then_quota(body):
        if state["quota"] and state["answered"] >= 3:
            return _daily_limit(body)
        state["answered"] += 1
        return answer(body)

    p = provider(three_then_quota)
    run_dir = os.fspath(tmp_path / "run")
    cfg = cfg_for(taskset_file, p.url, **{"generation.concurrency": 1,
                                          "endpoint.retries": 2})
    with pytest.raises(StoppedOnQuota) as e:
        execute(cfg, run_dir)
    # A string code is what `sys.exit` prints and exits 1 on.
    assert isinstance(e.value.code, str) and "\n" not in e.value.code
    assert RESETS_AT in e.value.code and "Rerun the same command" in e.value.code

    # One request hit the wall, it was not retried, and nothing started after it.
    assert len(p.requests) == 4
    gens = _gens(run_dir)
    assert len(gens) == 3 and not any(g["error"] for g in gens)

    with open(os.path.join(run_dir, artifacts.RUN)) as f:
        meta = json.load(f)
    assert meta["status"] == "stopped_on_quota"
    assert meta["quota_resets_at"] == RESETS_AT
    assert "daily_token_limit_exceeded" in meta["quota_error"]
    assert meta["n_errors"] == 0 and meta["n_missing"] == len(rows) - 3

    state["quota"] = False
    again = execute(cfg, run_dir)
    assert again["status"] == "completed"
    assert again["n_resumed"] == 3
    assert again["n_generated"] == len(rows) and again["n_errors"] == 0
    assert len(_gens(run_dir)) == len(rows)


def test_a_row_in_flight_when_the_quota_ends_still_lands(tmp_path, rows, provider):
    """Only the row that hit the wall and the rows not yet started are dropped."""
    answer, lock, n = answering(rows), threading.Lock(), {"n": 0}
    real_sleep = time.sleep

    def slow_first_then_quota(body):
        with lock:
            n["n"] += 1
            mine = n["n"]
        if mine == 1:
            real_sleep(1.0)
            return answer(body)
        return _daily_limit(body)

    p = provider(slow_first_then_quota)
    solver = ChatSolver(Endpoint(model="stub", base_url=p.url, timeout_s=10,
                                 retries=2))
    run_dir = os.fspath(tmp_path / "run")
    os.makedirs(run_dir)
    with pytest.raises(QuotaExhausted) as e:
        generate(rows, solver, run_dir, concurrency=2, progress=False)
    assert e.value.counts["generated"] == 1
    gens = _gens(run_dir)
    assert len(gens) == 1 and gens[0]["error"] is None
    assert len(p.requests) == 2


# --- (c) a far-off reset counts as an exhausted quota -------------------------


@pytest.mark.parametrize("reply,resets", [
    # A wait longer than `max_retry_after_s`, named by the header alone.
    ({"__status__": 429, "__headers__": {"Retry-After": "86400"},
      "error": {"message": "slow down"}}, "about a day"),
    # The same, named only by a reset time in the body.
    ({"__status__": 429, "error": {"message": "slow down",
                                   "detail": {"resets_at": "2099-01-01T00:00:00Z"}}},
     "2099-01-01T00:00:00Z"),
    # OpenAI's billing quota, which names no reset at all.
    ({"__status__": 429, "error": {"code": "insufficient_quota",
                                   "message": "You exceeded your current quota"}},
     None),
])
def test_a_far_off_reset_is_an_exhausted_quota(rows, provider, reply, resets):
    p = provider(lambda body: dict(reply))
    solver = ChatSolver(Endpoint(model="stub", base_url=p.url, timeout_s=10,
                                 retries=2))
    with pytest.raises(QuotaExhausted) as e:
        solver(rows[0])
    assert len(p.requests) == 1, "retried a quota that no retry can fix"
    if resets == "about a day":
        from datetime import datetime

        when = datetime.fromisoformat(e.value.resets_at.replace("Z", "+00:00"))
        assert abs(when.timestamp() - (time.time() + 86400)) < 60
    else:
        assert e.value.resets_at == resets


def test_a_far_future_retry_after_stops_a_run_without_error_rows(tmp_path, rows,
                                                                 taskset_file,
                                                                 provider):
    p = provider(lambda body: {"__status__": 429,
                               "__headers__": {"Retry-After": "86400"},
                               "error": {"message": "come back tomorrow"}})
    run_dir = os.fspath(tmp_path / "run")
    with pytest.raises(StoppedOnQuota) as e:
        execute(cfg_for(taskset_file, p.url, **{"generation.concurrency": 1}),
                run_dir)
    assert "Rerun the same command after" in e.value.code
    assert _gens(run_dir) == []
    with open(os.path.join(run_dir, artifacts.RUN)) as f:
        meta = json.load(f)
    assert meta["status"] == "stopped_on_quota" and meta["quota_resets_at"]


def _once(reply, rows):
    """A handler that sends `reply` to the first request and answers the rest."""
    answer, n = answering(rows), {"n": 0}

    def handler(body):
        n["n"] += 1
        return dict(reply) if n["n"] == 1 else answer(body)

    return handler


@pytest.mark.parametrize("resets_at,low,high", [
    # A clock far behind the provider's, or a field that is not the window.
    ("2099-01-01T00:00:00Z", 600, 601),
    # A clock ahead of it: the reset is already past.
    ("2000-01-01T00:00:00Z", 1, 2),
])
def test_a_rate_limit_code_never_turns_a_skewed_reset_into_a_stop(
        rows, provider, monkeypatch, resets_at, low, high):
    slept, _ = _recorded_sleeps(monkeypatch)
    p = provider(_once({"__status__": 429, "error": {
        "code": "rate_limit_exceeded", "message": "wait",
        "detail": {"resets_at": resets_at}}}, rows))
    attempt = ChatSolver(Endpoint(model="stub", base_url=p.url, timeout_s=10800,
                                  retries=0))(rows[0])
    assert attempt.error is None and attempt.attempts == 2
    assert len(slept) == 1 and low <= slept[0] <= high, slept


@pytest.mark.parametrize("retry_after", [
    "0", "-5", "Wed, 21 Oct 2015 07:28:00 GMT", "1e400", "nonsense"])
def test_a_retry_after_that_asks_for_no_wait_gets_the_ordinary_backoff(
        rows, provider, monkeypatch, retry_after):
    """Honoured, a zero would retry every half second for three hours."""
    slept, _ = _recorded_sleeps(monkeypatch)
    p = provider(lambda body: {"__status__": 429,
                               "__headers__": {"Retry-After": retry_after},
                               "error": {"message": "slow down"}})
    attempt = ChatSolver(Endpoint(model="stub", base_url=p.url, timeout_s=10800,
                                  retries=2))(rows[0])
    assert attempt.error_kind == "http_429" and attempt.attempts == 3
    assert slept == [5, 10]


def test_a_retry_after_under_a_second_is_floored(rows, provider, monkeypatch):
    slept, _ = _recorded_sleeps(monkeypatch)
    p = provider(_once({"__status__": 429, "__headers__": {"Retry-After": "0.2"},
                        "error": {"message": "slow down"}}, rows))
    attempt = ChatSolver(Endpoint(model="stub", base_url=p.url, retries=0))(rows[0])
    assert attempt.error is None
    assert len(slept) == 1 and 1 <= slept[0] <= 2, slept


def test_an_absurd_reset_is_never_a_crash(rows, provider, monkeypatch):
    """A reset past any calendar raised inside the retry loop, filing the row as
    `solver_raised`. It is still a quota; it just names no date."""
    slept, _ = _recorded_sleeps(monkeypatch)
    p = provider(lambda body: {"__status__": 429,
                               "__headers__": {"Retry-After": "99999999999999"},
                               "error": {"message": "never"}})
    with pytest.raises(QuotaExhausted) as e:
        ChatSolver(Endpoint(model="stub", base_url=p.url, retries=2))(rows[0])
    assert e.value.resets_at is None
    assert slept == []


def test_a_reset_in_milliseconds_is_read_as_one(rows, provider, monkeypatch):
    slept, _ = _recorded_sleeps(monkeypatch)
    soon_ms = int((time.time() + 30) * 1000)
    p = provider(_once({"__status__": 429, "error": {
        "message": "wait", "resets_at": soon_ms}}, rows))
    attempt = ChatSolver(Endpoint(model="stub", base_url=p.url, retries=0))(rows[0])
    assert attempt.error is None
    assert len(slept) == 1 and 25 <= slept[0] <= 32, slept

    far_ms = 4102444800000  # 2100-01-01T00:00:00Z
    p = provider(lambda body: {"__status__": 429, "error": {
        "message": "wait", "resets_at": far_ms}})
    with pytest.raises(QuotaExhausted) as e:
        ChatSolver(Endpoint(model="stub", base_url=p.url, retries=0))(rows[0])
    assert e.value.resets_at == "2100-01-01T00:00:00Z"


def test_a_quota_stop_over_the_error_rate_says_both(tmp_path, rows, taskset_file,
                                                    provider, monkeypatch):
    _recorded_sleeps(monkeypatch)
    n = {"n": 0}

    def fail_then_quota(body):
        n["n"] += 1
        if n["n"] == 1:
            return {"__status__": 400, "error": {"message": "bad"}}
        return _daily_limit(body)

    p = provider(fail_then_quota)
    with pytest.raises(StoppedOnQuota) as e:
        execute(cfg_for(taskset_file, p.url, **{"generation.concurrency": 1}),
                os.fspath(tmp_path / "run"))
    assert RESETS_AT in e.value.code and "max_error_rate" in e.value.code


# --- (d) every other failure is handled exactly as before ---------------------


@pytest.mark.parametrize("reply,kind", [
    ({"__status__": 500, "error": {"message": "boom"}}, "http_500"),
    # A `Retry-After` on anything but a 429 is not a rate limit.
    ({"__status__": 500, "__headers__": {"Retry-After": "1"},
      "error": {"message": "boom"}}, "http_500"),
    # A 429 that says nothing about when keeps the ordinary backoff.
    ({"__status__": 429, "error": {"message": "slow down"}}, "http_429"),
])
def test_an_ordinary_failure_spends_retries_and_becomes_an_error_row(
        tmp_path, rows, taskset_file, provider, monkeypatch, reply, kind):
    slept, _ = _recorded_sleeps(monkeypatch)
    p = provider(lambda body: dict(reply))
    run_dir = a_run(tmp_path, p.url, rows[:1], taskset_file, retries=2,
                    concurrency=1)
    (g,) = _gens(run_dir)
    assert g["error_kind"] == kind and g["attempts"] == 3
    assert slept == [5, 10]
    assert len(p.requests) == 3


def test_a_timeout_spends_retries_and_is_counted_as_before(tmp_path, rows,
                                                           taskset_file, provider,
                                                           monkeypatch):
    slept, real_sleep = _recorded_sleeps(monkeypatch)
    answer = answering(rows)

    def hangs(body):
        real_sleep(1.5)
        return answer(body)

    p = provider(hangs)
    run_dir = a_run(tmp_path, p.url, rows[:1], taskset_file, retries=1,
                    timeout_s=0.3, concurrency=1)
    (g,) = _gens(run_dir)
    assert g["error_kind"] == "timeout"
    assert g["attempts"] == 2 and g["requests_timed_out"] == 2
    assert slept == [5]
    cov = score_run(run_dir)["coverage"]
    assert cov["n_api_timeout"] == 1 and cov["n_requests_timed_out"] == 2
