"""A timeout is a silent loss, so the margin has to be reported before it is gone.

`timeout_s` was raised from 1800 to 5400 on the strength of one sweep where 1800
expired on 9.8%, 14.8% and 21.8% of one model's requests at levels 3, 6 and 9 --
and nothing then measured whether the new number was enough. The failure it
guards against is invisible while it is approaching: a run at 95% of its budget
and a run on a healthy endpoint produce identical artifacts right up to the day
the first one starts losing items, and each lost item is a whole completion the
server generates again on the retry.

Three things make it visible, and the third is the one that is easy to get
wrong. The latency the timeout actually bounds reaches `metrics.json` with the
timeout beside it. `n_api_error` is split by cause, so the failure mode that
moves with the token cap is countable on its own. And every request that hit the
wall is counted even when its row went on to answer -- because with the shipped
`retries: 2` a row is only *labelled* a timeout when all three of its requests
expire, so the first, cheapest warning is a row that recovered and would
otherwise appear in no number at all.
"""
from __future__ import annotations

import json
import os
import time

import pytest
from conftest import answering, completion
from test_a_reference_answer_scores_one_end_to_end import a_run

from evals import artifacts
from evals.client import Endpoint, kind_of
from evals.report import render, write_csv
from evals.run import generate
from evals.score import coverage, score_run
from evals.solver import ChatSolver
from evals.types import CLOSED_ERROR_KINDS, MALFORMED, Attempt


def _cells(line):
    """One markdown row's cells, label first."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def a_sample(**kw):
    """One `samples.jsonl` record, in the shape `coverage` reads."""
    out = {"id": "x", "task": "attack", "level": 3, "ordering": "last_link_elitist",
           "api_error": None, "api_error_kind": None, "truncated": False,
           "latency_s": None, "attempt_latency_s": None, "requests_timed_out": 0,
           "attempts": 1, "scorer_refused": False, "score": 1.0, "success": True,
           "no_answer_region": False, "answer_in_cot": False}
    out.update(kw)
    return out


def _no_backoff(monkeypatch):
    """Skip the five-second retry backoff, and hand back the real `sleep`.

    `evals.client.time` is the `time` module itself, so patching its `sleep`
    patches everyone's -- including a stub handler that is being deliberately
    slow. A handler that stopped sleeping would make these tests measure
    nothing, so it keeps the real one.
    """
    import evals.client as client_module

    real_sleep = time.sleep
    monkeypatch.setattr(client_module.time, "sleep", lambda _s: None)
    return real_sleep


# --- a request that hit the wall and recovered -------------------------------


def test_a_request_that_hit_the_wall_and_then_answered_is_still_counted(
        tmp_path, rows, taskset_file, provider, monkeypatch):
    """The whole point, and the case every other number here misses.

    `retries: 2` is shipped. A row carries `error_kind="timeout"` only when all
    three of its requests expired, and `attempt_latency_s` would say nothing if
    it were the *last* attempt -- the one that succeeded. So a row that hit the
    wall and recovered was absent from the error count, absent from the cause
    split, and absent from the percentiles, while having cost two full
    completions of server time. The report printed a healthy run.
    """
    real_sleep = _no_backoff(monkeypatch)
    seen = {"n": 0}

    def wall_on_the_first_try(body):
        seen["n"] += 1
        if seen["n"] == 1:
            real_sleep(1.5)
        return completion("<answer>x</answer>")

    p = provider(wall_on_the_first_try)
    run_dir = a_run(tmp_path, p.url, rows[:1], taskset_file, timeout_s=0.5,
                    retries=1, concurrency=1)
    c = score_run(run_dir)["coverage"]

    # The row answered, so nothing here is an error and nothing is a lost row.
    assert c["n_api_error"] == 0
    assert c["n_api_timeout"] == 0
    assert c["api_errors_by_kind"] == {}
    # And the wall was hit anyway. This is the only number that says so.
    assert c["n_requests_timed_out"] == 1
    # The headroom is measured from the slowest request, not the one that
    # worked, so the run does not read as comfortable either.
    assert c["n_near_timeout"] == 1
    assert c["latency_s_max"] >= 0.5


def test_the_headroom_comes_from_the_slowest_request_not_the_one_that_worked(
        rows, provider, monkeypatch):
    """Three clocks, three questions, and only one of them answers this one.

    `latency_s` adds the attempts up and passes the timeout after a single
    retry, which reads a run as out of budget while every request had room.
    The last attempt is the one that succeeded and says nothing about the one
    that nearly did not. The slowest attempt is the quantity `timeout_s` bounds.
    """
    real_sleep = _no_backoff(monkeypatch)
    calls = {"n": 0}

    def flaky(body):
        calls["n"] += 1
        if calls["n"] == 1:
            real_sleep(0.6)
            return {"__status__": 503, "error": {"message": "overloaded"}}
        return completion("<answer>x</answer>")

    p = provider(flaky)
    attempt = ChatSolver(Endpoint(model="stub", base_url=p.url, timeout_s=10,
                                  retries=1))(rows[0])

    assert attempt.error is None and attempt.attempts == 2
    # The slow attempt failed and the fast one answered, so the last-attempt
    # reading would be milliseconds and the whole-item reading would be more
    # than either request took.
    assert attempt.attempt_latency_s >= 0.6
    assert attempt.latency_s >= attempt.attempt_latency_s
    # Not a timeout: a 503 is not the wall.
    assert attempt.requests_timed_out == 0


def test_a_row_that_never_recovered_is_counted_in_both_places(
        tmp_path, rows, taskset_file, provider, monkeypatch):
    """Every request expired, so the row is lost *and* three requests were paid
    for. The two counts answer different questions and both are printed."""
    real_sleep = _no_backoff(monkeypatch)

    def always_past_the_wall(body):
        real_sleep(1.5)
        return completion("<answer>x</answer>")

    p = provider(always_past_the_wall)
    run_dir = a_run(tmp_path, p.url, rows[:1], taskset_file, timeout_s=0.3,
                    retries=2, concurrency=1)
    c = score_run(run_dir)["coverage"]

    # One row, lost. Three requests, all of them paid for in full.
    assert c["n_api_error"] == 1
    assert c["n_api_timeout"] == 1
    assert c["api_errors_by_kind"] == {"timeout": 1}
    assert c["n_requests_timed_out"] == 3
    # Nothing answered, so there is no latency to report -- and the counts above
    # are the loud thing, not a silently empty percentile.
    assert c["n_latency_measured"] == 0
    assert c["latency_s_p95"] is None


def test_the_wall_hit_count_reaches_the_report_and_the_csv(
        tmp_path, rows, taskset_file, provider, monkeypatch):
    """A number that stops at `metrics.json` is a number nobody reads."""
    import csv as _csv

    real_sleep = _no_backoff(monkeypatch)
    seen = {"n": 0}

    def wall_on_the_first_try(body):
        seen["n"] += 1
        if seen["n"] == 1:
            real_sleep(1.5)
        return completion("<answer>x</answer>")

    p = provider(wall_on_the_first_try)
    run_dir = a_run(tmp_path, p.url, rows[:1], taskset_file, timeout_s=0.5,
                    retries=1, concurrency=1)
    score_run(run_dir)
    with open(os.path.join(run_dir, "metrics.json")) as f:
        m = json.load(f)
    m["_label"] = "stub"

    text = render([m])
    header = [ln for ln in text.splitlines() if ln.startswith("| run |")][0]
    assert "requests that hit the timeout" in header
    row = [ln for ln in text.splitlines() if ln.startswith("| stub |")][0]
    # API errors 0, rows lost 0, requests that hit the wall 1.
    assert _cells(row)[3:6] == ["0", "0", "1"], row

    out = os.fspath(tmp_path / "report" / "results.csv")
    write_csv([m], out)
    with open(out) as f:
        got = list(_csv.DictReader(f))
    assert {r["n_requests_timed_out"] for r in got} == {"1"}


# --- the headroom itself -----------------------------------------------------


def test_the_timeout_a_run_used_reaches_the_file_its_latency_is_judged_by(
        tmp_path, rows, taskset_file, provider):
    """2,295s is comfortable under 5,400 and lost under 1,800.

    The percentile alone says neither, so the timeout it is measured against
    travels with it.
    """
    p = provider(answering(rows))
    run_dir = a_run(tmp_path, p.url, rows, taskset_file, timeout_s=1234)
    c = score_run(run_dir)["coverage"]

    assert c["timeout_s"] == 1234
    # And the line drawn on it, so the count below can be read without knowing
    # which fraction this version of the scorer used.
    assert c["near_timeout_s"] == 987.2
    assert c["n_near_timeout"] == 0
    assert c["n_latency_measured"] == len(rows)
    # Only that it is there. These fields are published rounded to a tenth of a
    # second -- a scale set by generations that take thousands of them -- and
    # this stub answers in milliseconds, so its honest maximum really is `0.0`.
    # That a request which merely succeeded gets clocked at all is pinned by
    # `test_a_request_that_simply_succeeded_is_timed_too`, which makes the
    # handler slow enough for the published precision to show it.
    assert c["latency_s_max"] is not None


def test_a_request_that_simply_succeeded_is_timed_too(
        tmp_path, rows, taskset_file, provider):
    """The ordinary path, which every other test here reaches past.

    Every latency assertion in this file was satisfiable by a number written on
    the way out of a *failure*: a timeout writes the wall, a retry writes the
    attempt that expired. Nothing pinned the clock on a request that just
    worked, so a healthy 12-row run could report `attempt_latency_s: 0.0`
    twelve times and p95 of zero seconds -- "every item infinitely far from the
    wall", the same failure this file names for a solver with no clock in it,
    reached through the client instead.

    So the handler is deliberately slow by a known amount and the recorded
    latency has to be at least that. A clock that is not started measures zero;
    a clock that is started cannot measure less than the sleep.
    """
    slow_by = 0.2

    def slow_but_fine(body):
        time.sleep(slow_by)
        return completion("<answer>x</answer>")

    p = provider(slow_but_fine)
    run_dir = a_run(tmp_path, p.url, rows[:2], taskset_file, timeout_s=30,
                    concurrency=1)
    gens = list(artifacts.read_jsonl(os.path.join(run_dir, artifacts.GENERATIONS)))
    assert all(g["error"] is None for g in gens), gens
    for g in gens:
        assert g["attempt_latency_s"] >= slow_by, g
        assert g["latency_s"] >= g["attempt_latency_s"], g

    c = score_run(run_dir)["coverage"]
    assert c["n_latency_measured"] == 2
    assert c["latency_s_p50"] >= slow_by
    assert c["latency_s_p95"] >= slow_by
    assert c["latency_s_max"] >= slow_by


def test_a_generation_near_the_wall_is_counted_and_a_fast_one_is_not():
    """The count is what decides anything; the percentile only says how close."""
    c = coverage([a_sample(attempt_latency_s=s) for s in (10.0, 79.0, 80.0, 99.0)],
                 timeout_s=100.0)

    assert c["near_timeout_s"] == 80.0
    # 80.0 is at the line and counts. A warning that fires only past the line
    # fires after the item it was meant to save.
    assert c["n_near_timeout"] == 2
    assert c["latency_s_max"] == 99.0
    assert c["latency_s_p50"] == 79.0
    assert c["latency_s_p95"] == 99.0


def test_the_percentile_is_a_latency_some_request_actually_had():
    """Nearest-rank, not interpolated: a timeout sized against a duration no
    request took is sized against nothing."""
    c = coverage([a_sample(attempt_latency_s=float(s)) for s in range(1, 101)],
                 timeout_s=1000.0)
    assert c["latency_s_p95"] == 95.0
    assert c["latency_s_p50"] == 50.0


def test_headroom_is_measured_per_request_and_not_per_item():
    """`timeout_s` bounds one request, so one request is what is compared to it.

    A row that retried costs more than the timeout in wall time while no single
    request came near it. Measured from the row, that run reads as out of budget
    when it has most of its budget left.
    """
    c = coverage([a_sample(latency_s=7000.0, attempt_latency_s=2000.0, attempts=3)],
                 timeout_s=5400.0)
    assert c["latency_s_max"] == 2000.0
    assert c["n_near_timeout"] == 0


def test_a_failed_generation_is_counted_but_never_averaged_into_the_latency():
    """A timeout's latency is the timeout, not a measurement of the model.

    Letting it into the percentile pins the percentile to the wall exactly when
    the wall is being hit, which is when the warning is needed; letting a 401's
    milliseconds in drags the percentile down while nothing is being exercised
    at all. Both are counted by cause instead.
    """
    records = [a_sample(score=None, api_error="timed out", api_error_kind="timeout",
                        attempt_latency_s=5400.0, requests_timed_out=3),
               a_sample(score=None, api_error="unauthorized",
                        api_error_kind="http_401", attempt_latency_s=0.01),
               a_sample(attempt_latency_s=12.0)]
    c = coverage(records, timeout_s=5400.0)

    assert c["n_latency_measured"] == 1
    assert c["latency_s_max"] == 12.0
    assert c["n_near_timeout"] == 0
    assert c["n_api_error"] == 2
    assert c["n_api_timeout"] == 1
    assert c["n_requests_timed_out"] == 3


def test_a_run_whose_generations_name_no_timeout_still_reports_its_latency():
    """No timeout recorded means no distance to the wall, and `0` is not that."""
    c = coverage([a_sample(attempt_latency_s=12.0)], timeout_s=None)
    assert c["latency_s_max"] == 12.0
    assert c["timeout_s"] is None
    assert c["near_timeout_s"] is None
    assert c["n_near_timeout"] is None


def test_a_solver_with_no_clock_publishes_no_latency_rather_than_zero(
        tmp_path, rows, taskset_file):
    """`evals/solver.py` advertises a solver with no model in it.

    Such a solver sets no latency, and a default of `0.0` would publish one
    measurement per row that nobody took as the healthiest endpoint ever recorded -- p95 of
    zero seconds, every item infinitely far from the wall.
    """
    def symbolic(row):
        return Attempt(completion="<answer>x</answer>")

    run_dir = os.fspath(tmp_path / "run")
    os.makedirs(run_dir)
    generate(rows[:2], symbolic, run_dir, concurrency=1, progress=False)
    gens = list(artifacts.read_jsonl(os.path.join(run_dir, artifacts.GENERATIONS)))
    assert [g["latency_s"] for g in gens] == [None, None]
    assert [g["attempt_latency_s"] for g in gens] == [None, None]

    c = coverage([a_sample(latency_s=None, attempt_latency_s=None)
                  for _ in gens], timeout_s=5400.0)
    assert c["n_latency_measured"] == 0
    assert c["latency_s_p95"] is None
    assert c["n_near_timeout"] == 0


# --- the timeout a generation was actually made under ------------------------


def test_the_yardstick_comes_from_the_generations_not_the_manifest(
        tmp_path, rows, taskset_file, provider):
    """`run.py` rewrites `run.json` on every invocation, and a changed timeout is
    not a changed run.

    `refuse_a_changed_run` deliberately ignores `timeout_s` -- a different
    deadline is not a different question, so refusing to resume over one would
    cost a sweep its generations for nothing. That was harmless until the
    timeout became the yardstick. Now a run resumed under a larger deadline
    would have its old generations measured against the new line.
    """
    p = provider(answering(rows))
    run_dir = a_run(tmp_path, p.url, rows[:2], taskset_file, timeout_s=900)
    run_path = os.path.join(run_dir, artifacts.RUN)
    meta = json.load(open(run_path))
    meta["endpoint"] = dict(meta["endpoint"], timeout_s=5400)
    artifacts.write_json(run_path, meta)

    c = score_run(run_dir)["coverage"]
    assert c["timeout_s"] == 900, "read the manifest instead of the generations"
    assert c["near_timeout_s"] == 720.0


def test_a_directory_holding_two_deadlines_reports_no_headroom(
        tmp_path, rows, taskset_file, provider):
    """One number cannot describe two walls, and the larger one is the flattering
    one. No headroom is the honest answer; the latencies are still reported."""
    p = provider(answering(rows))
    run_dir = a_run(tmp_path, p.url, rows[:2], taskset_file, timeout_s=900)
    path = os.path.join(run_dir, artifacts.GENERATIONS)
    mixed = list(artifacts.read_jsonl(path))
    mixed[0]["request"]["timeout_s"] = 5400
    artifacts.write_jsonl(path, mixed)

    c = score_run(run_dir)["coverage"]
    assert c["timeout_s"] is None
    assert c["near_timeout_s"] is None
    assert c["n_near_timeout"] is None
    # The latencies themselves are unaffected: an ambiguous wall removes the
    # comparison, never the measurement.
    assert c["n_latency_measured"] == 2


# --- the split by cause ------------------------------------------------------


def test_a_timeout_is_counted_apart_from_the_other_api_errors(
        tmp_path, rows, taskset_file, provider, monkeypatch):
    """Through a real socket, so the label is put on what the SDK actually raises."""
    real_sleep = _no_backoff(monkeypatch)

    def slow(body):
        real_sleep(1.5)
        return completion("<answer>x</answer>")

    p = provider(slow)
    run_dir = a_run(tmp_path, p.url, rows[:1], taskset_file, timeout_s=0.25)
    c = score_run(run_dir)["coverage"]

    assert c["n_api_error"] == 1
    assert c["n_api_timeout"] == 1
    assert c["api_errors_by_kind"] == {"timeout": 1}


def test_a_rejected_request_is_not_counted_as_a_timeout(tmp_path, rows,
                                                        taskset_file, provider):
    """A 401 and a timeout were one number, and only one of them moves with the
    token cap."""
    p = provider(lambda body: {"__status__": 401,
                               "error": {"message": "no key for you"}})
    run_dir = a_run(tmp_path, p.url, rows, taskset_file)
    c = score_run(run_dir)["coverage"]

    assert c["n_api_error"] == len(rows)
    assert c["n_api_timeout"] == 0
    assert c["api_errors_by_kind"] == {"http_401": len(rows)}
    assert c["n_requests_timed_out"] == 0


def test_a_content_filter_keeps_its_name_even_when_a_refusal_comes_with_it(
        tmp_path, rows, taskset_file, provider):
    """The pair is still a content filter, and `refusal` throws away which.

    `refusal` is the label only when the stop reason says nothing on its own.
    Preferring it whenever the field is populated collapses the provider's own
    reason into a bucket -- the exact loss the split exists to prevent.
    """
    def filtered(body):
        out = completion("", finish_reason="content_filter")
        out["choices"][0]["message"]["refusal"] = "I can't help with that."
        return out

    p = provider(filtered)
    run_dir = a_run(tmp_path, p.url, rows[:2], taskset_file)
    c = score_run(run_dir)["coverage"]
    assert c["api_errors_by_kind"] == {"finish_reason_content_filter": 2}


def test_a_refusal_with_nothing_else_to_call_it_is_called_a_refusal(
        tmp_path, rows, taskset_file, provider):
    """`stop` is an answer, so it cannot be the label for a row that refused."""
    def refused(body):
        out = completion("", finish_reason="stop")
        out["choices"][0]["message"]["refusal"] = "I can't help with that."
        return out

    p = provider(refused)
    run_dir = a_run(tmp_path, p.url, rows[:2], taskset_file)
    c = score_run(run_dir)["coverage"]
    assert c["api_errors_by_kind"] == {"refusal": 2}


def test_a_malformed_body_is_not_filed_under_the_status_it_arrived_with():
    """`APIResponseValidationError` carries a `status_code` and is not an HTTP
    failure. Probing the attribute filed a 200 whose JSON did not parse as
    `http_200`; the fix asks what raised, not what it happens to carry."""
    import httpx2
    from openai import APIResponseValidationError

    exc = APIResponseValidationError(
        response=httpx2.Response(200, request=httpx2.Request("POST", "http://x")),
        body=None)
    assert exc.status_code == 200
    assert kind_of(exc) == MALFORMED


def test_the_cause_is_recorded_where_it_happens_and_not_read_back_out_of_prose(
        tmp_path, rows, taskset_file, provider):
    """The label is on the generation, so scoring never parses an error message.

    A taxonomy reconstructed from the text is a taxonomy of how a provider words
    its errors, and every provider words them differently.
    """
    p = provider(lambda body: {"__status__": 429, "error": {"message": "slow down"}})
    run_dir = a_run(tmp_path, p.url, rows[:1], taskset_file)
    gens = list(artifacts.read_jsonl(os.path.join(run_dir, artifacts.GENERATIONS)))
    assert [g["error_kind"] for g in gens] == ["http_429"]

    score_run(run_dir)
    samples = list(artifacts.read_jsonl(
        os.path.join(run_dir, artifacts.SAMPLES)))
    assert [s["api_error_kind"] for s in samples] == ["http_429"]


def test_a_retry_is_still_spent_only_where_it_could_help(rows, provider):
    """Labelling the cause must not change which causes get retried."""
    seen = {"n": 0}

    def counts(body):
        seen["n"] += 1
        return {"__status__": 400, "error": {"message": "bad request"}}

    p = provider(counts)
    solver = ChatSolver(Endpoint(model="stub", base_url=p.url, timeout_s=10,
                                 retries=3))
    attempt = solver(rows[0])
    assert attempt.error_kind == "http_400"
    assert seen["n"] == 1, f"retried a 400 {seen['n']} times"


# --- the vocabulary, as a set ------------------------------------------------


def _produces(kind, rows, provider, tmp_path, taskset_file):
    """One generation carrying `kind`, produced the way production produces it."""
    if kind == "timeout":
        p = provider(lambda body: (time.sleep(1.5),
                                   completion("<answer>x</answer>"))[1])
        return a_run(tmp_path, p.url, rows[:1], taskset_file, timeout_s=0.25)
    if kind == "connection":
        # Nothing listening. A real `APIConnectionError` off a real socket.
        return a_run(tmp_path, "http://127.0.0.1:9/v1", rows[:1], taskset_file,
                     timeout_s=2)
    if kind == "malformed_response":
        p = provider(lambda body: {"id": "stub", "object": "chat.completion",
                                   "created": 0, "model": "stub", "choices": []})
        return a_run(tmp_path, p.url, rows[:1], taskset_file)
    if kind == "refusal":
        def refused(body):
            out = completion("", finish_reason="stop")
            out["choices"][0]["message"]["refusal"] = "no"
            return out

        p = provider(refused)
        return a_run(tmp_path, p.url, rows[:1], taskset_file)
    if kind == "solver_raised":
        def explodes(row):
            raise RuntimeError("solver is broken")

        run_dir = os.fspath(tmp_path / "run")
        os.makedirs(run_dir)
        generate(rows[:1], explodes, run_dir, concurrency=1, progress=False)
        return run_dir
    if kind == "solver_value_not_json":
        def unserialisable(row):
            return Attempt(value=object())

        run_dir = os.fspath(tmp_path / "run")
        os.makedirs(run_dir)
        generate(rows[:1], unserialisable, run_dir, concurrency=1, progress=False)
        return run_dir
    raise AssertionError(f"no case for {kind}")


#: `caller_error` has no socket that produces it -- it is a bug in our own call
#: path, a `TypeError` from a bad sampling value or an `ImportError` from a
#: missing dependency -- so it is checked against `kind_of` directly rather than
#: faked into a run.
NOT_FROM_A_RUN = {"caller_error": TypeError("temperature must be a number")}


@pytest.mark.parametrize("kind", sorted(CLOSED_ERROR_KINDS))
def test_every_label_in_the_vocabulary_is_produced_by_something(
        kind, tmp_path, rows, taskset_file, provider):
    """A label nothing produces is a column that silently never appears.

    Parametrised over `CLOSED_ERROR_KINDS` rather than over a list written here,
    so a label added to the vocabulary without a way to produce it fails this
    test instead of shipping as an empty bucket.
    """
    if kind in NOT_FROM_A_RUN:
        assert kind_of(NOT_FROM_A_RUN[kind]) == kind
        return
    run_dir = _produces(kind, rows, provider, tmp_path, taskset_file)
    gens = list(artifacts.read_jsonl(os.path.join(run_dir, artifacts.GENERATIONS)))
    assert [g["error_kind"] for g in gens] == [kind], gens


# --- older runs --------------------------------------------------------------


def test_a_generation_made_before_the_cause_was_recorded_is_not_guessed_at(
        tmp_path, rows, taskset_file, provider):
    """Every run in `outputs/runs/` predates this. None of them may crash it.

    `unclassified` is the honest answer in the split, and the timeout count is
    unknown rather than zero: an error whose cause nobody recorded is not
    evidence that the cause was not a timeout.
    """
    p = provider(lambda body: {"__status__": 500, "error": {"message": "boom"}})
    run_dir = a_run(tmp_path, p.url, rows[:2], taskset_file)
    path = os.path.join(run_dir, artifacts.GENERATIONS)
    old = []
    for g in artifacts.read_jsonl(path):
        # Exactly what a generation written before this change looks like.
        for gone in ("error_kind", "attempt_latency_s", "requests_timed_out"):
            g.pop(gone)
        g["request"].pop("timeout_s")
        g["latency_s"] = 3.5
        old.append(g)
    artifacts.write_jsonl(path, old)

    c = score_run(run_dir)["coverage"]
    assert c["api_errors_by_kind"] == {"unclassified": 2}
    # Not zero. Nobody counted, and "0 of 2 errors were timeouts" is a claim.
    assert c["n_api_timeout"] is None
    assert c["n_requests_timed_out"] is None
    # The deadline falls back to the manifest, which older runs do record.
    assert c["timeout_s"] == 10


def test_one_unclassified_error_makes_the_timeout_count_unknown():
    """A count that is right for some of the errors is not right.

    Reporting the classified ones only would print a lower bound with nothing
    saying so -- the same lie as rendering an absent count as zero, one layer
    down where `_int` cannot see it.
    """
    c = coverage([a_sample(score=None, api_error="timed out",
                           api_error_kind="timeout"),
                  a_sample(score=None, api_error="who knows", api_error_kind=None)],
                 timeout_s=100.0)
    assert c["n_api_timeout"] is None
    assert c["api_errors_by_kind"] == {"timeout": 1, "unclassified": 1}


def test_an_unmeasured_generation_makes_the_wall_hit_count_unknown_too():
    """A successful generation nobody counted is still a generation nobody
    counted.

    The guard is over every record, not only the errored ones. Scoping it to
    errors would read a stale successful generation -- one written before the
    counter existed, sitting in a directory beside newer ones after a resume --
    as having hit the wall zero times, which is a measurement it never made.
    The dash this produces can therefore mean two things, and
    `docs/evaluation.md` says so.
    """
    c = coverage([a_sample(requests_timed_out=None),
                  a_sample(requests_timed_out=0)], timeout_s=100.0)
    assert c["n_requests_timed_out"] is None
    # And nothing else is withheld: the latencies were measured and are given.
    assert c["n_api_error"] == 0


def test_the_terminal_says_not_recorded_rather_than_zero(
        tmp_path, rows, taskset_file, provider, capsys):
    """The line whose own comment says nobody opens `metrics.json` to check a
    thing they have no reason to suspect yet.

    `_count` and `_duration` exist for exactly one purpose -- keeping "nobody
    measured this" apart from "this was zero" -- and on the terminal that
    distinction is the whole message. A run with no timeout recorded and no
    wall-hit counter is every run currently on disk.
    """
    from evals.score import main

    p = provider(answering(rows))
    run_dir = a_run(tmp_path, p.url, rows[:2], taskset_file)
    path = os.path.join(run_dir, artifacts.GENERATIONS)
    old = []
    for g in artifacts.read_jsonl(path):
        g.pop("requests_timed_out")
        g["request"].pop("timeout_s")
        old.append(g)
    artifacts.write_jsonl(path, old)
    run_path = os.path.join(run_dir, artifacts.RUN)
    meta = json.load(open(run_path))
    meta["endpoint"].pop("timeout_s")
    artifacts.write_json(run_path, meta)

    assert main([run_dir]) == 0
    out = capsys.readouterr().out
    assert "requests that hit the timeout: not recorded" in out, out
    assert "timeout unknown" in out, out
    # The two numbers that would be wrong rather than absent.
    assert "requests that hit the timeout: 0" not in out
    assert "timeout 0s" not in out


def test_an_older_generations_latency_falls_back_to_the_whole_row(
        tmp_path, rows, taskset_file, provider):
    """The two differ only where a row was retried, and then the old number is
    the larger one -- it overstates the risk, which is the safe direction for a
    warning."""
    p = provider(answering(rows))
    run_dir = a_run(tmp_path, p.url, rows[:2], taskset_file, timeout_s=100)
    path = os.path.join(run_dir, artifacts.GENERATIONS)
    old = []
    for g in artifacts.read_jsonl(path):
        g.pop("attempt_latency_s")
        g["latency_s"] = 90.0
        old.append(g)
    artifacts.write_jsonl(path, old)

    c = score_run(run_dir)["coverage"]
    assert c["n_latency_measured"] == 2
    assert c["latency_s_max"] == 90.0
    assert c["n_near_timeout"] == 2


def test_a_metrics_file_without_any_of_these_fields_still_reports(tmp_path):
    """`outputs/runs/*/metrics.json` on disk has none of them."""
    from test_the_report_says_what_can_be_compared import a_metrics, written

    m = a_metrics("old-run")
    assert "latency_s_p95" not in m["coverage"]
    loaded = written(tmp_path, m)
    text = render(loaded)

    assert "## Latency headroom" in text
    rows_out = [ln for ln in text.splitlines() if ln.startswith("| old-run |")]
    # Every latency cell is a dash, not a zero. `0s` and `0` are measurements,
    # and this run has none: it would report an endpoint that never came near
    # its timeout, which is a claim nobody made.
    assert _cells(rows_out[-1])[1:] == ["-"] * 6, rows_out[-1]
    # Same for both timeout counts in the coverage table, where a `0` would sit
    # beside a real API-error total and read as part of it.
    assert _cells(rows_out[0])[4:6] == ["-", "-"], rows_out[0]
    # And an absent split is not an empty one, so the section is left out
    # entirely rather than printed as a table of zeros.
    assert "### API errors, by cause" not in text
    write_csv(loaded, os.fspath(tmp_path / "r" / "results.csv"))


# --- the report --------------------------------------------------------------


def test_the_report_prints_the_headroom_and_the_split(tmp_path, rows, taskset_file,
                                                      provider):
    """Numbers that stop at `metrics.json` are numbers nobody reads."""
    p = provider(lambda body: {"__status__": 401, "error": {"message": "nope"}})
    run_dir = a_run(tmp_path, p.url, rows[:2], taskset_file, timeout_s=1800)
    score_run(run_dir)
    with open(os.path.join(run_dir, "metrics.json")) as f:
        m = json.load(f)
    m["_label"] = "stub"

    text = render([m])
    assert "## Latency headroom" in text
    assert "1,800s" in text
    assert "### API errors, by cause" in text
    assert "| http_401 | 2 |" in text
    # The coverage table keeps the total and gains both subsets, so a reader who
    # only ever looks at the first table still sees the timeout numbers.
    header = [ln for ln in text.splitlines() if ln.startswith("| run |")][0]
    assert "rows lost to timeout" in header
    assert "requests that hit the timeout" in header


def test_the_near_wall_header_cannot_drift_from_the_number_it_names(tmp_path):
    """A header that says 80% while the scorer uses 90% is a wrong label on a
    right number, and nothing in the file would disagree with it."""
    from test_the_report_says_what_can_be_compared import a_metrics, written

    from evals.score import NEAR_TIMEOUT_FRACTION

    text = render(written(tmp_path, a_metrics("model-a")))
    assert f"at {NEAR_TIMEOUT_FRACTION:.0%} of timeout" in text


def test_the_timeout_counts_reach_the_csv(tmp_path, rows, taskset_file, provider):
    """The artifact people load into a dataframe, where the per-level shape is.

    The August rate rose from 9.8% to 21.8% between levels 3 and 9, and that is
    only visible sliced by level -- which the CSV is the only place to do.
    """
    import csv as _csv

    p = provider(lambda body: {"__status__": 401, "error": {"message": "nope"}})
    run_dir = a_run(tmp_path, p.url, rows[:2], taskset_file)
    score_run(run_dir)
    with open(os.path.join(run_dir, "metrics.json")) as f:
        m = json.load(f)
    m["_label"] = "stub"

    out = os.fspath(tmp_path / "report" / "results.csv")
    write_csv([m], out)
    with open(out) as f:
        got = list(_csv.DictReader(f))
    assert "n_api_timeout" in got[0]
    assert "n_requests_timed_out" in got[0]
    assert {r["n_api_timeout"] for r in got} == {"0"}
    assert {r["n_requests_timed_out"] for r in got} == {"0"}
    assert {r["n_api_error"] for r in got} == {"1"}
