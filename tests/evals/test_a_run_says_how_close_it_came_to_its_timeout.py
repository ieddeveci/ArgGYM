"""A timeout is a silent loss, so the margin has to be reported before it is gone.

`timeout_s` was raised from 1800 to 5400 on the strength of one sweep where 1800
expired on 9.8%, 14.8% and 21.8% of one model's requests at levels 3, 6 and 9 --
and nothing then measured whether the new number was enough. The failure it
guards against is invisible while it is approaching: a run at 95% of its budget
and a run on a healthy endpoint produce identical artifacts right up to the day
the first one starts losing items, and each lost item is a whole completion the
server generates again on the retry.

Two things make it visible, and both are tested here. The latency the timeout
actually bounds reaches `metrics.json` with the timeout beside it, and
`n_api_error` is split by cause so the one failure mode that moves with the
token cap can be counted on its own.
"""
from __future__ import annotations

import json
import os
import time

from conftest import answering, completion
from test_a_reference_answer_scores_one_end_to_end import a_run

from evals import artifacts
from evals.client import Endpoint
from evals.report import render, write_csv
from evals.score import coverage, score_run
from evals.solver import ChatSolver


def _cells(line):
    """One markdown row's cells, label first."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def a_sample(**kw):
    """One `samples.jsonl` record, in the shape `coverage` reads."""
    out = {"id": "x", "task": "attack", "level": 3, "ordering": "last_link_elitist",
           "api_error": None, "api_error_kind": None, "truncated": False,
           "latency_s": None, "attempt_latency_s": None, "attempts": 1,
           "scorer_refused": False, "score": 1.0, "success": True,
           "no_answer_region": False, "answer_in_cot": False}
    out.update(kw)
    return out


# --- the headroom itself -----------------------------------------------------


def test_the_timeout_a_run_used_reaches_the_file_its_latency_is_judged_by(
        tmp_path, rows, taskset_file, provider):
    """2,295s is comfortable under 5,400 and lost under 1,800.

    The percentile alone says neither, so the timeout it is measured against
    travels with it. It is only knowable from the endpoint the run recorded --
    `score.py` never builds an `Endpoint` and never sees the config the run was
    launched with.
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
    assert c["latency_s_max"] is not None


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


def test_the_client_times_each_attempt_and_not_only_the_whole_item(
        rows, provider, monkeypatch):
    """Both numbers, because they answer different questions.

    `latency_s` is what one row cost, retries and backoff included.
    `attempt_latency_s` is what the timeout is compared against. The moment a
    generation retries the two diverge, and only the second one can say whether
    `timeout_s` is big enough.
    """
    import evals.client as client_module

    # The retry backoff is five seconds and it is not what this measures. The
    # handler keeps a reference to the real one, because `evals.client.time` is
    # the `time` module itself -- patching its `sleep` patches everyone's, and
    # a stub that stopped sleeping would make this test measure nothing.
    real_sleep = time.sleep
    monkeypatch.setattr(client_module.time, "sleep", lambda _s: None)
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
    assert attempt.latency_s > 0.6
    # The second request did no sleeping at all, so the gap is the whole of the
    # first attempt. A wide margin, because this is a clock on a loaded box.
    assert attempt.attempt_latency_s < 0.4, (attempt.attempt_latency_s,
                                             attempt.latency_s)


def test_headroom_is_measured_per_request_and_not_per_item():
    """`timeout_s` bounds one request, so one request is what is compared to it.

    A generation that failed once and succeeded on the retry costs more than the
    timeout in wall time while no single request came near it. Measured from the
    item, that run reads as out of budget when it has most of its budget left.
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
                        attempt_latency_s=5400.0),
               a_sample(score=None, api_error="unauthorized",
                        api_error_kind="http_401", attempt_latency_s=0.01),
               a_sample(attempt_latency_s=12.0)]
    c = coverage(records, timeout_s=5400.0)

    assert c["n_latency_measured"] == 1
    assert c["latency_s_max"] == 12.0
    assert c["n_near_timeout"] == 0
    assert c["n_api_error"] == 2
    assert c["n_api_timeout"] == 1


def test_a_run_whose_manifest_names_no_timeout_still_reports_its_latency():
    """No timeout recorded means no distance to the wall, and `0` is not that."""
    c = coverage([a_sample(attempt_latency_s=12.0)], timeout_s=None)
    assert c["latency_s_max"] == 12.0
    assert c["timeout_s"] is None
    assert c["near_timeout_s"] is None
    assert c["n_near_timeout"] is None


# --- the split by cause ------------------------------------------------------


def test_a_timeout_is_counted_apart_from_the_other_api_errors(tmp_path, rows,
                                                              taskset_file, provider):
    """Through a real socket, so the label is put on what the SDK actually raises."""
    def slow(body):
        time.sleep(2.0)
        return completion("<answer>x</answer>")

    p = provider(slow)
    run_dir = a_run(tmp_path, p.url, rows[:1], taskset_file, timeout_s=0.25)
    c = score_run(run_dir)["coverage"]

    assert c["n_api_error"] == 1
    assert c["n_api_timeout"] == 1
    assert c["api_errors_by_kind"] == {"timeout": 1}
    # Nothing answered, so there is no latency to report -- and the count above
    # is the loud thing, not a silently empty percentile.
    assert c["n_latency_measured"] == 0
    assert c["latency_s_p95"] is None


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


def test_a_provider_that_refused_is_not_an_endpoint_that_was_unreachable(
        tmp_path, rows, taskset_file, provider):
    """`content_filter` is the provider's policy; a 500 is its health."""
    p = provider(lambda body: completion("", finish_reason="content_filter"))
    run_dir = a_run(tmp_path, p.url, rows, taskset_file)
    c = score_run(run_dir)["coverage"]

    assert c["api_errors_by_kind"] == {"finish_reason_content_filter": len(rows)}
    assert c["n_api_timeout"] == 0


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

    samples_of = score_run(run_dir) and list(artifacts.read_jsonl(
        os.path.join(run_dir, artifacts.SAMPLES)))
    assert [s["api_error_kind"] for s in samples_of] == ["http_429"]


def test_a_solver_that_raises_is_not_filed_as_a_provider_failure(rows):
    """Our own bug and the endpoint's are different findings."""
    from evals.run import generate

    def explodes(row):
        raise RuntimeError("solver is broken")

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        generate(rows[:1], explodes, d, concurrency=1, progress=False)
        gens = list(artifacts.read_jsonl(os.path.join(d, artifacts.GENERATIONS)))
    assert [g["error_kind"] for g in gens] == ["solver_raised"]


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


# --- older runs --------------------------------------------------------------


def test_a_generation_made_before_the_cause_was_recorded_is_not_guessed_at(
        tmp_path, rows, taskset_file, provider):
    """Every run in `outputs/runs/` predates this. None of them may crash it.

    `unclassified` is the honest answer. Deducing the cause from the message
    would put a number in the timeout column that was never measured.
    """
    p = provider(lambda body: {"__status__": 500, "error": {"message": "boom"}})
    run_dir = a_run(tmp_path, p.url, rows[:2], taskset_file)
    path = os.path.join(run_dir, artifacts.GENERATIONS)
    old = []
    for g in artifacts.read_jsonl(path):
        # Exactly what a generation written before this change looks like.
        g.pop("error_kind")
        g.pop("attempt_latency_s")
        g["latency_s"] = 3.5
        old.append(g)
    artifacts.write_jsonl(path, old)

    c = score_run(run_dir)["coverage"]
    assert c["api_errors_by_kind"] == {"unclassified": 2}
    assert c["n_api_timeout"] == 0


def test_an_older_generations_latency_falls_back_to_the_whole_item(
        tmp_path, rows, taskset_file, provider):
    """The two differ only where a generation was retried, and then the old
    number is the larger one -- it overstates the risk, which is the safe
    direction for a warning."""
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
    # Same for the timeout count in the coverage table, where a `0` would sit
    # beside a real API-error total and read as part of it.
    assert _cells(rows_out[0])[4] == "-", rows_out[0]
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
    # The coverage table keeps the total and gains the subset, so a reader who
    # only ever looks at the first table still sees a timeout count.
    header = [ln for ln in text.splitlines() if ln.startswith("| run |")][0]
    assert "timeouts" in header


def test_the_timeout_count_reaches_the_csv(tmp_path, rows, taskset_file, provider):
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
    assert {r["n_api_timeout"] for r in got} == {"0"}
    assert {r["n_api_error"] for r in got} == {"1"}
