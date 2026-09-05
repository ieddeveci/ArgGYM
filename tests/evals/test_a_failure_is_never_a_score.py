"""An API error, a truncation and a wrong answer are three different events.

Folding any of them into the others reports infrastructure trouble as a
reasoning result, which on the previous sweep is exactly what happened: three of
seven models lost most of their items to the token cap and their means were read
as reasoning.
"""
from __future__ import annotations

import os

from conftest import answering, completion
from test_a_reference_answer_scores_one_end_to_end import a_run

from evals import artifacts
from evals.client import Endpoint
from evals.run import generate
from evals.score import score_one, score_run
from evals.solver import ChatSolver


def test_an_api_error_is_recorded_as_an_error_and_not_as_zero(tmp_path, rows,
                                                              taskset_file, provider):
    p = provider(lambda body: {"__status__": 500, "error": {"message": "boom"}})
    run_dir = a_run(tmp_path, p.url, rows, taskset_file)
    metrics = score_run(run_dir)

    assert metrics["coverage"]["n_api_error"] == len(rows)
    assert metrics["coverage"]["n_scored"] == 0
    # Coverage says what was not measured and carries no score of any kind. A
    # pooled figure over every record would be a cross-task mean, which is the
    # one number `aggregate` exists to refuse.
    assert not ({"mean", "mean_untruncated", "success_rate", "corrected", "floor"}
                & set(metrics["coverage"])), metrics["coverage"]
    for v in metrics["by_task"].values():
        assert v["mean"] is None and v["success_rate"] is None

    samples = list(artifacts.read_jsonl(os.path.join(run_dir, artifacts.SAMPLES)))
    assert all(s["score"] is None and s["reason"] == "api_error" for s in samples)


def test_a_truncated_generation_is_flagged_and_kept_out_of_the_censored_mean(
        tmp_path, rows, taskset_file, provider):
    p = provider(answering(rows, finish_reason="length"))
    run_dir = a_run(tmp_path, p.url, rows, taskset_file)
    metrics = score_run(run_dir)

    assert metrics["coverage"]["truncated_rate"] == 1.0
    for v in metrics["by_task"].values():
        # The answer is still there and still correct, so the raw mean is 1.0.
        assert v["mean"] == 1.0
        # Every generation was cut off, so there is no untruncated mean at all
        # -- and saying "none" is the honest report, not carrying the raw one
        # over into the column that claims to exclude them.
        assert v["mean_untruncated"] is None


def test_a_completion_with_no_answer_region_is_told_apart_from_a_wrong_one(
        tmp_path, rows, taskset_file, provider):
    p = provider(lambda body: completion("I would rather not say."))
    run_dir = a_run(tmp_path, p.url, rows, taskset_file)
    metrics = score_run(run_dir)

    assert metrics["coverage"]["no_answer_region_rate"] == 1.0
    # It is still scored -- the text is handed to the scorer whole -- but the
    # flag is what separates a formatting failure from a reasoning failure.
    assert metrics["coverage"]["n_scored"] == len(rows)


def test_a_scorer_that_refuses_a_row_is_flagged_and_never_averaged(rows):
    """The previous harness caught this with a broad except and published 0.000."""
    broken = dict(rows[0])
    broken["metadata"] = dict(broken["metadata"], pyarg_version="0.0.0-not-installed")
    record = score_one({"id": broken["id"], "task": broken["task"], "level": 3,
                        "ordering": "last_link_elitist", "completion": "<answer>x</answer>"},
                       broken, "xml_tags")
    assert record["scorer_refused"] is True
    assert record["score"] is None
    assert record["reason"] != "api_error"


def test_a_solver_that_raises_costs_its_row_and_not_the_run(tmp_path, rows,
                                                            taskset_file):
    def explodes(row):
        raise RuntimeError("solver is broken")

    run_dir = os.fspath(tmp_path / "run")
    os.makedirs(run_dir)
    counts = generate(rows, explodes, run_dir, concurrency=2, progress=False)
    assert counts["generated"] == len(rows)
    assert counts["errors"] == len(rows)
    gens = list(artifacts.read_jsonl(os.path.join(run_dir, artifacts.GENERATIONS)))
    assert all("solver raised: RuntimeError" in g["error"] for g in gens)


def test_a_retry_is_spent_only_where_it_could_help(rows, provider):
    """A 400 will not succeed on the third attempt; a 503 might."""
    seen = {"n": 0}

    def flaky(body):
        seen["n"] += 1
        return {"__status__": 400, "error": {"message": "bad request"}}

    p = provider(flaky)
    solver = ChatSolver(Endpoint(model="stub", base_url=p.url, timeout_s=10, retries=3))
    attempt = solver(rows[0])
    assert attempt.error is not None
    assert seen["n"] == 1, f"retried a 400 {seen['n']} times"


def test_a_floor_that_could_not_be_measured_is_reported_and_not_absorbed(
        tmp_path, rows, taskset_file, provider, monkeypatch):
    """`-` in the floor column must not mean two different things.

    `arggym.floors` scores every row with each constant strategy and does not
    guard, so it was called inside a bare `except` -- and one ungradeable row
    then removed a task's floor and its `corrected` column silently, rendering
    identically to a task that has no floor at all.
    """
    import arggym

    p = provider(answering(rows))
    run_dir = a_run(tmp_path, p.url, rows, taskset_file)

    def refuses(_rows):
        raise RuntimeError("engine version mismatch")

    monkeypatch.setattr(arggym, "floors", refuses)
    metrics = score_run(run_dir)

    assert metrics["_meta"]["floors_unmeasured"] == [
        "RuntimeError: engine version mismatch"]
    for v in metrics["by_task"].values():
        assert "floor" not in v and "corrected" not in v
        assert v["floor_error"] == "RuntimeError: engine version mismatch"
        # The scores themselves are unaffected: a missing floor removes the
        # correction, never the measurement.
        assert v["mean"] == 1.0
