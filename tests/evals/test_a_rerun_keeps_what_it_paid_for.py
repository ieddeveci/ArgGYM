"""Generations cost hours. A rerun must not buy them again.

A machine reboot on the previous harness nearly cost 439 completed generations,
and a whole finished sweep was lost to a cleanup job because nothing durable was
written until the end.
"""
from __future__ import annotations

import os

from conftest import answering, completion

from evals import artifacts
from evals.client import Endpoint
from evals.run import generate
from evals.solver import ChatSolver


def test_a_rerun_skips_what_already_succeeded(tmp_path, rows, provider):
    p = provider(answering(rows))
    run_dir = os.fspath(tmp_path / "run")
    os.makedirs(run_dir)
    solver = ChatSolver(Endpoint(model="stub", base_url=p.url, retries=0))

    generate(rows[:1], solver, run_dir, concurrency=1, progress=False)
    done = artifacts.completed_ids(run_dir)
    assert done == {rows[0]["id"]}

    todo = [r for r in rows if r["id"] not in done]
    before = len(p.requests)
    generate(todo, solver, run_dir, concurrency=1, progress=False)
    assert len(p.requests) - before == len(rows) - 1, "regenerated a finished row"


def test_an_errored_row_is_not_treated_as_finished(tmp_path, rows, provider):
    p = provider(lambda body: {"__status__": 503, "error": {"message": "down"}})
    run_dir = os.fspath(tmp_path / "run")
    os.makedirs(run_dir)
    generate(rows, ChatSolver(Endpoint(model="stub", base_url=p.url, retries=0)),
             run_dir, concurrency=1, progress=False)
    assert artifacts.completed_ids(run_dir) == set()


def test_a_retry_supersedes_the_failure_it_replaces(tmp_path, rows):
    """Both lines stay on disk; the good one is the one that counts."""
    run_dir = os.fspath(tmp_path / "run")
    os.makedirs(run_dir)
    path = os.path.join(run_dir, artifacts.GENERATIONS)
    with artifacts.Appender(path) as a:
        a.write({"id": "x", "error": "down", "completion": ""})
        a.write({"id": "x", "error": None, "completion": "<answer>ok</answer>"})
    assert len(list(artifacts.read_jsonl(path))) == 2
    assert artifacts.best_per_id(run_dir)["x"]["completion"] == "<answer>ok</answer>"
    assert artifacts.completed_ids(run_dir) == {"x"}


def test_a_later_failure_does_not_discard_a_generation_already_paid_for(tmp_path):
    """The two readers must agree, or an item is lost while looking complete.

    Reading simply the last record made `completed_ids` say "done" -- so resume
    skipped the id forever -- while scoring took the trailing error and recorded
    `score: None`. The good generation sat one line up and was never scored.
    """
    run_dir = os.fspath(tmp_path / "run")
    os.makedirs(run_dir)
    with artifacts.Appender(os.path.join(run_dir, artifacts.GENERATIONS)) as a:
        a.write({"id": "x", "error": None, "completion": "<answer>ok</answer>"})
        a.write({"id": "x", "error": "503 later", "completion": ""})
    assert artifacts.completed_ids(run_dir) == {"x"}
    assert artifacts.best_per_id(run_dir)["x"]["error"] is None


def test_not_resuming_starts_the_directory_over(tmp_path):
    """`Appender` appends, so leaving the old file would be resume by accident."""
    run_dir = os.fspath(tmp_path / "run")
    os.makedirs(run_dir)
    with artifacts.Appender(os.path.join(run_dir, artifacts.GENERATIONS)) as a:
        a.write({"id": "x", "error": None, "completion": "old"})
    artifacts.restart(run_dir)
    assert artifacts.completed_ids(run_dir) == set()


def test_each_result_is_on_disk_before_the_next_call_returns(tmp_path, rows, provider):
    """Streaming, not a write at the end: a crash costs the item in flight only."""
    seen = []

    def watching(body):
        seen.append(len(list(artifacts.read_jsonl(
            os.path.join(run_dir, artifacts.GENERATIONS)))))
        return completion("<answer>x</answer>")

    p = provider(watching)
    run_dir = os.fspath(tmp_path / "run")
    os.makedirs(run_dir)
    generate(rows, ChatSolver(Endpoint(model="stub", base_url=p.url, retries=0)),
             run_dir, concurrency=1, progress=False)
    # The second request saw the first result already written.
    assert seen[-1] == len(rows) - 1, seen


def test_a_manifest_is_never_left_half_written(tmp_path):
    path = os.fspath(tmp_path / "run.json")
    artifacts.write_json(path, {"status": "running"})
    artifacts.write_json(path, {"status": "completed"})
    import json
    assert json.load(open(path))["status"] == "completed"
    assert not os.path.exists(path + ".tmp")
