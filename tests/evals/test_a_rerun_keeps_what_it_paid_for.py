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


def test_a_torn_last_line_costs_the_item_and_not_the_run(tmp_path):
    """A reboot mid-append leaves half a line, and that is the case this is for.

    `json.loads` raising there took down `best_per_id`, resume, scoring and the
    manifest's error rate together -- the whole run, to save the item in flight.
    """
    run_dir = os.fspath(tmp_path / "run")
    os.makedirs(run_dir)
    path = os.path.join(run_dir, artifacts.GENERATIONS)
    with artifacts.Appender(path) as a:
        a.write({"id": "x", "error": None, "completion": "<answer>ok</answer>"})
    with open(path, "a") as f:
        f.write('{"id": "y", "error": null, "compl')

    assert artifacts.completed_ids(run_dir) == {"x"}
    assert artifacts.best_per_id(run_dir)["x"]["completion"] == "<answer>ok</answer>"


def test_damage_anywhere_but_the_last_line_is_refused(tmp_path):
    """Only the end of an appended file can be a torn write.

    Skipping a broken line in the middle would drop a generation that was paid
    for and report the run as one item shorter, with nothing saying so.
    """
    import pytest

    run_dir = os.fspath(tmp_path / "run")
    os.makedirs(run_dir)
    path = os.path.join(run_dir, artifacts.GENERATIONS)
    with open(path, "w") as f:
        f.write('{"id": "x", "err\n')
        f.write('{"id": "y", "error": null, "completion": "ok"}\n')
    with pytest.raises(artifacts.Torn) as e:
        artifacts.completed_ids(run_dir)
    assert "line 1 of 2" in str(e.value)
