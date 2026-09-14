"""The whole pipeline, from a prompt to a scored table.

A provider that replies with each row's own reference answer must score 1.0 on
every task. If it does not, then something between composing the prompt and
reading the score is losing the answer, and this is the test that says so before
a sweep spends money finding out.
"""
from __future__ import annotations

import json
import os

from conftest import answering

from evals import artifacts, taskset
from evals.client import Endpoint
from evals.run import generate
from evals.score import score_run
from evals.solver import ChatSolver


def a_run(tmp_path, url, rows, taskset_file, template="xml_tags", timeout_s=10):
    """One run on disk, manifest included.

    `timeout_s` is a parameter because it is a number the scorer reads back out
    of the manifest: the latency headroom in `metrics.json` is measured against
    it, so a test of that has to be able to set it.
    """
    run_dir = os.fspath(tmp_path / "run")
    os.makedirs(run_dir)
    manifest, _ = taskset.load(taskset_file)
    solver = ChatSolver(
        Endpoint(model="stub", base_url=url, timeout_s=timeout_s, retries=0),
        template=template)
    artifacts.write_json(os.path.join(run_dir, artifacts.RUN), {
        "status": "completed", "taskset": taskset_file,
        "taskset_hash": manifest["taskset_hash"], "template": template,
        "endpoint": solver.client.endpoint.redacted(), "n_selected": len(rows)})
    generate(rows, solver, run_dir, concurrency=2, progress=False)
    return run_dir


def test_a_reference_answer_scores_one_on_every_task(tmp_path, rows, taskset_file,
                                                     provider):
    p = provider(answering(rows))
    run_dir = a_run(tmp_path, p.url, rows, taskset_file)
    metrics = score_run(run_dir)

    assert metrics["coverage"]["n_scored"] == len(rows)
    assert metrics["coverage"]["n_api_error"] == 0
    for task, v in metrics["by_task"].items():
        assert v["mean"] == 1.0, (task, v)
        assert v["success_rate"] == 1.0, (task, v)
        # A floor is published beside every score, because a score without one
        # cannot be read (`docs/dataset-contract.md` section 10).
        assert v["floor"] is not None, task


def test_the_prompt_is_written_before_the_answer_is_known(tmp_path, rows,
                                                          taskset_file, provider):
    """A run that dies mid-flight must still say what it asked."""
    p = provider(answering(rows))
    run_dir = a_run(tmp_path, p.url, rows, taskset_file)
    # `generate` does not write prompts -- `run.py` does, before calling it --
    # so here we check the weaker thing the artifacts guarantee: every
    # generation carries the request that produced it.
    gens = list(artifacts.read_jsonl(os.path.join(run_dir, artifacts.GENERATIONS)))
    assert len(gens) == len(rows)
    for g in gens:
        assert g["request"]["model"] == "stub"
        assert g["request"]["n_messages"] >= 1


def test_scoring_never_reaches_the_provider(tmp_path, rows, taskset_file, provider):
    """Scoring is offline, so a stopped provider changes nothing."""
    p = provider(answering(rows))
    run_dir = a_run(tmp_path, p.url, rows, taskset_file)
    first = score_run(run_dir)
    p.stop()
    again = score_run(run_dir)
    assert first["by_task"] == again["by_task"]


def test_a_run_scored_against_another_taskset_is_refused(tmp_path, rows,
                                                         taskset_file, provider):
    """Answers scored against questions they were not asked is a silent wrong number."""
    from evals.score import TasksetMismatch

    p = provider(answering(rows))
    run_dir = a_run(tmp_path, p.url, rows, taskset_file)
    run_path = os.path.join(run_dir, artifacts.RUN)
    meta = json.load(open(run_path))
    meta["taskset_hash"] = "0" * 32
    artifacts.write_json(run_path, meta)

    try:
        score_run(run_dir)
    except TasksetMismatch as e:
        assert "taskset_hash" in str(e)
    else:
        raise AssertionError("scored a run against a taskset it did not use")
