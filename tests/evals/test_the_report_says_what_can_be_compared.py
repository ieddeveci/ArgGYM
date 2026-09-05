"""Reporting rules from `docs/dataset-contract.md` section 10.

Chance floors beside the scores, because at level 3 `semantics_query` sits at
0.490 and a model scoring 0.45 there is doing worse than answering the same thing
every time. And no unweighted mean across tasks: twelve metrics of four kinds,
with floors spanning half the range, average into a number that moves mostly
with which tasks are in the basket. The previous harness published one anyway
and called it a ranking aid.
"""
from __future__ import annotations

import json
import os

from evals.report import load_metrics, render, write_csv


def a_metrics(label, taskset_hash="abc", versions=None, **overrides):
    by_task = {
        "status_query": {"n": 4, "n_scored": 4, "n_api_error": 0,
                         "n_scorer_refused": 0, "mean": 0.4, "mean_untruncated": 0.5,
                         "success_rate": 0.25, "floor": 0.375, "corrected": 0.04,
                         "truncated_rate": 0.25, "no_answer_region_rate": 0.0,
                         "answer_in_cot_rate": 0.0, "zero_with_region": 1},
    }
    return {"_meta": {"run_dir": f"/runs/{label}", "endpoint": {"model": label},
                      "taskset_hash": taskset_hash, "run_status": "completed",
                      "taskset_versions": versions or {"prompt_version": 3},
                      "n_scored": 4},
            "coverage": {"n": 4, "n_scored": 4, "n_api_error": 0,
                         "truncated_rate": 0.25, "no_answer_region_rate": 0.0,
                         "mean": 0.4},
            "by_task": by_task, "by_task_level": {}, "by_task_ordering": {},
            **overrides}


def written(tmp_path, *metrics):
    paths = []
    for i, m in enumerate(metrics):
        d = tmp_path / f"run{i}"
        d.mkdir()
        (d / "metrics.json").write_text(json.dumps(m))
        paths.append(os.fspath(d))
    return load_metrics(paths)


def test_the_report_gives_no_overall_mean(tmp_path):
    text = render(written(tmp_path, a_metrics("model-a"), a_metrics("model-b")))
    # The property, not the word: no row of any table aggregates across tasks.
    # (The prose says "no overall mean is given", so grepping for the phrase
    # would pass on a report that also printed one.)
    labels = {line.split("|")[1].strip().lower()
              for line in text.splitlines() if line.startswith("| ")}
    assert not (labels & {"overall", "average", "mean", "macro-avg", "macro", "all",
                          "total"}), labels
    assert "macro" not in text.lower()
    assert "No overall mean is given." in text


def test_every_score_column_carries_its_floor(tmp_path):
    text = render(written(tmp_path, a_metrics("model-a")))
    assert "| floor |" in text
    assert "0.375" in text
    assert "Chance-corrected" in text


def test_coverage_is_printed_before_any_score(tmp_path):
    """A run that lost its items to the token cap has not been measured."""
    text = render(written(tmp_path, a_metrics("model-a")))
    assert text.index("## Coverage") < text.index("## Mean score, by task")
    assert "25.0%" in text


def test_runs_on_different_tasksets_are_called_incomparable(tmp_path):
    text = render(written(tmp_path, a_metrics("a", taskset_hash="one"),
                          a_metrics("b", taskset_hash="two")))
    assert "different tasksets" in text
    assert "not comparable" in text


def test_runs_on_different_prompt_versions_are_called_out(tmp_path):
    text = render(written(tmp_path, a_metrics("a", versions={"prompt_version": 3}),
                          a_metrics("b", versions={"prompt_version": 4})))
    assert "different prompt or scoring versions" in text


def test_the_csv_carries_every_grouping(tmp_path):
    m = a_metrics("model-a")
    m["by_task_level"] = {"status_query|L3": dict(m["by_task"]["status_query"])}
    m["by_task_ordering"] = {"status_query|last_link_elitist":
                             dict(m["by_task"]["status_query"])}
    out = os.fspath(tmp_path / "report")
    write_csv(written(tmp_path, m), os.path.join(out, "results.csv"))
    text = open(os.path.join(out, "results.csv")).read()
    assert "L3" in text and "last_link_elitist" in text
    assert "floor" in text.splitlines()[0]
