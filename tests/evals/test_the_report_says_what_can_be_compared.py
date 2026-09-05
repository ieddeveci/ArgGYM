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

import pytest

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
                      "template": "xml_tags", "elicitation": "none",
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
    #
    # Labels are stripped of markdown emphasis first. Matching the raw cell let
    # `| **overall** |` through, and a summary row someone adds later is far
    # more likely to be bolded than not.
    labels = {_plain(line.split("|")[1]) for line in text.splitlines()
              if line.startswith("| ")}
    assert not (labels & {"overall", "average", "mean", "macro-avg", "macro", "all",
                          "total", "aggregate", "summary"}), labels
    assert "macro" not in text.lower()
    assert "No overall mean is given." in text


def _plain(cell: str) -> str:
    return cell.strip().strip("*_` ").lower()


def test_the_csv_gives_no_overall_mean_either(tmp_path):
    """The CSV is what a person loads into a dataframe, and nothing checked it."""
    m = a_metrics("model-a")
    out = os.fspath(tmp_path / "report")
    write_csv(written(tmp_path, m), os.path.join(out, "results.csv"))
    import csv as _csv

    with open(os.path.join(out, "results.csv")) as f:
        tasks = {_plain(r["task"]) for r in _csv.DictReader(f)}
    assert not (tasks & {"overall", "average", "mean", "macro-avg", "macro", "all",
                         "total", "aggregate", "summary"}), tasks


def test_the_csv_carries_every_stat_the_scorer_produces(tmp_path):
    """A stat dropped on the way to the CSV is a flag nobody sees."""
    m = a_metrics("model-a")
    out = os.fspath(tmp_path / "report")
    write_csv(written(tmp_path, m), os.path.join(out, "results.csv"))
    header = open(os.path.join(out, "results.csv")).readline()
    for field in ("answer_in_cot_rate", "n_scorer_refused", "truncated_rate",
                  "no_answer_region_rate", "zero_with_region", "n_scored"):
        assert field in header, field


def test_a_run_that_was_never_scored_is_not_silently_dropped(tmp_path):
    from evals.report import NotScored

    scored = written(tmp_path, a_metrics("model-a"))
    unscored = tmp_path / "unscored"
    unscored.mkdir()
    with pytest.raises(NotScored) as e:
        load_metrics([os.path.dirname(scored[0]["_meta"]["run_dir"]) and
                      os.fspath(tmp_path / "run0"), os.fspath(unscored)])
    assert "unscored" in str(e.value)


def test_two_runs_of_one_model_get_different_columns(tmp_path):
    """Elicitation exists so two runs are an experiment, not a collision."""
    a = a_metrics("gpt-5")
    b = a_metrics("gpt-5")
    b["_meta"]["elicitation"] = "cot"
    labels = [m["_label"] for m in written(tmp_path, a, b)]
    assert len(set(labels)) == 2, labels
    assert any("cot" in x for x in labels)


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
