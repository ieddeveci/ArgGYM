"""Reporting rules from `docs/dataset-contract.md` section 10.

Chance floors beside the scores, because at level 3 `semantics_query` sits at
0.806 and a model scoring 0.45 there is doing worse than answering the same thing
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
                         "answer_in_cot_rate": 0.0, "zero_with_region": 1,
                         "bloat_rate": 0.0},
    }
    return {"_meta": {"run_dir": f"/runs/{label}", "endpoint": {"model": label},
                      "template": "xml_tags", "elicitation": "cot",
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
                  "no_answer_region_rate", "zero_with_region", "bloat_rate",
                  "n_scored"):
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
    b["_meta"]["elicitation"] = "none"
    labels = [m["_label"] for m in written(tmp_path, a, b)]
    assert len(set(labels)) == 2, labels
    # The default elicitation goes unnamed; the ablation is named.
    assert sorted(labels) == ["gpt-5", "gpt-5/none"], labels


def test_two_texts_under_one_elicitation_name_get_different_columns(tmp_path):
    """A reworded `cot` keeps its name, so the text is what separates the runs."""
    runs = []
    for text in ("Work the problem out step by step.",
                 "Think the problem through step by step before you answer."):
        m = a_metrics("gpt-5")
        m["_meta"]["elicitation_config"] = {"name": "cot", "system": text,
                                            "prefix": "", "suffix": ""}
        runs.append(m)
    labels = [m["_label"] for m in written(tmp_path, *runs)]
    assert len(set(labels)) == 2, labels
    assert all("elicitation.system=" in x for x in labels), labels


def test_every_score_column_carries_its_floor(tmp_path):
    text = render(written(tmp_path, a_metrics("model-a")))
    assert "| floor |" in text
    assert "0.375" in text
    assert "Chance-corrected" in text


def test_the_report_shows_the_bloat_rate_without_a_floor(tmp_path):
    """The rate the docs say to read beside a construction task's mean."""
    m = a_metrics("model-a")
    m["by_task"]["status_query"]["bloat_rate"] = 0.125
    text = render(written(tmp_path, m))
    section = text[text.index("## Bloat rate, by task"):]
    table = [line for line in section.splitlines() if line.startswith("|")]
    assert table[0] == "| task | model-a |"
    assert "| status_query | 0.125 (4) |" in table


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


def test_two_token_caps_of_one_model_get_different_columns(tmp_path):
    """`run.py` treats a changed cap as a different run; a report must too.

    Measured on this branch: one model at a 24,576-token cap scored eight of
    twelve tasks at exactly 0.000, and at 57,344 six of those eight came back
    at 1.000. Both columns were headed `Qwen/Qwen3.6-27B`, and `results.csv`
    carried that one string in its `run` field for both.
    """
    a, b = a_metrics("qwen"), a_metrics("qwen")
    a["_meta"]["endpoint"] = {"model": "qwen", "sampling": {"max_tokens": 24576}}
    b["_meta"]["endpoint"] = {"model": "qwen", "sampling": {"max_tokens": 57344}}
    got = [m["_label"] for m in written(tmp_path, a, b)]
    assert len(set(got)) == 2, got
    assert any("24576" in x for x in got) and any("57344" in x for x in got)
    # Only what differs. Temperature is the same in both, so it is not noise in
    # the header.
    assert not any("temperature" in x for x in got)


def test_two_copies_of_one_run_get_different_columns(tmp_path):
    """The tie-break cannot be a field that travels inside the copy.

    `_meta.run_dir` is written at scoring time, so two copies of one run carry
    the same one and both columns came out `gpt-5 [gpt-5]` -- the collision this
    branch exists to break. Copying a finished run is how these files travel;
    `docs/evaluation.md` says to do it.
    """
    m = a_metrics("gpt-5")
    got = [x["_label"] for x in written(tmp_path, m, dict(m))]
    assert len(set(got)) == 2, got
    assert {"gpt-5 [run0]", "gpt-5 [run1]"} == set(got), got


def test_one_run_keeps_a_plain_column_name(tmp_path):
    """Disambiguation is for collisions, not a tax on the common case."""
    m = a_metrics("gpt-5")
    m["_meta"]["endpoint"] = {"model": "gpt-5", "sampling": {"max_tokens": 4096}}
    assert [x["_label"] for x in written(tmp_path, m)] == ["gpt-5"]


def test_the_untruncated_mean_is_counted_over_untruncated_items(tmp_path):
    """A mean of one printed as `(4)` is the confusion the bracket exists to stop."""
    m = a_metrics("model-a")
    m["by_task"]["status_query"].update(n_scored=4, n_untruncated=1,
                                        mean_untruncated=0.5)
    text = render(written(tmp_path, m))
    untruncated = text[text.index("## Mean over untruncated"):]
    assert "0.500 (1)" in untruncated, untruncated
    assert "0.500 (4)" not in untruncated


def test_a_floor_one_run_could_not_measure_is_marked(tmp_path):
    """A borrowed floor makes `corrected` unreproducible from the table it is in.

    One ungradeable row leaves a run with no floor for that task, and the column
    then showed the *other* run's floor with nothing saying so.
    """
    a, b = a_metrics("model-a"), a_metrics("model-b")
    b["by_task"]["status_query"].pop("floor")
    b["by_task"]["status_query"].pop("corrected")
    text = render(written(tmp_path, a, b))
    assert "0.375*" in text
    assert "could not measure a floor" in text


def test_a_stat_missing_from_the_csv_fails_loudly(tmp_path):
    """The artifact people load into a dataframe must not drop a flag quietly.

    The comment claimed `extrasaction` would catch this; the row was filtered to
    the field list first, so the check could not fire.
    """
    m = a_metrics("model-a")
    m["by_task"]["status_query"]["a_stat_nobody_added_here"] = 1
    with pytest.raises(ValueError) as e:
        write_csv(written(tmp_path, m),
                  os.path.join(os.fspath(tmp_path / "report"), "results.csv"))
    assert "a_stat_nobody_added_here" in str(e.value)
