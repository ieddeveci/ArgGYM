"""The figures are drawn from artifacts `evals/score.py` wrote, in this test.

An earlier figure script read `metrics["overall"]` and a per-task `mean_score`.
The harness writes `coverage` and `mean`, and had done for months; nothing ran the script, so nobody found out until the next sweep needed a
chart (#142). A fixture with the field names typed in by hand would have gone
stale beside it, so every metrics file here comes out of `score_run`, and a
rename in `evals/score.py` fails these tests rather than the sweep.

The one run these tests are built on is deliberately messy: one task answers
`500`, another is cut off at the cap, and between them the run is 29%
contaminated. A clean fixture leaves the API-error panel, the cap line and the
dashed-and-hollow styling unexecuted, which is three of the behaviours this
module claims.
"""
from __future__ import annotations

import json
import os
import shutil
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

import pytest
from conftest import answering
from test_a_reference_answer_scores_one_end_to_end import a_run

import arggym
from arggym.core.freeze import taskset_hash
from evals import figures
from evals.report import NotScored
from evals.score import score_run

#: status_query plus the six construction tasks, which is what figure 3 contrasts.
#: Seven tasks over two levels is the smallest fixture that gives every figure a
#: real x-axis; the whole roster would quadruple the floor measurements for a
#: wider legend and nothing else.
TASKS = ("status_query",) + figures.CONSTRUCTION
LEVELS = (3, 6)
ORDERING = "last_link_elitist"
#: Answered with a 500 rather than an answer, so the figures meet an API error,
#: a record with no completion tokens and a group whose `mean` is None.
BROKEN = "attack"
#: Answered, but cut off at the cap, so `Run.cap` returns a number and figure 5
#: draws the line. Four of fourteen items lost puts the run over CONTAM_LIMIT,
#: which is what makes figures 2 and 3 draw it dashed and hollow.
CUT_OFF = "defence"
CAP = 4096


@pytest.fixture(scope="module")
def graded_rows() -> List[Dict[str, Any]]:
    out = []
    for level in LEVELS:
        for task in TASKS:
            row = arggym.TaskDataset(task, level, ORDERING, size=1, seed=0)[0]
            row["metadata"]["source_index"] = len(out)
            out.append(row)
    return out


def a_taskset(path: str, rows: List[Dict[str, Any]]) -> str:
    with open(path, "w") as f:
        f.write(json.dumps({"__manifest__": {
            "taskset_hash": taskset_hash(rows), "n_items": len(rows),
            "versions": {"arggym": arggym.__version__}}}) + "\n")
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return path


@pytest.fixture()
def graded_taskset(tmp_path, graded_rows) -> str:
    return a_taskset(os.fspath(tmp_path / "taskset.jsonl"), graded_rows)


def answering_badly(rows: List[Dict[str, Any]]) -> Callable:
    """Reference answers, except one task that 500s and one that is cut off."""
    good = answering(rows)
    cut = answering(rows, finish_reason="length")
    keyed = {r["question"]: r["task"] for r in rows}

    def handler(body: Dict[str, Any]) -> Dict[str, Any]:
        user = [m for m in body["messages"] if m["role"] == "user"][-1]["content"]
        hit = max((k for k in keyed if k in user), key=len, default=None)
        task = keyed.get(hit) if hit is not None else None
        if task == BROKEN:
            return {"__status__": 500, "error": {"message": "boom"}}
        if task == CUT_OFF:
            return cut(body)
        return good(body)

    return handler


@pytest.fixture()
def scored_run(tmp_path, graded_rows, graded_taskset, provider) -> str:
    p = provider(answering_badly(graded_rows))
    run_dir = a_run(tmp_path, p.url, graded_rows, graded_taskset,
                    sampling={"max_tokens": CAP})
    score_run(run_dir)
    return run_dir


@pytest.fixture()
def another_taskset_run(tmp_path, graded_rows, provider) -> str:
    """A run on genuinely different questions, so its hash differs by itself.

    Editing `taskset_hash` into a copy would test a field this test wrote, which
    is how the guard it checks came to be reachable through `.get` unnoticed.
    """
    rows = graded_rows[:-1]
    path = a_taskset(os.fspath(tmp_path / "other.jsonl"), rows)
    p = provider(answering(rows))
    run_dir = a_run(tmp_path / "other", p.url, rows, path)
    score_run(run_dir)
    return run_dir


def a_copy(tmp_path, run_dir: str, name: str) -> str:
    out = os.fspath(tmp_path / name)
    shutil.copytree(run_dir, out)
    return out


def test_every_figure_is_drawn_from_a_run_the_scorer_produced(tmp_path, scored_run):
    out = os.fspath(tmp_path / "figures")
    assert figures.main([scored_run, "-o", out]) == 0

    # The level axis comes from the records, not from a list of levels the
    # August sweep happened to run.
    assert figures.levels_of(figures.load([scored_run])) == list(LEVELS)
    drawn = sorted(os.listdir(out))
    assert drawn == ["fig1_contamination.png", "fig2_macro_by_level.png",
                     "fig3_recognise_vs_construct.png", "fig4_task_profile.png",
                     "fig5_tokens_vs_truncation.png"], drawn
    for name in drawn:
        # A PNG header and some content. matplotlib writes a file for an empty
        # axes too, so existence alone would pass on five blank charts.
        blob = open(os.path.join(out, name), "rb").read()
        assert blob[:8] == b"\x89PNG\r\n\x1a\n", name
        assert len(blob) > 10_000, (name, len(blob))


def test_the_fixture_run_is_contaminated_enough_to_be_drawn_as_one(scored_run):
    """Otherwise the dashed-and-hollow branch never executes here."""
    run = figures.load([scored_run])[0]
    assert figures.contamination(run) > figures.CONTAM_LIMIT
    assert figures._style(run, "#000000")["linestyle"] == "--"
    assert figures._style(run, "#000000")["markerfacecolor"] == "none"


def test_a_cap_is_drawn_only_where_a_generation_hit_it(scored_run,
                                                       another_taskset_run):
    """A line at a cap nothing reached claims a constraint that is not in play."""
    assert figures.load([scored_run])[0].cap == CAP
    # Every item answered and stopped, so this run's cap bound nothing.
    assert figures.load([another_taskset_run])[0].cap is None


def test_a_task_that_never_answered_is_absent_and_not_zero(scored_run):
    """An API error is not a score, in the figure as in the table.

    Every `attack` item 500s, so its group has no mean. Drawing that as 0.0
    would put a model that was never measured at the bottom of figure 4.
    """
    run = figures.load([scored_run])[0]
    assert run.mean_at(BROKEN, LEVELS[0]) is None
    assert run.mean_at("status_query", LEVELS[0]) == 1.0
    # ... and the macro-average is over the tasks that have one.
    assert figures.macro(run, LEVELS[0]) == 1.0
    assert 0 < figures.errored(run, LEVELS[0]) < 1
    assert 0 < figures.truncated(run, LEVELS[0]) < 1


def test_a_renamed_metric_stops_the_figures_instead_of_drawing_zeros(scored_run):
    """The failure mode of #142, made to fail here first.

    The old script reached every score through `.get(...)`, so the day the field
    was renamed it drew a chart of zeros rather than raising. Subscripting is
    the guard, and this is what says so.
    """
    path = os.path.join(scored_run, "metrics.json")
    m = json.load(open(path))
    for group in m["by_task_level"].values():
        group["mean_score"] = group.pop("mean")
    json.dump(m, open(path, "w"))

    run = figures.load([scored_run])[0]
    with pytest.raises(KeyError) as e:
        run.mean_at("status_query", LEVELS[0])
    assert "mean" in str(e.value)


def test_a_renamed_taskset_hash_stops_the_figures_instead_of_disarming_the_guard(
        tmp_path, scored_run):
    """The guard below is the reason this module errors where `report.py` warns.

    Read through `.get`, a rename makes every run report `None`, the set of
    hashes holds one element, and two sweeps on different tasksets draw one
    curve with nothing said. The key is the scorer's, so the rename is applied
    to what the scorer wrote.
    """
    for d in (scored_run, a_copy(tmp_path, scored_run, "twin")):
        path = os.path.join(d, "metrics.json")
        m = json.load(open(path))
        m["_meta"]["taskset_fingerprint"] = m["_meta"].pop("taskset_hash")
        json.dump(m, open(path, "w"))

    with pytest.raises(KeyError) as e:
        figures.load([scored_run, os.fspath(tmp_path / "twin")])
    assert "taskset_hash" in str(e.value)


def test_runs_on_different_tasksets_are_refused_rather_than_drawn_together(
        scored_run, another_taskset_run):
    """`report.py` warns; a curve has no room to carry the warning.

    Two sweeps built on different tasksets asked different questions, so a line
    through both compares models on different exams. The August script guarded
    this with a `__v2` suffix on the directory name, which only held while
    whoever named the directory remembered to.
    """
    with pytest.raises(figures.Incomparable) as e:
        figures.load([scored_run, another_taskset_run])
    assert "different tasksets" in str(e.value)


def test_two_runs_of_one_taskset_are_drawn_apart(tmp_path, scored_run):
    twin = a_copy(tmp_path, scored_run, "twin")
    runs = figures.load([scored_run, twin])
    assert len({r.label for r in runs}) == 2, [r.label for r in runs]

    style = figures.palette(runs)
    assert len({(v["color"], v["marker"]) for v in style.values()}) == 2
    out = os.fspath(tmp_path / "figures")
    assert figures.main([scored_run, twin, "-o", out]) == 0
    assert len(os.listdir(out)) == 5


def test_a_title_counts_the_tasks_each_run_has_and_not_their_union(tmp_path,
                                                                   scored_run):
    """A mean of one task under a title claiming six is the defect being repaired.

    `macro` and the construction mean are per run; a count taken across runs
    describes no line on the chart. A two-task run drawn at 1.0 sits above every
    honest run, and the title says it is a mean of six.
    """
    thin = a_copy(tmp_path, scored_run, "thin")
    path = os.path.join(thin, "metrics.json")
    m = json.load(open(path))
    m["by_task"] = {"status_query": m["by_task"]["status_query"],
                    "attack_defense": m["by_task"]["attack_defense"]}
    json.dump(m, open(path, "w"))

    runs = figures.load([scored_run, thin])
    assert figures._construction_count(runs) == "1-6"
    assert figures._counted(len(r.tasks) for r in runs) == "2-7"
    # One run, one number: the range is for disagreement, not a tax on the
    # common case.
    assert figures._construction_count(figures.load([scored_run])) == "6"


def test_more_runs_than_the_palette_can_tell_apart_are_refused():
    """The old roster silently dropped an unknown model; this must not merge two.

    `tab10` has ten colours, so a marker list of ten would share its period and
    run eleven would come back identical to run one.
    """
    many = [SimpleNamespace(label=f"run-{i}") for i in range(10 * len(figures.MARKERS))]
    style = figures.palette(many)
    assert len({(v["color"], v["marker"]) for v in style.values()}) == len(many)

    with pytest.raises(figures.TooMany):
        figures.palette(many + [SimpleNamespace(label="one-too-many")])


def test_a_profile_is_of_the_run_that_measured_the_most(tmp_path,
                                                        another_taskset_run):
    """Contamination alone ties, and the tie broke on how the labels sorted.

    Both runs here are clean, so the contamination key is exactly equal and the
    second key decides. A profile of two items is not a profile.
    """
    thin = a_copy(tmp_path, another_taskset_run, "thin")
    keep = list(open(os.path.join(thin, "samples.jsonl")))[:2]
    open(os.path.join(thin, "samples.jsonl"), "w").writelines(keep)

    runs = figures.load([another_taskset_run, thin])
    assert len({figures.contamination(r) for r in runs}) == 1, "not a tie"
    assert len(figures.profile_of(runs).records) == max(len(r.records) for r in runs)
    # And the tie does not turn on the order the runs arrive in.
    assert figures.profile_of(runs[::-1]) is figures.profile_of(runs)


def test_a_run_without_its_records_is_named_rather_than_drawn_empty(tmp_path,
                                                                    scored_run):
    """Figures 1 and 5 are per item, and `metrics.json` holds no items."""
    half = a_copy(tmp_path, scored_run, "half")
    os.remove(os.path.join(half, "samples.jsonl"))

    with pytest.raises(NotScored) as e:
        figures.load([half])
    assert "samples.jsonl" in str(e.value)

