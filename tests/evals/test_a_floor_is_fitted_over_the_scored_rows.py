"""A group's floor is measured over the rows its mean is over.

`corrected` is `(mean - floor) / (1 - floor)`, and the mean skips a record whose
score is None -- an API error or a scorer refusal. A floor that still counted
those rows would describe different items than the mean it corrects. Errors
are not spread evenly either: timeouts land on the long, high-level
generations, and on `semantics_query` a floor over every row sat 0.026 below
the floor over the scored ones when level 9 was lost.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

import arggym
from evals.score import aggregate

TASK = "semantics_query"
LEVELS = (1, 3, 6, 9)
ORDERINGS = ("last_link_elitist", "last_link_democratic",
             "weakest_link_elitist", "weakest_link_democratic")


@pytest.fixture(scope="module")
def taskset() -> List[Dict[str, Any]]:
    return [arggym.TaskDataset(TASK, lv, od, size=1, seed=s)[0]
            for lv in LEVELS for od in ORDERINGS for s in (0, 1)]


def _records(rows, errored) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        lv, od = r["metadata"]["level"], r["metadata"]["ordering"]
        err = errored(lv)
        out.append({"id": r["id"], "task": TASK, "level": lv, "ordering": od,
                    "truncated": False, "reason": "api_error" if err else "ok",
                    "requests_timed_out": 0,
                    "api_error": "timeout" if err else None,
                    "api_error_kind": "timeout" if err else None,
                    "score": None if err else 0.5,
                    "success": None if err else False})
    return out


def _floor(rows, fit_rows) -> float:
    return round(arggym.floors(rows, fit_rows=fit_rows)[TASK]["floor"], 4)


def test_an_errored_row_leaves_the_floor(taskset):
    records = _records(taskset, lambda lv: lv == 9)
    got = aggregate(records, {r["id"]: r for r in taskset})["by_task"][TASK]

    scored = [r for r in taskset if r["metadata"]["level"] != 9]
    want = _floor(scored, taskset)
    # The case is only a test if losing level 9 moves the floor at all.
    assert want != _floor(taskset, taskset)
    assert got["n_scored"] == len(scored)
    assert got["floor"] == want
    assert got["corrected"] == round(arggym.corrected(got["mean"], want), 4)


def test_a_group_with_no_scored_row_has_no_floor(taskset):
    records = _records(taskset, lambda lv: lv == 9)
    got = aggregate(records, {r["id"]: r for r in taskset})["by_task_level"]

    lost = got[f"{TASK}|L9"]
    assert lost["n_scored"] == 0 and lost["mean"] is None
    # Neither a floor beside a None mean, nor a floor_error: nothing failed to
    # be measured, there was simply nothing to measure.
    assert not {"floor", "corrected", "floor_error"} & set(lost), lost
    assert "floor" in got[f"{TASK}|L6"]


def test_a_run_without_errors_keeps_its_floor(taskset):
    records = _records(taskset, lambda lv: False)
    got = aggregate(records, {r["id"]: r for r in taskset})["by_task"][TASK]
    assert got["floor"] == _floor(taskset, taskset)
