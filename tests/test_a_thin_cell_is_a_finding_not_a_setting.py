"""The export densifies, and says what it skipped.

`docs/dataset-contract.md` section 7. The generator is pure and raises on a
failed seed; this is the one place that scans past a failure, so this is the one
place that can record it. A cell that quietly ships short is the failure mode
being designed out.
"""
import dataclasses
import json
import os
import signal
import threading
import time
from concurrent.futures.process import BrokenProcessPool

import pytest

from arggym.core import freeze as F
from arggym.core.build import BuildReport
from arggym.core.spec import SeedPolicy, TasksetSpec, load

CHEAP = TasksetSpec(tasks=("claim_chain",), levels=(3,),
                    orderings=("last_link_elitist",),
                    seeds=SeedPolicy(start=0, take=2, scan_limit=10))


class _Refusing:
    """A dataset that fails the seeds named, and defers the rest."""

    def __init__(self, real, bad):
        self._real, self._bad = real, bad

    def build_at(self, k):
        if k in self._bad:
            return None, BuildReport(None, 3, {"refused_by_the_test": 3})
        return self._real.build_at(k)


def _patch(monkeypatch, bad):
    real_cls = F.TaskDataset

    def factory(*a, **kw):
        return _Refusing(real_cls(*a, **kw), bad)

    monkeypatch.setattr(F, "TaskDataset", factory)


class _Wasteful:
    """A dataset whose every item cost `calls` candidates, the first of which built."""

    def __init__(self, real, calls):
        self._real, self._calls = real, calls

    def build_at(self, k):
        entry, report = self._real.build_at(k)
        return entry, BuildReport(report.item, self._calls,
                                  {"refused_by_the_test": self._calls - 1})


class _Unproven:
    """A dataset that reports the seeds named as built with an unproven minimum.

    Its report carries a refusal of its own. The skip record is supposed to keep
    the candidate rejections that seed had already made, and a fake whose report
    had none could not tell a record that keeps them from one that drops them.
    """

    def __init__(self, real, bad):
        self._real, self._bad = real, bad

    def build_at(self, k):
        entry, report = self._real.build_at(k)
        if entry is None or k not in self._bad:
            return entry, report
        entry["metadata"]["gold"].setdefault("metadata", {})["minimality_proven"] = False
        return entry, BuildReport(report.item, report.calls + 2,
                                  {"refused_by_the_test": 2})


def test_a_filled_cell_records_the_seeds_it_used(tmp_path):
    m = F.freeze(CHEAP, str(tmp_path / "t.jsonl"), verbose=False)
    cell = m["cells"]["claim_chain|L3|last_link_elitist"]
    assert cell["seeds_used"] == [0, 1]
    assert cell["acceptance_rate"] == 1.0


def test_a_skipped_seed_is_named_and_the_cell_still_fills(monkeypatch, tmp_path):
    _patch(monkeypatch, {0, 2})
    m = F.freeze(CHEAP, str(tmp_path / "t.jsonl"), verbose=False)
    cell = m["cells"]["claim_chain|L3|last_link_elitist"]
    assert cell["seeds_used"] == [1, 3]
    assert [s["seed"] for s in cell["seeds_skipped"]] == [0, 2]
    assert cell["n"] == 2


def test_a_cell_that_cannot_fill_fails_naming_itself(monkeypatch, tmp_path):
    _patch(monkeypatch, set(range(10)))
    with pytest.raises(SystemExit, match=r"claim_chain\|L3\|last_link_elitist yielded 0 of 2"):
        F.freeze(CHEAP, str(tmp_path / "t.jsonl"), verbose=False)


def test_a_degraded_cell_is_refused_rather_than_shipped(monkeypatch, tmp_path):
    # Filling 2 items out of 6 seeds is 0.33 acceptance, below the 0.5 default.
    # The cell could ship; the point is that it should not do so silently.
    _patch(monkeypatch, {0, 1, 2, 4})
    with pytest.raises(SystemExit, match="acceptance 0.33"):
        F.freeze(CHEAP, str(tmp_path / "t.jsonl"), verbose=False)


def test_a_degraded_cell_ships_when_the_spec_says_so(monkeypatch, tmp_path):
    _patch(monkeypatch, {0, 1, 2, 4})
    lenient = TasksetSpec(tasks=("claim_chain",), levels=(3,),
                          orderings=("last_link_elitist",),
                          seeds=SeedPolicy(start=0, take=2, scan_limit=10),
                          min_acceptance=0.3)
    m = F.freeze(lenient, str(tmp_path / "t.jsonl"), verbose=False)
    assert m["cells"]["claim_chain|L3|last_link_elitist"]["seeds_used"] == [3, 5]


def test_a_cell_that_discards_most_of_what_it_builds_is_refused_when_the_spec_says(
        monkeypatch, tmp_path):
    # Every seed builds, so seed acceptance is 1.00 and the seed guard is silent;
    # the cell reached its two items by throwing away eight candidates (#114).
    real_cls = F.TaskDataset
    monkeypatch.setattr(F, "TaskDataset", lambda *a, **kw: _Wasteful(real_cls(*a, **kw), 5))
    strict = TasksetSpec(tasks=("claim_chain",), levels=(3,),
                         orderings=("last_link_elitist",),
                         seeds=SeedPolicy(start=0, take=2, scan_limit=10),
                         min_build_acceptance=0.5)
    with pytest.raises(SystemExit, match="build acceptance 0.20"):
        F.freeze(strict, str(tmp_path / "t.jsonl"), verbose=False)
    m = F.freeze(CHEAP, str(tmp_path / "t.jsonl"), verbose=False)
    cell = m["cells"]["claim_chain|L3|last_link_elitist"]
    assert cell["acceptance_rate"] == 1.0 and cell["build_acceptance_rate"] == 0.2


def test_a_row_whose_minimum_the_search_gave_up_on_is_skipped_and_named(monkeypatch, tmp_path):
    # A stated minimum the search could not prove is an upper bound wearing the
    # name of a minimum, and the bloat gate reads it as one (#124). The seed is
    # skipped like one that built nothing, and the skip says why.
    real_cls = F.TaskDataset
    monkeypatch.setattr(F, "TaskDataset", lambda *a, **kw: _Unproven(real_cls(*a, **kw), {0}))
    m = F.freeze(CHEAP, str(tmp_path / "t.jsonl"), verbose=False)
    cell = m["cells"]["claim_chain|L3|last_link_elitist"]
    assert cell["seeds_used"] == [1, 2]
    assert [(s["seed"], s["reason"]) for s in cell["seeds_skipped"]] == [(0, "minimality_unproven")]
    assert cell["build_rejections"]["minimality_unproven"] == 1
    # And the skip keeps what that seed had already refused, the way a seed that
    # built nothing does. Writing `{"minimality_unproven": 1}` instead drops them,
    # leaving a record that reads as a clean seed with one late failure.
    assert cell["seeds_skipped"][0]["reasons"] == {"minimality_unproven": 1,
                                                   "refused_by_the_test": 2}


def test_the_same_spec_twice_gives_the_same_hash(tmp_path):
    a = F.freeze(CHEAP, str(tmp_path / "a.jsonl"), verbose=False)
    b = F.freeze(CHEAP, str(tmp_path / "b.jsonl"), verbose=False)
    assert a["taskset_hash"] == b["taskset_hash"]
    assert a["cells"] == b["cells"]


def test_a_parallel_freeze_writes_the_serial_file_byte_for_byte(tmp_path):
    # Six cells over three generators, filled out of spec order: the slow-first
    # schedule starts the `weakest_link_democratic` cells before the others.
    spec = load("tests/data/one-cheap-cell.yaml")
    F.freeze(spec, str(tmp_path / "serial.jsonl"), verbose=False, workers=1)
    F.freeze(spec, str(tmp_path / "parallel.jsonl"), verbose=False, workers=2)
    assert (tmp_path / "serial.jsonl").read_bytes() == (tmp_path / "parallel.jsonl").read_bytes()


def test_a_parallel_freeze_refuses_the_cell_a_serial_one_refuses(tmp_path):
    # Two cells keep half their candidates, `status_query` fourth in spec order and
    # `preference_construction` sixth. Serial stops at the fourth; parallel has to
    # name the same cell whichever of the two finishes first.
    spec = dataclasses.replace(load("tests/data/one-cheap-cell.yaml"),
                               min_build_acceptance=0.9)
    messages = []
    for workers in (1, 2):
        with pytest.raises(SystemExit) as err:
            F.freeze(spec, str(tmp_path / "t.jsonl"), verbose=False, workers=workers)
        assert err.type is F.CellUnfilled
        messages.append(str(err.value))
    assert messages[0].startswith("status_query|L3|weakest_link_democratic filled")
    assert messages[0] == messages[1]


def _refuse_the_first_cell_last(task, level, ordering, spec):
    """A `fill_cell` that refuses every cell, the first in spec order a second late."""
    if ordering == "last_link_elitist":
        time.sleep(1.0)
    raise F.CellUnfilled(f"{task}|L{level}|{ordering} refused by the test")


def test_a_parallel_freeze_names_the_first_failing_cell_in_spec_order_not_in_time(
        monkeypatch, tmp_path):
    # The cheap cells above fail in spec order whichever way results are read, so
    # they cannot tell spec order from completion order. Here the second cell
    # fails a second before the first; reading results as they complete would
    # name the second.
    two = TasksetSpec(tasks=("claim_chain",), levels=(3,),
                      orderings=("last_link_elitist", "weakest_link_elitist"),
                      seeds=SeedPolicy(start=0, take=2, scan_limit=10))
    monkeypatch.setattr(F, "fill_cell", _refuse_the_first_cell_last)
    with pytest.raises(F.CellUnfilled, match=r"^claim_chain\|L3\|last_link_elitist refused"):
        F.freeze(two, str(tmp_path / "t.jsonl"), verbose=False, workers=2)


def _die_in_the_first_cell(task, level, ordering, spec):
    """A `fill_cell` whose first cell ends its process, the way an OOM kill does."""
    if ordering == "last_link_elitist":
        os._exit(1)
    time.sleep(30)


class _Hung(Exception):
    pass


def _raise_hung(signum, frame):
    raise _Hung("the freeze was still waiting on a dead worker")


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="needs SIGALRM")
def test_a_worker_that_dies_fails_the_freeze_rather_than_hanging_it(monkeypatch, tmp_path):
    # A pool that replaces a dead worker without failing its task leaves the
    # freeze waiting forever. The alarm turns that hang into a failure here; the
    # other cell's 30 s shows the workers are stopped rather than waited for.
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("signal handlers can only be set from the main thread")
    two = TasksetSpec(tasks=("claim_chain",), levels=(3,),
                      orderings=("last_link_elitist", "weakest_link_elitist"),
                      seeds=SeedPolicy(start=0, take=2, scan_limit=10))
    monkeypatch.setattr(F, "fill_cell", _die_in_the_first_cell)
    previous = signal.signal(signal.SIGALRM, _raise_hung)
    signal.alarm(15)
    start = time.monotonic()
    try:
        with pytest.raises(BrokenProcessPool):
            F.freeze(two, str(tmp_path / "t.jsonl"), verbose=False, workers=2)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    assert time.monotonic() - start < 10


def test_skipping_a_seed_changes_the_hash(monkeypatch, tmp_path):
    # Different items, so a different taskset. The skip list is what tells you
    # a generator moved rather than a renderer.
    plain = F.freeze(CHEAP, str(tmp_path / "a.jsonl"), verbose=False)
    _patch(monkeypatch, {0})
    shifted = F.freeze(CHEAP, str(tmp_path / "b.jsonl"), verbose=False)
    assert plain["taskset_hash"] != shifted["taskset_hash"]
    assert plain["cells"] != shifted["cells"]


def test_the_manifest_records_what_the_taskset_was_built_with(tmp_path):
    m = F.freeze(CHEAP, str(tmp_path / "t.jsonl"), verbose=False)
    v = m["versions"]
    assert v["arggym"] and v["pyarg"] and v["python"]
    assert v["prompt_version"] and v["scoring_version"] and v["theory_schema"]
    assert m["spec"]["seeds"]["scan_limit"] == 10, "the scan limit is part of the procedure"


def test_the_minimum_caveat_travels_with_the_file(tmp_path):
    p = tmp_path / "t.jsonl"
    F.freeze(CHEAP, str(p), verbose=False)
    assert "not proven globally minimal" in F.read_manifest(str(p))["minimum_caveat"]


def test_rows_are_numbered_across_the_whole_file_not_per_cell(tmp_path):
    two = TasksetSpec(tasks=("claim_chain",), levels=(3,),
                      orderings=("last_link_elitist", "weakest_link_elitist"),
                      seeds=SeedPolicy(start=0, take=2, scan_limit=10))
    p = tmp_path / "t.jsonl"
    F.freeze(two, str(p), verbose=False)
    rows = [json.loads(x) for x in open(p)][1:]
    assert [r["metadata"]["source_index"] for r in rows] == [0, 1, 2, 3]
