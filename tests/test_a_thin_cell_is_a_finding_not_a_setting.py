"""The export densifies, and says what it skipped.

`docs/dataset-contract.md` section 7. The generator is pure and raises on a
failed seed; this is the one place that scans past a failure, so this is the one
place that can record it. A cell that quietly ships short is the failure mode
being designed out.
"""
import json

import pytest

from arggym.core import freeze as F
from arggym.core.build import BuildReport
from arggym.core.spec import SeedPolicy, TasksetSpec

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


def test_the_same_spec_twice_gives_the_same_hash(tmp_path):
    a = F.freeze(CHEAP, str(tmp_path / "a.jsonl"), verbose=False)
    b = F.freeze(CHEAP, str(tmp_path / "b.jsonl"), verbose=False)
    assert a["taskset_hash"] == b["taskset_hash"]
    assert a["cells"] == b["cells"]


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
