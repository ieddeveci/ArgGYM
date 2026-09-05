"""The manifest is how a number is traced back to what produced it.

And how "not measured" is kept from reading as "measured as failing": the
previous harness's error guard broke a level loop and silently dropped two cells
from a published report, where a missing cell and a failing one looked the same.
"""
from __future__ import annotations

import pytest

from evals import taskset


def test_a_filter_that_matches_nothing_is_refused(rows):
    """Silently matching nothing reports a missing sweep as a clean run."""
    with pytest.raises(ValueError) as e:
        taskset.select(rows, levels=[99])
    assert "match no row" in str(e.value)

    with pytest.raises(ValueError):
        taskset.select(rows, tasks=["no_such_task"])


def test_a_filter_slices_without_rebuilding(rows):
    one = taskset.select(rows, tasks=[rows[0]["task"]])
    assert [r["id"] for r in one] == [rows[0]["id"]]
    assert len(taskset.select(rows, limit=1)) == 1
    assert len(taskset.select(rows)) == len(rows)


def test_a_taskset_carries_its_manifest(taskset_file, rows):
    manifest, loaded = taskset.load(taskset_file)
    assert manifest["taskset_hash"]
    assert manifest["n_items"] == len(rows)
    assert [r["id"] for r in loaded] == [r["id"] for r in rows]


def test_a_file_with_no_manifest_still_loads(tmp_path, rows):
    import json
    p = tmp_path / "bare.jsonl"
    p.write_text("\n".join(json.dumps(r, default=str) for r in rows))
    manifest, loaded = taskset.load(str(p))
    assert manifest == {}
    assert len(loaded) == len(rows)


def test_the_manifest_names_the_key_variable_and_never_the_key(monkeypatch):
    from evals.client import Endpoint

    monkeypatch.setenv("A_SECRET_VAR", "sk-do-not-write-this-down")
    recorded = Endpoint(model="m", api_key_env="A_SECRET_VAR").redacted()
    assert "sk-do-not-write-this-down" not in repr(recorded)
    assert recorded["api_key_env"] == "A_SECRET_VAR"
