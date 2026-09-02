"""A cell whose generator returns None is recorded, not silently dropped (issue #24).

_export_row is stubbed so no real item is generated.
"""
from __future__ import annotations

import json

import pytest

from arggym.core import export

MISSING = (6, export.WEAKEST_LINK, 1)


class _Fake:
    prompt = "p"
    reference = "r"
    metadata = {}


def _fake_row(task, lv, o, s):
    if (lv, o, s) == MISSING:
        return None
    return _Fake(), 1.0, [], 1


def _stub(monkeypatch, fn=_fake_row):
    monkeypatch.setattr(export, "_export_row", fn)


def _manifest(path):
    with open(path) as f:
        return json.loads(f.readline())["__manifest__"]


def test_default_raises_and_writes_nothing(tmp_path, monkeypatch):
    _stub(monkeypatch)
    out = tmp_path / "sq.jsonl"
    with pytest.raises(SystemExit) as info:
        export.export_task(str(out), "status_query")
    msg = str(info.value)
    assert "status_query" in msg
    assert "L6" in msg and export.WEAKEST_LINK in msg and "seed 1" in msg
    assert not out.exists()


def test_allow_missing_writes_and_records_the_cell(tmp_path, monkeypatch, capsys):
    _stub(monkeypatch)
    out = tmp_path / "sq.jsonl"
    export.export_task(str(out), "status_query", allow_missing=True)
    m = _manifest(out)
    assert m["n_requested"] == 20
    assert m["n_items"] == 19
    assert m["missing_cells"] == [{"level": 6, "ordering": export.WEAKEST_LINK, "seed": 1}]
    with open(out) as f:
        assert len(f.readlines()) == 20  # manifest + 19 rows
    lines = capsys.readouterr().out.splitlines()
    wrote = next(i for i, l in enumerate(lines) if l.startswith("wrote 19 items"))
    assert "L6" in lines[wrote + 1] and "seed 1" in lines[wrote + 1]


def test_complete_grid_has_no_missing_cells(tmp_path, monkeypatch):
    _stub(monkeypatch, lambda task, lv, o, s: (_Fake(), 1.0, [], 1))
    out = tmp_path / "sq.jsonl"
    export.export_task(str(out), "status_query")
    m = _manifest(out)
    assert m["missing_cells"] == []
    assert m["n_requested"] == m["n_items"] == 20
