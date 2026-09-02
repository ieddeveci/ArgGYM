"""export_task writes rows whose references verify against freshly generated items."""
from __future__ import annotations

import json

import pytest

from arggym.core import export

from .conftest import Cell, generate
from .registry import MODES


@pytest.mark.parametrize("task", sorted(export._EXPORTABLE))
def test_exported_rows_verify(task, tmp_path):
    path = tmp_path / f"{task}.jsonl"
    export.export_task(str(path), task, levels=(3,), seeds=(0,))
    lines = path.read_text().splitlines()
    manifest = json.loads(lines[0])["__manifest__"]
    rows = [json.loads(l) for l in lines[1:]]
    assert rows, f"{task}: no rows exported"
    assert manifest["n_items"] == len(rows)
    assert manifest["n_valid"] == manifest["n_items"], manifest
    for row in rows:
        assert isinstance(row["reference"], str)
        cell = Cell(task, row["level"], row["ordering"], row["seed"])
        item = generate(cell)
        assert item is not None, cell.short
        result = MODES[task].score(row["reference"], item)
        assert result["score"] == pytest.approx(1.0), (cell.short, result["reason"])


def test_every_mode_is_exportable():
    assert set(export._EXPORTABLE) == set(MODES)
