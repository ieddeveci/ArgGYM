"""Every cell generates, its reference scores 1.0, and generation is deterministic."""
from __future__ import annotations

import pytest

from arggym import inspector

from .conftest import KNOWN_MISSING, generate
from .registry import MODES


def test_registry_matches_inspector():
    assert set(MODES) == {m[0] for m in inspector.MODES}


def test_every_cell_generates(cell):
    item = generate(cell)
    if cell in KNOWN_MISSING:
        assert item is None, f"{cell.short} is in KNOWN_MISSING but generates; remove it"
    else:
        assert item is not None, f"{cell.short}: generator returned None"


def test_reference_scores_one(cell, item):
    adapter = MODES[cell.mode]
    result = adapter.score(adapter.reference(item), item)
    assert result.score == pytest.approx(1.0), f"{cell.short}: {result.reason}"


def test_generation_is_deterministic(cell, item):
    adapter = MODES[cell.mode]
    again = adapter.make(cell.level, cell.seed, cell.ordering)
    assert again is not None
    assert adapter.prompt(again) == adapter.prompt(item)
    assert adapter.reference(again) == adapter.reference(item)
