"""Grid fixtures and the per-process item cache for the end-to-end suite.

`cell` is parametrized over FULL_GRID for every mode; cells outside FAST_GRID carry the
`slow` marker, and `pyproject.toml` deselects them by default. `fast_cell` covers FAST_GRID
only. A test limits itself with `@pytest.mark.families("label_map", ...)` or
`@pytest.mark.modes("status_query", ...)`.

The cache is a module dict, so it lives per process: under `pytest -n auto` each xdist
worker generates its own copy of the cells it is handed. That is deliberate; there is no
cross-process cache.
"""
from __future__ import annotations

from typing import Any, Dict, NamedTuple, Optional

import pytest

from arggym.core.curriculum import LAST_LINK, WEAKEST_LINK
from arggym.core.spec import LEVELS

from .registry import MODES

ORDERINGS = (LAST_LINK, WEAKEST_LINK)


class Cell(NamedTuple):
    mode: str
    level: int
    ordering: str
    seed: int

    @property
    def short(self) -> str:
        return f"{self.mode}-L{self.level}-{self.ordering.split('_')[0]}-s{self.seed}"


FAST_GRID = [(level, ordering, 0) for level in (3, 6, 9) for ordering in ORDERINGS]
# The level axis comes from the spec; restating it here is how a grid and a suite drift
# apart, which is #30. The ordering axis is sampled on purpose -- the elitist member of
# each family -- because these tests are about the item a builder produces and the
# democratic readings are covered where they change an answer.
FULL_GRID = [(level, ordering, seed) for level in LEVELS
             for ordering in ORDERINGS for seed in (0, 1)]

# Cells whose generator is known to return None. A listed cell that does generate fails
# test_every_cell_generates, so the list cannot go stale silently.
KNOWN_MISSING: set = set()

_CACHE: Dict[Cell, Any] = {}


def generate(cell: Cell) -> Optional[Any]:
    """The item for a cell, generated once per process."""
    if cell not in _CACHE:
        _CACHE[cell] = MODES[cell.mode].make(cell.level, cell.seed, cell.ordering)
    return _CACHE[cell]


def _cells(modes, grid, slow_outside_fast: bool):
    fast = set(FAST_GRID)
    params = []
    for mode in modes:
        for level, ordering, seed in grid:
            cell = Cell(mode, level, ordering, seed)
            marks = [pytest.mark.slow] if slow_outside_fast and (level, ordering, seed) not in fast else []
            params.append(pytest.param(cell, id=cell.short, marks=marks))
    return params


def pytest_generate_tests(metafunc):
    fam = metafunc.definition.get_closest_marker("families")
    only = metafunc.definition.get_closest_marker("modes")
    modes = [m for m, a in MODES.items()
             if (fam is None or a.family in fam.args) and (only is None or m in only.args)]
    if "cell" in metafunc.fixturenames:
        metafunc.parametrize("cell", _cells(modes, FULL_GRID, slow_outside_fast=True))
    if "fast_cell" in metafunc.fixturenames:
        metafunc.parametrize("fast_cell", _cells(modes, FAST_GRID, slow_outside_fast=False))
    if "mode" in metafunc.fixturenames:
        metafunc.parametrize("mode", modes)


def _item_or_skip(cell: Cell):
    item = generate(cell)
    if item is None:
        pytest.skip(f"{cell.short}: no item (test_every_cell_generates reports this)")
    return item


@pytest.fixture
def item(cell: Cell):
    return _item_or_skip(cell)


@pytest.fixture
def fast_item(fast_cell: Cell):
    return _item_or_skip(fast_cell)
