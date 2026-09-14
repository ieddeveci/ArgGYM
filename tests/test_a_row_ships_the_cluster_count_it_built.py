"""`semantics_query` publishes the cluster count it built the item from.

`build` held two counts. `n_cluster = 2` sized the name pool, `_n_cluster` drew the
number of clusters the loop actually built, and the metadata shipped the first one, so
all 40 exported rows said 2 -- a number `randint` cannot return at any level (#141). The
two names are one underscore apart and nine lines apart, and nothing in the repo read
`metadata["n_clusters"]`, so no test could see it.

The guard is the band rather than the loop variable, because the band is the thing a
test can hold from outside `build`: `cluster_bounds` is the draw's own bounds, so a
count outside it was never drawn, whatever produced it. A count inside the band but
constant is the other half of the same failure, which is why the second test is here --
2 was in no band, but a future 4 would be.

The column is `last_link_elitist` at seed 0 over levels 1 to 15, following #123: the
level axis is cheap here, 0.94 s for the whole file, and the ordering and seed axes
belong to `tests/e2e`. The three counts the exported grid draws, 3, 4 and 5, all appear
in this one column.
"""
from __future__ import annotations

import pytest

from arggym.tasks import semantics_query as sq

SEED, ORDERING = 0, "last_link_elitist"
#: The whole curriculum range, not the five exported levels. A spec may name any level
#: in this range and freeze it, and the count was wrong at every one of them.
LEVELS = tuple(range(1, 16))

#: Two tests walk the same fifteen cells, so one build serves both.
_CACHE: dict = {}


def item_at(level: int):
    if level not in _CACHE:
        it = sq.make_item(level, SEED, ORDERING)
        assert it is not None, f"no semantics_query item at level {level} seed {SEED}"
        _CACHE[level] = it
    return _CACHE[level]


@pytest.mark.parametrize("level", LEVELS)
def test_the_count_a_row_ships_is_one_the_generator_can_draw(level):
    lo, hi = sq.cluster_bounds(level)
    shipped = item_at(level).metadata["n_clusters"]
    assert lo <= shipped <= hi, (
        f"level {level} ships n_clusters={shipped}, outside the {lo}-{hi} the draw can "
        f"return, so the metadata is reporting something the generator did not build")


def test_the_count_moves_with_the_item():
    """A constant inside the band would pass the test above and still be a constant."""
    seen = {level: item_at(level).metadata["n_clusters"] for level in LEVELS}
    assert len(set(seen.values())) > 1, (
        f"every level ships the same n_clusters over {len(LEVELS)} cells: {seen}. The "
        f"count is drawn per item, so one value across the curriculum means the field "
        f"is a constant wearing a draw's name")
