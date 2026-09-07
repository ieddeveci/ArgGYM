"""The mid-chain target is on exactly the levels and seeds that schedule it.

`mid_target` is `level >= 6 and seed % 2 == 1`, and where the chains are deep enough and
numerous enough the item takes it. Twice a fix elsewhere in the generator has emptied that
path out of a band of levels while every cell still produced a valid item, which is a
defect nothing else notices: #40 lost levels 10 to 14 when the target index landed on a
spliced branch rule, and the naive fix for #93 -- make the mid-chain target rule strict
where it sits -- would have lost levels 6 and 9, because at depth 3 the target sat on the
chain's first rule and a strict rule there leaves nothing for a counter-argument to break.

So the schedule is asserted as a closed form over the whole level range rather than at the
cells that once broke, and the level-10 cell #40 was about keeps its own test with the
gold check attached.

The two tests check for a build differently because the two entry points answer
differently. `build` answers `CAItem | Rejected` (#115), and a `Rejected` is truthy and
passes `is not None`, so the level-10 test would go vacuously green under an
always-reject mutation if it asked that question; it asks `isinstance` instead.
`make_item` still answers `Optional[CAItem]`, so the sweep asks `is not None`.
"""
from __future__ import annotations

import pytest

from arggym.core.build import Rejected
from arggym.core.scoring import score_item
from arggym.core.spec import ALL_ORDERINGS, SEEDS
from arggym.tasks.counter_argument import (
    LAST_LINK,
    WEAKEST_LINK,
    as_score_input,
    build,
    make_item,
)

ALL_LEVELS = tuple(range(1, 16))


def test_the_mid_chain_path_is_reachable_from_the_exported_seeds():
    """The path needs an odd seed, so the schedule below rests on `SEEDS`."""
    assert any(s % 2 == 1 for s in SEEDS), (
        f"no odd seed in {SEEDS}, so no exported item takes the mid-chain target and the "
        f"schedule below would hold vacuously")


@pytest.mark.parametrize("ordering", [LAST_LINK, WEAKEST_LINK])
def test_mid_chain_target_builds_and_gold_verifies(ordering):
    item = build(10, 1, ordering)
    # Not `is not None`: a Rejected satisfies that, and this assertion is the
    # whole test.
    assert not isinstance(item, Rejected), (
        f"affected cell (level 10, odd seed) rejected the build: {item.reason}")
    assert item.metadata["mid_chain_target"], "item lost its mid-chain target"
    ref = score_item(item.reference, as_score_input(item))
    assert ref.score == pytest.approx(1.0), f"gold regrades at {ref.score}: {ref.reason}"


@pytest.mark.slow
@pytest.mark.parametrize("allow_strict", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("ordering", ALL_ORDERINGS)
@pytest.mark.parametrize("level", ALL_LEVELS)
def test_the_mid_chain_target_is_on_exactly_the_scheduled_cells(
        level, ordering, seed, allow_strict):
    item = make_item(level, seed, ordering, allow_strict=allow_strict)
    assert item is not None, f"L{level} {ordering} seed {seed} allow_strict={allow_strict}"
    want = level >= 6 and seed % 2 == 1
    assert item.metadata["mid_chain_target"] is want, (
        f"L{level} {ordering} seed {seed} allow_strict={allow_strict}: the mid-chain "
        f"target is {item.metadata['mid_chain_target']} where the schedule says {want}")
