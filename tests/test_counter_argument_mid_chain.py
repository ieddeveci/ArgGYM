"""Mid-chain-target cells at depth-4 levels build, and their gold verifies.

At levels 10-14 every chain splices a branch rule at chain position 1, the same
position the mid-chain target index points at. Before the fix for issue #40 the
index landed on the branch rule, build returned None for every odd build seed,
and the mid-chain path vanished from those levels. This locks in both halves:
build succeeds with metadata["mid_chain_target"] true, and the reference still
scores 1.0.
"""
from __future__ import annotations

import pytest

from arggym.core.scoring import score_item
from arggym.tasks.counter_argument import LAST_LINK, WEAKEST_LINK, as_score_input, build


@pytest.mark.parametrize("ordering", [LAST_LINK, WEAKEST_LINK])
def test_mid_chain_target_builds_and_gold_verifies(ordering):
    item = build(10, 1, ordering)
    assert item is not None, "affected cell (level 10, odd seed) rejected the build"
    assert item.metadata["mid_chain_target"], "item lost its mid-chain target"
    ref = score_item(item.reference, as_score_input(item))
    assert ref.score == pytest.approx(1.0), f"gold regrades at {ref.score}: {ref.reason}"
