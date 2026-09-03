"""A chain that passes through a junction needs its branch premise ranked too.

Under a democratic strength ordering one argument is weaker than another only when every
one of its elements is weaker than something in the other; under an elitist ordering a
single weak element is enough. `counter_argument` builds a chain that can pass through a
junction, and each junction branch rests on an ordinary premise of its own. The reference
solution ranked the branch rule and the chain's root premise, but never the branch premise,
so under the democratic orderings the chain stayed incomparable to the counter-argument,
defeated it back, and both claims came out UNDECIDED. Level 12 under `weakest_link_democratic`
is where that became fatal: every retry failed on both seeds, and `export-all` wrote two
files short.

These tests pin the outcome, not the directive list. Whatever the reference solution
contains, applying it to the theory the prompt shows has to make -target justified and
target overruled.
"""
from __future__ import annotations

import pytest

from arggym.aspic.api import ASPICVerifier
from arggym.core.export import ALL_ORDERINGS
from arggym.core.scoring import parse_answer
from arggym.tasks import counter_argument as ca

# The cell that issue-level testing is really about. Named rather than searched for: it is
# the cheapest cell on the grid that carries a junction branch on the path to the target
# under a democratic ordering, and it generated nothing before this was fixed.
DEAD_CELL = (12, "weakest_link_democratic")

# Cheap cells for the round-trip property, covering both ways the reference is built:
# level 3 has no decoy and level 9 is the lowest level that turns one on. Level 6 is left
# out on cost alone, since its democratic seed-0 cell runs for eight seconds.
CHEAP = [(lv, o, s) for lv in (3, 9) for o in ALL_ORDERINGS for s in (0, 1)]


def solves(item) -> bool:
    """Apply the item's own reference answer to its own theory and read the statuses.

    Preferences go in after everything else, the way the engine requires: a preference may
    only name a rule or premise the framework already has.
    """
    ops = parse_answer(item.reference).ops
    plain = [o for o in ops if o.kind not in ("prefer_rule", "prefer_premise")]
    prefs = [o for o in ops if o.kind in ("prefer_rule", "prefer_premise")]
    v = ASPICVerifier.from_operations(list(item.base_ops) + plain + prefs,
                                      ordering=item.ordering)
    return (str(v.status("-" + item.target)) == "JUSTIFIED"
            and str(v.status(item.target)) == "OVERRULED"
            and v.is_consistent())


@pytest.mark.parametrize("level,ordering,seed", CHEAP)
def test_reference_answer_solves_its_own_item(level, ordering, seed):
    item = ca.make_item(level, seed, ordering)
    assert item is not None, f"L{level}/{ordering}/s{seed} generated nothing"
    assert solves(item), f"L{level}/{ordering}/s{seed}: the reference answer does not solve it"


def test_a_rejected_build_now_generates():
    """The cheapest build the fix changes, so the default suite covers it.

    `make_item(6, 0, ...)` walks build seeds 0 through 13. On main every even one of those
    was rejected for want of the branch premise ranking, and the retry loop hid it by
    settling on an odd seed. Seed 4 is the cheapest of the rejected ones: 0.00s to fail on
    main against 2s to build here. Levels 3 and 9 above cannot carry this, because no
    junction branch lands on the path to the target there and the new directives never
    fire.
    """
    item = ca.build(6, 4, "weakest_link_democratic")
    assert item is not None, "a build the branch premise ranking should have rescued"
    assert solves(item)


@pytest.mark.slow
@pytest.mark.parametrize("seed", (0, 1))
@pytest.mark.parametrize("allow_strict", (False, True))
def test_the_democratic_junction_cell_generates(seed, allow_strict):
    """Slow: this cell runs the exact subset search that the extra directives enlarge."""
    level, ordering = DEAD_CELL
    item = ca.make_item(level, seed, ordering, allow_strict=allow_strict)
    assert item is not None, f"L{level}/{ordering}/s{seed} generated nothing"
    assert solves(item)
