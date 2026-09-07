"""`semantics_query` has to ask something the grounded extension does not answer.

`status_under` implements grounded, sceptical preferred, credulous preferred, stable
and eager. `SEMANTICS_BY_LEVEL` scheduled two of them, so three reached no item at any
level, and level 3 resolved to grounded alone — the same engine call `status_query`
makes, with the same label set (#36).
"""
from __future__ import annotations

import pytest

from arggym.aspic.api import ASPICVerifier
from arggym.core.spec import ALL_ORDERINGS, LEVELS, SEEDS
from arggym.tasks import semantics_query as sq

GRID = LEVELS
CELLS = [(lv, o, s) for lv in GRID for o in ALL_ORDERINGS for s in SEEDS]

# The schedule is a dict lookup, so asking it about every level costs nothing and a
# level the grid skips today is a level a spec may name tomorrow.
SCHEDULE = tuple(range(1, 16))

IMPLEMENTED = (sq.GROUNDED, sq.SCEPT_PREF, sq.CRED_PREF, sq.STABLE, sq.EAGER)


def test_every_implemented_semantics_reaches_some_level():
    scheduled = {s for lv in SCHEDULE for s in sq.semantics_for(lv)}
    unreachable = sorted(set(IMPLEMENTED) - scheduled)
    assert not unreachable, (f"{unreachable} are implemented in status_under and asked at "
                             f"no level, so the code paths ship dead")


# Strict, so the day #108 is settled these xpass and the marker has to go.
BELOW_THREE = pytest.mark.xfail(strict=True, reason=(
    "#108: SEMANTICS_BY_LEVEL schedules grounded alone below level 3, so the task is "
    "status_query reworded there; whether it should exist there is a curriculum decision"))


@pytest.mark.parametrize("level", [pytest.param(lv, marks=[BELOW_THREE] if lv < 3 else [])
                                   for lv in SCHEDULE])
def test_every_level_asks_past_grounded(level):
    assert set(sq.semantics_for(level)) - {sq.GROUNDED}, (
        f"level {level} resolves to {sq.semantics_for(level)}, which is status_query reworded")


@pytest.mark.parametrize("level,ordering,seed", CELLS)
def test_the_grounded_map_alone_does_not_answer_the_item(level, ordering, seed):
    """The claim behind the task's existence, checked per item rather than per schedule."""
    it = sq.make_item(level, seed, ordering)
    assert it is not None, f"L{level} {ordering} seed {seed} produced no item"
    grounded = ASPICVerifier.from_operations(list(it.base_ops), ordering=ordering).status_map()
    disagree = [(c, s) for (c, s) in it.queries
                if it.gold[(c, s)] != grounded.get(c)]
    assert disagree, ("every queried status equals the grounded status of the same claim, so "
                      "the grounded extension answers the whole item")


@pytest.mark.parametrize("level,ordering,seed", CELLS)
def test_some_claim_is_answered_differently_by_two_semantics(level, ordering, seed):
    it = sq.make_item(level, seed, ordering)
    assert it is not None
    assert it.metadata["n_diverging_claims"] > 0


@pytest.mark.parametrize("level,ordering,seed", CELLS)
def test_the_reference_still_scores_itself(level, ordering, seed):
    it = sq.make_item(level, seed, ordering)
    assert it is not None
    assert sq.score(it.reference, it).score == pytest.approx(1.0)


@pytest.mark.parametrize("level", GRID)
def test_the_stable_sentence_appears_where_stable_is_asked(level):
    asks_stable = sq.STABLE in sq.semantics_for(level)
    for o in ALL_ORDERINGS:
        for s in SEEDS:
            it = sq.make_item(level, s, o)
            assert it is not None
            assert ("no stable extension" in it.prompt) is asks_stable
