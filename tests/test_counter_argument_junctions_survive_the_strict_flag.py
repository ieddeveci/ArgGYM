"""How much structure a counter_argument theory carries is not the strict schedule's call.

A junction is a rule drawing on more than one line at once, and `counter_argument` splices
one into every chain at each point `_jpts` names. `_jpts` is `{1}` from level 5 to level 14
and `{2}` at 15, and the mid-chain target sits at index 1 at every depth the curriculum
reaches, so the two indices collide from 5 to 14. The damage starts at 6: at level 5
`mid_target` is off, `junction` is None, `tgt_j` falls back to `depth - 1`, and the guard's
own `not last` already excluded that. From 6 up, once #93 moved the strict flag onto the
target-reaching rule, the splice guard `not strict` fired there and every strict chain lost
its junction. Level 12 odd seed went from 5 junctions to 2, `n_rules` 33 to 28, `n_atoms`
44 to 34; level 15 was untouched, because `{2}` and index 1 do not meet. Nothing failed:
the budget was still computed, every cell still built, and both existing junction tests are
attack_defense-only. That is the shape of #48, and this file is what would have caught it.

The count is read off `base_ops` and the budget is recomputed from the schedule, for the
reason `test_junction_budget_tracks_attackers.py` gives: a test that reads the generator's
own number agrees with it by construction. `counter_argument` records neither, and adding
them would move every one of its 80 rows for two keys nothing reads -- `freeze.py` counts
the metadata schema as part of the row, so it would cost a `PROMPT_VERSION` bump to say
what `base_ops` already says.

Three assertions, none of which restates the splice arithmetic:

- the theory delivers no more than the budget buys, and not nothing where it buys some;
- the count does not move with the seed or with `allow_strict`, because how many strict
  rules a theory happens to draw may not decide how much structure it carries;
- the count grows across the grid, which is the #48 symptom stated directly.

**Only the second one catches the regression this file is named for.** Run against the
`not strict` guard, it fails at levels 6, 9 and 12 and the other two stay green: the loss
was partial, so the budget bound still held and the count still grew. All three share a
blind spot, and it is worth knowing before trusting them -- a *uniform* loss, level 12
falling from 5 junctions to 1 across both seeds and both arms, passes every one of them.
What is asserted is that the strict schedule does not decide the count, not that the count
is right.
"""
from __future__ import annotations

import collections

import pytest

from arggym.core.curriculum import junctions_for
from arggym.core.spec import ALL_ORDERINGS, LEVELS, SEEDS
from arggym.tasks import counter_argument as ca

CHEAP = [(lv, o, s) for lv in (3, 6) for o in ALL_ORDERINGS for s in SEEDS]
GRID = [(lv, o, s) for lv in LEVELS for o in ALL_ORDERINGS for s in SEEDS]
DEAR = [c for c in GRID if c not in CHEAP]


def _junctions(item) -> int:
    return sum(1 for o in item.base_ops
               if o.kind in ("defeasible", "strict") and len(o.antecedents) > 1)


def _budget(item) -> int:
    """What the schedule bought, from the two numbers the item already records.

    Independent of the generator's count, not of its call: the second argument is
    reconstructed here, so a generator that started passing a different one would move the
    budget and this with it. It catches a wrong count, which is the defect; it would not
    catch a re-scaled budget.
    """
    return junctions_for(item.level,
                         max(1, item.metadata["n_chains"] * item.metadata["chain_depth"]))


def _the_budget_is_spent(level: int, ordering: str, seed: int) -> None:
    for allow_strict in (False, True):
        it = ca.make_item(level, seed, ordering, allow_strict=allow_strict)
        where = f"L{level} {ordering} seed {seed} allow_strict={allow_strict}"
        assert it is not None, where
        built, budget = _junctions(it), _budget(it)
        # Not equality. The budget is a density figure over an approximated rule count,
        # the splice skips each chain's last rule, and `_jpts` clamps at `depth - 2`, so
        # level 15 buys 8 and fits 6. Under-delivering to zero is the defect.
        assert built <= budget, f"{where}: {built} junctions against a budget of {budget}"
        assert built or not budget, (
            f"{where}: the schedule bought {budget} junctions and the theory has none")


@pytest.mark.parametrize("level,ordering,seed", CHEAP)
def test_the_theory_delivers_the_junctions_the_schedule_bought(level, ordering, seed):
    _the_budget_is_spent(level, ordering, seed)


@pytest.mark.slow
@pytest.mark.parametrize("level,ordering,seed", DEAR)
def test_the_theory_delivers_them_at_every_exported_cell(level, ordering, seed):
    _the_budget_is_spent(level, ordering, seed)


@pytest.mark.slow
@pytest.mark.parametrize("level", LEVELS)
def test_the_junction_count_is_the_same_whatever_the_strict_schedule_does(level):
    """The regression itself: strict chains skipped the splice and shipped thinner.

    Both seeds and both arms, because the strict flags are shuffled per build seed and
    `n_strict` is shared between the arms (#167) -- so a count that
    tracked the flags fans out across exactly these sixteen items, which is how the
    defect looked: seed 0 kept `0 2 3 5 6` across the grid while seed 1 read `0 1 2 2 6`.
    """
    counts = {}
    for ordering in ALL_ORDERINGS:
        for seed in SEEDS:
            for allow_strict in (False, True):
                it = ca.make_item(level, seed, ordering, allow_strict=allow_strict)
                assert it is not None
                counts[(ordering, seed, allow_strict)] = _junctions(it)
    assert len(set(counts.values())) == 1, (
        f"L{level}: the junction count depends on the strict schedule, "
        f"{sorted(set(counts.values()))} across the cell: "
        + ", ".join(f"{o} s{s} strict={a}: {n}" for (o, s, a), n in sorted(counts.items())))


@pytest.mark.slow
def test_the_junctions_grow_across_the_grid():
    """The #48 symptom: a budget that rises while the theory stays flat."""
    by_level = collections.defaultdict(set)
    for level, ordering, seed in GRID:
        it = ca.make_item(level, seed, ordering, allow_strict=True)
        assert it is not None
        by_level[level].add(_junctions(it))
    lo, hi = min(LEVELS), max(LEVELS)
    assert max(by_level[hi]) > max(by_level[lo]), (
        f"junctions do not grow: {dict(sorted((k, sorted(v)) for k, v in by_level.items()))}")
