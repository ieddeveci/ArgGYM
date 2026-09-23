"""`counter_argument`'s stated minimum is the walk's length, under weakest-link too.

The walk rebuts the target from the premise the question builds on, unless the theory
already has a rule for the contrary, then undercuts each rule concluding the target, or
the nearest defeasible rule feeding it where that rule is strict. An undercut defeats
whatever the ordering says, so the walk costs one line per chain under every ordering.

Under weakest-link the generator used to offer only preferences for a defeasible chain,
one per element of it, and recorded that as the proven minimum: 14 on level 6 seed 0
under `weakest_link_democratic`, where the 4-line walk scores 1.0 (#184). The bloat gate
reads the minimum, so a 28-line answer on that row still scored 0.75.

Under last-link, outranking a defeasible chain's final rule costs one line, the same as
undercutting it, so the walk ties the searched reference. The walk enters the candidate bank
after that reference and only a strictly shorter candidate replaces it
(`counter_argument.py`, the `>= bar` test in the bank loop), so the last-link gold keeps its
preferences. The card says so, and the last test below holds it to that.

Read off `base_ops`, which is the theory the prompt prints, so the walk is written the
way an answerer would write it rather than taken from the generator.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List

import pytest

from arggym.core.scoring import score_item
from arggym.tasks import counter_argument as ca

#: Weakest-link cells that build in a few seconds each, across the level bands: one
#: chain count per band, the decoy from level 9, and the level-6 cell #184 was filed on.
CELLS = [
    (1, "weakest_link_elitist", 0),
    (6, "weakest_link_elitist", 0),
    (9, "weakest_link_elitist", 1),
    (15, "weakest_link_elitist", 0),
    (3, "weakest_link_democratic", 0),
    (6, "weakest_link_democratic", 0),
    (9, "weakest_link_democratic", 0),
]
#: The same levels under last-link, where the walk ties the reference rather than beating it.
LAST_LINK_CELLS = [
    (1, "last_link_elitist", 0),
    (6, "last_link_democratic", 1),
    (9, "last_link_elitist", 0),
    (15, "last_link_democratic", 0),
]


def _walk(item) -> List[str]:
    rules = [o for o in item.base_ops if o.kind in ("defeasible", "strict")]
    by_cons = defaultdict(list)
    for r in rules:
        by_cons[r.consequent].append(r)

    def cut(r):
        if r.kind == "defeasible":
            return r
        for a in r.antecedents:
            for feeder in by_cons.get(a, []):
                found = cut(feeder)
                if found is not None:
                    return found
        return None

    names: List[str] = []
    for r in by_cons.get(item.target, []):
        c = cut(r)
        assert c is not None, f"nothing defeasible under {r.name}"
        if c.name not in names:
            names.append(c.name)
    lines = [f"[defeasible z{i}: {item.seed_lit} => -{n}]" for i, n in enumerate(names)]
    if not by_cons.get("-" + item.target):
        lines.insert(0, f"[defeasible w: {item.seed_lit} => -{item.target}]")
    return lines


@pytest.mark.parametrize("level,ordering,seed", CELLS)
def test_the_recorded_minimum_is_the_walk(level, ordering, seed):
    item = ca.make_item(level, seed, ordering)
    assert item is not None
    walk = _walk(item)
    where = f"L{level} {ordering} seed {seed}"
    got = score_item("\n".join(walk), ca.as_score_input(item))
    assert got.score == pytest.approx(1.0), (
        f"{where}: the walk scores {got.score} ({got.reason}):\n" + "\n".join(walk))
    assert item.min_directives == len(walk), (
        f"{where}: the row states a minimum of {item.min_directives} where the "
        f"{len(walk)}-line walk answers it, so the bloat gate is graded against the "
        f"wrong number:\n" + "\n".join(walk))


@pytest.mark.parametrize("level,ordering,seed", LAST_LINK_CELLS)
def test_under_last_link_the_walk_ties_and_the_gold_keeps_its_preferences(level, ordering, seed):
    item = ca.make_item(level, seed, ordering)
    assert item is not None
    walk = _walk(item)
    where = f"L{level} {ordering} seed {seed}"
    assert item.min_directives == len(walk), (
        f"{where}: minimum {item.min_directives} against a {len(walk)}-line walk")
    assert "[prefer_rule:" in item.reference, (
        f"{where}: the last-link gold no longer carries a rule preference, so the tie with "
        f"the walk went the other way and the card's account of the gold is wrong:\n"
        f"{item.reference}")
