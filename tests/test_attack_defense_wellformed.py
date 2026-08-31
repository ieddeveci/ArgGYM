"""No literal in an attack_defense theory is declared twice.

An axiom is unattackable and a premise is undermineable, so a literal declared as both
contradicts itself on the face of the prompt; NOTATION.md section 1 rules it out. Each builder
used to draw names from several independent pools over one 1560-string universe, so the pools
collided. Every cell below produced a malformed theory before that was fixed.
"""
from __future__ import annotations

from collections import Counter

import pytest

from arggym.core.curriculum import ATTACK, DEFENCE, MIXED
from arggym.structures.chains import LAST_LINK, WEAKEST_LINK
from arggym.tasks import attack_defense

CELLS = [
    (11, 1, LAST_LINK, ATTACK),      # was: axiom km3 declared twice
    (9, 2, LAST_LINK, DEFENCE),      # was: premise cm5 declared twice
    (3, 2, WEAKEST_LINK, MIXED),     # was: kq3 declared axiom and premise
    (12, 0, LAST_LINK, MIXED),       # was: premises cp8 and lp8 declared twice
]


@pytest.mark.parametrize("level,seed,ordering,mode", CELLS)
def test_no_literal_declared_twice(level, seed, ordering, mode):
    item = attack_defense.make_item(level, seed, ordering, mode=mode)
    assert item is not None, "generation failed"

    axioms = [o.content for o in item.base_ops if o.kind == "axiom"]
    premises = [o.content for o in item.base_ops if o.kind == "premise"]

    both = sorted(set(axioms) & set(premises))
    assert not both, f"declared axiom and premise: {both}"
    assert not [k for k, v in Counter(axioms).items() if v > 1], "axiom declared twice"
    assert not [k for k, v in Counter(premises).items() if v > 1], "premise declared twice"
