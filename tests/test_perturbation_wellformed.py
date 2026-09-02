"""No rule of a perturbation item shares a name with an atom or with another rule.

The theory and the perturbation form one namespace: the model reads them in a single prompt,
and NOTATION.md section 1 rules out an atom that shares a name with a rule. The perturbation's
rules used to be named from the atom pool, so a perturbation rule regularly took the name of a
literal, and now and then the name of a rule already in the theory. Every cell below produced a
malformed item before that was fixed.

An undercut is written as a rule concluding a negated rule name (NOTATION.md section 2), so a
negated rule name in a rule position is a rule reference rather than an atom occurrence. A
premise or axiom content is always a literal, negated or not.
"""
from __future__ import annotations

from collections import defaultdict

import pytest

from arggym.core.invariants import split_atoms_and_rules
from arggym.structures.chains import LAST_LINK, WEAKEST_LINK
from arggym.tasks import perturbation

CELLS = [
    (6, 0, LAST_LINK),      # was: perturbation rule kq6, also a literal
    (9, 2, LAST_LINK),      # was: perturbation rule in6, also a literal
    (12, 0, WEAKEST_LINK),  # was: perturbation rules dw9 and hp8, both literals
    (12, 2, WEAKEST_LINK),  # was: lp7 a literal, and cy2 the name of a theory rule too
    (15, 1, LAST_LINK),     # was: perturbation rule bm3, also a literal
]


def _rules_and_atoms(ops):
    bound = defaultdict(set)
    for o in ops:
        if o.kind in ("defeasible", "strict") and o.name:
            bound[o.name].add((tuple(o.antecedents or ()), o.consequent))
    atoms, _ = split_atoms_and_rules(ops)
    return bound, atoms


@pytest.mark.parametrize("level,seed,ordering", CELLS)
def test_rule_names_are_not_atoms(level, seed, ordering):
    item = perturbation.make_item(level, seed, ordering)
    assert item is not None, "generation failed"

    ops = list(item.base_ops) + list(item.pert_ops)
    bound, atoms = _rules_and_atoms(ops)

    clash = sorted(atoms & set(bound))
    assert not clash, f"rule name also used as a literal: {clash}"
    dup = sorted(n for n, v in bound.items() if len(v) > 1)
    assert not dup, f"one name bound to two different rules: {dup}"
