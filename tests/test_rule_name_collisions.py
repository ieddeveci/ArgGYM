"""Rule names never collide with atoms, and counter_argument rule names carry no role.

Issue #34: split_atoms_and_rules built atoms as the complement of the rule names, so
the guard `atoms & rules` in the task builders could never fire, and randomize_rule_names
could hand a rule the name of an existing atom. Issue #26: counter_argument named its
chain rules d1..dN and its enrichment rules lx_9xx, so the prefix told a solver which
rules were noise and which one was the decoy.
"""
from __future__ import annotations

import re

import pytest

from arggym.aspic.engine import Operation
from arggym.core.invariants import randomize_rule_names, split_atoms_and_rules
from arggym.core.scoring import score_item
from arggym.tasks import claim_chain, counter_argument, defeat_diagnosis, semantics_query, status_query

LAST_LINK, WEAKEST_LINK = counter_argument.LAST_LINK, counter_argument.WEAKEST_LINK
RULE_POOL = [f"{a}{b}{c}" for a in "cdfghjklmnpqrstvwxz" for b in "aeiouy" for c in "0123456789"]


def _collisions(ops):
    """Independent spelling of the fixed definition, so the test does not trust the library."""
    rules = {o.name for o in ops if o.kind in ("defeasible", "strict") and o.name}
    atoms = set()
    for o in ops:
        for a in (o.antecedents or ()):
            atoms.add(a.lstrip("-"))
        if o.consequent and not (o.consequent.startswith("-") and o.consequent[1:] in rules):
            atoms.add(o.consequent.lstrip("-"))
        if o.content:
            atoms.add(o.content.lstrip("-"))
    return atoms & rules


def test_split_undercut_target_is_not_a_collision():
    ops = [Operation(kind="premise", content="a"),
           Operation(kind="premise", content="b"),
           Operation(kind="defeasible", name="r1", antecedents=("a",), consequent="q"),
           Operation(kind="defeasible", name="r2", antecedents=("b",), consequent="-r1")]
    atoms, rules = split_atoms_and_rules(ops)
    assert atoms == {"a", "b", "q"}
    assert rules == {"r1", "r2"}
    assert not atoms & rules


def test_split_rule_named_after_atom_is_a_collision():
    ops = [Operation(kind="premise", content="a"),
           Operation(kind="defeasible", name="q", antecedents=("a",), consequent="q")]
    atoms, rules = split_atoms_and_rules(ops)
    assert atoms & rules == {"q"}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_randomize_rule_names_never_picks_an_atom(seed):
    free = set(RULE_POOL[:10])
    taken = [n for n in RULE_POOL if n not in free]
    ops = [Operation(kind="premise", content=n) for n in taken]
    ops += [Operation(kind="defeasible", name=f"r{i}", antecedents=(taken[i],), consequent=taken[i + 1])
            for i in range(5)]
    renamed, mapping = randomize_rule_names(ops, seed)
    assert len(mapping) == 5, "rename fell back to the old names"
    names = {o.name for o in renamed if o.kind == "defeasible"}
    assert names <= free, f"rule named after an atom: {sorted(names - free)}"
    assert not _collisions(renamed)


@pytest.mark.parametrize("level", [9, 12])
@pytest.mark.parametrize("ordering", [LAST_LINK, WEAKEST_LINK])
def test_counter_argument_rule_names_carry_no_role(level, ordering):
    it = counter_argument.make_item(level, 0, ordering, allow_strict=False)
    assert it is not None
    names = [o.name for o in it.base_ops if o.kind in ("defeasible", "strict")]
    assert names
    assert not [n for n in names if n.startswith("lx")], names
    assert not [n for n in names if re.fullmatch(r"d\d+", n)], names
    if it.metadata["decoy_present"]:
        shapes = {re.sub(r"\d", "0", re.sub(r"[a-z]", "a", n)) for n in names}
        assert len(shapes) == 1, f"a name shape singles out a rule: {sorted(shapes)}"
    assert not _collisions(it.base_ops)
    ref = score_item(it.reference, counter_argument.as_score_input(it))
    assert ref["score"] == pytest.approx(1.0), f"gold regrades at {ref['score']}: {ref.get('reason')}"


@pytest.mark.parametrize("task", [status_query, claim_chain, defeat_diagnosis, semantics_query])
def test_guarded_tasks_have_no_atom_rule_collision(task):
    it = task.make_item(6, 0, LAST_LINK)
    assert it is not None
    assert not _collisions(it.base_ops)
