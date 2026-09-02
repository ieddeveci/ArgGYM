"""Rule names never collide with atoms, and counter_argument rule names carry no role.

Issue #34: split_atoms_and_rules built atoms as the complement of the rule names, so
randomize_rule_names could hand a rule the name of an existing atom. Issue #26:
counter_argument named its chain rules d1..dN and its enrichment rules lx_9xx, so the
prefix told a solver which rules were noise and which one was the decoy.

The library definition of a collision is pinned by the test_split_* cases below,
and every mode is checked against it. The same property over the whole grid, computed
without the library, lives in tests/e2e/test_wellformed.py.
"""
from __future__ import annotations

import pytest

from arggym.aspic.engine import Operation
from arggym.core.invariants import RULE_POOL, randomize_rule_names, split_atoms_and_rules
from arggym.core.scoring import score_item
from arggym.tasks import counter_argument
from tests.e2e.registry import MODES

LAST_LINK, WEAKEST_LINK = counter_argument.LAST_LINK, counter_argument.WEAKEST_LINK


def _collisions(ops):
    atoms, rules = split_atoms_and_rules(ops)
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


def test_split_counts_a_preference_operand_as_an_atom():
    """A prefer_premise operand is the one place an atom appears undeclared."""
    ops = [Operation(kind="premise", content="a"),
           Operation(kind="defeasible", name="r1", antecedents=("a",), consequent="q"),
           Operation(kind="prefer_premise", stronger="zz9", weaker="a")]
    atoms, rules = split_atoms_and_rules(ops)
    assert "zz9" in atoms
    ops.append(Operation(kind="defeasible", name="zz9", antecedents=("a",), consequent="q"))
    atoms, rules = split_atoms_and_rules(ops)
    assert atoms & rules == {"zz9"}


def test_split_counts_an_undeclared_antecedent_as_an_atom():
    ops = [Operation(kind="defeasible", name="r1", antecedents=("ax4",), consequent="q")]
    atoms, _ = split_atoms_and_rules(ops)
    assert "ax4" in atoms


def test_split_ignores_prefer_rule_operands():
    """Those operands are rule names, so counting them would invent a collision."""
    ops = [Operation(kind="premise", content="a"),
           Operation(kind="defeasible", name="r1", antecedents=("a",), consequent="q"),
           Operation(kind="defeasible", name="r2", antecedents=("a",), consequent="q"),
           Operation(kind="prefer_rule", stronger="r1", weaker="r2")]
    atoms, rules = split_atoms_and_rules(ops)
    assert not atoms & rules


def test_randomize_rule_names_raises_when_the_pool_cannot_cover():
    """The old silent fallback kept the original names and broke the guarantee."""
    ops = [Operation(kind="premise", content=n) for n in RULE_POOL]
    ops.append(Operation(kind="defeasible", name="d1",
                         antecedents=(RULE_POOL[0],), consequent=RULE_POOL[1]))
    with pytest.raises(ValueError):
        randomize_rule_names(ops, 0)


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
    assert set(names) <= set(RULE_POOL), f"rule not named from the pool: {sorted(set(names) - set(RULE_POOL))}"
    assert not _collisions(it.base_ops)
    ref = score_item(it.reference, counter_argument.as_score_input(it))
    assert ref["score"] == pytest.approx(1.0), f"gold regrades at {ref['score']}: {ref.get('reason')}"


@pytest.mark.parametrize("mode", sorted(MODES))
def test_no_mode_names_a_rule_after_an_atom(mode):
    """Every mode, not only the ones that used to carry a guard of their own."""
    adapter = MODES[mode]
    item = adapter.make(6, 0, LAST_LINK)
    assert item is not None, "generation failed"
    clash = _collisions(adapter.theory_ops(item))
    assert not clash, f"{mode}: names used as both atom and rule: {sorted(clash)}"
