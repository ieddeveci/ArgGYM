"""Structural properties of the theory each prompt shows, computed here without the library.

An atom is a name that appears as a premise or axiom, as a rule antecedent, as a positive
consequent, or in a premise preference. A negated consequent whose bare name is a rule is an
undercut target, not an atom (NOTATION.md section 2), so it is left out on purpose.

Known failures are listed in KNOWN_FAILURES by the issue that tracks them. A test calls
pytest.xfail from inside when the property fails for a listed mode, so a cell that happens to
hold simply passes; a strict decorator would XPASS on those cells and turn the suite red.
Remove an entry when its issue closes and the property then holds on the grid.
"""
from __future__ import annotations

import re
from collections import Counter

import pytest

from arggym.core.scoring import _PREF, _PREMISE, _RULE

from .registry import MODES

RULE_KINDS = ("defeasible", "strict")
PREMISE_KINDS = ("premise", "axiom")

# Rule names from the randomized pool are two letters and a digit (invariants.randomize_rule_names).
# Anything else is a fixed prefix plus a counter, and the prefix is what leaks structure.
_POOL_NAME = re.compile(r"^[a-z][a-z]\d$")


def name_class(name: str) -> str:
    """'xi7' -> 'pool'; 'lx_901' -> 'lx_#'; 'd3' -> 'd#'; 'w_2' -> 'w_#'."""
    if _POOL_NAME.match(name):
        return "pool"
    return re.sub(r"\d+", "#", name)


# {property: {reason: modes}}; see the module docstring.
KNOWN_FAILURES = {
    "atom_named_like_a_rule": {},
    "duplicate_declaration": {},
    "rule_names_reveal_structure": {},
}


def xfail_if_known(prop: str, mode: str, detail: str) -> None:
    for reason, modes in KNOWN_FAILURES[prop].items():
        if mode in modes:
            pytest.xfail(f"{reason}: {detail}")


def bare(lit: str) -> str:
    return lit[1:] if lit.startswith("-") else lit


def rule_names(ops):
    return [o.name for o in ops if o.kind in RULE_KINDS]


def atoms(ops):
    out = set()
    for o in ops:
        if o.kind in PREMISE_KINDS:
            out.add(bare(o.content))
        elif o.kind in RULE_KINDS:
            out.update(bare(a) for a in o.antecedents)
            if not o.consequent.startswith("-"):
                out.add(o.consequent)
        elif o.kind == "prefer_premise":
            out.add(bare(o.stronger))
            out.add(bare(o.weaker))
    return out


def test_no_atom_named_like_a_rule(cell, item):
    ops = MODES[cell.mode].theory_ops(item)
    clash = atoms(ops) & set(rule_names(ops))
    detail = f"{cell.short}: names used as both atom and rule: {sorted(clash)}"
    if clash:
        xfail_if_known("atom_named_like_a_rule", cell.mode, detail)
    assert not clash, detail


def test_no_duplicate_declarations(cell, item):
    ops = MODES[cell.mode].theory_ops(item)
    literals = Counter(o.content for o in ops if o.kind in PREMISE_KINDS)
    dup_lits = sorted(l for l, n in literals.items() if n > 1)
    names = Counter(rule_names(ops))
    dup_names = sorted(r for r, n in names.items() if n > 1)
    detail = f"{cell.short}: literal declared twice: {dup_lits}; rule name bound twice: {dup_names}"
    if dup_lits or dup_names:
        xfail_if_known("duplicate_declaration", cell.mode, detail)
    assert not dup_lits and not dup_names, detail


def test_rule_names_do_not_reveal_structure(cell, item):
    ops = MODES[cell.mode].theory_ops(item)
    classes = Counter(name_class(r) for r in rule_names(ops))
    detail = f"{cell.short}: rule-name classes {dict(classes)}"
    if len(classes) != 1:
        xfail_if_known("rule_names_reveal_structure", cell.mode, detail)
    assert len(classes) == 1, detail


def test_every_name_parses_in_an_answer(cell, item):
    ops = MODES[cell.mode].theory_ops(item)
    for r in rule_names(ops):
        assert _RULE.match(f"[defeasible {r}: a => b]"), f"{cell.short}: rule name {r!r}"
        assert _PREF.match(f"[prefer_rule: {r} > {r}]"), f"{cell.short}: rule name {r!r}"
    for a in sorted(atoms(ops)):
        assert _PREMISE.match(f"[premise: {a}]"), f"{cell.short}: atom {a!r}"
        assert _PREMISE.match(f"[premise: -{a}]"), f"{cell.short}: atom {a!r}"
        assert _PREF.match(f"[prefer_premise: {a} > -{a}]"), f"{cell.short}: atom {a!r}"
