"""A level-indexed switch has to take both values on the levels we actually export.

Twice now a condition on the level has been constant across the whole exported grid, so
the branch behind it was written, reviewed and never reached an item: #30 and #69. Both
were invisible, because a switch that never fires still generates valid items.

The check evaluates the condition rather than pattern-matching it, and it evaluates the
whole guard rather than the innermost comparison, because `level >= 4 and level % 5 == 3`
varies in isolation and fires at no level the branch can be reached at. Two holes remain,
both requiring the level to leave the expression: aliasing it through an intermediate
variable, and offsetting it by a named constant.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from arggym.aspic.api import ASPICVerifier
from arggym.core.spec import ALL_ORDERINGS, LEVELS, SEEDS
from arggym.tasks import status_query as sq

GRID = LEVELS
CELLS = [(lv, o, s) for lv in GRID for o in ALL_ORDERINGS for s in SEEDS]

LEVEL_NAMES = {"level", "lv", "L"}
SRC_DIRS = ("arggym/tasks", "arggym/structures", "arggym/core")


def _sources():
    root = pathlib.Path(__file__).resolve().parent.parent
    for d in SRC_DIRS:
        yield from sorted((root / d).glob("*.py"))


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _has_level_modulo(node) -> bool:
    return any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mod)
               and _names(n.left) & LEVEL_NAMES
               for n in ast.walk(node))


def level_conditions(tree):
    """Outermost boolean tests that turn on a modulo of the level and nothing else.

    Outermost, because an enclosing `level >= 4 and ...` is part of when the branch
    fires. Taking the inner comparison alone is the same mistake the check is about.
    """
    taken = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Compare, ast.BoolOp)):
            continue
        if not _has_level_modulo(node):
            continue
        names = _names(node)
        if not names or not names <= LEVEL_NAMES:
            continue
        if any(node in set(ast.walk(t)) for t in taken):
            continue
        taken.append(node)
    return [(n, sorted(_names(n))) for n in taken]


def _constant_over_the_grid(node, names):
    code = compile(ast.Expression(node), "<check>", "eval")
    return {bool(eval(code, {}, {n: lv for n in names})) for lv in GRID}


def test_no_condition_on_the_level_is_constant_across_the_grid():
    collapsed = []
    for path in _sources():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node, names in level_conditions(tree):
            values = _constant_over_the_grid(node, names)
            if len(values) == 1:
                collapsed.append(f"{path.name}:{node.lineno}  {ast.unparse(node)} "
                                 f"is {values.pop()} at every exported level")
    assert not collapsed, ("a level switch takes one value on the whole grid, so the branch "
                           "behind it never ships:\n  " + "\n  ".join(collapsed))


@pytest.mark.parametrize("src,collapses", [
    ("shared = level >= 5 and (level % 3 == 2)", True),      # #30 as written
    ("want_split = (level % 3 != 0)", True),                 # #69 as written
    ("want_split = (lv % 3 != 0)", True),                    # the same, renamed
    ("want_split = ((level - 0) % 6 == 1)", True),           # varies in residue, never true
    ("x = level >= 4 and level % 5 == 3", True),             # only true at an excluded level
    ("shared = level >= 9 and (level // 3) % 2 == 1", False),  # the #30 fix
    ("want_split = level % 2 == 0", False),
])
def test_the_check_agrees_with_the_cases_it_exists_for(src, collapses):
    """A guard nobody has seen fail is a guard nobody has tested."""
    found = level_conditions(ast.parse(src))
    assert len(found) == 1, ast.dump(ast.parse(src))
    node, names = found[0]
    assert (len(_constant_over_the_grid(node, names)) == 1) is collapses, src


def test_a_comparison_with_no_level_modulo_is_not_flagged():
    assert level_conditions(ast.parse("x = level > 3 % 2")) == []


# --- the construct behind the switch -------------------------------------------------

def _construct(want_split, ordering):
    ops, ridx = [], [0]
    goal = sq._weakest_link_split(ops, iter(f"wl{i}" for i in range(40)), ridx,
                                  want_split=want_split)
    return str(ASPICVerifier.from_operations(ops, ordering=ordering).status(goal))


def test_the_split_is_what_separates_the_two_readings():
    """The whole table, because three of its four rows already held before the fix.

    Split, only one of the two premises is weaker than the single-premise argument:
    enough for the elitist reading, not for the democratic one. Ranked, both readings
    agree. The `False` column is what the two added rule preferences buy.
    """
    got = {(o, ws): _construct(ws, o)
           for o in ("weakest_link_elitist", "weakest_link_democratic")
           for ws in (True, False)}
    assert got == {
        ("weakest_link_elitist", True): "JUSTIFIED",
        ("weakest_link_elitist", False): "JUSTIFIED",
        ("weakest_link_democratic", True): "UNDECIDED",
        ("weakest_link_democratic", False): "JUSTIFIED",
    }, got


def test_both_splits_ship_on_the_grid():
    """Read off the items that ship, not the builds that were attempted and thrown away."""
    got = {"weakest_link_split": set(), "last_link_split": set()}
    for lv, o, s in CELLS:
        it = sq.make_item(lv, s, o)
        assert it is not None, f"L{lv} {o} seed {s} produced no item"
        for k in got:
            got[k].add(it.metadata[k])
    assert {True, False} <= got["weakest_link_split"], got["weakest_link_split"]
    assert {True, False} <= got["last_link_split"], got["last_link_split"]
