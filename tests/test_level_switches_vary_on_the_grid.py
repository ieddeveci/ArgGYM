"""A level-indexed switch has to take both values on the levels we actually export.

Twice a condition on the level was constant across the whole exported grid, so the branch
behind it was written, reviewed and never reached an item: #30 and #69. Both were
invisible, because a switch that never fires still generates valid items.

Both were sampling failures. The grid stepped by 3, so `level % 3` was the constant 0 on
it and `level % 3 != 0` picked out ten levels the export never asked for. The grid exports
all fifteen levels now, and that retires the sampling failure outright: a switch keyed on
the level either fires somewhere the export looks or fires at no level at all. The
conditions those two issues shipped all vary over the grid today, and
`test_the_switches_that_shipped_broken_now_reach_an_item` holds the grid to it -- narrow
the grid back to a step that hides them and that test goes red rather than this bug class
coming back unannounced.

What the check catches on a grid this wide is the other half of the same defect: a guard
that fires at no level the curriculum defines, or at every one. `level % 4 == 5` and
`level >= 20 and level % 3 == 0` are dead branches, `level % 1 == 0` is a guard that
guards nothing, and none of them is visible in a generated item either.

The check evaluates the condition rather than pattern-matching it, and it evaluates the
whole guard rather than the innermost comparison, because `level >= 4 and level % 5 == 3`
varies in isolation and fires at no level the branch can be reached at. Two holes remain,
both requiring the level to leave the expression: aliasing it through an intermediate
variable, and offsetting it by a named constant.

GRID is the exported grid and has to stay it: it is what makes the two readings above one
test. Today the grid is the whole curriculum, so "constant on the grid" means "dead";
narrow it and the same line means "never sampled" again.
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
                           "behind it never ships. The grid exports every level, so this "
                           "is a guard that fires at no level at all rather than one the "
                           "export happens not to sample:\n  " + "\n  ".join(collapsed))


#: The conditions #30 and #69 shipped, as they were written. Every one of them fires at
#: some level between 1 and 15 -- that was never the defect -- and the defect was that the
#: five-level grid sampled none of those levels.
SHIPPED_BROKEN = [
    "shared = level >= 5 and (level % 3 == 2)",       # #30 as written: 5, 8, 11, 14
    "want_split = (level % 3 != 0)",                  # #69 as written: all but 3, 6, 9, 12, 15
    "want_split = (lv % 3 != 0)",                     # the same, renamed
    "want_split = ((level - 0) % 6 == 1)",            # 7 and 13
    "x = level >= 4 and level % 5 == 3",              # 8 and 13
]

#: Guards that reach no item on any grid, because they reach no level. These are what the
#: check still catches now that the grid samples nothing away, and they are the cases that
#: keep `_constant_over_the_grid` under test.
DEAD = [
    "want_split = level % 4 == 5",                    # no residue mod 4 is 5
    "x = level >= 20 and level % 3 == 0",             # no level is 20
    "want_split = level % 5 == 0 and level % 5 == 1", # cannot be both
    "want_split = level % 1 == 0",                    # true everywhere, guards nothing
]

#: Guards that fire on part of the grid and not the rest, which is all that is asked.
LIVE = [
    "shared = level >= 9 and (level // 3) % 2 == 1",  # the #30 fix
    "want_split = level % 2 == 0",
]


@pytest.mark.parametrize("src", SHIPPED_BROKEN + DEAD + LIVE)
def test_the_check_reads_the_whole_guard_and_not_a_piece_of_it(src):
    """A guard nobody has seen fail is a guard nobody has tested.

    This half of the meta-test is about `level_conditions`, which is where the subtlety
    is: it has to take `level >= 4 and level % 5 == 3` whole rather than the comparison
    inside it, follow the level through a rename, and see the modulo in `(level // 3) % 2`.
    None of that depends on which levels the grid exports.
    """
    found = level_conditions(ast.parse(src))
    assert len(found) == 1, ast.dump(ast.parse(src))


@pytest.mark.parametrize("src", DEAD)
def test_a_guard_that_reaches_no_level_is_flagged(src):
    node, names = level_conditions(ast.parse(src))[0]
    assert len(_constant_over_the_grid(node, names)) == 1, (
        f"{src} takes one value at every level of the grid and the check missed it")


@pytest.mark.parametrize("src", LIVE)
def test_a_guard_that_fires_on_part_of_the_grid_is_not_flagged(src):
    node, names = level_conditions(ast.parse(src))[0]
    assert len(_constant_over_the_grid(node, names)) == 2, src


@pytest.mark.parametrize("src", SHIPPED_BROKEN)
def test_the_switches_that_shipped_broken_now_reach_an_item(src):
    """The grid's receipt for #30 and #69, and the reason this file no longer owns them.

    These five conditions are why the check above exists, and not one of them collapses on
    today's grid: each fires at some level between 1 and 15, and the grid exports every
    level between 1 and 15. That is the bug class retired rather than guarded -- the miss
    took a grid that skipped levels, and there is no longer one.

    Which is why it is asserted here. Narrowing the grid is the one change that could put
    these back, and it would show up as this test going red on the condition it reopens,
    naming the issue, instead of as a branch someone notices was never reached.
    """
    node, names = level_conditions(ast.parse(src))[0]
    assert len(_constant_over_the_grid(node, names)) == 2, (
        f"{src} is constant on the grid {GRID} again; this is how #30 and #69 shipped "
        f"branches no exported item reached")


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
