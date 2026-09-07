"""The junction budget has to follow the attacker count, and the metadata has to be true.

Both attack/defence builders computed the budget as `(n if "n" in dir() else 3) * 4`. In
`build_defence_item` a local `n` exists and the guard passes; in `build_mixed_item` the
local is `n_atk`, so the guard was permanently false and the budget was frozen at `3 * 4`
however many attackers the level asked for (#48). Nothing failed: a frozen budget still
generates valid items, it just stops scaling.

A budget is not a count. `build_mixed` places junctions only on the shared stem, so it
keeps `min(budget, shared_depth)` of them; `build_attack` divides an allocation across the
chains, loses the remainder, and then gets whatever junctions the chosen configs build on
their own. So the budget is a ceiling and the count is what the item has. Both are
recorded, and the tests hold them apart.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from arggym.core.curriculum import ATTACK, DEFENCE, MIXED, junctions_for
from arggym.core.spec import ALL_ORDERINGS, SEEDS
from arggym.tasks import attack_defense as ad

#: The curriculum range, not the five `LEVELS` exports. The budget held on the exported
#: grid and broke at 10 and 11, which no test built (#97). A spec naming an off-grid level
#: freezes without complaint, so the invariant has to cover the range the curriculum spans.
#: The generators themselves clamp nothing and build well above 15.
GRID = tuple(range(1, 16))
CELLS = [(lv, o, s) for lv in GRID for o in ALL_ORDERINGS for s in SEEDS]
MODES = [ATTACK, DEFENCE, MIXED]

#: The level `attack_defense` gates junctions at. Written out rather than read off
#: `ad.JUNCTION_LEVEL`: a formula that reads the constant restates whatever the generator
#: currently says and cannot see it move. Pinned below, so changing the gate takes changing
#: this file in the same commit.
GATE = 6

NAME_PROBES = {"dir", "locals", "globals", "vars"}

#: Three tests walk the same 120 cells per mode, so one pass serves all three. Under xdist
#: they scatter across workers and each rebuilds, which is what the suite actually runs;
#: the cache is what keeps a serial run of this file cheap.
_CACHE: dict = {}


def _items(mode):
    if mode not in _CACHE:
        _CACHE[mode] = [(lv, o, s, it) for lv, o, s in CELLS
                        if (it := ad.make_item(lv, s, o, mode=mode)) is not None]
    return _CACHE[mode]


def test_the_junction_gate_is_the_level_the_curriculum_chose():
    """`GATE` is the recorded decision and `JUNCTION_LEVEL` is the code that applies it.

    Moving the flag is a curriculum change -- it takes junctions off `defence` and
    `attack_defense` at every level it passes over -- so the two have to move together.
    """
    assert ad.JUNCTION_LEVEL == GATE, (
        f"the generator gates junctions at {ad.JUNCTION_LEVEL}, this file records {GATE}; "
        "whichever is right, moving one alone changes the curriculum silently")


def test_no_builder_asks_whether_one_of_its_own_locals_exists():
    """`"x" in dir()` reads as a fallback and behaves as a constant. It is never right here."""
    root = pathlib.Path(__file__).resolve().parent.parent
    hits = []
    for path in sorted((root / "arggym").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if not isinstance(node, ast.Compare):
                continue
            for op, cmp in zip(node.ops, node.comparators):
                if (isinstance(op, (ast.In, ast.NotIn)) and isinstance(cmp, ast.Call)
                        and isinstance(cmp.func, ast.Name) and cmp.func.id in NAME_PROBES):
                    hits.append(f"{path.relative_to(root)}:{node.lineno}  {ast.unparse(node)}")
    assert not hits, ("a name-existence probe decides a value; whichever branch it picks, it "
                      "picks the same one forever:\n  " + "\n  ".join(hits))


@pytest.mark.parametrize("mode", [DEFENCE, MIXED])
def test_the_budget_is_the_one_the_attacker_count_implies(mode):
    seen = 0
    for lv, o, s, it in _items(mode):
        seen += 1
        # Both builders gate junctions on the flag and honestly record 0 below it, while
        # `junctions_for` starts paying out at `curriculum.JUNCTION_START`, one level lower.
        # Level 5 is the only level in the gap and the exported grid skips it, so the ungated
        # formula looked right. The divergence is a curriculum decision, argued in the
        # comment on `JUNCTION_LEVEL` itself, so the formula states the gate rather than
        # reading it off the generator.
        want = junctions_for(lv, max(1, it.metadata["n_attackers"] * 4)) if lv >= GATE else 0
        assert it.metadata["junction_budget"] == want, (
            f"L{lv} {o} seed {s}: budget {it.metadata['junction_budget']}, "
            f"{want} implied by {it.metadata['n_attackers']} attackers")
    assert seen, f"no {mode} item generated on the grid"


@pytest.mark.parametrize("mode", MODES)
def test_the_recorded_count_is_the_number_of_junctions_in_the_theory(mode):
    """Counted off `base_ops`, so it cannot drift from the item a reader receives."""
    seen = 0
    for lv, o, s, it in _items(mode):
        seen += 1
        built = sum(1 for op in it.base_ops
                    if op.kind in ("defeasible", "strict") and len(op.antecedents) > 1)
        assert it.metadata["n_junctions"] == built, f"L{lv} {o} seed {s} {mode}"
        assert built <= it.metadata["junction_budget"], (
            f"L{lv} {o} seed {s} {mode}: {built} built against a budget of "
            f"{it.metadata['junction_budget']}")
    assert seen, f"no {mode} item generated"


@pytest.mark.parametrize("mode", MODES)
def test_junctions_still_grow_across_the_grid(mode):
    """The symptom of #48: with the budget frozen, mixed was flat from level 6 upward."""
    by_level = {}
    for lv, o, s, it in _items(mode):
        by_level.setdefault(lv, set()).add(it.metadata["n_junctions"])
    above = sorted(lv for lv in by_level if lv >= 6)
    assert above, f"no {mode} item above level 6"
    assert max(by_level[above[-1]]) > min(by_level[above[0]]), {
        lv: sorted(v) for lv, v in by_level.items()}
