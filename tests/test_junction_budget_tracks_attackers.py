"""The junction budget has to follow the attacker count it is derived from.

Both attack/defence builders computed it as `(n if "n" in dir() else 3) * 4`. In
`build_defence_item` a local `n` exists and the guard passes; in `build_mixed_item`
the local is `n_atk`, so the guard was permanently false and the budget was frozen at
`3 * 4` (#48). Nothing failed: a frozen budget still generates valid items, it just
stops scaling.
"""
from __future__ import annotations

import inspect
import pathlib
import re

import pytest

from arggym.core.curriculum import DEFENCE, MIXED, junctions_for
from arggym.core.export import ALL_ORDERINGS, export_task
from arggym.tasks import attack_defense as ad

GRID = inspect.signature(export_task).parameters["levels"].default
SEEDS = inspect.signature(export_task).parameters["seeds"].default

_NAME_GUARD = re.compile(r"\bin\s+(?:dir|locals|globals|vars)\(\)")


def test_no_builder_tests_for_the_existence_of_a_local():
    """`"x" in dir()` reads as a fallback and behaves as a constant. It is never right here."""
    root = pathlib.Path(__file__).resolve().parent.parent
    hits = []
    for d in ("arggym/tasks", "arggym/structures", "arggym/core"):
        for path in sorted((root / d).glob("*.py")):
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                if _NAME_GUARD.search(line):
                    hits.append(f"{path.name}:{lineno}  {line.strip()}")
    assert not hits, ("a name-existence guard decides a curriculum value; whichever branch it "
                      "picks, it picks the same one forever:\n  " + "\n  ".join(hits))


@pytest.mark.parametrize("mode", [DEFENCE, MIXED])
def test_the_recorded_budget_is_the_one_the_attacker_count_implies(mode):
    seen = 0
    for lv in GRID:
        for o in ALL_ORDERINGS:
            for s in SEEDS:
                it = ad.make_item(lv, s, o, mode=mode)
                if it is None:
                    continue
                seen += 1
                want = junctions_for(lv, max(1, it.metadata["n_attackers"] * 4))
                assert it.metadata["n_junctions"] == want, (
                    f"L{lv} {o} seed {s}: {it.metadata['n_junctions']} recorded, "
                    f"{want} implied by {it.metadata['n_attackers']} attackers")
    assert seen, f"no {mode} item generated on the grid"


def test_mixed_junctions_still_grow_at_the_top_of_the_grid():
    """The symptom: frozen at `3 * 4`, the recorded budget was 3 from level 6 upward."""
    got = {}
    for lv in GRID:
        for o in ALL_ORDERINGS:
            for s in SEEDS:
                it = ad.make_item(lv, s, o, mode=MIXED)
                if it is not None:
                    got.setdefault(lv, set()).add(it.metadata["n_junctions"])
    junctioned = sorted(lv for lv in got if lv >= 6)
    assert junctioned, f"no mixed item generated above level 6: {sorted(got)}"
    lo, hi = junctioned[0], junctioned[-1]
    assert max(got[hi]) > min(got[lo]), (
        f"the budget does not grow across the grid: {{lv: sorted(v) for lv, v in got.items()}}")
