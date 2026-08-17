"""The ArgGYM_v2 level schedule must actually be a ladder.

Levels are recipes, not a scalar, so nothing forces rung N+1 to differ from rung
N or to be harder than it. The 2026-08-11 sweep found three places where it did
not hold, and each was invisible in the scores until someone went looking:

  * `defence` produced byte-identical recipes at L12 and L15 -- a fifteen-level
    ladder with four distinct rungs at the top.
  * `counter_argument_strict` needed MORE directives at L3 (3.5) than at L6
    (3.0), so the first rung was the second-hardest and models scored 0.125
    against 0.600 one rung later.
  * `preference_construction` gated shared conflicts on `level % 3 == 2`, which
    fires at levels 5, 8, 11 and 14 -- and the evaluated grid is 3, 6, 9, 12,
    15. The feature existed and no evaluated item ever contained it, which left
    the efficiency half of that task's score measuring nothing in all 30 cells.

These tests pin the repairs. They check the generator's own difficulty knobs,
not model scores, so they stay meaningful without a GPU.
"""
import sys
from pathlib import Path

import pytest

V2 = Path(__file__).resolve().parent.parent / "ArgGYM_v2"

# The levels the benchmark actually evaluates. A defect that hides between these
# rungs is a defect nobody ever sees, so this is the grid the tests use.
LEVELS = (3, 6, 9, 12, 15)


def _v2_import(name: str):
    """Import a v2 module, with v1's same-named packages evicted first.

    v1 and v2 both ship top-level `core`, `tasks` and `aspic` packages. Whichever
    is imported first wins for the whole process, so a full-suite run would
    otherwise resolve `tasks.attack_defense` against whichever tree a previously
    collected test happened to touch.
    """
    if str(V2) not in sys.path:
        sys.path.insert(0, str(V2))
    for mod in list(sys.modules):
        if mod.split(".")[0] in ("core", "tasks", "aspic", "structures"):
            owner = getattr(sys.modules[mod], "__file__", "") or ""
            if str(V2) not in owner:
                del sys.modules[mod]
    import importlib
    return importlib.import_module(name)


@pytest.fixture(scope="module")
def tasks():
    return {
        "attack_defense": _v2_import("tasks.attack_defense"),
        "counter_argument": _v2_import("tasks.counter_argument"),
        "preference_construction": _v2_import("tasks.preference_construction"),
        "status_query": _v2_import("tasks.status_query"),
        "formalization": _v2_import("tasks.formalization"),
    }


def test_defence_rungs_are_all_distinct(tasks):
    """No two evaluated levels may share a defence recipe."""
    knobs = {L: tasks["attack_defense"]._defence_knobs(L) for L in LEVELS}
    dupes = [(a, b) for i, a in enumerate(LEVELS) for b in LEVELS[i + 1:]
             if knobs[a] == knobs[b]]
    assert not dupes, f"identical defence recipes at {dupes}: {knobs}"


def test_defence_knobs_never_decrease(tasks):
    """Attackers, strict attackers and both depths are difficulty knobs."""
    seqs = list(zip(*(tasks["attack_defense"]._defence_knobs(L) for L in LEVELS)))
    for name, seq in zip(("n_attackers", "n_strict", "support_depth", "attacker_depth"), seqs):
        assert all(seq[i + 1] >= seq[i] for i in range(len(seq) - 1)), f"{name} dips: {seq}"


def test_defence_does_not_unlock_strict_attacks_on_a_counting_rung(tasks):
    """A new *kind* of attack must not arrive on a rung that also adds one.

    Bundling them is what made a single step cost more than the rest of the
    ladder combined: gemma-4-31b-it fell 0.575 to 0.025 across L6->L9 and never
    recovered.
    """
    kn = tasks["attack_defense"]._defence_knobs
    for lo, hi in zip(LEVELS, LEVELS[1:]):
        n_lo, s_lo, _, _ = kn(lo)
        n_hi, s_hi, _, _ = kn(hi)
        assert not (s_hi > s_lo and n_hi > n_lo), (
            f"L{lo}->L{hi} adds a strict attacker ({s_lo}->{s_hi}) and an "
            f"attacker ({n_lo}->{n_hi}) on the same rung")


def test_counter_argument_strict_first_rung_is_the_cheapest(tasks):
    """L3 must not demand more directives than any later rung."""
    ca = tasks["counter_argument"]
    means = []
    for L in LEVELS[:3]:
        mins = [it.min_directives for it in
                (ca.make_item(L, s, "last_link_elitist", allow_strict=True) for s in range(8))
                if it is not None]
        assert mins, f"no counter_argument_strict items built at L{L}"
        means.append(sum(mins) / len(mins))
    assert means[0] <= min(means[1:]) + 1e-9, (
        f"L3 needs {means[0]:.2f} directives, more than a later rung: {means}")


def test_preference_construction_minimum_beats_one_per_goal(tasks):
    """Shared conflicts must make the minimum smaller than the goal count.

    This is the whole basis of the efficiency term: if a correct answer is
    necessarily minimal then `0.5 + 0.5 * efficiency` is pass/fail in disguise,
    which is what the 2026-08-11 sweep measured.
    """
    pc = tasks["preference_construction"]
    items = [it for it in (pc.make_item(15, s, "last_link_elitist") for s in range(10))
             if it is not None]
    assert items, "no preference_construction items built at L15"
    gaps = [len(it.goals) - it.min_directives for it in items]
    assert max(gaps) > 0, (
        f"no item at L15 has a minimum below its goal count (gaps {gaps}); "
        "efficiency cannot drop below 1.0 and the score is pass/fail")


@pytest.mark.parametrize("task", ["status_query", "formalization"])
def test_conjunctive_antecedents_are_generated(tasks, task):
    """`a AND b => c` must actually appear in generated theories.

    The DSL, the parser and the engine have always supported conjunction. No
    generator emitted it, so every item in the benchmark was solvable by a model
    that believes a rule takes exactly one antecedent, and nothing could tell
    that model apart from a correct one.
    """
    import re
    mod = tasks[task]
    pat = re.compile(r"\[(?:defeasible|strict) \w+: \w+ AND \w+ (?:=>|->)")
    seen = 0
    for s in range(8):
        it = mod.make_item(15, s, "last_link_elitist")
        if it is None:
            continue
        text = getattr(it, "theory_text", None) or it.reference
        seen += len(pat.findall(text))
    assert seen > 0, f"{task} generated no conjunctive rule at L15"
