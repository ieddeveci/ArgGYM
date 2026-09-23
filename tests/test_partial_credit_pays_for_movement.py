"""Partial credit on a construction task pays for what the answer moved (#188).

The scorer used to read partial credit off the final theory alone. A goal that
started `UNDECIDED` and was asked to become `JUSTIFIED` or `OVERRULED` earned 0.4 of
its share when the answer left it where it was; a goal already at its wanted status
before the answer counted as met; a subgoal that was never `JUSTIFIED` counted as
defeated. So a legal answer that changed nothing, such as restating a preference the
theory already printed, scored up to 0.23 on `preference_construction`, `defence` and
`attack_defense`, while the empty answer and the reported floor sat at 0.0.

Each goal's and subgoal's starting status now comes from the engine on the base
theory. The hand-built tests take the branches one at a time on theories small
enough to read, including the other direction: an answer that breaks a goal which
started met earns nothing for it. The sweep checks the no-op on generated rows,
where the reference still scores 1.0.
"""
from __future__ import annotations

import pytest

from arggym.core import rows
from arggym.core.dataset import TaskDataset
from arggym.core.scoring import parse, score_item, subgoals_from

# Starting statuses under last_link_elitist, from the engine: a, p, d, e and g are
# JUSTIFIED; s and c are UNDECIDED, each with one rule for it and one against it and
# no preference between them. g has two supports, r3 through s and r4 from a.
THEORY = """[premise: a]
[premise: p]
[defeasible r1: a => s]
[defeasible r2: p => -s]
[defeasible r3: s => g]
[defeasible r4: a => g]
[defeasible r5: a => d]
[defeasible r6: p => e]
[defeasible r7: a => c]
[defeasible r8: p => -c]"""
BASE = parse(THEORY, {})

#: A legal preference between two rules that attack nothing: it changes no status.
NO_OP = "[prefer_rule: r5 > r6]"


def _item(goals, subgoals=False):
    item = {"base_ops": BASE, "ordering": "last_link_elitist", "goals": goals,
            "min_directives": 1}
    if subgoals:
        item["subgoals"] = subgoals_from(goals, BASE)
    return item


def test_a_goal_that_started_undecided_earns_nothing_for_staying_there():
    # c starts UNDECIDED and should be OVERRULED; a starts JUSTIFIED, as wanted.
    item = _item([{"claim": "c", "want": "OVERRULED"}, {"claim": "a", "want": "JUSTIFIED"}])
    r = score_item(NO_OP, item)
    assert r.reason == "goal_not_met"
    # Was 0.175: 0.4 for c left UNDECIDED plus 1.0 for a, which the answer never touched.
    assert r.score == 0.0
    assert r.diagnostics["deadlock_not_defeat"] == 0, "the answer created no deadlock"


def test_moving_a_goal_into_undecided_keeps_its_share():
    # d starts JUSTIFIED and should be OVERRULED. A bare rebuttal makes it UNDECIDED,
    # which is movement toward the goal. a, already met, leaves the average, so the
    # 0.4 is d's alone: 0.25 * 0.4.
    item = _item([{"claim": "d", "want": "OVERRULED"}, {"claim": "a", "want": "JUSTIFIED"}])
    r = score_item("[defeasible zz: p => -d]", item)
    assert [g["got"] for g in r.diagnostics["goals_met"]] == ["UNDECIDED", "JUSTIFIED"]
    assert r.score == 0.1
    assert r.diagnostics["deadlock_not_defeat"] == 1


def test_a_subgoal_that_never_stood_is_not_counted_as_defeated():
    # g should be OVERRULED; its supports are s and a. s starts UNDECIDED, so the
    # answer cannot be credited for s not being JUSTIFIED at the end.
    item = _item([{"claim": "g", "want": "OVERRULED"}], subgoals=True)
    assert item["subgoals"] == ["s", "a"]
    r = score_item(NO_OP, item)
    assert r.diagnostics["subgoals_defeated"] == "0/1"
    # Was 0.0375: 0.3 * (1 of 2 subgoals "defeated") * 0.25.
    assert r.score == 0.0


def test_a_subgoal_the_answer_pushed_to_justified_counts_against_it():
    # s starts UNDECIDED. Preferring r1 over r2 settles it JUSTIFIED, which
    # strengthens a support of g, the claim to overrule: counted, and not defeated.
    item = _item([{"claim": "g", "want": "OVERRULED"}], subgoals=True)
    r = score_item("[prefer_rule: r1 > r2]", item)
    assert r.diagnostics["subgoals_defeated"] == "0/2"
    assert r.score == 0.0


def test_defeating_one_of_two_supports_earns_subgoal_credit():
    # g stands on two premises through two rules. Rebutting a leaves g JUSTIFIED
    # through q, so the goal is unmoved but half its supports are defeated:
    # 0.25 * 0.3 * 1/2.
    base = parse("[premise: a] [premise: q] [defeasible r1: a => g] "
                 "[defeasible r2: q => g]", {})
    goals = [{"claim": "g", "want": "OVERRULED"}]
    item = {"base_ops": base, "ordering": "last_link_elitist", "goals": goals,
            "min_directives": 1, "subgoals": subgoals_from(goals, base)}
    r = score_item("[premise: -a]", item)
    assert r.diagnostics["goals_met"][0]["got"] == "JUSTIFIED"
    assert r.diagnostics["subgoals_defeated"] == "1/2"
    assert r.score == 0.0375


BREAK = parse("[premise: a] [premise: p] [defeasible r1: a => d]", {})


@pytest.mark.parametrize("answer,score", [
    # d is overruled, as wanted, and a, met before the answer, is overruled too:
    # (1.0 + 0.0) / 2 * 0.25. Leaving a out of the average would pay the full 0.25.
    ("[premise: -a] [prefer_premise: -a > a]", 0.125),
    # d and a both end UNDECIDED. d moved toward its want and earns 0.4; a moved
    # away from it and earns nothing, not the 0.4 an UNDECIDED goal otherwise gets.
    ("[premise: -a]", 0.05),
])
def test_breaking_a_goal_that_started_met_is_not_free(answer, score):
    item = {"base_ops": BREAK, "ordering": "last_link_elitist", "min_directives": 2,
            "goals": [{"claim": "d", "want": "OVERRULED"},
                      {"claim": "a", "want": "JUSTIFIED"}]}
    r = score_item(answer, item)
    assert r.reason == "goal_not_met"
    assert r.score == score


TASKS = ("preference_construction", "defence", "attack_defense")
LEVELS = (1, 3, 5, 9)
CELLS = [(t, lv) for t in TASKS for lv in LEVELS]


@pytest.mark.parametrize("task,level", CELLS, ids=[f"{t}-L{lv}" for t, lv in CELLS])
def test_restating_a_preference_the_theory_already_has_scores_zero(task, level):
    row, _ = TaskDataset(task, level, "last_link_elitist", size=1, seed=0).build_at(0)
    pref = next(line for line in row["question"].splitlines()
                if line.startswith("[prefer_"))
    r = rows.score(pref, row)
    assert r.score == 0.0, (pref, r.reason, r.diagnostics.get("goals_met"))
    assert rows.score(row["reference_answer"], row).score == 1.0
