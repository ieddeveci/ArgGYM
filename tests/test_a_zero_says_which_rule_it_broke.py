"""A reason string sends a reader somewhere. It should be the right place.

The construction prompt states two ways to score zero after reaching the goals:
an unreadable directive, and leaving the theory inconsistent. The second came
back as `goal_not_met`, which is the one thing that had not happened.
"""
import pytest

from arggym.aspic.engine import Operation as Op
from arggym.core.scoring import score_item

ORDERING = "last_link_elitist"


def _item(base, goals, **kw):
    return {"base_ops": base, "ordering": ORDERING, "goals": goals,
            "min_directives": 1, **kw}


# Both z and -z are derived strictly from axioms, so nothing can defeat either
# and the theory contradicts itself whatever the answer does. An answer may not
# add axioms or strict rules, so this shape has to be built rather than reached.
INCONSISTENT = [
    Op(kind="axiom", content="a"), Op(kind="axiom", content="b"),
    Op(kind="strict", name="s1", antecedents=("a",), consequent="z"),
    Op(kind="strict", name="s2", antecedents=("b",), consequent="-z"),
    Op(kind="premise", content="p"),
    Op(kind="defeasible", name="r1", antecedents=("p",), consequent="q"),
]


def test_meeting_every_goal_in_a_broken_theory_says_so():
    r = score_item("[defeasible n1: p => q2]",
                   _item(INCONSISTENT, [{"claim": "q", "want": "JUSTIFIED"}]))
    assert r.reason == "inconsistent_theory"
    assert r.score == 0.0 and r.success is False
    # The goal really was met; that is what makes the old reason misleading.
    assert r.diagnostics["goals_met"] == [
        {"claim": "q", "want": "JUSTIFIED", "got": "JUSTIFIED"}]


def test_missing_a_goal_still_says_goal_not_met():
    base = [Op(kind="premise", content="p"),
            Op(kind="defeasible", name="r1", antecedents=("p",), consequent="q")]
    r = score_item("[defeasible n1: p => q2]",
                   _item(base, [{"claim": "q", "want": "OVERRULED"}]))
    assert r.reason == "goal_not_met"
    assert r.success is False


@pytest.mark.parametrize("answer,reason", [
    ("", "no_directives"),
    ("I think the answer is probably nothing", "unparseable_tokens"),
    ("[axiom: zz]", "all_directives_illegal"),
])
def test_every_early_zero_names_what_went_wrong_and_is_not_a_success(answer, reason):
    base = [Op(kind="premise", content="p"),
            Op(kind="defeasible", name="r1", antecedents=("p",), consequent="q")]
    r = score_item(answer, _item(base, [{"claim": "q", "want": "OVERRULED"}]))
    assert r.reason.startswith(reason)
    # The bug this guards: success was absent on these paths, and the harness
    # filtered on `is not None`, so format failures left the denominator.
    assert r.success is False
