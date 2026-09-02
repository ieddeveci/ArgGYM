"""The shared construction scorer drops a bad directive and charges for it (issues #8, #21).

Before the fix a preference the engine rejects (unknown rule, self-preference,
premise preference naming an axiom) raised inside ``from_operations`` and the
whole answer scored 0.0 as ``engine_rejected``; and a directive dropped by
``check_legality`` cost nothing because ``n_used`` counted only kept lines.
Now every written directive counts toward economy, and preferences are
validated up front with the engine's own conditions.
"""
from __future__ import annotations

import pytest

from arggym.aspic.engine import Operation
from arggym.core.scoring import score_item

# a, d1 => p ; b, d2 => -p ; c, d3 => q. p wins under d1 > d2. x is an axiom.
BASE = [
    Operation(kind="premise", content="a"),
    Operation(kind="premise", content="b"),
    Operation(kind="premise", content="c"),
    Operation(kind="axiom", content="x"),
    Operation(kind="defeasible", name="d1", antecedents=("a",), consequent="p"),
    Operation(kind="defeasible", name="d2", antecedents=("b",), consequent="-p"),
    Operation(kind="defeasible", name="d3", antecedents=("c",), consequent="q"),
]
GOLD = "[prefer_rule: d1 > d2]"


def _item(**kw):
    d = {"base_ops": BASE, "ordering": "last_link_elitist",
         "goals": [{"claim": "p", "want": "JUSTIFIED"}, {"claim": "-p", "want": "OVERRULED"}],
         "min_directives": 1}
    d.update(kw)
    return d


def _ans(*lines):
    return "[answer]\n" + "\n".join(lines) + "\n[/answer]"


def test_gold_scores_one():
    r = score_item(_ans(GOLD), _item())
    assert r["score"] == pytest.approx(1.0), r


@pytest.mark.parametrize("extra, reason", [
    ("[prefer_rule: d99 > d98]", "unknown_rule_preference:d99>d98"),
    ("[prefer_rule: d1 > d1]", "self_preference:d1"),
    ("[prefer_premise: x > a]", "unknown_premise_preference:x>a"),
    ("[prefer_premise: a > a]", "self_preference:a"),
])
def test_engine_rejectable_preference_is_dropped_and_charged(extra, reason):
    r = score_item(_ans(GOLD, extra), _item())
    assert not r["reason"].startswith("engine_rejected"), r
    assert r["success"] is True
    assert r["diagnostics"]["illegal"] == [reason]
    assert r["diagnostics"]["n_illegal"] == 1
    assert r["diagnostics"]["n_used"] == 2
    assert r["score"] == pytest.approx(0.75)  # 0.5 + 0.5 * (1/2)


def test_unknown_name_wins_over_self_preference_like_the_engine():
    r = score_item(_ans(GOLD, "[prefer_rule: d99 > d99]"), _item())
    assert r["diagnostics"]["illegal"] == ["unknown_rule_preference:d99>d99"]


def test_preference_may_name_the_answers_own_rule():
    item = _item(min_directives=2)
    r = score_item(_ans("[defeasible d4: c => -p]", "[prefer_rule: d1 > d4]", GOLD), item)
    assert not r["reason"].startswith("engine_rejected"), r
    assert r["diagnostics"]["illegal"] == []
    assert r["success"] is True


def test_illegal_strict_rule_counts_toward_economy():
    r = score_item(_ans(GOLD, "[strict s1: c -> q]"), _item())
    assert r["success"] is True
    assert r["diagnostics"]["illegal"] == ["illegal_strict_rule:s1"]
    assert r["diagnostics"]["n_used"] == 2
    assert r["score"] == pytest.approx(0.75)


def test_illegal_lines_can_bloat():
    extras = ["[premise: zzz9]", "[prefer_rule: d98 > d97]"]
    r = score_item(_ans(GOLD, *extras), _item())
    assert r["diagnostics"]["n_used"] == 3
    assert r["reason"].startswith("bloated:3_used_vs_1_minimum")
    assert r["score"] == 0.0


def test_all_directives_illegal_still_fires():
    r = score_item(_ans("[prefer_rule: d99 > d98]", "[axiom: y]"), _item())
    assert r["score"] == 0.0
    assert r["reason"] == "all_directives_illegal"
    assert r["diagnostics"]["n_illegal"] == 2
    assert r["diagnostics"]["n_used"] == 2


def test_no_input_here_reaches_engine_rejected():
    extras = ["[prefer_rule: d99 > d98]", "[prefer_rule: d1 > d1]", "[prefer_premise: x > a]",
              "[prefer_premise: q > a]", "[prefer_rule: d1 > x]"]
    for e in extras:
        r = score_item(_ans(GOLD, e), _item())
        assert not r["reason"].startswith("engine_rejected"), (e, r)
