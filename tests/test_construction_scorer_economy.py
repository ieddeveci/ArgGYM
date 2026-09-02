"""The shared construction scorer drops a bad directive and charges for it (issues #8, #21).

Before the fix a preference the engine rejects (unknown rule, self-preference,
premise preference naming an axiom) raised inside ``from_operations`` and the
whole answer scored 0.0 as ``engine_rejected``; and a directive dropped by
``check_legality`` cost nothing because ``n_used`` counted only kept lines.
Now every written line counts toward economy, the engine judges each
preference on its own, and bloat is judged before the goals.
"""
from __future__ import annotations

import pytest

from arggym.aspic.api import ASPICVerifier
from arggym.aspic.engine import Operation
from arggym.core.scoring import build_framework, score_item
from arggym.tasks.attack_defense import reference_ok

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
    assert r["diagnostics"]["illegal"] == []


@pytest.mark.parametrize("extra, reason", [
    ("[prefer_rule: d99 > d98]", "rejected_preference:d99>d98"),
    ("[prefer_rule: d1 > d1]", "rejected_preference:d1>d1"),
    ("[prefer_rule: d99 > d99]", "rejected_preference:d99>d99"),
    ("[prefer_rule: d1 > x]", "rejected_preference:d1>x"),
    ("[prefer_premise: x > a]", "rejected_preference:x>a"),
    ("[prefer_premise: q > a]", "rejected_preference:q>a"),
    ("[prefer_premise: a > a]", "rejected_preference:a>a"),
])
def test_engine_rejected_preference_is_dropped_and_charged(extra, reason):
    r = score_item(_ans(GOLD, extra), _item())
    assert not r["reason"].startswith("engine_rejected"), r
    assert r["success"] is True
    assert r["diagnostics"]["illegal"] == [reason]
    assert len(r["diagnostics"]["rejected_detail"]) == 1
    assert r["diagnostics"]["rejected_detail"][0]
    assert r["diagnostics"]["n_used"] == 2
    assert r["score"] == pytest.approx(0.75)  # 0.5 + 0.5 * (1/2)


def test_preference_may_name_the_answers_own_rule():
    item = _item(min_directives=2)
    r = score_item(_ans("[defeasible d4: c => -p]", "[prefer_rule: d1 > d4]", GOLD), item)
    assert not r["reason"].startswith("engine_rejected"), r
    assert r["diagnostics"]["illegal"] == []
    assert r["success"] is True


def test_engine_path_matches_from_operations():
    added = [Operation(kind="defeasible", name="d4", antecedents=("c",), consequent="-p"),
             Operation(kind="prefer_rule", stronger="d1", weaker="d4"),
             Operation(kind="prefer_rule", stronger="d1", weaker="d2")]
    fw, rejected, detail = build_framework(BASE, added, "last_link_elitist")
    assert rejected == [] and detail == []
    ours = ASPICVerifier(fw).status_map()
    ref = ASPICVerifier.from_operations(BASE + added, ordering="last_link_elitist").status_map()
    assert ours == ref


def test_illegal_strict_rule_counts_toward_economy():
    r = score_item(_ans(GOLD, "[strict s1: c -> q]"), _item())
    assert r["success"] is True
    assert r["diagnostics"]["illegal"] == ["illegal_strict_rule:s1"]
    assert r["diagnostics"]["n_used"] == 2
    assert r["score"] == pytest.approx(0.75)


@pytest.mark.parametrize("first", ["[prefer_rule: d1 > d2]", "[prefer_rule: d3 > d2]"])
def test_bloat_is_judged_before_the_goals(first):
    # Reviewer's inputs: correct and wrong first line, same two junk lines, minimum 1.
    r = score_item(_ans(first, "[axiom: z]", "[premise: z]"), _item())
    assert r["diagnostics"]["n_used"] == 3
    assert r["reason"] == "bloated:3_used_vs_1_minimum"
    assert r["score"] == 0.0
    assert r["diagnostics"]["goals_met"] == []


def test_bloat_gate_skipped_when_minimum_unknown():
    r = score_item(_ans(GOLD, "[axiom: z]", "[premise: z]"), _item(min_directives=None))
    assert r["reason"] == "success_but_minimum_unknown"
    assert r["score"] == 0.5


@pytest.mark.parametrize("rule", ["[defeasible d1: c => -p]", "[strict d1: c -> -p]"])
def test_duplicate_rule_name_is_dropped(rule):
    item = _item(allow_strict=True)
    r = score_item(_ans(rule, GOLD), item)
    assert r["diagnostics"]["illegal"] == ["duplicate_rule_name:d1"]
    assert r["success"] is True
    assert r["score"] == pytest.approx(0.75)


def test_rule_name_repeated_within_the_answer_is_dropped():
    item = _item(min_directives=3)
    r = score_item(_ans("[defeasible d4: c => -p]", "[defeasible d4: b => q]", GOLD), item)
    assert r["diagnostics"]["illegal"] == ["duplicate_rule_name:d4"]


def test_all_directives_illegal_still_fires():
    r = score_item(_ans("[prefer_rule: d99 > d98]", "[axiom: y]"), _item())
    assert r["score"] == 0.0
    assert r["reason"] == "all_directives_illegal"
    assert r["diagnostics"]["illegal"] == ["illegal_new_axiom:y", "rejected_preference:d99>d98"]
    assert r["diagnostics"]["n_used"] == 2


def test_reference_gate_requires_no_illegal_line():
    assert reference_ok({"score": 1.0, "diagnostics": {"illegal": []}})
    assert not reference_ok({"score": 1.0, "diagnostics": {"illegal": ["rejected_preference:d9>d8"]}})
    assert not reference_ok({"score": 0.9, "diagnostics": {"illegal": []}})
