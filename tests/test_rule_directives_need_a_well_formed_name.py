"""A rule directive needs a name, a space before it, and the arrow its kind uses.

The pattern was `\\s*` with an optional name and either arrow, so `[stricttest: a => b]`
parsed as a strict rule named "test", `[strict: a => b]` was accepted under an invented
name, and a strict rule written with the defeasible arrow was taken as strict anyway.
None of it cost a point (#19). What a legal name looks like is now in the prompt (#20).
"""
from __future__ import annotations

import pytest

from arggym.core.answers import UnparseableAnswer
from arggym.core.prompting import permitted_block
from arggym.core.scoring import parse_answer, score_item
from arggym.core.spec import ALL_ORDERINGS, SEEDS
from arggym.tasks import counter_argument as ca
from arggym.tasks import formalization as fz


def _fz_parse(line):
    """`(operations, unreadable)`, which `parse` no longer returns.

    Text that spells out no answer raises rather than scoring, so the count these
    tests assert on comes from the exception. A caller wanting the count reads it
    from the diagnostics; a caller wanting the answer catches nothing.
    """
    try:
        return fz.parse(line), 0
    except UnparseableAnswer as e:
        return [], e.diagnostics["n_unparseable"]


WELL_FORMED = [
    ("[defeasible n1: a => b]", "defeasible", "n1"),
    ("[strict n1: a -> b]", "strict", "n1"),
    ("[defeasible r_2: a AND b => c]", "defeasible", "r_2"),
]

MALFORMED = [
    ("[stricttest: a => b]", "no space, so the kind swallowed the name"),
    ("[defeasiblex: a => b]", "no space"),
    ("[strict: a -> b]", "no name at all"),
    ("[defeasible: a => b]", "no name at all"),
    ("[strict n1: a => b]", "defeasible arrow on a strict rule"),
    ("[defeasible n1: a -> b]", "strict arrow on a defeasible rule"),
    ("[defeasible 1n: a => b]", "a name that does not start with a letter"),
    ("[defeasible _x: a => b]", "a name that does not start with a letter"),
]


@pytest.mark.parametrize("line,kind,name", WELL_FORMED)
def test_a_well_formed_rule_still_parses(line, kind, name):
    p = parse_answer(line)
    assert p.n_unparseable == 0, p.unparseable_examples
    assert [(o.kind, o.name) for o in p.ops] == [(kind, name)]


@pytest.mark.parametrize("line,why", MALFORMED)
def test_a_malformed_rule_is_counted_as_unparseable(line, why):
    p = parse_answer(line)
    assert p.ops == [], f"{why}: parsed as {[(o.kind, o.name) for o in p.ops]}"
    assert p.n_unparseable == 1, why


def test_a_malformed_rule_costs_the_answer_its_score():
    """The point of #19: it used to cost nothing."""
    it = ca.make_item(3, 0, "last_link_elitist")
    good = score_item(it.reference, ca.as_score_input(it))
    assert good.score == pytest.approx(1.0)
    sneaked = it.reference + "\n[stricttest: a => b]"
    bad = score_item(sneaked, ca.as_score_input(it))
    assert bad.score == 0.0
    assert bad.reason.startswith("unparseable_lines")


@pytest.mark.parametrize("line,kind,name", WELL_FORMED)
def test_formalization_reads_a_rule_the_same_way(line, kind, name):
    """It writes its own parser and its own notation line, and they had drifted apart."""
    ops, bad = _fz_parse(line)
    assert bad == 0
    assert [(o.kind, o.name) for o in ops] == [(kind, name)]


@pytest.mark.parametrize("line,why", MALFORMED)
def test_formalization_rejects_the_same_shapes(line, why):
    ops, bad = _fz_parse(line)
    assert ops == [] and bad == 1, why


def test_the_prompt_says_what_a_legal_name_is():
    for allow_strict in (False, True):
        block = permitted_block(allow_strict)
        assert "starts with a letter" in block
        assert "separated from it by a space" in block
        assert "=> for a defeasible rule and -> for a strict one" in block


@pytest.mark.parametrize("level", (3, 6))
@pytest.mark.parametrize("allow_strict", [False, True])
def test_the_cheap_references_are_well_formed(level, allow_strict):
    """The tightened pattern must not reject the answers the generator writes."""
    for o in ALL_ORDERINGS:
        for s in SEEDS:
            it = ca.make_item(level, s, o, allow_strict=allow_strict)
            assert it is not None
            assert parse_answer(it.reference).n_unparseable == 0, it.reference
            assert score_item(it.reference, ca.as_score_input(it)).score == pytest.approx(1.0)
