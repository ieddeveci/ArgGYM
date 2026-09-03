"""Scorer properties on the fast-grid items: junk, hedges, surplus, case, order."""
from __future__ import annotations

import re

import pytest

from .registry import CHAIN, CONSTRUCTION, DIAGNOSIS, LABEL_MAP, MODES
from .test_wellformed import rule_names

_ANSWER = re.compile(r"\[answer\](.*?)\[/answer\]", re.S)
STATUSES = ("justified", "overruled", "undecided")


def body_lines(reference: str):
    return [l for l in _ANSWER.search(reference).group(1).splitlines() if l.strip()]


def wrap(lines) -> str:
    return "[answer]\n" + "\n".join(lines) + "\n[/answer]"


def flip_status(line: str) -> str:
    """'x: justified' -> 'x: overruled'; the head is kept verbatim."""
    head, _, status = line.rpartition(":")
    new = next(s for s in STATUSES if s != status.strip().lower())
    return f"{head}: {new}"


def test_stray_token_is_rejected(fast_cell, fast_item):
    adapter = MODES[fast_cell.mode]
    lines = body_lines(adapter.reference(fast_item))
    result = adapter.score(wrap(lines + ["banana"]), fast_item)
    assert result["score"] == 0.0, result
    assert result["reason"] == f"{adapter.junk_reason}:1", result["reason"]


@pytest.mark.families(LABEL_MAP, DIAGNOSIS)
def test_contradicted_line_scores_as_one_wrong_line(fast_cell, fast_item):
    adapter = MODES[fast_cell.mode]
    lines = body_lines(adapter.reference(fast_item))
    if adapter.family == DIAGNOSIS:
        idx = next(i for i, l in enumerate(lines) if l.startswith("status:"))
    else:
        idx = 0
    flipped = flip_status(lines[idx])
    wrong = adapter.score(wrap(lines[:idx] + [flipped] + lines[idx + 1:]), fast_item)["score"]
    hedge_after = adapter.score(wrap(lines + [flipped]), fast_item)["score"]
    hedge_before = adapter.score(wrap([flipped] + lines), fast_item)["score"]
    assert hedge_after == pytest.approx(wrong), (hedge_after, wrong)
    assert hedge_before == pytest.approx(wrong), (hedge_before, wrong)


@pytest.mark.families(CONSTRUCTION)
def test_surplus_directive_costs_the_same_whatever_its_kind(fast_cell, fast_item):
    adapter = MODES[fast_cell.mode]
    lines = body_lines(adapter.reference(fast_item))
    rule = rule_names(adapter.theory_ops(fast_item))[0]
    surplus = {"inert_premise": "[premise: zzz9]",
               "unknown_rule_preference": "[prefer_rule: zzz9 > zzz8]",
               "self_preference": f"[prefer_rule: {rule} > {rule}]"}
    scores = {kind: adapter.score(wrap(lines + [directive]), fast_item)["score"]
              for kind, directive in surplus.items()}
    assert 0.0 not in scores.values(), scores
    assert len(set(scores.values())) == 1, scores


@pytest.mark.modes("status_query")
def test_status_query_claim_ids_are_case_insensitive(fast_cell, fast_item):
    adapter = MODES[fast_cell.mode]
    lines = [f"{head.upper()}:{tail}" for head, _, tail
             in (l.partition(":") for l in body_lines(adapter.reference(fast_item)))]
    result = adapter.score(wrap(lines), fast_item)
    assert result["score"] == pytest.approx(1.0), result


@pytest.mark.modes(*[m for m, a in MODES.items() if a.family != CHAIN])
def test_reversed_reference_scores_one(fast_cell, fast_item):
    adapter = MODES[fast_cell.mode]
    lines = body_lines(adapter.reference(fast_item))
    result = adapter.score(wrap(lines[::-1]), fast_item)
    assert result["score"] == pytest.approx(1.0), result


@pytest.mark.families(CHAIN)
def test_claim_chain_reversed_line_is_not_full_credit(fast_cell, fast_item):
    adapter = MODES[fast_cell.mode]
    lines = body_lines(adapter.reference(fast_item))
    result = adapter.score(wrap(lines[::-1]), fast_item)
    assert result["score"] < 1.0, result
