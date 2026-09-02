"""A claim answered twice with different statuses scores as one wrong prediction.

Both scorers collect predictions into a dict keyed by claim, so an answer that hedged by naming
two statuses for the same claim used to be scored on whichever line came last (issue #42).
A contradicted claim now counts as one prediction that is never a true positive, so the hedge
scores exactly like a single wrong answer in either line order. Identical duplicate lines are
not a contradiction.
"""
from __future__ import annotations

import pytest

from arggym.tasks import semantics_query as smq
from arggym.tasks import status_query as sq

STATUSES = ("justified", "overruled", "undecided")


def _wrap(lines):
    return "[answer]\n" + "\n".join(lines) + "\n[/answer]"


def _other_status(gold_status: str) -> str:
    return next(s for s in STATUSES if s.upper() != gold_status)


# --- status_query -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sq_item():
    item = sq.make_item(4, 0)
    assert item is not None
    return item


def _sq_gold_lines(item):
    return [f"{c}: {s.lower()}" for c, s in item.gold.items()]


def test_sq_gold_is_exact(sq_item):
    r = sq.score(_wrap(_sq_gold_lines(sq_item)), sq_item)
    assert r["score"] == 1.0
    assert r["exact_match"] is True
    assert r["diagnostics"]["contradicted"] == []


@pytest.mark.parametrize("contradiction_first", [False, True])
def test_sq_contradiction_scores_as_one_wrong_in_either_order(sq_item, contradiction_first):
    gold_lines = _sq_gold_lines(sq_item)
    claim, gold_status = next(iter(sq_item.gold.items()))
    wrong_line = f"{claim}: {_other_status(gold_status)}"

    one_wrong = [wrong_line if ln.startswith(f"{claim}:") else ln for ln in gold_lines]
    expected = sq.score(_wrap(one_wrong), sq_item)["score"]
    assert expected < 1.0

    hedged = [wrong_line] + gold_lines if contradiction_first else gold_lines + [wrong_line]
    r = sq.score(_wrap(hedged), sq_item)
    assert r["reason"] == "ok"
    assert r["score"] == expected
    assert r["exact_match"] is False
    assert r["diagnostics"]["contradicted"] == [claim]


def test_sq_identical_duplicate_is_not_a_contradiction(sq_item):
    gold_lines = _sq_gold_lines(sq_item)
    r = sq.score(_wrap(gold_lines + [gold_lines[0]]), sq_item)
    assert r["score"] == 1.0
    assert r["exact_match"] is True
    assert r["diagnostics"]["contradicted"] == []


def test_sq_junk_token_still_rejected(sq_item):
    r = sq.score(_wrap(_sq_gold_lines(sq_item) + ["blah"]), sq_item)
    assert r["score"] == 0.0
    assert r["reason"].startswith("unparseable_tokens:")


# --- semantics_query ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def smq_item():
    item = smq.make_item(4, 0)
    assert item is not None
    return item


def _smq_gold_lines(item):
    return [f"{c} under {s}: {v.lower().replace('_', ' ')}" for (c, s), v in item.gold.items()]


def test_smq_gold_is_exact(smq_item):
    r = smq.score(_wrap(_smq_gold_lines(smq_item)), smq_item)
    assert r["score"] == 1.0
    assert r["exact_match"] is True
    assert r["diagnostics"]["contradicted"] == []


@pytest.mark.parametrize("contradiction_first", [False, True])
def test_smq_contradiction_scores_as_one_wrong_in_either_order(smq_item, contradiction_first):
    gold_lines = _smq_gold_lines(smq_item)
    (claim, sem), gold_status = next(iter(smq_item.gold.items()))
    prefix = f"{claim} under {sem}:"
    wrong_line = f"{prefix} {_other_status(gold_status)}"

    one_wrong = [wrong_line if ln.startswith(prefix) else ln for ln in gold_lines]
    expected = smq.score(_wrap(one_wrong), smq_item)["score"]
    assert expected < 1.0

    hedged = [wrong_line] + gold_lines if contradiction_first else gold_lines + [wrong_line]
    r = smq.score(_wrap(hedged), smq_item)
    assert r["reason"] == "ok"
    assert r["score"] == expected
    assert r["exact_match"] is False
    assert r["diagnostics"]["contradicted"] == [f"{claim} under {sem}"]


def test_smq_identical_duplicate_is_not_a_contradiction(smq_item):
    gold_lines = _smq_gold_lines(smq_item)
    r = smq.score(_wrap(gold_lines + [gold_lines[0]]), smq_item)
    assert r["score"] == 1.0
    assert r["exact_match"] is True
    assert r["diagnostics"]["contradicted"] == []


def test_smq_junk_token_still_rejected(smq_item):
    r = smq.score(_wrap(_smq_gold_lines(smq_item) + ["blah"]), smq_item)
    assert r["score"] == 0.0
    assert r["reason"].startswith("unparseable_tokens:")
