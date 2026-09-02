"""A claim answered twice with different statuses scores as one wrong prediction.

status_query, semantics_query and perturbation collect predictions into a dict keyed by claim,
so an answer that hedged by naming two statuses for the same claim used to be scored on whichever
line came last (issue #42); defeat_diagnosis read only the first `status:` line, so the same hedge
was scored on whichever line came first. A contradicted claim now counts as one prediction that is
never a true positive, so the hedge scores exactly like a single wrong answer in either line order,
and the `wrong` diagnostics name it. Identical duplicate lines are not a contradiction.
"""
from __future__ import annotations

import pytest

from arggym.tasks import defeat_diagnosis as dd
from arggym.tasks import perturbation as pt
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
    said = [ln.split(": ")[1].upper() for ln in hedged if ln.startswith(f"{claim}:")]
    assert r["diagnostics"]["wrong"] == [f"{claim}:said {'/'.join(said)}, is {gold_status}"]


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
    said = [ln[len(prefix):].strip().upper().replace(" ", "_")
            for ln in hedged if ln.startswith(prefix)]
    assert r["diagnostics"]["wrong"] == [f"{prefix} said {'/'.join(said)}, is {gold_status}"]


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


# --- perturbation -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pt_item():
    item = pt.make_item(4, 0)
    assert item is not None
    return item


def _pt_gold_lines(item):
    return [f"{c}: {s.lower()}" for c, s in item.gold.items()]


def test_pt_gold_is_exact(pt_item):
    r = pt.score(_wrap(_pt_gold_lines(pt_item)), pt_item)
    assert r["score"] == 1.0
    assert r["exact_match"] is True
    assert r["diagnostics"]["contradicted"] == []


@pytest.mark.parametrize("contradiction_first", [False, True])
def test_pt_contradiction_scores_as_one_wrong_in_either_order(pt_item, contradiction_first):
    gold_lines = _pt_gold_lines(pt_item)
    claim, gold_status = next(iter(pt_item.gold.items()))
    wrong_line = f"{claim}: {_other_status(gold_status)}"

    one_wrong = [wrong_line if ln.startswith(f"{claim}:") else ln for ln in gold_lines]
    expected = pt.score(_wrap(one_wrong), pt_item)["score"]
    assert expected < 1.0

    hedged = [wrong_line] + gold_lines if contradiction_first else gold_lines + [wrong_line]
    r = pt.score(_wrap(hedged), pt_item)
    assert r["reason"] == "ok"
    assert r["score"] == expected
    assert r["exact_match"] is False
    assert r["diagnostics"]["contradicted"] == [claim]
    assert r["diagnostics"]["wrong_status"] == [claim]
    assert r["diagnostics"]["n_predicted"] == len(pt_item.gold)


def test_pt_identical_duplicate_is_not_a_contradiction(pt_item):
    gold_lines = _pt_gold_lines(pt_item)
    r = pt.score(_wrap(gold_lines + [gold_lines[0]]), pt_item)
    assert r["score"] == 1.0
    assert r["exact_match"] is True
    assert r["diagnostics"]["contradicted"] == []


# --- defeat_diagnosis ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def dd_item():
    item = dd.make_item(4, 0)
    assert item is not None
    return item


def _dd_reference_lines(item):
    lines = item.reference.strip().splitlines()
    assert lines[0] == "[answer]" and lines[-1] == "[/answer]"
    return lines[1:-1]


def _dd_status_line(lines):
    return next(ln for ln in lines if ln.lower().startswith("status"))


def test_dd_reference_is_exact(dd_item):
    r = dd.score(dd_item.reference, dd_item)
    assert r["score"] == 1.0
    assert r["exact_match"] is True
    assert r["diagnostics"]["status_contradicted"] is False


@pytest.mark.parametrize("contradiction_first", [False, True])
def test_dd_hedged_status_scores_as_wrong_status_in_either_order(dd_item, contradiction_first):
    lines = _dd_reference_lines(dd_item)
    status_line = _dd_status_line(lines)
    wrong_line = f"status: {_other_status(dd_item.claim_status)}"

    one_wrong = [wrong_line if ln is status_line else ln for ln in lines]
    expected = dd.score(_wrap(one_wrong), dd_item)["score"]
    assert expected < 1.0

    hedged = [wrong_line] + lines if contradiction_first else lines + [wrong_line]
    r = dd.score(_wrap(hedged), dd_item)
    assert r["reason"] == "ok"
    assert r["score"] == expected
    assert r["status_correct"] is False
    assert r["exact_match"] is False
    assert r["diagnostics"]["status_contradicted"] is True


def test_dd_identical_status_repeat_is_not_a_contradiction(dd_item):
    lines = _dd_reference_lines(dd_item)
    r = dd.score(_wrap(lines + [_dd_status_line(lines)]), dd_item)
    assert r["score"] == 1.0
    assert r["exact_match"] is True
    assert r["diagnostics"]["status_contradicted"] is False
