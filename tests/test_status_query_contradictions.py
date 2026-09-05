"""A claim answered twice with different statuses scores as one wrong prediction.

status_query, semantics_query and perturbation collect predictions into a dict keyed by claim,
so an answer that hedged by naming two statuses for the same claim used to be scored on whichever
line came last (issue #42); defeat_diagnosis read only the first `status:` line, so the same hedge
was scored on whichever line came first. A contradicted claim now counts as one prediction that is
never a true positive, so the hedge scores exactly like a single wrong answer in either line order,
and the `wrong` diagnostics name it. Identical duplicate lines are not a contradiction. Claim keys
are case-folded, so `DL6: x` and `dl6: y` are the same claim, and an all-uppercase answer scores
full (issue #41).
"""
from __future__ import annotations

import pytest

from arggym.tasks import defeat_diagnosis as dd
from arggym.tasks import perturbation as pt
from arggym.tasks import semantics_query as smq
from arggym.tasks import status_query as sq

STATUSES = ("justified", "overruled", "undecided")


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
    r = sq.score("\n".join(_sq_gold_lines(sq_item)), sq_item)
    assert r.score == 1.0
    assert r.diagnostics["exact_match"] is True
    assert r.diagnostics["contradicted"] == []


@pytest.mark.parametrize("contradiction_first", [False, True])
def test_sq_contradiction_scores_as_one_wrong_in_either_order(sq_item, contradiction_first):
    gold_lines = _sq_gold_lines(sq_item)
    claim, gold_status = next(iter(sq_item.gold.items()))
    wrong_line = f"{claim}: {_other_status(gold_status)}"

    one_wrong = [wrong_line if ln.startswith(f"{claim}:") else ln for ln in gold_lines]
    expected = sq.score("\n".join(one_wrong), sq_item).score
    assert expected < 1.0

    hedged = [wrong_line] + gold_lines if contradiction_first else gold_lines + [wrong_line]
    r = sq.score("\n".join(hedged), sq_item)
    assert r.reason == "ok"
    assert r.score == expected
    assert r.diagnostics["exact_match"] is False
    assert r.diagnostics["contradicted"] == [claim]
    said = [ln.split(": ")[1].upper() for ln in hedged if ln.startswith(f"{claim}:")]
    assert r.diagnostics["wrong"] == [f"{claim}:said {'/'.join(said)}, is {gold_status}"]


def test_sq_identical_duplicate_is_not_a_contradiction(sq_item):
    gold_lines = _sq_gold_lines(sq_item)
    r = sq.score("\n".join(gold_lines + [gold_lines[0]]), sq_item)
    assert r.score == 1.0
    assert r.diagnostics["exact_match"] is True
    assert r.diagnostics["contradicted"] == []


def test_sq_junk_token_still_rejected(sq_item):
    r = sq.score("\n".join(_sq_gold_lines(sq_item) + ["blah"]), sq_item)
    assert r.score == 0.0
    assert r.reason.startswith("unparseable_tokens:")


# --- semantics_query ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def smq_item():
    item = smq.make_item(4, 0)
    assert item is not None
    return item


def _smq_gold_lines(item):
    return [f"{c} under {s}: {v.lower().replace('_', ' ')}" for (c, s), v in item.gold.items()]


def test_smq_gold_is_exact(smq_item):
    r = smq.score("\n".join(_smq_gold_lines(smq_item)), smq_item)
    assert r.score == 1.0
    assert r.diagnostics["exact_match"] is True
    assert r.diagnostics["contradicted"] == []


@pytest.mark.parametrize("contradiction_first", [False, True])
def test_smq_contradiction_scores_as_one_wrong_in_either_order(smq_item, contradiction_first):
    gold_lines = _smq_gold_lines(smq_item)
    (claim, sem), gold_status = next(iter(smq_item.gold.items()))
    prefix = f"{claim} under {sem}:"
    wrong_line = f"{prefix} {_other_status(gold_status)}"

    one_wrong = [wrong_line if ln.startswith(prefix) else ln for ln in gold_lines]
    expected = smq.score("\n".join(one_wrong), smq_item).score
    assert expected < 1.0

    hedged = [wrong_line] + gold_lines if contradiction_first else gold_lines + [wrong_line]
    r = smq.score("\n".join(hedged), smq_item)
    assert r.reason == "ok"
    assert r.score == expected
    assert r.diagnostics["exact_match"] is False
    assert r.diagnostics["contradicted"] == [f"{claim} under {sem}"]
    said = [ln[len(prefix):].strip().upper().replace(" ", "_")
            for ln in hedged if ln.startswith(prefix)]
    assert r.diagnostics["wrong"] == [f"{prefix} said {'/'.join(said)}, is {gold_status}"]


def test_smq_identical_duplicate_is_not_a_contradiction(smq_item):
    gold_lines = _smq_gold_lines(smq_item)
    r = smq.score("\n".join(gold_lines + [gold_lines[0]]), smq_item)
    assert r.score == 1.0
    assert r.diagnostics["exact_match"] is True
    assert r.diagnostics["contradicted"] == []


def test_smq_junk_token_still_rejected(smq_item):
    r = smq.score("\n".join(_smq_gold_lines(smq_item) + ["blah"]), smq_item)
    assert r.score == 0.0
    assert r.reason.startswith("unparseable_tokens:")


# --- perturbation -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pt_item():
    item = pt.make_item(4, 0)
    assert item is not None
    return item


def _pt_gold_lines(item):
    return [f"{c}: {s.lower()}" for c, s in item.gold.items()]


def test_pt_gold_is_exact(pt_item):
    r = pt.score("\n".join(_pt_gold_lines(pt_item)), pt_item)
    assert r.score == 1.0
    assert r.diagnostics["exact_match"] is True
    assert r.diagnostics["contradicted"] == []


@pytest.mark.parametrize("contradiction_first", [False, True])
def test_pt_contradiction_scores_as_one_wrong_in_either_order(pt_item, contradiction_first):
    gold_lines = _pt_gold_lines(pt_item)
    claim, gold_status = next(iter(pt_item.gold.items()))
    wrong_line = f"{claim}: {_other_status(gold_status)}"

    one_wrong = [wrong_line if ln.startswith(f"{claim}:") else ln for ln in gold_lines]
    expected = pt.score("\n".join(one_wrong), pt_item).score
    assert expected < 1.0

    hedged = [wrong_line] + gold_lines if contradiction_first else gold_lines + [wrong_line]
    r = pt.score("\n".join(hedged), pt_item)
    assert r.reason == "ok"
    assert r.score == expected
    assert r.diagnostics["exact_match"] is False
    assert r.diagnostics["contradicted"] == [claim]
    assert r.diagnostics["wrong_status"] == [claim]
    assert r.diagnostics["n_predicted"] == len(pt_item.gold)


def test_pt_identical_duplicate_is_not_a_contradiction(pt_item):
    gold_lines = _pt_gold_lines(pt_item)
    r = pt.score("\n".join(gold_lines + [gold_lines[0]]), pt_item)
    assert r.score == 1.0
    assert r.diagnostics["exact_match"] is True
    assert r.diagnostics["contradicted"] == []


# --- defeat_diagnosis ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def dd_item():
    item = dd.make_item(4, 0)
    assert item is not None
    return item


def _dd_reference_lines(item):
    """The lines of the reference answer."""
    return item.reference.strip().splitlines()


def _dd_status_line(lines):
    return next(ln for ln in lines if ln.lower().startswith("status"))


def test_dd_reference_is_exact(dd_item):
    r = dd.score(dd_item.reference, dd_item)
    assert r.score == 1.0
    assert r.diagnostics["exact_match"] is True
    assert r.diagnostics["status_contradicted"] is False


@pytest.mark.parametrize("contradiction_first", [False, True])
def test_dd_hedged_status_scores_as_wrong_status_in_either_order(dd_item, contradiction_first):
    lines = _dd_reference_lines(dd_item)
    status_line = _dd_status_line(lines)
    wrong_line = f"status: {_other_status(dd_item.claim_status)}"

    one_wrong = [wrong_line if ln is status_line else ln for ln in lines]
    expected = dd.score("\n".join(one_wrong), dd_item).score
    assert expected < 1.0

    hedged = [wrong_line] + lines if contradiction_first else lines + [wrong_line]
    r = dd.score("\n".join(hedged), dd_item)
    assert r.reason == "ok"
    assert r.score == expected
    assert r.diagnostics["status_correct"] is False
    assert r.diagnostics["exact_match"] is False
    assert r.diagnostics["status_contradicted"] is True


def test_dd_identical_status_repeat_is_not_a_contradiction(dd_item):
    lines = _dd_reference_lines(dd_item)
    r = dd.score("\n".join(lines + [_dd_status_line(lines)]), dd_item)
    assert r.score == 1.0
    assert r.diagnostics["exact_match"] is True
    assert r.diagnostics["status_contradicted"] is False


# --- case-variant and normalised keys -----------------------------------------------------


def test_sq_case_variant_hedge_is_a_contradiction(sq_item):
    gold_lines = _sq_gold_lines(sq_item)
    claim, gold_status = next(iter(sq_item.gold.items()))
    one_wrong = [f"{claim}: {_other_status(gold_status)}" if ln.startswith(f"{claim}:") else ln
                 for ln in gold_lines]
    expected = sq.score("\n".join(one_wrong), sq_item).score

    hedge = f"{claim.upper()}: {_other_status(gold_status)}"
    r = sq.score("\n".join(gold_lines + [hedge]), sq_item)
    assert r.score == expected < 1.0
    assert r.diagnostics["contradicted"] == [claim]


def test_sq_all_uppercase_answer_scores_full(sq_item):
    r = sq.score("\n".join([ln.upper() for ln in _sq_gold_lines(sq_item)]), sq_item)
    assert r.score == 1.0
    assert r.diagnostics["exact_match"] is True


def test_smq_case_variant_hedge_is_a_contradiction(smq_item):
    gold_lines = _smq_gold_lines(smq_item)
    (claim, sem), gold_status = next(iter(smq_item.gold.items()))
    prefix = f"{claim} under {sem}:"
    one_wrong = [f"{prefix} {_other_status(gold_status)}" if ln.startswith(prefix) else ln
                 for ln in gold_lines]
    expected = smq.score("\n".join(one_wrong), smq_item).score

    hedge = f"{claim.upper()} under {sem.upper()}: {_other_status(gold_status)}"
    r = smq.score("\n".join(gold_lines + [hedge]), smq_item)
    assert r.score == expected < 1.0
    assert r.diagnostics["contradicted"] == [f"{claim} under {sem}"]


def test_smq_semantics_word_hedge_is_a_contradiction(smq_item):
    gold_lines = _smq_gold_lines(smq_item)
    (claim, sem), gold_status = next(iter(smq_item.gold.items()))
    prefix = f"{claim} under {sem}:"
    one_wrong = [f"{prefix} {_other_status(gold_status)}" if ln.startswith(prefix) else ln
                 for ln in gold_lines]
    expected = smq.score("\n".join(one_wrong), smq_item).score

    hedge = f"{claim} under {sem} semantics: {_other_status(gold_status)}"
    r = smq.score("\n".join(gold_lines + [hedge]), smq_item)
    assert r.reason == "ok"
    assert r.score == expected < 1.0
    assert r.diagnostics["contradicted"] == [f"{claim} under {sem}"]


def test_pt_case_variant_hedge_is_a_contradiction(pt_item):
    gold_lines = _pt_gold_lines(pt_item)
    claim, gold_status = next(iter(pt_item.gold.items()))
    one_wrong = [f"{claim}: {_other_status(gold_status)}" if ln.startswith(f"{claim}:") else ln
                 for ln in gold_lines]
    expected = pt.score("\n".join(one_wrong), pt_item).score

    hedge = f"{claim.upper()}: {_other_status(gold_status)}"
    r = pt.score("\n".join(gold_lines + [hedge]), pt_item)
    assert r.score == expected < 1.0
    d = r.diagnostics
    assert d["contradicted"] == [claim]
    assert d["wrong_status"] == [claim]
    assert d["false_positives"] == []
    assert d["n_lines"] == len(gold_lines) + 1
    assert d["n_predicted"] == len(gold_lines)
    assert d["n_contradicted"] == 1


# --- defeat_diagnosis: kind hedge and status words inside a record --------------------------

KINDS = ("undermine", "undercut", "rebut")


def _dd_first_record(lines):
    return next(ln for ln in lines if ln.startswith("defeated_at"))


@pytest.mark.parametrize("contradiction_first", [False, True])
def test_dd_kind_hedge_scores_as_one_wrong_kind(dd_item, contradiction_first):
    lines = _dd_reference_lines(dd_item)
    record = _dd_first_record(lines)
    kind = record.split("kind: ")[1].strip()
    wrong_record = record.replace(f"kind: {kind}", f"kind: {next(k for k in KINDS if k != kind)}")

    one_wrong = [wrong_record if ln == record else ln for ln in lines]
    expected = dd.score("\n".join(one_wrong), dd_item).score
    assert expected < 1.0

    hedged = ([lines[0], wrong_record] + lines[1:] if contradiction_first
              else lines + [wrong_record])
    r = dd.score("\n".join(hedged), dd_item)
    assert r.reason == "ok"
    assert r.score == expected
    assert r.diagnostics["status_correct"] is True
    assert r.diagnostics["exact_match"] is False
    assert r.diagnostics["kind_contradicted"] is True
    assert r.diagnostics["status_contradicted"] is False


@pytest.mark.parametrize("value", ["justified", "correct"])
def test_dd_status_word_inside_a_record_is_not_a_status_line(dd_item, value):
    lines = _dd_reference_lines(dd_item)
    record = _dd_first_record(lines)
    if value == "correct":
        value = dd_item.claim_status.lower()
    annotated = [f"{ln}; status: {value}" if ln == record else ln for ln in lines]
    r = dd.score("\n".join(annotated), dd_item)
    assert r.score == 1.0
    assert r.diagnostics["exact_match"] is True
    assert r.diagnostics["status_contradicted"] is False
    assert r.diagnostics["kind_contradicted"] is False


# --- early returns carry the contradiction keys ---------------------------------------------


@pytest.mark.parametrize("answer", ["", "blah"])
def test_early_returns_carry_contradicted_key(sq_item, smq_item, pt_item, dd_item, answer):
    for mod, item in ((sq, sq_item), (smq, smq_item), (pt, pt_item)):
        r = mod.score(answer, item)
        assert r.reason != "ok"
        assert r.diagnostics["contradicted"] == []
    r = dd.score(answer, dd_item)
    assert r.score == 0.0
    assert r.diagnostics["status_contradicted"] is False
    assert r.diagnostics["kind_contradicted"] is False
