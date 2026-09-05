"""A prediction comes from a line, not from anywhere a status word turns up.

The three label-map scorers matched `claim: status` anywhere in the answer, so
`re.findall` on "My conclusion: overruled" returned `("conclusion", "overruled")`
and a sentence quietly became a claim the model never made. `defeat_diagnosis`
had the same shape one level up: a `survives_because` written under a record that
named no defeater was carried over to the record above it.

Both patterns are now the line the prompt already asks for -- "one line per
claim", "one line per failure point" -- with room for a bullet in front and a
full stop behind, because those are how a model writes a list and neither
changes what the line says.
"""
from __future__ import annotations

import pytest

from arggym.tasks import defeat_diagnosis as dd
from arggym.tasks import perturbation as pt
from arggym.tasks import semantics_query as smq
from arggym.tasks import status_query as sq

LEVEL, ORDERING, SEED = 3, "last_link_elitist", 0


@pytest.fixture(scope="module")
def sq_item():
    it = sq.make_item(LEVEL, SEED, ORDERING)
    assert it is not None
    return it


def gold_lines(item):
    return [f"{c}: {v.lower()}" for c, v in item.gold.items()]


def test_a_sentence_that_ends_in_a_status_is_not_a_prediction(sq_item):
    """The bug this file exists for.

    The claim `conclusion` is not in the theory, so the injected pair was a false
    positive: it lowered precision on an answer whose every real line was right.
    """
    lines = gold_lines(sq_item)
    r = sq.score("\n".join(["My conclusion: overruled"] + lines), sq_item)
    assert "conclusion" not in (r.diagnostics.get("false_positives") or [])
    assert r.diagnostics.get("n_predicted", 0) <= len(sq_item.gold)


def test_a_prose_line_inside_the_fence_is_still_junk(sq_item):
    """Ignoring the sentence is not the same as accepting it.

    Everything between the delimiters is answer, so a sentence there is an
    unreadable line and the answer scores zero. What changed is that it can no
    longer add a claim to the prediction on its way out.
    """
    r = sq.score("\n".join(["My conclusion: overruled"] + gold_lines(sq_item)), sq_item)
    assert r.score == 0.0
    assert r.reason.startswith("unparseable_tokens:")


@pytest.mark.parametrize("decorate, why", [
    (lambda l: f"- {l}", "a bullet"),
    (lambda l: f"* {l}", "an asterisk bullet"),
    (lambda l: f"{l}.", "a full stop"),
])
def test_how_a_model_writes_a_list_still_reads(sq_item, decorate, why):
    lines = [decorate(l) for l in gold_lines(sq_item)]
    assert sq.score("\n".join(lines), sq_item).score == pytest.approx(1.0), why


def test_a_numbered_list_still_reads(sq_item):
    lines = [f"{i + 1}. {l}" for i, l in enumerate(gold_lines(sq_item))]
    assert sq.score("\n".join(lines), sq_item).score == pytest.approx(1.0)


def test_a_negated_claim_keeps_its_minus():
    """`- ao0: overruled` is a bullet; `-ao0: overruled` is a claim about -ao0.

    The bullet has to be followed by a space, and that one space is the whole
    difference. Eat it and every negated literal in `perturbation` becomes a
    prediction about the wrong claim.
    """
    it = pt.make_item(LEVEL, SEED, ORDERING)
    assert it is not None
    assert any(c.startswith("-") for c in it.gold), "no negated claim in this cell"
    assert pt.score(it.reference(), it).score == pytest.approx(1.0)
    bulleted = "\n".join(f"- {l}" for l in it.reference().splitlines())
    assert pt.score(bulleted, it).score == pytest.approx(1.0)


def test_semantics_query_reads_a_whole_line_too():
    it = smq.make_item(LEVEL, SEED, ORDERING)
    assert it is not None
    assert smq.score(it.reference, it).score == pytest.approx(1.0)
    bulleted = "\n".join(f"* {l}" for l in it.reference.splitlines())
    assert smq.score(bulleted, it).score == pytest.approx(1.0)


def test_a_reason_written_under_a_malformed_record_is_not_credited_to_the_one_above():
    """`defeat_diagnosis`'s carry-over, which was stateful and reached backwards.

    The answer below gives its first record no `survives_because` at all, and
    writes that reason two lines later under a record that names no defeater.
    Scoring it as if the first record had carried it credits the model for a
    field it wrote about something else.
    """
    it = dd.make_item(6, SEED, ORDERING)
    assert it is not None
    first = next((d for d in it.diagnoses if d.get("survives_because")), None)
    if first is None:
        pytest.skip("this cell asks for no survival reason")

    lines = [f"status: {it.claim_status.lower()}"]
    for d in it.diagnoses:
        parts = [f"defeated_at: {d['defeated_at']}", f"defeater: {d['defeater']}",
                 f"kind: {d['kind']}"]
        if d.get("survives_because") and d is not first:
            parts.append(f"survives_because: {d['survives_because']}")
        lines.append("; ".join(parts))
        if d is first:
            lines.append("defeated_at: qq9; kind: rebut")
            lines.append(f"survives_because: {first['survives_because']}")

    r = dd.score("\n".join(lines), it)
    total = r.diagnostics["survives_because_total"]
    assert r.diagnostics["survives_because_correct"] == total - 1
    assert r.diagnostics["exact_match"] is False
