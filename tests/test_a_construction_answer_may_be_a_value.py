"""A construction answer is a list of operations; text is one way to write it.

`docs/dataset-contract.md` section 4. A solver using a JSON schema, a tool call or
constrained decoding submits the operations and never imitates our serialization; a
solver returning text calls `parse` first. Both reach `score_value`, and the score is
the same object either way -- which is the property that makes the seam worth having
rather than two scorers that agree today.
"""
from __future__ import annotations

import pytest

from arggym.aspic.engine import Operation
from arggym.core.answers import UnparseableAnswer
from arggym.core.scoring import parse, score_item, score_value
from arggym.tasks import attack_defense as ad
from arggym.tasks import counter_argument as ca
from arggym.tasks import preference_construction as pc

LEVEL, SEED, ORDERING = 3, 0, "last_link_elitist"

VARIANTS = {
    "preference_construction": (lambda: pc.make_item(LEVEL, SEED, ORDERING),
                                lambda i: i.as_score_input()),
    "counter_argument": (lambda: ca.make_item(LEVEL, SEED, ORDERING, allow_strict=False),
                         ca.as_score_input),
    "counter_argument_strict": (lambda: ca.make_item(LEVEL, SEED, ORDERING,
                                                     allow_strict=True),
                                ca.as_score_input),
    "attack": (lambda: ad.make_item(LEVEL, SEED, ORDERING, mode="attack"),
               lambda i: i.as_score_input()),
    "defence": (lambda: ad.make_item(LEVEL, SEED, ORDERING, mode="defence"),
                lambda i: i.as_score_input()),
    "attack_defense": (lambda: ad.make_item(LEVEL, SEED, ORDERING, mode="attack_defense"),
                       lambda i: i.as_score_input()),
}


@pytest.fixture(scope="module")
def items():
    out = {}
    for task, (make, as_input) in VARIANTS.items():
        it = make()
        assert it is not None, f"{task}: no item at level {LEVEL} seed {SEED}"
        out[task] = (it, as_input(it))
    return out


@pytest.mark.parametrize("task", sorted(VARIANTS))
def test_the_value_path_and_the_text_path_give_the_same_result(task, items):
    it, score_input = items[task]
    from_text = score_item(it.reference, score_input)
    from_value = score_value(parse(it.reference, score_input), score_input)
    assert from_value.as_dict() == from_text.as_dict()


@pytest.mark.parametrize("task", sorted(VARIANTS))
def test_a_solver_that_never_writes_text_can_answer(task, items):
    """No string is constructed anywhere on this path."""
    it, score_input = items[task]
    ops = parse(it.reference, score_input)
    assert all(isinstance(o, Operation) for o in ops)
    r = score_value(ops, score_input)
    assert r.score == pytest.approx(1.0) and r.success is True


@pytest.mark.parametrize("task", sorted(VARIANTS))
def test_an_empty_answer_is_a_value_rather_than_a_parse_failure(task, items):
    """Submitting nothing is an answer. Only text that cannot be read at all raises."""
    _, score_input = items[task]
    assert parse("", score_input) == []
    r = score_value([], score_input)
    assert r.score == 0.0 and r.reason == "no_directives" and r.success is False


def test_text_that_cannot_be_read_raises_rather_than_scoring(items):
    _, score_input = items["preference_construction"]
    with pytest.raises(UnparseableAnswer) as e:
        parse("I do not think anything can be done here.", score_input)
    assert e.value.reason.startswith("unparseable_lines:")
    assert e.value.diagnostics["n_unparseable"] > 0


def test_score_item_turns_that_back_into_a_row(items):
    """A harness scoring a taskset needs a row, not an exception, for a bad generation."""
    _, score_input = items["preference_construction"]
    r = score_item("I do not think anything can be done here.", score_input)
    assert r.score == 0.0 and r.success is False
    assert r.reason.startswith("unparseable_lines:")
    assert r.diagnostics["unparseable_examples"]


def test_the_operations_a_solver_submits_are_not_required_to_come_from_our_parser(items):
    """The seam is only real if a hand-built value works."""
    it, score_input = items["preference_construction"]
    mine = [Operation(kind="prefer_rule", stronger="nope1", weaker="nope2")]
    r = score_value(mine, score_input)
    assert r.success is False
    assert r.reason in ("all_directives_illegal", "goal_not_met")
