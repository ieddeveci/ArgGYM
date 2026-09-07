"""A construction answer is a list of operations; text is one way to write it.

`docs/dataset-contract.md` section 4. A solver using a JSON schema, a tool call or
constrained decoding submits the operations and never imitates our serialization; a
solver returning text calls `parse` first. Both reach `score_value`, and the score is
the same object either way -- which is the property that makes the seam worth having
rather than two scorers that agree today.

Every level of one cheap column, not level 3 alone (#110): the seam is a property of
an item, and a spec may name any level the curriculum spans.
"""
from __future__ import annotations

import pytest

from arggym.aspic.engine import Operation
from arggym.core.answers import UnparseableAnswer
from arggym.core.scoring import parse, score_item, score_value
from arggym.tasks import attack_defense as ad
from arggym.tasks import counter_argument as ca
from arggym.tasks import preference_construction as pc

SEED, ORDERING = 0, "last_link_elitist"

VARIANTS = {
    "preference_construction": (lambda lv: pc.make_item(lv, SEED, ORDERING),
                                lambda i: i.as_score_input()),
    "counter_argument": (lambda lv: ca.make_item(lv, SEED, ORDERING, allow_strict=False),
                         ca.as_score_input),
    "counter_argument_strict": (lambda lv: ca.make_item(lv, SEED, ORDERING,
                                                        allow_strict=True),
                                ca.as_score_input),
    "attack": (lambda lv: ad.make_item(lv, SEED, ORDERING, mode="attack"),
               lambda i: i.as_score_input()),
    "defence": (lambda lv: ad.make_item(lv, SEED, ORDERING, mode="defence"),
                lambda i: i.as_score_input()),
    "attack_defense": (lambda lv: ad.make_item(lv, SEED, ORDERING, mode="attack_defense"),
                       lambda i: i.as_score_input()),
}

#: `last_link_elitist` at seed 0 builds all six variants at all fifteen levels in
#: seconds, while `weakest_link_democratic` is where the generation cost lives (#111).
LEVELS = tuple(range(1, 16))
CELLS = [(task, level) for task in sorted(VARIANTS) for level in LEVELS]

#: Three tests walk all 90 cells and three more read the `preference_construction`
#: column, so one build serves six. Under xdist they scatter across workers and each
#: rebuilds what it is handed; the cache is what keeps a serial run of this file cheap.
_CACHE: dict = {}


def item_at(task: str, level: int):
    """The item and the mapping `core.scoring` takes, both cached."""
    if (task, level) not in _CACHE:
        make, as_input = VARIANTS[task]
        it = make(level)
        assert it is not None, f"{task}: no item at level {level} seed {SEED}"
        _CACHE[task, level] = (it, as_input(it))
    return _CACHE[task, level]


@pytest.fixture
def cell(task, level):
    return item_at(task, level)


@pytest.mark.parametrize("task,level", CELLS)
def test_the_value_path_and_the_text_path_give_the_same_result(task, cell):
    it, score_input = cell
    from_text = score_item(it.reference, score_input)
    from_value = score_value(parse(it.reference, score_input), score_input)
    assert from_value.as_dict() == from_text.as_dict()


@pytest.mark.parametrize("task,level", CELLS)
def test_a_solver_that_never_writes_text_can_answer(task, cell):
    """No string is constructed anywhere on this path."""
    it, score_input = cell
    ops = parse(it.reference, score_input)
    assert all(isinstance(o, Operation) for o in ops)
    r = score_value(ops, score_input)
    assert r.score == pytest.approx(1.0) and r.success is True


@pytest.mark.parametrize("task,level", CELLS)
def test_an_empty_answer_is_a_value_rather_than_a_parse_failure(task, cell):
    """Submitting nothing is an answer. Only text that cannot be read at all raises."""
    _, score_input = cell
    assert parse("", score_input) == []
    r = score_value([], score_input)
    assert r.score == 0.0 and r.reason == "no_directives" and r.success is False


@pytest.mark.parametrize("level", LEVELS)
def test_text_that_cannot_be_read_raises_rather_than_scoring(level):
    _, score_input = item_at("preference_construction", level)
    with pytest.raises(UnparseableAnswer) as e:
        parse("I do not think anything can be done here.", score_input)
    assert e.value.reason.startswith("unparseable_lines:")
    assert e.value.diagnostics["n_unparseable"] > 0


@pytest.mark.parametrize("level", LEVELS)
def test_score_item_turns_that_back_into_a_row(level):
    """A harness scoring a taskset needs a row, not an exception, for a bad generation."""
    _, score_input = item_at("preference_construction", level)
    r = score_item("I do not think anything can be done here.", score_input)
    assert r.score == 0.0 and r.success is False
    assert r.reason.startswith("unparseable_lines:")
    assert r.diagnostics["unparseable_examples"]


@pytest.mark.parametrize("level", LEVELS)
def test_the_operations_a_solver_submits_are_not_required_to_come_from_our_parser(level):
    """The seam is only real if a hand-built value works."""
    _, score_input = item_at("preference_construction", level)
    mine = [Operation(kind="prefer_rule", stronger="nope1", weaker="nope2")]
    r = score_value(mine, score_input)
    assert r.success is False
    assert r.reason in ("all_directives_illegal", "goal_not_met")
