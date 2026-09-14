"""A label-map answer is a claim-to-status mapping; text is one way to write it.

`docs/dataset-contract.md` section 4. A solver using a JSON schema, a tool call or
constrained decoding submits the mapping and never imitates our line format; a solver
returning text calls `parse` first.

What establishes the seam is the three tests that build the mapping out of `item.gold`
and score it against the text path. A test comparing `score(text)` against
`score_value(parse(text))` used to open this file; `score` *is* that expression
(`arggym/tasks/status_query.py`), so it compared an expression with itself and held
whatever `score_value` did (#122). It is gone rather than repaired, because the three
below are the repair.

Four tests here are level-invariant by construction, reading a fixed prose string, an
empty answer or an unrecognizable mapping, so the case count is not a count of distinct
checks. They ride on builds the swept tests have already paid for.

One wrinkle only text has: an answer can label the same claim twice with different
statuses. The scorer counts such a claim as one prediction that is never correct and
reports it under `contradicted`, so `score_value` takes a sequence of statuses as well
as a single one, and `parse` produces a sequence only where the text said several things.
"""
from __future__ import annotations

import pytest

from arggym.core.answers import ScoreResult, UnparseableAnswer
from arggym.tasks import perturbation as pt
from arggym.tasks import semantics_query as smq
from arggym.tasks import status_query as sq

SEED, ORDERING = 0, "last_link_elitist"
STATUSES = ("JUSTIFIED", "OVERRULED", "UNDECIDED")
PROSE = "The theory is intricate and I would rather explain my reasoning than label anything."

MODULES = {"status_query": sq, "semantics_query": smq, "perturbation": pt}

#: The whole curriculum range, not the five exported levels. The value-versus-text seam
#: is a property of an item, and it was asserted at level 3 alone (#110) -- a spec may
#: name any level in this range and freeze it. One seed and one ordering, because those
#: two axes are `tests/e2e`'s and `weakest_link_democratic` is where the generation cost
#: lives (#111).
LEVELS = tuple(range(1, 16))
CELLS = [(task, level) for task in sorted(MODULES) for level in LEVELS]


def line_of(task: str, key, status: str) -> str:
    """The one line the prompt asks for, for one claim of one task."""
    written = status.lower().replace("_", " ")
    if task == "semantics_query":
        claim, semantics = key
        return f"{claim} under {semantics}: {written}"
    return f"{key}: {written}"


def text_of(task: str, value) -> str:
    return "\n".join(line_of(task, k, s) for k, s in value.items())


def other_status(status: str) -> str:
    return next(s for s in STATUSES if s != status)


#: Nine tests walk the same 45 cells and a tenth walks the `perturbation` column, so
#: one build serves all ten. Under xdist they scatter across workers and each rebuilds
#: what it is handed; the cache is what keeps a serial run of this file cheap.
_CACHE: dict = {}


def item_at(task: str, level: int):
    if (task, level) not in _CACHE:
        it = MODULES[task].make_item(level, SEED, ORDERING)
        assert it is not None, f"{task}: no item at level {level} seed {SEED}"
        _CACHE[task, level] = it
    return _CACHE[task, level]


@pytest.fixture
def item(task, level):
    return item_at(task, level)


@pytest.mark.parametrize("task,level", CELLS)
def test_a_solver_may_write_one_status_per_claim_as_a_plain_string(task, item):
    """The shape a mapping actually has: no sequence anywhere, and no string built."""
    module = MODULES[task]
    value = dict(item.gold)
    assert all(isinstance(s, str) for s in value.values())
    result = module.score_value(value, item)
    assert isinstance(result, ScoreResult)
    assert result.score == pytest.approx(1.0) and result.success is True
    assert result.as_dict() == module.score(text_of(task, item.gold), item).as_dict()


@pytest.mark.parametrize("task,level", CELLS)
def test_parse_reads_the_reference_into_the_gold_mapping(task, item):
    module = MODULES[task]
    assert module.parse(text_of(task, item.gold), item) == {k: [s] for k, s in item.gold.items()}


@pytest.mark.parametrize("task,level", CELLS)
def test_one_wrong_label_costs_the_same_as_a_value_as_it_does_as_text(task, item):
    module = MODULES[task]
    key, gold_status = next(iter(item.gold.items()))
    value = {**item.gold, key: other_status(gold_status)}
    from_text = module.score(text_of(task, value), item)
    assert 0.0 < from_text.score < 1.0
    assert module.score_value(value, item).as_dict() == from_text.as_dict()


@pytest.mark.parametrize("task,level", CELLS)
def test_a_claim_labelled_twice_arrives_as_a_sequence(task, item):
    """The one shape only text can produce, and the one the mapping still has to carry."""
    module = MODULES[task]
    key, gold_status = next(iter(item.gold.items()))
    hedged = text_of(task, item.gold) + "\n" + line_of(task, key, other_status(gold_status))
    parsed = module.parse(hedged, item)
    assert parsed[key] == [gold_status, other_status(gold_status)]

    value = {**item.gold, key: (gold_status, other_status(gold_status))}
    from_text = module.score(hedged, item)
    assert module.score_value(value, item).as_dict() == from_text.as_dict()
    assert from_text.diagnostics["contradicted"]


@pytest.mark.parametrize("task,level", CELLS)
def test_a_claim_labelled_twice_the_same_way_is_one_prediction(task, item):
    """Folding a repeat is a scoring decision, so `parse` keeps it and `score_value` folds it."""
    module = MODULES[task]
    key, gold_status = next(iter(item.gold.items()))
    repeated = text_of(task, item.gold) + "\n" + line_of(task, key, gold_status)
    assert module.parse(repeated, item)[key] == [gold_status, gold_status]
    result = module.score(repeated, item)
    assert result.score == pytest.approx(1.0) and result.success is True
    assert result.diagnostics["contradicted"] == []


@pytest.mark.parametrize("task,level", CELLS)
def test_prose_raises_rather_than_scoring_zero(task, item):
    module = MODULES[task]
    with pytest.raises(UnparseableAnswer) as raised:
        module.parse(PROSE, item)
    assert raised.value.reason.startswith("unparseable_")


@pytest.mark.parametrize("task,level", CELLS)
def test_score_turns_that_back_into_a_row(task, item):
    """A harness scoring a whole taskset needs a row, not an exception."""
    module = MODULES[task]
    with pytest.raises(UnparseableAnswer) as raised:
        module.parse(PROSE, item)
    result = module.score(PROSE, item)
    assert result.score == 0.0 and result.success is False
    assert result.reason == raised.value.reason
    assert result.diagnostics.items() >= raised.value.diagnostics.items()
    assert result.diagnostics["contradicted"] == []


@pytest.mark.parametrize("task,level", CELLS)
def test_an_answer_that_names_no_claim_raises(task, item):
    module = MODULES[task]
    with pytest.raises(UnparseableAnswer):
        module.parse("", item)
    assert module.score("", item).score == 0.0


@pytest.mark.parametrize("task,level", CELLS)
def test_score_value_scores_a_mapping_it_cannot_recognize_rather_than_raising(task, item):
    """A solver that submits a value has done its own parsing, and its failures are its own."""
    module = MODULES[task]
    key = ("zzz9", "grounded") if task == "semantics_query" else "zzz9"
    result = module.score_value({key: "PERHAPS"}, item)
    assert result.score == 0.0 and result.success is False
    assert result.reason == "ok"


@pytest.mark.parametrize("level", LEVELS)
def test_perturbation_reads_none_as_the_empty_mapping(level):
    """`none` says no claim changed status, which is what an empty mapping says."""
    item = item_at("perturbation", level)
    assert pt.parse("none", item) == {}
    assert pt.score_value({}, item).as_dict() == pt.score("none", item).as_dict()
    assert pt.score("none", item).reason == "predicted_none"
