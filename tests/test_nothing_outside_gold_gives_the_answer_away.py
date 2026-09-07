"""Hiding `metadata.gold` has to be enough.

`docs/dataset-contract.md` section 2 promises one rule: a harness that does not
show the model `metadata.gold` has not shown it the answer. That is only true if
nothing answer-bearing sits anywhere else, and it was not -- the generator's own
statistics were copied in beside it, and they name the gold status, the label
distribution, and the answer's length.

This is the test that makes the promise checkable rather than remembered.
"""
import json

import pytest

import arggym

# Reference answers are short, and the reference is public, so a substring scan
# over the non-gold metadata is a real check rather than a proxy.
TASKS = arggym.task_names()
ORDERING = "last_link_elitist"

#: The curriculum range, not the five exported levels. What a row publishes beside its
#: gold is a property of the row, and it was checked at level 3 alone (#110) while a spec
#: may name any level in this range and freeze it. One cheap column: `last_link_elitist`
#: at seed 0, since `weakest_link_democratic` is where the generation cost lives (#111).
LEVELS = tuple(range(1, 16))
CELLS = [(task, level) for task in sorted(TASKS) for level in LEVELS]

#: Three tests walk all 180 cells, and two more read the `defeat_diagnosis` and
#: `status_query` columns a second time, so one build serves three readers and four
#: in those two columns. Under xdist they scatter across workers and each rebuilds
#: what it is handed; the cache is what keeps a serial run of this file cheap.
_CACHE: dict = {}


def row_at(task: str, level: int) -> dict:
    if (task, level) not in _CACHE:
        _CACHE[task, level] = arggym.create(task, level=level, ordering=ORDERING,
                                            size=1)[0]
    return _CACHE[task, level]


def outside_gold(entry: dict) -> str:
    """Everything a harness may show a model: the row's metadata without `gold`."""
    return json.dumps({k: v for k, v in entry["metadata"].items() if k != "gold"},
                      sort_keys=True)


@pytest.mark.parametrize("task,level", CELLS)
def test_the_reference_answer_appears_nowhere_outside_gold(task, level):
    entry = row_at(task, level)
    blob = outside_gold(entry)
    for line in entry["reference_answer"].splitlines():
        line = line.strip()
        if len(line) > 8:  # skip fragments too short to be evidence
            assert line not in blob, (
                f"{task} L{level}: reference line is readable outside gold")


@pytest.mark.parametrize("task,level", CELLS)
def test_the_generator_statistics_are_treated_as_gold(task, level):
    # Denied by default rather than filtered by an allowlist: the first key
    # somebody forgot to list would be a silent leak in every published row.
    entry = row_at(task, level)
    assert "stats" not in entry["metadata"]
    # The row key is the item attribute it came from.
    assert entry["metadata"]["gold"]["metadata"] is not None


@pytest.mark.parametrize("level", LEVELS)
def test_the_status_a_diagnosis_must_state_is_not_published(level):
    entry = row_at("defeat_diagnosis", level)
    assert "claim_status" not in outside_gold(entry)
    assert entry["metadata"]["gold"]["metadata"]["claim_status"]


@pytest.mark.parametrize("level", LEVELS)
def test_the_label_distribution_of_a_query_is_not_published(level):
    assert "status_counts" not in outside_gold(row_at("status_query", level))


@pytest.mark.parametrize("task,level", CELLS)
def test_what_is_left_outside_gold_is_still_enough_to_ask_the_question(task, level):
    # The other half of the rule: hiding gold must not hide the item. A harness
    # needs the question, the coordinates, and the theory to display an item.
    entry = row_at(task, level)
    meta = entry["metadata"]
    assert entry["question"]
    for k in ("source_dataset", "seed", "level", "ordering", "checker", "answer_shape"):
        assert k in meta, f"{task} L{level}: {k} is not readable without gold"
