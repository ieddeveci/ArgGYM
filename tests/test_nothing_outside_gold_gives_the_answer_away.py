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


@pytest.mark.parametrize("task", TASKS)
def test_the_reference_answer_appears_nowhere_outside_gold(task):
    entry = arggym.create(task, level=3, ordering="last_link_elitist", size=1)[0]
    public = {k: v for k, v in entry["metadata"].items() if k != "gold"}
    blob = json.dumps(public, sort_keys=True)
    for line in entry["reference_answer"].splitlines():
        line = line.strip()
        if len(line) > 8:  # skip fragments too short to be evidence
            assert line not in blob, f"{task}: reference line is readable outside gold"


@pytest.mark.parametrize("task", TASKS)
def test_the_generator_statistics_are_treated_as_gold(task):
    # Denied by default rather than filtered by an allowlist: the first key
    # somebody forgot to list would be a silent leak in every published row.
    entry = arggym.create(task, level=3, ordering="last_link_elitist", size=1)[0]
    assert "stats" not in entry["metadata"]
    # The row key is the item attribute it came from.
    assert entry["metadata"]["gold"]["metadata"] is not None


def test_the_status_a_diagnosis_must_state_is_not_published():
    entry = arggym.create("defeat_diagnosis", level=3,
                          ordering="last_link_elitist", size=1)[0]
    public = json.dumps({k: v for k, v in entry["metadata"].items() if k != "gold"})
    assert "claim_status" not in public
    assert entry["metadata"]["gold"]["metadata"]["claim_status"]


def test_the_label_distribution_of_a_query_is_not_published():
    entry = arggym.create("status_query", level=3,
                          ordering="last_link_elitist", size=1)[0]
    public = json.dumps({k: v for k, v in entry["metadata"].items() if k != "gold"})
    assert "status_counts" not in public


@pytest.mark.parametrize("task", TASKS)
def test_what_is_left_outside_gold_is_still_enough_to_ask_the_question(task):
    # The other half of the rule: hiding gold must not hide the item. A harness
    # needs the question, the coordinates, and the theory to display an item.
    entry = arggym.create(task, level=3, ordering="last_link_elitist", size=1)[0]
    meta = entry["metadata"]
    assert entry["question"]
    for k in ("source_dataset", "seed", "level", "ordering", "checker", "answer_shape"):
        assert k in meta, f"{task}: {k} is not readable without gold"
