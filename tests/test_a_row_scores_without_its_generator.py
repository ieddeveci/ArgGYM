"""A frozen row scores to the number the item it came from would.

`docs/dataset-contract.md` sections 2, 4 and 5. The v1 harness scored a stored
generation by calling `_rebuild(task, level, ordering, seed)` and regenerating
the item, which pins every stored run to one generator commit. A row that
carries everything its scorer reads removes the rebuild.

Scoring the gold is not the test. A `to_dict` that drops a field can still
accept the reference and grade a model's answer wrongly, so every case here is
checked on wrong answers too -- built from the reference by dropping a line,
flipping a label, adding a stray token and reversing the order -- and compared
field for field, `diagnostics` included.
"""
import json
from importlib import import_module

import pytest

from arggym.core import registry, rows
from arggym.core.dataset import TaskDataset
from arggym.core.scoring import score_item

ORDERINGS = ("last_link_elitist", "weakest_link_democratic")
TASKS = registry.task_names()
CELLS = [(t, o) for t in TASKS for o in ORDERINGS]
IDS = [f"{t}-{o}" for t, o in CELLS]

_BUILT = {}


def cell(task, ordering, level=3):
    """One item and its row, built once per cell for the whole module."""
    key = (task, ordering, level)
    if key not in _BUILT:
        ds = TaskDataset(task, level, ordering, size=1, seed=0)
        item = ds.spec.make_item(level, 0, ordering)
        assert item is not None, f"{task} L{level} {ordering} built no item"
        _BUILT[key] = (item, ds.entry(item, 0, 0))
    return _BUILT[key]


def live_score(task, item, text):
    """The score through the item, which is the path the row has to reproduce."""
    spec = registry.get(task)
    mod = import_module(f"arggym.tasks.{spec.module}")
    if spec.scorer != registry.ENGINE:
        return mod.score(text, item)
    payload = (mod.as_score_input(item) if hasattr(mod, "as_score_input")
               else item.as_score_input())
    return score_item(text, payload)


def wrong_answers(reference):
    """Four ways to be wrong, all built from the reference rather than invented."""
    lines = [x for x in reference.splitlines() if x.strip()]
    flipped = (reference.replace("justified", "overruled") if "justified" in reference
               else reference.replace("=>", "->"))
    return {"drop_a_line": "\n".join(lines[1:]),
            "flip_a_label": flipped,
            "stray_token": reference + "\nzz9",
            "reverse_the_order": "\n".join(reversed(lines))}


@pytest.mark.parametrize("task,ordering", CELLS, ids=IDS)
def test_the_reference_scores_the_same_from_the_row_as_from_the_item(task, ordering):
    item, row = cell(task, ordering)
    ref = row["reference_answer"]
    assert rows.score(ref, row) == live_score(task, item, ref)
    assert rows.score(ref, row).score == 1.0


@pytest.mark.parametrize("task,ordering", CELLS, ids=IDS)
def test_a_wrong_answer_scores_the_same_from_the_row_as_from_the_item(task, ordering):
    item, row = cell(task, ordering)
    for label, text in wrong_answers(row["reference_answer"]).items():
        assert rows.score(text, row) == live_score(task, item, text), label


@pytest.mark.parametrize("task,ordering", CELLS, ids=IDS)
def test_a_row_that_has_been_through_json_scores_identically(task, ordering):
    # The hazard this catches is the tuple: `semantics_query` keys its gold by
    # (claim, semantics) and `claim_chain` holds its line by identity into the
    # theory. JSON has neither, and both fail by missing every lookup rather
    # than by raising.
    _item, row = cell(task, ordering)
    back = json.loads(json.dumps(row))
    for text in [row["reference_answer"], *wrong_answers(row["reference_answer"]).values()]:
        assert rows.score(text, back) == rows.score(text, row)


def _keys_of(row):
    """Every key of the row's metadata, the two containers included."""
    meta = row["metadata"]
    return ([(meta, k, k) for k in meta]
            + [(meta["state"], k, f"state.{k}") for k in meta["state"]]
            + [(meta["gold"], k, f"gold.{k}") for k in meta["gold"]])


@pytest.mark.parametrize("task,ordering", CELLS, ids=IDS)
def test_a_row_that_lost_a_key_refuses_rather_than_scores_differently(task, ordering):
    _item, row = cell(task, ordering)
    ref = row["reference_answer"]
    whole = rows.score(ref, row)
    refused = set()
    for holder, key, name in _keys_of(row):
        cut = json.loads(json.dumps(row))
        holder_cut = cut["metadata"]
        for part in name.split(".")[:-1]:
            holder_cut = holder_cut[part]
        holder_cut.pop(key)
        try:
            assert rows.score(ref, cut) == whole, (
                f"{task}: dropping metadata.{name} changed the score silently")
        except (KeyError, ValueError):
            refused.add(name)
    spec = registry.get(task)
    # The keys that refuse are exactly the ones the scorer reads: the task, the
    # schema, the ordering, the theory, and the declared state and gold. Every
    # other key of the row can be dropped without moving the number.
    expected = ({"source_dataset", "theory_schema", "ordering"}
                | set(spec.theory_fields)
                | {f"state.{n}" for n, _ in spec.state_fields}
                | {f"gold.{n}" for n, _ in spec.gold_fields}
                | ({"state"} if spec.state_fields else set())
                | ({"gold"} if spec.gold_fields else set()))
    assert refused == expected


def test_the_answer_bearing_fields_are_all_under_gold():
    # One rule -- do not show the model `metadata.gold` -- rather than twelve.
    for task in TASKS:
        gold_names = {n for n, _ in registry.get(task).gold_fields}
        _item, row = cell(task, "last_link_elitist")
        assert set(row["metadata"]["gold"]) == gold_names, task
        assert not gold_names & set(row["metadata"]["state"]), task
        # `state` and `gold` themselves are the two containers; three tasks name
        # their label map `gold`, so the row reads `metadata.gold.gold`.
        assert not gold_names & (set(row["metadata"]) - {"state", "gold"}), task


def test_the_directive_count_and_the_survivors_are_gold():
    # Neither is in the question: the prompt never says how many directives an
    # answer needs, and `survivors` names the claims that did not change.
    _i, attack = cell("attack", "last_link_elitist")
    assert "min_directives" in attack["metadata"]["gold"]
    _i, pert = cell("perturbation", "last_link_elitist")
    assert "survivors" in pert["metadata"]["gold"]


def test_the_justifying_line_is_written_as_positions_in_the_theory():
    # line_ops is a sublist of base_ops. Writing the operations again would say
    # the same thing twice and let the copy disagree with the theory.
    item, row = cell("claim_chain", "last_link_elitist")
    idx = row["metadata"]["gold"]["line_ops"]
    assert idx and all(isinstance(i, int) for i in idx)
    assert [item.base_ops[i] for i in idx] == item.line_ops


def test_the_semantics_gold_comes_back_keyed_by_pairs():
    item, row = cell("semantics_query", "last_link_elitist")
    _spec, payload = rows.score_input(json.loads(json.dumps(row)))
    assert payload.gold == item.gold
    assert payload.queries == item.queries
    assert all(isinstance(q, tuple) for q in payload.queries)


def test_subgoals_is_recomputed_rather_than_stored():
    # It is a property over the goals and the theory. A stored copy would drift
    # the first time the property changed.
    item, row = cell("attack_defense", "last_link_elitist")
    assert "subgoals" not in json.dumps(row)
    _spec, payload = rows.score_input(row)
    assert payload["subgoals"] == item.subgoals


def test_a_row_from_a_schema_this_build_cannot_read_is_refused():
    _item, row = cell("status_query", "last_link_elitist")
    cut = json.loads(json.dumps(row))
    cut["metadata"]["theory_schema"] += 1
    with pytest.raises(ValueError):
        rows.score(cut["reference_answer"], cut)


def test_scoring_dispatches_on_the_row_not_on_the_dataset():
    # A concatenation hands every row to one `score`, so the row has to say
    # which task grades it.
    ds = TaskDataset("status_query", 3, "last_link_elitist", size=1)
    _item, row = cell("claim_chain", "last_link_elitist")
    assert ds.score(row["reference_answer"], row).score == 1.0


def test_a_freeze_records_what_each_reference_scored(tmp_path):
    # The old export computed this and counted how many reached 1.0
    # (`core/export.py`). Rows carry it now, one per row.
    from arggym.core import freeze as F
    from arggym.core.spec import SeedPolicy, TasksetSpec

    spec = TasksetSpec(tasks=("claim_chain",), levels=(3,),
                       orderings=("last_link_elitist",), seeds=SeedPolicy(take=2))
    manifest = F.freeze(spec, str(tmp_path / "t.jsonl"), verbose=False)
    assert manifest["n_verified"] == manifest["n_items"] == 2
    with open(tmp_path / "t.jsonl") as f:
        rows_read = [json.loads(line) for line in f][1:]
    assert [r["metadata"]["reference_score"] for r in rows_read] == [1.0, 1.0]


def test_a_reference_that_does_not_score_1_stops_the_freeze(monkeypatch, tmp_path):
    # A taskset whose own answers fail is not publishable, and counting the
    # failures in a manifest nobody reads is a check already lost.
    from arggym.core import freeze as F
    from arggym.core.spec import SeedPolicy, TasksetSpec

    monkeypatch.setattr(F, "reference_score", lambda entry: 0.5)
    spec = TasksetSpec(tasks=("claim_chain",), levels=(3,),
                       orderings=("last_link_elitist",), seeds=SeedPolicy(take=2))
    with pytest.raises(SystemExit, match="does not score 1.0"):
        F.freeze(spec, str(tmp_path / "t.jsonl"), verbose=False)


@pytest.mark.slow
@pytest.mark.parametrize("task,ordering", CELLS, ids=IDS)
def test_the_same_holds_a_level_up(task, ordering):
    item, row = cell(task, ordering, level=6)
    for text in [row["reference_answer"], *wrong_answers(row["reference_answer"]).values()]:
        assert rows.score(text, row) == live_score(task, item, text)
