"""A dataset object is one difficulty, addressed by seed.

`docs/dataset-contract.md` section 6. The property under test is that `ds[k]`
depends on `k` and nothing else -- not on whether `ds[k-1]` built, not on the
order the caller asked in, not on how many items were requested.
"""
import pytest

from arggym.core.answers import DEFAULT_TEMPLATE, AnswerTemplate
from arggym.core.dataset import BuildFailed, ConcatDataset, TaskDataset, create, from_spec
from arggym.core.spec import SeedPolicy, TasksetSpec

CHEAP = dict(task="claim_chain", level=3, ordering="last_link_elitist")


def test_the_item_at_an_index_is_the_item_from_that_seed():
    ds = create(**CHEAP, size=3, seed=0)
    assert [e["metadata"]["seed"] for e in ds] == [0, 1, 2]


def test_an_index_does_not_depend_on_the_ones_below_it():
    # The property that a dense index would destroy: reading ds[2] alone must
    # give what reading 0, 1, 2 in order gives.
    whole = list(create(**CHEAP, size=3, seed=0))
    assert create(**CHEAP, size=3, seed=0)[2] == whole[2]


def test_starting_elsewhere_shifts_the_seed_and_the_id():
    a = create(**CHEAP, size=1, seed=7)[0]
    assert a["metadata"]["seed"] == 7
    assert a["id"].endswith("/s7")


def test_asking_for_more_items_does_not_change_the_old_ones():
    assert create(**CHEAP, size=1, seed=0)[0] == create(**CHEAP, size=9, seed=0)[0]


def test_a_failed_build_names_the_cell_rather_than_moving_on():
    # Skipping to the next seed would make a cell that rejects most of its
    # draws look like a full one. See #72.
    class Stubborn(TaskDataset):
        def __getitem__(self, idx):
            raise BuildFailed(f"{self.task} L{self.level} {self.ordering} seed {idx}")

    with pytest.raises(BuildFailed, match="seed 4"):
        Stubborn(**CHEAP, size=5)[4]


def test_an_index_past_the_end_is_an_index_error():
    with pytest.raises(IndexError):
        create(**CHEAP, size=2)[2]


def test_a_row_carries_the_theory_its_question_shows():
    e = create(**CHEAP, size=1)[0]
    assert e["metadata"]["base_ops"], "the row must carry the theory to be scorable offline"
    assert e["question"] and e["reference_answer"]


def test_the_answer_bearing_operations_sit_under_gold():
    # claim_chain's line_ops is the justifying line the answer must reproduce,
    # so it must not sit beside the theory where a harness would display it.
    e = create(**CHEAP, size=1)[0]
    assert "line_ops" in e["metadata"]["gold"]
    assert "line_ops" not in e["metadata"]


def test_formalization_publishes_no_theory_because_it_has_none():
    # Its question is prose and its operations are the gold answer.
    e = create(task="formalization", level=3, ordering="last_link_elitist", size=1)[0]
    assert "base_ops" not in e["metadata"]
    assert "reference_ops" in e["metadata"]["gold"]


def test_perturbation_carries_both_of_its_operation_lists():
    e = create(task="perturbation", level=3, ordering="last_link_elitist", size=1)[0]
    assert e["metadata"]["base_ops"] and e["metadata"]["pert_ops"]


def test_a_row_asks_for_no_convention_unless_one_was_requested():
    """Where the answer goes is the harness's sentence, so the question omits it and
    the row says so rather than leaving a reader to infer it from the prompt text."""
    e = create(**CHEAP, size=1)[0]
    assert e["metadata"]["answer_template"] is None
    assert DEFAULT_TEMPLATE.instruction not in e["question"]


def test_asking_for_a_template_adds_one_sentence_and_records_it():
    """The same item either way: only the question is re-rendered."""
    plain = create(**CHEAP, size=1)[0]
    boxed = AnswerTemplate("boxed", "\\boxed{", "}")
    other = create(**CHEAP, size=1, template=boxed)[0]
    assert other["metadata"]["answer_template"] == "boxed"
    assert other["question"] == f"{plain['question']}\n{boxed.instruction}"
    assert other["reference_answer"] == plain["reference_answer"]
    assert other["metadata"]["gold"] == plain["metadata"]["gold"]


def test_source_dataset_is_the_registry_key_so_a_composite_can_dispatch():
    e = create(task="counter_argument_strict", level=3,
               ordering="last_link_elitist", size=1)[0]
    assert e["metadata"]["source_dataset"] == "counter_argument_strict"


def test_a_concatenation_reads_its_parts_end_to_end():
    a = create(**CHEAP, size=2)
    b = create(task="status_query", level=3, ordering="last_link_elitist", size=2)
    both = ConcatDataset([a, b])
    assert len(both) == 4
    assert [e["task"] for e in both] == ["claim_chain"] * 2 + ["status_query"] * 2
    assert both[3] == b[1]


def test_a_spec_expands_to_every_cell_once():
    s = TasksetSpec(tasks=("claim_chain",), levels=(3,),
                    orderings=("last_link_elitist", "weakest_link_elitist"),
                    seeds=SeedPolicy(take=2))
    ds = from_spec(s)
    assert len(ds) == 4
    assert {(e["metadata"]["ordering"], e["metadata"]["seed"]) for e in ds} == {
        ("last_link_elitist", 0), ("last_link_elitist", 1),
        ("weakest_link_elitist", 0), ("weakest_link_elitist", 1)}
