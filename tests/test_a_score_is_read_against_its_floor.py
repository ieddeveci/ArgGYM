"""What an uninformed answer scores, per task.

#9, and `docs/dataset-card.md`. A benchmark number means nothing without the
number an uninformed answer gets: measured here, answering "justified" to every
query scores 0.375 on `status_query`, so a model at 0.3 there is worse than a
fixed reply.

The strategies are deliberately dumb -- what a model that read the answer format
and nothing else can produce. They are not attacks on the scorer. One constant
for the whole item is not the widest such answer, which is #95: where the
answer key carries a component the task fixes, the uninformed answer is one
constant per value of it.
"""
import json

import pytest

import arggym
from arggym.core.floors import (
    _CANDIDATES,
    STRATEGIES,
    _key_groups,
    corrected,
    floor_for,
    floor_strategy,
    floors,
)
from arggym.core.freeze import freeze
from arggym.core.spec import SeedPolicy, TasksetSpec

LABEL_TASKS = ("status_query", "semantics_query", "perturbation")


@pytest.fixture(scope="module")
def rows(tmp_path_factory):
    spec = TasksetSpec(tasks=LABEL_TASKS + ("claim_chain", "counter_argument"),
                       levels=(3,), orderings=("last_link_elitist",),
                       seeds=SeedPolicy(take=2, scan_limit=20))
    p = tmp_path_factory.mktemp("f") / "t.jsonl"
    freeze(spec, str(p), verbose=False)
    return [json.loads(x) for x in open(p)][1:]


def test_every_task_gets_a_floor(rows):
    got = floors(rows)
    assert set(got) == set(LABEL_TASKS + ("claim_chain", "counter_argument"))
    for task, v in got.items():
        assert 0.0 <= v["floor"] <= 1.0, task
        # Not `in STRATEGIES`: a fitted strategy is not one of the named
        # constants, and every strategy that ran is a key of `by_strategy`.
        assert v["strategy"] in v["by_strategy"]


@pytest.mark.parametrize("task", LABEL_TASKS)
def test_a_constant_label_is_actually_submitted(task, rows):
    # The trap this guards: a strategy whose text does not match a task's answer
    # format submits nothing, and the task reports a floor of 0.0. An
    # unmeasured floor reads exactly like a task where guessing does not pay,
    # which is the opposite of the truth. semantics_query measured 0.000 that
    # way before the strategy learned its phrasing, and it is the highest floor
    # in the set.
    for row in [r for r in rows if r["task"] == task]:
        text = STRATEGIES["all_justified"](row)
        assert text.strip(), f"{task}: the constant strategy produced no answer"


def test_answering_justified_to_everything_pays_on_the_query_tasks(rows):
    got = floors(rows)
    # `semantics_query` asks under several semantics at once, and the status
    # that pays differs between them: over the shipped grid one constant scores
    # 0.459 and one constant per semantics 0.671, so a floor from one constant
    # reported 0.54 of room above chance where there is 0.33 (#95).
    assert got["semantics_query"]["floor"] > 0.7, got["semantics_query"]
    assert got["status_query"]["floor"] > 0.2, got["status_query"]


def test_only_a_vocabulary_the_task_fixes_becomes_a_key_group(rows):
    # The two halves of `semantics_query`'s answer key differ in where they come
    # from: the semantics is five names the task schedules by level, the claim is
    # drawn per item from a pool of two thousand. A constant per semantics is a
    # map an uninformed answerer writes down in advance; a constant per sampled
    # claim would be the gold spelled as a map, so only the first earns a group.
    assert _key_groups([r for r in rows if r["task"] == "semantics_query"]) == (1,)
    for task in ("status_query", "perturbation", "claim_chain"):
        assert _key_groups([r for r in rows if r["task"] == task]) == (), task


def test_a_floor_says_what_reached_it(rows):
    # `0.000 empty` means the search found nothing that fits the answer format,
    # which is a different sentence from "guessing does not pay here". The number
    # alone spells the two the same way, and a fitted floor is unreadable without
    # the map it fitted.
    got = floors(rows)
    assert floor_strategy(got["status_query"]) == got["status_query"]["strategy"]

    sem = got["semantics_query"]
    assert sem["strategy"] == "per_key_group"
    assert set(sem["detail"]) == {"grounded", "credulous preferred"}
    assert floor_strategy(sem).startswith("per_key_group(credulous preferred=")


def test_every_candidate_is_a_status_the_scorer_accepts():
    # A candidate the parser drops is a search that measures nothing. `no stable
    # extension` is the answer `semantics_query` asks for when a theory has none,
    # and the search never tried it -- #95 in miniature.
    from arggym.tasks.semantics_query import _PAIR

    for status in _CANDIDATES:
        assert _PAIR.findall(f"x under stable: {status}") == [("x", "stable", status)]


def test_a_directive_task_cannot_be_guessed(rows):
    # An engine-checked answer either reaches the goals or does not, so no
    # constant text scores. That asymmetry is worth stating: the two families
    # are not on the same scale.
    assert floors(rows)["counter_argument"]["floor"] == 0.0


def test_the_floor_is_the_best_constant_not_their_average(rows):
    # A reader comparing a model against chance asks whether it beat the
    # easiest thing that works, not the average of several dumb things.
    value, best, means, _ = floor_for([r for r in rows if r["task"] == "status_query"])
    assert value == max(means.values())
    assert means[best] == value


def test_chance_correction_puts_a_floor_at_zero_and_perfect_at_one():
    assert corrected(0.49, 0.49) == 0.0
    assert corrected(1.0, 0.49) == 1.0
    assert corrected(0.745, 0.49) == pytest.approx(0.5, abs=1e-3)


def test_a_floor_is_measured_from_the_question_not_from_the_gold(rows):
    # A strategy that read metadata.gold would not be uninformed, and would
    # report a floor of 1.0 everywhere.
    row = next(r for r in rows if r["task"] == "status_query")
    stripped = dict(row, metadata=dict(row["metadata"]))
    stripped["metadata"]["gold"] = {}
    assert STRATEGIES["all_justified"](stripped) == STRATEGIES["all_justified"](row)


def test_the_floors_command_reports_per_task(tmp_path, capsys):
    # Never one number: the twelve metrics are of four kinds and their floors
    # differ by an order of magnitude.
    spec = TasksetSpec(tasks=("status_query", "claim_chain"), levels=(3,),
                       orderings=("last_link_elitist",),
                       seeds=SeedPolicy(take=1, scan_limit=10))
    p = tmp_path / "t.jsonl"
    freeze(spec, str(p), verbose=False)
    got = floors([json.loads(x) for x in open(p)][1:])
    assert len(got) == 2
    assert got["status_query"]["n"] == 1


def test_the_public_api_exposes_the_floor(rows):
    assert hasattr(arggym, "floors")
