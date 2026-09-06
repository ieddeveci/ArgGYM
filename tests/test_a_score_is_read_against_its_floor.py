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

A fitted map is searched against labels, so two things it must not become have
tests of their own here: the gold, when the map has an entry per coordinate; and
a hindsight number, when the map is fitted on the same small group it corrects.
"""
import itertools
import json

import pytest

import arggym
from arggym.core import floors as floors_module
from arggym.core.floors import (
    _CANDIDATES,
    FITTED,
    STRATEGIES,
    FittedFloorIsGold,
    _asked,
    _by_group,
    _group_of,
    _key_groups,
    _mean,
    corrected,
    floor_for,
    floor_strategy,
    floors,
)
from arggym.core.freeze import freeze
from arggym.core.spec import ALL_ORDERINGS, SeedPolicy, TasksetSpec

LABEL_TASKS = ("status_query", "semantics_query", "perturbation")
TASKS = LABEL_TASKS + ("claim_chain", "formalization", "counter_argument")


@pytest.fixture(scope="module")
def rows(tmp_path_factory):
    # Four orderings, for eight rows a task. A fitted map is admitted only while
    # it stays far smaller than the coordinates it answers, and two rows cannot
    # pay for one constant per semantics.
    spec = TasksetSpec(tasks=TASKS, levels=(3,), orderings=ALL_ORDERINGS,
                       seeds=SeedPolicy(take=2, scan_limit=20))
    p = tmp_path_factory.mktemp("f") / "t.jsonl"
    freeze(spec, str(p), verbose=False)
    return [json.loads(x) for x in open(p)][1:]


def of(rows, task):
    return [r for r in rows if r["task"] == task]


def test_every_task_gets_a_floor(rows):
    got = floors(rows)
    assert set(got) == set(TASKS)
    for task, v in got.items():
        assert 0.0 <= v["floor"] <= 1.0, task
        # Against the two registries rather than against `by_strategy`, which is
        # the dict the winner was chosen out of and would agree with anything.
        assert v["strategy"] in STRATEGIES or v["strategy"] in FITTED, task


@pytest.mark.parametrize("task", LABEL_TASKS)
def test_a_constant_label_is_actually_submitted(task, rows):
    # The trap this guards: a strategy whose text does not match a task's answer
    # format submits nothing, and the task reports a floor of 0.0. An
    # unmeasured floor reads exactly like a task where guessing does not pay,
    # which is the opposite of the truth. semantics_query measured 0.000 that
    # way before the strategy learned its phrasing, and it is the highest floor
    # in the set.
    for row in of(rows, task):
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
    assert _key_groups(of(rows, "semantics_query")) == (1,)
    for task in ("status_query", "perturbation", "claim_chain"):
        assert _key_groups(of(rows, task)) == (), task


def test_rows_that_repeat_an_ask_list_do_not_earn_a_key_group(rows):
    # A vocabulary that does not grow with the rows is not enough on its own:
    # rows repeating one ask list hold the claim fixed too, and a map over
    # (claim, semantics) is the gold in another notation. A report fits over the
    # rows a run actually scored, so a filtered run can hand this in.
    repeated = of(rows, "semantics_query")[:1] * 6
    assert _key_groups(repeated) == ()

    value, best, _, detail = floor_for(repeated)
    assert best in STRATEGIES and detail is None
    assert value < 1.0


def test_a_map_with_an_entry_per_coordinate_refuses_to_be_a_floor(rows, monkeypatch):
    # With the budget lifted, the same rows fit one constant per (claim,
    # semantics) and score 1.000. That must not be reported: `corrected` divides
    # by `1 - floor`, and at a floor of 1.0 it returns 0.0 for every score in the
    # group, which reads as a task no model can beat.
    monkeypatch.setattr(floors_module, "MIN_COORDS_PER_ENTRY", 0)
    with pytest.raises(FittedFloorIsGold):
        floor_for(of(rows, "semantics_query")[:1] * 4)


def test_greedy_finds_the_map_an_exhaustive_search_finds(rows):
    # The fit pins the other groups and varies one, which is the exact argmax
    # only because every coordinate is answered exactly once and the row's score
    # is then affine in its correct lines. Ranking a group by its own lines
    # instead reweights the rows and misses maps this finds, so the property is
    # worth a test rather than a paragraph.
    sem = of(rows, "semantics_query")
    idx = _key_groups(sem)
    groups = sorted({g for row in sem for g in _by_group(row, idx)})

    best, argmax = -1.0, []
    for combo in itertools.product(_CANDIDATES, repeat=len(groups)):
        chosen = dict(zip(groups, combo))
        value = _mean(lambda row: "\n".join(
            f"{c}: {chosen[_group_of(c, idx)]}" for c in _asked(row)), sem)
        if value > best:
            best, argmax = value, [chosen]
        elif value == best:
            argmax.append(chosen)

    make, detail = FITTED["per_key_group"](sem)
    assert {g: detail[" ".join(g)] for g in groups} in argmax
    # `pair_f1` is rounded to four decimals per row and the mean is rounded
    # again, so two ways of adding the same map's rows can land either side of a
    # tie at the fourth decimal. The map is what the search has to get right.
    assert _mean(make, sem) == pytest.approx(best, abs=1e-4)


def test_a_fitted_map_comes_from_the_task_not_from_the_group_it_corrects(rows):
    # A level's mean is corrected by that level's floor, and a map searched
    # against the labels of the rows it then corrects is optimistic by
    # hindsight -- 0.088 on `semantics_query` at level 9 of the shipped grid. The
    # map belongs to the dataset, the number to the rows.
    sem = of(rows, "semantics_query")
    half = sem[:4]

    assert floors(half)["semantics_query"]["strategy"] != "per_key_group"
    handed = floors(half, fit_rows=sem)["semantics_query"]
    assert handed["strategy"] == "per_key_group"
    assert handed["detail"] == floors(sem)["semantics_query"]["detail"]


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


def test_every_candidate_is_a_status_the_scorer_accepts(rows):
    # A candidate the parser drops is a search that measures nothing. `no stable
    # extension` is the answer `semantics_query` asks for when a theory has none,
    # and the search never tried it -- #95 in miniature. Submitted the way a
    # model would, so this fails if the scorer stops taking one of them.
    row = of(rows, "semantics_query")[0]
    for status in _CANDIDATES:
        answer = "\n".join(f"{c}: {status}" for c in _asked(row))
        assert arggym.score_row(answer, row).reason == "ok", status


def test_a_key_group_that_reads_the_answer_format_changes_nothing(rows, monkeypatch):
    # `formalization`'s question lists no coordinates, so `_asked` falls through
    # to the literals of the answer-format example and returns the same two on
    # every item. That passes the fixed-vocabulary rule on the shipped grid, and
    # it is inert: the task scores an operation list, so a label line is worth
    # zero whichever status it carries.
    monkeypatch.setattr(floors_module, "MIN_COORDS_PER_ENTRY", 0)
    got = floors(of(rows, "formalization"))["formalization"]
    assert got["floor"] == 0.0 and got["strategy"] == "empty"
    assert got["by_strategy"]["per_key_group"] == 0.0


def test_a_directive_task_cannot_be_guessed(rows):
    # An engine-checked answer either reaches the goals or does not, so no
    # constant text scores. That asymmetry is worth stating: the two families
    # are not on the same scale.
    assert floors(rows)["counter_argument"]["floor"] == 0.0


def test_the_floor_is_the_best_constant_not_their_average(rows):
    # A reader comparing a model against chance asks whether it beat the
    # easiest thing that works, not the average of several dumb things.
    value, best, means, _ = floor_for(of(rows, "status_query"))
    assert value == max(means.values())
    assert means[best] == value


def test_chance_correction_puts_a_floor_at_zero_and_perfect_at_one():
    assert corrected(0.49, 0.49) == 0.0
    assert corrected(1.0, 0.49) == 1.0
    assert corrected(0.745, 0.49) == pytest.approx(0.5, abs=1e-3)


def test_a_floor_is_measured_from_the_question_not_from_the_gold(rows):
    # A strategy that read metadata.gold would not be uninformed, and would
    # report a floor of 1.0 everywhere.
    row = of(rows, "status_query")[0]
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
