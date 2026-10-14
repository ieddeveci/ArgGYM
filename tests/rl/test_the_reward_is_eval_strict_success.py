"""The reward is 1 exactly when the eval harness would score the completion a success.

Gold answers pay 1 on every task, wrong answers 0, truncated completions `None`
(out of the group baseline), and the constant-label answers the reward-risk
analysis found earning partial credit pay 0. The batch metrics count a
truncated completion as a failure, as eval does.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from arggym.core.floors import STRATEGIES
from evals import taskset
from evals.score import score_one
from rl.reward import batch_metrics, make_reward, split_reasoning

ROOT = Path(__file__).resolve().parents[2]
EOS, PAD, OTHER = 151645, 151643, 42


@pytest.fixture(scope="module")
def rows():
    return taskset.load(str(ROOT / "data/taskset-lite.jsonl"))[1]


def one_per_task(rows):
    seen = {}
    for r in rows:
        seen.setdefault(r["task"], r)
    return list(seen.values())


def completion(answer: str, think: str = "Working it out.") -> str:
    return f"<think>\n{think}\n</think>\n\n<answer>\n{answer}\n</answer>"


def reward(texts, rows, ids=None, **kw):
    fn = make_reward([EOS, PAD])
    ids = ids or [[OTHER, EOS]] * len(texts)
    return fn(completions=[[{"role": "assistant", "content": t}] for t in texts],
              completion_ids=ids, row_json=[json.dumps(r) for r in rows], **kw)


def test_a_gold_answer_pays_one_on_every_task(rows):
    sample = one_per_task(rows)
    assert len(sample) == 12
    got = reward([completion(r["reference_answer"]) for r in sample], sample)
    assert got == [1.0] * 12


def test_a_wrong_answer_pays_zero(rows):
    sample = one_per_task(rows)
    assert reward([completion("I am not sure.") for _ in sample], sample) == [0.0] * 12
    assert reward(["no answer region at all" for _ in sample], sample) == [0.0] * 12


def test_a_truncated_completion_is_unscorable_even_with_the_gold_answer(rows):
    sample = one_per_task(rows)
    texts = [completion(r["reference_answer"]) for r in sample]
    assert reward(texts, sample, ids=[[OTHER, OTHER]] * 12) == [None] * 12
    # Ending on padding is finished, as in TRL's own truncation test.
    assert reward(texts, sample, ids=[[OTHER, PAD]] * 12) == [1.0] * 12


@pytest.mark.parametrize("task", ["semantics_query", "perturbation", "status_query"])
def test_a_constant_label_answer_pays_nothing_for_its_partial_credit(rows, task):
    sample = [r for r in rows if r["task"] == task]
    partial_seen = False
    for name in ("all_justified", "all_overruled", "all_undecided"):
        answers = [STRATEGIES[name](r) for r in sample]
        got = reward([completion(a) for a in answers], sample)
        for a, r, g in zip(answers, sample, got):
            rec = score_one({"id": r["id"], "task": task, "level": 1, "ordering": "x",
                             "completion": f"<answer>\n{a}\n</answer>"}, r, "xml_tags")
            assert g == (1.0 if rec["success"] else 0.0)
            partial_seen |= g == 0.0 and rec["score"] > 0
    # The point of the test: the floor exists, and the reward does not pay it.
    assert partial_seen


def test_an_answer_left_inside_the_reasoning_scores_as_eval_scores_it(rows):
    r = one_per_task(rows)[0]
    text = f"<think>\n<answer>\n{r['reference_answer']}\n</answer>\n</think>\n\nDone."
    rec = score_one({"id": r["id"], "task": r["task"], "level": 1, "ordering": "x",
                     **split_reasoning(text)}, r, "xml_tags")
    assert rec["answer_in_cot"] and rec["success"]
    assert reward([text], [r]) == [1.0]


def test_split_matches_the_reasoning_parser():
    assert split_reasoning("<think>\nA\n</think>\n\nB") == {"reasoning": "A", "completion": "B"}
    assert split_reasoning("no think") == {"reasoning": "", "completion": "no think"}


def test_metrics_split_groups_lengths_and_count_truncation_as_failure(rows):
    def rec(i, success, truncated=False, length=10):
        return {"id": f"p{i}", "task": "attack", "success": success,
                "score": 1.0 if success else 0.3, "truncated": truncated,
                "length": length, "no_answer_region": False}

    records = ([rec(0, True), rec(0, True)] + [rec(1, False), rec(1, False, True, 99)]
               + [rec(2, True, length=20), rec(2, False, length=40)]
               + [rec(3, False, True), rec(3, False, True)])
    m = batch_metrics(records)
    assert m["arggym_success_rate"] == 3 / 8
    assert m["arggym_truncated_rate"] == 3 / 8
    assert m["arggym_groups/all_correct"] == 1 / 4
    assert m["arggym_groups/all_wrong"] == 1 / 4
    assert m["arggym_groups/no_scored_member"] == 1 / 4
    assert m["arggym_length/correct"] == (10 + 10 + 20) / 3
    assert m["arggym_length/wrong"] == (10 + 40) / 2
    assert m["arggym_success_rate/attack"] == 3 / 8


def test_the_reward_logs_through_trl_hooks(rows):
    sample = one_per_task(rows)[:2]
    metrics, extra = {}, {}
    reward([completion(sample[0]["reference_answer"]), completion("x")], sample,
           log_metric=metrics.__setitem__, log_extra=extra.__setitem__)
    assert metrics["arggym_success_rate"] == 0.5
    assert extra["success"] == [True, False]
    assert set(extra) >= {"task", "level", "partial_score", "reason", "truncated"}
