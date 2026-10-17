"""The GSM8K control pays 1 exactly when the last `\\boxed{}` is the gold number.

Commas and dollar signs are formatting, not a different answer; a completion
with no box pays 0; a truncated one is `None`, out of the group baseline, as in
the ArgGYM reward.
"""
from __future__ import annotations

import pytest

from rl.gsm8k import batch_metrics, gold, make_reward

EOS, PAD, OTHER = 151645, 151643, 42


def reward(texts, golds, ids=None):
    fn = make_reward([EOS, PAD], gather=lambda x: x)
    ids = ids or [[OTHER, EOS]] * len(texts)
    return fn(completions=[[{"role": "assistant", "content": t}] for t in texts],
              completion_ids=ids, gold=golds, id=[f"p{i}" for i in range(len(texts))])


def done(answer: str) -> str:
    return f"<think>\nSome steps.\n</think>\n\nSo the answer is $\\boxed{{{answer}}}$."


@pytest.mark.parametrize("answer, value", [
    ("18", 18), ("1,234", 1234), ("\\$18", 18), ("$1,000,000", 1e6), ("-7", -7),
    ("0.5", 0.5), ("3.50", 3.5), (" 42 ", 42), ("{18}", None),
])
def test_a_boxed_number_pays_one_when_it_is_the_gold(answer, value):
    if value is None:  # braces inside the box are not a number
        assert reward([done(answer)], [18.0]) == [0.0]
        return
    assert reward([done(answer)], [float(value)]) == [1.0]
    assert reward([done(answer)], [float(value) + 1]) == [0.0]


def test_the_last_box_is_the_answer():
    text = "<think>\nmaybe \\boxed{5}\n</think>\n\n\\boxed{6} no, \\boxed{7}"
    assert reward([text, text], [7.0, 5.0]) == [1.0, 0.0]


def test_no_box_or_no_number_pays_zero():
    assert reward(["The answer is 18.", done("eighteen"), done("18 apples")],
                  [18.0] * 3) == [0.0] * 3


def test_a_truncated_completion_is_unscorable_even_when_right():
    assert reward([done("18")], [18.0], ids=[[OTHER, OTHER]]) == [None]
    # Ending on padding is finished, as in the ArgGYM reward.
    assert reward([done("18")], [18.0], ids=[[OTHER, PAD]]) == [1.0]


def test_the_gold_is_the_number_after_the_hashes():
    assert gold("She makes 9 * 2 = $<<9*2=18>>18.\n#### 18") == 18.0
    assert gold("...\n#### 1,600") == 1600.0
    assert gold("...\n#### -3") == -3.0


def test_metrics_count_truncation_as_failure_and_find_groups_without_signal():
    def rec(i, success, truncated=False, boxed=True, length=10):
        return {"id": f"p{i}", "success": success, "boxed": boxed,
                "truncated": truncated, "length": length}

    m = batch_metrics([rec(0, True), rec(0, True), rec(1, True), rec(1, False, boxed=False),
                       rec(2, True, truncated=True), rec(2, False, length=30)])
    assert m["gsm8k_success_rate"] == 3 / 6
    assert m["gsm8k_truncated_rate"] == 1 / 6
    assert m["gsm8k_no_box_rate"] == 1 / 5
    assert m["gsm8k_length/wrong"] == 20
    # p0 agrees and p2 has one scored member: only p1 carries a gradient.
    assert m["gsm8k_groups/no_signal"] == pytest.approx(2 / 3)
