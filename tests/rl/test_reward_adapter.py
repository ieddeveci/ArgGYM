from __future__ import annotations

import json

import arggym

from evals import taskset
from rl.generate import TEMPLATE, prompt_messages
from rl.reward import arggym_reward


def _frozen_status_row():
    _manifest, rows = taskset.load("data/taskset.jsonl", verify=True)
    return next(r for r in rows if r["task"] == "status_query")


def test_reward_is_exactly_existing_arggym_score():
    # A frozen row exercises the same row-only scoring seam used by RL and does
    # not require regenerating the theory. This makes the parity test cheap.
    row = _frozen_status_row()
    completion = f"<answer>\n{row['reference_answer']}\n</answer>"
    got = arggym_reward([completion], [json.dumps(row)], [TEMPLATE])
    want = arggym.score_row(row["reference_answer"], row).score
    assert got == [want]


def test_prompt_does_not_expose_gold_or_reference_answer():
    row = _frozen_status_row()
    rendered = json.dumps(prompt_messages(row), sort_keys=True)
    assert "reference_answer" not in rendered
    assert '"gold"' not in rendered
    before = prompt_messages(row)
    row["metadata"]["gold"] = {"poison": "DO NOT LEAK"}
    row["reference_answer"] = "DO NOT LEAK"
    assert prompt_messages(row) == before


def test_missing_answer_region_is_not_rewarded():
    row = _frozen_status_row()
    # Even a literally correct reference answer is not a submitted answer when
    # it occurs outside the requested final region.
    got = arggym_reward([row["reference_answer"]], [json.dumps(row)], [TEMPLATE])
    assert got == [0.0]
