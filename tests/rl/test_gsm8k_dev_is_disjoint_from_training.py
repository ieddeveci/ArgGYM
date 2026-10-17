"""The GSM8K control's dev split is fixed, held out, and new to the policy.

Reads `openai/gsm8k` from the Hugging Face cache; skips when it is not there.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rl import config as rl_config
from rl import gsm8k

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "rl/configs/gsm8k-qwen3-0.6b.yaml"


@pytest.fixture(scope="module")
def data():
    return rl_config.load(str(CONFIG))["data"]


@pytest.fixture(scope="module")
def splits(data):
    try:
        return gsm8k.load_splits(data)
    except (ConnectionError, OSError) as e:
        pytest.skip(f"openai/gsm8k is not in the cache: {e}")


def test_dev_shares_no_question_with_training(splits, data):
    train, dev = splits
    assert len(dev) == data["dev_size"] and len(train) == 7473
    assert {r["id"] for r in dev} <= {f"test-{i}" for i in range(1319)}
    assert not {r["question"].strip() for r in dev} & {r["question"].strip() for r in train}


def test_dev_is_the_same_rows_on_every_load(splits, data):
    again = gsm8k.load_splits(data)[1]
    assert [r["id"] for r in again] == [r["id"] for r in splits[1]]


def test_every_row_has_a_numeric_gold(splits):
    for rows in splits:
        for r in gsm8k.trl_rows(rows):
            assert isinstance(r["gold"], float)
            assert r["prompt"][-1]["content"].endswith(gsm8k.INSTRUCTION)
