"""The GSM8K control's dev split is fixed, held out, and new to the policy.

Reads `openai/gsm8k` from the Hugging Face cache; skips when it is not there.
"""
from __future__ import annotations

import json
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


def _shards(tmp_path, counts):
    for i in range(2):
        rows = [{"id": k, "passes": p, "samples": 8} for k, p in list(counts.items())[i::2]]
        (tmp_path / f"shard-{i}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_the_screen_keeps_questions_whose_passes_lie_in_the_range(tmp_path):
    ids = [f"train-{i}" for i in range(7473)] + [f"test-{i}" for i in range(1319)]
    counts = {k: i % 9 for i, k in enumerate(ids)}
    _shards(tmp_path, counts)
    kept = gsm8k.screened({"path": str(tmp_path), "samples": 8, "passes": [2, 6]})
    assert kept == {k for k, p in counts.items() if 2 <= p <= 6}


def test_a_missing_shard_is_refused(tmp_path):
    _shards(tmp_path, {f"train-{i}": 3 for i in range(7473)} | {f"test-{i}": 3 for i in range(1319)})
    (tmp_path / "shard-1.jsonl").unlink()
    with pytest.raises(ValueError, match="shard is missing"):
        gsm8k.screened({"path": str(tmp_path), "samples": 8, "passes": [2, 6]})
