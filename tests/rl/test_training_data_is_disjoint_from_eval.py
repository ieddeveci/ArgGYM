"""No training or dev row is an eval row, by seed, by question, or by theory.

Seed windows are checked before anything is built; a pool is then checked row
by row against both frozen eval tasksets. The pool built here is one seed a cell
at level 1, where theories are smallest and a collision most likely. Any pool
already built under `outputs/rl/pools/` is checked too.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import arggym
from evals import taskset
from rl import config as rl_config
from rl import pool

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ("rl/configs/qwen3-4b.yaml", "rl/configs/smoke.yaml")


@pytest.fixture(scope="module")
def eval_keys():
    return pool.keys(pool.eval_rows())


@pytest.fixture(scope="module")
def small_pool(tmp_path_factory):
    cfg = rl_config.load(str(ROOT / CONFIGS[1]))
    cfg["output_root"] = str(tmp_path_factory.mktemp("rl"))
    cfg["data"]["levels"] = [1]
    cfg["data"]["train"]["take"] = 1
    pool.build(cfg, workers=8)
    return cfg


@pytest.mark.parametrize("path", CONFIGS)
def test_the_configured_seed_windows_miss_every_eval_window(path):
    pool.check_windows(rl_config.load(str(ROOT / path)))


def test_an_overlapping_window_is_refused_before_building():
    cfg = rl_config.load(str(ROOT / CONFIGS[1]))
    cfg["data"]["train"]["seed_start"] = 30
    with pytest.raises(SystemExit, match="seed windows overlap"):
        pool.check_windows(cfg)
    cfg = rl_config.load(str(ROOT / CONFIGS[1]))
    cfg["data"]["dev"]["seed_start"] = cfg["data"]["train"]["seed_start"] + 1
    with pytest.raises(SystemExit, match="seed windows overlap"):
        pool.check_windows(cfg)


def _check(cfg, eval_keys):
    train, dev = pool.load_split(cfg, "train"), pool.load_split(cfg, "dev")
    eval_seeds = range(0, 40)
    for split in (train, dev):
        assert {r["task"] for r in split} == set(arggym.task_names())
        assert {r["metadata"]["level"] for r in split} == set(cfg["data"]["levels"])
        assert not {r["metadata"]["seed"] for r in split} & set(eval_seeds)
        q, t = pool.keys(split)
        assert not q & eval_keys[0], "a pool question is an eval question"
        assert not t & eval_keys[1], "a pool theory is an eval theory"
    tq, tt = pool.keys(train)
    dq, dt = pool.keys(dev)
    assert not dq & tq and not dt & tt, "a dev row is a training row"
    assert not {r["id"] for r in dev} & {r["id"] for r in train}


def test_a_built_pool_shares_nothing_with_eval(small_pool, eval_keys):
    _check(small_pool, eval_keys)


def test_a_row_eval_holds_is_dropped(eval_keys):
    _, rows = taskset.load(str(ROOT / "data/taskset-lite.jsonl"))
    kept, dropped = pool.drop_seen(rows[:5], eval_keys)
    assert kept == [] and sum(dropped.values()) == 5


def test_the_dev_split_rotates_orderings_and_covers_every_cell(small_pool):
    dev = pool.load_split(small_pool, "dev")
    cells = {(r["task"], r["metadata"]["level"]) for r in dev}
    assert len(dev) == len(cells) == 12 * len(small_pool["data"]["levels"])


BUILT = sorted((ROOT / "outputs/rl/pools").glob("*/summary.json"))


@pytest.mark.parametrize("summary", BUILT, ids=[p.parent.name for p in BUILT])
def test_every_pool_on_disk_shares_nothing_with_eval(summary, eval_keys):
    import json

    data = json.loads(summary.read_text())["data"]
    cfg = {"data": data, "output_root": "outputs/rl"}
    assert pool.pool_dir(cfg) == summary.parent
    _check(cfg, eval_keys)


def test_a_pool_frozen_under_other_rules_is_refused(small_pool):
    import json

    path = pool.pool_dir(small_pool) / "dev.jsonl"
    lines = path.read_text().splitlines()
    head = json.loads(lines[0])
    head["__manifest__"]["versions"]["scoring_version"] -= 1
    path.write_text("\n".join([json.dumps(head), *lines[1:]]) + "\n")
    try:
        with pytest.raises(SystemExit, match="scoring_version"):
            pool.load_split(small_pool, "dev")
        with pytest.raises(SystemExit, match="scoring_version"):
            pool.main([str(ROOT / CONFIGS[1]), f"output_root={small_pool['output_root']}",
                       "data.levels=[1]", "data.train.take=1"])
    finally:
        path.write_text("\n".join(lines) + "\n")


def test_a_formalization_theory_is_keyed_by_its_reference_theory(eval_keys):
    _, rows = taskset.load(str(ROOT / "data/taskset-lite.jsonl"))
    forms = [r for r in rows if r["task"] == "formalization"]
    a, b = forms[0], json_copy(forms[0])
    b["metadata"]["gold"]["reference_ops"] = b["metadata"]["gold"]["reference_ops"][:-1]
    assert a["metadata"]["state"] == b["metadata"]["state"]
    assert pool.theory_key(a) != pool.theory_key(b)
    # The same theory asked under another ordering is the same item.
    c = json_copy(a)
    c["metadata"]["ordering"] = "weakest_link_democratic"
    assert pool.theory_key(c) == pool.theory_key(a)


def json_copy(row):
    import json

    return json.loads(json.dumps(row))
