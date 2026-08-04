"""The inference path must accept either benchmark's row shape.

v1 nests gold under `entry` and identifies an item by (mode, idx); ArgGYM_v2
stores `reference` at the top level and identifies by (ordering, seed). The
runner writes gold to disk *before* inference, so a shape mismatch does not
surface as a bad score -- it kills the run before a single token is generated,
once per (model, level) cell, which is how it burned a sweep launch.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals import artifacts


def _v1_row():
    return {
        "sample_id": "status_query__symbolic__L01__000",
        "task": "status_query", "mode": "symbolic", "level": 1, "idx": 0,
        "prompt": "the v1 prompt",
        "entry": {"answer": "a1: justified", "metadata": {"n": 3}},
    }


def _v2_row():
    return {
        "sample_id": "status_query__ll__L03__000",
        "task": "status_query", "ordering": "last_link_elitist", "level": 3,
        "seed": 0, "prompt": "the v2 prompt",
        "reference": "[answer]\na1: justified\n[/answer]",
        "min_directives": 8, "metadata": {"n_queried": 8},
    }


def _write(tmp_path, row):
    artifacts.write_sample_input(tmp_path, row)
    d = artifacts.sample_dir(tmp_path, row["sample_id"])
    return (d / "input.txt").read_text(), json.loads((d / "gold.json").read_text())


def test_v1_row_round_trips(tmp_path):
    prompt, gold = _write(tmp_path, _v1_row())
    assert prompt == "the v1 prompt"
    assert gold["gold_answer"] == "a1: justified"
    assert gold["metadata"] == {"n": 3}
    assert gold["mode"] == "symbolic" and gold["idx"] == 0


def test_v2_row_round_trips(tmp_path):
    prompt, gold = _write(tmp_path, _v2_row())
    assert prompt == "the v2 prompt"
    assert gold["gold_answer"] == "[answer]\na1: justified\n[/answer]"
    assert gold["metadata"] == {"n_queried": 8}
    assert gold["ordering"] == "last_link_elitist" and gold["seed"] == 0
    assert gold["min_directives"] == 8


def test_each_shape_omits_the_other_s_fields(tmp_path):
    """Absent, not null: a gold.json carrying `mode: null` invites a reader to
    treat v2 as a mode-less v1 run rather than a different benchmark."""
    _, v1 = _write(tmp_path, _v1_row())
    _, v2 = _write(tmp_path, _v2_row())
    for k in ("ordering", "seed", "min_directives"):
        assert k not in v1
    for k in ("mode", "idx"):
        assert k not in v2


def test_fields_common_to_both_are_always_written(tmp_path):
    for row in (_v1_row(), _v2_row()):
        _, gold = _write(tmp_path, row)
        for k in ("sample_id", "task", "level", "gold_answer"):
            assert k in gold, f"{k} missing for {row['sample_id']}"
