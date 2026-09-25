"""A run packed into results/ and unpacked again scores to the same metrics.

`scripts/results.sh` keeps three files of a run and rebuilds the rest, so the
thing to check is that those three are enough: a teammate who unpacks a shared
run and rescores it has to get the numbers that were committed beside it.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest
from conftest import answering
from test_a_reference_answer_scores_one_end_to_end import a_run

from evals.score import score_run

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "results.sh")

pytestmark = pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd is not installed")


def results(cmd, *ids, runs, packed, check=True):
    env = {**os.environ, "RUNS_DIR": os.fspath(runs), "RESULTS_DIR": os.fspath(packed)}
    return subprocess.run(["bash", SCRIPT, cmd, *ids], env=env, check=check,
                          capture_output=True, text=True)


def test_an_unpacked_run_rescores_to_its_committed_metrics(tmp_path, rows, taskset_file,
                                                          provider):
    p = provider(answering(rows))
    runs, packed, restored = tmp_path / "runs", tmp_path / "results", tmp_path / "restored"
    original = runs / "stub__xml_tags__cot"
    os.makedirs(runs)
    os.rename(a_run(tmp_path, p.url, rows, taskset_file), original)
    want = score_run(os.fspath(original))

    results("add", original.name, runs=runs, packed=packed)
    assert sorted(os.listdir(packed / original.name)) == [
        "generations.jsonl.zst", "metrics.json", "run.json"]

    results("unpack", runs=restored, packed=packed)
    got = score_run(os.fspath(restored / original.name))
    for m in (want, got):
        del m["_meta"]["run_dir"]
    assert json.loads(json.dumps(got)) == json.loads(json.dumps(want))


def test_only_a_completed_run_is_packed(tmp_path, rows, taskset_file, provider):
    p = provider(answering(rows))
    runs = tmp_path / "runs"
    run = runs / "live"
    os.makedirs(runs)
    os.rename(a_run(tmp_path, p.url, rows, taskset_file), run)
    score_run(os.fspath(run))
    manifest = json.loads((run / "run.json").read_text())
    (run / "run.json").write_text(json.dumps({**manifest, "status": "running"}, indent=2))

    out = results("add", "live", runs=runs, packed=tmp_path / "results", check=False)
    assert out.returncode != 0
    assert "completed" in out.stderr
    assert not (tmp_path / "results" / "live").exists()
