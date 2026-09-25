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


def results(cmd, *ids, runs, packed, repo, check=True, env=None):
    env = {**os.environ, "RUNS_DIR": os.fspath(runs), "RESULTS_DIR": os.fspath(packed),
           **(env or {})}
    return subprocess.run(["bash", SCRIPT, cmd, *ids], env=env, cwd=repo, check=check,
                          capture_output=True, text=True)


def a_repo(tmp_path, taskset_file):
    """A checkout with the taskset as data/taskset.jsonl and Git LFS set up.

    The script packs against the checkout it is called in, so the stub taskset
    has to sit where a real one does: directly under the repo's data/.
    """
    repo = tmp_path / "repo"
    os.makedirs(repo / "data")
    shutil.copy(taskset_file, repo / "data" / "taskset.jsonl")
    subprocess.run(["git", "init", "-q", os.fspath(repo)], check=True)
    subprocess.run(["git", "-C", os.fspath(repo), "config", "filter.lfs.clean",
                    "git-lfs clean -- %f"], check=True)
    return repo


def a_packable_run(tmp_path, url, rows, taskset_file, name):
    """A finished run in <tmp>/runs/<name>, generated against <repo>/data/taskset.jsonl."""
    repo = a_repo(tmp_path, taskset_file)
    runs = tmp_path / "runs"
    os.makedirs(runs)
    run = runs / name
    os.rename(a_run(tmp_path, url, rows, os.fspath(repo / "data" / "taskset.jsonl")), run)
    return repo, runs, run


def test_an_unpacked_run_rescores_to_its_committed_metrics(tmp_path, rows, taskset_file,
                                                          provider, monkeypatch):
    p = provider(answering(rows))
    repo, runs, original = a_packable_run(tmp_path, p.url, rows, taskset_file,
                                          "stub__xml_tags__cot")
    packed, restored = tmp_path / "results", tmp_path / "restored"
    # A 429 that a later retry superseded, which scoring has to keep ignoring.
    gens = original / "generations.jsonl"
    lines = gens.read_text().splitlines(keepends=True)
    failed = {**json.loads(lines[0]), "error": "HTTP 429: slow down", "error_kind": "http_429"}
    gens.write_text(json.dumps(failed) + "\n" + "".join(lines))
    want = score_run(os.fspath(original))

    results("add", original.name, runs=runs, packed=packed, repo=repo)
    assert sorted(os.listdir(packed / original.name)) == [
        "generations.jsonl.zst", "metrics.json", "run.json"]
    assert json.loads((packed / original.name / "run.json").read_text())["taskset"] == \
        "data/taskset.jsonl"
    meta = json.loads((packed / original.name / "metrics.json").read_text())["_meta"]
    assert (meta["taskset"], meta["run_dir"]) == (
        "data/taskset.jsonl", f"outputs/runs/{original.name}")

    results("unpack", runs=restored, packed=packed, repo=repo)
    # The unpacked metrics.json is what a teammate reads without rescoring.
    committed = json.loads((restored / original.name / "metrics.json").read_text())
    monkeypatch.chdir(repo)  # the packed taskset path is relative to the repo root
    got = score_run(os.fspath(restored / original.name))
    for m in (want, got, committed):
        del m["_meta"]["run_dir"], m["_meta"]["taskset"]
    want = json.loads(json.dumps(want))
    assert json.loads(json.dumps(got)) == want
    assert committed == want


def test_only_a_completed_run_is_packed(tmp_path, rows, taskset_file, provider):
    p = provider(answering(rows))
    repo, runs, run = a_packable_run(tmp_path, p.url, rows, taskset_file, "live")
    score_run(os.fspath(run))
    manifest = json.loads((run / "run.json").read_text())
    (run / "run.json").write_text(json.dumps({**manifest, "status": "running"}, indent=2))

    out = results("add", "live", runs=runs, packed=tmp_path / "results", repo=repo,
                  check=False)
    assert out.returncode != 0
    assert "completed" in out.stderr
    assert not (tmp_path / "results" / "live").exists()


def test_a_run_whose_taskset_the_repo_lacks_is_not_packed(tmp_path, rows, taskset_file,
                                                         provider):
    """The committed copy has another hash, or the run's taskset has no committed copy."""
    p = provider(answering(rows))
    repo, runs, run = a_packable_run(tmp_path, p.url, rows, taskset_file, "run")
    score_run(os.fspath(run))
    manifest = json.loads((run / "run.json").read_text())
    packed = tmp_path / "results"

    (run / "run.json").write_text(json.dumps({**manifest, "taskset_hash": "0" * 32}, indent=2))
    out = results("add", "run", runs=runs, packed=packed, repo=repo, check=False)
    assert out.returncode != 0
    assert "0" * 32 in out.stderr and manifest["taskset_hash"] in out.stderr

    for elsewhere in ("/elsewhere/taskset.jsonl", "/elsewhere/data/sub/taskset.jsonl",
                      "/elsewhere/data/other.jsonl"):
        (run / "run.json").write_text(json.dumps({**manifest, "taskset": elsewhere}, indent=2))
        out = results("add", "run", runs=runs, packed=packed, repo=repo, check=False)
        assert out.returncode != 0, elsewhere
        assert elsewhere in out.stderr
    assert not (packed / "run").exists()


def test_nothing_is_packed_without_git_lfs(tmp_path, rows, taskset_file, provider):
    """Git ignores an undefined `filter=lfs`, so the .zst would land in plain git."""
    p = provider(answering(rows))
    repo, runs, run = a_packable_run(tmp_path, p.url, rows, taskset_file, "run")
    score_run(os.fspath(run))
    subprocess.run(["git", "-C", os.fspath(repo), "config", "--unset", "filter.lfs.clean"],
                   check=True)
    out = results("add", "run", runs=runs, packed=tmp_path / "results", repo=repo,
                  check=False, env={"GIT_CONFIG_GLOBAL": os.devnull,
                                    "GIT_CONFIG_NOSYSTEM": "1"})
    assert out.returncode != 0
    assert "git lfs install" in out.stderr
    assert not (tmp_path / "results" / "run").exists()
