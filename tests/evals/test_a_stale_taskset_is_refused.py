"""A run refuses a taskset frozen under other rules than the installed scorer's.

A file frozen under an older `PROMPT_VERSION` asks questions that do not state
rules the scorer enforces; #154 found a shipped file two versions behind, whose
480 rows all scored zero for one stray word no question forbade.
"""
from __future__ import annotations

import json
import os

import pytest
from conftest import answering
from test_a_run_is_reachable_from_the_command_line import cfg_for

from arggym.core.freeze import PROMPT_VERSION, SCORING_VERSION
from evals.run import execute
from evals.score import score_run


def with_versions(src: str, dst: str, **versions) -> str:
    """A copy of `src` whose manifest records `versions` in place of its own."""
    with open(src) as f:
        lines = f.readlines()
    head = json.loads(lines[0])
    if versions.pop("drop", False):
        head["__manifest__"]["versions"] = {}
    head["__manifest__"]["versions"].update(versions)
    with open(dst, "w") as f:
        f.write(json.dumps(head) + "\n")
        f.writelines(lines[1:])
    return dst


def test_a_matching_taskset_runs(tmp_path, rows, taskset_file, provider):
    p = provider(answering(rows))
    meta = execute(cfg_for(taskset_file, p.url), os.fspath(tmp_path / "run"))
    assert meta["status"] == "completed"


@pytest.mark.parametrize("field,installed", [("prompt_version", PROMPT_VERSION),
                                             ("scoring_version", SCORING_VERSION)])
def test_a_stale_version_is_refused_before_any_call(tmp_path, rows, taskset_file,
                                                    provider, field, installed):
    p = provider(answering(rows))
    stale = with_versions(taskset_file, os.fspath(tmp_path / "stale.jsonl"),
                          **{field: installed - 1})
    with pytest.raises(SystemExit) as e:
        execute(cfg_for(stale, p.url), os.fspath(tmp_path / "run"))
    msg = str(e.value)
    assert f"{field} {installed - 1} (installed: {installed})" in msg
    assert stale in msg and "run_id=" in msg
    # stale.jsonl is no shipped file, so no make target rebuilds it.
    assert f"arggym freeze -c <spec> -o {stale}" in msg and "make freeze" not in msg
    assert p.requests == []


@pytest.mark.parametrize("shipped,target", [("taskset-lite.jsonl", "`make freeze-lite`"),
                                            ("taskset.jsonl", "`make freeze`")])
def test_a_stale_shipped_file_names_the_target_that_rebuilds_it(
        tmp_path, rows, taskset_file, provider, shipped, target):
    p = provider(answering(rows))
    (tmp_path / "data").mkdir()
    stale = with_versions(taskset_file, os.fspath(tmp_path / "data" / shipped),
                          prompt_version=PROMPT_VERSION - 1)
    with pytest.raises(SystemExit) as e:
        execute(cfg_for(stale, p.url), os.fspath(tmp_path / "run"))
    msg = str(e.value)
    assert target in msg and f"arggym freeze -c <spec> -o {stale}" in msg


def test_a_file_newer_than_the_checkout_asks_for_an_update(tmp_path, rows,
                                                           taskset_file, provider):
    p = provider(answering(rows))
    newer = with_versions(taskset_file, os.fspath(tmp_path / "newer.jsonl"),
                          prompt_version=PROMPT_VERSION + 1)
    with pytest.raises(SystemExit, match="update this checkout"):
        execute(cfg_for(newer, p.url), os.fspath(tmp_path / "run"))


def test_a_manifest_without_versions_is_warned_about(tmp_path, rows, taskset_file,
                                                     provider, capsys):
    """Every freeze writes both; their absence means a hand-built file, not a stale one."""
    p = provider(answering(rows))
    bare = with_versions(taskset_file, os.fspath(tmp_path / "bare.jsonl"), drop=True)
    meta = execute(cfg_for(bare, p.url), os.fspath(tmp_path / "run"))
    assert meta["status"] == "completed"
    assert "records no prompt_version or scoring_version" in capsys.readouterr().err


def test_a_file_with_no_manifest_is_refused(tmp_path, rows, taskset_file, provider):
    p = provider(answering(rows))
    with open(taskset_file) as f:
        lines = f.readlines()
    path = os.fspath(tmp_path / "headless.jsonl")
    with open(path, "w") as f:
        f.writelines(lines[1:])
    with pytest.raises(SystemExit, match="no manifest line"):
        execute(cfg_for(path, p.url), os.fspath(tmp_path / "run"))


def test_an_old_file_runs_on_request_and_rescores_under_the_installed_scorer(
        tmp_path, rows, taskset_file, provider, capsys):
    p = provider(answering(rows))
    stale = with_versions(taskset_file, os.fspath(tmp_path / "stale.jsonl"),
                          prompt_version=PROMPT_VERSION - 1)
    run_dir = os.fspath(tmp_path / "run")
    run = execute(cfg_for(stale, p.url, allow_stale_taskset="true"), run_dir)
    assert "running anyway" in capsys.readouterr().err
    assert run["allow_stale_taskset"] is True
    assert (run["prompt_version"], run["scoring_version"]) == (PROMPT_VERSION,
                                                               SCORING_VERSION)
    meta = score_run(run_dir)["_meta"]
    assert meta["scoring_version"] == SCORING_VERSION
    assert meta["taskset_versions"]["prompt_version"] == PROMPT_VERSION - 1


def test_a_run_made_under_the_override_does_not_resume_without_it(
        tmp_path, rows, taskset_file, provider):
    p = provider(answering(rows))
    run_dir = os.fspath(tmp_path / "run")
    execute(cfg_for(taskset_file, p.url, allow_stale_taskset="true"), run_dir)
    with pytest.raises(SystemExit, match="allow_stale_taskset"):
        execute(cfg_for(taskset_file, p.url), run_dir)
