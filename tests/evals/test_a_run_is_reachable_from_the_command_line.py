"""What `run.py` does as a whole, through a composed config.

Every other test here calls `generate` and hand-writes `run.json`, which left
the run's own behaviour untested: the manifest, the prompts written before any
call, resume, and the error-rate gate. Three defects lived in that gap, the
worst being a resume that could never fire because the run directory carried a
timestamp.
"""
from __future__ import annotations

import json
import os

import pytest
from conftest import answering
from hydra import compose, initialize_config_dir

from evals import artifacts
from evals.run import RunFailed, execute

CONF = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "evals", "conf")


def cfg_for(taskset_file, url, **overrides):
    over = [f"taskset={taskset_file}", "model=stub", f"model.base_url={url}",
            "generation.concurrency=2", "endpoint.retries=0",
            "hydra.job.chdir=false"]
    over += [f"{k}={v}" for k, v in overrides.items()]
    with initialize_config_dir(version_base=None, config_dir=CONF):
        return compose(config_name="config", overrides=over)


def test_a_second_identical_run_regenerates_nothing(tmp_path, rows, taskset_file,
                                                    provider):
    """Resume is the feature the run directory's name exists to make possible."""
    p = provider(answering(rows))
    run_dir = os.fspath(tmp_path / "run")

    first = execute(cfg_for(taskset_file, p.url), run_dir)
    assert first["n_generated"] == len(rows)
    assert first["n_resumed"] == 0
    calls = len(p.requests)

    second = execute(cfg_for(taskset_file, p.url), run_dir)
    assert len(p.requests) == calls, "regenerated work already paid for"
    assert second["n_resumed"] == len(rows)
    # The manifest describes the run, not the last invocation. It read
    # `n_generated: 0` when it counted only what this pass did.
    assert second["n_generated"] == len(rows)
    assert second["status"] == "completed"


def test_the_run_directory_is_named_by_what_identifies_the_run():
    """A timestamped directory made resume unreachable through the CLI."""
    import yaml

    with open(os.path.join(CONF, "config.yaml")) as f:
        raw = f.read()
    conf = yaml.safe_load(raw)
    assert "now:" not in conf["hydra"]["run"]["dir"], conf["hydra"]["run"]["dir"]
    assert conf["hydra"]["run"]["dir"] == "outputs/runs/${run_id}"
    for part in ("model.name", "template.name", "elicitation.name"):
        assert part in conf["run_id"], part


def test_a_directory_will_not_take_a_second_configuration(tmp_path, rows,
                                                          taskset_file, provider):
    """Two models under one manifest is a mixture nothing downstream can undo."""
    p = provider(answering(rows))
    run_dir = os.fspath(tmp_path / "run")
    execute(cfg_for(taskset_file, p.url), run_dir)

    with pytest.raises(SystemExit) as e:
        execute(cfg_for(taskset_file, p.url, **{"model.model": "a-different-model"}),
                run_dir)
    assert "endpoint.model" in str(e.value)

    with pytest.raises(SystemExit) as e:
        execute(cfg_for(taskset_file, p.url, template="square_tags"), run_dir)
    assert "template" in str(e.value)


def test_not_resuming_discards_the_old_generations(tmp_path, rows, taskset_file,
                                                   provider):
    p = provider(answering(rows))
    run_dir = os.fspath(tmp_path / "run")
    execute(cfg_for(taskset_file, p.url), run_dir)
    # A different model would be refused on resume; with resume off the
    # directory starts over, so it is allowed.
    meta = execute(cfg_for(taskset_file, p.url, resume="false",
                           **{"model.model": "another"}), run_dir)
    assert meta["n_resumed"] == 0
    assert meta["n_generated"] == len(rows)
    assert len(list(artifacts.read_jsonl(
        os.path.join(run_dir, artifacts.GENERATIONS)))) == len(rows)


def test_the_prompts_are_on_disk_before_any_call(tmp_path, rows, taskset_file,
                                                 provider):
    """A run that dies mid-flight must still say what it asked."""
    seen = {}

    def note(body):
        seen["prompts"] = list(artifacts.read_jsonl(
            os.path.join(run_dir, artifacts.PROMPTS)))
        return answering(rows)(body)

    p = provider(note)
    run_dir = os.fspath(tmp_path / "run")
    execute(cfg_for(taskset_file, p.url), run_dir)
    assert len(seen["prompts"]) == len(rows), "prompts were written after the calls"
    assert all(r["user"] for r in seen["prompts"])


def test_a_run_that_could_not_reach_the_provider_is_marked_failed(tmp_path, rows,
                                                                  taskset_file,
                                                                  provider):
    """A dead endpoint and a bad model give the same low score; only one is a finding."""
    p = provider(lambda body: {"__status__": 503, "error": {"message": "down"}})
    run_dir = os.fspath(tmp_path / "run")
    with pytest.raises(RunFailed) as e:
        execute(cfg_for(taskset_file, p.url), run_dir)
    assert "max_error_rate" in str(e.value)

    with open(os.path.join(run_dir, artifacts.RUN)) as f:
        meta = json.load(f)
    assert meta["status"] == "failed"
    assert meta["error_rate"] == 1.0
    # The generations are kept, so the run is resumable.
    assert os.path.exists(os.path.join(run_dir, artifacts.GENERATIONS))


def test_the_manifest_records_the_taskset_it_actually_read(tmp_path, rows,
                                                           taskset_file, provider,
                                                           monkeypatch):
    monkeypatch.setenv("ARGGYM_STUB_KEY", "sk-must-not-be-written-down")
    p = provider(answering(rows))
    run_dir = os.fspath(tmp_path / "run")
    meta = execute(cfg_for(taskset_file, p.url), run_dir)
    from evals import taskset as ts

    manifest, _ = ts.load(taskset_file)
    assert meta["taskset_hash"] == manifest["taskset_hash"]
    assert meta["template"] == "xml_tags"

    # The manifest names the variable and never its value, and the file on disk
    # is what a run gets published with.
    with open(os.path.join(run_dir, artifacts.RUN)) as f:
        written = f.read()
    assert "sk-must-not-be-written-down" not in written
    assert meta["endpoint"]["api_key_env"] == "ARGGYM_STUB_KEY"
