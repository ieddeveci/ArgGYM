"""The CLI wiring: command names, and the paths export writes to.

Generation itself is covered by test_smoke; here export_task is stubbed out so these stay fast.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from arggym.cli import app

runner = CliRunner()


@pytest.fixture
def written():
    """Capture (task, path) pairs instead of generating anything."""
    calls = []
    with patch("arggym.core.export.export_task",
               lambda path, task, allow_missing=False: calls.append((task, path))):
        yield calls


def test_tasks_lists_every_exportable_task():
    from arggym.core import export

    result = runner.invoke(app, ["tasks"])
    assert result.exit_code == 0
    assert result.output.split() == sorted(export._EXPORTABLE)


def test_export_defaults_to_data_dir(written):
    assert runner.invoke(app, ["export", "status_query"]).exit_code == 0
    assert written == [("status_query", "data/status_query.jsonl")]


def test_export_honours_an_explicit_path(written):
    assert runner.invoke(app, ["export", "status_query", "out.jsonl"]).exit_code == 0
    assert written == [("status_query", "out.jsonl")]


def test_export_all_covers_every_task_under_one_directory(written):
    from arggym.core import export

    assert runner.invoke(app, ["export-all", "somewhere"]).exit_code == 0
    assert [t for t, _ in written] == sorted(export._EXPORTABLE)
    assert all(p.startswith("somewhere/") for _, p in written)


def test_unknown_task_is_rejected():
    """Not stubbed: the rejection lives in export_task, and it fires before any generation."""
    result = runner.invoke(app, ["export", "bogus"])
    assert result.exit_code != 0
    assert "unknown task bogus" in str(result.exception) + result.output


def test_the_cli_offers_no_command_it_cannot_run():
    # `gates` and `gate` imported a `validation` package that is not in the repo
    # and not distributed, so a fresh clone met two commands that could only
    # fail. A benchmark whose quality gates are invisible to the people meant to
    # trust it is worse than one that does not advertise them (#52).
    listed = runner.invoke(app, ["--help"]).output
    for gone in ("gates", "gate "):
        assert gone not in listed
    assert runner.invoke(app, ["gates"]).exit_code != 0


def test_allow_missing_is_passed_through():
    seen = []
    with patch("arggym.core.export.export_task",
               lambda path, task, allow_missing=False: seen.append(allow_missing)):
        assert runner.invoke(app, ["export", "status_query", "--allow-missing"]).exit_code == 0
        assert runner.invoke(app, ["export", "status_query"]).exit_code == 0
        assert runner.invoke(app, ["export-all", "somewhere", "--allow-missing"]).exit_code == 0
    n = len(seen)
    assert seen[:2] == [True, False] and all(seen[2:]) and n > 2


def test_export_all_attempts_every_task_and_fails_at_the_end():
    """A task with missing cells must not stop the others, but must make the command fail."""
    from arggym.core import export

    attempted = []

    def stub(path, task, allow_missing=False):
        attempted.append(task)
        if task == "status_query":
            raise SystemExit(f"{task}: 2 of 20 cells returned no item")

    with patch("arggym.core.export.export_task", stub):
        result = runner.invoke(app, ["export-all", "somewhere"])
    assert result.exit_code == 1
    assert attempted == sorted(export._EXPORTABLE)
    assert "status_query" in result.output
