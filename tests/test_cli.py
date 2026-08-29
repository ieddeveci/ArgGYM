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
    with patch("arggym.core.export.export_task", lambda path, task: calls.append((task, path))):
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


def test_gates_report_the_missing_validation_package():
    result = runner.invoke(app, ["gates"])
    assert result.exit_code == 1
    assert "validation" in result.output
