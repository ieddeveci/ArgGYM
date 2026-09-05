"""The CLI wiring: which commands exist, and what they refuse.

Generation itself is covered by test_smoke; the freeze path by
test_a_thin_cell_is_a_finding_not_a_setting.
"""
from __future__ import annotations

import json

from typer.testing import CliRunner

from arggym.cli import app
from arggym.core import registry

runner = CliRunner()


def test_tasks_lists_every_registered_task():
    result = runner.invoke(app, ["tasks"])
    assert result.exit_code == 0
    assert result.output.split() == list(registry.task_names())


def test_freeze_writes_a_taskset_from_a_spec(tmp_path):
    out = tmp_path / "t.jsonl"
    result = runner.invoke(app, ["freeze", "-c", "tests/data/one-cheap-cell.yaml",
                                 "-o", str(out)])
    assert result.exit_code == 0, result.output
    lines = out.read_text().splitlines()
    assert "__manifest__" in json.loads(lines[0])
    assert len(lines) == 13  # manifest plus 6 cells x 2 items


def test_a_spec_naming_an_unknown_task_is_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"tasks": ["bogus"], "levels": [3],
                               "orderings": ["last_link_elitist"]}))
    result = runner.invoke(app, ["freeze", "-c", str(bad), "-o", str(tmp_path / "x.jsonl")])
    assert result.exit_code != 0
    assert "unknown task" in str(result.exception) + result.output


def test_floors_reports_per_task(tmp_path):
    out = tmp_path / "t.jsonl"
    assert runner.invoke(app, ["freeze", "-c", "tests/data/one-cheap-cell.yaml",
                               "-o", str(out)]).exit_code == 0
    result = runner.invoke(app, ["floors", str(out)])
    assert result.exit_code == 0
    assert "status_query" in result.output
    assert "claim_chain" in result.output


def test_the_cli_offers_no_command_it_cannot_run():
    # `gates` and `gate` imported a `validation` package that is not in the repo,
    # so a fresh clone met two commands that could only fail (#52).
    #
    # `export` and `export-all` are gone for a different reason: they wrote
    # `goals`, `min_directives` and the raw statistics blob at the top level of
    # every row, so a taskset written that way leaked the answer size. `freeze`
    # puts all of it under `metadata.gold`, and is the only way to write one now.
    # Read the commands off the app rather than out of its rendered help. Help is
    # drawn by rich, which picks its box characters from the terminal's width,
    # encoding and colour support, so a test that scrapes those characters is
    # testing the terminal. This one passed locally and found no commands at all
    # on CI, where the same help renders without them.
    names = {c.name or c.callback.__name__.replace("_", "-")
             for c in app.registered_commands}
    assert names == {"tasks", "freeze", "floors", "inspect"}, names
    # Offering a command means it runs. Each one loads its own imports lazily, so
    # `--help` is what proves the module behind it is importable at all.
    for name in sorted(names):
        assert runner.invoke(app, [name, "--help"]).exit_code == 0, name
    for gone in (["gates"], ["export", "status_query"], ["export-all", "somewhere"]):
        assert runner.invoke(app, gone).exit_code != 0
