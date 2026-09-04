from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

DEFAULT_OUT_DIR = Path("data")

app = typer.Typer(
    help="A generator and grader for defeasible-argumentation tasks in ASPIC+.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def tasks() -> None:
    """List the tasks that can be exported."""
    from arggym.core import export

    for name in sorted(export._EXPORTABLE):
        typer.echo(name)


ALLOW_MISSING_HELP = "Write the file even when some (level, ordering, seed) cells produced no item."


@app.command()
def export(
    task: str = typer.Argument(..., help="Task to export; see 'arggym tasks'."),
    path: Optional[Path] = typer.Argument(None, help="Output file. Defaults to data/<task>.jsonl."),
    allow_missing: bool = typer.Option(False, "--allow-missing", help=ALLOW_MISSING_HELP),
) -> None:
    """Export one task as JSONL."""
    from arggym.core import export as export_mod

    out = path if path is not None else DEFAULT_OUT_DIR / f"{task}.jsonl"
    export_mod.export_task(str(out), task, allow_missing=allow_missing)


@app.command("export-all")
def export_all(
    directory: Path = typer.Argument(DEFAULT_OUT_DIR, help="Directory to write the JSONL into."),
    allow_missing: bool = typer.Option(False, "--allow-missing", help=ALLOW_MISSING_HELP),
) -> None:
    """Export every task as JSONL. Attempts every task; fails at the end if any had missing cells."""
    from arggym.core import export as export_mod

    failed = []
    for task in sorted(export_mod._EXPORTABLE):
        try:
            export_mod.export_task(str(directory / f"{task}.jsonl"), task,
                                   allow_missing=allow_missing)
        except SystemExit as e:
            typer.echo(str(e), err=True)
            failed.append(task)
    if failed:
        typer.echo(f"export-all: {len(failed)} task(s) not written for missing cells: "
                   f"{', '.join(failed)}", err=True)
        raise typer.Exit(1)


@app.command()
def freeze(
    config: Path = typer.Option(Path("tasksets/standard.yaml"), "--config", "-c",
                                help="Taskset spec to build."),
    out: Path = typer.Option(Path("data/taskset.jsonl"), "--out", "-o",
                             help="Output JSONL."),
) -> None:
    """Build a taskset from a spec, into one JSONL with a manifest line."""
    from arggym.core.freeze import freeze as run
    from arggym.core.spec import load

    run(load(str(config)), str(out))


@app.command()
def floors(
    taskset: Path = typer.Argument(..., help="A frozen taskset JSONL."),
) -> None:
    """What a constant answer scores on each task.

    A score means nothing without the number an uninformed answer gets, and a
    floor is a property of the scorer, so it moves whenever scoring policy does.
    """
    import json

    from arggym.core.floors import floors as measure

    rows = []
    with open(taskset) as f:
        first = json.loads(f.readline())
        if "__manifest__" not in first:
            rows.append(first)
        rows.extend(json.loads(line) for line in f)

    typer.echo(f"{'task':26s} {'n':>4s} {'floor':>7s}  best constant answer")
    for task, v in measure(rows).items():
        typer.echo(f"{task:26s} {v['n']:4d} {v['floor']:7.3f}  {v['strategy']}")
    typer.echo("\nReport these beside the scores. A result below its floor is "
               "worse than answering the same thing every time.")


@app.command()
def inspect() -> None:
    """Start the inspector on http://127.0.0.1:5000."""
    try:
        from arggym import inspector
    except ImportError:
        # Flask is an extra, so scoring model outputs in CI does not pull a web
        # framework (#54). Only this one command needs it.
        typer.echo("the inspector needs Flask: pip install 'arggym[inspector]'", err=True)
        raise typer.Exit(1)

    inspector.app.run(debug=False, port=5000)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
