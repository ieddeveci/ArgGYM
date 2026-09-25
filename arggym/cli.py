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
    """List the registered task names."""
    from arggym.core import registry

    for name in registry.task_names():
        typer.echo(name)


@app.command()
def freeze(
    config: Path = typer.Option(Path("data/taskset.yaml"), "--config", "-c",
                                help="Taskset spec to build."),
    out: Path = typer.Option(Path("data/taskset.jsonl"), "--out", "-o",
                             help="Output JSONL."),
    workers: Optional[int] = typer.Option(
        None, "--workers", "-j", min=1, show_default=False,
        help="Processes filling cells. 1 is a serial freeze. The output is the "
             "same file whatever the count. Defaults to the cores this process "
             "may use."),
) -> None:
    """Build a taskset from a spec, into one JSONL with a manifest line."""
    from arggym.core.freeze import freeze as run
    from arggym.core.freeze import usable_cores
    from arggym.core.spec import load

    run(load(str(config)), str(out), workers=workers or usable_cores())


@app.command()
def floors(
    taskset: Path = typer.Argument(..., help="A frozen taskset JSONL."),
) -> None:
    """What an uninformed answer scores on each task.

    A score means nothing without the number an uninformed answer gets, and a
    floor is a property of the scorer, so it moves whenever scoring policy does.
    """
    import json

    from arggym.core.floors import floor_strategy
    from arggym.core.floors import floors as measure

    rows = []
    with open(taskset) as f:
        first = json.loads(f.readline())
        if "__manifest__" not in first:
            rows.append(first)
        rows.extend(json.loads(line) for line in f)

    # The strategy, not just the number: a floor of 0.000 from `empty` says the
    # search found nothing that fits the answer format, and a floor a fitted map
    # reached is only readable beside the map.
    typer.echo(f"{'task':26s} {'n':>4s} {'floor':>7s}  best uninformed answer")
    for task, v in measure(rows).items():
        typer.echo(f"{task:26s} {v['n']:4d} {v['floor']:7.3f}  {floor_strategy(v)}")
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
