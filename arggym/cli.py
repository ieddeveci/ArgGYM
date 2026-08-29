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


def _load_gates():
    """The optional local validation suite. It is not distributed with ArgGYM."""
    try:
        from validation import gates
    except ImportError:
        typer.echo("gates are unavailable: no importable 'validation' package", err=True)
        raise typer.Exit(1)
    return gates


@app.command()
def tasks() -> None:
    """List the tasks that can be exported."""
    from arggym.core import export

    for name in sorted(export._EXPORTABLE):
        typer.echo(name)


@app.command()
def export(
    task: str = typer.Argument(..., help="Task to export; see 'arggym tasks'."),
    path: Optional[Path] = typer.Argument(None, help="Output file. Defaults to data/<task>.jsonl."),
) -> None:
    """Export one task as JSONL."""
    from arggym.core import export as export_mod

    out = path if path is not None else DEFAULT_OUT_DIR / f"{task}.jsonl"
    export_mod.export_task(str(out), task)


@app.command("export-all")
def export_all(
    directory: Path = typer.Argument(DEFAULT_OUT_DIR, help="Directory to write the JSONL into."),
) -> None:
    """Export every task as JSONL."""
    from arggym.core import export as export_mod

    for task in sorted(export_mod._EXPORTABLE):
        export_mod.export_task(str(directory / f"{task}.jsonl"), task)


@app.command()
def inspect() -> None:
    """Start the inspector on http://127.0.0.1:5000."""
    from arggym import inspector

    inspector.app.run(debug=False, port=5000)


@app.command()
def gates() -> None:
    """List the gates in the optional local validation suite."""
    for name in sorted(_load_gates().ALL):
        typer.echo(name)


@app.command()
def gate(name: str = typer.Argument(..., help="Gate to run; see 'arggym gates'.")) -> None:
    """Run one gate from the optional local validation suite."""
    suite = _load_gates()
    if name not in suite.ALL:
        typer.echo(f"unknown gate {name!r}; try 'arggym gates'", err=True)
        raise typer.Exit(1)
    suite.ALL[name]()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
