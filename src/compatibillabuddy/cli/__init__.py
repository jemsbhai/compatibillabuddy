"""CLI entry point for compatibillabuddy."""

from typing import Optional

import typer

from compatibillabuddy import __version__

app = typer.Typer(
    name="compatibuddy",
    help="Hardware-aware dependency compatibility for Python ML stacks.",
    no_args_is_help=True,
)


def version_callback(value: bool):
    if value:
        typer.echo(f"compatibillabuddy {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
):
    """Hardware-aware dependency compatibility for Python ML stacks."""
