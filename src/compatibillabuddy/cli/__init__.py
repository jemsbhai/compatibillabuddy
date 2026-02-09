"""CLI entry point for compatibillabuddy."""

import typer

from compatibillabuddy import __version__
from compatibillabuddy.cli.agent import agent_command
from compatibillabuddy.cli.doctor import doctor_command
from compatibillabuddy.cli.repair import repair_command

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
    version: bool | None = typer.Option(
        None,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
):
    """Hardware-aware dependency compatibility for Python ML stacks."""


app.command(name="doctor")(doctor_command)
app.command(name="agent")(agent_command)
app.command(name="repair")(repair_command)
