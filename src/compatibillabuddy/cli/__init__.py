"""CLI entry point for compatibillabuddy."""

import typer

app = typer.Typer(
    name="compatibuddy",
    help="Hardware-aware dependency compatibility for Python ML stacks.",
    no_args_is_help=True,
)


@app.command()
def version():
    """Show version information."""
    from compatibillabuddy import __version__

    typer.echo(f"compatibillabuddy {__version__}")
