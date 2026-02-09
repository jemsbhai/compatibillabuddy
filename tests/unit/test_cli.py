"""Verify CLI entry point works."""

from typer.testing import CliRunner

from compatibillabuddy.cli import app

runner = CliRunner()


def test_cli_no_args_shows_help():
    """Running with no arguments should show help text (exit code 2 = no args)."""
    result = runner.invoke(app)
    assert result.exit_code in (0, 2)
    assert "Usage" in result.output


def test_cli_version_flag():
    """The --version flag should print the current version."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.2.0" in result.stdout


def test_cli_version_short_flag():
    """The -V flag should also print the current version."""
    result = runner.invoke(app, ["-V"])
    assert result.exit_code == 0
    assert "0.2.0" in result.stdout
