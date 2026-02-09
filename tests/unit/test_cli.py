"""Verify CLI entry point works."""

from typer.testing import CliRunner

from compatibillabuddy.cli import app

runner = CliRunner()


def test_cli_no_args_shows_help():
    """Running with no arguments should show help text."""
    result = runner.invoke(app)
    assert result.exit_code == 0
    assert "Hardware-aware" in result.stdout or "Usage" in result.stdout


def test_cli_version_command():
    """The version command should print the current version."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout
