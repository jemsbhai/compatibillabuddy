"""Tests for CLI repair command — non-interactive autonomous repair.

TDD: tests written BEFORE implementation.
All tests mock AgentSession to avoid real API calls.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from compatibillabuddy.cli import app

runner = CliRunner()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPAIR_MODULE = "compatibillabuddy.cli.repair"


def _mock_repair_result(
    *, success: bool = True, summary: str = "All fixed.", log: list | None = None
):
    """Build a mock RepairResult."""
    from compatibillabuddy.agent.core import RepairResult

    return RepairResult(
        summary=summary,
        repair_log=log or [],
        success=success,
    )


def _patch_session(repair_result):
    """Return a context manager that patches AgentSession for the repair CLI."""
    mock_session = MagicMock()
    mock_session.auto_repair.return_value = repair_result

    mock_cls = MagicMock(return_value=mock_session)
    return patch(f"{REPAIR_MODULE}.AgentSession", mock_cls), mock_session


# ===========================================================================
# Tests
# ===========================================================================


class TestRepairCommandHelp:
    """Basic CLI registration and help tests."""

    def test_repair_appears_in_help(self):
        result = runner.invoke(app, ["--help"])
        assert "repair" in result.output

    def test_repair_help_text(self):
        result = runner.invoke(app, ["repair", "--help"])
        assert result.exit_code == 0
        assert "repair" in result.output.lower() or "autonomous" in result.output.lower()


class TestRepairCommandExecution:
    """Tests for repair command execution flow."""

    def test_repair_dry_run_default(self):
        """Default invocation uses dry_run=True."""
        repair_result = _mock_repair_result()
        session_patch, mock_session = _patch_session(repair_result)

        with (
            session_patch,
            patch(f"{REPAIR_MODULE}.resolve_api_key", return_value="fake-key"),
        ):
            result = runner.invoke(app, ["repair"])

        assert result.exit_code == 0
        mock_session.auto_repair.assert_called_once()
        call_kwargs = mock_session.auto_repair.call_args[1]
        assert call_kwargs["dry_run"] is True

    def test_repair_live_mode(self):
        """--live flag sets dry_run=False."""
        repair_result = _mock_repair_result()
        session_patch, mock_session = _patch_session(repair_result)

        with (
            session_patch,
            patch(f"{REPAIR_MODULE}.resolve_api_key", return_value="fake-key"),
        ):
            result = runner.invoke(app, ["repair", "--live"])

        assert result.exit_code == 0
        call_kwargs = mock_session.auto_repair.call_args[1]
        assert call_kwargs["dry_run"] is False

    def test_repair_max_retries(self):
        """--max-retries is passed through to auto_repair."""
        repair_result = _mock_repair_result()
        session_patch, mock_session = _patch_session(repair_result)

        with (
            session_patch,
            patch(f"{REPAIR_MODULE}.resolve_api_key", return_value="fake-key"),
        ):
            result = runner.invoke(app, ["repair", "--max-retries", "7"])

        assert result.exit_code == 0
        call_kwargs = mock_session.auto_repair.call_args[1]
        assert call_kwargs["max_retries"] == 7

    def test_repair_exit_code_0_on_success(self):
        """Successful repair returns exit code 0."""
        repair_result = _mock_repair_result(success=True)
        session_patch, _ = _patch_session(repair_result)

        with (
            session_patch,
            patch(f"{REPAIR_MODULE}.resolve_api_key", return_value="fake-key"),
        ):
            result = runner.invoke(app, ["repair"])

        assert result.exit_code == 0

    def test_repair_exit_code_1_on_failure(self):
        """Failed repair returns exit code 1."""
        repair_result = _mock_repair_result(success=False, summary="Could not fix all issues.")
        session_patch, _ = _patch_session(repair_result)

        with (
            session_patch,
            patch(f"{REPAIR_MODULE}.resolve_api_key", return_value="fake-key"),
        ):
            result = runner.invoke(app, ["repair"])

        assert result.exit_code == 1

    def test_repair_shows_summary(self):
        """Output contains the agent's summary text."""
        repair_result = _mock_repair_result(summary="Fixed 3 issues. Env is healthy.")
        session_patch, _ = _patch_session(repair_result)

        with (
            session_patch,
            patch(f"{REPAIR_MODULE}.resolve_api_key", return_value="fake-key"),
        ):
            result = runner.invoke(app, ["repair"])

        assert "Fixed 3 issues" in result.output

    def test_repair_json_format(self):
        """--format json outputs structured JSON with summary and log."""
        log = [
            {"tool": "tool_snapshot_environment", "args": {}, "result": {"packages": []}},
            {"tool": "tool_run_doctor", "args": {}, "result": {"issues": []}},
        ]
        repair_result = _mock_repair_result(summary="All clear.", log=log)
        session_patch, _ = _patch_session(repair_result)

        with (
            session_patch,
            patch(f"{REPAIR_MODULE}.resolve_api_key", return_value="fake-key"),
        ):
            result = runner.invoke(app, ["repair", "--format", "json"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "summary" in parsed
        assert "repair_log" in parsed
        assert "success" in parsed


class TestRepairCommandErrors:
    """Tests for error handling in repair command."""

    def test_repair_missing_api_key(self):
        """Exits with error message when no API key is available."""
        with patch(
            f"{REPAIR_MODULE}.resolve_api_key",
            side_effect=ValueError("No API key found"),
        ):
            result = runner.invoke(app, ["repair"])

        assert result.exit_code == 1
        assert "api key" in result.output.lower() or "GEMINI_API_KEY" in result.output

    def test_repair_model_option(self):
        """--model option is passed through to AgentConfig."""
        repair_result = _mock_repair_result()
        session_patch, _ = _patch_session(repair_result)

        with (
            session_patch as mock_cls_patch,
            patch(f"{REPAIR_MODULE}.resolve_api_key", return_value="fake-key"),
        ):
            result = runner.invoke(app, ["repair", "--model", "gemini-2.5-pro"])

        assert result.exit_code == 0
        # Check AgentConfig was created with the right model
        call_args = mock_cls_patch.call_args
        config_arg = call_args[0][0] if call_args[0] else call_args[1].get("config")
        assert config_arg.model == "gemini-2.5-pro"
