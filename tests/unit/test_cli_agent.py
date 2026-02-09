"""Tests for CLI agent command — interactive chat REPL.

TDD: tests written BEFORE implementation.
All tests mock AgentSession to avoid real API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from compatibillabuddy.cli import app

runner = CliRunner()

AGENT_MODULE = "compatibillabuddy.cli.agent"


# ===========================================================================
# Tests
# ===========================================================================


class TestAgentCommandHelp:
    """Basic CLI registration and help tests."""

    def test_agent_appears_in_help(self):
        result = runner.invoke(app, ["--help"])
        assert "agent" in result.output

    def test_agent_help_text(self):
        result = runner.invoke(app, ["agent", "--help"])
        assert result.exit_code == 0
        assert "chat" in result.output.lower() or "interactive" in result.output.lower()


class TestAgentCommandErrors:
    """Tests for error handling in agent command."""

    def test_agent_missing_api_key(self):
        """Exits with error when no API key is available."""
        with patch(
            f"{AGENT_MODULE}.resolve_api_key",
            side_effect=ValueError("No API key found"),
        ):
            result = runner.invoke(app, ["agent"], input="\n")

        assert result.exit_code == 1
        assert "api key" in result.output.lower() or "GEMINI_API_KEY" in result.output

    def test_agent_model_option(self):
        """--model option is accepted."""
        mock_session = MagicMock()
        mock_session.chat.return_value = "Hello!"

        with (
            patch(f"{AGENT_MODULE}.AgentSession", return_value=mock_session),
            patch(f"{AGENT_MODULE}.resolve_api_key", return_value="fake-key"),
        ):
            # Send "exit" immediately to quit the REPL
            result = runner.invoke(app, ["agent", "--model", "gemini-2.5-pro"], input="exit\n")

        assert result.exit_code == 0


class TestAgentCommandInteraction:
    """Tests for the interactive chat loop."""

    def test_agent_single_turn(self):
        """Agent processes one message then exits on 'exit'."""
        mock_session = MagicMock()
        mock_session.chat.return_value = "You have an NVIDIA RTX 4090."

        with (
            patch(f"{AGENT_MODULE}.AgentSession", return_value=mock_session),
            patch(f"{AGENT_MODULE}.resolve_api_key", return_value="fake-key"),
        ):
            result = runner.invoke(app, ["agent"], input="What GPU do I have?\nexit\n")

        assert result.exit_code == 0
        assert "RTX 4090" in result.output

    def test_agent_quit_command(self):
        """'quit' also exits the REPL."""
        mock_session = MagicMock()

        with (
            patch(f"{AGENT_MODULE}.AgentSession", return_value=mock_session),
            patch(f"{AGENT_MODULE}.resolve_api_key", return_value="fake-key"),
        ):
            result = runner.invoke(app, ["agent"], input="quit\n")

        assert result.exit_code == 0
        mock_session.chat.assert_not_called()
