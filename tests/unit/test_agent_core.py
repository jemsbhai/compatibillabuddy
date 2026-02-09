"""Tests for agent core — AgentSession with mocked Gemini client.

TDD: tests written BEFORE implementation in agent/core.py.
All tests mock google.genai — no real API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from compatibillabuddy.agent.config import AgentConfig

# ---------------------------------------------------------------------------
# Helpers to build mock Gemini responses
# ---------------------------------------------------------------------------


def _mock_text_response(text: str) -> MagicMock:
    """Build a mock Gemini response with only text, no tool calls."""
    part = MagicMock()
    part.text = text
    part.function_call = None

    candidate = MagicMock()
    candidate.content.parts = [part]

    response = MagicMock()
    response.candidates = [candidate]
    return response


def _mock_tool_call_response(fn_name: str, fn_args: dict) -> MagicMock:
    """Build a mock Gemini response with a single function call."""
    fc = MagicMock()
    fc.name = fn_name
    fc.args = fn_args

    part = MagicMock()
    part.text = None
    part.function_call = fc

    candidate = MagicMock()
    candidate.content.parts = [part]

    response = MagicMock()
    response.candidates = [candidate]
    return response


def _make_config() -> AgentConfig:
    return AgentConfig(api_key="fake-key-for-testing")


# ===========================================================================
# Tests
# ===========================================================================


class TestAgentSessionInit:
    """Tests for AgentSession initialization."""

    def test_session_init_stores_config(self):
        from compatibillabuddy.agent.core import AgentSession

        config = _make_config()
        with patch("compatibillabuddy.agent.core.genai"):
            session = AgentSession(config)
        assert session.config is config

    def test_session_registers_tools(self):
        from compatibillabuddy.agent.core import AgentSession

        config = _make_config()
        with patch("compatibillabuddy.agent.core.genai"):
            session = AgentSession(config)
        tool_names = set(session.tool_map.keys())
        expected = {
            "tool_probe_hardware",
            "tool_inspect_environment",
            "tool_run_doctor",
            "tool_explain_issue",
            "tool_search_rules",
        }
        assert expected == tool_names


class TestAgentSessionChat:
    """Tests for AgentSession.chat() method."""

    def test_chat_simple_text_response(self):
        """Returns text when Gemini replies with no tool calls."""
        from compatibillabuddy.agent.core import AgentSession

        config = _make_config()
        mock_genai = MagicMock()
        mock_chat = MagicMock()
        mock_chat.send_message.return_value = _mock_text_response("Your environment looks fine!")

        with patch("compatibillabuddy.agent.core.genai", mock_genai):
            session = AgentSession(config)
            session._chat = mock_chat
            result = session.chat("Is my setup OK?")

        assert result == "Your environment looks fine!"

    def test_chat_single_tool_call(self):
        """Dispatches one tool call, returns final text."""
        from compatibillabuddy.agent.core import AgentSession

        config = _make_config()
        mock_genai = MagicMock()
        mock_chat = MagicMock()

        # First response: tool call, second response: text
        mock_chat.send_message.side_effect = [
            _mock_tool_call_response("tool_probe_hardware", {}),
            _mock_text_response("You have an NVIDIA RTX 4090."),
        ]

        with patch("compatibillabuddy.agent.core.genai", mock_genai):
            session = AgentSession(config)
            session._chat = mock_chat
            # Mock the tool function
            session.tool_map["tool_probe_hardware"] = MagicMock(
                return_value={"os_name": "Linux", "gpus": []}
            )
            result = session.chat("What GPU do I have?")

        assert "RTX 4090" in result

    def test_chat_multi_tool_calls(self):
        """Handles multiple sequential tool calls in one turn."""
        from compatibillabuddy.agent.core import AgentSession

        config = _make_config()
        mock_genai = MagicMock()
        mock_chat = MagicMock()

        # Tool call 1 → tool call 2 → final text
        mock_chat.send_message.side_effect = [
            _mock_tool_call_response("tool_probe_hardware", {}),
            _mock_tool_call_response("tool_inspect_environment", {}),
            _mock_text_response("Diagnosis complete."),
        ]

        with patch("compatibillabuddy.agent.core.genai", mock_genai):
            session = AgentSession(config)
            session._chat = mock_chat
            session.tool_map["tool_probe_hardware"] = MagicMock(
                return_value={"os_name": "Linux", "gpus": []}
            )
            session.tool_map["tool_inspect_environment"] = MagicMock(
                return_value={"python_version": "3.11.7", "packages": []}
            )
            result = session.chat("Diagnose my environment")

        assert result == "Diagnosis complete."
        assert mock_chat.send_message.call_count == 3

    def test_chat_unknown_tool_returns_error(self):
        """Gracefully handles unknown tool name from Gemini."""
        from compatibillabuddy.agent.core import AgentSession

        config = _make_config()
        mock_genai = MagicMock()
        mock_chat = MagicMock()

        # Gemini calls a tool that doesn't exist, then gives text
        mock_chat.send_message.side_effect = [
            _mock_tool_call_response("nonexistent_tool", {}),
            _mock_text_response("Sorry, I couldn't do that."),
        ]

        with patch("compatibillabuddy.agent.core.genai", mock_genai):
            session = AgentSession(config)
            session._chat = mock_chat
            result = session.chat("Do something weird")

        # Should not crash — returns the final text
        assert isinstance(result, str)


class TestAgentToolDispatch:
    """Tests for tool dispatch mapping."""

    def test_tool_dispatch_maps_correctly(self):
        """Each tool name dispatches to the right function."""
        from compatibillabuddy.agent import tools
        from compatibillabuddy.agent.core import AgentSession

        config = _make_config()
        with patch("compatibillabuddy.agent.core.genai"):
            session = AgentSession(config)

        assert session.tool_map["tool_probe_hardware"] is tools.tool_probe_hardware
        assert session.tool_map["tool_inspect_environment"] is tools.tool_inspect_environment
        assert session.tool_map["tool_run_doctor"] is tools.tool_run_doctor
        assert session.tool_map["tool_explain_issue"] is tools.tool_explain_issue
        assert session.tool_map["tool_search_rules"] is tools.tool_search_rules


class TestMissingGenai:
    """Test behavior when google-genai is not installed."""

    def test_missing_genai_raises(self):
        """Clear error if google-genai not installed."""

        from compatibillabuddy.agent import core as core_module

        # Temporarily remove genai from the module
        original_genai = getattr(core_module, "genai", None)
        core_module.genai = None

        try:
            config = _make_config()
            with pytest.raises(ImportError, match="google-genai"):
                AgentSession_cls = core_module.AgentSession
                AgentSession_cls(config)
        finally:
            core_module.genai = original_genai
