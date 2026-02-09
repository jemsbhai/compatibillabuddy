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
        diagnostic_tools = {
            "tool_probe_hardware",
            "tool_inspect_environment",
            "tool_run_doctor",
            "tool_explain_issue",
            "tool_search_rules",
        }
        assert diagnostic_tools.issubset(tool_names)


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


class TestRepairToolsRegistered:
    """Tests that repair tools are registered in the agent session."""

    def test_repair_tools_in_tool_map(self):
        from compatibillabuddy.agent.core import AgentSession

        config = _make_config()
        with patch("compatibillabuddy.agent.core.genai"):
            session = AgentSession(config)

        repair_tools = {
            "tool_snapshot_environment",
            "tool_run_pip",
            "tool_verify_fix",
            "tool_rollback",
        }
        assert repair_tools.issubset(set(session.tool_map.keys()))

    def test_all_nine_tools_registered(self):
        """5 diagnostic + 4 repair = 9 total tools."""
        from compatibillabuddy.agent.core import AgentSession

        config = _make_config()
        with patch("compatibillabuddy.agent.core.genai"):
            session = AgentSession(config)

        assert len(session.tool_map) == 9

    def test_dispatch_repair_tools(self):
        """_dispatch_tool works for all 4 repair tools."""
        from compatibillabuddy.agent import tools
        from compatibillabuddy.agent.core import AgentSession

        config = _make_config()
        with patch("compatibillabuddy.agent.core.genai"):
            session = AgentSession(config)

        assert session.tool_map["tool_snapshot_environment"] is tools.tool_snapshot_environment
        assert session.tool_map["tool_run_pip"] is tools.tool_run_pip
        assert session.tool_map["tool_verify_fix"] is tools.tool_verify_fix
        assert session.tool_map["tool_rollback"] is tools.tool_rollback


class TestAutoRepair:
    """Tests for AgentSession.auto_repair() method."""

    def test_auto_repair_returns_repair_result(self):
        """auto_repair() returns a RepairResult dataclass."""
        from compatibillabuddy.agent.core import AgentSession, RepairResult

        config = _make_config()
        mock_genai = MagicMock()
        mock_chat = MagicMock()
        mock_chat.send_message.return_value = _mock_text_response(
            "No issues found. Environment is healthy."
        )

        with patch("compatibillabuddy.agent.core.genai", mock_genai):
            session = AgentSession(config)
            session._chat = mock_chat
            result = session.auto_repair(dry_run=True)

        assert isinstance(result, RepairResult)
        assert isinstance(result.summary, str)
        assert isinstance(result.repair_log, list)

    def test_auto_repair_dry_run_in_prompt(self):
        """dry_run=True is communicated in the prompt sent to Gemini."""
        from compatibillabuddy.agent.core import AgentSession

        config = _make_config()
        mock_genai = MagicMock()
        mock_chat = MagicMock()
        mock_chat.send_message.return_value = _mock_text_response("Dry run complete.")

        with patch("compatibillabuddy.agent.core.genai", mock_genai):
            session = AgentSession(config)
            session._chat = mock_chat
            session.auto_repair(dry_run=True)

        sent_prompt = mock_chat.send_message.call_args[0][0]
        assert "dry_run" in sent_prompt.lower() or "dry run" in sent_prompt.lower()

    def test_auto_repair_live_mode_in_prompt(self):
        """dry_run=False instructs Gemini to execute for real."""
        from compatibillabuddy.agent.core import AgentSession

        config = _make_config()
        mock_genai = MagicMock()
        mock_chat = MagicMock()
        mock_chat.send_message.return_value = _mock_text_response("Repairs complete.")

        with patch("compatibillabuddy.agent.core.genai", mock_genai):
            session = AgentSession(config)
            session._chat = mock_chat
            session.auto_repair(dry_run=False)

        sent_prompt = mock_chat.send_message.call_args[0][0]
        assert "dry_run=false" in sent_prompt.lower() or "execute" in sent_prompt.lower()

    def test_auto_repair_with_tool_calls(self):
        """auto_repair handles the tool-call loop and logs actions."""
        from compatibillabuddy.agent.core import AgentSession

        config = _make_config()
        mock_genai = MagicMock()
        mock_chat = MagicMock()

        # Simulate: snapshot → doctor → text reply
        mock_chat.send_message.side_effect = [
            _mock_tool_call_response("tool_snapshot_environment", {}),
            _mock_tool_call_response("tool_run_doctor", {}),
            _mock_text_response("Found 1 issue. Environment needs torch upgrade."),
        ]

        with patch("compatibillabuddy.agent.core.genai", mock_genai):
            session = AgentSession(config)
            session._chat = mock_chat
            session.tool_map["tool_snapshot_environment"] = MagicMock(
                return_value={"timestamp": "2025-01-15T00:00:00", "packages": ["torch==2.1.0"]}
            )
            session.tool_map["tool_run_doctor"] = MagicMock(
                return_value={"issues": [{"severity": 1, "description": "CUDA mismatch"}]}
            )
            result = session.auto_repair(dry_run=True)

        assert len(result.repair_log) == 2  # snapshot + doctor
        assert result.repair_log[0]["tool"] == "tool_snapshot_environment"
        assert result.repair_log[1]["tool"] == "tool_run_doctor"

    def test_auto_repair_max_retries_respected(self):
        """auto_repair accepts a max_retries parameter."""
        from compatibillabuddy.agent.core import AgentSession

        config = _make_config()
        mock_genai = MagicMock()
        mock_chat = MagicMock()
        mock_chat.send_message.return_value = _mock_text_response("Done.")

        with patch("compatibillabuddy.agent.core.genai", mock_genai):
            session = AgentSession(config)
            session._chat = mock_chat
            result = session.auto_repair(dry_run=True, max_retries=5)

        # Should complete without error — max_retries is accepted
        assert isinstance(result.summary, str)


class TestEventCallback:
    """Tests for the on_event callback system."""

    def test_on_event_receives_tool_calls(self):
        """Callback is called for each tool dispatch during chat."""
        from compatibillabuddy.agent.core import AgentSession

        config = _make_config()
        mock_genai = MagicMock()
        mock_chat = MagicMock()

        mock_chat.send_message.side_effect = [
            _mock_tool_call_response("tool_probe_hardware", {}),
            _mock_text_response("Done."),
        ]

        events = []

        with patch("compatibillabuddy.agent.core.genai", mock_genai):
            session = AgentSession(config, on_event=lambda e: events.append(e))
            session._chat = mock_chat
            session.tool_map["tool_probe_hardware"] = MagicMock(
                return_value={"os_name": "Linux", "gpus": []}
            )
            session.chat("Check my hardware")

        assert len(events) >= 1
        assert events[0]["type"] == "tool_call"
        assert events[0]["tool"] == "tool_probe_hardware"

    def test_on_event_optional(self):
        """Session works fine without a callback."""
        from compatibillabuddy.agent.core import AgentSession

        config = _make_config()
        mock_genai = MagicMock()
        mock_chat = MagicMock()
        mock_chat.send_message.return_value = _mock_text_response("Hello!")

        with patch("compatibillabuddy.agent.core.genai", mock_genai):
            session = AgentSession(config)
            session._chat = mock_chat
            result = session.chat("Hi")

        assert result == "Hello!"

    def test_auto_repair_fires_events(self):
        """auto_repair() fires events through the callback."""
        from compatibillabuddy.agent.core import AgentSession

        config = _make_config()
        mock_genai = MagicMock()
        mock_chat = MagicMock()

        mock_chat.send_message.side_effect = [
            _mock_tool_call_response("tool_snapshot_environment", {}),
            _mock_text_response("Environment healthy."),
        ]

        events = []

        with patch("compatibillabuddy.agent.core.genai", mock_genai):
            session = AgentSession(config, on_event=lambda e: events.append(e))
            session._chat = mock_chat
            session.tool_map["tool_snapshot_environment"] = MagicMock(
                return_value={"timestamp": "2025-01-15T00:00:00", "packages": []}
            )
            session.auto_repair(dry_run=True)

        tool_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_events) >= 1


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
