"""Live integration tests for autonomous repair against real Gemini API.

These tests require a valid GEMINI_API_KEY environment variable.
They are auto-skipped in CI and should be run manually:

    $env:GEMINI_API_KEY = "your-key"
    pytest tests/integration/test_repair_live.py -m integration -v
"""

from __future__ import annotations

import json
import os

import pytest

from compatibillabuddy.agent.config import AgentConfig

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set",
    ),
]


@pytest.fixture()
def agent_session():
    """Create a live AgentSession with the real API key."""
    from compatibillabuddy.agent.core import AgentSession

    config = AgentConfig(api_key=os.environ["GEMINI_API_KEY"])
    return AgentSession(config)


@pytest.fixture()
def agent_session_with_events():
    """Create a live AgentSession that captures events."""
    from compatibillabuddy.agent.core import AgentSession

    events: list[dict] = []
    config = AgentConfig(api_key=os.environ["GEMINI_API_KEY"])
    session = AgentSession(config, on_event=lambda e: events.append(e))
    return session, events


class TestLiveAutoRepair:
    """Live integration tests for the autonomous repair loop."""

    def test_auto_repair_dry_run(self, agent_session):
        """auto_repair(dry_run=True) completes and returns a RepairResult."""
        from compatibillabuddy.agent.core import RepairResult

        result = agent_session.auto_repair(dry_run=True)

        assert isinstance(result, RepairResult)
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0
        assert isinstance(result.repair_log, list)

    def test_auto_repair_uses_tools(self, agent_session):
        """Agent calls at least snapshot and doctor tools during repair."""
        result = agent_session.auto_repair(dry_run=True)

        tool_names = [entry["tool"] for entry in result.repair_log]
        # Agent should at minimum snapshot the environment and run doctor
        assert "tool_snapshot_environment" in tool_names
        assert "tool_run_doctor" in tool_names

    def test_auto_repair_events_fire(self, agent_session_with_events):
        """on_event callback receives tool_call events during auto_repair."""
        session, events = agent_session_with_events

        session.auto_repair(dry_run=True)

        tool_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_events) >= 2  # at least snapshot + doctor
        # Every event should have tool name and result
        for event in tool_events:
            assert "tool" in event
            assert "result" in event

    def test_chat_repair_question(self, agent_session):
        """Agent uses tools to answer repair-related questions."""
        result = agent_session.chat(
            "Can you snapshot my environment and then run a diagnosis? "
            "Use tool_snapshot_environment first, then tool_run_doctor."
        )

        assert isinstance(result, str)
        assert len(result) > 0


class TestLiveRepairCli:
    """Live integration test for the repair CLI command."""

    def test_repair_cli_dry_run_json(self):
        """compatibuddy repair --format json produces valid JSON."""
        from typer.testing import CliRunner

        from compatibillabuddy.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["repair", "--format", "json"])

        # Should complete (exit 0 or 1 depending on environment health)
        assert result.exit_code in (0, 1)

        # Output should be valid JSON
        parsed = json.loads(result.output)
        assert "summary" in parsed
        assert "repair_log" in parsed
        assert "success" in parsed
        assert isinstance(parsed["repair_log"], list)
