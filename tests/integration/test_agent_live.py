"""Live integration tests for AgentSession against real Gemini API.

These tests require a valid GEMINI_API_KEY environment variable.
They are auto-skipped in CI and should be run manually:

    $env:GEMINI_API_KEY = "your-key"
    pytest tests/integration/ -m integration -v
"""

from __future__ import annotations

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


class TestLiveAgent:
    """Integration tests against real Gemini API."""

    def test_live_simple_greeting(self, agent_session):
        """Agent responds to a simple greeting with text."""
        result = agent_session.chat("Hello! What can you help me with?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_live_doctor_tool_call(self, agent_session):
        """Agent uses tool_run_doctor when asked to diagnose."""
        result = agent_session.chat(
            "Please run a full compatibility diagnosis on my environment. "
            "Use the tool_run_doctor tool."
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_live_hardware_tool_call(self, agent_session):
        """Agent uses tool_probe_hardware when asked about hardware."""
        result = agent_session.chat(
            "What hardware am I running on? Use the tool_probe_hardware tool to check."
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_live_multi_turn(self, agent_session):
        """Two messages in sequence, context maintained."""
        first = agent_session.chat("My name is TestUser. Remember that for this conversation.")
        assert isinstance(first, str)

        second = agent_session.chat("What name did I just tell you?")
        assert isinstance(second, str)
        assert "testuser" in second.lower() or "TestUser" in second
