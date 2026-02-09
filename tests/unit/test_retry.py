"""Tests for retry/backoff logic in agent core.

TDD: tests written BEFORE implementation.
All tests mock Gemini client — no real API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from compatibillabuddy.agent.config import AgentConfig


def _make_config(**overrides) -> AgentConfig:
    defaults = {"api_key": "fake-key"}
    defaults.update(overrides)
    return AgentConfig(**defaults)


def _mock_text_response(text: str) -> MagicMock:
    part = MagicMock()
    part.text = text
    part.function_call = None
    candidate = MagicMock()
    candidate.content.parts = [part]
    response = MagicMock()
    response.candidates = [candidate]
    return response


def _make_api_error(code: int, message: str = "error"):
    """Build a fake exception that mimics google.genai.errors.ClientError."""

    class FakeClientError(Exception):
        def __init__(self, code, message):
            self.code = code
            self.message = message
            super().__init__(f"{code} {message}")

    return FakeClientError(code, message)


# ===========================================================================
# AgentConfig retry defaults
# ===========================================================================


class TestConfigRetryDefaults:
    """AgentConfig should have retry configuration fields."""

    def test_config_has_retry_defaults(self):
        config = _make_config()
        assert config.max_api_retries >= 1
        assert config.base_retry_delay > 0

    def test_config_custom_retry_values(self):
        config = _make_config(max_api_retries=10, base_retry_delay=0.5)
        assert config.max_api_retries == 10
        assert config.base_retry_delay == 0.5


# ===========================================================================
# _send_with_retry
# ===========================================================================


class TestSendWithRetry:
    """Tests for the retry wrapper around chat.send_message."""

    def test_send_with_retry_success_first_try(self):
        """No retry when call succeeds on first attempt."""
        from compatibillabuddy.agent.core import AgentSession

        config = _make_config()
        mock_genai = MagicMock()
        mock_chat = MagicMock()
        expected = _mock_text_response("Hello!")
        mock_chat.send_message.return_value = expected

        with patch("compatibillabuddy.agent.core.genai", mock_genai):
            session = AgentSession(config)
            session._chat = mock_chat
            result = session._send_with_retry("Hi")

        assert result is expected
        assert mock_chat.send_message.call_count == 1

    def test_send_with_retry_retries_on_429(self):
        """Retries on rate limit error, succeeds on 2nd try."""
        from compatibillabuddy.agent.core import AgentSession

        config = _make_config(base_retry_delay=0.01)
        mock_genai = MagicMock()
        mock_chat = MagicMock()

        error_429 = _make_api_error(429, "RESOURCE_EXHAUSTED")
        expected = _mock_text_response("Success after retry!")

        mock_chat.send_message.side_effect = [error_429, expected]

        with (
            patch("compatibillabuddy.agent.core.genai", mock_genai),
            patch("compatibillabuddy.agent.core.time.sleep"),
        ):
            session = AgentSession(config)
            session._chat = mock_chat
            result = session._send_with_retry("Hi")

        assert result is expected
        assert mock_chat.send_message.call_count == 2

    def test_send_with_retry_exponential_backoff(self):
        """Sleep times double each retry."""
        from compatibillabuddy.agent.core import AgentSession

        config = _make_config(max_api_retries=4, base_retry_delay=1.0)
        mock_genai = MagicMock()
        mock_chat = MagicMock()

        error_429 = _make_api_error(429, "RESOURCE_EXHAUSTED")
        expected = _mock_text_response("Finally!")

        # Fail 3 times, succeed on 4th
        mock_chat.send_message.side_effect = [error_429, error_429, error_429, expected]

        with (
            patch("compatibillabuddy.agent.core.genai", mock_genai),
            patch("compatibillabuddy.agent.core.time.sleep") as mock_sleep,
        ):
            session = AgentSession(config)
            session._chat = mock_chat
            result = session._send_with_retry("Hi")

        assert result is expected
        # Backoff: 1.0, 2.0, 4.0
        sleep_calls = [c[0][0] for c in mock_sleep.call_args_list]
        assert sleep_calls[0] == pytest.approx(1.0)
        assert sleep_calls[1] == pytest.approx(2.0)
        assert sleep_calls[2] == pytest.approx(4.0)

    def test_send_with_retry_gives_up_after_max(self):
        """Raises after exhausting all retries."""
        from compatibillabuddy.agent.core import AgentSession

        config = _make_config(max_api_retries=2, base_retry_delay=0.01)
        mock_genai = MagicMock()
        mock_chat = MagicMock()

        error_429 = _make_api_error(429, "RESOURCE_EXHAUSTED")
        mock_chat.send_message.side_effect = [error_429, error_429, error_429]

        with (
            patch("compatibillabuddy.agent.core.genai", mock_genai),
            patch("compatibillabuddy.agent.core.time.sleep"),
        ):
            session = AgentSession(config)
            session._chat = mock_chat

            with pytest.raises(Exception, match="429"):
                session._send_with_retry("Hi")

        # 1 initial + 2 retries = 3 total calls
        assert mock_chat.send_message.call_count == 3

    def test_send_with_retry_no_retry_on_400(self):
        """Non-transient errors (400) raise immediately without retry."""
        from compatibillabuddy.agent.core import AgentSession

        config = _make_config(base_retry_delay=0.01)
        mock_genai = MagicMock()
        mock_chat = MagicMock()

        error_400 = _make_api_error(400, "BAD_REQUEST")
        mock_chat.send_message.side_effect = error_400

        with patch("compatibillabuddy.agent.core.genai", mock_genai):
            session = AgentSession(config)
            session._chat = mock_chat

            with pytest.raises(Exception, match="400"):
                session._send_with_retry("Hi")

        assert mock_chat.send_message.call_count == 1

    def test_send_with_retry_retries_on_500(self):
        """Server errors (500, 503) are also retried."""
        from compatibillabuddy.agent.core import AgentSession

        config = _make_config(base_retry_delay=0.01)
        mock_genai = MagicMock()
        mock_chat = MagicMock()

        error_503 = _make_api_error(503, "SERVICE_UNAVAILABLE")
        expected = _mock_text_response("Recovered!")

        mock_chat.send_message.side_effect = [error_503, expected]

        with (
            patch("compatibillabuddy.agent.core.genai", mock_genai),
            patch("compatibillabuddy.agent.core.time.sleep"),
        ):
            session = AgentSession(config)
            session._chat = mock_chat
            result = session._send_with_retry("Hi")

        assert result is expected
        assert mock_chat.send_message.call_count == 2

    def test_send_with_retry_emits_events(self):
        """Retry events fire through the on_event callback."""
        from compatibillabuddy.agent.core import AgentSession

        config = _make_config(base_retry_delay=0.01)
        mock_genai = MagicMock()
        mock_chat = MagicMock()

        error_429 = _make_api_error(429, "RESOURCE_EXHAUSTED")
        expected = _mock_text_response("OK!")

        mock_chat.send_message.side_effect = [error_429, expected]

        events = []

        with (
            patch("compatibillabuddy.agent.core.genai", mock_genai),
            patch("compatibillabuddy.agent.core.time.sleep"),
        ):
            session = AgentSession(config, on_event=lambda e: events.append(e))
            session._chat = mock_chat
            session._send_with_retry("Hi")

        retry_events = [e for e in events if e["type"] == "retry"]
        assert len(retry_events) == 1
        assert retry_events[0]["attempt"] == 1
        assert "429" in str(retry_events[0]["error"])


# ===========================================================================
# chat() and auto_repair() use retry
# ===========================================================================


class TestChatAndRepairUseRetry:
    """Verify chat() and auto_repair() go through _send_with_retry."""

    def test_chat_uses_retry(self):
        """chat() calls _send_with_retry instead of raw send_message."""
        from compatibillabuddy.agent.core import AgentSession

        config = _make_config()
        mock_genai = MagicMock()

        with patch("compatibillabuddy.agent.core.genai", mock_genai):
            session = AgentSession(config)
            session._send_with_retry = MagicMock(return_value=_mock_text_response("Hello!"))
            result = session.chat("Hi")

        assert result == "Hello!"
        session._send_with_retry.assert_called()

    def test_auto_repair_uses_retry(self):
        """auto_repair() calls _send_with_retry instead of raw send_message."""
        from compatibillabuddy.agent.core import AgentSession

        config = _make_config()
        mock_genai = MagicMock()

        with patch("compatibillabuddy.agent.core.genai", mock_genai):
            session = AgentSession(config)
            session._send_with_retry = MagicMock(return_value=_mock_text_response("Repair done."))
            result = session.auto_repair(dry_run=True)

        assert result.summary == "Repair done."
        session._send_with_retry.assert_called()
