"""Tests for agent configuration — API key handling and model selection.

TDD: tests written BEFORE implementation in agent/config.py.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from compatibillabuddy.agent.config import (
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    AgentConfig,
    resolve_api_key,
)

# ===========================================================================
# API key resolution
# ===========================================================================


class TestResolveApiKey:
    """Tests for resolve_api_key()."""

    def test_config_from_env_var(self):
        """Picks up GEMINI_API_KEY from environment."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-123"}):
            key = resolve_api_key()
        assert key == "test-key-123"

    def test_explicit_key_overrides_env(self):
        """Explicit key takes precedence over env var."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key"}):
            key = resolve_api_key(explicit_key="explicit-key")
        assert key == "explicit-key"

    def test_missing_key_raises(self):
        """Raises ValueError if no key provided and no env var."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure GEMINI_API_KEY is not set
            env = os.environ.copy()
            env.pop("GEMINI_API_KEY", None)
            with (
                patch.dict(os.environ, env, clear=True),
                pytest.raises(ValueError, match="GEMINI_API_KEY"),
            ):
                resolve_api_key()


# ===========================================================================
# AgentConfig
# ===========================================================================


class TestAgentConfig:
    """Tests for AgentConfig model."""

    def test_config_default_model(self):
        """Default model is gemini-3-flash-preview."""
        config = AgentConfig(api_key="test-key")
        assert config.model == "gemini-3-flash-preview"

    def test_config_custom_model(self):
        """Can override model selection."""
        config = AgentConfig(api_key="test-key", model="gemini-3-pro-preview")
        assert config.model == "gemini-3-pro-preview"

    def test_invalid_model_rejected(self):
        """Rejects model strings not in supported list."""
        with pytest.raises(ValueError, match="not a supported model"):
            AgentConfig(api_key="test-key", model="gpt-4o")


# ===========================================================================
# Supported models
# ===========================================================================


class TestSupportedModels:
    """Tests for SUPPORTED_MODELS constant."""

    def test_supported_models_contains_all_four(self):
        """All 4 expected models are in SUPPORTED_MODELS."""
        expected = {
            "gemini-3-flash-preview",
            "gemini-3-pro-preview",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        }
        assert expected == set(SUPPORTED_MODELS)

    def test_default_model_is_in_supported(self):
        """DEFAULT_MODEL is in SUPPORTED_MODELS."""
        assert DEFAULT_MODEL in SUPPORTED_MODELS

    def test_default_model_value(self):
        """DEFAULT_MODEL is gemini-3-flash-preview."""
        assert DEFAULT_MODEL == "gemini-3-flash-preview"
