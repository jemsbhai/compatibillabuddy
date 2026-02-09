"""Agent configuration — API key resolution and model selection.

Handles GEMINI_API_KEY lookup (env var or explicit) and validates
model selection against the supported models list.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, field_validator

SUPPORTED_MODELS: list[str] = [
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

DEFAULT_MODEL: str = "gemini-3-flash-preview"


def resolve_api_key(explicit_key: str | None = None) -> str:
    """Resolve the Gemini API key.

    Priority: explicit_key > GEMINI_API_KEY env var.

    Args:
        explicit_key: Directly provided API key, takes precedence.

    Returns:
        The resolved API key string.

    Raises:
        ValueError: If no key is found from any source.
    """
    if explicit_key:
        return explicit_key

    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key

    raise ValueError(
        "No API key found. Set the GEMINI_API_KEY environment variable or pass a key explicitly."
    )


class AgentConfig(BaseModel):
    """Configuration for the Gemini-powered agent.

    Attributes:
        api_key: Gemini API key.
        model: Model identifier string. Must be in SUPPORTED_MODELS.
        max_api_retries: Max retry attempts on transient API errors.
        base_retry_delay: Initial delay in seconds for exponential backoff.
    """

    api_key: str
    model: str = DEFAULT_MODEL
    max_api_retries: int = 5
    base_retry_delay: float = 1.0

    @field_validator("model")
    @classmethod
    def model_must_be_supported(cls, v: str) -> str:
        if v not in SUPPORTED_MODELS:
            raise ValueError(
                f"{v!r} is not a supported model. Choose from: {', '.join(SUPPORTED_MODELS)}"
            )
        return v
