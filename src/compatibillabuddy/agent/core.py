"""Agent core — Gemini-powered diagnostic session with tool calling.

Manages the Gemini client, tool registration, and the multi-turn
conversation loop with automatic tool dispatch.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

try:
    from google import genai
except ImportError:
    genai = None  # type: ignore[assignment]

from compatibillabuddy.agent import tools as tool_functions
from compatibillabuddy.agent.config import AgentConfig

# Maximum tool-call round-trips per chat() call to prevent infinite loops
_MAX_TOOL_ROUNDS = 10

_SYSTEM_PROMPT = """\
You are Compatibillabuddy, a hardware-aware Python dependency diagnostic assistant.

You help developers diagnose and fix compatibility issues in their ML/AI Python
environments — CUDA mismatches, NumPy ABI breaks, driver conflicts, and more.

You have access to these tools:
- tool_probe_hardware: Detect OS, CPU, GPU, CUDA version
- tool_inspect_environment: List installed Python packages
- tool_run_doctor: Run a full compatibility diagnosis
- tool_explain_issue: Get detailed explanation of a specific issue
- tool_search_rules: Search the knowledge base for rules about a package

When a user asks about their environment, USE the tools to get real data.
Do not guess — run the diagnostics. Be concise and actionable.
"""


class AgentSession:
    """A multi-turn conversation session with the Gemini-powered agent.

    Handles tool registration, dispatch, and the tool-call loop.

    Args:
        config: AgentConfig with API key and model selection.

    Raises:
        ImportError: If google-genai is not installed.
    """

    def __init__(self, config: AgentConfig) -> None:
        if genai is None:
            raise ImportError(
                "google-genai is required for the agent. "
                "Install it with: pip install compatibillabuddy[agent]"
            )

        self.config = config

        # Build tool dispatch map
        self.tool_map: dict[str, Callable[..., Any]] = {
            "tool_probe_hardware": tool_functions.tool_probe_hardware,
            "tool_inspect_environment": tool_functions.tool_inspect_environment,
            "tool_run_doctor": tool_functions.tool_run_doctor,
            "tool_explain_issue": tool_functions.tool_explain_issue,
            "tool_search_rules": tool_functions.tool_search_rules,
        }

        # Initialize Gemini client
        self._client = genai.Client(api_key=config.api_key)

        # Build tool declarations for Gemini
        self._tools = self._build_tool_declarations()

        # Start a chat session
        self._chat = self._client.chats.create(
            model=config.model,
            config=genai.types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                tools=self._tools,
            ),
        )

    def _build_tool_declarations(self) -> list:
        """Build Gemini tool declarations from our tool functions."""
        declarations = []
        for _name, func in self.tool_map.items():
            declarations.append(func)
        return declarations

    def chat(self, user_message: str) -> str:
        """Send a message and handle the tool-call loop.

        Args:
            user_message: The user's input text.

        Returns:
            The agent's final text response after all tool calls are resolved.
        """
        response = self._chat.send_message(user_message)

        for _ in range(_MAX_TOOL_ROUNDS):
            # Check if the response contains a tool call
            part = response.candidates[0].content.parts[0]

            if part.function_call is None:
                # Pure text response — we're done
                return part.text or ""

            # Dispatch the tool call
            fc = part.function_call
            tool_result = self._dispatch_tool(fc.name, dict(fc.args) if fc.args else {})

            # Send tool result back to Gemini
            response = self._chat.send_message(json.dumps(tool_result, default=str))

        # Exhausted max rounds — return whatever we have
        part = response.candidates[0].content.parts[0]
        return part.text or "(Agent reached maximum tool call rounds)"

    def _dispatch_tool(self, tool_name: str, args: dict) -> dict:
        """Dispatch a tool call by name.

        Args:
            tool_name: Name of the tool to call.
            args: Arguments to pass to the tool.

        Returns:
            The tool's return dict, or an error dict if unknown.
        """
        func = self.tool_map.get(tool_name)
        if func is None:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            return func(**args)
        except Exception as e:
            return {"error": f"Tool {tool_name} failed: {e}"}
