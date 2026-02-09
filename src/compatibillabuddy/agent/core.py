"""Agent core — Gemini-powered diagnostic session with tool calling.

Manages the Gemini client, tool registration, and the multi-turn
conversation loop with automatic tool dispatch.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

try:
    from google import genai
except ImportError:
    genai = None  # type: ignore[assignment]

from compatibillabuddy.agent import tools as tool_functions
from compatibillabuddy.agent.config import AgentConfig

# Maximum tool-call round-trips per chat()/auto_repair() call
_MAX_TOOL_ROUNDS = 25

# HTTP status codes that should trigger a retry
_RETRYABLE_CODES: frozenset[int] = frozenset({429, 500, 502, 503})

_SYSTEM_PROMPT = """\
You are Compatibillabuddy, a hardware-aware Python dependency diagnostic and repair agent.

You help developers diagnose AND fix compatibility issues in their ML/AI Python
environments — CUDA mismatches, NumPy ABI breaks, driver conflicts, and more.

## Diagnostic Tools
- tool_probe_hardware: Detect OS, CPU, GPU, CUDA version
- tool_inspect_environment: List installed Python packages
- tool_run_doctor: Run a full compatibility diagnosis
- tool_explain_issue: Get detailed explanation of a specific issue
- tool_search_rules: Search the knowledge base for rules about a package

## Repair Tools
- tool_snapshot_environment: Capture pip freeze as a rollback point (ALWAYS do this first)
- tool_run_pip: Execute pip install/uninstall with safety guardrails
- tool_verify_fix: Re-run doctor and compare before/after to verify improvement
- tool_rollback: Restore packages to a previous snapshot if fixes made things worse

## Rules
- When a user asks about their environment, USE the tools to get real data. Do not guess.
- Be concise and actionable.
- When repairing: ALWAYS snapshot first, then diagnose, then fix one issue at a time,
  then verify. If verification shows new problems, rollback and try an alternative.
- Explain what you are doing at each step.
"""

_AUTO_REPAIR_PROMPT_TEMPLATE = """\
You are now in autonomous repair mode. Your goal is to diagnose and {action} all \
compatibility issues in this Python environment.

## Protocol — follow these steps IN ORDER:
1. Call tool_snapshot_environment() to save a rollback point.
2. Call tool_run_doctor() to get the full diagnosis.
3. If there are no issues, report success and stop.
4. Analyze the issues and plan a fix order (fix the most critical/blocking issues first).
5. For each issue:
   a. Call tool_run_pip(action=..., package=..., dry_run={dry_run}) to apply the fix.
   b. Call tool_verify_fix(previous_diagnosis_json=...) to check improvement.
   c. If improved, continue to the next issue.
   d. If new issues appeared, call tool_rollback(snapshot_json=...) and try an alternative.
6. After all fixes, provide a final summary with:
   - What was found
   - What was {action_past} (or would be fixed in dry-run mode)
   - What still needs attention
   - The full changelog of actions taken

## Settings
- dry_run={dry_run} — {dry_run_instruction}
- max_retries={max_retries} — give up on a single issue after this many failed attempts

## Important
- ALWAYS snapshot before making any changes.
- Fix ONE issue at a time, then verify before moving on.
- If you cannot fix an issue after {max_retries} attempts, skip it and move to the next.
- Never uninstall pip, setuptools, wheel, or compatibillabuddy.

Begin now.
"""


@dataclass
class RepairResult:
    """Structured result from an autonomous repair session.

    Attributes:
        summary: The agent's final text summary of the repair session.
        repair_log: List of dicts recording each tool call and its result.
        success: Whether the repair session completed without errors.
    """

    summary: str
    repair_log: list[dict[str, Any]] = field(default_factory=list)
    success: bool = True


class AgentSession:
    """A multi-turn conversation session with the Gemini-powered agent.

    Handles tool registration, dispatch, and the tool-call loop.

    Args:
        config: AgentConfig with API key and model selection.
        on_event: Optional callback for real-time event notifications.
            Called with a dict containing event type and details.

    Raises:
        ImportError: If google-genai is not installed.
    """

    def __init__(
        self,
        config: AgentConfig,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if genai is None:
            raise ImportError(
                "google-genai is required for the agent. "
                "Install it with: pip install compatibillabuddy[agent]"
            )

        self.config = config
        self._on_event = on_event

        # Build tool dispatch map — 5 diagnostic + 4 repair = 9 tools
        self.tool_map: dict[str, Callable[..., Any]] = {
            "tool_probe_hardware": tool_functions.tool_probe_hardware,
            "tool_inspect_environment": tool_functions.tool_inspect_environment,
            "tool_run_doctor": tool_functions.tool_run_doctor,
            "tool_explain_issue": tool_functions.tool_explain_issue,
            "tool_search_rules": tool_functions.tool_search_rules,
            "tool_snapshot_environment": tool_functions.tool_snapshot_environment,
            "tool_run_pip": tool_functions.tool_run_pip,
            "tool_verify_fix": tool_functions.tool_verify_fix,
            "tool_rollback": tool_functions.tool_rollback,
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

    def _emit_event(self, event: dict[str, Any]) -> None:
        """Fire an event to the callback if one is registered."""
        if self._on_event is not None:
            self._on_event(event)

    def _send_with_retry(self, message: str) -> Any:
        """Send a message with exponential backoff on transient errors.

        Retries on 429 (rate limit), 500, 502, 503 errors.
        Non-transient errors (e.g. 400) raise immediately.

        Args:
            message: The message text to send.

        Returns:
            The Gemini API response.

        Raises:
            Exception: After exhausting retries, or on non-transient errors.
        """
        last_error: Exception | None = None
        max_attempts = 1 + self.config.max_api_retries

        for attempt in range(max_attempts):
            try:
                return self._chat.send_message(message)
            except Exception as e:
                error_code = getattr(e, "code", None)
                if error_code not in _RETRYABLE_CODES:
                    raise

                last_error = e

                if attempt < self.config.max_api_retries:
                    delay = self.config.base_retry_delay * (2**attempt)
                    self._emit_event(
                        {
                            "type": "retry",
                            "attempt": attempt + 1,
                            "max_retries": self.config.max_api_retries,
                            "delay": delay,
                            "error": str(e),
                        }
                    )
                    time.sleep(delay)

        raise last_error  # type: ignore[misc]

    def chat(self, user_message: str) -> str:
        """Send a message and handle the tool-call loop.

        Args:
            user_message: The user's input text.

        Returns:
            The agent's final text response after all tool calls are resolved.
        """
        response = self._send_with_retry(user_message)

        for _ in range(_MAX_TOOL_ROUNDS):
            part = response.candidates[0].content.parts[0]

            if part.function_call is None:
                return part.text or ""

            # Dispatch the tool call
            fc = part.function_call
            args = dict(fc.args) if fc.args else {}
            tool_result = self._dispatch_tool(fc.name, args)

            self._emit_event(
                {
                    "type": "tool_call",
                    "tool": fc.name,
                    "args": args,
                    "result": tool_result,
                }
            )

            response = self._send_with_retry(json.dumps(tool_result, default=str))

        part = response.candidates[0].content.parts[0]
        return part.text or "(Agent reached maximum tool call rounds)"

    def auto_repair(
        self,
        dry_run: bool = True,
        max_retries: int = 3,
    ) -> RepairResult:
        """Run the autonomous repair loop.

        The agent plans, executes fixes, verifies results, and self-corrects.
        All tool calls are logged in the returned RepairResult.

        Args:
            dry_run: If True (default), the agent shows what it would do
                without executing pip commands.
            max_retries: Maximum fix attempts per issue before skipping.

        Returns:
            RepairResult with summary, repair log, and success status.
        """
        repair_log: list[dict[str, Any]] = []

        action = "fix" if not dry_run else "plan fixes for"
        action_past = "fixed" if not dry_run else "planned"
        dry_run_instruction = (
            "SHOW what you would do but do NOT execute pip commands"
            if dry_run
            else "EXECUTE pip commands for real"
        )

        prompt = _AUTO_REPAIR_PROMPT_TEMPLATE.format(
            dry_run=dry_run,
            max_retries=max_retries,
            action=action,
            action_past=action_past,
            dry_run_instruction=dry_run_instruction,
        )

        response = self._send_with_retry(prompt)

        for _ in range(_MAX_TOOL_ROUNDS):
            part = response.candidates[0].content.parts[0]

            if part.function_call is None:
                summary = part.text or ""
                return RepairResult(
                    summary=summary,
                    repair_log=repair_log,
                    success=True,
                )

            # Dispatch the tool call
            fc = part.function_call
            args = dict(fc.args) if fc.args else {}
            tool_result = self._dispatch_tool(fc.name, args)

            log_entry = {
                "tool": fc.name,
                "args": args,
                "result": tool_result,
            }
            repair_log.append(log_entry)

            self._emit_event(
                {
                    "type": "tool_call",
                    **log_entry,
                }
            )

            response = self._send_with_retry(json.dumps(tool_result, default=str))

        # Exhausted max rounds
        part = response.candidates[0].content.parts[0]
        summary = part.text or "(Agent reached maximum tool call rounds)"
        return RepairResult(
            summary=summary,
            repair_log=repair_log,
            success=False,
        )

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
