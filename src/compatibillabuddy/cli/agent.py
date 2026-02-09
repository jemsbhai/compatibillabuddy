"""CLI agent command — interactive chat REPL with the Gemini agent.

Provides a multi-turn conversation interface for diagnosing and
repairing ML environment issues interactively.
"""

from typing import Optional

import typer

from compatibillabuddy.agent.config import DEFAULT_MODEL, AgentConfig, resolve_api_key
from compatibillabuddy.agent.core import AgentSession


def agent_command(
    model: str = typer.Option(  # noqa: B008
        DEFAULT_MODEL,
        "--model",
        "-m",
        help="Gemini model to use.",
    ),
    api_key: Optional[str] = typer.Option(  # noqa: B008, UP045
        None,
        "--api-key",
        envvar="GEMINI_API_KEY",
        help="Gemini API key (or set GEMINI_API_KEY env var).",
    ),
) -> None:
    """Interactive chat with the Compatibillabuddy AI agent.

    Start a multi-turn conversation to diagnose and fix your ML environment.
    Type 'exit' or 'quit' to end the session.

    Requires a Gemini API key (GEMINI_API_KEY env var or --api-key).
    """
    # Resolve API key
    try:
        key = resolve_api_key(api_key)
    except ValueError:
        typer.echo(
            "Error: No Gemini API key found. "
            "Set GEMINI_API_KEY environment variable or pass --api-key.",
        )
        raise typer.Exit(code=1) from None

    typer.echo("\n🤖 Compatibillabuddy Agent")
    typer.echo("   Type 'exit' or 'quit' to end the session.\n")

    config = AgentConfig(api_key=key, model=model)
    session = AgentSession(config)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            typer.echo("Goodbye!")
            break

        try:
            response = session.chat(user_input)
            typer.echo(f"\nAgent: {response}\n")
        except Exception as e:
            typer.echo(f"\nError: {e}\n", err=True)
