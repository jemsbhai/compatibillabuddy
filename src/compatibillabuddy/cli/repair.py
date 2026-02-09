"""CLI repair command — non-interactive autonomous repair.

Runs the Gemini-powered agent in autonomous mode to diagnose and fix
compatibility issues. Dry-run by default for safety.
"""

import json
from typing import Optional

import typer

from compatibillabuddy.agent.config import DEFAULT_MODEL, AgentConfig, resolve_api_key
from compatibillabuddy.agent.core import AgentSession


def repair_command(
    live: bool = typer.Option(  # noqa: B008
        False,
        "--live",
        help="Execute pip commands for real. Without this flag, runs in dry-run mode.",
    ),
    max_retries: int = typer.Option(  # noqa: B008
        3,
        "--max-retries",
        help="Max fix attempts per issue before skipping.",
    ),
    format: str = typer.Option(  # noqa: B008
        "console",
        "--format",
        "-f",
        help="Output format: 'console' or 'json'.",
    ),
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
    """Autonomous repair — diagnose and fix ML environment issues.

    By default runs in dry-run mode, showing what it WOULD do without
    executing any pip commands. Use --live to execute repairs for real.

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

    dry_run = not live

    # Show mode banner
    if format != "json":
        mode = "[DRY RUN]" if dry_run else "[LIVE MODE]"
        typer.echo(f"\n🔧 Compatibillabuddy Repair {mode}")
        typer.echo(f"   Model: {model}")
        typer.echo(f"   Max retries per issue: {max_retries}")
        typer.echo("")

    config = AgentConfig(api_key=key, model=model)
    session = AgentSession(config)

    result = session.auto_repair(dry_run=dry_run, max_retries=max_retries)

    if format == "json":
        output = json.dumps(
            {
                "summary": result.summary,
                "repair_log": result.repair_log,
                "success": result.success,
            },
            indent=2,
            default=str,
        )
        typer.echo(output)
    else:
        typer.echo(result.summary)

    if not result.success:
        raise typer.Exit(code=1)
