"""CLI doctor command: run diagnosis and output report.

Calls diagnose() and formats the result for console or JSON output.
Designed for both human use and agent interop (--format json).
"""

from __future__ import annotations

from pathlib import Path

import typer

from compatibillabuddy.engine.doctor import diagnose
from compatibillabuddy.engine.report import format_report_console, format_report_json


def doctor(
    format: str = typer.Option(  # noqa: B008
        "console",
        "--format",
        "-f",
        help="Output format: 'console' (human-readable) or 'json' (machine-readable).",
    ),
    output: Path | None = typer.Option(  # noqa: B008
        None,
        "--output",
        "-o",
        help="Write report to file instead of stdout.",
    ),
) -> None:
    """Run a full compatibility diagnosis of your ML environment.

    Probes hardware, inspects installed packages, and evaluates
    compatibility rules. Returns exit code 1 if errors are found.
    """
    result = diagnose()

    report = (
        format_report_json(result)
        if format == "json"
        else format_report_console(result)
    )

    if output is not None:
        output.write_text(report, encoding="utf-8")
    else:
        typer.echo(report)

    if result.has_errors:
        raise typer.Exit(code=1)
