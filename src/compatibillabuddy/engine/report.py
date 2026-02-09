"""Report formatting for DiagnosisResult.

Two output modes:
- JSON: structured, machine-readable (for piping to tools / Gemini agent)
- Console: human-readable with Rich formatting and severity-colored badges
"""

from __future__ import annotations

from io import StringIO

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from compatibillabuddy.engine.models import DiagnosisResult, Severity

# Severity → Rich colour mapping
_SEVERITY_STYLES: dict[Severity, str] = {
    Severity.ERROR: "bold red",
    Severity.WARNING: "bold yellow",
    Severity.INFO: "bold blue",
}


def format_report_json(result: DiagnosisResult) -> str:
    """Serialize a DiagnosisResult to a pretty-printed JSON string.

    The output round-trips cleanly via
    ``DiagnosisResult.model_validate_json()``.
    """
    return result.model_dump_json(indent=2)


def format_report_console(result: DiagnosisResult) -> str:
    """Render a DiagnosisResult as a human-readable Rich-formatted string.

    Sections: Hardware → Environment → Issues → Timing → Verdict.
    """
    buf = StringIO()
    console = Console(
        file=buf, width=100, force_terminal=False, no_color=True
    )

    _render_hardware(console, result)
    _render_environment(console, result)
    _render_issues(console, result)
    _render_timing(console, result)
    _render_verdict(console, result)

    return buf.getvalue()


def _render_hardware(console: Console, result: DiagnosisResult) -> None:
    """Render the hardware summary panel."""
    hw = result.hardware
    hw_lines: list[str] = [
        f"OS:     {hw.os_name} {hw.os_version}",
        f"CPU:    {hw.cpu_name} ({hw.cpu_arch})",
        f"Python: {hw.python_version}",
    ]
    for gpu in hw.gpus:
        extras = _gpu_extras(gpu)
        hw_lines.append(
            f"GPU:    {gpu.name} (driver {gpu.driver_version}{extras})"
        )
    console.print(
        Panel("\n".join(hw_lines), title="Hardware", border_style="cyan")
    )


def _gpu_extras(gpu) -> str:  # noqa: ANN001
    """Build the extra details string for a GPU line."""
    parts: list[str] = []
    if gpu.cuda_version:
        parts.append(f"CUDA {gpu.cuda_version}")
    if gpu.rocm_version:
        parts.append(f"ROCm {gpu.rocm_version}")
    if gpu.vram_mb:
        parts.append(f"{gpu.vram_mb} MB VRAM")
    return ", " + ", ".join(parts) if parts else ""


def _render_environment(
    console: Console, result: DiagnosisResult
) -> None:
    """Render the environment summary panel."""
    env = result.environment
    pkg_count = len(env.packages)
    console.print(
        Panel(
            f"Python: {env.python_version}\n"
            f"Packages installed: {pkg_count}",
            title="Environment",
            border_style="cyan",
        )
    )


def _render_issues(console: Console, result: DiagnosisResult) -> None:
    """Render the issues section — one panel per issue, or a clean banner."""
    if not result.issues:
        console.print(
            Panel(
                "No issues found! ✓",
                title="Diagnosis",
                border_style="green",
            )
        )
        return

    for issue in result.issues:
        style = _SEVERITY_STYLES.get(issue.severity, "")
        header = Text()
        header.append(f"[{issue.severity.name}]", style=style)
        header.append(f" {issue.category}")

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("key", style="dim", width=12)
        table.add_column("value")

        table.add_row("Description", issue.description)
        if issue.affected_packages:
            table.add_row(
                "Packages", ", ".join(issue.affected_packages)
            )
        if issue.fix_suggestion is not None:
            table.add_row("Fix:", issue.fix_suggestion)

        border = style.split()[-1] if style else "white"
        console.print(Panel(table, title=header, border_style=border))


def _render_timing(console: Console, result: DiagnosisResult) -> None:
    """Render the timing metadata panel."""
    console.print(
        Panel(
            f"Hardware probe:  {result.hardware_probe_seconds:.2f}s\n"
            f"Env inspection:  {result.environment_inspect_seconds:.2f}s\n"
            f"Rule evaluation: {result.rule_evaluation_seconds:.2f}s\n"
            f"Total:           {result.total_seconds:.2f}s",
            title="Timing",
            border_style="dim",
        )
    )


def _render_verdict(console: Console, result: DiagnosisResult) -> None:
    """Render the final verdict panel."""
    errors = sum(
        1 for i in result.issues if i.severity == Severity.ERROR
    )
    warnings = sum(
        1 for i in result.issues if i.severity == Severity.WARNING
    )
    infos = sum(
        1 for i in result.issues if i.severity == Severity.INFO
    )

    if errors:
        parts = [f"{errors} error{'s' if errors != 1 else ''}"]
        if warnings:
            parts.append(
                f"{warnings} warning{'s' if warnings != 1 else ''}"
            )
        if infos:
            parts.append(f"{infos} info")
        verdict = "Found " + ", ".join(parts)
        console.print(
            Panel(verdict, title="Verdict", border_style="red")
        )
    elif warnings:
        parts = [f"{warnings} warning{'s' if warnings != 1 else ''}"]
        if infos:
            parts.append(f"{infos} info")
        verdict = "Found " + ", ".join(parts)
        console.print(
            Panel(verdict, title="Verdict", border_style="yellow")
        )
    else:
        console.print(
            Panel("All clear!", title="Verdict", border_style="green")
        )
