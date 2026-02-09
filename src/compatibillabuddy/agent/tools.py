"""Agent tool definitions — thin wrappers for Gemini function calling.

Each tool wraps an existing compatibillabuddy function, returning a plain
dict that is JSON-serializable. Docstrings serve as Gemini tool descriptions.
"""

from __future__ import annotations

from compatibillabuddy.engine.doctor import diagnose
from compatibillabuddy.engine.models import DiagnosisResult
from compatibillabuddy.hardware.inspector import inspect_environment
from compatibillabuddy.hardware.probe import probe_hardware
from compatibillabuddy.kb.engine import load_bundled_rulepacks


def tool_probe_hardware() -> dict:
    """Probe the current machine's hardware.

    Detects OS, CPU, Python version, and GPU info (NVIDIA via nvidia-smi).
    Returns a dictionary with the full hardware profile.
    """
    hw = probe_hardware()
    return hw.model_dump()


def tool_inspect_environment() -> dict:
    """Inspect the current Python environment.

    Lists all installed packages with their versions by calling pip inspect.
    Returns a dictionary with python version and installed packages.
    """
    env = inspect_environment()
    return env.model_dump()


def tool_run_doctor() -> dict:
    """Run a full compatibility diagnosis.

    Probes hardware, inspects installed packages, evaluates compatibility
    rules, and returns all discovered issues sorted by severity.
    """
    result = diagnose()
    return result.model_dump()


def tool_explain_issue(issue_index: int, diagnosis_json: str) -> dict:
    """Explain a specific issue from a diagnosis result in detail.

    Args:
        issue_index: Zero-based index of the issue in the diagnosis issues list.
        diagnosis_json: JSON string of a DiagnosisResult from tool_run_doctor.

    Returns:
        Dictionary with the issue details, or an error dict if index is invalid.
    """
    try:
        result = DiagnosisResult.model_validate_json(diagnosis_json)
    except Exception as e:
        return {"error": f"Failed to parse diagnosis JSON: {e}"}

    if issue_index < 0 or issue_index >= len(result.issues):
        return {
            "error": (
                f"Issue index {issue_index} out of range. "
                f"There are {len(result.issues)} issues (0-{len(result.issues) - 1})."
            )
        }

    issue = result.issues[issue_index]
    return {
        "severity": issue.severity.name,
        "category": issue.category,
        "description": issue.description,
        "affected_packages": issue.affected_packages,
        "fix_suggestion": issue.fix_suggestion,
    }


def tool_search_rules(package_name: str) -> dict:
    """Search bundled rulepacks for rules that mention a specific package.

    Args:
        package_name: The package name to search for (case-insensitive).

    Returns:
        Dictionary with a list of matching rules.
    """
    rules = load_bundled_rulepacks()
    name_lower = package_name.lower()

    matching = []
    for rule in rules:
        if rule.when.package_installed and rule.when.package_installed.lower() == name_lower:
            matching.append(rule.model_dump())
            continue
        if name_lower in rule.description.lower():
            matching.append(rule.model_dump())

    return {"package": package_name, "rules": matching}
