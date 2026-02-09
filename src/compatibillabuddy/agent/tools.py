"""Agent tool definitions — thin wrappers for Gemini function calling.

Each tool wraps an existing compatibillabuddy function, returning a plain
dict that is JSON-serializable. Docstrings serve as Gemini tool descriptions.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from compatibillabuddy.engine.doctor import diagnose
from compatibillabuddy.engine.models import DiagnosisResult
from compatibillabuddy.hardware.inspector import inspect_environment
from compatibillabuddy.hardware.probe import probe_hardware
from compatibillabuddy.kb.engine import load_bundled_rulepacks

# ---------------------------------------------------------------------------
# Repair tool constants
# ---------------------------------------------------------------------------

_PROTECTED_PACKAGES: frozenset[str] = frozenset({"pip", "setuptools", "wheel", "compatibillabuddy"})
_ALLOWED_PIP_ACTIONS: frozenset[str] = frozenset({"install", "uninstall"})
_MAX_PIP_OPS: int = 10
_pip_op_count: int = 0


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
        try:
            if rule.when.package_installed and rule.when.package_installed.lower() == name_lower:
                matching.append(rule.model_dump(mode="json"))
                continue
            if name_lower in rule.description.lower():
                matching.append(rule.model_dump(mode="json"))
        except Exception:
            # Serialization edge case — include minimal info
            matching.append({"id": rule.id, "description": rule.description})

    return {"package": package_name, "rules": matching}


# ===========================================================================
# Repair tools
# ===========================================================================


def _is_in_virtualenv() -> bool:
    """Check if running inside a virtual environment or conda env."""
    import os

    if sys.prefix != sys.base_prefix:
        return True
    return bool(os.environ.get("CONDA_PREFIX"))


def _extract_base_package_name(package: str) -> str:
    """Extract the base package name from a specifier like 'torch==2.1.0+cu121'."""
    for sep in ("==", ">=", "<=", "!=", "~=", ">", "<"):
        if sep in package:
            return package.split(sep)[0].strip().lower()
    return package.strip().lower()


def tool_snapshot_environment() -> dict:
    """Capture the current environment state as a rollback point.

    Runs pip freeze and records a timestamped snapshot of all installed
    packages. Use this BEFORE making any changes so you can rollback.

    Returns:
        Dictionary with 'timestamp' (ISO format) and 'packages' (list of
        specifier strings like 'torch==2.1.0').
    """
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
    )
    packages = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    return {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "packages": packages,
    }


def tool_run_pip(action: str, package: str, dry_run: bool = True) -> dict:
    """Execute a pip install or uninstall with safety guardrails.

    Args:
        action: Either 'install' or 'uninstall'.
        package: Package specifier (e.g. 'torch==2.1.0+cu121').
        dry_run: If True (default), shows the command without executing.

    Returns:
        Dictionary with 'success', 'action', 'package', and result details.
        On failure, includes 'error' with a human-readable message.
    """
    global _pip_op_count  # noqa: PLW0603

    # --- Safety: valid action ---
    if action not in _ALLOWED_PIP_ACTIONS:
        return {
            "success": False,
            "error": f"Invalid action '{action}'. Allowed actions: install, uninstall.",
        }

    # --- Safety: must be in virtualenv ---
    if not _is_in_virtualenv():
        return {
            "success": False,
            "error": (
                "Refused: not running inside a virtual environment. "
                "Create a venv first to protect your system Python."
            ),
        }

    # --- Safety: blocklist ---
    base_name = _extract_base_package_name(package)
    if base_name in _PROTECTED_PACKAGES:
        return {
            "success": False,
            "error": (
                f"Refused: '{base_name}' is on the protected blocklist. "
                f"Protected packages: {', '.join(sorted(_PROTECTED_PACKAGES))}."
            ),
        }

    # --- Safety: operation limit ---
    if _pip_op_count >= _MAX_PIP_OPS:
        return {
            "success": False,
            "error": (
                f"Operation limit reached ({_MAX_PIP_OPS} pip commands per session). "
                "Start a new session to continue."
            ),
        }

    # --- Build command ---
    cmd = [sys.executable, "-m", "pip", action, package]
    if action == "uninstall":
        cmd.append("--yes")

    # --- Dry run ---
    if dry_run:
        return {
            "success": True,
            "dry_run": True,
            "action": action,
            "package": package,
            "command": " ".join(cmd),
        }

    # --- Execute ---
    _pip_op_count += 1
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return {
            "success": False,
            "action": action,
            "package": package,
            "error": result.stderr or result.stdout,
        }

    return {
        "success": True,
        "dry_run": False,
        "action": action,
        "package": package,
        "output": result.stdout,
    }


def tool_verify_fix(previous_diagnosis_json: str) -> dict:
    """Re-run the doctor and compare against a previous diagnosis.

    Args:
        previous_diagnosis_json: JSON string from a prior tool_run_doctor() call.

    Returns:
        Dictionary comparing previous vs current issue counts, indicating
        whether the environment improved.
    """
    try:
        previous = json.loads(previous_diagnosis_json)
    except (json.JSONDecodeError, TypeError) as e:
        return {"error": f"Failed to parse previous diagnosis JSON: {e}"}

    previous_issues = previous.get("issues", [])
    previous_count = len(previous_issues)

    current_result = diagnose()
    current_count = len(current_result.issues)

    resolved = max(0, previous_count - current_count)
    new = max(0, current_count - previous_count)

    return {
        "previous_issue_count": previous_count,
        "current_issue_count": current_count,
        "issues_resolved": resolved,
        "new_issues": new,
        "improved": current_count < previous_count,
        "current_diagnosis": current_result.model_dump(),
    }


def tool_rollback(snapshot_json: str) -> dict:
    """Restore packages to a previously captured snapshot.

    Args:
        snapshot_json: JSON string from a prior tool_snapshot_environment() call.

    Returns:
        Dictionary with 'success' and 'restored_count', or 'error' on failure.
    """
    # --- Safety: must be in virtualenv ---
    if not _is_in_virtualenv():
        return {
            "success": False,
            "error": (
                "Refused: not running inside a virtual environment. "
                "Cannot rollback on system Python."
            ),
        }

    try:
        snapshot = json.loads(snapshot_json)
    except (json.JSONDecodeError, TypeError) as e:
        return {"success": False, "error": f"Failed to parse snapshot JSON: {e}"}

    packages = snapshot.get("packages", [])
    if not packages:
        return {"success": False, "error": "Snapshot contains no packages."}

    # Write packages to a temp requirements file and pip install
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="compatibuddy_rollback_"
    ) as f:
        f.write("\n".join(packages))
        req_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", req_path],
            capture_output=True,
            text=True,
        )
    finally:
        Path(req_path).unlink(missing_ok=True)

    if result.returncode != 0:
        return {
            "success": False,
            "error": f"Rollback failed: {result.stderr or result.stdout}",
        }

    return {
        "success": True,
        "restored_count": len(packages),
        "timestamp": snapshot.get("timestamp", "unknown"),
    }
