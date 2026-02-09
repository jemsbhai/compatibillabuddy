"""Environment inspector: detect what's installed in a Python environment.

Calls `pip inspect` as a subprocess and parses the stable JSON output
into an EnvironmentInventory model. No pip internals are imported.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Optional

from compatibillabuddy.engine.models import EnvironmentInventory, InstalledPackage


def inspect_environment(python_executable: Optional[str] = None) -> EnvironmentInventory:
    """Inspect the current (or specified) Python environment.

    Calls `pip inspect` and parses the JSON output.

    Args:
        python_executable: Path to a Python interpreter. Defaults to sys.executable.

    Returns:
        EnvironmentInventory with all installed packages.

    Raises:
        RuntimeError: If pip is not found or pip inspect fails.
    """
    python = python_executable or sys.executable
    raw_json = _run_pip_inspect(python)
    return parse_pip_inspect(raw_json)


def parse_pip_inspect(raw_json: str) -> EnvironmentInventory:
    """Parse raw JSON output from `pip inspect` into an EnvironmentInventory.

    Args:
        raw_json: The JSON string from pip inspect stdout.

    Returns:
        EnvironmentInventory populated from the JSON.

    Raises:
        ValueError: If the JSON is invalid or missing required fields.
    """
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse pip inspect JSON: {e}") from e

    if "environment" not in data:
        raise ValueError("pip inspect output missing 'environment' key")

    env = data["environment"]
    python_version = env.get("python_version", "")
    python_executable = env.get("python_executable", "")

    packages = []
    for entry in data.get("installed", []):
        meta = entry.get("metadata", {})
        name = meta.get("name", "")
        version = meta.get("version", "")

        if not name:
            continue

        requires = meta.get("requires_dist", []) or []
        installer = entry.get("installer", None)
        location = entry.get("metadata_location", None)

        packages.append(
            InstalledPackage(
                name=name,
                version=version,
                requires=requires,
                installer=installer,
                location=location,
            )
        )

    return EnvironmentInventory(
        python_version=python_version,
        python_executable=python_executable,
        packages=packages,
    )


def _run_pip_inspect(python_executable: str) -> str:
    """Run `pip inspect` and return raw JSON stdout.

    Args:
        python_executable: Path to the Python interpreter to use.

    Returns:
        Raw JSON string from pip inspect.

    Raises:
        RuntimeError: If pip is not found or the command fails.
    """
    try:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"

        result = subprocess.run(
            [python_executable, "-m", "pip", "inspect"],
            capture_output=True,
            text=True,
            timeout=60,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"pip not found at '{python_executable}'. "
            "Ensure pip is installed in the target environment."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("pip inspect timed out after 60 seconds.")

    if result.returncode != 0:
        raise RuntimeError(
            f"pip inspect failed (exit code {result.returncode}): {result.stderr}"
        )

    return result.stdout
