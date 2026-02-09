"""Tests for CLI doctor command.

TDD: tests written BEFORE wiring the doctor command into the CLI.
All tests mock diagnose() to avoid real subprocess calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from compatibillabuddy.cli import app
from compatibillabuddy.engine.models import (
    CompatIssue,
    DiagnosisResult,
    EnvironmentInventory,
    GpuInfo,
    GpuVendor,
    HardwareProfile,
    InstalledPackage,
    Severity,
)

runner = CliRunner()

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_HW = HardwareProfile(
    os_name="Linux",
    os_version="6.5.0",
    cpu_arch="x86_64",
    cpu_name="AMD Ryzen 9 7950X",
    python_version="3.11.7",
    gpus=[
        GpuInfo(
            vendor=GpuVendor.NVIDIA,
            name="NVIDIA RTX 4090",
            driver_version="545.29.06",
            cuda_version="12.3",
            vram_mb=24576,
        )
    ],
)

_ENV = EnvironmentInventory(
    python_version="3.11.7",
    python_executable="/usr/bin/python3.11",
    packages=[
        InstalledPackage(name="torch", version="2.1.0"),
        InstalledPackage(name="numpy", version="1.26.4"),
    ],
)

_ERROR_ISSUE = CompatIssue(
    severity=Severity.ERROR,
    category="cuda-mismatch",
    description="torch 2.1.0 requires CUDA 11.8 but CUDA 12.3 detected",
    affected_packages=["torch"],
    fix_suggestion="pip install torch==2.1.0+cu121",
)

_WARNING_ISSUE = CompatIssue(
    severity=Severity.WARNING,
    category="numpy-abi",
    description="pandas 2.2.0 was built against NumPy 1.x ABI",
    affected_packages=["pandas", "numpy"],
    fix_suggestion=None,
)


def _clean_result() -> DiagnosisResult:
    return DiagnosisResult(
        hardware=_HW,
        environment=_ENV,
        issues=[],
        hardware_probe_seconds=0.05,
        environment_inspect_seconds=0.12,
        rule_evaluation_seconds=0.003,
        total_seconds=0.173,
    )


def _error_result() -> DiagnosisResult:
    return DiagnosisResult(
        hardware=_HW,
        environment=_ENV,
        issues=[_ERROR_ISSUE, _WARNING_ISSUE],
        hardware_probe_seconds=0.05,
        environment_inspect_seconds=0.12,
        rule_evaluation_seconds=0.003,
        total_seconds=0.173,
    )


def _warning_only_result() -> DiagnosisResult:
    return DiagnosisResult(
        hardware=_HW,
        environment=_ENV,
        issues=[_WARNING_ISSUE],
        hardware_probe_seconds=0.05,
        environment_inspect_seconds=0.12,
        rule_evaluation_seconds=0.003,
        total_seconds=0.173,
    )


DIAGNOSE_PATH = "compatibillabuddy.cli.doctor.diagnose"

# ===========================================================================
# Tests
# ===========================================================================


class TestDoctorCommand:
    """Tests for `compatibuddy doctor` CLI command."""

    def test_doctor_runs_default(self):
        """Default invocation exits 0 with console output."""
        with patch(DIAGNOSE_PATH, return_value=_clean_result()):
            result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        assert "No issues found" in result.output

    def test_doctor_json_format(self):
        """--format json produces valid JSON output."""
        with patch(DIAGNOSE_PATH, return_value=_clean_result()):
            result = runner.invoke(app, ["doctor", "--format", "json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert isinstance(parsed, dict)
        assert "hardware" in parsed

    def test_doctor_exit_code_1_on_errors(self):
        """Exit code 1 when ERROR-severity issues are found."""
        with patch(DIAGNOSE_PATH, return_value=_error_result()):
            result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 1

    def test_doctor_exit_code_0_warnings_only(self):
        """Exit code 0 when only WARNING issues, no ERRORs."""
        with patch(DIAGNOSE_PATH, return_value=_warning_only_result()):
            result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0

    def test_doctor_output_to_file(self, tmp_path: Path):
        """--output writes report to a file instead of stdout."""
        outfile = tmp_path / "report.txt"
        with patch(DIAGNOSE_PATH, return_value=_clean_result()):
            result = runner.invoke(
                app, ["doctor", "--output", str(outfile)]
            )
        assert result.exit_code == 0
        contents = outfile.read_text(encoding="utf-8")
        assert "No issues found" in contents

    def test_doctor_console_contains_hardware(self):
        """Console output includes hardware info from the result."""
        with patch(DIAGNOSE_PATH, return_value=_clean_result()):
            result = runner.invoke(app, ["doctor"])
        assert "NVIDIA RTX 4090" in result.output
        assert "Linux" in result.output

    def test_doctor_json_roundtrips(self):
        """JSON output deserializes back to DiagnosisResult."""
        with patch(DIAGNOSE_PATH, return_value=_error_result()):
            result = runner.invoke(app, ["doctor", "--format", "json"])
        restored = DiagnosisResult.model_validate_json(result.output)
        assert restored.issue_count == 2
        assert restored.issues[0].severity == Severity.ERROR

    def test_doctor_json_clean_no_rich_markup(self):
        """JSON output contains no ANSI escape codes or Rich markup."""
        with patch(DIAGNOSE_PATH, return_value=_error_result()):
            result = runner.invoke(app, ["doctor", "--format", "json"])
        # ANSI escape codes start with \x1b[
        assert "\x1b[" not in result.output
        # No Rich markup tags like [bold red]
        assert "[bold" not in result.output
