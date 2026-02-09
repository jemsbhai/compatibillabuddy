"""Tests for engine.report — console and JSON report formatting.

TDD: these tests are written BEFORE the implementation in engine/report.py.
"""

from __future__ import annotations

import json

import pytest

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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def hw_profile() -> HardwareProfile:
    """Hardware profile with one NVIDIA GPU."""
    return HardwareProfile(
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


@pytest.fixture()
def env_inventory() -> EnvironmentInventory:
    """Environment with a few ML packages."""
    return EnvironmentInventory(
        python_version="3.11.7",
        python_executable="/usr/bin/python3.11",
        packages=[
            InstalledPackage(name="torch", version="2.1.0"),
            InstalledPackage(name="numpy", version="1.26.4"),
            InstalledPackage(name="pandas", version="2.2.0"),
        ],
    )


@pytest.fixture()
def clean_result(
    hw_profile: HardwareProfile,
    env_inventory: EnvironmentInventory,
) -> DiagnosisResult:
    """DiagnosisResult with no issues."""
    return DiagnosisResult(
        hardware=hw_profile,
        environment=env_inventory,
        issues=[],
        hardware_probe_seconds=0.05,
        environment_inspect_seconds=0.12,
        rule_evaluation_seconds=0.003,
        total_seconds=0.173,
    )


@pytest.fixture()
def error_issue() -> CompatIssue:
    return CompatIssue(
        severity=Severity.ERROR,
        category="cuda-mismatch",
        description=("torch 2.1.0 requires CUDA 11.8 but CUDA 12.3 detected"),
        affected_packages=["torch"],
        fix_suggestion="pip install torch==2.1.0+cu121",
    )


@pytest.fixture()
def warning_issue() -> CompatIssue:
    return CompatIssue(
        severity=Severity.WARNING,
        category="numpy-abi",
        description="pandas 2.2.0 was built against NumPy 1.x ABI",
        affected_packages=["pandas", "numpy"],
        fix_suggestion=None,
    )


@pytest.fixture()
def info_issue() -> CompatIssue:
    return CompatIssue(
        severity=Severity.INFO,
        category="deprecation",
        description=("numpy 1.26.x will drop Python 3.9 support soon"),
        affected_packages=["numpy"],
        fix_suggestion=None,
    )


@pytest.fixture()
def result_with_issues(
    hw_profile: HardwareProfile,
    env_inventory: EnvironmentInventory,
    error_issue: CompatIssue,
    warning_issue: CompatIssue,
    info_issue: CompatIssue,
) -> DiagnosisResult:
    """DiagnosisResult with one of each severity level."""
    return DiagnosisResult(
        hardware=hw_profile,
        environment=env_inventory,
        issues=[error_issue, warning_issue, info_issue],
        hardware_probe_seconds=0.05,
        environment_inspect_seconds=0.12,
        rule_evaluation_seconds=0.003,
        total_seconds=0.173,
    )


# ===========================================================================
# JSON report tests
# ===========================================================================


class TestFormatReportJson:
    """Tests for format_report_json()."""

    def test_json_is_valid_json(self, clean_result: DiagnosisResult):
        from compatibillabuddy.engine.report import format_report_json

        output = format_report_json(clean_result)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_json_roundtrip_clean(self, clean_result: DiagnosisResult):
        from compatibillabuddy.engine.report import format_report_json

        output = format_report_json(clean_result)
        restored = DiagnosisResult.model_validate_json(output)
        assert restored.issue_count == 0
        assert restored.hardware.os_name == clean_result.hardware.os_name
        assert restored.environment.python_version == clean_result.environment.python_version

    def test_json_roundtrip_with_issues(self, result_with_issues: DiagnosisResult):
        from compatibillabuddy.engine.report import format_report_json

        output = format_report_json(result_with_issues)
        restored = DiagnosisResult.model_validate_json(output)
        assert restored.issue_count == 3
        assert restored.issues[0].severity == Severity.ERROR
        assert restored.issues[1].severity == Severity.WARNING
        assert restored.issues[2].severity == Severity.INFO

    def test_json_contains_hardware_fields(self, clean_result: DiagnosisResult):
        from compatibillabuddy.engine.report import format_report_json

        output = format_report_json(clean_result)
        parsed = json.loads(output)
        assert parsed["hardware"]["os_name"] == "Linux"
        assert parsed["hardware"]["gpus"][0]["name"] == "NVIDIA RTX 4090"
        assert parsed["hardware"]["gpus"][0]["cuda_version"] == "12.3"

    def test_json_contains_timing(self, clean_result: DiagnosisResult):
        from compatibillabuddy.engine.report import format_report_json

        output = format_report_json(clean_result)
        parsed = json.loads(output)
        assert parsed["total_seconds"] == pytest.approx(0.173)
        assert parsed["hardware_probe_seconds"] == pytest.approx(0.05)


# ===========================================================================
# Console report tests
# ===========================================================================


class TestFormatReportConsole:
    """Tests for format_report_console()."""

    def test_console_clean_env(self, clean_result: DiagnosisResult):
        from compatibillabuddy.engine.report import format_report_console

        output = format_report_console(clean_result)
        assert "No issues found" in output

    def test_console_shows_hardware(self, clean_result: DiagnosisResult):
        from compatibillabuddy.engine.report import format_report_console

        output = format_report_console(clean_result)
        assert "Linux" in output
        assert "NVIDIA RTX 4090" in output

    def test_console_shows_python_version(self, clean_result: DiagnosisResult):
        from compatibillabuddy.engine.report import format_report_console

        output = format_report_console(clean_result)
        assert "3.11.7" in output

    def test_console_shows_error_issue(self, result_with_issues: DiagnosisResult):
        from compatibillabuddy.engine.report import format_report_console

        output = format_report_console(result_with_issues)
        assert "ERROR" in output
        assert "cuda-mismatch" in output
        assert "torch 2.1.0 requires CUDA 11.8 but CUDA 12.3 detected" in output

    def test_console_shows_warning_issue(self, result_with_issues: DiagnosisResult):
        from compatibillabuddy.engine.report import format_report_console

        output = format_report_console(result_with_issues)
        assert "WARNING" in output
        assert "numpy-abi" in output

    def test_console_shows_fix_suggestion(self, result_with_issues: DiagnosisResult):
        from compatibillabuddy.engine.report import format_report_console

        output = format_report_console(result_with_issues)
        assert "pip install torch==2.1.0+cu121" in output

    def test_console_omits_fix_when_none(
        self,
        hw_profile: HardwareProfile,
        env_inventory: EnvironmentInventory,
        warning_issue: CompatIssue,
    ):
        """When fix_suggestion is None, no 'Fix:' line should appear."""
        from compatibillabuddy.engine.report import format_report_console

        result = DiagnosisResult(
            hardware=hw_profile,
            environment=env_inventory,
            issues=[warning_issue],
        )
        output = format_report_console(result)
        assert "Fix:" not in output

    def test_console_shows_affected_packages(self, result_with_issues: DiagnosisResult):
        from compatibillabuddy.engine.report import format_report_console

        output = format_report_console(result_with_issues)
        assert "torch" in output
        assert "pandas" in output

    def test_console_shows_timing(self, clean_result: DiagnosisResult):
        from compatibillabuddy.engine.report import format_report_console

        output = format_report_console(clean_result)
        assert "0.17" in output

    def test_console_verdict_with_errors(self, result_with_issues: DiagnosisResult):
        from compatibillabuddy.engine.report import format_report_console

        output = format_report_console(result_with_issues)
        assert "1 error" in output.lower()

    def test_console_verdict_warnings_only(
        self,
        hw_profile: HardwareProfile,
        env_inventory: EnvironmentInventory,
        warning_issue: CompatIssue,
    ):
        from compatibillabuddy.engine.report import format_report_console

        result = DiagnosisResult(
            hardware=hw_profile,
            environment=env_inventory,
            issues=[warning_issue],
        )
        output = format_report_console(result)
        lower = output.lower()
        assert "1 warning" in lower
