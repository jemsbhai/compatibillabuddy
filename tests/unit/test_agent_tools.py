"""Tests for agent tool definitions — thin wrappers for Gemini function calling.

TDD: tests written BEFORE implementation in agent/tools.py.
All tests mock underlying functions to avoid subprocess calls.
"""

from __future__ import annotations

import json
from unittest.mock import patch

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
# Shared test data
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

_DIAGNOSIS = DiagnosisResult(
    hardware=_HW,
    environment=_ENV,
    issues=[
        CompatIssue(
            severity=Severity.ERROR,
            category="cuda-mismatch",
            description="torch 2.1.0 requires CUDA 11.8 but CUDA 12.3 detected",
            affected_packages=["torch"],
            fix_suggestion="pip install torch==2.1.0+cu121",
        ),
        CompatIssue(
            severity=Severity.WARNING,
            category="numpy-abi",
            description="pandas 2.2.0 was built against NumPy 1.x ABI",
            affected_packages=["pandas", "numpy"],
            fix_suggestion=None,
        ),
    ],
    hardware_probe_seconds=0.05,
    environment_inspect_seconds=0.12,
    rule_evaluation_seconds=0.003,
    total_seconds=0.173,
)


# ===========================================================================
# tool_probe_hardware
# ===========================================================================


class TestToolProbeHardware:
    """Tests for tool_probe_hardware()."""

    def test_returns_dict(self):
        from compatibillabuddy.agent.tools import tool_probe_hardware

        with patch("compatibillabuddy.agent.tools.probe_hardware", return_value=_HW):
            result = tool_probe_hardware()
        assert isinstance(result, dict)
        assert "os_name" in result
        assert "python_version" in result

    def test_has_gpu_info(self):
        from compatibillabuddy.agent.tools import tool_probe_hardware

        with patch("compatibillabuddy.agent.tools.probe_hardware", return_value=_HW):
            result = tool_probe_hardware()
        assert len(result["gpus"]) == 1
        assert result["gpus"][0]["name"] == "NVIDIA RTX 4090"
        assert result["gpus"][0]["cuda_version"] == "12.3"


# ===========================================================================
# tool_inspect_environment
# ===========================================================================


class TestToolInspectEnvironment:
    """Tests for tool_inspect_environment()."""

    def test_returns_dict(self):
        from compatibillabuddy.agent.tools import tool_inspect_environment

        with patch(
            "compatibillabuddy.agent.tools.inspect_environment",
            return_value=_ENV,
        ):
            result = tool_inspect_environment()
        assert isinstance(result, dict)
        assert "packages" in result
        assert "python_version" in result

    def test_packages_present(self):
        from compatibillabuddy.agent.tools import tool_inspect_environment

        with patch(
            "compatibillabuddy.agent.tools.inspect_environment",
            return_value=_ENV,
        ):
            result = tool_inspect_environment()
        names = [p["name"] for p in result["packages"]]
        assert "torch" in names
        assert "numpy" in names


# ===========================================================================
# tool_run_doctor
# ===========================================================================


class TestToolRunDoctor:
    """Tests for tool_run_doctor()."""

    def test_returns_dict(self):
        from compatibillabuddy.agent.tools import tool_run_doctor

        with patch(
            "compatibillabuddy.agent.tools.diagnose",
            return_value=_DIAGNOSIS,
        ):
            result = tool_run_doctor()
        assert isinstance(result, dict)
        assert "issues" in result
        assert result["timing"]["total_seconds"] > 0

    def test_issues_serialized(self):
        from compatibillabuddy.agent.tools import tool_run_doctor

        with patch(
            "compatibillabuddy.agent.tools.diagnose",
            return_value=_DIAGNOSIS,
        ):
            result = tool_run_doctor()
        assert isinstance(result["issues"], list)
        assert len(result["issues"]) == 2
        assert result["issues"][0]["severity"] == "ERROR"
        assert result["issues"][0]["category"] == "cuda-mismatch"


# ===========================================================================
# tool_explain_issue
# ===========================================================================


class TestToolExplainIssue:
    """Tests for tool_explain_issue()."""

    def test_valid_index(self):
        from compatibillabuddy.agent.tools import tool_explain_issue

        diagnosis_json = _DIAGNOSIS.model_dump_json()
        result = tool_explain_issue(issue_index=0, diagnosis_json=diagnosis_json)
        assert isinstance(result, dict)
        assert "severity" in result
        assert "description" in result
        assert "cuda-mismatch" in result["category"]

    def test_invalid_index(self):
        from compatibillabuddy.agent.tools import tool_explain_issue

        diagnosis_json = _DIAGNOSIS.model_dump_json()
        result = tool_explain_issue(issue_index=99, diagnosis_json=diagnosis_json)
        assert "error" in result


# ===========================================================================
# tool_search_rules
# ===========================================================================


class TestToolSearchRules:
    """Tests for tool_search_rules()."""

    def test_finds_match(self):
        from compatibillabuddy.agent.tools import tool_search_rules

        result = tool_search_rules(package_name="torch")
        assert isinstance(result, dict)
        assert "rules" in result
        assert len(result["rules"]) > 0

    def test_no_match(self):
        from compatibillabuddy.agent.tools import tool_search_rules

        result = tool_search_rules(package_name="nonexistent-pkg-xyz")
        assert isinstance(result, dict)
        assert len(result["rules"]) == 0


# ===========================================================================
# JSON serializability
# ===========================================================================


class TestJsonSerializable:
    """All tool outputs must be json.dumps()-able."""

    def test_probe_hardware_serializable(self):
        from compatibillabuddy.agent.tools import tool_probe_hardware

        with patch("compatibillabuddy.agent.tools.probe_hardware", return_value=_HW):
            result = tool_probe_hardware()
        assert json.dumps(result)

    def test_inspect_environment_serializable(self):
        from compatibillabuddy.agent.tools import tool_inspect_environment

        with patch(
            "compatibillabuddy.agent.tools.inspect_environment",
            return_value=_ENV,
        ):
            result = tool_inspect_environment()
        assert json.dumps(result)

    def test_run_doctor_serializable(self):
        from compatibillabuddy.agent.tools import tool_run_doctor

        with patch(
            "compatibillabuddy.agent.tools.diagnose",
            return_value=_DIAGNOSIS,
        ):
            result = tool_run_doctor()
        assert json.dumps(result)

    def test_explain_issue_serializable(self):
        from compatibillabuddy.agent.tools import tool_explain_issue

        result = tool_explain_issue(issue_index=0, diagnosis_json=_DIAGNOSIS.model_dump_json())
        assert json.dumps(result)

    def test_search_rules_serializable(self):
        from compatibillabuddy.agent.tools import tool_search_rules

        result = tool_search_rules(package_name="torch")
        assert json.dumps(result)
