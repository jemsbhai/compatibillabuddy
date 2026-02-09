"""Tests for the doctor diagnostic orchestrator — written BEFORE implementation (TDD).

Covers:
- Full diagnosis with injected dependencies (no subprocess calls)
- Severity sorting (errors first, then warnings, then info)
- Timing metadata population
- Default behavior (calls probe/inspector/loader when args are None)
- Error propagation from subsystems
- Edge cases: empty rules, clean environment, no GPU
"""

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


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def hw_nvidia_cuda118():
    """NVIDIA GPU with CUDA 11.8 — triggers torch >=2.4 CUDA mismatch."""
    return HardwareProfile(
        os_name="Linux",
        os_version="6.1.0",
        cpu_arch="x86_64",
        cpu_name="Intel Xeon",
        python_version="3.12.0",
        gpus=[
            GpuInfo(
                vendor=GpuVendor.NVIDIA,
                name="RTX 3090",
                driver_version="520.61",
                cuda_version="11.8",
                vram_mb=24576,
            )
        ],
    )


@pytest.fixture
def hw_cpu_only():
    """CPU-only machine — no GPU rules should fire."""
    return HardwareProfile(
        os_name="Linux",
        os_version="6.1.0",
        cpu_arch="x86_64",
        cpu_name="Intel Xeon",
        python_version="3.12.0",
    )


@pytest.fixture
def env_torch24_numpy2():
    """Environment with torch 2.4 and numpy 2.0 + old pandas."""
    return EnvironmentInventory(
        python_version="3.12.0",
        python_executable="/usr/bin/python3",
        packages=[
            InstalledPackage(name="torch", version="2.4.0"),
            InstalledPackage(name="numpy", version="2.0.0"),
            InstalledPackage(name="pandas", version="2.0.3"),
        ],
    )


@pytest.fixture
def env_clean():
    """Clean environment with no ML packages."""
    return EnvironmentInventory(
        python_version="3.12.0",
        python_executable="/usr/bin/python3",
        packages=[
            InstalledPackage(name="pip", version="24.0"),
            InstalledPackage(name="setuptools", version="69.0"),
        ],
    )


@pytest.fixture
def bundled_rules():
    """Load the real bundled rulepacks for integration-style tests."""
    from compatibillabuddy.kb.engine import load_bundled_rulepacks

    return load_bundled_rulepacks()


# ===========================================================================
# Core diagnosis tests
# ===========================================================================


class TestDiagnose:
    """Tests for the diagnose() orchestrator function."""

    def test_clean_environment_no_issues(self, hw_cpu_only, env_clean, bundled_rules):
        """A clean environment with no ML packages should produce zero issues."""
        from compatibillabuddy.engine.doctor import diagnose

        result = diagnose(hardware=hw_cpu_only, env=env_clean, rules=bundled_rules)

        assert isinstance(result, DiagnosisResult)
        assert result.issue_count == 0
        assert result.has_errors is False
        assert result.has_warnings is False

    def test_cuda_mismatch_detected(self, hw_nvidia_cuda118, env_torch24_numpy2, bundled_rules):
        """Torch 2.4 on CUDA 11.8 should trigger a CUDA mismatch error."""
        from compatibillabuddy.engine.doctor import diagnose

        result = diagnose(
            hardware=hw_nvidia_cuda118,
            env=env_torch24_numpy2,
            rules=bundled_rules,
        )

        cuda_issues = [i for i in result.issues if i.category == "cuda_mismatch"]
        assert len(cuda_issues) >= 1
        assert cuda_issues[0].severity == Severity.ERROR

    def test_numpy_abi_issue_detected(self, hw_cpu_only, env_torch24_numpy2, bundled_rules):
        """numpy 2.0 + pandas 2.0.3 should trigger a numpy ABI error."""
        from compatibillabuddy.engine.doctor import diagnose

        result = diagnose(
            hardware=hw_cpu_only,
            env=env_torch24_numpy2,
            rules=bundled_rules,
        )

        abi_issues = [i for i in result.issues if i.category == "numpy_abi"]
        assert len(abi_issues) >= 1
        assert abi_issues[0].severity == Severity.ERROR

    def test_multiple_issues_detected(self, hw_nvidia_cuda118, env_torch24_numpy2, bundled_rules):
        """An environment with multiple problems should report all of them."""
        from compatibillabuddy.engine.doctor import diagnose

        result = diagnose(
            hardware=hw_nvidia_cuda118,
            env=env_torch24_numpy2,
            rules=bundled_rules,
        )

        # Should have at least: CUDA mismatch + numpy ABI + pandas deprecation
        assert result.issue_count >= 3

    def test_empty_rules_no_issues(self, hw_nvidia_cuda118, env_torch24_numpy2):
        """An empty rule list should produce zero issues regardless of environment."""
        from compatibillabuddy.engine.doctor import diagnose

        result = diagnose(
            hardware=hw_nvidia_cuda118,
            env=env_torch24_numpy2,
            rules=[],
        )

        assert result.issue_count == 0

    def test_result_contains_hardware_profile(self, hw_nvidia_cuda118, env_clean, bundled_rules):
        """The result should contain the exact hardware profile passed in."""
        from compatibillabuddy.engine.doctor import diagnose

        result = diagnose(
            hardware=hw_nvidia_cuda118,
            env=env_clean,
            rules=bundled_rules,
        )

        assert result.hardware is hw_nvidia_cuda118

    def test_result_contains_environment(self, hw_cpu_only, env_torch24_numpy2, bundled_rules):
        """The result should contain the exact environment inventory passed in."""
        from compatibillabuddy.engine.doctor import diagnose

        result = diagnose(
            hardware=hw_cpu_only,
            env=env_torch24_numpy2,
            rules=bundled_rules,
        )

        assert result.environment is env_torch24_numpy2


# ===========================================================================
# Severity sorting tests
# ===========================================================================


class TestDiagnoseSorting:
    """Issues should be sorted by severity: ERROR first, then WARNING, then INFO."""

    def test_issues_sorted_by_severity(self):
        """When multiple severity levels fire, errors should come first."""
        from compatibillabuddy.engine.doctor import diagnose
        from compatibillabuddy.kb.engine import Rule, RuleCondition

        # Craft rules that fire in INFO, ERROR, WARNING order (wrong order)
        rules = [
            Rule(
                id="info-rule",
                severity=Severity.INFO,
                category="deprecation",
                description="Something deprecated",
                when=RuleCondition(package_installed="torch"),
            ),
            Rule(
                id="error-rule",
                severity=Severity.ERROR,
                category="cuda_mismatch",
                description="CUDA broken",
                when=RuleCondition(package_installed="torch"),
            ),
            Rule(
                id="warning-rule",
                severity=Severity.WARNING,
                category="coinstall",
                description="Potential conflict",
                when=RuleCondition(package_installed="torch"),
            ),
        ]

        hw = HardwareProfile(
            os_name="Linux", os_version="6.1.0", cpu_arch="x86_64",
            cpu_name="Intel Xeon", python_version="3.12.0",
        )
        env = EnvironmentInventory(
            python_version="3.12.0", python_executable="/usr/bin/python3",
            packages=[InstalledPackage(name="torch", version="2.4.0")],
        )

        result = diagnose(hardware=hw, env=env, rules=rules)

        assert result.issue_count == 3
        assert result.issues[0].severity == Severity.ERROR
        assert result.issues[1].severity == Severity.WARNING
        assert result.issues[2].severity == Severity.INFO

    def test_same_severity_preserves_order(self):
        """Issues with the same severity should preserve rule evaluation order."""
        from compatibillabuddy.engine.doctor import diagnose
        from compatibillabuddy.kb.engine import Rule, RuleCondition

        rules = [
            Rule(
                id="error-a",
                severity=Severity.ERROR,
                category="cuda_mismatch",
                description="Error A",
                when=RuleCondition(package_installed="torch"),
            ),
            Rule(
                id="error-b",
                severity=Severity.ERROR,
                category="numpy_abi",
                description="Error B",
                when=RuleCondition(package_installed="torch"),
            ),
        ]

        hw = HardwareProfile(
            os_name="Linux", os_version="6.1.0", cpu_arch="x86_64",
            cpu_name="Intel Xeon", python_version="3.12.0",
        )
        env = EnvironmentInventory(
            python_version="3.12.0", python_executable="/usr/bin/python3",
            packages=[InstalledPackage(name="torch", version="2.4.0")],
        )

        result = diagnose(hardware=hw, env=env, rules=rules)

        assert result.issues[0].description == "Error A"
        assert result.issues[1].description == "Error B"


# ===========================================================================
# Timing metadata tests
# ===========================================================================


class TestDiagnoseTiming:
    """Timing metadata should be populated by diagnose()."""

    def test_total_seconds_populated(self, hw_cpu_only, env_clean, bundled_rules):
        """total_seconds should be > 0 after a diagnosis run."""
        from compatibillabuddy.engine.doctor import diagnose

        result = diagnose(hardware=hw_cpu_only, env=env_clean, rules=bundled_rules)

        assert result.total_seconds >= 0.0

    def test_rule_evaluation_seconds_populated(self, hw_cpu_only, env_clean, bundled_rules):
        """rule_evaluation_seconds should be >= 0 when rules are provided."""
        from compatibillabuddy.engine.doctor import diagnose

        result = diagnose(hardware=hw_cpu_only, env=env_clean, rules=bundled_rules)

        assert result.rule_evaluation_seconds >= 0.0

    def test_injected_deps_skip_probe_and_inspect_timing(self, hw_cpu_only, env_clean, bundled_rules):
        """When hardware/env are injected, their timing should be 0.0."""
        from compatibillabuddy.engine.doctor import diagnose

        result = diagnose(hardware=hw_cpu_only, env=env_clean, rules=bundled_rules)

        assert result.hardware_probe_seconds == 0.0
        assert result.environment_inspect_seconds == 0.0


# ===========================================================================
# Default behavior tests (mocked subprocess calls)
# ===========================================================================


class TestDiagnoseDefaults:
    """When args are None, diagnose() should call the real subsystems."""

    def test_calls_probe_when_hardware_is_none(self, env_clean, bundled_rules):
        """diagnose(hardware=None) should call probe_hardware()."""
        from unittest.mock import patch

        from compatibillabuddy.engine.doctor import diagnose

        mock_hw = HardwareProfile(
            os_name="MockOS", os_version="1.0", cpu_arch="x86_64",
            cpu_name="Mock CPU", python_version="3.12.0",
        )

        with patch(
            "compatibillabuddy.engine.doctor.probe_hardware", return_value=mock_hw
        ) as mock_probe:
            result = diagnose(hardware=None, env=env_clean, rules=bundled_rules)

        mock_probe.assert_called_once()
        assert result.hardware.os_name == "MockOS"
        assert result.hardware_probe_seconds >= 0.0

    def test_calls_inspector_when_env_is_none(self, hw_cpu_only, bundled_rules):
        """diagnose(env=None) should call inspect_environment()."""
        from unittest.mock import patch

        from compatibillabuddy.engine.doctor import diagnose

        mock_env = EnvironmentInventory(
            python_version="3.12.0", python_executable="/mock/python",
        )

        with patch(
            "compatibillabuddy.engine.doctor.inspect_environment", return_value=mock_env
        ) as mock_inspect:
            result = diagnose(hardware=hw_cpu_only, env=None, rules=bundled_rules)

        mock_inspect.assert_called_once()
        assert result.environment.python_executable == "/mock/python"
        assert result.environment_inspect_seconds >= 0.0

    def test_loads_bundled_rules_when_rules_is_none(self, hw_cpu_only, env_clean):
        """diagnose(rules=None) should call load_bundled_rulepacks()."""
        from unittest.mock import patch

        from compatibillabuddy.engine.doctor import diagnose

        with patch(
            "compatibillabuddy.engine.doctor.load_bundled_rulepacks", return_value=[]
        ) as mock_load:
            result = diagnose(hardware=hw_cpu_only, env=env_clean, rules=None)

        mock_load.assert_called_once()
        assert result.issue_count == 0


# ===========================================================================
# Error propagation tests
# ===========================================================================


class TestDiagnoseErrors:
    """Errors from subsystems should propagate cleanly."""

    def test_probe_failure_propagates(self, env_clean, bundled_rules):
        """If probe_hardware() raises, diagnose() should propagate the error."""
        from unittest.mock import patch

        from compatibillabuddy.engine.doctor import diagnose

        with patch(
            "compatibillabuddy.engine.doctor.probe_hardware",
            side_effect=RuntimeError("nvidia-smi exploded"),
        ):
            with pytest.raises(RuntimeError, match="nvidia-smi exploded"):
                diagnose(hardware=None, env=env_clean, rules=bundled_rules)

    def test_inspector_failure_propagates(self, hw_cpu_only, bundled_rules):
        """If inspect_environment() raises, diagnose() should propagate the error."""
        from unittest.mock import patch

        from compatibillabuddy.engine.doctor import diagnose

        with patch(
            "compatibillabuddy.engine.doctor.inspect_environment",
            side_effect=RuntimeError("pip not found"),
        ):
            with pytest.raises(RuntimeError, match="pip not found"):
                diagnose(hardware=hw_cpu_only, env=None, rules=bundled_rules)
