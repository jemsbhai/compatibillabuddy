"""Tests for core data models — written BEFORE implementation (TDD)."""

import pytest
from pydantic import ValidationError


class TestHardwareProfile:
    """Tests for HardwareProfile model."""

    def test_create_full_nvidia_profile(self):
        """A complete profile with NVIDIA GPU should store all fields."""
        from compatibillabuddy.engine.models import GpuInfo, HardwareProfile

        gpu = GpuInfo(
            vendor="nvidia",
            name="NVIDIA GeForce RTX 4090",
            driver_version="560.94",
            cuda_version="12.6",
            compute_capability="8.9",
            vram_mb=24564,
        )
        profile = HardwareProfile(
            os_name="Windows",
            os_version="10.0.22631",
            cpu_arch="x86_64",
            cpu_name="AMD Ryzen 9 7945HX",
            python_version="3.12.0",
            gpus=[gpu],
        )

        assert profile.os_name == "Windows"
        assert profile.cpu_arch == "x86_64"
        assert profile.python_version == "3.12.0"
        assert len(profile.gpus) == 1
        assert profile.gpus[0].vendor == "nvidia"
        assert profile.gpus[0].cuda_version == "12.6"

    def test_create_cpu_only_profile(self):
        """A profile with no GPUs should be valid."""
        from compatibillabuddy.engine.models import HardwareProfile

        profile = HardwareProfile(
            os_name="Linux",
            os_version="6.1.0",
            cpu_arch="x86_64",
            cpu_name="Intel Xeon E5-2690",
            python_version="3.11.5",
        )

        assert profile.gpus == []

    def test_has_nvidia_gpu(self):
        """has_nvidia_gpu should return True only when an NVIDIA GPU is present."""
        from compatibillabuddy.engine.models import GpuInfo, HardwareProfile

        cpu_only = HardwareProfile(
            os_name="Linux",
            os_version="6.1.0",
            cpu_arch="x86_64",
            cpu_name="Intel Xeon",
            python_version="3.12.0",
        )
        assert cpu_only.has_nvidia_gpu is False

        with_nvidia = HardwareProfile(
            os_name="Linux",
            os_version="6.1.0",
            cpu_arch="x86_64",
            cpu_name="Intel Xeon",
            python_version="3.12.0",
            gpus=[
                GpuInfo(
                    vendor="nvidia",
                    name="RTX 4090",
                    driver_version="560.94",
                )
            ],
        )
        assert with_nvidia.has_nvidia_gpu is True

    def test_has_amd_gpu(self):
        """has_amd_gpu should return True only when an AMD GPU is present."""
        from compatibillabuddy.engine.models import GpuInfo, HardwareProfile

        with_amd = HardwareProfile(
            os_name="Linux",
            os_version="6.1.0",
            cpu_arch="x86_64",
            cpu_name="AMD EPYC",
            python_version="3.12.0",
            gpus=[
                GpuInfo(
                    vendor="amd",
                    name="Radeon RX 7900 XTX",
                    driver_version="6.3.6",
                    rocm_version="6.0",
                )
            ],
        )
        assert with_amd.has_amd_gpu is True
        assert with_amd.has_nvidia_gpu is False

    def test_serialization_roundtrip(self):
        """HardwareProfile should serialize to JSON and back without data loss."""
        from compatibillabuddy.engine.models import GpuInfo, HardwareProfile

        original = HardwareProfile(
            os_name="Linux",
            os_version="6.1.0",
            cpu_arch="x86_64",
            cpu_name="Intel Xeon",
            python_version="3.12.0",
            gpus=[
                GpuInfo(
                    vendor="nvidia",
                    name="A100",
                    driver_version="535.129",
                    cuda_version="12.2",
                    compute_capability="8.0",
                    vram_mb=81920,
                )
            ],
        )

        json_str = original.model_dump_json()
        restored = HardwareProfile.model_validate_json(json_str)
        assert restored == original

    def test_gpu_vendor_must_be_valid(self):
        """GPU vendor must be one of the known vendors."""
        from compatibillabuddy.engine.models import GpuInfo

        with pytest.raises(ValidationError):
            GpuInfo(vendor="intel_arc", name="Arc A770", driver_version="1.0")


class TestInstalledPackage:
    """Tests for InstalledPackage model."""

    def test_create_package(self):
        from compatibillabuddy.engine.models import InstalledPackage

        pkg = InstalledPackage(
            name="numpy",
            version="1.26.4",
        )
        assert pkg.name == "numpy"
        assert pkg.version == "1.26.4"

    def test_package_with_metadata(self):
        from compatibillabuddy.engine.models import InstalledPackage

        pkg = InstalledPackage(
            name="torch",
            version="2.4.0",
            requires=["numpy", "sympy", "networkx"],
            location="/home/user/.venv/lib/python3.12/site-packages",
            installer="pip",
        )
        assert "numpy" in pkg.requires
        assert pkg.installer == "pip"

    def test_requires_defaults_to_empty(self):
        from compatibillabuddy.engine.models import InstalledPackage

        pkg = InstalledPackage(name="six", version="1.16.0")
        assert pkg.requires == []


class TestEnvironmentInventory:
    """Tests for EnvironmentInventory model."""

    def test_create_inventory(self):
        from compatibillabuddy.engine.models import EnvironmentInventory, InstalledPackage

        inv = EnvironmentInventory(
            python_version="3.12.0",
            python_executable="/usr/bin/python3",
            packages=[
                InstalledPackage(name="numpy", version="1.26.4"),
                InstalledPackage(name="torch", version="2.4.0"),
            ],
        )
        assert inv.python_version == "3.12.0"
        assert len(inv.packages) == 2

    def test_get_package_by_name(self):
        """Should be able to look up a package by name (case-insensitive)."""
        from compatibillabuddy.engine.models import EnvironmentInventory, InstalledPackage

        inv = EnvironmentInventory(
            python_version="3.12.0",
            python_executable="/usr/bin/python3",
            packages=[
                InstalledPackage(name="numpy", version="1.26.4"),
                InstalledPackage(name="torch", version="2.4.0"),
            ],
        )
        assert inv.get_package("numpy").version == "1.26.4"
        assert inv.get_package("NumPy").version == "1.26.4"
        assert inv.get_package("nonexistent") is None

    def test_serialization_roundtrip(self):
        from compatibillabuddy.engine.models import EnvironmentInventory, InstalledPackage

        original = EnvironmentInventory(
            python_version="3.12.0",
            python_executable="/usr/bin/python3",
            packages=[
                InstalledPackage(name="numpy", version="1.26.4"),
            ],
        )
        json_str = original.model_dump_json()
        restored = EnvironmentInventory.model_validate_json(json_str)
        assert restored == original


class TestCompatIssue:
    """Tests for CompatIssue model."""

    def test_create_error_issue(self):
        from compatibillabuddy.engine.models import CompatIssue, Severity

        issue = CompatIssue(
            severity=Severity.ERROR,
            category="cuda_mismatch",
            description="PyTorch 2.4.0 requires CUDA >= 12.1, but driver supports max CUDA 11.8",
            affected_packages=["torch"],
            fix_suggestion="Install PyTorch with CUDA 11.8: pip install torch --index-url https://download.pytorch.org/whl/cu118",
        )
        assert issue.severity == Severity.ERROR
        assert "torch" in issue.affected_packages
        assert issue.fix_suggestion is not None

    def test_create_warning_issue(self):
        from compatibillabuddy.engine.models import CompatIssue, Severity

        issue = CompatIssue(
            severity=Severity.WARNING,
            category="numpy_abi",
            description="numpy 2.0.0 has ABI changes that may break pandas 1.5.3",
            affected_packages=["numpy", "pandas"],
        )
        assert issue.severity == Severity.WARNING
        assert issue.fix_suggestion is None

    def test_create_info_issue(self):
        from compatibillabuddy.engine.models import CompatIssue, Severity

        issue = CompatIssue(
            severity=Severity.INFO,
            category="deprecation",
            description="sklearn.externals.joblib is deprecated, use joblib directly",
            affected_packages=["scikit-learn"],
        )
        assert issue.severity == Severity.INFO

    def test_severity_ordering(self):
        """Errors should sort before warnings, which sort before info."""
        from compatibillabuddy.engine.models import Severity

        assert Severity.ERROR.value < Severity.WARNING.value < Severity.INFO.value

    def test_serialization_roundtrip(self):
        from compatibillabuddy.engine.models import CompatIssue, Severity

        original = CompatIssue(
            severity=Severity.ERROR,
            category="cuda_mismatch",
            description="CUDA version mismatch",
            affected_packages=["torch"],
            fix_suggestion="Downgrade torch",
        )
        json_str = original.model_dump_json()
        restored = CompatIssue.model_validate_json(json_str)
        assert restored == original

    def test_category_must_not_be_empty(self):
        from compatibillabuddy.engine.models import CompatIssue, Severity

        with pytest.raises(ValidationError):
            CompatIssue(
                severity=Severity.ERROR,
                category="",
                description="Something broke",
                affected_packages=["torch"],
            )


class TestDiagnosisResult:
    """Tests for DiagnosisResult model."""

    def test_create_clean_result(self):
        """A diagnosis with no issues should be valid."""
        from compatibillabuddy.engine.models import (
            DiagnosisResult,
            EnvironmentInventory,
            HardwareProfile,
        )

        hw = HardwareProfile(
            os_name="Linux",
            os_version="6.1.0",
            cpu_arch="x86_64",
            cpu_name="Intel Xeon",
            python_version="3.12.0",
        )
        env = EnvironmentInventory(
            python_version="3.12.0",
            python_executable="/usr/bin/python3",
        )
        result = DiagnosisResult(hardware=hw, environment=env)

        assert result.issue_count == 0
        assert result.has_errors is False
        assert result.has_warnings is False

    def test_create_result_with_issues(self):
        """A diagnosis with mixed severity issues."""
        from compatibillabuddy.engine.models import (
            CompatIssue,
            DiagnosisResult,
            EnvironmentInventory,
            HardwareProfile,
            Severity,
        )

        hw = HardwareProfile(
            os_name="Linux",
            os_version="6.1.0",
            cpu_arch="x86_64",
            cpu_name="Intel Xeon",
            python_version="3.12.0",
        )
        env = EnvironmentInventory(
            python_version="3.12.0",
            python_executable="/usr/bin/python3",
        )
        issues = [
            CompatIssue(
                severity=Severity.ERROR, category="cuda_mismatch", description="CUDA too old"
            ),
            CompatIssue(
                severity=Severity.WARNING,
                category="coinstall",
                description="Torch and TF co-installed",
            ),
            CompatIssue(
                severity=Severity.INFO, category="deprecation", description="API deprecated"
            ),
        ]
        result = DiagnosisResult(hardware=hw, environment=env, issues=issues)

        assert result.issue_count == 3
        assert result.has_errors is True
        assert result.has_warnings is True

    def test_has_errors_false_when_only_warnings(self):
        """has_errors should be False if there are only warnings and info."""
        from compatibillabuddy.engine.models import (
            CompatIssue,
            DiagnosisResult,
            EnvironmentInventory,
            HardwareProfile,
            Severity,
        )

        hw = HardwareProfile(
            os_name="Linux",
            os_version="6.1.0",
            cpu_arch="x86_64",
            cpu_name="Intel Xeon",
            python_version="3.12.0",
        )
        env = EnvironmentInventory(
            python_version="3.12.0",
            python_executable="/usr/bin/python3",
        )
        result = DiagnosisResult(
            hardware=hw,
            environment=env,
            issues=[
                CompatIssue(
                    severity=Severity.WARNING,
                    category="coinstall",
                    description="Potential conflict",
                ),
                CompatIssue(severity=Severity.INFO, category="deprecation", description="Old API"),
            ],
        )

        assert result.has_errors is False
        assert result.has_warnings is True

    def test_has_warnings_false_when_only_errors(self):
        """has_warnings should be False if there are only errors."""
        from compatibillabuddy.engine.models import (
            CompatIssue,
            DiagnosisResult,
            EnvironmentInventory,
            HardwareProfile,
            Severity,
        )

        hw = HardwareProfile(
            os_name="Linux",
            os_version="6.1.0",
            cpu_arch="x86_64",
            cpu_name="Intel Xeon",
            python_version="3.12.0",
        )
        env = EnvironmentInventory(
            python_version="3.12.0",
            python_executable="/usr/bin/python3",
        )
        result = DiagnosisResult(
            hardware=hw,
            environment=env,
            issues=[
                CompatIssue(
                    severity=Severity.ERROR, category="cuda_mismatch", description="CUDA broken"
                ),
            ],
        )

        assert result.has_errors is True
        assert result.has_warnings is False

    def test_timing_defaults_to_zero(self):
        """All timing fields should default to 0.0."""
        from compatibillabuddy.engine.models import (
            DiagnosisResult,
            EnvironmentInventory,
            HardwareProfile,
        )

        hw = HardwareProfile(
            os_name="Linux",
            os_version="6.1.0",
            cpu_arch="x86_64",
            cpu_name="Intel Xeon",
            python_version="3.12.0",
        )
        env = EnvironmentInventory(
            python_version="3.12.0",
            python_executable="/usr/bin/python3",
        )
        result = DiagnosisResult(hardware=hw, environment=env)

        assert result.hardware_probe_seconds == 0.0
        assert result.environment_inspect_seconds == 0.0
        assert result.rule_evaluation_seconds == 0.0
        assert result.total_seconds == 0.0

    def test_timing_stores_values(self):
        """Timing fields should store provided values."""
        from compatibillabuddy.engine.models import (
            DiagnosisResult,
            EnvironmentInventory,
            HardwareProfile,
        )

        hw = HardwareProfile(
            os_name="Linux",
            os_version="6.1.0",
            cpu_arch="x86_64",
            cpu_name="Intel Xeon",
            python_version="3.12.0",
        )
        env = EnvironmentInventory(
            python_version="3.12.0",
            python_executable="/usr/bin/python3",
        )
        result = DiagnosisResult(
            hardware=hw,
            environment=env,
            hardware_probe_seconds=1.23,
            environment_inspect_seconds=4.56,
            rule_evaluation_seconds=0.01,
            total_seconds=5.80,
        )

        assert result.hardware_probe_seconds == 1.23
        assert result.environment_inspect_seconds == 4.56
        assert result.rule_evaluation_seconds == 0.01
        assert result.total_seconds == 5.80

    def test_serialization_roundtrip(self):
        """DiagnosisResult should serialize to JSON and back without data loss."""
        from compatibillabuddy.engine.models import (
            CompatIssue,
            DiagnosisResult,
            EnvironmentInventory,
            HardwareProfile,
            InstalledPackage,
            Severity,
        )

        hw = HardwareProfile(
            os_name="Linux",
            os_version="6.1.0",
            cpu_arch="x86_64",
            cpu_name="Intel Xeon",
            python_version="3.12.0",
        )
        env = EnvironmentInventory(
            python_version="3.12.0",
            python_executable="/usr/bin/python3",
            packages=[InstalledPackage(name="torch", version="2.4.0")],
        )
        original = DiagnosisResult(
            hardware=hw,
            environment=env,
            issues=[
                CompatIssue(
                    severity=Severity.ERROR,
                    category="cuda_mismatch",
                    description="CUDA too old",
                    affected_packages=["torch"],
                    fix_suggestion="Upgrade CUDA",
                ),
            ],
            hardware_probe_seconds=1.0,
            environment_inspect_seconds=2.0,
            rule_evaluation_seconds=0.5,
            total_seconds=3.5,
        )

        json_str = original.model_dump_json()
        restored = DiagnosisResult.model_validate_json(json_str)
        assert restored == original

    def test_issues_defaults_to_empty(self):
        """Issues list should default to empty."""
        from compatibillabuddy.engine.models import (
            DiagnosisResult,
            EnvironmentInventory,
            HardwareProfile,
        )

        hw = HardwareProfile(
            os_name="Linux",
            os_version="6.1.0",
            cpu_arch="x86_64",
            cpu_name="Intel Xeon",
            python_version="3.12.0",
        )
        env = EnvironmentInventory(
            python_version="3.12.0",
            python_executable="/usr/bin/python3",
        )
        result = DiagnosisResult(hardware=hw, environment=env)
        assert result.issues == []
