"""Tests for hardware probe module — written BEFORE implementation (TDD).

Most tests mock system calls so they run on any machine.
One integration test (marked slow) runs the real probe.
"""

import json
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from compatibillabuddy.engine.models import GpuVendor, HardwareProfile


# ---------------------------------------------------------------------------
# Fixtures: canned system responses
# ---------------------------------------------------------------------------


@pytest.fixture
def nvidia_smi_csv_single():
    """nvidia-smi output for a single RTX 4090."""
    return "NVIDIA GeForce RTX 4090, 560.94, 24564\n"


@pytest.fixture
def nvidia_smi_csv_multi():
    """nvidia-smi output for dual A100s."""
    return (
        "NVIDIA A100-SXM4-80GB, 535.129.03, 81920\n"
        "NVIDIA A100-SXM4-80GB, 535.129.03, 81920\n"
    )


@pytest.fixture
def nvidia_smi_cuda_version():
    """nvidia-smi output showing CUDA version."""
    return (
        "Wed Feb  5 10:30:00 2025\n"
        "+-------------------------+\n"
        "| NVIDIA-SMI 560.94       Driver Version: 560.94       CUDA Version: 12.6 |\n"
        "+-------------------------+\n"
    )


# ---------------------------------------------------------------------------
# OS / CPU detection tests
# ---------------------------------------------------------------------------


class TestOsCpuDetection:
    """Test OS and CPU field population."""

    def test_os_name_populated(self):
        from compatibillabuddy.hardware.probe import probe_hardware

        profile = probe_hardware()
        assert profile.os_name in ("Windows", "Linux", "Darwin")

    def test_os_version_populated(self):
        from compatibillabuddy.hardware.probe import probe_hardware

        profile = probe_hardware()
        assert len(profile.os_version) > 0

    def test_cpu_arch_populated(self):
        from compatibillabuddy.hardware.probe import probe_hardware

        profile = probe_hardware()
        assert profile.cpu_arch in ("x86_64", "AMD64", "aarch64", "arm64", "x86")

    def test_cpu_name_populated(self):
        from compatibillabuddy.hardware.probe import probe_hardware

        profile = probe_hardware()
        assert len(profile.cpu_name) > 0

    def test_python_version_matches_runtime(self):
        from compatibillabuddy.hardware.probe import probe_hardware

        profile = probe_hardware()
        expected = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        assert profile.python_version == expected


# ---------------------------------------------------------------------------
# NVIDIA GPU detection tests (mocked)
# ---------------------------------------------------------------------------


class TestNvidiaDetection:
    """Test NVIDIA GPU detection via mocked nvidia-smi calls."""

    def test_single_nvidia_gpu(self, nvidia_smi_csv_single, nvidia_smi_cuda_version):
        """Should detect a single NVIDIA GPU with correct fields."""
        from compatibillabuddy.hardware.probe import _detect_nvidia_gpus

        def mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
            if "--query-gpu" in cmd_str:
                result.stdout = nvidia_smi_csv_single
            else:
                result.stdout = nvidia_smi_cuda_version
            return result

        with patch("subprocess.run", side_effect=mock_run):
            gpus = _detect_nvidia_gpus()

        assert len(gpus) == 1
        assert gpus[0].vendor == GpuVendor.NVIDIA
        assert gpus[0].name == "NVIDIA GeForce RTX 4090"
        assert gpus[0].driver_version == "560.94"
        assert gpus[0].vram_mb == 24564
        assert gpus[0].cuda_version == "12.6"

    def test_multiple_nvidia_gpus(self, nvidia_smi_csv_multi, nvidia_smi_cuda_version):
        """Should detect multiple GPUs."""
        from compatibillabuddy.hardware.probe import _detect_nvidia_gpus

        def mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
            if "--query-gpu" in cmd_str:
                result.stdout = nvidia_smi_csv_multi
            else:
                result.stdout = nvidia_smi_cuda_version
            return result

        with patch("subprocess.run", side_effect=mock_run):
            gpus = _detect_nvidia_gpus()

        assert len(gpus) == 2
        assert all(g.vendor == GpuVendor.NVIDIA for g in gpus)
        assert all(g.vram_mb == 81920 for g in gpus)

    def test_nvidia_smi_not_found(self):
        """Should return empty list when nvidia-smi is not available."""
        from compatibillabuddy.hardware.probe import _detect_nvidia_gpus

        with patch("subprocess.run", side_effect=FileNotFoundError):
            gpus = _detect_nvidia_gpus()

        assert gpus == []

    def test_nvidia_smi_fails(self):
        """Should return empty list when nvidia-smi exits non-zero."""
        from compatibillabuddy.hardware.probe import _detect_nvidia_gpus

        result = MagicMock()
        result.returncode = 1
        result.stdout = ""

        with patch("subprocess.run", return_value=result):
            gpus = _detect_nvidia_gpus()

        assert gpus == []

    def test_cuda_version_parsed_from_nvidia_smi(self, nvidia_smi_cuda_version):
        """Should extract CUDA version from nvidia-smi header output."""
        from compatibillabuddy.hardware.probe import _parse_cuda_version

        cuda_ver = _parse_cuda_version(nvidia_smi_cuda_version)
        assert cuda_ver == "12.6"

    def test_cuda_version_none_when_missing(self):
        """Should return None when CUDA version can't be parsed."""
        from compatibillabuddy.hardware.probe import _parse_cuda_version

        cuda_ver = _parse_cuda_version("some garbage output")
        assert cuda_ver is None


# ---------------------------------------------------------------------------
# Full probe tests (mocked)
# ---------------------------------------------------------------------------


class TestProbeHardwareMocked:
    """Test the full probe_hardware() function with mocked system calls."""

    def test_probe_with_nvidia_gpu(self, nvidia_smi_csv_single, nvidia_smi_cuda_version):
        """Full probe on a system with an NVIDIA GPU."""
        from compatibillabuddy.hardware.probe import probe_hardware

        def mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
            if "--query-gpu" in cmd_str:
                result.stdout = nvidia_smi_csv_single
            else:
                result.stdout = nvidia_smi_cuda_version
            return result

        with patch("subprocess.run", side_effect=mock_run):
            profile = probe_hardware()

        assert isinstance(profile, HardwareProfile)
        assert profile.has_nvidia_gpu is True
        assert len(profile.gpus) == 1

    def test_probe_cpu_only(self):
        """Full probe on a CPU-only system (nvidia-smi not found)."""
        from compatibillabuddy.hardware.probe import probe_hardware

        with patch("subprocess.run", side_effect=FileNotFoundError):
            profile = probe_hardware()

        assert isinstance(profile, HardwareProfile)
        assert profile.has_nvidia_gpu is False
        assert profile.gpus == []
        # OS/CPU fields should still be populated
        assert len(profile.os_name) > 0
        assert len(profile.cpu_arch) > 0

    def test_probe_returns_valid_json(self, nvidia_smi_csv_single, nvidia_smi_cuda_version):
        """The profile should serialize to valid JSON."""
        from compatibillabuddy.hardware.probe import probe_hardware

        def mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
            if "--query-gpu" in cmd_str:
                result.stdout = nvidia_smi_csv_single
            else:
                result.stdout = nvidia_smi_cuda_version
            return result

        with patch("subprocess.run", side_effect=mock_run):
            profile = probe_hardware()

        json_str = profile.model_dump_json()
        parsed = json.loads(json_str)
        assert "os_name" in parsed
        assert "gpus" in parsed


# ---------------------------------------------------------------------------
# Integration test: runs real probe on THIS machine
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestProbeHardwareReal:
    """Run the real hardware probe. Only meaningful on the dev machine."""

    def test_real_probe_returns_profile(self):
        """The probe should return a valid HardwareProfile on any machine."""
        from compatibillabuddy.hardware.probe import probe_hardware

        profile = probe_hardware()

        assert isinstance(profile, HardwareProfile)
        assert profile.os_name in ("Windows", "Linux", "Darwin")
        assert len(profile.python_version) > 0

    def test_real_probe_serializes(self):
        """Real probe output should serialize cleanly."""
        from compatibillabuddy.hardware.probe import probe_hardware

        profile = probe_hardware()
        json_str = profile.model_dump_json(indent=2)
        restored = HardwareProfile.model_validate_json(json_str)
        assert restored.os_name == profile.os_name
