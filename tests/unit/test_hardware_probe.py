"""Tests for hardware probe module — written BEFORE implementation (TDD).

Most tests mock system calls so they run on any machine.
One integration test (marked slow) runs the real probe.
"""

import json
import sys
from pathlib import Path
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
    return "NVIDIA A100-SXM4-80GB, 535.129.03, 81920\nNVIDIA A100-SXM4-80GB, 535.129.03, 81920\n"


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

        with (
            patch("subprocess.run", side_effect=mock_run),
            patch("compatibillabuddy.hardware.probe._detect_cudnn_version", return_value=None),
        ):
            gpus = _detect_nvidia_gpus()

        assert len(gpus) == 1
        assert gpus[0].vendor == GpuVendor.NVIDIA
        assert gpus[0].name == "NVIDIA GeForce RTX 4090"
        assert gpus[0].driver_version == "560.94"
        assert gpus[0].vram_mb == 24564
        assert gpus[0].cuda_version == "12.6"
        assert gpus[0].cudnn_version is None

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

        with (
            patch("subprocess.run", side_effect=mock_run),
            patch("compatibillabuddy.hardware.probe._detect_cudnn_version", return_value=None),
        ):
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
# cuDNN detection tests (mocked)
# ---------------------------------------------------------------------------


# Realistic cudnn_version.h content (cuDNN 9.3.0)
CUDNN_9_HEADER = """\
#ifndef CUDNN_VERSION_H_
#define CUDNN_VERSION_H_

#define CUDNN_MAJOR 9
#define CUDNN_MINOR 3
#define CUDNN_PATCHLEVEL 0

#define CUDNN_VERSION (CUDNN_MAJOR * 10000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)

#endif /* CUDNN_VERSION_H_ */
"""

# Realistic cudnn_version.h content (cuDNN 8.9.7)
CUDNN_8_HEADER = """\
#ifndef CUDNN_VERSION_H_
#define CUDNN_VERSION_H_

#define CUDNN_MAJOR 8
#define CUDNN_MINOR 9
#define CUDNN_PATCHLEVEL 7

#define CUDNN_VERSION (CUDNN_MAJOR * 10000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)

#endif /* CUDNN_VERSION_H_ */
"""


class TestCudnnVersionParsing:
    """Test parsing of cudnn_version.h header content."""

    def test_parse_cudnn_9(self):
        """Should parse cuDNN 9.3.0 header correctly."""
        from compatibillabuddy.hardware.probe import _parse_cudnn_version_header

        version = _parse_cudnn_version_header(CUDNN_9_HEADER)
        assert version == "9.3.0"

    def test_parse_cudnn_8(self):
        """Should parse cuDNN 8.9.7 header correctly."""
        from compatibillabuddy.hardware.probe import _parse_cudnn_version_header

        version = _parse_cudnn_version_header(CUDNN_8_HEADER)
        assert version == "8.9.7"

    def test_parse_returns_none_for_garbage(self):
        """Should return None for non-header content."""
        from compatibillabuddy.hardware.probe import _parse_cudnn_version_header

        assert _parse_cudnn_version_header("not a header") is None

    def test_parse_returns_none_for_empty(self):
        """Should return None for empty string."""
        from compatibillabuddy.hardware.probe import _parse_cudnn_version_header

        assert _parse_cudnn_version_header("") is None

    def test_parse_returns_none_for_partial_defines(self):
        """Should return None if only some defines are present."""
        from compatibillabuddy.hardware.probe import _parse_cudnn_version_header

        partial = "#define CUDNN_MAJOR 9\n#define CUDNN_MINOR 3\n"
        assert _parse_cudnn_version_header(partial) is None


class TestCudnnDetection:
    """Test cuDNN version detection via file system lookup."""

    def test_detect_from_cuda_path_env(self, tmp_path):
        """Should find cudnn_version.h under CUDA_PATH environment variable."""
        from compatibillabuddy.hardware.probe import _detect_cudnn_version

        include_dir = tmp_path / "include"
        include_dir.mkdir()
        header = include_dir / "cudnn_version.h"
        header.write_text(CUDNN_9_HEADER)

        with patch.dict("os.environ", {"CUDA_PATH": str(tmp_path)}):
            version = _detect_cudnn_version()

        assert version == "9.3.0"

    def test_detect_returns_none_when_no_header_found(self, tmp_path):
        """Should return None when no cudnn_version.h exists anywhere."""
        from compatibillabuddy.hardware.probe import _detect_cudnn_version

        # Point CUDA_PATH to an empty directory and mock Path.read_text
        # to raise FileNotFoundError for the Linux fallback paths.
        empty_dir = tmp_path / "no_cuda"
        empty_dir.mkdir()

        original_read_text = Path.read_text

        def guarded_read_text(self, *args, **kwargs):
            # Block the Linux fallback paths so the test is deterministic
            blocked = ("/usr/include/cudnn_version.h", "/usr/local/cuda/include/cudnn_version.h")
            if str(self) in blocked:
                raise FileNotFoundError(f"Mocked: {self}")
            return original_read_text(self, *args, **kwargs)

        with (
            patch.dict("os.environ", {"CUDA_PATH": str(empty_dir)}, clear=True),
            patch.object(Path, "read_text", guarded_read_text),
        ):
            version = _detect_cudnn_version()

        assert version is None

    def test_detect_returns_none_when_header_is_garbage(self, tmp_path):
        """Should return None when cudnn_version.h exists but has bad content."""
        from compatibillabuddy.hardware.probe import _detect_cudnn_version

        include_dir = tmp_path / "include"
        include_dir.mkdir()
        header = include_dir / "cudnn_version.h"
        header.write_text("this is not a real header")

        with patch.dict("os.environ", {"CUDA_PATH": str(tmp_path)}):
            version = _detect_cudnn_version()

        assert version is None


class TestNvidiaGpusCudnnIntegration:
    """Test that _detect_nvidia_gpus populates cudnn_version on GpuInfo."""

    def test_gpus_have_cudnn_version_when_detected(
        self, nvidia_smi_csv_single, nvidia_smi_cuda_version, tmp_path
    ):
        """GpuInfo should have cudnn_version set when cuDNN is found."""
        from compatibillabuddy.hardware.probe import _detect_nvidia_gpus

        include_dir = tmp_path / "include"
        include_dir.mkdir()
        header = include_dir / "cudnn_version.h"
        header.write_text(CUDNN_9_HEADER)

        def mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
            if "--query-gpu" in cmd_str:
                result.stdout = nvidia_smi_csv_single
            else:
                result.stdout = nvidia_smi_cuda_version
            return result

        with (
            patch("subprocess.run", side_effect=mock_run),
            patch.dict("os.environ", {"CUDA_PATH": str(tmp_path)}),
        ):
            gpus = _detect_nvidia_gpus()

        assert len(gpus) == 1
        assert gpus[0].cudnn_version == "9.3.0"

    def test_gpus_have_cudnn_none_when_not_detected(
        self, nvidia_smi_csv_single, nvidia_smi_cuda_version, tmp_path
    ):
        """GpuInfo should have cudnn_version=None when cuDNN is not found."""
        from compatibillabuddy.hardware.probe import _detect_nvidia_gpus

        # Point CUDA_PATH to an empty directory and block Linux fallback paths
        empty_dir = tmp_path / "no_cuda"
        empty_dir.mkdir()

        original_read_text = Path.read_text

        def guarded_read_text(self, *args, **kwargs):
            blocked = ("/usr/include/cudnn_version.h", "/usr/local/cuda/include/cudnn_version.h")
            if str(self) in blocked:
                raise FileNotFoundError(f"Mocked: {self}")
            return original_read_text(self, *args, **kwargs)

        def mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
            if "--query-gpu" in cmd_str:
                result.stdout = nvidia_smi_csv_single
            else:
                result.stdout = nvidia_smi_cuda_version
            return result

        with (
            patch("subprocess.run", side_effect=mock_run),
            patch.dict("os.environ", {"CUDA_PATH": str(empty_dir)}, clear=True),
            patch.object(Path, "read_text", guarded_read_text),
        ):
            gpus = _detect_nvidia_gpus()

        assert len(gpus) == 1
        assert gpus[0].cudnn_version is None


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

        with (
            patch("subprocess.run", side_effect=mock_run),
            patch("compatibillabuddy.hardware.probe._detect_cudnn_version", return_value=None),
        ):
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

        with (
            patch("subprocess.run", side_effect=mock_run),
            patch("compatibillabuddy.hardware.probe._detect_cudnn_version", return_value=None),
        ):
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
