"""Hardware probe: detect CPU, OS, GPU, and driver info without heavy dependencies.

Uses only stdlib + subprocess calls to nvidia-smi (no Python GPU libraries needed).
This avoids the bootstrap problem — we can diagnose an environment without importing
the packages we're trying to fix.
"""

from __future__ import annotations

import platform
import re
import subprocess
import sys
from typing import Optional

from compatibillabuddy.engine.models import GpuInfo, GpuVendor, HardwareProfile


def probe_hardware() -> HardwareProfile:
    """Probe the current machine and return a complete HardwareProfile.

    Safe to call on any machine — returns a CPU-only profile if no GPU is found.
    """
    gpus: list[GpuInfo] = []
    gpus.extend(_detect_nvidia_gpus())
    # Future: gpus.extend(_detect_amd_gpus())
    # Future: gpus.extend(_detect_apple_gpus())

    return HardwareProfile(
        os_name=_detect_os_name(),
        os_version=_detect_os_version(),
        cpu_arch=_detect_cpu_arch(),
        cpu_name=_detect_cpu_name(),
        python_version=_detect_python_version(),
        gpus=gpus,
    )


# ---------------------------------------------------------------------------
# OS / CPU helpers
# ---------------------------------------------------------------------------


def _detect_os_name() -> str:
    """Return normalized OS name: 'Windows', 'Linux', or 'Darwin'."""
    return platform.system()


def _detect_os_version() -> str:
    """Return OS version string."""
    return platform.version()


def _detect_cpu_arch() -> str:
    """Return CPU architecture (e.g. 'x86_64', 'AMD64', 'aarch64')."""
    return platform.machine()


def _detect_cpu_name() -> str:
    """Return CPU model name. Best-effort — falls back to platform.processor()."""
    name = platform.processor()
    if name:
        return name

    # On Linux, platform.processor() can be empty — try /proc/cpuinfo
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.strip().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (FileNotFoundError, PermissionError):
        pass

    return platform.machine()


def _detect_python_version() -> str:
    """Return Python version as 'MAJOR.MINOR.MICRO'."""
    v = sys.version_info
    return f"{v.major}.{v.minor}.{v.micro}"


# ---------------------------------------------------------------------------
# NVIDIA GPU detection
# ---------------------------------------------------------------------------


def _detect_nvidia_gpus() -> list[GpuInfo]:
    """Detect NVIDIA GPUs by calling nvidia-smi.

    Returns an empty list if nvidia-smi is not available or fails.
    """
    csv_output = _run_nvidia_smi_query()
    if csv_output is None:
        return []

    cuda_version = _get_cuda_version_from_nvidia_smi()

    gpus = []
    for line in csv_output.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue

        name, driver_version, vram_str = parts[0], parts[1], parts[2]

        try:
            vram_mb = int(vram_str)
        except ValueError:
            vram_mb = None

        gpus.append(
            GpuInfo(
                vendor=GpuVendor.NVIDIA,
                name=name,
                driver_version=driver_version,
                cuda_version=cuda_version,
                vram_mb=vram_mb,
            )
        )

    return gpus


def _run_nvidia_smi_query() -> Optional[str]:
    """Run nvidia-smi --query-gpu and return CSV output, or None on failure."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _get_cuda_version_from_nvidia_smi() -> Optional[str]:
    """Get CUDA version from nvidia-smi header output."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        return _parse_cuda_version(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _parse_cuda_version(nvidia_smi_output: str) -> Optional[str]:
    """Extract CUDA version from nvidia-smi header output.

    Looks for 'CUDA Version: XX.Y' in the output.
    """
    match = re.search(r"CUDA Version:\s*(\d+\.\d+)", nvidia_smi_output)
    if match:
        return match.group(1)
    return None
