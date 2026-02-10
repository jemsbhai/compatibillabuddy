"""Tests for the ONNX Runtime rulepack (onnxruntime.toml).

Each test constructs a synthetic HardwareProfile + EnvironmentInventory
and evaluates the bundled rules to verify the ONNX Runtime rulepack fires
(or does not fire) under the expected conditions.

Rule sources:
- ort-gpu-1.19-needs-cuda-12: ONNX Runtime CUDA EP docs —
  "Starting with version 1.19, CUDA 12.x becomes the default version
  when distributing ONNX Runtime GPU packages in PyPI."
  https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- ort-cpu-gpu-coinstall: Well-documented issue — onnxruntime and
  onnxruntime-gpu both provide the same 'onnxruntime' Python module,
  causing import conflicts and undefined behavior.
"""

from __future__ import annotations

from compatibillabuddy.engine.models import (
    EnvironmentInventory,
    GpuInfo,
    GpuVendor,
    HardwareProfile,
    InstalledPackage,
    Severity,
)
from compatibillabuddy.kb.engine import evaluate_rules, load_bundled_rulepacks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nvidia_hw(cuda_version: str = "12.4") -> HardwareProfile:
    """Build a HardwareProfile with a single NVIDIA GPU."""
    return HardwareProfile(
        os_name="Linux",
        os_version="6.5.0",
        cpu_arch="x86_64",
        cpu_name="AMD Ryzen 9",
        python_version="3.12.0",
        gpus=[
            GpuInfo(
                vendor=GpuVendor.NVIDIA,
                name="RTX 4090",
                driver_version="550.54",
                cuda_version=cuda_version,
                vram_mb=24576,
            )
        ],
    )


def _cpu_only_hw() -> HardwareProfile:
    """Build a HardwareProfile with no GPU."""
    return HardwareProfile(
        os_name="Linux",
        os_version="6.5.0",
        cpu_arch="x86_64",
        cpu_name="Intel Core i9",
        python_version="3.12.0",
        gpus=[],
    )


def _env(*packages: tuple[str, str]) -> EnvironmentInventory:
    """Build an EnvironmentInventory from (name, version) tuples."""
    return EnvironmentInventory(
        python_version="3.12.0",
        python_executable="/usr/bin/python3",
        packages=[InstalledPackage(name=name, version=version) for name, version in packages],
    )


def _get_rules():
    """Load all bundled rules (includes onnxruntime.toml once it exists)."""
    return load_bundled_rulepacks()


# ---------------------------------------------------------------------------
# ort-gpu-1.19-needs-cuda-12
# ---------------------------------------------------------------------------


class TestOrtGpuCuda12Required:
    """onnxruntime-gpu >= 1.19 requires CUDA 12.x on NVIDIA GPUs."""

    def test_fires_when_cuda_11(self):
        """onnxruntime-gpu 1.20.0 + CUDA 11.8 → should fire ERROR."""
        rules = _get_rules()
        hw = _nvidia_hw(cuda_version="11.8")
        env = _env(("onnxruntime-gpu", "1.20.0"))
        issues = evaluate_rules(rules, env, hw)

        ort_issues = [
            i
            for i in issues
            if i.category == "cuda_mismatch" and "onnxruntime" in i.description.lower()
        ]
        assert len(ort_issues) >= 1
        assert ort_issues[0].severity == Severity.ERROR

    def test_fires_for_1_19_exactly(self):
        """onnxruntime-gpu 1.19.0 + CUDA 11.8 → should fire."""
        rules = _get_rules()
        hw = _nvidia_hw(cuda_version="11.8")
        env = _env(("onnxruntime-gpu", "1.19.0"))
        issues = evaluate_rules(rules, env, hw)

        ort_issues = [
            i
            for i in issues
            if i.category == "cuda_mismatch" and "onnxruntime" in i.description.lower()
        ]
        assert len(ort_issues) >= 1

    def test_does_not_fire_when_cuda_12(self):
        """onnxruntime-gpu 1.20.0 + CUDA 12.4 → should NOT fire."""
        rules = _get_rules()
        hw = _nvidia_hw(cuda_version="12.4")
        env = _env(("onnxruntime-gpu", "1.20.0"))
        issues = evaluate_rules(rules, env, hw)

        ort_issues = [
            i
            for i in issues
            if i.category == "cuda_mismatch" and "onnxruntime" in i.description.lower()
        ]
        assert len(ort_issues) == 0

    def test_does_not_fire_for_older_ort(self):
        """onnxruntime-gpu 1.18.0 + CUDA 11.8 → should NOT fire this rule."""
        rules = _get_rules()
        hw = _nvidia_hw(cuda_version="11.8")
        env = _env(("onnxruntime-gpu", "1.18.0"))
        issues = evaluate_rules(rules, env, hw)

        ort_issues = [
            i
            for i in issues
            if i.category == "cuda_mismatch"
            and "onnxruntime" in i.description.lower()
            and "CUDA 12" in i.description
        ]
        assert len(ort_issues) == 0

    def test_does_not_fire_on_cpu_only(self):
        """onnxruntime-gpu 1.20.0 on CPU-only → should NOT fire."""
        rules = _get_rules()
        hw = _cpu_only_hw()
        env = _env(("onnxruntime-gpu", "1.20.0"))
        issues = evaluate_rules(rules, env, hw)

        ort_issues = [
            i
            for i in issues
            if i.category == "cuda_mismatch" and "onnxruntime" in i.description.lower()
        ]
        assert len(ort_issues) == 0

    def test_does_not_fire_when_not_installed(self):
        """No onnxruntime-gpu → should NOT fire."""
        rules = _get_rules()
        hw = _nvidia_hw(cuda_version="11.8")
        env = _env(("numpy", "1.26.0"))
        issues = evaluate_rules(rules, env, hw)

        ort_issues = [
            i
            for i in issues
            if i.category == "cuda_mismatch" and "onnxruntime" in i.description.lower()
        ]
        assert len(ort_issues) == 0


# ---------------------------------------------------------------------------
# ort-cpu-gpu-coinstall
# ---------------------------------------------------------------------------


class TestOrtCpuGpuCoinstall:
    """onnxruntime and onnxruntime-gpu both installed → conflict."""

    def test_fires_when_both_installed(self):
        """onnxruntime + onnxruntime-gpu → should fire ERROR."""
        rules = _get_rules()
        hw = _nvidia_hw(cuda_version="12.4")
        env = _env(("onnxruntime", "1.20.0"), ("onnxruntime-gpu", "1.20.0"))
        issues = evaluate_rules(rules, env, hw)

        coinstall_issues = [
            i
            for i in issues
            if i.category == "coinstall_conflict" and "onnxruntime" in i.description.lower()
        ]
        assert len(coinstall_issues) >= 1
        assert coinstall_issues[0].severity == Severity.ERROR

    def test_fires_on_cpu_machine_too(self):
        """Both installed on CPU-only machine → should still fire."""
        rules = _get_rules()
        hw = _cpu_only_hw()
        env = _env(("onnxruntime", "1.20.0"), ("onnxruntime-gpu", "1.20.0"))
        issues = evaluate_rules(rules, env, hw)

        coinstall_issues = [
            i
            for i in issues
            if i.category == "coinstall_conflict" and "onnxruntime" in i.description.lower()
        ]
        assert len(coinstall_issues) >= 1

    def test_does_not_fire_with_only_cpu_package(self):
        """Only onnxruntime (CPU) installed → should NOT fire."""
        rules = _get_rules()
        hw = _cpu_only_hw()
        env = _env(("onnxruntime", "1.20.0"))
        issues = evaluate_rules(rules, env, hw)

        coinstall_issues = [
            i
            for i in issues
            if i.category == "coinstall_conflict" and "onnxruntime" in i.description.lower()
        ]
        assert len(coinstall_issues) == 0

    def test_does_not_fire_with_only_gpu_package(self):
        """Only onnxruntime-gpu installed → should NOT fire."""
        rules = _get_rules()
        hw = _nvidia_hw(cuda_version="12.4")
        env = _env(("onnxruntime-gpu", "1.20.0"))
        issues = evaluate_rules(rules, env, hw)

        coinstall_issues = [
            i
            for i in issues
            if i.category == "coinstall_conflict" and "onnxruntime" in i.description.lower()
        ]
        assert len(coinstall_issues) == 0


# ---------------------------------------------------------------------------
# ort-gpu-1.19-needs-cudnn-9
# ---------------------------------------------------------------------------


def _nvidia_hw_cudnn(
    cuda_version: str = "12.4", cudnn_version: str | None = "9.3.0"
) -> HardwareProfile:
    """Build a HardwareProfile with NVIDIA GPU and cuDNN version."""
    return HardwareProfile(
        os_name="Linux",
        os_version="6.5.0",
        cpu_arch="x86_64",
        cpu_name="AMD Ryzen 9",
        python_version="3.12.0",
        gpus=[
            GpuInfo(
                vendor=GpuVendor.NVIDIA,
                name="RTX 4090",
                driver_version="550.54",
                cuda_version=cuda_version,
                cudnn_version=cudnn_version,
                vram_mb=24576,
            )
        ],
    )


class TestOrtGpuCudnn9Required:
    """onnxruntime-gpu >= 1.19 (default PyPI package) requires cuDNN 9.x.

    Source: ORT CUDA EP docs — "ONNX Runtime built with cuDNN 8.x is not
    compatible with cuDNN 9.x, and vice versa." The default PyPI package
    since 1.19 is built against cuDNN 9.
    https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
    """

    def test_fires_when_cudnn_8(self):
        """onnxruntime-gpu 1.20.0 + cuDNN 8.9.7 → should fire ERROR."""
        rules = _get_rules()
        hw = _nvidia_hw_cudnn(cuda_version="12.4", cudnn_version="8.9.7")
        env = _env(("onnxruntime-gpu", "1.20.0"))
        issues = evaluate_rules(rules, env, hw)

        cudnn_issues = [
            i
            for i in issues
            if i.category == "cudnn_mismatch" and "onnxruntime" in i.description.lower()
        ]
        assert len(cudnn_issues) >= 1
        assert cudnn_issues[0].severity == Severity.ERROR

    def test_fires_for_1_19_exactly(self):
        """onnxruntime-gpu 1.19.0 + cuDNN 8.9.7 → should fire."""
        rules = _get_rules()
        hw = _nvidia_hw_cudnn(cuda_version="12.4", cudnn_version="8.9.7")
        env = _env(("onnxruntime-gpu", "1.19.0"))
        issues = evaluate_rules(rules, env, hw)

        cudnn_issues = [
            i
            for i in issues
            if i.category == "cudnn_mismatch" and "onnxruntime" in i.description.lower()
        ]
        assert len(cudnn_issues) >= 1

    def test_does_not_fire_when_cudnn_9(self):
        """onnxruntime-gpu 1.20.0 + cuDNN 9.3.0 → should NOT fire."""
        rules = _get_rules()
        hw = _nvidia_hw_cudnn(cuda_version="12.4", cudnn_version="9.3.0")
        env = _env(("onnxruntime-gpu", "1.20.0"))
        issues = evaluate_rules(rules, env, hw)

        cudnn_issues = [
            i
            for i in issues
            if i.category == "cudnn_mismatch" and "onnxruntime" in i.description.lower()
        ]
        assert len(cudnn_issues) == 0

    def test_does_not_fire_for_older_ort(self):
        """onnxruntime-gpu 1.18.0 + cuDNN 8.9.7 → should NOT fire this rule."""
        rules = _get_rules()
        hw = _nvidia_hw_cudnn(cuda_version="12.4", cudnn_version="8.9.7")
        env = _env(("onnxruntime-gpu", "1.18.0"))
        issues = evaluate_rules(rules, env, hw)

        cudnn_issues = [
            i
            for i in issues
            if i.category == "cudnn_mismatch" and "onnxruntime" in i.description.lower()
        ]
        assert len(cudnn_issues) == 0

    def test_does_not_fire_when_cudnn_not_detected(self):
        """onnxruntime-gpu 1.20.0 + cuDNN=None → should NOT fire."""
        rules = _get_rules()
        hw = _nvidia_hw_cudnn(cuda_version="12.4", cudnn_version=None)
        env = _env(("onnxruntime-gpu", "1.20.0"))
        issues = evaluate_rules(rules, env, hw)

        cudnn_issues = [
            i
            for i in issues
            if i.category == "cudnn_mismatch" and "onnxruntime" in i.description.lower()
        ]
        assert len(cudnn_issues) == 0

    def test_does_not_fire_on_cpu_only(self):
        """onnxruntime-gpu 1.20.0 on CPU-only → should NOT fire."""
        rules = _get_rules()
        hw = _cpu_only_hw()
        env = _env(("onnxruntime-gpu", "1.20.0"))
        issues = evaluate_rules(rules, env, hw)

        cudnn_issues = [
            i
            for i in issues
            if i.category == "cudnn_mismatch" and "onnxruntime" in i.description.lower()
        ]
        assert len(cudnn_issues) == 0
