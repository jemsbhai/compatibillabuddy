"""Tests for the JAX rulepack (jax.toml).

Each test constructs a synthetic HardwareProfile + EnvironmentInventory
and evaluates the bundled rules to verify the JAX rulepack fires (or
does not fire) under the expected conditions.

Rule sources:
- jax-0.4.31-needs-cuda-12.1: JAX changelog — "JAX now supports CUDA 12.1
  or newer only. Support for CUDA 11.8 has been dropped." (v0.4.31)
- jax-old-needs-cuda-11.8: JAX pre-0.4.31 required CUDA >=11.8 for GPU
- jax-numpy-too-old: JAX changelog — "The minimum NumPy version is now 1.24"
  (applies to JAX >=0.4.31)
- jax-torch-cuda-conflict: Well-documented real-world issue — JAX and PyTorch
  both ship their own CUDA runtime, causing potential library conflicts
"""

from __future__ import annotations

from compatibillabuddy.engine.models import (
    CompatIssue,
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


def _find_issues(issues: list[CompatIssue], rule_id_prefix: str) -> list[CompatIssue]:
    """Find issues whose description or category matches a rule pattern."""
    return [i for i in issues if rule_id_prefix in (i.category or "")]


def _get_jax_rules():
    """Load all bundled rules (includes jax.toml once it exists)."""
    return load_bundled_rulepacks()


# ---------------------------------------------------------------------------
# jax-0.4.31-needs-cuda-12.1
# ---------------------------------------------------------------------------


class TestJaxCuda121Required:
    """JAX >= 0.4.31 requires CUDA >= 12.1 on NVIDIA GPUs."""

    def test_fires_when_cuda_too_old(self):
        """JAX 0.5.0 + CUDA 11.8 → should fire ERROR."""
        rules = _get_jax_rules()
        hw = _nvidia_hw(cuda_version="11.8")
        env = _env(("jax", "0.5.0"), ("jaxlib", "0.5.0"))
        issues = evaluate_rules(rules, env, hw)

        cuda_issues = [
            i for i in issues if i.category == "cuda_mismatch" and "jax" in i.description.lower()
        ]
        assert len(cuda_issues) >= 1
        assert cuda_issues[0].severity == Severity.ERROR

    def test_fires_for_0_4_31(self):
        """JAX 0.4.31 exactly + CUDA 12.0 → should fire (12.0 < 12.1)."""
        rules = _get_jax_rules()
        hw = _nvidia_hw(cuda_version="12.0")
        env = _env(("jax", "0.4.31"), ("jaxlib", "0.4.31"))
        issues = evaluate_rules(rules, env, hw)

        cuda_issues = [
            i for i in issues if i.category == "cuda_mismatch" and "jax" in i.description.lower()
        ]
        assert len(cuda_issues) >= 1

    def test_does_not_fire_when_cuda_sufficient(self):
        """JAX 0.5.0 + CUDA 12.4 → should NOT fire."""
        rules = _get_jax_rules()
        hw = _nvidia_hw(cuda_version="12.4")
        env = _env(("jax", "0.5.0"), ("jaxlib", "0.5.0"))
        issues = evaluate_rules(rules, env, hw)

        cuda_issues = [
            i for i in issues if i.category == "cuda_mismatch" and "jax" in i.description.lower()
        ]
        assert len(cuda_issues) == 0

    def test_does_not_fire_on_cpu_only(self):
        """JAX 0.5.0 on CPU-only machine → should NOT fire."""
        rules = _get_jax_rules()
        hw = _cpu_only_hw()
        env = _env(("jax", "0.5.0"), ("jaxlib", "0.5.0"))
        issues = evaluate_rules(rules, env, hw)

        cuda_issues = [
            i for i in issues if i.category == "cuda_mismatch" and "jax" in i.description.lower()
        ]
        assert len(cuda_issues) == 0

    def test_does_not_fire_when_jax_not_installed(self):
        """No JAX installed → should NOT fire."""
        rules = _get_jax_rules()
        hw = _nvidia_hw(cuda_version="11.8")
        env = _env(("numpy", "1.26.0"))
        issues = evaluate_rules(rules, env, hw)

        cuda_issues = [
            i for i in issues if i.category == "cuda_mismatch" and "jax" in i.description.lower()
        ]
        assert len(cuda_issues) == 0


# ---------------------------------------------------------------------------
# jax-old-needs-cuda-11.8
# ---------------------------------------------------------------------------


class TestJaxOldCuda118Required:
    """JAX >= 0.4.1, < 0.4.31 requires CUDA >= 11.8 on NVIDIA GPUs."""

    def test_fires_when_cuda_too_old(self):
        """JAX 0.4.20 + CUDA 11.7 → should fire ERROR."""
        rules = _get_jax_rules()
        hw = _nvidia_hw(cuda_version="11.7")
        env = _env(("jax", "0.4.20"), ("jaxlib", "0.4.20"))
        issues = evaluate_rules(rules, env, hw)

        cuda_issues = [
            i for i in issues if i.category == "cuda_mismatch" and "jax" in i.description.lower()
        ]
        assert len(cuda_issues) >= 1
        assert cuda_issues[0].severity == Severity.ERROR

    def test_does_not_fire_when_cuda_118(self):
        """JAX 0.4.20 + CUDA 11.8 → should NOT fire."""
        rules = _get_jax_rules()
        hw = _nvidia_hw(cuda_version="11.8")
        env = _env(("jax", "0.4.20"), ("jaxlib", "0.4.20"))
        issues = evaluate_rules(rules, env, hw)

        cuda_issues = [
            i for i in issues if i.category == "cuda_mismatch" and "jax" in i.description.lower()
        ]
        assert len(cuda_issues) == 0

    def test_does_not_fire_for_new_jax(self):
        """JAX 0.4.31 + CUDA 11.7 → should NOT fire this rule (different rule applies)."""
        rules = _get_jax_rules()
        hw = _nvidia_hw(cuda_version="11.7")
        env = _env(("jax", "0.4.31"), ("jaxlib", "0.4.31"))
        issues = evaluate_rules(rules, env, hw)

        # The newer rule (jax-0.4.31-needs-cuda-12.1) should fire instead,
        # but this older rule (which says "requires CUDA >= 11.8") should NOT fire
        old_rule_issues = [
            i
            for i in issues
            if i.category == "cuda_mismatch"
            and "jax" in i.description.lower()
            and "requires CUDA >= 11.8" in i.description
        ]
        assert len(old_rule_issues) == 0


# ---------------------------------------------------------------------------
# jax-numpy-too-old
# ---------------------------------------------------------------------------


class TestJaxNumpyCompat:
    """JAX >= 0.4.31 requires NumPy >= 1.24."""

    def test_fires_when_numpy_too_old(self):
        """JAX 0.5.0 + NumPy 1.23.5 → should fire ERROR."""
        rules = _get_jax_rules()
        hw = _cpu_only_hw()
        env = _env(("jax", "0.5.0"), ("jaxlib", "0.5.0"), ("numpy", "1.23.5"))
        issues = evaluate_rules(rules, env, hw)

        numpy_issues = [
            i for i in issues if i.category == "numpy_compat" and "jax" in i.description.lower()
        ]
        assert len(numpy_issues) >= 1
        assert numpy_issues[0].severity == Severity.ERROR

    def test_does_not_fire_when_numpy_sufficient(self):
        """JAX 0.5.0 + NumPy 1.26.0 → should NOT fire."""
        rules = _get_jax_rules()
        hw = _cpu_only_hw()
        env = _env(("jax", "0.5.0"), ("jaxlib", "0.5.0"), ("numpy", "1.26.0"))
        issues = evaluate_rules(rules, env, hw)

        numpy_issues = [
            i for i in issues if i.category == "numpy_compat" and "jax" in i.description.lower()
        ]
        assert len(numpy_issues) == 0

    def test_does_not_fire_for_older_jax(self):
        """JAX 0.4.20 + NumPy 1.22.0 → should NOT fire (older JAX, looser req)."""
        rules = _get_jax_rules()
        hw = _cpu_only_hw()
        env = _env(("jax", "0.4.20"), ("jaxlib", "0.4.20"), ("numpy", "1.22.0"))
        issues = evaluate_rules(rules, env, hw)

        numpy_issues = [
            i for i in issues if i.category == "numpy_compat" and "jax" in i.description.lower()
        ]
        assert len(numpy_issues) == 0

    def test_does_not_fire_without_numpy(self):
        """JAX 0.5.0 without NumPy → should NOT fire (numpy not installed)."""
        rules = _get_jax_rules()
        hw = _cpu_only_hw()
        env = _env(("jax", "0.5.0"), ("jaxlib", "0.5.0"))
        issues = evaluate_rules(rules, env, hw)

        numpy_issues = [
            i for i in issues if i.category == "numpy_compat" and "jax" in i.description.lower()
        ]
        assert len(numpy_issues) == 0


# ---------------------------------------------------------------------------
# jax-torch-cuda-conflict
# ---------------------------------------------------------------------------


class TestJaxTorchCudaConflict:
    """JAX + PyTorch on NVIDIA GPU → potential CUDA runtime conflicts."""

    def test_fires_when_both_installed_on_nvidia(self):
        """JAX + PyTorch on NVIDIA → should fire WARNING."""
        rules = _get_jax_rules()
        hw = _nvidia_hw(cuda_version="12.4")
        env = _env(("jax", "0.4.35"), ("jaxlib", "0.4.35"), ("torch", "2.4.0"))
        issues = evaluate_rules(rules, env, hw)

        coinstall_issues = [
            i
            for i in issues
            if i.category == "coinstall_conflict"
            and "jax" in i.description.lower()
            and "torch" in i.description.lower()
        ]
        assert len(coinstall_issues) >= 1
        assert coinstall_issues[0].severity == Severity.WARNING

    def test_does_not_fire_on_cpu_only(self):
        """JAX + PyTorch on CPU-only → should NOT fire (no CUDA conflict)."""
        rules = _get_jax_rules()
        hw = _cpu_only_hw()
        env = _env(("jax", "0.4.35"), ("jaxlib", "0.4.35"), ("torch", "2.4.0"))
        issues = evaluate_rules(rules, env, hw)

        coinstall_issues = [
            i
            for i in issues
            if i.category == "coinstall_conflict"
            and "jax" in i.description.lower()
            and "torch" in i.description.lower()
        ]
        assert len(coinstall_issues) == 0

    def test_does_not_fire_without_torch(self):
        """JAX alone on NVIDIA → should NOT fire."""
        rules = _get_jax_rules()
        hw = _nvidia_hw(cuda_version="12.4")
        env = _env(("jax", "0.4.35"), ("jaxlib", "0.4.35"))
        issues = evaluate_rules(rules, env, hw)

        coinstall_issues = [
            i
            for i in issues
            if i.category == "coinstall_conflict"
            and "jax" in i.description.lower()
            and "torch" in i.description.lower()
        ]
        assert len(coinstall_issues) == 0

    def test_does_not_fire_without_jax(self):
        """PyTorch alone on NVIDIA → should NOT fire from jax rules."""
        rules = _get_jax_rules()
        hw = _nvidia_hw(cuda_version="12.4")
        env = _env(("torch", "2.4.0"))
        issues = evaluate_rules(rules, env, hw)

        # There might be other rules from ml_core.toml, but not the jax-torch one
        coinstall_issues = [
            i
            for i in issues
            if i.category == "coinstall_conflict"
            and "jax" in i.description.lower()
            and "torch" in i.description.lower()
        ]
        assert len(coinstall_issues) == 0
