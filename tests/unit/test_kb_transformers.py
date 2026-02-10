"""Tests for the Transformers rulepack (transformers.toml).

Each test constructs a synthetic HardwareProfile + EnvironmentInventory
and evaluates the bundled rules to verify the Transformers rulepack fires
(or does not fire) under the expected conditions.

Rule sources:
- transformers-pytorch-too-old: HF Transformers GitHub issue #39780 shows
  setup.py declares torch>=2.1 in the [torch] extra.
  https://github.com/huggingface/transformers/issues/39780
- transformers-pytorch-untested: HF Transformers README states
  "Transformers works with Python 3.9+, and PyTorch 2.4+."
  https://github.com/huggingface/transformers
  Installation docs: "tested on Python 3.9+ and PyTorch 2.2+"
  https://huggingface.co/docs/transformers/en/installation
"""

from __future__ import annotations

from compatibillabuddy.engine.models import (
    EnvironmentInventory,
    HardwareProfile,
    InstalledPackage,
    Severity,
)
from compatibillabuddy.kb.engine import evaluate_rules, load_bundled_rulepacks


def _get_rules():
    return load_bundled_rulepacks()


def _cpu_only_hw() -> HardwareProfile:
    return HardwareProfile(
        os_name="Linux",
        os_version="6.5.0",
        cpu_arch="x86_64",
        cpu_name="AMD Ryzen 9",
        python_version="3.12.0",
        gpus=[],
    )


def _env(*packages: tuple[str, str]) -> EnvironmentInventory:
    return EnvironmentInventory(
        python_version="3.12.0",
        python_executable="/usr/bin/python3",
        packages=[InstalledPackage(name=n, version=v) for n, v in packages],
    )


# ---------------------------------------------------------------------------
# transformers-pytorch-too-old
# ---------------------------------------------------------------------------


class TestTransformersPytorchTooOld:
    """transformers >= 4.48 + torch < 2.1 → ERROR.

    The [torch] extra in setup.py declares torch>=2.1 as a hard minimum.
    """

    def test_fires_when_torch_too_old(self):
        """transformers 4.48.0 + torch 2.0.1 → should fire ERROR."""
        rules = _get_rules()
        hw = _cpu_only_hw()
        env = _env(("transformers", "4.48.0"), ("torch", "2.0.1"))
        issues = evaluate_rules(rules, env, hw)

        dep_issues = [
            i
            for i in issues
            if i.category == "dependency_version"
            and "transformers" in i.description.lower()
            and "pytorch" in i.description.lower()
        ]
        assert len(dep_issues) >= 1
        assert dep_issues[0].severity == Severity.ERROR

    def test_fires_for_very_old_torch(self):
        """transformers 4.50.0 + torch 1.13.0 → should fire ERROR."""
        rules = _get_rules()
        hw = _cpu_only_hw()
        env = _env(("transformers", "4.50.0"), ("torch", "1.13.0"))
        issues = evaluate_rules(rules, env, hw)

        dep_issues = [
            i
            for i in issues
            if i.category == "dependency_version" and "transformers" in i.description.lower()
        ]
        assert len(dep_issues) >= 1
        assert dep_issues[0].severity == Severity.ERROR

    def test_does_not_fire_when_torch_sufficient(self):
        """transformers 4.48.0 + torch 2.1.0 → should NOT fire ERROR."""
        rules = _get_rules()
        hw = _cpu_only_hw()
        env = _env(("transformers", "4.48.0"), ("torch", "2.1.0"))
        issues = evaluate_rules(rules, env, hw)

        error_issues = [
            i
            for i in issues
            if i.category == "dependency_version"
            and i.severity == Severity.ERROR
            and "transformers" in i.description.lower()
        ]
        assert len(error_issues) == 0

    def test_does_not_fire_for_older_transformers(self):
        """transformers 4.40.0 + torch 2.0.1 → should NOT fire."""
        rules = _get_rules()
        hw = _cpu_only_hw()
        env = _env(("transformers", "4.40.0"), ("torch", "2.0.1"))
        issues = evaluate_rules(rules, env, hw)

        dep_issues = [
            i
            for i in issues
            if i.category == "dependency_version"
            and "transformers" in i.description.lower()
            and i.severity == Severity.ERROR
        ]
        assert len(dep_issues) == 0

    def test_does_not_fire_without_torch(self):
        """transformers 4.50.0 without torch → should NOT fire."""
        rules = _get_rules()
        hw = _cpu_only_hw()
        env = _env(("transformers", "4.50.0"))
        issues = evaluate_rules(rules, env, hw)

        dep_issues = [
            i
            for i in issues
            if i.category == "dependency_version" and "transformers" in i.description.lower()
        ]
        assert len(dep_issues) == 0


# ---------------------------------------------------------------------------
# transformers-pytorch-untested
# ---------------------------------------------------------------------------


class TestTransformersPytorchUntested:
    """transformers >= 4.48 + torch >= 2.1, < 2.4 → WARNING.

    PyTorch 2.1-2.3 may work but is below the tested/recommended range.
    The README states "PyTorch 2.4+" and install docs say "tested on PyTorch 2.2+".
    """

    def test_fires_warning_for_torch_2_2(self):
        """transformers 4.48.0 + torch 2.2.0 → should fire WARNING."""
        rules = _get_rules()
        hw = _cpu_only_hw()
        env = _env(("transformers", "4.48.0"), ("torch", "2.2.0"))
        issues = evaluate_rules(rules, env, hw)

        warn_issues = [
            i
            for i in issues
            if i.category == "dependency_version"
            and i.severity == Severity.WARNING
            and "transformers" in i.description.lower()
        ]
        assert len(warn_issues) >= 1

    def test_fires_warning_for_torch_2_3(self):
        """transformers 4.50.0 + torch 2.3.1 → should fire WARNING."""
        rules = _get_rules()
        hw = _cpu_only_hw()
        env = _env(("transformers", "4.50.0"), ("torch", "2.3.1"))
        issues = evaluate_rules(rules, env, hw)

        warn_issues = [
            i
            for i in issues
            if i.category == "dependency_version"
            and i.severity == Severity.WARNING
            and "transformers" in i.description.lower()
        ]
        assert len(warn_issues) >= 1

    def test_does_not_fire_for_torch_2_4(self):
        """transformers 4.48.0 + torch 2.4.0 → should NOT fire warning."""
        rules = _get_rules()
        hw = _cpu_only_hw()
        env = _env(("transformers", "4.48.0"), ("torch", "2.4.0"))
        issues = evaluate_rules(rules, env, hw)

        warn_issues = [
            i
            for i in issues
            if i.category == "dependency_version"
            and i.severity == Severity.WARNING
            and "transformers" in i.description.lower()
        ]
        assert len(warn_issues) == 0

    def test_does_not_fire_for_torch_2_5(self):
        """transformers 4.50.0 + torch 2.5.0 → should NOT fire warning."""
        rules = _get_rules()
        hw = _cpu_only_hw()
        env = _env(("transformers", "4.50.0"), ("torch", "2.5.0"))
        issues = evaluate_rules(rules, env, hw)

        warn_issues = [
            i
            for i in issues
            if i.category == "dependency_version"
            and i.severity == Severity.WARNING
            and "transformers" in i.description.lower()
        ]
        assert len(warn_issues) == 0

    def test_does_not_fire_for_older_transformers(self):
        """transformers 4.40.0 + torch 2.2.0 → should NOT fire."""
        rules = _get_rules()
        hw = _cpu_only_hw()
        env = _env(("transformers", "4.40.0"), ("torch", "2.2.0"))
        issues = evaluate_rules(rules, env, hw)

        warn_issues = [
            i
            for i in issues
            if i.category == "dependency_version"
            and i.severity == Severity.WARNING
            and "transformers" in i.description.lower()
        ]
        assert len(warn_issues) == 0
