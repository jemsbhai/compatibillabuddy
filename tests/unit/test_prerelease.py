"""Focused tests for pre-release version handling in rule matching.

This file exists because PEP 440 pre-release semantics create a real gap
in our rule engine: a user with torch==2.4.0rc1 on CUDA 11.8 should be
warned, but PEP 440 says 2.4.0rc1 does NOT satisfy >=2.4.

We must handle this explicitly, since ML users frequently install
pre-release / nightly builds (e.g. torch nightly, tf-nightly).
"""

import textwrap

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from compatibillabuddy.engine.models import (
    EnvironmentInventory,
    GpuInfo,
    GpuVendor,
    HardwareProfile,
    InstalledPackage,
    Severity,
)


# ---------------------------------------------------------------------------
# First: prove the PEP 440 behavior that creates the gap
# ---------------------------------------------------------------------------


class TestPEP440PreReleaseBehavior:
    """Document actual PEP 440 behavior so we understand what we're working with."""

    def test_rc_sorts_before_release(self):
        """2.4.0rc1 is strictly LESS THAN 2.4.0 in PEP 440 ordering."""
        assert Version("2.4.0rc1") < Version("2.4.0")
        assert Version("2.4.0rc1") < Version("2.4")

    def test_rc_not_in_gte_specifier(self):
        """PEP 440 >=2.4 EXCLUDES 2.4.0rc1 — even with prereleases=True.

        This is the root cause of our gap. The packaging library considers
        2.4.0rc1 as "not yet 2.4", so >=2.4 doesn't match it.
        """
        spec = SpecifierSet(">=2.4")
        v = Version("2.4.0rc1")
        # This is the actual behavior — >=2.4 does NOT include 2.4.0rc1
        assert spec.contains(v, prereleases=True) is False

    def test_rc_not_in_lt_specifier_either(self):
        """PEP 440 <2.0 also EXCLUDES 2.0.0rc1.

        This seems counterintuitive (2.0.0rc1 < 2.0.0 in sort order),
        but the packaging library treats pre-releases of the boundary
        version specially.
        """
        spec = SpecifierSet("<2.0")
        v = Version("2.0.0rc1")
        assert spec.contains(v, prereleases=True) is False

    def test_rc_IS_in_gte_when_specifier_includes_pre(self):
        """>=2.4.0rc1 DOES match 2.4.0rc1."""
        spec = SpecifierSet(">=2.4.0rc1")
        v = Version("2.4.0rc1")
        assert spec.contains(v, prereleases=True) is True

    def test_dev_and_nightly_versions(self):
        """torch nightly versions like 2.5.0.dev20240101 also won't match >=2.5."""
        spec = SpecifierSet(">=2.5")
        v = Version("2.5.0.dev20240101")
        assert spec.contains(v, prereleases=True) is False


# ---------------------------------------------------------------------------
# Now: test that OUR engine handles this correctly
# ---------------------------------------------------------------------------


PRERELEASE_RULEPACK = textwrap.dedent("""\
    [meta]
    name = "test-prerelease"
    version = "1.0.0"

    [[rules]]
    id = "torch-cuda-needs-12.1"
    severity = "error"
    category = "cuda_mismatch"
    description = "PyTorch {torch_version} requires CUDA >= 12.1, but system has CUDA {cuda_version}"
    fix = "Install compatible PyTorch"

    [rules.when]
    package_version = {torch = ">=2.4"}
    gpu_vendor = "nvidia"
    cuda_version = "<12.1"

    [[rules]]
    id = "numpy2-abi"
    severity = "error"
    category = "numpy_abi"
    description = "numpy {numpy_version} ABI break vs pandas {pandas_version}"
    fix = "Upgrade pandas"

    [rules.when]
    package_version = {numpy = ">=2.0", pandas = "<2.1"}
""")


def _make_env(*packages):
    return EnvironmentInventory(
        python_version="3.12.0",
        python_executable="/usr/bin/python3",
        packages=[InstalledPackage(name=n, version=v) for n, v in packages],
    )


@pytest.fixture
def hw_cuda_old():
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
            )
        ],
    )


class TestPreReleaseRuleMatching:
    """Verify our rule engine correctly handles pre-release package versions."""

    def test_torch_rc_fires_cuda_rule(self, hw_cuda_old):
        """torch 2.4.0rc1 on CUDA 11.8 MUST trigger the warning.

        A release candidate has the same CUDA requirements as the release.
        """
        from compatibillabuddy.kb.engine import evaluate_rules, load_rules_from_toml

        env = _make_env(("torch", "2.4.0rc1"))
        rules = load_rules_from_toml(PRERELEASE_RULEPACK)
        issues = evaluate_rules(rules, env, hw_cuda_old)

        cuda_issues = [i for i in issues if i.category == "cuda_mismatch"]
        assert len(cuda_issues) == 1, (
            "CRITICAL: torch 2.4.0rc1 on CUDA 11.8 was not flagged! "
            "Pre-release versions must trigger the same rules as their base release."
        )

    def test_torch_dev_fires_cuda_rule(self, hw_cuda_old):
        """torch 2.4.0.dev20240101 (nightly) on CUDA 11.8 MUST trigger the warning."""
        from compatibillabuddy.kb.engine import evaluate_rules, load_rules_from_toml

        env = _make_env(("torch", "2.4.0.dev20240101"))
        rules = load_rules_from_toml(PRERELEASE_RULEPACK)
        issues = evaluate_rules(rules, env, hw_cuda_old)

        cuda_issues = [i for i in issues if i.category == "cuda_mismatch"]
        assert len(cuda_issues) == 1, (
            "CRITICAL: torch nightly 2.4.0.dev on CUDA 11.8 was not flagged!"
        )

    def test_torch_alpha_fires_cuda_rule(self, hw_cuda_old):
        """torch 2.4.0a1 on CUDA 11.8 MUST trigger the warning."""
        from compatibillabuddy.kb.engine import evaluate_rules, load_rules_from_toml

        env = _make_env(("torch", "2.4.0a1"))
        rules = load_rules_from_toml(PRERELEASE_RULEPACK)
        issues = evaluate_rules(rules, env, hw_cuda_old)

        cuda_issues = [i for i in issues if i.category == "cuda_mismatch"]
        assert len(cuda_issues) == 1

    def test_torch_beta_fires_cuda_rule(self, hw_cuda_old):
        """torch 2.4.0b2 on CUDA 11.8 MUST trigger the warning."""
        from compatibillabuddy.kb.engine import evaluate_rules, load_rules_from_toml

        env = _make_env(("torch", "2.4.0b2"))
        rules = load_rules_from_toml(PRERELEASE_RULEPACK)
        issues = evaluate_rules(rules, env, hw_cuda_old)

        cuda_issues = [i for i in issues if i.category == "cuda_mismatch"]
        assert len(cuda_issues) == 1

    def test_numpy_rc_fires_abi_rule(self, hw_cuda_old):
        """numpy 2.0.0rc2 + old pandas should trigger ABI rule."""
        from compatibillabuddy.kb.engine import evaluate_rules, load_rules_from_toml

        env = _make_env(("numpy", "2.0.0rc2"), ("pandas", "1.5.3"))
        rules = load_rules_from_toml(PRERELEASE_RULEPACK)
        issues = evaluate_rules(rules, env, hw_cuda_old)

        abi_issues = [i for i in issues if i.category == "numpy_abi"]
        assert len(abi_issues) == 1

    def test_stable_release_still_works(self, hw_cuda_old):
        """Stable versions must still match correctly (no regression)."""
        from compatibillabuddy.kb.engine import evaluate_rules, load_rules_from_toml

        env = _make_env(("torch", "2.4.0"))
        rules = load_rules_from_toml(PRERELEASE_RULEPACK)
        issues = evaluate_rules(rules, env, hw_cuda_old)

        cuda_issues = [i for i in issues if i.category == "cuda_mismatch"]
        assert len(cuda_issues) == 1

    def test_old_stable_does_not_false_positive(self, hw_cuda_old):
        """torch 2.3.1 should NOT trigger the >=2.4 rule, even with prerelease handling."""
        from compatibillabuddy.kb.engine import evaluate_rules, load_rules_from_toml

        env = _make_env(("torch", "2.3.1"))
        rules = load_rules_from_toml(PRERELEASE_RULEPACK)
        issues = evaluate_rules(rules, env, hw_cuda_old)

        cuda_issues = [i for i in issues if i.category == "cuda_mismatch"]
        assert len(cuda_issues) == 0

    def test_old_prerelease_does_not_false_positive(self, hw_cuda_old):
        """torch 2.3.0rc1 should NOT trigger the >=2.4 rule."""
        from compatibillabuddy.kb.engine import evaluate_rules, load_rules_from_toml

        env = _make_env(("torch", "2.3.0rc1"))
        rules = load_rules_from_toml(PRERELEASE_RULEPACK)
        issues = evaluate_rules(rules, env, hw_cuda_old)

        cuda_issues = [i for i in issues if i.category == "cuda_mismatch"]
        assert len(cuda_issues) == 0


class TestVersionMatchesWithPreRelease:
    """Direct unit tests for _version_matches with pre-release awareness."""

    def test_rc_matches_gte_base(self):
        """2.4.0rc1 should match >=2.4 in our engine."""
        from compatibillabuddy.kb.engine import _version_matches

        assert _version_matches("2.4.0rc1", ">=2.4") is True

    def test_dev_matches_gte_base(self):
        """2.4.0.dev1 should match >=2.4 in our engine."""
        from compatibillabuddy.kb.engine import _version_matches

        assert _version_matches("2.4.0.dev20240101", ">=2.4") is True

    def test_alpha_matches_gte_base(self):
        from compatibillabuddy.kb.engine import _version_matches

        assert _version_matches("2.4.0a1", ">=2.4") is True

    def test_beta_matches_gte_base(self):
        from compatibillabuddy.kb.engine import _version_matches

        assert _version_matches("2.4.0b3", ">=2.4") is True

    def test_old_rc_does_not_match_gte_newer(self):
        """2.3.0rc1 should NOT match >=2.4."""
        from compatibillabuddy.kb.engine import _version_matches

        assert _version_matches("2.3.0rc1", ">=2.4") is False

    def test_rc_matches_lt_higher(self):
        """2.0.0rc1 should match <2.1 (it's definitely less than 2.1)."""
        from compatibillabuddy.kb.engine import _version_matches

        assert _version_matches("2.0.0rc1", "<2.1") is True

    def test_stable_versions_unchanged(self):
        """Stable versions should work exactly as before."""
        from compatibillabuddy.kb.engine import _version_matches

        assert _version_matches("2.4.0", ">=2.4") is True
        assert _version_matches("2.3.9", ">=2.4") is False
        assert _version_matches("11.8", "<12.1") is True
        assert _version_matches("12.6", "<12.1") is False

    def test_post_release_unchanged(self):
        from compatibillabuddy.kb.engine import _version_matches

        assert _version_matches("2.0.0.post1", ">=2.0") is True
