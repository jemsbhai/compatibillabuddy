"""Tests for knowledge base and rulepack engine — written BEFORE implementation (TDD).

Comprehensive test suite covering:
- TOML rule loading and validation
- Version specifier matching (PEP 440)
- Hardware condition matching
- Template variable substitution
- End-to-end rule evaluation
- Every bundled rulepack rule with realistic scenarios
- Error handling and edge cases
"""

import textwrap

import pytest
from packaging.version import Version

from compatibillabuddy.engine.models import (
    CompatIssue,
    EnvironmentInventory,
    GpuInfo,
    GpuVendor,
    HardwareProfile,
    InstalledPackage,
    Severity,
)


# ===========================================================================
# Fixtures: hardware profiles
# ===========================================================================


@pytest.fixture
def hw_nvidia_old_driver():
    """NVIDIA GPU with old driver — max CUDA 11.8."""
    return HardwareProfile(
        os_name="Linux",
        os_version="6.1.0",
        cpu_arch="x86_64",
        cpu_name="Intel Xeon E5-2690",
        python_version="3.12.0",
        gpus=[
            GpuInfo(
                vendor=GpuVendor.NVIDIA,
                name="RTX 3090",
                driver_version="520.61",
                cuda_version="11.8",
                compute_capability="8.6",
                vram_mb=24576,
            )
        ],
    )


@pytest.fixture
def hw_nvidia_mid_driver():
    """NVIDIA GPU with mid-range driver — CUDA 12.1."""
    return HardwareProfile(
        os_name="Linux",
        os_version="6.1.0",
        cpu_arch="x86_64",
        cpu_name="Intel Xeon",
        python_version="3.12.0",
        gpus=[
            GpuInfo(
                vendor=GpuVendor.NVIDIA,
                name="A100-SXM4-80GB",
                driver_version="535.129.03",
                cuda_version="12.1",
                compute_capability="8.0",
                vram_mb=81920,
            )
        ],
    )


@pytest.fixture
def hw_nvidia_new_driver():
    """NVIDIA GPU with latest driver — CUDA 12.6."""
    return HardwareProfile(
        os_name="Windows",
        os_version="10.0.22631",
        cpu_arch="AMD64",
        cpu_name="AMD Ryzen 9 7945HX",
        python_version="3.12.0",
        gpus=[
            GpuInfo(
                vendor=GpuVendor.NVIDIA,
                name="RTX 4090",
                driver_version="560.94",
                cuda_version="12.6",
                compute_capability="8.9",
                vram_mb=24564,
            )
        ],
    )


@pytest.fixture
def hw_nvidia_dual_gpu():
    """Machine with two NVIDIA GPUs (different CUDA should not happen, but tests edge)."""
    return HardwareProfile(
        os_name="Linux",
        os_version="6.1.0",
        cpu_arch="x86_64",
        cpu_name="Intel Xeon",
        python_version="3.12.0",
        gpus=[
            GpuInfo(
                vendor=GpuVendor.NVIDIA,
                name="A100-SXM4-80GB",
                driver_version="535.129.03",
                cuda_version="12.1",
                vram_mb=81920,
            ),
            GpuInfo(
                vendor=GpuVendor.NVIDIA,
                name="A100-SXM4-80GB",
                driver_version="535.129.03",
                cuda_version="12.1",
                vram_mb=81920,
            ),
        ],
    )


@pytest.fixture
def hw_amd_gpu():
    """AMD GPU with ROCm."""
    return HardwareProfile(
        os_name="Linux",
        os_version="6.1.0",
        cpu_arch="x86_64",
        cpu_name="AMD EPYC 7763",
        python_version="3.12.0",
        gpus=[
            GpuInfo(
                vendor=GpuVendor.AMD,
                name="Radeon Instinct MI250X",
                driver_version="6.3.6",
                rocm_version="6.0",
            )
        ],
    )


@pytest.fixture
def hw_apple_silicon():
    """Apple Silicon Mac."""
    return HardwareProfile(
        os_name="Darwin",
        os_version="23.4.0",
        cpu_arch="arm64",
        cpu_name="Apple M3 Max",
        python_version="3.12.0",
        gpus=[
            GpuInfo(
                vendor=GpuVendor.APPLE,
                name="Apple M3 Max",
                driver_version="macOS 14.4",
            )
        ],
    )


@pytest.fixture
def hw_cpu_only():
    """CPU-only machine, no GPU at all."""
    return HardwareProfile(
        os_name="Linux",
        os_version="6.1.0",
        cpu_arch="x86_64",
        cpu_name="Intel Xeon E5-2690",
        python_version="3.12.0",
    )


# ===========================================================================
# Fixtures: environment inventories
# ===========================================================================


def _make_env(*packages: tuple[str, str]) -> EnvironmentInventory:
    """Helper to create an EnvironmentInventory from (name, version) tuples."""
    return EnvironmentInventory(
        python_version="3.12.0",
        python_executable="/usr/bin/python3",
        packages=[InstalledPackage(name=n, version=v) for n, v in packages],
    )


@pytest.fixture
def env_torch24():
    return _make_env(("torch", "2.4.0"), ("numpy", "1.26.4"))


@pytest.fixture
def env_torch22():
    return _make_env(("torch", "2.2.0"), ("numpy", "1.26.4"))


@pytest.fixture
def env_torch20():
    return _make_env(("torch", "2.0.1"), ("numpy", "1.24.0"))


@pytest.fixture
def env_tf216():
    return _make_env(("tensorflow", "2.16.0"), ("numpy", "1.26.4"))


@pytest.fixture
def env_tf215():
    return _make_env(("tensorflow", "2.15.0"), ("numpy", "1.26.4"))


@pytest.fixture
def env_numpy2_old_pandas():
    return _make_env(("numpy", "2.0.0"), ("pandas", "1.5.3"))


@pytest.fixture
def env_numpy2_new_pandas():
    return _make_env(("numpy", "2.0.0"), ("pandas", "2.2.0"))


@pytest.fixture
def env_numpy2_old_sklearn():
    return _make_env(("numpy", "2.0.0"), ("scikit-learn", "1.3.2"))


@pytest.fixture
def env_numpy2_new_sklearn():
    return _make_env(("numpy", "2.0.0"), ("scikit-learn", "1.4.2"))


@pytest.fixture
def env_numpy2_old_scipy():
    return _make_env(("numpy", "2.0.0"), ("scipy", "1.11.4"))


@pytest.fixture
def env_numpy2_new_scipy():
    return _make_env(("numpy", "2.0.0"), ("scipy", "1.13.0"))


@pytest.fixture
def env_torch_and_tf():
    return _make_env(("torch", "2.4.0"), ("tensorflow", "2.16.0"), ("numpy", "1.26.4"))


@pytest.fixture
def env_sklearn_13():
    return _make_env(("scikit-learn", "1.3.0"))


@pytest.fixture
def env_sklearn_12():
    return _make_env(("scikit-learn", "1.2.2"))


@pytest.fixture
def env_pandas2():
    return _make_env(("pandas", "2.0.0"), ("numpy", "1.26.4"))


@pytest.fixture
def env_pandas1():
    return _make_env(("pandas", "1.5.3"), ("numpy", "1.26.4"))


@pytest.fixture
def env_healthy():
    """Compatible stack — should produce no errors."""
    return _make_env(
        ("numpy", "1.26.4"),
        ("pandas", "2.2.0"),
        ("scikit-learn", "1.4.0"),
        ("scipy", "1.13.0"),
    )


@pytest.fixture
def env_empty():
    return _make_env()


# ===========================================================================
# Sample TOML rulepack for unit tests
# ===========================================================================


SAMPLE_RULEPACK_TOML = textwrap.dedent("""\
    [meta]
    name = "test-rules"
    version = "1.0.0"
    description = "Test rulepack for unit tests"

    [[rules]]
    id = "torch-cuda-mismatch"
    severity = "error"
    category = "cuda_mismatch"
    description = "PyTorch {torch_version} requires CUDA >= 12.1, but system has CUDA {cuda_version}"
    fix = "Install PyTorch for CUDA 11.8: pip install torch --index-url https://download.pytorch.org/whl/cu118"

    [rules.when]
    package_installed = "torch"
    package_version = {torch = ">=2.4"}
    gpu_vendor = "nvidia"
    cuda_version = "<12.1"

    [[rules]]
    id = "numpy2-pandas-abi"
    severity = "error"
    category = "numpy_abi"
    description = "numpy {numpy_version} has ABI changes that break pandas {pandas_version} (< 2.1)"
    fix = "Upgrade pandas: pip install 'pandas>=2.1'"

    [rules.when]
    package_version = {numpy = ">=2.0", pandas = "<2.1"}

    [[rules]]
    id = "sklearn-deprecation-joblib"
    severity = "info"
    category = "deprecation"
    description = "sklearn.externals.joblib was removed in scikit-learn {scikit_learn_version}"
    fix = "Use 'import joblib' directly instead of 'from sklearn.externals import joblib'"

    [rules.when]
    package_version = {scikit-learn = ">=1.3"}
""")


# ===========================================================================
# 1. TOML Rule Loading
# ===========================================================================


class TestRuleLoading:
    """Test loading and parsing of TOML rulepack files."""

    def test_load_rules_from_string(self):
        from compatibillabuddy.kb.engine import load_rules_from_toml

        rules = load_rules_from_toml(SAMPLE_RULEPACK_TOML)
        assert len(rules) == 3

    def test_rule_has_required_fields(self):
        from compatibillabuddy.kb.engine import load_rules_from_toml

        rules = load_rules_from_toml(SAMPLE_RULEPACK_TOML)
        rule = rules[0]
        assert rule.id == "torch-cuda-mismatch"
        assert rule.severity == Severity.ERROR
        assert rule.category == "cuda_mismatch"
        assert len(rule.description) > 0
        assert rule.fix is not None
        assert len(rule.fix) > 0

    def test_rule_conditions_parsed(self):
        from compatibillabuddy.kb.engine import load_rules_from_toml

        rules = load_rules_from_toml(SAMPLE_RULEPACK_TOML)
        rule = rules[0]
        assert rule.when.package_installed == "torch"
        assert rule.when.gpu_vendor == "nvidia"
        assert rule.when.cuda_version == "<12.1"
        assert "torch" in rule.when.package_version
        assert rule.when.package_version["torch"] == ">=2.4"

    def test_rule_without_fix(self):
        """Rules without a 'fix' field should have fix=None."""
        toml_str = textwrap.dedent("""\
            [meta]
            name = "test"
            version = "1.0.0"

            [[rules]]
            id = "test-no-fix"
            severity = "warning"
            category = "test"
            description = "A test rule with no fix"

            [rules.when]
            package_version = {numpy = ">=1.0"}
        """)
        from compatibillabuddy.kb.engine import load_rules_from_toml

        rules = load_rules_from_toml(toml_str)
        assert len(rules) == 1
        assert rules[0].fix is None

    def test_rule_with_no_conditions(self):
        """A rule with no conditions should load (but always match)."""
        toml_str = textwrap.dedent("""\
            [meta]
            name = "test"
            version = "1.0.0"

            [[rules]]
            id = "always-fire"
            severity = "info"
            category = "global"
            description = "This rule always fires"
        """)
        from compatibillabuddy.kb.engine import load_rules_from_toml

        rules = load_rules_from_toml(toml_str)
        assert len(rules) == 1

    def test_load_invalid_toml_raises(self):
        from compatibillabuddy.kb.engine import load_rules_from_toml

        with pytest.raises(ValueError, match="Failed to parse"):
            load_rules_from_toml("this is [[[ not valid toml")

    def test_load_empty_rules(self):
        from compatibillabuddy.kb.engine import load_rules_from_toml

        toml_str = textwrap.dedent("""\
            [meta]
            name = "empty"
            version = "1.0.0"
        """)
        rules = load_rules_from_toml(toml_str)
        assert rules == []

    def test_severity_mapping_all_levels(self):
        """Ensure all severity strings map correctly."""
        from compatibillabuddy.kb.engine import load_rules_from_toml

        toml_str = textwrap.dedent("""\
            [meta]
            name = "test"
            version = "1.0.0"

            [[rules]]
            id = "err"
            severity = "error"
            category = "test"
            description = "error level"

            [[rules]]
            id = "warn"
            severity = "warning"
            category = "test"
            description = "warning level"

            [[rules]]
            id = "inf"
            severity = "info"
            category = "test"
            description = "info level"
        """)
        rules = load_rules_from_toml(toml_str)
        assert rules[0].severity == Severity.ERROR
        assert rules[1].severity == Severity.WARNING
        assert rules[2].severity == Severity.INFO

    def test_unknown_severity_defaults_to_warning(self):
        """Unknown severity string should default to WARNING for safety."""
        from compatibillabuddy.kb.engine import load_rules_from_toml

        toml_str = textwrap.dedent("""\
            [meta]
            name = "test"
            version = "1.0.0"

            [[rules]]
            id = "unknown-sev"
            severity = "critical"
            category = "test"
            description = "unknown severity"
        """)
        rules = load_rules_from_toml(toml_str)
        assert rules[0].severity == Severity.WARNING


# ===========================================================================
# 2. Version Matching (PEP 440) — direct unit tests
# ===========================================================================


class TestVersionMatching:
    """Exhaustive tests for the _version_matches helper."""

    def test_gte_match(self):
        from compatibillabuddy.kb.engine import _version_matches

        assert _version_matches("2.4.0", ">=2.4") is True
        assert _version_matches("2.4.1", ">=2.4") is True
        assert _version_matches("3.0.0", ">=2.4") is True
        assert _version_matches("2.3.9", ">=2.4") is False

    def test_lt_match(self):
        from compatibillabuddy.kb.engine import _version_matches

        assert _version_matches("11.8", "<12.1") is True
        assert _version_matches("12.0", "<12.1") is True
        assert _version_matches("12.1", "<12.1") is False
        assert _version_matches("12.6", "<12.1") is False

    def test_exact_match(self):
        from compatibillabuddy.kb.engine import _version_matches

        assert _version_matches("2.4.0", "==2.4.0") is True
        assert _version_matches("2.4.1", "==2.4.0") is False

    def test_not_equal(self):
        from compatibillabuddy.kb.engine import _version_matches

        assert _version_matches("2.4.0", "!=2.3.0") is True
        assert _version_matches("2.3.0", "!=2.3.0") is False

    def test_compound_specifier(self):
        from compatibillabuddy.kb.engine import _version_matches

        assert _version_matches("2.2.0", ">=2.1,<2.4") is True
        assert _version_matches("2.1.0", ">=2.1,<2.4") is True
        assert _version_matches("2.4.0", ">=2.1,<2.4") is False
        assert _version_matches("2.0.9", ">=2.1,<2.4") is False

    def test_compatible_release(self):
        """~= operator: compatible release."""
        from compatibillabuddy.kb.engine import _version_matches

        assert _version_matches("1.4.2", "~=1.4") is True
        assert _version_matches("1.5.0", "~=1.4") is True
        assert _version_matches("2.0.0", "~=1.4") is False

    def test_pre_release_version(self):
        """Pre-releases are normalized to their base version for rule matching.

        In strict PEP 440, 2.0.0rc1 does NOT satisfy >=2.0. But for
        compatibility diagnosis, a release candidate of 2.0 has the same
        hardware/ABI requirements as 2.0 itself. So our engine normalizes
        pre-releases to their base version before matching.

        See tests/unit/test_prerelease.py for exhaustive coverage.
        """
        from compatibillabuddy.kb.engine import _version_matches

        # Our engine: pre-release of 2.0 matches >=2.0 (base version fallback)
        assert _version_matches("2.0.0rc1", ">=2.0") is True
        assert _version_matches("2.0.0rc1", ">=1.0") is True
        assert _version_matches("2.0.0rc1", ">=2.0.0rc1") is True
        # Pre-release of 2.0 does NOT match >=2.1
        assert _version_matches("2.0.0rc1", ">=2.1") is False

    def test_post_release_version(self):
        from compatibillabuddy.kb.engine import _version_matches

        assert _version_matches("2.0.0.post1", ">=2.0") is True

    def test_invalid_installed_version(self):
        """Invalid version strings should return False (not crash)."""
        from compatibillabuddy.kb.engine import _version_matches

        assert _version_matches("not.a.version.at.all", ">=1.0") is False

    def test_invalid_specifier(self):
        """Invalid specifier strings should return False (not crash)."""
        from compatibillabuddy.kb.engine import _version_matches

        assert _version_matches("2.0.0", ">>>invalid<<<") is False

    def test_two_component_version(self):
        """Versions like '12.1' (CUDA) should work."""
        from compatibillabuddy.kb.engine import _version_matches

        assert _version_matches("12.1", ">=12.0") is True
        assert _version_matches("11.8", "<12.1") is True
        assert _version_matches("12.6", "<12.1") is False


# ===========================================================================
# 3. Template Variable Substitution
# ===========================================================================


class TestTemplateSubstitution:
    """Test description template filling."""

    def test_single_variable(self):
        from compatibillabuddy.kb.engine import _fill_template

        result = _fill_template("CUDA version is {cuda_version}", {"cuda_version": "11.8"})
        assert result == "CUDA version is 11.8"

    def test_multiple_variables(self):
        from compatibillabuddy.kb.engine import _fill_template

        result = _fill_template(
            "{pkg} {ver} needs CUDA {cuda}",
            {"pkg": "torch", "ver": "2.4.0", "cuda": "12.1"},
        )
        assert result == "torch 2.4.0 needs CUDA 12.1"

    def test_missing_variable_left_as_is(self):
        """Unknown template variables should remain in the string, not crash."""
        from compatibillabuddy.kb.engine import _fill_template

        result = _fill_template("version is {unknown_var}", {})
        assert result == "version is {unknown_var}"

    def test_no_variables(self):
        from compatibillabuddy.kb.engine import _fill_template

        result = _fill_template("No variables here", {"cuda": "12.1"})
        assert result == "No variables here"

    def test_empty_template(self):
        from compatibillabuddy.kb.engine import _fill_template

        result = _fill_template("", {"cuda": "12.1"})
        assert result == ""

    def test_hyphenated_package_name_normalized(self, hw_cpu_only, env_sklearn_13):
        """scikit-learn should become scikit_learn_version in template vars."""
        from compatibillabuddy.kb.engine import evaluate_rules, load_rules_from_toml

        rules = load_rules_from_toml(SAMPLE_RULEPACK_TOML)
        issues = evaluate_rules(rules, env_sklearn_13, hw_cpu_only)

        dep_issues = [i for i in issues if i.category == "deprecation"]
        assert len(dep_issues) == 1
        # The template had {scikit_learn_version}, should be filled
        assert "1.3.0" in dep_issues[0].description
        assert "{scikit_learn_version}" not in dep_issues[0].description


# ===========================================================================
# 4. Rule Condition Evaluation — Edge Cases
# ===========================================================================


class TestRuleConditionEdgeCases:
    """Test edge cases in rule condition matching."""

    def test_rule_with_no_conditions_always_fires(self, hw_cpu_only, env_empty):
        """A rule with no conditions should always fire."""
        from compatibillabuddy.kb.engine import evaluate_rules, load_rules_from_toml

        toml_str = textwrap.dedent("""\
            [meta]
            name = "test"
            version = "1.0.0"

            [[rules]]
            id = "always-fire"
            severity = "info"
            category = "global"
            description = "This rule always fires"
        """)
        rules = load_rules_from_toml(toml_str)
        issues = evaluate_rules(rules, env_empty, hw_cpu_only)
        assert len(issues) == 1
        assert issues[0].category == "global"

    def test_package_installed_only(self, hw_cpu_only):
        """Rule with only package_installed condition."""
        from compatibillabuddy.kb.engine import evaluate_rules, load_rules_from_toml

        toml_str = textwrap.dedent("""\
            [meta]
            name = "test"
            version = "1.0.0"

            [[rules]]
            id = "torch-present"
            severity = "info"
            category = "notice"
            description = "PyTorch is installed"

            [rules.when]
            package_installed = "torch"
        """)
        rules = load_rules_from_toml(toml_str)

        env_with = _make_env(("torch", "2.4.0"))
        env_without = _make_env(("numpy", "1.26.4"))

        assert len(evaluate_rules(rules, env_with, hw_cpu_only)) == 1
        assert len(evaluate_rules(rules, env_without, hw_cpu_only)) == 0

    def test_gpu_vendor_only(self, hw_nvidia_new_driver, hw_amd_gpu, hw_cpu_only):
        """Rule with only gpu_vendor condition."""
        from compatibillabuddy.kb.engine import evaluate_rules, load_rules_from_toml

        toml_str = textwrap.dedent("""\
            [meta]
            name = "test"
            version = "1.0.0"

            [[rules]]
            id = "nvidia-present"
            severity = "info"
            category = "notice"
            description = "NVIDIA GPU detected"

            [rules.when]
            gpu_vendor = "nvidia"
        """)
        rules = load_rules_from_toml(toml_str)
        env = _make_env()

        assert len(evaluate_rules(rules, env, hw_nvidia_new_driver)) == 1
        assert len(evaluate_rules(rules, env, hw_amd_gpu)) == 0
        assert len(evaluate_rules(rules, env, hw_cpu_only)) == 0

    def test_cuda_version_without_nvidia(self, hw_amd_gpu):
        """CUDA version condition on a non-NVIDIA machine should not fire."""
        from compatibillabuddy.kb.engine import evaluate_rules, load_rules_from_toml

        toml_str = textwrap.dedent("""\
            [meta]
            name = "test"
            version = "1.0.0"

            [[rules]]
            id = "cuda-old"
            severity = "error"
            category = "cuda"
            description = "CUDA is old"

            [rules.when]
            cuda_version = "<12.0"
        """)
        rules = load_rules_from_toml(toml_str)
        issues = evaluate_rules(rules, _make_env(), hw_amd_gpu)
        assert len(issues) == 0

    def test_all_conditions_combined(self, hw_nvidia_old_driver):
        """Rule with package_installed + package_version + gpu_vendor + cuda_version."""
        from compatibillabuddy.kb.engine import evaluate_rules, load_rules_from_toml

        rules = load_rules_from_toml(SAMPLE_RULEPACK_TOML)
        env = _make_env(("torch", "2.4.0"))
        issues = evaluate_rules(rules, env, hw_nvidia_old_driver)

        cuda_issues = [i for i in issues if i.category == "cuda_mismatch"]
        assert len(cuda_issues) == 1

    def test_all_conditions_fail_if_one_fails(self, hw_nvidia_new_driver):
        """All conditions must match — if CUDA is fine, rule should not fire."""
        from compatibillabuddy.kb.engine import evaluate_rules, load_rules_from_toml

        rules = load_rules_from_toml(SAMPLE_RULEPACK_TOML)
        env = _make_env(("torch", "2.4.0"))
        issues = evaluate_rules(rules, env, hw_nvidia_new_driver)

        cuda_issues = [i for i in issues if i.category == "cuda_mismatch"]
        assert len(cuda_issues) == 0

    def test_multi_package_version_all_must_match(self, hw_cpu_only):
        """When package_version has multiple packages, ALL must match."""
        from compatibillabuddy.kb.engine import evaluate_rules, load_rules_from_toml

        rules = load_rules_from_toml(SAMPLE_RULEPACK_TOML)

        # Both match: numpy >= 2.0 AND pandas < 2.1
        env_both = _make_env(("numpy", "2.0.0"), ("pandas", "1.5.3"))
        issues = evaluate_rules(rules, env_both, hw_cpu_only)
        assert any(i.category == "numpy_abi" for i in issues)

        # Only one matches: numpy >= 2.0 but pandas >= 2.1
        env_one = _make_env(("numpy", "2.0.0"), ("pandas", "2.2.0"))
        issues = evaluate_rules(rules, env_one, hw_cpu_only)
        assert not any(i.category == "numpy_abi" for i in issues)

        # One package missing entirely
        env_missing = _make_env(("numpy", "2.0.0"))
        issues = evaluate_rules(rules, env_missing, hw_cpu_only)
        assert not any(i.category == "numpy_abi" for i in issues)

    def test_multi_gpu_uses_first_for_cuda(self, hw_nvidia_dual_gpu):
        """With multiple GPUs, CUDA version should come from the first."""
        from compatibillabuddy.kb.engine import _get_system_cuda_version

        cuda_ver = _get_system_cuda_version(hw_nvidia_dual_gpu)
        assert cuda_ver == "12.1"


# ===========================================================================
# 5. End-to-End Rule Evaluation with Sample Rulepack
# ===========================================================================


class TestRuleEvaluation:
    """Test matching rules against hardware + environment."""

    def test_cuda_mismatch_detected(self, hw_nvidia_old_driver, env_torch24):
        from compatibillabuddy.kb.engine import evaluate_rules, load_rules_from_toml

        rules = load_rules_from_toml(SAMPLE_RULEPACK_TOML)
        issues = evaluate_rules(rules, env_torch24, hw_nvidia_old_driver)

        cuda_issues = [i for i in issues if i.category == "cuda_mismatch"]
        assert len(cuda_issues) == 1
        assert cuda_issues[0].severity == Severity.ERROR
        assert "11.8" in cuda_issues[0].description

    def test_cuda_mismatch_not_triggered_new_driver(self, hw_nvidia_new_driver, env_torch24):
        from compatibillabuddy.kb.engine import evaluate_rules, load_rules_from_toml

        rules = load_rules_from_toml(SAMPLE_RULEPACK_TOML)
        issues = evaluate_rules(rules, env_torch24, hw_nvidia_new_driver)

        cuda_issues = [i for i in issues if i.category == "cuda_mismatch"]
        assert len(cuda_issues) == 0

    def test_cuda_rule_skipped_on_cpu_only(self, hw_cpu_only, env_torch24):
        from compatibillabuddy.kb.engine import evaluate_rules, load_rules_from_toml

        rules = load_rules_from_toml(SAMPLE_RULEPACK_TOML)
        issues = evaluate_rules(rules, env_torch24, hw_cpu_only)

        cuda_issues = [i for i in issues if i.category == "cuda_mismatch"]
        assert len(cuda_issues) == 0

    def test_cuda_rule_skipped_on_amd(self, hw_amd_gpu, env_torch24):
        """NVIDIA CUDA rules should not fire on AMD hardware."""
        from compatibillabuddy.kb.engine import evaluate_rules, load_rules_from_toml

        rules = load_rules_from_toml(SAMPLE_RULEPACK_TOML)
        issues = evaluate_rules(rules, env_torch24, hw_amd_gpu)

        cuda_issues = [i for i in issues if i.category == "cuda_mismatch"]
        assert len(cuda_issues) == 0

    def test_numpy_abi_break_detected(self, hw_cpu_only, env_numpy2_old_pandas):
        from compatibillabuddy.kb.engine import evaluate_rules, load_rules_from_toml

        rules = load_rules_from_toml(SAMPLE_RULEPACK_TOML)
        issues = evaluate_rules(rules, env_numpy2_old_pandas, hw_cpu_only)

        abi_issues = [i for i in issues if i.category == "numpy_abi"]
        assert len(abi_issues) == 1
        assert abi_issues[0].severity == Severity.ERROR

    def test_numpy_abi_not_triggered_when_pandas_new(self, hw_cpu_only, env_numpy2_new_pandas):
        from compatibillabuddy.kb.engine import evaluate_rules, load_rules_from_toml

        rules = load_rules_from_toml(SAMPLE_RULEPACK_TOML)
        issues = evaluate_rules(rules, env_numpy2_new_pandas, hw_cpu_only)

        abi_issues = [i for i in issues if i.category == "numpy_abi"]
        assert len(abi_issues) == 0

    def test_deprecation_info_detected(self, hw_cpu_only, env_sklearn_13):
        from compatibillabuddy.kb.engine import evaluate_rules, load_rules_from_toml

        rules = load_rules_from_toml(SAMPLE_RULEPACK_TOML)
        issues = evaluate_rules(rules, env_sklearn_13, hw_cpu_only)

        dep_issues = [i for i in issues if i.category == "deprecation"]
        assert len(dep_issues) == 1
        assert dep_issues[0].severity == Severity.INFO

    def test_deprecation_not_triggered_old_sklearn(self, hw_cpu_only, env_sklearn_12):
        from compatibillabuddy.kb.engine import evaluate_rules, load_rules_from_toml

        rules = load_rules_from_toml(SAMPLE_RULEPACK_TOML)
        issues = evaluate_rules(rules, env_sklearn_12, hw_cpu_only)

        dep_issues = [i for i in issues if i.category == "deprecation"]
        assert len(dep_issues) == 0

    def test_healthy_env_no_errors(self, hw_cpu_only, env_healthy):
        from compatibillabuddy.kb.engine import evaluate_rules, load_rules_from_toml

        rules = load_rules_from_toml(SAMPLE_RULEPACK_TOML)
        issues = evaluate_rules(rules, env_healthy, hw_cpu_only)

        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_missing_package_skips_rule(self, hw_nvidia_old_driver, env_empty):
        from compatibillabuddy.kb.engine import evaluate_rules, load_rules_from_toml

        rules = load_rules_from_toml(SAMPLE_RULEPACK_TOML)
        issues = evaluate_rules(rules, env_empty, hw_nvidia_old_driver)

        cuda_issues = [i for i in issues if i.category == "cuda_mismatch"]
        assert len(cuda_issues) == 0

    def test_issues_contain_affected_packages(self, hw_cpu_only, env_numpy2_old_pandas):
        from compatibillabuddy.kb.engine import evaluate_rules, load_rules_from_toml

        rules = load_rules_from_toml(SAMPLE_RULEPACK_TOML)
        issues = evaluate_rules(rules, env_numpy2_old_pandas, hw_cpu_only)

        abi_issues = [i for i in issues if i.category == "numpy_abi"]
        assert len(abi_issues) == 1
        assert "numpy" in abi_issues[0].affected_packages
        assert "pandas" in abi_issues[0].affected_packages

    def test_description_template_populated(self, hw_nvidia_old_driver, env_torch24):
        from compatibillabuddy.kb.engine import evaluate_rules, load_rules_from_toml

        rules = load_rules_from_toml(SAMPLE_RULEPACK_TOML)
        issues = evaluate_rules(rules, env_torch24, hw_nvidia_old_driver)

        cuda_issues = [i for i in issues if i.category == "cuda_mismatch"]
        assert len(cuda_issues) == 1
        assert "11.8" in cuda_issues[0].description
        assert "{cuda_version}" not in cuda_issues[0].description

    def test_multiple_rules_fire_simultaneously(self, hw_nvidia_old_driver):
        """Multiple rules can fire for the same environment."""
        from compatibillabuddy.kb.engine import evaluate_rules, load_rules_from_toml

        env = _make_env(
            ("torch", "2.4.0"),
            ("numpy", "2.0.0"),
            ("pandas", "1.5.3"),
            ("scikit-learn", "1.4.0"),
        )
        rules = load_rules_from_toml(SAMPLE_RULEPACK_TOML)
        issues = evaluate_rules(rules, env, hw_nvidia_old_driver)

        categories = {i.category for i in issues}
        # Should fire: cuda_mismatch, numpy_abi, deprecation
        assert "cuda_mismatch" in categories
        assert "numpy_abi" in categories
        assert "deprecation" in categories
        assert len(issues) == 3


# ===========================================================================
# 6. Bundled Rulepack Tests — every rule in ml_core.toml
# ===========================================================================


class TestBundledRulepacks:
    """Test loading and validation of all bundled rulepack files."""

    def test_load_bundled_rulepacks(self):
        from compatibillabuddy.kb.engine import load_bundled_rulepacks

        rules = load_bundled_rulepacks()
        assert isinstance(rules, list)
        assert len(rules) > 0

    def test_bundled_rules_have_unique_ids(self):
        from compatibillabuddy.kb.engine import load_bundled_rulepacks

        rules = load_bundled_rulepacks()
        ids = [r.id for r in rules]
        assert len(ids) == len(set(ids)), f"Duplicate rule IDs: {[x for x in ids if ids.count(x) > 1]}"

    def test_all_bundled_rules_have_descriptions(self):
        from compatibillabuddy.kb.engine import load_bundled_rulepacks

        rules = load_bundled_rulepacks()
        for rule in rules:
            assert len(rule.description) > 0, f"Rule {rule.id} has empty description"

    def test_all_bundled_rules_have_categories(self):
        from compatibillabuddy.kb.engine import load_bundled_rulepacks

        rules = load_bundled_rulepacks()
        for rule in rules:
            assert len(rule.category) > 0, f"Rule {rule.id} has empty category"

    def test_all_bundled_error_rules_have_fix(self):
        """Every ERROR-level rule should provide a fix suggestion."""
        from compatibillabuddy.kb.engine import load_bundled_rulepacks

        rules = load_bundled_rulepacks()
        for rule in rules:
            if rule.severity == Severity.ERROR:
                assert rule.fix is not None and len(rule.fix) > 0, (
                    f"ERROR rule {rule.id} has no fix suggestion"
                )


class TestBundledMlCoreRules:
    """Test each specific rule in ml_core.toml against realistic scenarios."""

    @pytest.fixture(autouse=True)
    def _load_bundled(self):
        from compatibillabuddy.kb.engine import load_bundled_rulepacks

        self.rules = load_bundled_rulepacks()

    def _issues_with_id(self, issues, rule_id):
        return [i for i in issues if i.category == self._rule_category(rule_id)]

    def _rule_category(self, rule_id):
        for r in self.rules:
            if r.id == rule_id:
                return r.category
        raise ValueError(f"No rule with id={rule_id}")

    def _eval(self, env, hw):
        from compatibillabuddy.kb.engine import evaluate_rules

        return evaluate_rules(self.rules, env, hw)

    def _fires(self, rule_id, env, hw):
        """Check if a specific rule fires."""
        from compatibillabuddy.kb.engine import evaluate_rules

        issues = evaluate_rules(self.rules, env, hw)
        # Match by checking the description contains content from the rule
        rule = next(r for r in self.rules if r.id == rule_id)
        return any(
            i.category == rule.category and i.severity == rule.severity
            for i in issues
        )

    # --- torch CUDA rules ---

    def test_torch24_cuda_old_fires(self, hw_nvidia_old_driver, env_torch24):
        assert self._fires("torch-2.4-needs-cuda-12.1", env_torch24, hw_nvidia_old_driver)

    def test_torch24_cuda_new_does_not_fire(self, hw_nvidia_new_driver, env_torch24):
        issues = self._eval(env_torch24, hw_nvidia_new_driver)
        cuda_errors = [i for i in issues if i.category == "cuda_mismatch"]
        assert len(cuda_errors) == 0

    def test_torch24_cuda_mid_does_not_fire(self, hw_nvidia_mid_driver, env_torch24):
        """CUDA 12.1 should satisfy torch 2.4's requirement."""
        issues = self._eval(env_torch24, hw_nvidia_mid_driver)
        cuda_errors = [i for i in issues if i.category == "cuda_mismatch"]
        assert len(cuda_errors) == 0

    def test_torch22_cuda_very_old_fires(self, hw_nvidia_old_driver):
        """torch 2.2 needs CUDA >= 11.8, CUDA 11.7 should fail."""
        hw = HardwareProfile(
            os_name="Linux",
            os_version="6.1.0",
            cpu_arch="x86_64",
            cpu_name="Intel Xeon",
            python_version="3.12.0",
            gpus=[
                GpuInfo(
                    vendor=GpuVendor.NVIDIA,
                    name="GTX 1080",
                    driver_version="470.82",
                    cuda_version="11.4",
                )
            ],
        )
        assert self._fires("torch-2.1-needs-cuda-11.8", _make_env(("torch", "2.2.0")), hw)

    def test_torch22_cuda_118_does_not_fire(self, hw_nvidia_old_driver):
        """torch 2.2 with CUDA 11.8 should be fine."""
        env = _make_env(("torch", "2.2.0"))
        issues = self._eval(env, hw_nvidia_old_driver)
        # hw_nvidia_old_driver has CUDA 11.8, which satisfies >=11.8
        # The rule "torch-2.1-needs-cuda-11.8" checks cuda < 11.8
        matching = [i for i in issues if "torch" in i.affected_packages and i.category == "cuda_mismatch"]
        assert len(matching) == 0

    # --- TensorFlow CUDA rules ---

    def test_tf216_cuda_old_fires(self, hw_nvidia_old_driver, env_tf216):
        assert self._fires("tensorflow-2.16-needs-cuda-12.3", env_tf216, hw_nvidia_old_driver)

    def test_tf216_cuda_new_does_not_fire(self, hw_nvidia_new_driver, env_tf216):
        issues = self._eval(env_tf216, hw_nvidia_new_driver)
        tf_cuda = [i for i in issues if "tensorflow" in i.affected_packages and i.category == "cuda_mismatch"]
        assert len(tf_cuda) == 0

    def test_tf215_does_not_fire(self, hw_nvidia_old_driver, env_tf215):
        """TensorFlow < 2.16 should not trigger the 12.3 CUDA rule."""
        issues = self._eval(env_tf215, hw_nvidia_old_driver)
        tf_cuda = [i for i in issues if "tensorflow" in i.affected_packages and i.category == "cuda_mismatch"]
        assert len(tf_cuda) == 0

    # --- NumPy ABI rules ---

    def test_numpy2_old_pandas_fires(self, hw_cpu_only, env_numpy2_old_pandas):
        assert self._fires("numpy2-pandas-abi", env_numpy2_old_pandas, hw_cpu_only)

    def test_numpy2_new_pandas_does_not_fire(self, hw_cpu_only, env_numpy2_new_pandas):
        issues = self._eval(env_numpy2_new_pandas, hw_cpu_only)
        abi = [i for i in issues if i.category == "numpy_abi" and "pandas" in i.affected_packages]
        assert len(abi) == 0

    def test_numpy2_old_sklearn_fires(self, hw_cpu_only, env_numpy2_old_sklearn):
        assert self._fires("numpy2-sklearn-abi", env_numpy2_old_sklearn, hw_cpu_only)

    def test_numpy2_new_sklearn_does_not_fire(self, hw_cpu_only, env_numpy2_new_sklearn):
        issues = self._eval(env_numpy2_new_sklearn, hw_cpu_only)
        abi = [i for i in issues if i.category == "numpy_abi" and "scikit-learn" in i.affected_packages]
        assert len(abi) == 0

    def test_numpy2_old_scipy_fires(self, hw_cpu_only, env_numpy2_old_scipy):
        assert self._fires("numpy2-scipy-abi", env_numpy2_old_scipy, hw_cpu_only)

    def test_numpy2_new_scipy_does_not_fire(self, hw_cpu_only, env_numpy2_new_scipy):
        issues = self._eval(env_numpy2_new_scipy, hw_cpu_only)
        abi = [i for i in issues if i.category == "numpy_abi" and "scipy" in i.affected_packages]
        assert len(abi) == 0

    # --- Co-installation warning ---

    def test_torch_and_tf_coinstall_warns(self, hw_cpu_only, env_torch_and_tf):
        assert self._fires("torch-tf-coinstall", env_torch_and_tf, hw_cpu_only)

    def test_torch_only_no_coinstall_warning(self, hw_cpu_only, env_torch24):
        issues = self._eval(env_torch24, hw_cpu_only)
        coinstall = [i for i in issues if i.category == "coinstall_conflict"]
        assert len(coinstall) == 0

    # --- Deprecation rules ---

    def test_sklearn_joblib_deprecation_fires(self, hw_cpu_only, env_sklearn_13):
        assert self._fires("sklearn-joblib-removed", env_sklearn_13, hw_cpu_only)

    def test_sklearn_joblib_deprecation_not_old(self, hw_cpu_only, env_sklearn_12):
        issues = self._eval(env_sklearn_12, hw_cpu_only)
        dep = [i for i in issues if i.category == "deprecation"]
        assert len(dep) == 0

    def test_pandas_append_deprecation_fires(self, hw_cpu_only, env_pandas2):
        assert self._fires("pandas-append-removed", env_pandas2, hw_cpu_only)

    def test_pandas_append_deprecation_not_old(self, hw_cpu_only, env_pandas1):
        issues = self._eval(env_pandas1, hw_cpu_only)
        dep = [i for i in issues if i.category == "deprecation" and "pandas" in i.affected_packages]
        assert len(dep) == 0
