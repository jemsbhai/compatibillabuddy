"""Knowledge base rule engine: load TOML rulepacks and evaluate against environments.

Rules are curated TOML files that encode known-bad package/hardware combinations.
The engine matches rules against a HardwareProfile + EnvironmentInventory and
produces CompatIssue lists for the doctor command and AI agent.
"""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, Field

from compatibillabuddy.engine.models import (
    CompatIssue,
    EnvironmentInventory,
    GpuVendor,
    HardwareProfile,
    Severity,
)

# ---------------------------------------------------------------------------
# Rule models
# ---------------------------------------------------------------------------


class RuleCondition(BaseModel):
    """Conditions that must ALL be true for a rule to fire."""

    package_installed: str | None = None
    package_version: dict[str, str] = Field(default_factory=dict)
    gpu_vendor: str | None = None
    cuda_version: str | None = None


class Rule(BaseModel):
    """A single compatibility rule from a rulepack."""

    id: str
    severity: Severity
    category: str
    description: str
    fix: str | None = None
    when: RuleCondition = Field(default_factory=RuleCondition)


# ---------------------------------------------------------------------------
# Rule loading
# ---------------------------------------------------------------------------


_SEVERITY_MAP = {
    "error": Severity.ERROR,
    "warning": Severity.WARNING,
    "info": Severity.INFO,
}


def load_rules_from_toml(toml_string: str) -> list[Rule]:
    """Parse a TOML rulepack string into a list of Rule objects.

    Args:
        toml_string: Raw TOML content of a rulepack.

    Returns:
        List of Rule objects.

    Raises:
        ValueError: If the TOML is invalid.
    """
    try:
        data = tomllib.loads(toml_string)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"Failed to parse rulepack TOML: {e}") from e

    raw_rules = data.get("rules", [])
    rules = []

    for raw in raw_rules:
        severity = _SEVERITY_MAP.get(raw.get("severity", ""), Severity.WARNING)
        when_data = raw.get("when", {})

        condition = RuleCondition(
            package_installed=when_data.get("package_installed"),
            package_version=when_data.get("package_version", {}),
            gpu_vendor=when_data.get("gpu_vendor"),
            cuda_version=when_data.get("cuda_version"),
        )

        rules.append(
            Rule(
                id=raw["id"],
                severity=severity,
                category=raw["category"],
                description=raw["description"],
                fix=raw.get("fix"),
                when=condition,
            )
        )

    return rules


def load_bundled_rulepacks() -> list[Rule]:
    """Load all bundled .toml rulepack files shipped with the package.

    Returns:
        Combined list of rules from all bundled rulepacks.
    """
    rulepacks_dir = Path(__file__).parent / "rulepacks"
    all_rules: list[Rule] = []

    if not rulepacks_dir.is_dir():
        return all_rules

    for toml_path in sorted(rulepacks_dir.glob("*.toml")):
        content = toml_path.read_text(encoding="utf-8")
        all_rules.extend(load_rules_from_toml(content))

    return all_rules


# ---------------------------------------------------------------------------
# Rule evaluation
# ---------------------------------------------------------------------------


def evaluate_rules(
    rules: list[Rule],
    env: EnvironmentInventory,
    hardware: HardwareProfile,
) -> list[CompatIssue]:
    """Evaluate rules against an environment and hardware profile.

    Args:
        rules: List of rules to evaluate.
        env: Current environment inventory.
        hardware: Current hardware profile.

    Returns:
        List of CompatIssue for each rule that fires.
    """
    issues: list[CompatIssue] = []

    for rule in rules:
        if _rule_matches(rule, env, hardware):
            template_vars = _build_template_vars(rule, env, hardware)
            description = _fill_template(rule.description, template_vars)
            affected = _get_affected_packages(rule, env)

            issues.append(
                CompatIssue(
                    severity=rule.severity,
                    category=rule.category,
                    description=description,
                    affected_packages=affected,
                    fix_suggestion=rule.fix,
                )
            )

    return issues


def _rule_matches(
    rule: Rule,
    env: EnvironmentInventory,
    hardware: HardwareProfile,
) -> bool:
    """Check if ALL conditions in a rule are satisfied.

    Returns False as soon as any condition fails (short-circuit).
    """
    cond = rule.when

    # Check: specific package must be installed
    if cond.package_installed is not None and env.get_package(cond.package_installed) is None:
        return False

    # Check: package version constraints
    if cond.package_version:
        for pkg_name, version_spec_str in cond.package_version.items():
            pkg = env.get_package(pkg_name)
            if pkg is None:
                return False
            if not _version_matches(pkg.version, version_spec_str):
                return False

    # Check: GPU vendor
    if cond.gpu_vendor is not None:
        vendor = cond.gpu_vendor.lower()
        if (
            vendor == "nvidia"
            and not hardware.has_nvidia_gpu
            or vendor == "amd"
            and not hardware.has_amd_gpu
            or vendor == "apple"
            and not hardware.has_apple_gpu
        ):
            return False

    # Check: CUDA version constraint
    if cond.cuda_version is not None:
        cuda_ver = _get_system_cuda_version(hardware)
        if cuda_ver is None:
            return False
        if not _version_matches(cuda_ver, cond.cuda_version):
            return False

    return True


def _version_matches(installed_version: str, specifier_str: str) -> bool:
    """Check if an installed version satisfies a PEP 440 specifier.

    For compatibility rule matching, pre-release versions (rc, dev, alpha, beta)
    are normalized to their base release before comparison. This is because a
    release candidate of X has the same hardware/ABI requirements as X itself.

    For example:
        - torch 2.4.0rc1 has the same CUDA requirements as torch 2.4.0
        - numpy 2.0.0.dev1 has the same ABI surface as numpy 2.0.0

    This intentionally differs from strict PEP 440 specifier matching, which
    excludes pre-releases from ranges like >=2.4. That behavior is correct for
    package resolution but wrong for compatibility diagnosis.

    Args:
        installed_version: The version string of the installed package.
        specifier_str: A PEP 440 specifier (e.g. ">=2.0", "<12.1").

    Returns:
        True if the version (or its base release) matches the specifier.
    """
    try:
        version = Version(installed_version)
        specifier = SpecifierSet(specifier_str)

        # First try exact match (handles stable versions and pre-release-aware specs)
        if specifier.contains(version, prereleases=True):
            return True

        # If the version is a pre-release and didn't match, try its base release.
        # A pre-release of X should trigger the same compatibility rules as X.
        if version.is_prerelease or version.is_devrelease:
            base = Version(version.base_version)
            if specifier.contains(base, prereleases=True):
                return True

        return False
    except (InvalidVersion, InvalidSpecifier):
        return False


def _get_system_cuda_version(hardware: HardwareProfile) -> str | None:
    """Extract the CUDA version from the hardware profile.

    Returns the CUDA version from the first NVIDIA GPU, or None.
    """
    for gpu in hardware.gpus:
        if gpu.vendor == GpuVendor.NVIDIA and gpu.cuda_version is not None:
            return gpu.cuda_version
    return None


def _build_template_vars(
    rule: Rule,
    env: EnvironmentInventory,
    hardware: HardwareProfile,
) -> dict[str, str]:
    """Build a dict of template variables for description formatting.

    Includes package versions (as {package_name}_version) and hardware fields.
    """
    variables: dict[str, str] = {}

    # Add package versions
    for pkg_name in rule.when.package_version:
        pkg = env.get_package(pkg_name)
        if pkg is not None:
            safe_name = pkg_name.replace("-", "_")
            variables[f"{safe_name}_version"] = pkg.version

    # Add hardware variables
    cuda_ver = _get_system_cuda_version(hardware)
    if cuda_ver is not None:
        variables["cuda_version"] = cuda_ver

    if hardware.gpus:
        variables["gpu_name"] = hardware.gpus[0].name
        variables["driver_version"] = hardware.gpus[0].driver_version

    # For CUDA mismatch rules, try to extract the min required CUDA
    if "cuda_version" in rule.when.__dict__ and rule.when.cuda_version:
        match = re.search(r"[\d.]+", rule.when.cuda_version)
        if match:
            variables["min_cuda"] = match.group()

    return variables


def _fill_template(template: str, variables: dict[str, str]) -> str:
    """Fill in {variable} placeholders in a template string.

    Unknown placeholders are left as-is to avoid KeyError.
    """
    result = template
    for key, value in variables.items():
        result = result.replace(f"{{{key}}}", value)
    return result


def _get_affected_packages(rule: Rule, env: EnvironmentInventory) -> list[str]:
    """Determine which packages are affected by a rule firing.

    Uses the package names from the rule's version conditions.
    """
    affected: list[str] = []

    # From package_installed
    if rule.when.package_installed:
        pkg = env.get_package(rule.when.package_installed)
        if pkg is not None and pkg.name not in affected:
            affected.append(pkg.name)

    # From package_version constraints
    for pkg_name in rule.when.package_version:
        pkg = env.get_package(pkg_name)
        if pkg is not None and pkg.name not in affected:
            affected.append(pkg.name)

    return affected
