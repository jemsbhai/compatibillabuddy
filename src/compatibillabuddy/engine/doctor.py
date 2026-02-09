"""Doctor: diagnostic orchestrator that wires probe + inspector + KB engine.

This is the core diagnostic pipeline. It collects hardware info, environment
packages, and evaluates compatibility rules to produce a DiagnosisResult.

Supports dependency injection for testing — pass pre-built hardware/env/rules
to avoid subprocess calls. When arguments are None, calls the real subsystems.
"""

from __future__ import annotations

import time
from typing import Optional

from compatibillabuddy.engine.models import (
    DiagnosisResult,
    EnvironmentInventory,
    HardwareProfile,
)
from compatibillabuddy.hardware.inspector import inspect_environment
from compatibillabuddy.hardware.probe import probe_hardware
from compatibillabuddy.kb.engine import Rule, evaluate_rules, load_bundled_rulepacks


def diagnose(
    hardware: Optional[HardwareProfile] = None,
    env: Optional[EnvironmentInventory] = None,
    rules: Optional[list[Rule]] = None,
) -> DiagnosisResult:
    """Run a full compatibility diagnosis.

    Probes hardware, inspects the Python environment, evaluates all
    compatibility rules, and returns a sorted DiagnosisResult.

    Args:
        hardware: Pre-built hardware profile, or None to auto-detect.
        env: Pre-built environment inventory, or None to auto-detect.
        rules: List of rules to evaluate, or None to load bundled rulepacks.

    Returns:
        DiagnosisResult with issues sorted by severity (errors first).

    Raises:
        RuntimeError: If hardware probe or environment inspection fails
            when auto-detecting (i.e. when the corresponding arg is None).
    """
    total_start = time.monotonic()

    # --- Phase 1: Hardware probe ---
    hw_seconds = 0.0
    if hardware is None:
        hw_start = time.monotonic()
        hardware = probe_hardware()
        hw_seconds = time.monotonic() - hw_start

    # --- Phase 2: Environment inspection ---
    env_seconds = 0.0
    if env is None:
        env_start = time.monotonic()
        env = inspect_environment()
        env_seconds = time.monotonic() - env_start

    # --- Phase 3: Rule loading ---
    if rules is None:
        rules = load_bundled_rulepacks()

    # --- Phase 4: Rule evaluation ---
    eval_start = time.monotonic()
    issues = evaluate_rules(rules, env, hardware)
    eval_seconds = time.monotonic() - eval_start

    # Sort by severity: ERROR (1) first, then WARNING (2), then INFO (3).
    # Stable sort preserves original rule evaluation order within each level.
    issues.sort(key=lambda i: i.severity.value)

    total_seconds = time.monotonic() - total_start

    return DiagnosisResult(
        hardware=hardware,
        environment=env,
        issues=issues,
        hardware_probe_seconds=hw_seconds,
        environment_inspect_seconds=env_seconds,
        rule_evaluation_seconds=eval_seconds,
        total_seconds=total_seconds,
    )
