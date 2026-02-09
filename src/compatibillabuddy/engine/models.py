"""Core data models for compatibillabuddy.

These models flow through the entire system:
- Hardware probe outputs HardwareProfile
- Environment inspector outputs EnvironmentInventory
- Doctor/KB produce CompatIssue lists
- Agent passes them to Gemini as structured tool results
- Reports render them for humans
"""

from __future__ import annotations

import enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class GpuVendor(str, enum.Enum):
    """Known GPU vendors."""

    NVIDIA = "nvidia"
    AMD = "amd"
    APPLE = "apple"


class Severity(int, enum.Enum):
    """Issue severity levels. Lower value = more severe."""

    ERROR = 1
    WARNING = 2
    INFO = 3


# ---------------------------------------------------------------------------
# Hardware models
# ---------------------------------------------------------------------------


class GpuInfo(BaseModel):
    """Information about a single GPU."""

    vendor: GpuVendor
    name: str
    driver_version: str

    # NVIDIA-specific
    cuda_version: Optional[str] = None
    compute_capability: Optional[str] = None

    # AMD-specific
    rocm_version: Optional[str] = None

    # Shared
    vram_mb: Optional[int] = None


class HardwareProfile(BaseModel):
    """Complete hardware fingerprint of the current (or target) machine.

    Produced by the hardware probe module. Consumed by the KB rule engine,
    the doctor command, and the AI agent for context.
    """

    os_name: str
    os_version: str
    cpu_arch: str
    cpu_name: str
    python_version: str
    gpus: list[GpuInfo] = Field(default_factory=list)

    @property
    def has_nvidia_gpu(self) -> bool:
        """True if at least one NVIDIA GPU is detected."""
        return any(gpu.vendor == GpuVendor.NVIDIA for gpu in self.gpus)

    @property
    def has_amd_gpu(self) -> bool:
        """True if at least one AMD GPU is detected."""
        return any(gpu.vendor == GpuVendor.AMD for gpu in self.gpus)

    @property
    def has_apple_gpu(self) -> bool:
        """True if an Apple Silicon GPU (Metal/MPS) is detected."""
        return any(gpu.vendor == GpuVendor.APPLE for gpu in self.gpus)


# ---------------------------------------------------------------------------
# Environment models
# ---------------------------------------------------------------------------


class InstalledPackage(BaseModel):
    """A single installed Python package."""

    name: str
    version: str
    requires: list[str] = Field(default_factory=list)
    location: Optional[str] = None
    installer: Optional[str] = None


class EnvironmentInventory(BaseModel):
    """Snapshot of the current Python environment.

    Produced by parsing `pip inspect` or `pip list --format=json` output.
    """

    python_version: str
    python_executable: str
    packages: list[InstalledPackage] = Field(default_factory=list)

    def get_package(self, name: str) -> Optional[InstalledPackage]:
        """Look up a package by name (case-insensitive).

        Returns None if the package is not installed.
        """
        name_lower = name.lower()
        for pkg in self.packages:
            if pkg.name.lower() == name_lower:
                return pkg
        return None


# ---------------------------------------------------------------------------
# Compatibility issue models
# ---------------------------------------------------------------------------


class DiagnosisResult(BaseModel):
    """Complete result of a compatibility diagnosis run.

    Bundles hardware profile, environment inventory, discovered issues,
    and timing metadata. Produced by the doctor module. Consumed by
    the CLI reporter, JSON export, and AI agent context.
    """

    hardware: HardwareProfile
    environment: EnvironmentInventory
    issues: list["CompatIssue"] = Field(default_factory=list)

    # Timing metadata (seconds)
    hardware_probe_seconds: float = 0.0
    environment_inspect_seconds: float = 0.0
    rule_evaluation_seconds: float = 0.0
    total_seconds: float = 0.0

    @property
    def has_errors(self) -> bool:
        """True if any issues have ERROR severity."""
        return any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        """True if any issues have WARNING severity."""
        return any(i.severity == Severity.WARNING for i in self.issues)

    @property
    def issue_count(self) -> int:
        """Total number of diagnosed issues."""
        return len(self.issues)


class CompatIssue(BaseModel):
    """A single diagnosed compatibility issue.

    Produced by the KB rule engine or the doctor command.
    Consumed by the explain module and the AI agent.
    """

    severity: Severity
    category: str = Field(..., min_length=1)
    description: str
    affected_packages: list[str] = Field(default_factory=list)
    fix_suggestion: Optional[str] = None

    @field_validator("category")
    @classmethod
    def category_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("category must not be blank")
        return v
