"""Tests for environment inspector module — written BEFORE implementation (TDD).

Tests mock `pip inspect` subprocess output so they run anywhere.
One integration test runs the real inspector on this machine.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from compatibillabuddy.engine.models import EnvironmentInventory

# ---------------------------------------------------------------------------
# Fixtures: canned pip inspect output
# ---------------------------------------------------------------------------


@pytest.fixture
def pip_inspect_output_minimal():
    """Minimal pip inspect JSON with two packages."""
    return json.dumps(
        {
            "version": "1",
            "pip_version": "24.0",
            "installed": [
                {
                    "metadata": {
                        "metadata_version": "2.1",
                        "name": "numpy",
                        "version": "1.26.4",
                    },
                    "metadata_location": "/usr/lib/python3.12/site-packages/numpy-1.26.4.dist-info",
                    "installer": "pip",
                    "requested": True,
                },
                {
                    "metadata": {
                        "metadata_version": "2.1",
                        "name": "pandas",
                        "version": "2.2.0",
                        "requires_dist": [
                            "numpy>=1.23.2",
                            "python-dateutil>=2.8.2",
                            "pytz>=2020.1",
                            "tzdata>=2022.7",
                        ],
                    },
                    "metadata_location": "/usr/lib/python3.12/site-packages/pandas-2.2.0.dist-info",
                    "installer": "pip",
                    "requested": True,
                },
            ],
            "environment": {
                "python_version": "3.12.0",
                "python_executable": "/usr/bin/python3.12",
            },
        }
    )


@pytest.fixture
def pip_inspect_output_ml_stack():
    """pip inspect JSON simulating a typical ML stack."""
    return json.dumps(
        {
            "version": "1",
            "pip_version": "24.0",
            "installed": [
                {
                    "metadata": {
                        "metadata_version": "2.1",
                        "name": "torch",
                        "version": "2.4.0",
                        "requires_dist": [
                            "filelock",
                            "typing-extensions>=4.8.0",
                            "sympy",
                            "networkx",
                            "jinja2",
                            "fsspec",
                            "nvidia-cuda-nvrtc-cu12==12.1.105",
                            "nvidia-cuda-runtime-cu12==12.1.105",
                            "nvidia-cublas-cu12==12.1.3.1",
                        ],
                    },
                    "metadata_location": (
                        "/home/user/.venv/lib/python3.12/"
                        "site-packages/torch-2.4.0.dist-info"
                    ),
                    "installer": "pip",
                    "requested": True,
                },
                {
                    "metadata": {
                        "metadata_version": "2.1",
                        "name": "numpy",
                        "version": "2.0.0",
                    },
                    "metadata_location": (
                        "/home/user/.venv/lib/python3.12/"
                        "site-packages/numpy-2.0.0.dist-info"
                    ),
                    "installer": "pip",
                    "requested": True,
                },
                {
                    "metadata": {
                        "metadata_version": "2.1",
                        "name": "scikit-learn",
                        "version": "1.4.0",
                        "requires_dist": [
                            "numpy>=1.19.5",
                            "scipy>=1.6.0",
                            "joblib>=1.2.0",
                            "threadpoolctl>=3.1.0",
                        ],
                    },
                    "metadata_location": (
                        "/home/user/.venv/lib/python3.12/"
                        "site-packages/scikit_learn-1.4.0.dist-info"
                    ),
                    "installer": "pip",
                    "requested": True,
                },
            ],
            "environment": {
                "python_version": "3.12.0",
                "python_executable": "/home/user/.venv/bin/python",
            },
        }
    )


@pytest.fixture
def pip_inspect_output_empty():
    """pip inspect JSON for a fresh venv with no packages."""
    return json.dumps(
        {
            "version": "1",
            "pip_version": "24.0",
            "installed": [],
            "environment": {
                "python_version": "3.12.0",
                "python_executable": "/home/user/.venv/bin/python",
            },
        }
    )


# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------


class TestParsePipInspect:
    """Test parsing of pip inspect JSON output."""

    def test_parse_minimal(self, pip_inspect_output_minimal):
        from compatibillabuddy.hardware.inspector import parse_pip_inspect

        inv = parse_pip_inspect(pip_inspect_output_minimal)

        assert isinstance(inv, EnvironmentInventory)
        assert inv.python_version == "3.12.0"
        assert inv.python_executable == "/usr/bin/python3.12"
        assert len(inv.packages) == 2

    def test_parse_package_fields(self, pip_inspect_output_minimal):
        from compatibillabuddy.hardware.inspector import parse_pip_inspect

        inv = parse_pip_inspect(pip_inspect_output_minimal)

        numpy = inv.get_package("numpy")
        assert numpy is not None
        assert numpy.name == "numpy"
        assert numpy.version == "1.26.4"
        assert numpy.installer == "pip"

    def test_parse_requires_dist(self, pip_inspect_output_minimal):
        from compatibillabuddy.hardware.inspector import parse_pip_inspect

        inv = parse_pip_inspect(pip_inspect_output_minimal)

        pandas = inv.get_package("pandas")
        assert pandas is not None
        assert len(pandas.requires) == 4
        assert "numpy>=1.23.2" in pandas.requires

    def test_parse_ml_stack(self, pip_inspect_output_ml_stack):
        from compatibillabuddy.hardware.inspector import parse_pip_inspect

        inv = parse_pip_inspect(pip_inspect_output_ml_stack)

        assert len(inv.packages) == 3
        torch = inv.get_package("torch")
        assert torch is not None
        assert torch.version == "2.4.0"
        assert any("nvidia-cuda-nvrtc" in r for r in torch.requires)

    def test_parse_empty_env(self, pip_inspect_output_empty):
        from compatibillabuddy.hardware.inspector import parse_pip_inspect

        inv = parse_pip_inspect(pip_inspect_output_empty)

        assert len(inv.packages) == 0
        assert inv.python_version == "3.12.0"

    def test_parse_invalid_json_raises(self):
        from compatibillabuddy.hardware.inspector import parse_pip_inspect

        with pytest.raises(ValueError, match="Failed to parse"):
            parse_pip_inspect("this is not json {{{")

    def test_parse_missing_environment_key(self):
        from compatibillabuddy.hardware.inspector import parse_pip_inspect

        bad_json = json.dumps({"version": "1", "installed": []})
        with pytest.raises(ValueError, match="environment"):
            parse_pip_inspect(bad_json)


# ---------------------------------------------------------------------------
# Subprocess invocation tests (mocked)
# ---------------------------------------------------------------------------


class TestInspectEnvironment:
    """Test the full inspect_environment() function with mocked subprocess."""

    def test_inspect_calls_pip(self, pip_inspect_output_minimal):
        from compatibillabuddy.hardware.inspector import inspect_environment

        result = MagicMock()
        result.returncode = 0
        result.stdout = pip_inspect_output_minimal

        with patch("subprocess.run", return_value=result) as mock_run:
            inspect_environment()

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "pip" in cmd
        assert "inspect" in cmd

    def test_inspect_returns_inventory(self, pip_inspect_output_minimal):
        from compatibillabuddy.hardware.inspector import inspect_environment

        result = MagicMock()
        result.returncode = 0
        result.stdout = pip_inspect_output_minimal

        with patch("subprocess.run", return_value=result):
            inv = inspect_environment()

        assert isinstance(inv, EnvironmentInventory)
        assert len(inv.packages) == 2

    def test_inspect_pip_not_found(self):
        from compatibillabuddy.hardware.inspector import inspect_environment

        with (
            patch("subprocess.run", side_effect=FileNotFoundError),
            pytest.raises(RuntimeError, match="pip"),
        ):
            inspect_environment()

    def test_inspect_pip_fails(self):
        from compatibillabuddy.hardware.inspector import inspect_environment

        result = MagicMock()
        result.returncode = 1
        result.stderr = "some error"

        with (
            patch("subprocess.run", return_value=result),
            pytest.raises(RuntimeError, match="pip inspect"),
        ):
            inspect_environment()

    def test_inspect_custom_python(self, pip_inspect_output_minimal):
        """Should allow specifying a custom Python executable."""
        from compatibillabuddy.hardware.inspector import inspect_environment

        result = MagicMock()
        result.returncode = 0
        result.stdout = pip_inspect_output_minimal

        with patch("subprocess.run", return_value=result) as mock_run:
            inspect_environment(python_executable="/path/to/venv/bin/python")

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/path/to/venv/bin/python"


# ---------------------------------------------------------------------------
# Integration test: runs real pip inspect on THIS environment
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestInspectEnvironmentReal:
    """Run the real environment inspector."""

    def test_real_inspect_returns_inventory(self):
        from compatibillabuddy.hardware.inspector import inspect_environment

        inv = inspect_environment()

        assert isinstance(inv, EnvironmentInventory)
        assert len(inv.python_version) > 0
        assert len(inv.packages) > 0

    def test_real_inspect_finds_pytest(self):
        """pytest must be installed since we're running it."""
        from compatibillabuddy.hardware.inspector import inspect_environment

        inv = inspect_environment()
        pkg = inv.get_package("pytest")
        assert pkg is not None

    def test_real_inspect_finds_self(self):
        """compatibillabuddy itself should be installed (editable)."""
        from compatibillabuddy.hardware.inspector import inspect_environment

        inv = inspect_environment()
        pkg = inv.get_package("compatibillabuddy")
        assert pkg is not None
        assert pkg.version == "0.1.0"
