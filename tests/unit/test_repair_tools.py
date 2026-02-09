"""Tests for agent repair tools — snapshot, pip execution, verify, rollback.

TDD: tests written BEFORE implementation in agent/tools.py.
All tests mock subprocess/system calls to avoid side effects.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FAKE_FREEZE = "torch==2.1.0\nnumpy==1.26.4\npandas==2.2.0\n"

_FAKE_SNAPSHOT = {
    "timestamp": "2025-01-15T10:30:00",
    "packages": ["torch==2.1.0", "numpy==1.26.4", "pandas==2.2.0"],
}


def _make_diagnosis_dict(*, num_issues: int = 2) -> dict:
    """Build a minimal diagnosis dict for testing verify_fix."""
    issues = []
    for i in range(num_issues):
        issues.append(
            {
                "severity": 1,
                "category": f"issue-{i}",
                "description": f"Test issue {i}",
                "affected_packages": [f"pkg-{i}"],
                "fix_suggestion": f"fix {i}",
            }
        )
    return {
        "hardware": {
            "os_name": "Linux",
            "os_version": "6.5.0",
            "cpu_arch": "x86_64",
            "cpu_name": "Test CPU",
            "python_version": "3.11.7",
            "gpus": [],
        },
        "environment": {
            "python_version": "3.11.7",
            "python_executable": "/usr/bin/python3.11",
            "packages": [],
        },
        "issues": issues,
        "hardware_probe_seconds": 0.01,
        "environment_inspect_seconds": 0.01,
        "rule_evaluation_seconds": 0.01,
        "total_seconds": 0.03,
    }


# ===========================================================================
# tool_snapshot_environment
# ===========================================================================


class TestToolSnapshotEnvironment:
    """Tests for tool_snapshot_environment()."""

    def test_snapshot_returns_dict(self):
        from compatibillabuddy.agent.tools import tool_snapshot_environment

        with patch("compatibillabuddy.agent.tools.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=_FAKE_FREEZE, stderr="")
            result = tool_snapshot_environment()

        assert isinstance(result, dict)
        assert "packages" in result
        assert "timestamp" in result

    def test_snapshot_captures_packages(self):
        from compatibillabuddy.agent.tools import tool_snapshot_environment

        with patch("compatibillabuddy.agent.tools.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=_FAKE_FREEZE, stderr="")
            result = tool_snapshot_environment()

        assert result["packages"] == [
            "torch==2.1.0",
            "numpy==1.26.4",
            "pandas==2.2.0",
        ]


# ===========================================================================
# tool_run_pip
# ===========================================================================


class TestToolRunPip:
    """Tests for tool_run_pip()."""

    def setup_method(self):
        """Reset the operation counter before each test."""
        import compatibillabuddy.agent.tools as tools_mod

        tools_mod._pip_op_count = 0

    def test_run_pip_install_success(self):
        from compatibillabuddy.agent.tools import tool_run_pip

        with (
            patch("compatibillabuddy.agent.tools.sys") as mock_sys,
            patch("compatibillabuddy.agent.tools.subprocess.run") as mock_run,
        ):
            mock_sys.prefix = "/home/user/myenv"
            mock_sys.base_prefix = "/usr"
            mock_run.return_value = MagicMock(
                returncode=0, stdout="Successfully installed torch-2.1.0", stderr=""
            )
            result = tool_run_pip(action="install", package="torch==2.1.0", dry_run=False)

        assert result["success"] is True
        assert result["action"] == "install"
        assert result["package"] == "torch==2.1.0"

    def test_run_pip_rejects_system_python(self):
        from compatibillabuddy.agent.tools import tool_run_pip

        with patch("compatibillabuddy.agent.tools.sys") as mock_sys:
            mock_sys.prefix = "/usr"
            mock_sys.base_prefix = "/usr"
            with patch.dict("os.environ", {}, clear=True):
                result = tool_run_pip(action="install", package="torch==2.1.0", dry_run=False)

        assert result["success"] is False
        assert "venv" in result["error"].lower() or "virtual" in result["error"].lower()

    def test_run_pip_rejects_blocklisted_package(self):
        from compatibillabuddy.agent.tools import tool_run_pip

        with patch("compatibillabuddy.agent.tools.sys") as mock_sys:
            mock_sys.prefix = "/home/user/myenv"
            mock_sys.base_prefix = "/usr"
            result = tool_run_pip(action="uninstall", package="pip", dry_run=False)

        assert result["success"] is False
        assert "blocklist" in result["error"].lower() or "protected" in result["error"].lower()

    def test_run_pip_rejects_invalid_action(self):
        from compatibillabuddy.agent.tools import tool_run_pip

        with patch("compatibillabuddy.agent.tools.sys") as mock_sys:
            mock_sys.prefix = "/home/user/myenv"
            mock_sys.base_prefix = "/usr"
            result = tool_run_pip(action="upgrade", package="torch", dry_run=False)

        assert result["success"] is False
        assert "action" in result["error"].lower()

    def test_run_pip_dry_run(self):
        from compatibillabuddy.agent.tools import tool_run_pip

        with patch("compatibillabuddy.agent.tools.sys") as mock_sys:
            mock_sys.prefix = "/home/user/myenv"
            mock_sys.base_prefix = "/usr"
            mock_sys.executable = "/home/user/myenv/bin/python"
            result = tool_run_pip(action="install", package="torch==2.1.0", dry_run=True)

        assert result["dry_run"] is True
        assert "command" in result

    def test_run_pip_max_operations(self):
        import compatibillabuddy.agent.tools as tools_mod
        from compatibillabuddy.agent.tools import tool_run_pip

        tools_mod._pip_op_count = tools_mod._MAX_PIP_OPS

        with patch("compatibillabuddy.agent.tools.sys") as mock_sys:
            mock_sys.prefix = "/home/user/myenv"
            mock_sys.base_prefix = "/usr"
            result = tool_run_pip(action="install", package="torch==2.1.0", dry_run=False)

        assert result["success"] is False
        assert "limit" in result["error"].lower() or "max" in result["error"].lower()


# ===========================================================================
# tool_verify_fix
# ===========================================================================


class TestToolVerifyFix:
    """Tests for tool_verify_fix()."""

    def test_verify_fix_issues_reduced(self):
        from compatibillabuddy.agent.tools import tool_verify_fix

        previous = _make_diagnosis_dict(num_issues=3)
        current = _make_diagnosis_dict(num_issues=1)

        with patch("compatibillabuddy.agent.tools.diagnose") as mock_diag:
            from compatibillabuddy.engine.models import DiagnosisResult

            mock_diag.return_value = DiagnosisResult.model_validate(current)
            result = tool_verify_fix(previous_diagnosis_json=json.dumps(previous))

        assert result["previous_issue_count"] == 3
        assert result["current_issue_count"] == 1
        assert result["issues_resolved"] == 2
        assert result["improved"] is True

    def test_verify_fix_new_issues(self):
        from compatibillabuddy.agent.tools import tool_verify_fix

        previous = _make_diagnosis_dict(num_issues=1)
        current = _make_diagnosis_dict(num_issues=3)

        with patch("compatibillabuddy.agent.tools.diagnose") as mock_diag:
            from compatibillabuddy.engine.models import DiagnosisResult

            mock_diag.return_value = DiagnosisResult.model_validate(current)
            result = tool_verify_fix(previous_diagnosis_json=json.dumps(previous))

        assert result["previous_issue_count"] == 1
        assert result["current_issue_count"] == 3
        assert result["new_issues"] == 2
        assert result["improved"] is False


# ===========================================================================
# tool_rollback
# ===========================================================================


class TestToolRollback:
    """Tests for tool_rollback()."""

    def test_rollback_restores_packages(self):
        from compatibillabuddy.agent.tools import tool_rollback

        with (
            patch("compatibillabuddy.agent.tools.sys") as mock_sys,
            patch("compatibillabuddy.agent.tools.subprocess.run") as mock_run,
        ):
            mock_sys.prefix = "/home/user/myenv"
            mock_sys.base_prefix = "/usr"
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = tool_rollback(snapshot_json=json.dumps(_FAKE_SNAPSHOT))

        assert result["success"] is True
        assert result["restored_count"] == 3
        # Verify pip install was called with the snapshot packages
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        cmd = call_args[0][0] if call_args[0] else call_args[1].get("args", [])
        assert "install" in cmd

    def test_rollback_rejects_system_python(self):
        from compatibillabuddy.agent.tools import tool_rollback

        with patch("compatibillabuddy.agent.tools.sys") as mock_sys:
            mock_sys.prefix = "/usr"
            mock_sys.base_prefix = "/usr"
            with patch.dict("os.environ", {}, clear=True):
                result = tool_rollback(snapshot_json=json.dumps(_FAKE_SNAPSHOT))

        assert result["success"] is False
        assert "venv" in result["error"].lower() or "virtual" in result["error"].lower()


# ===========================================================================
# JSON serializability for repair tools
# ===========================================================================


class TestRepairToolsJsonSerializable:
    """All repair tool outputs must be json.dumps()-able."""

    def setup_method(self):
        import compatibillabuddy.agent.tools as tools_mod

        tools_mod._pip_op_count = 0

    def test_all_repair_tools_json_serializable(self):
        from compatibillabuddy.agent.tools import (
            tool_rollback,
            tool_run_pip,
            tool_snapshot_environment,
            tool_verify_fix,
        )

        # snapshot
        with patch("compatibillabuddy.agent.tools.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=_FAKE_FREEZE, stderr="")
            snap = tool_snapshot_environment()
        assert json.dumps(snap)

        # run_pip (dry run — no subprocess needed)
        with patch("compatibillabuddy.agent.tools.sys") as mock_sys:
            mock_sys.prefix = "/home/user/myenv"
            mock_sys.base_prefix = "/usr"
            mock_sys.executable = "/home/user/myenv/bin/python"
            pip_result = tool_run_pip(action="install", package="torch", dry_run=True)
        assert json.dumps(pip_result)

        # verify_fix
        previous = _make_diagnosis_dict(num_issues=2)
        current = _make_diagnosis_dict(num_issues=1)
        with patch("compatibillabuddy.agent.tools.diagnose") as mock_diag:
            from compatibillabuddy.engine.models import DiagnosisResult

            mock_diag.return_value = DiagnosisResult.model_validate(current)
            verify = tool_verify_fix(previous_diagnosis_json=json.dumps(previous))
        assert json.dumps(verify)

        # rollback (rejected — system python, still must be serializable)
        with patch("compatibillabuddy.agent.tools.sys") as mock_sys:
            mock_sys.prefix = "/usr"
            mock_sys.base_prefix = "/usr"
            with patch.dict("os.environ", {}, clear=True):
                rb = tool_rollback(snapshot_json=json.dumps(_FAKE_SNAPSHOT))
        assert json.dumps(rb)
