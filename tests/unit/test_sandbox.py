"""Tests for subprocess sandbox."""

from __future__ import annotations

import pytest

from src.sandbox.subprocess_sandbox import SubprocessSandbox


@pytest.fixture
def sandbox(tmp_path):
    return SubprocessSandbox(runs_dir=tmp_path)


class TestSubprocessSandbox:
    async def test_execute_simple_command(self, sandbox, tmp_path):
        """Run 'echo hello', verify stdout."""
        result = await sandbox.execute("echo hello", cwd=tmp_path)
        assert result.exit_code == 0
        assert result.stdout.strip() == "hello"
        assert result.timed_out is False

    async def test_execute_timeout(self, sandbox, tmp_path):
        """Run 'sleep 10' with 1s timeout, verify timed_out=True."""
        result = await sandbox.execute("sleep 10", cwd=tmp_path, timeout=1)
        assert result.timed_out is True
        assert result.exit_code == -1

    async def test_workspace_creation_and_cleanup(self, sandbox):
        """Create workspace, verify exists, cleanup, verify gone."""
        workspace = await sandbox.setup_workspace("test-run-123")
        assert workspace.exists()
        assert (workspace / ".venv").exists()

        await sandbox.cleanup_workspace("test-run-123")
        assert not workspace.exists()

    async def test_execute_captures_stderr(self, sandbox, tmp_path):
        """Verify stderr is captured."""
        result = await sandbox.execute(
            "python -c \"import sys; sys.stderr.write('error msg')\"",
            cwd=tmp_path,
        )
        assert "error msg" in result.stderr

    async def test_execute_nonzero_exit_code(self, sandbox, tmp_path):
        """Verify non-zero exit code is captured."""
        result = await sandbox.execute("exit 42", cwd=tmp_path)
        assert result.exit_code == 42

    async def test_install_packages(self, sandbox):
        """Verify install_packages runs pip in the venv."""
        workspace = await sandbox.setup_workspace("test-pip-run")
        try:
            result = await sandbox.install_packages(["pip"], cwd=workspace)
            # pip installing itself should succeed
            assert result.exit_code == 0
        finally:
            await sandbox.cleanup_workspace("test-pip-run")

    async def test_cleanup_nonexistent_workspace(self, sandbox):
        """Cleanup a non-existent workspace should not raise."""
        await sandbox.cleanup_workspace("nonexistent-run-id")
