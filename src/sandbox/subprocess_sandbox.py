"""Subprocess-based sandbox for code execution."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

from src.config import get_settings
from src.sandbox.base import ExecutionResult, SandboxProvider

logger = logging.getLogger(__name__)

# Limit stdout/stderr capture to 50KB each to prevent memory blowup
MAX_OUTPUT_BYTES = 50 * 1024


class SubprocessSandbox(SandboxProvider):
    """Code execution sandbox using subprocess with venv isolation.

    The sandbox runs INSIDE the Docker container. We're not doing Docker-in-Docker.
    The isolation is: separate venv + subprocess with timeout + resource limits.
    """

    def __init__(self, runs_dir: Optional[Path] = None) -> None:
        self._runs_dir = runs_dir or get_settings().runs_dir
        # Resolve the Python executable — `python` may not exist on all systems
        self._python = shutil.which("python3") or shutil.which("python") or sys.executable

    async def execute(
        self,
        command: str,
        cwd: Path,
        timeout: int = 120,
        env: Optional[dict[str, str]] = None,
    ) -> ExecutionResult:
        """Run a command in the sandbox, capture all output."""
        start = time.perf_counter()

        # Build environment: merge os.environ with provided env, add venv to PATH
        exec_env = os.environ.copy()
        venv_path = cwd / ".venv"
        if venv_path.exists():
            exec_env["PATH"] = f"{venv_path / 'bin'}:{exec_env.get('PATH', '')}"
            exec_env["VIRTUAL_ENV"] = str(venv_path)
        if env:
            exec_env.update(env)

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(cwd),
                env=exec_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                # Kill the process tree on timeout
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
                duration_ms = int((time.perf_counter() - start) * 1000)
                logger.warning(
                    f"Command timed out after {timeout}s: {command[:100]}"
                )
                return ExecutionResult(
                    stdout="",
                    stderr=f"Execution timed out after {timeout} seconds",
                    exit_code=-1,
                    duration_ms=duration_ms,
                    timed_out=True,
                )

            duration_ms = int((time.perf_counter() - start) * 1000)

            # Truncate output to prevent memory blowup
            stdout = stdout_bytes[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
            stderr = stderr_bytes[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")

            # Collect artifacts (files created/modified in workspace)
            artifacts = self._collect_artifacts(cwd)

            logger.info(
                f"Command completed: exit_code={process.returncode}, "
                f"duration={duration_ms}ms, command={command[:100]}"
            )

            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=process.returncode or 0,
                duration_ms=duration_ms,
                artifacts=artifacts,
            )

        except Exception as e:
            duration_ms = int((time.perf_counter() - start) * 1000)
            logger.error(f"Execution failed: {e}")
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                exit_code=-1,
                duration_ms=duration_ms,
            )

    async def install_packages(
        self, packages: list[str], cwd: Path
    ) -> ExecutionResult:
        """Install Python packages in the sandbox venv."""
        venv_pip = cwd / ".venv" / "bin" / "pip"
        if not venv_pip.exists():
            # Create venv if it doesn't exist yet
            await self.execute(f"{self._python} -m venv {cwd / '.venv'}", cwd=cwd)

        package_str = " ".join(packages)
        return await self.execute(
            f"{venv_pip} install {package_str} --quiet",
            cwd=cwd,
            timeout=120,
        )

    async def setup_workspace(self, run_id: str) -> Path:
        """Create an isolated workspace directory for a run."""
        workspace = self._runs_dir / run_id / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)

        # Create a Python venv inside the workspace, inheriting system packages
        # to avoid reinstalling large deps (yfinance, matplotlib, etc.) each run.
        result = await self.execute(
            f"{self._python} -m venv --system-site-packages {workspace / '.venv'}",
            cwd=workspace,
            timeout=60,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to create venv for {run_id}: {result.stderr}")

        logger.info(f"Workspace created: {workspace}")
        return workspace

    async def cleanup_workspace(self, run_id: str) -> None:
        """Clean up a workspace directory."""
        workspace = self._runs_dir / run_id / "workspace"
        if workspace.exists():
            try:
                shutil.rmtree(workspace)
                logger.info(f"Workspace cleaned up: {workspace}")
            except Exception as e:
                logger.warning(f"Failed to clean up workspace {workspace}: {e}")
        else:
            logger.warning(f"Workspace not found for cleanup: {workspace}")

    @staticmethod
    def _collect_artifacts(cwd: Path) -> list[Path]:
        """Collect artifact file paths from the workspace."""
        artifacts: list[Path] = []
        artifacts_dir = cwd.parent / "artifacts" if cwd.name == "workspace" else cwd / "artifacts"
        if artifacts_dir.exists():
            for path in artifacts_dir.rglob("*"):
                if path.is_file():
                    artifacts.append(path)
        return artifacts
