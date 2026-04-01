"""Sandbox provider abstract base class and data models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class ExecutionResult(BaseModel):
    """Result of a sandbox code execution."""

    stdout: str = Field(default="", description="Standard output")
    stderr: str = Field(default="", description="Standard error")
    exit_code: int = Field(description="Process exit code")
    duration_ms: int = Field(description="Execution duration in milliseconds")
    artifacts: list[Path] = Field(default_factory=list, description="Produced artifact paths")
    timed_out: bool = Field(default=False, description="Whether execution timed out")


class SandboxProvider(ABC):
    """Abstract base for code execution sandboxes."""

    @abstractmethod
    async def execute(
        self,
        command: str,
        cwd: Path,
        timeout: int = 120,
        env: Optional[dict[str, str]] = None,
    ) -> ExecutionResult:
        """Execute a command in the sandbox.

        Args:
            command: Shell command to execute.
            cwd: Working directory.
            timeout: Timeout in seconds.
            env: Additional environment variables.
        """
        ...

    @abstractmethod
    async def install_packages(
        self, packages: list[str], cwd: Path
    ) -> ExecutionResult:
        """Install Python packages in the sandbox.

        Args:
            packages: List of pip package specifiers.
            cwd: Working directory (with venv).
        """
        ...

    @abstractmethod
    async def setup_workspace(self, run_id: str) -> Path:
        """Create an isolated workspace directory for a run.

        Args:
            run_id: Unique run identifier.

        Returns:
            Path to the workspace directory.
        """
        ...

    @abstractmethod
    async def cleanup_workspace(self, run_id: str) -> None:
        """Clean up a workspace directory.

        Args:
            run_id: Unique run identifier.
        """
        ...
