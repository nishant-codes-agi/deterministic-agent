"""Agent abstract base class and event models."""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional, Protocol

from pydantic import BaseModel, Field

from src.models.enums import AgentPhase
from src.models.traces import DecisionTrace, PathLockConfig


class AgentEventType(str, enum.Enum):
    """Types of agent events for observability."""

    RUN_STARTED = "run_started"
    PHASE_CHANGED = "phase_changed"
    DECISION_MADE = "decision_made"
    CODE_WRITTEN = "code_written"
    EXECUTION_STARTED = "execution_started"
    EXECUTION_COMPLETED = "execution_completed"
    ERROR_HIT = "error_hit"
    RECOVERY_STARTED = "recovery_started"
    RUN_COMPLETED = "run_completed"
    COST_UPDATE = "cost_update"


class AgentEvent(BaseModel):
    """An event emitted during agent execution."""

    event_type: AgentEventType = Field(description="Type of event")
    run_id: str = Field(description="Run ID this event belongs to")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the event occurred",
    )
    phase: Optional[AgentPhase] = Field(default=None, description="Current agent phase")
    payload: dict = Field(default_factory=dict, description="Event-specific data")


class EventHandler(Protocol):
    """Protocol for handling agent events."""

    async def handle(self, event: AgentEvent) -> None:
        """Handle an agent event.

        Args:
            event: The agent event to handle.
        """
        ...


class CodingAgent(ABC):
    """Abstract base for the coding agent."""

    @abstractmethod
    async def run(
        self,
        task: str,
        lock_config: Optional[PathLockConfig] = None,
        event_handler: Optional[EventHandler] = None,
    ) -> DecisionTrace:
        """Execute the agent on a task.

        Args:
            task: Natural language task description.
            lock_config: Optional path locking configuration.
            event_handler: Optional event handler for observability.

        Returns:
            Complete decision trace of the run.
        """
        ...
