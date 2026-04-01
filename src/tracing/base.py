"""Decision tracer abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from src.models.decisions import DecisionPoint, ExecutionRecord, LLMCallRecord
from src.models.enums import RunStatus
from src.models.traces import DecisionTrace


class DecisionTracer(ABC):
    """Abstract base for decision trace capture."""

    @abstractmethod
    async def record_decision(self, decision: DecisionPoint) -> None:
        """Record a decision point.

        Args:
            decision: The decision point to record.
        """
        ...

    @abstractmethod
    async def record_execution(self, execution: ExecutionRecord) -> None:
        """Record an execution attempt.

        Args:
            execution: The execution record to store.
        """
        ...

    @abstractmethod
    async def record_llm_call(self, call: LLMCallRecord) -> None:
        """Record an LLM API call.

        Args:
            call: The LLM call record to store.
        """
        ...

    @abstractmethod
    async def get_decisions(self) -> list[DecisionPoint]:
        """Get all recorded decisions.

        Returns:
            List of decision points in order.
        """
        ...

    @abstractmethod
    async def finalize(
        self, status: RunStatus, error: Optional[str] = None
    ) -> DecisionTrace:
        """Finalize the trace and return the complete DecisionTrace.

        Args:
            status: Final status of the run.
            error: Error message if failed.

        Returns:
            The complete decision trace.
        """
        ...
