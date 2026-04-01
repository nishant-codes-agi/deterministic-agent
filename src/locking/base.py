"""Path locker abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from src.models.decisions import DecisionPoint


class PathLocker(ABC):
    """Abstract base for path locking mechanisms."""

    @abstractmethod
    def is_locked(self, decision_id: str) -> bool:
        """Check if a decision is locked.

        Args:
            decision_id: The decision point ID to check.

        Returns:
            True if the decision is locked.
        """
        ...

    @abstractmethod
    def get_locked_value(self, decision_id: str) -> Optional[str]:
        """Get the locked value for a decision.

        Args:
            decision_id: The decision point ID.

        Returns:
            The locked value, or None if not locked.
        """
        ...

    @abstractmethod
    def get_lock_prompt(self, question: str, locked_value: str) -> str:
        """Generate the prompt injection text for a locked decision.

        Args:
            question: The decision question.
            locked_value: The value to lock to.

        Returns:
            Prompt text to inject.
        """
        ...

    @abstractmethod
    def verify_decision(self, decision: DecisionPoint) -> bool:
        """Verify that a decision matches the locked value.

        Returns True if the decision matches the lock or is unlocked.

        Args:
            decision: The decision point to verify.

        Returns:
            True if valid.
        """
        ...
