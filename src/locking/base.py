"""Path locker abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from src.models.decisions import DecisionPoint
from src.models.enums import DecisionCategory


class PathLocker(ABC):
    """Abstract base for path locking mechanisms.

    Locks decisions from a source trace so that a new run follows the same
    decision path. Supports full lock, partial lock (by IDs, sequence, or
    category), and fork lock (override one decision, lock the rest).
    """

    @abstractmethod
    def is_locked(
        self,
        question: str,
        category: DecisionCategory,
        sequence_number: int,
    ) -> bool:
        """Check if a decision at this point should be locked.

        Match by question text similarity, not just ID (because IDs may
        differ across runs). Uses normalized question matching.

        Args:
            question: The decision question text.
            category: The decision category.
            sequence_number: The ordinal position in the run.

        Returns:
            True if the decision is locked.
        """
        ...

    @abstractmethod
    def get_locked_value(
        self,
        question: str,
        category: DecisionCategory,
    ) -> Optional[str]:
        """Get the value to lock to. Returns None if not locked.

        Args:
            question: The decision question text.
            category: The decision category.

        Returns:
            The locked value, or None if not locked.
        """
        ...

    @abstractmethod
    def get_lock_prompt(self, question: str, locked_value: str) -> str:
        """Generate prompt injection text for a locked decision.

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

    @abstractmethod
    def get_all_lock_prompts(self) -> str:
        """Generate combined prompt injection text for ALL locked decisions.

        Used to inject into the planning prompt upfront so the LLM sees
        all constraints at once.

        Returns:
            Combined prompt text for all locked decisions.
        """
        ...

    @property
    @abstractmethod
    def verification_failures(self) -> list[dict]:
        """Return list of verification failures encountered during the run."""
        ...
