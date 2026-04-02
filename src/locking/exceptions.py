"""Exceptions for the path locking system."""

from __future__ import annotations


class PathLockViolationError(Exception):
    """Raised when LLM ignores a locked decision after max retries.

    This means the LLM repeatedly chose a different value than the one
    specified by the lock, even after increasingly forceful prompt injection.
    """

    def __init__(
        self, question: str, expected: str, got: str, attempts: int
    ) -> None:
        self.question = question
        self.expected = expected
        self.got = got
        self.attempts = attempts
        super().__init__(
            f"Path lock violation: for '{question}', "
            f"expected '{expected}' but got '{got}' after {attempts} attempts"
        )


class PathLockDivergenceError(Exception):
    """Raised when a locked choice is no longer valid at runtime.

    For example, if the locked decision was to use a library that has since
    been removed, or an API endpoint that no longer exists.
    """

    def __init__(
        self, question: str, locked_value: str, reason: str
    ) -> None:
        self.question = question
        self.locked_value = locked_value
        self.reason = reason
        super().__init__(
            f"Path lock divergence: for '{question}', "
            f"locked value '{locked_value}' is no longer valid: {reason}"
        )
