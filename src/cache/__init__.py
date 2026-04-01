"""Cache backend protocol and interface."""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class CacheBackend(Protocol):
    """Protocol for cache backends."""

    async def get(self, key: str) -> Optional[str]:
        """Get a value by key."""
        ...

    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Set a value with optional TTL."""
        ...

    async def delete(self, key: str) -> None:
        """Delete a key."""
        ...

    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        ...
