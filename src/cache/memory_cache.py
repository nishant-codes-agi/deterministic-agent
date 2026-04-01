"""In-memory LRU cache implementing CacheBackend."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from collections import OrderedDict
from typing import Optional

logger = logging.getLogger(__name__)


class InMemoryCache:
    """In-memory LRU cache with TTL support.

    Uses OrderedDict for LRU behavior. Thread-safe via asyncio.Lock.
    Stores (value, expires_at) tuples, lazy-expires on get.
    """

    def __init__(self, max_size_mb: int = 100) -> None:
        self._store: OrderedDict[str, tuple[str, Optional[float]]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._current_size_bytes = 0

    async def get(self, key: str) -> Optional[str]:
        """Get a value by key. Returns None if expired or missing."""
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None

            value, expires_at = entry
            # Lazy expiry check
            if expires_at is not None and time.time() > expires_at:
                del self._store[key]
                self._current_size_bytes -= self._estimate_size(key, value)
                logger.debug(f"Cache key expired: {key}")
                return None

            # Move to end for LRU
            self._store.move_to_end(key)
            logger.debug(f"Cache hit: {key}")
            return value

    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Set a value with optional TTL in seconds."""
        expires_at = (time.time() + ttl) if ttl else None
        entry_size = self._estimate_size(key, value)

        async with self._lock:
            # Remove old entry if exists
            if key in self._store:
                old_value, _ = self._store[key]
                self._current_size_bytes -= self._estimate_size(key, old_value)
                del self._store[key]

            # Evict LRU entries if over max size
            while self._current_size_bytes + entry_size > self._max_size_bytes and self._store:
                evicted_key, (evicted_value, _) = self._store.popitem(last=False)
                self._current_size_bytes -= self._estimate_size(evicted_key, evicted_value)
                logger.debug(f"Cache evicted: {evicted_key}")

            self._store[key] = (value, expires_at)
            self._current_size_bytes += entry_size
            self._store.move_to_end(key)

    async def delete(self, key: str) -> None:
        """Delete a key."""
        async with self._lock:
            entry = self._store.pop(key, None)
            if entry is not None:
                value, _ = entry
                self._current_size_bytes -= self._estimate_size(key, value)

    async def exists(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
        value = await self.get(key)
        return value is not None

    @staticmethod
    def _estimate_size(key: str, value: str) -> int:
        """Estimate memory size of a key-value pair."""
        return sys.getsizeof(key) + sys.getsizeof(value)
