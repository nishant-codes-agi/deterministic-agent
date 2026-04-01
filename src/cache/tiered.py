"""Tiered cache orchestrating L1 (memory) -> L2 (Redis) -> L3 (filesystem)."""

from __future__ import annotations

import logging
from typing import Optional

from src.cache.filesystem_cache import FilesystemCache
from src.cache.memory_cache import InMemoryCache
from src.cache.redis_cache import RedisCache

logger = logging.getLogger(__name__)


class TieredCache:
    """Three-tier cache: in-memory (L1) -> Redis (L2) -> filesystem (L3).

    - get(): Check L1, then L2, then L3. Backfill higher tiers on hit.
    - set(): Always write to L1 + L2. Write to L3 only if persist=True.
    - delete(): Delete from all tiers.

    Constructor takes optional Redis -- if None, L2 falls back to a second
    InMemoryCache instance (for when Redis is unavailable).
    """

    def __init__(
        self,
        memory: Optional[InMemoryCache] = None,
        redis: Optional[RedisCache] = None,
        filesystem: Optional[FilesystemCache] = None,
    ) -> None:
        self._l1 = memory or InMemoryCache()
        # If Redis is unavailable, use a second memory cache as L2
        self._l2: InMemoryCache | RedisCache = redis or InMemoryCache()
        self._l3 = filesystem or FilesystemCache()
        self._has_redis = redis is not None

    async def get(self, key: str) -> Optional[str]:
        """Get from the cache, checking tiers in order."""
        # L1: memory
        value = await self._l1.get(key)
        if value is not None:
            logger.debug(f"Tiered cache L1 hit: {key}")
            return value

        # L2: Redis (or fallback memory)
        value = await self._l2.get(key)
        if value is not None:
            logger.debug(f"Tiered cache L2 hit: {key}")
            # Backfill L1
            await self._l1.set(key, value)
            return value

        # L3: filesystem
        value = await self._l3.get(key)
        if value is not None:
            logger.debug(f"Tiered cache L3 hit: {key}")
            # Backfill L1 + L2
            await self._l1.set(key, value)
            await self._l2.set(key, value)
            return value

        return None

    async def set(
        self, key: str, value: str, ttl: Optional[int] = None, persist: bool = False
    ) -> None:
        """Set in the cache. Always writes L1+L2. Writes L3 if persist=True."""
        await self._l1.set(key, value, ttl=ttl)
        await self._l2.set(key, value, ttl=ttl)
        if persist:
            await self._l3.set(key, value)

    async def delete(self, key: str) -> None:
        """Delete from all tiers."""
        await self._l1.delete(key)
        await self._l2.delete(key)
        await self._l3.delete(key)

    async def exists(self, key: str) -> bool:
        """Check if a key exists in any tier."""
        return (
            await self._l1.exists(key)
            or await self._l2.exists(key)
            or await self._l3.exists(key)
        )
