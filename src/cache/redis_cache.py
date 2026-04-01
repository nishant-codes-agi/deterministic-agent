"""Redis cache implementing CacheBackend."""

from __future__ import annotations

import json
import logging
from typing import Optional

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# Namespace prefix for all keys to avoid collisions
KEY_PREFIX = "dagent:"


class RedisCache:
    """Redis cache with graceful degradation.

    All operations have try/except -- on ConnectionError, log warning and return
    None/False. The agent should never crash because Redis is down.
    """

    def __init__(self, redis_url: str) -> None:
        self._client = aioredis.from_url(redis_url, decode_responses=True)

    def _key(self, key: str) -> str:
        """Prefix key for namespacing."""
        return f"{KEY_PREFIX}{key}"

    async def get(self, key: str) -> Optional[str]:
        """Get a value by key."""
        try:
            value = await self._client.get(self._key(key))
            if value is not None:
                logger.debug(f"Redis cache hit: {key}")
            return value
        except (aioredis.ConnectionError, OSError) as e:
            logger.warning(f"Redis get failed: {e}")
            return None

    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Set a value with optional TTL using SETEX."""
        try:
            if ttl:
                await self._client.setex(self._key(key), ttl, value)
            else:
                await self._client.set(self._key(key), value)
        except (aioredis.ConnectionError, OSError) as e:
            logger.warning(f"Redis set failed: {e}")

    async def delete(self, key: str) -> None:
        """Delete a key."""
        try:
            await self._client.delete(self._key(key))
        except (aioredis.ConnectionError, OSError) as e:
            logger.warning(f"Redis delete failed: {e}")

    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        try:
            return bool(await self._client.exists(self._key(key)))
        except (aioredis.ConnectionError, OSError) as e:
            logger.warning(f"Redis exists failed: {e}")
            return False

    async def increment_cost(self, run_id: str, amount: float) -> float:
        """Atomically increment the cost counter for a run."""
        try:
            result = await self._client.incrbyfloat(
                self._key(f"cost:{run_id}"), amount
            )
            return float(result)
        except (aioredis.ConnectionError, OSError) as e:
            logger.warning(f"Redis increment_cost failed: {e}")
            return 0.0

    async def publish_event(self, channel: str, event_json: str) -> None:
        """Publish an event for streaming."""
        try:
            await self._client.publish(self._key(channel), event_json)
        except (aioredis.ConnectionError, OSError) as e:
            logger.warning(f"Redis publish failed: {e}")

    async def ping(self) -> bool:
        """Health check."""
        try:
            return bool(await self._client.ping())
        except (aioredis.ConnectionError, OSError) as e:
            logger.warning(f"Redis ping failed: {e}")
            return False

    async def close(self) -> None:
        """Close the Redis connection."""
        try:
            await self._client.close()
        except Exception as e:
            logger.warning(f"Redis close failed: {e}")
