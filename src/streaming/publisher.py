"""Redis-backed event publisher and multi-handler compositor."""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from src.agent.base import AgentEvent, EventHandler

logger = logging.getLogger(__name__)


class RedisEventPublisher:
    """Publishes agent events to Redis pub/sub for real-time streaming.

    Channel pattern: ``dagent:stream:{run_id}``

    Gracefully degrades if Redis is unavailable — events are silently
    dropped so the agent run is never interrupted by streaming failures.
    """

    def __init__(self, redis_cache) -> None:
        """Initialize with a RedisCache instance (may be None)."""
        self._redis = redis_cache

    async def handle(self, event: AgentEvent) -> None:
        """Publish event JSON to the run's Redis channel."""
        if self._redis is None:
            return
        channel = f"stream:{event.run_id}"
        try:
            await self._redis.publish_event(channel, event.model_dump_json())
        except Exception as e:
            logger.warning(f"Failed to publish event to Redis: {e}")


class MultiEventHandler:
    """Composes multiple EventHandler instances into one.

    Events are dispatched to all handlers. If any handler raises,
    it is logged and the remaining handlers still receive the event.
    """

    def __init__(self, handlers: Sequence[EventHandler]) -> None:
        self._handlers = list(handlers)

    async def handle(self, event: AgentEvent) -> None:
        """Dispatch event to all composed handlers."""
        for handler in self._handlers:
            try:
                await handler.handle(event)
            except Exception as e:
                logger.warning(
                    f"Event handler {type(handler).__name__} failed: {e}"
                )
