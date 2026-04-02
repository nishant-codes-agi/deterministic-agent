"""Cost tracking with optional Redis atomic counters."""

from __future__ import annotations

import logging
from typing import Optional

from src.cache.redis_cache import RedisCache

logger = logging.getLogger(__name__)


class CostLimitExceededError(Exception):
    """Raised when the run exceeds its cost budget."""

    def __init__(self, current_cost: float, max_cost: float) -> None:
        self.current_cost = current_cost
        self.max_cost = max_cost
        super().__init__(
            f"Cost limit exceeded: ${current_cost:.4f} > ${max_cost:.4f}"
        )


class CostTracker:
    """Tracks accumulated cost for a run.

    Uses Redis INCRBYFLOAT when available for atomic cross-process accuracy.
    Falls back to in-memory counter otherwise.
    """

    def __init__(
        self,
        run_id: str,
        max_cost_usd: float,
        redis: Optional[RedisCache] = None,
    ) -> None:
        self._run_id = run_id
        self._max_cost_usd = max_cost_usd
        self._redis = redis
        self._local_cost: float = 0.0

    @property
    def current_cost(self) -> float:
        """Get the current accumulated cost."""
        return self._local_cost

    async def add_cost(self, amount: float) -> float:
        """Add cost and check limit. Returns new total.

        Raises CostLimitExceededError if the limit is exceeded.
        """
        if amount <= 0:
            return self._local_cost

        # Try Redis atomic increment first
        if self._redis:
            total = await self._redis.increment_cost(self._run_id, amount)
            if total > 0:
                self._local_cost = total
            else:
                # Redis failed, fall back to local
                self._local_cost += amount
        else:
            self._local_cost += amount

        # Check limit
        if self._max_cost_usd > 0 and self._local_cost > self._max_cost_usd:
            raise CostLimitExceededError(self._local_cost, self._max_cost_usd)

        return self._local_cost
