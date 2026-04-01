"""Dependency injection container for the application."""

from __future__ import annotations

import logging
from typing import Optional

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


class Container:
    """Simple DI container that lazily initializes and holds references.

    Not a framework — just a plain class that creates and holds deps.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._db_engine = None
        self._db_session_factory = None
        self._redis_client = None
        self._cache_backend = None
        self._llm_provider = None

    @property
    def settings(self) -> Settings:
        """Get application settings."""
        return self._settings

    async def startup(self) -> None:
        """Initialize all dependencies."""
        logger.info("Starting up dependency container")

        # DB engine + session factory
        try:
            from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

            self._db_engine = create_async_engine(
                self._settings.database.url,
                echo=self._settings.database.echo,
                pool_size=self._settings.database.pool_size,
            )
            self._db_session_factory = async_sessionmaker(
                self._db_engine, expire_on_commit=False
            )
            logger.info("Database engine created")
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}")

        # Redis client (graceful degradation)
        try:
            import redis.asyncio as aioredis

            self._redis_client = aioredis.from_url(
                self._settings.redis.url,
                decode_responses=True,
            )
            await self._redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis unavailable, continuing without it: {e}")
            self._redis_client = None

    async def shutdown(self) -> None:
        """Close all connections."""
        logger.info("Shutting down dependency container")
        if self._redis_client:
            await self._redis_client.close()
        if self._db_engine:
            await self._db_engine.dispose()

    def get_db_session_factory(self):
        """Get the async session factory."""
        return self._db_session_factory

    def get_redis(self):
        """Get the Redis client (may be None if unavailable)."""
        return self._redis_client

    def get_settings_ref(self) -> Settings:
        """Get settings reference."""
        return self._settings
