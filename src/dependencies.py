"""Dependency injection container for the application."""

from __future__ import annotations

import logging
from typing import Optional

from src.cache.memory_cache import InMemoryCache
from src.cache.redis_cache import RedisCache
from src.cache.tiered import TieredCache
from src.config import Settings, get_settings
from src.llm.base import LLMProvider
from src.llm.factory import create_llm_provider
from src.sandbox.base import SandboxProvider
from src.sandbox.subprocess_sandbox import SubprocessSandbox

logger = logging.getLogger(__name__)


class Container:
    """Simple DI container that lazily initializes and holds references.

    Not a framework -- just a plain class that creates and holds deps.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._db_engine = None
        self._db_session_factory = None
        self._redis_cache: Optional[RedisCache] = None
        self._tiered_cache: Optional[TieredCache] = None
        self._llm_provider: Optional[LLMProvider] = None
        self._sandbox: Optional[SandboxProvider] = None

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

        # Redis cache (graceful degradation)
        try:
            self._redis_cache = RedisCache(self._settings.redis.url)
            if await self._redis_cache.ping():
                logger.info("Redis connection established")
            else:
                logger.warning("Redis ping failed, continuing without it")
                self._redis_cache = None
        except Exception as e:
            logger.warning(f"Redis unavailable, continuing without it: {e}")
            self._redis_cache = None

        # Tiered cache: L1 (memory) -> L2 (Redis or memory) -> L3 (filesystem)
        self._tiered_cache = TieredCache(
            memory=InMemoryCache(max_size_mb=self._settings.redis.max_memory_mb),
            redis=self._redis_cache,
        )

        # LLM provider
        try:
            self._llm_provider = create_llm_provider(self._settings.llm)
            logger.info(
                f"LLM provider created: {self._llm_provider.provider_name}"
            )
        except Exception as e:
            logger.error(f"Failed to create LLM provider: {e}")

        # Sandbox
        self._sandbox = SubprocessSandbox(runs_dir=self._settings.runs_dir)
        logger.info("Sandbox provider created")

    async def shutdown(self) -> None:
        """Close all connections."""
        logger.info("Shutting down dependency container")
        if self._redis_cache:
            await self._redis_cache.close()
        if self._db_engine:
            await self._db_engine.dispose()

    def get_db_session_factory(self):
        """Get the async session factory."""
        return self._db_session_factory

    def get_redis(self) -> Optional[RedisCache]:
        """Get the Redis cache (may be None if unavailable)."""
        return self._redis_cache

    def get_llm(self) -> LLMProvider:
        """Get the LLM provider."""
        if self._llm_provider is None:
            raise RuntimeError("LLM provider not initialized. Call startup() first.")
        return self._llm_provider

    def get_cache(self) -> TieredCache:
        """Get the tiered cache."""
        if self._tiered_cache is None:
            raise RuntimeError("Cache not initialized. Call startup() first.")
        return self._tiered_cache

    def get_sandbox(self) -> SandboxProvider:
        """Get the sandbox provider."""
        if self._sandbox is None:
            raise RuntimeError("Sandbox not initialized. Call startup() first.")
        return self._sandbox

    def get_settings_ref(self) -> Settings:
        """Get settings reference."""
        return self._settings
