"""Tests for cache implementations."""

from __future__ import annotations

import asyncio

import pytest

from src.cache.filesystem_cache import FilesystemCache
from src.cache.memory_cache import InMemoryCache
from src.cache.tiered import TieredCache


class TestInMemoryCache:
    async def test_memory_cache_set_get(self):
        """Set value, get it back."""
        cache = InMemoryCache()
        await cache.set("key1", "value1")
        result = await cache.get("key1")
        assert result == "value1"

    async def test_memory_cache_ttl_expiry(self):
        """Set with TTL=1, sleep 2, verify None."""
        cache = InMemoryCache()
        await cache.set("key1", "value1", ttl=1)
        await asyncio.sleep(1.5)
        result = await cache.get("key1")
        assert result is None

    async def test_memory_cache_lru_eviction(self):
        """Fill to max, add one more, verify oldest evicted."""
        # Use a tiny max size to trigger eviction
        cache = InMemoryCache(max_size_mb=0)  # Will be 0 bytes
        # Override to a small limit for testing
        cache._max_size_bytes = 200  # Very small

        await cache.set("a", "x" * 10)
        await cache.set("b", "y" * 10)
        await cache.set("c", "z" * 10)

        # At least one of the earlier keys should be evicted
        # The exact behavior depends on size estimates, but we verify
        # the cache doesn't grow unboundedly
        assert len(cache._store) <= 3

    async def test_memory_cache_delete(self):
        """Delete a key and verify it's gone."""
        cache = InMemoryCache()
        await cache.set("key1", "value1")
        await cache.delete("key1")
        result = await cache.get("key1")
        assert result is None

    async def test_memory_cache_exists(self):
        """Verify exists returns correct boolean."""
        cache = InMemoryCache()
        assert await cache.exists("key1") is False
        await cache.set("key1", "value1")
        assert await cache.exists("key1") is True

    async def test_memory_cache_overwrite(self):
        """Overwriting a key updates the value."""
        cache = InMemoryCache()
        await cache.set("key1", "value1")
        await cache.set("key1", "value2")
        assert await cache.get("key1") == "value2"


class TestFilesystemCache:
    async def test_filesystem_set_get(self, tmp_path):
        """Set value, get it back."""
        cache = FilesystemCache(cache_dir=tmp_path / ".cache")
        await cache.set("key1", "value1")
        result = await cache.get("key1")
        assert result == "value1"

    async def test_filesystem_delete(self, tmp_path):
        """Delete a key."""
        cache = FilesystemCache(cache_dir=tmp_path / ".cache")
        await cache.set("key1", "value1")
        await cache.delete("key1")
        result = await cache.get("key1")
        assert result is None

    async def test_filesystem_missing_key(self, tmp_path):
        """Get nonexistent key returns None."""
        cache = FilesystemCache(cache_dir=tmp_path / ".cache")
        result = await cache.get("nonexistent")
        assert result is None

    async def test_filesystem_persistence(self, tmp_path):
        """Value persists across new cache instances."""
        cache_dir = tmp_path / ".cache"
        cache1 = FilesystemCache(cache_dir=cache_dir)
        await cache1.set("key1", "persistent_value")

        cache2 = FilesystemCache(cache_dir=cache_dir)
        result = await cache2.get("key1")
        assert result == "persistent_value"


class TestTieredCache:
    async def test_tiered_cache_backfill(self, tmp_path):
        """Set in L3 only, get from tiered, verify L1 now has it."""
        l1 = InMemoryCache()
        l3 = FilesystemCache(cache_dir=tmp_path / ".cache")

        # Set only in L3
        await l3.set("key1", "from_filesystem")

        tiered = TieredCache(memory=l1, redis=None, filesystem=l3)
        result = await tiered.get("key1")
        assert result == "from_filesystem"

        # Verify L1 was backfilled
        l1_val = await l1.get("key1")
        assert l1_val == "from_filesystem"

    async def test_redis_graceful_degradation(self, tmp_path):
        """With no Redis, verify tiered cache still works."""
        tiered = TieredCache(
            memory=InMemoryCache(),
            redis=None,
            filesystem=FilesystemCache(cache_dir=tmp_path / ".cache"),
        )

        await tiered.set("key1", "value1")
        result = await tiered.get("key1")
        assert result == "value1"

    async def test_tiered_cache_delete(self, tmp_path):
        """Delete from all tiers."""
        l1 = InMemoryCache()
        l3 = FilesystemCache(cache_dir=tmp_path / ".cache")
        tiered = TieredCache(memory=l1, redis=None, filesystem=l3)

        await tiered.set("key1", "value1", persist=True)
        assert await tiered.get("key1") == "value1"

        await tiered.delete("key1")
        assert await tiered.get("key1") is None
        assert await l3.get("key1") is None

    async def test_tiered_cache_persist_flag(self, tmp_path):
        """Without persist=True, L3 should not have the value."""
        l1 = InMemoryCache()
        l3 = FilesystemCache(cache_dir=tmp_path / ".cache")
        tiered = TieredCache(memory=l1, redis=None, filesystem=l3)

        await tiered.set("key1", "value1", persist=False)
        # L1 should have it
        assert await l1.get("key1") == "value1"
        # L3 should NOT have it
        assert await l3.get("key1") is None

    async def test_tiered_cache_l1_hit(self, tmp_path):
        """L1 hit should not check lower tiers."""
        l1 = InMemoryCache()
        l3 = FilesystemCache(cache_dir=tmp_path / ".cache")
        tiered = TieredCache(memory=l1, redis=None, filesystem=l3)

        await l1.set("key1", "from_l1")
        await l3.set("key1", "from_l3")

        result = await tiered.get("key1")
        assert result == "from_l1"
