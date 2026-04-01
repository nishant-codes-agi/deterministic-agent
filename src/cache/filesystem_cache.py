"""Filesystem cache implementing CacheBackend."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from src.config import get_settings

logger = logging.getLogger(__name__)


class FilesystemCache:
    """Filesystem-based persistent cache.

    Stores JSON files at {RUNS_DIR}/.cache/{key_hash}.json.
    No TTL -- this cache is persistent.
    Uses atomic writes (write to .tmp file, then os.rename) for safety.
    """

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self._cache_dir = cache_dir or (get_settings().runs_dir / ".cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, key: str) -> Path:
        """Hash key to a file path. SHA-256, first 16 chars as filename."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self._cache_dir / f"{key_hash}.json"

    async def get(self, key: str) -> Optional[str]:
        """Get a value by key."""
        path = self._key_path(key)
        try:
            if not path.exists():
                return None
            data = json.loads(path.read_text())
            # Verify the key matches (handle hash collisions)
            if data.get("key") != key:
                return None
            logger.debug(f"Filesystem cache hit: {key}")
            return data["value"]
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(f"Filesystem cache get failed: {e}")
            return None

    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Set a value. TTL is ignored for filesystem cache (persistent)."""
        path = self._key_path(key)
        data = json.dumps({"key": key, "value": value})
        try:
            # Atomic write: write to temp file, then rename
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._cache_dir), suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(data)
                os.rename(tmp_path, str(path))
            except Exception:
                # Clean up temp file on error
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as e:
            logger.warning(f"Filesystem cache set failed: {e}")

    async def delete(self, key: str) -> None:
        """Delete a key."""
        path = self._key_path(key)
        try:
            if path.exists():
                path.unlink()
        except OSError as e:
            logger.warning(f"Filesystem cache delete failed: {e}")

    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        value = await self.get(key)
        return value is not None
