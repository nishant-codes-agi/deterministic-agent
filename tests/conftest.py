"""Shared test fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def test_settings():
    """Create test settings with test database URL."""
    from src.config import Settings

    return Settings(
        database={"url": "postgresql+asyncpg://agent:agent@localhost:5432/deterministic_agent_test"},
        redis={"url": "redis://localhost:6379/1"},
        runs_dir="./test_runs",
    )
