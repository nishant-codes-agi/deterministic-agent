"""Shared test fixtures."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import pytest

from src.models.enums import RunStatus


@pytest.fixture
def test_settings():
    """Create test settings with test database URL."""
    from src.config import Settings

    return Settings(
        database={"url": "postgresql+asyncpg://agent:agent@localhost:5432/deterministic_agent_test"},
        redis={"url": "redis://localhost:6379/1"},
        runs_dir="./test_runs",
    )


@pytest.fixture(scope="session")
def recorded_runs_from_disk() -> Optional[list]:
    """Load the 5 recorded runs from the runs/ directory.

    Returns None if runs haven't been recorded yet (scripts/record_runs.py).
    Tests using this fixture should skip if None.
    """
    from src.tracing.store import TraceStore

    runs_dir = Path("runs")
    if not runs_dir.exists():
        return None

    store = TraceStore(runs_dir=runs_dir)

    # List all successful runs
    runs = asyncio.get_event_loop().run_until_complete(
        store.list_runs(status=RunStatus.SUCCESS, limit=5)
    ) if asyncio.get_event_loop().is_running() else asyncio.run(
        store.list_runs(status=RunStatus.SUCCESS, limit=5)
    )

    if len(runs) < 5:
        return None

    # Load full traces
    async def _load():
        traces = []
        for r in runs[:5]:
            trace = await store.get_trace(r.run_id)
            if trace:
                traces.append(trace)
        return traces

    try:
        traces = asyncio.run(_load())
    except RuntimeError:
        # Event loop already running
        return None

    return traces if len(traces) >= 5 else None


@pytest.fixture
def recorded_run(recorded_runs_from_disk):
    """Single recorded run for locking tests. Requires real runs."""
    if recorded_runs_from_disk is None:
        pytest.skip("No recorded runs available. Run scripts/record_runs.py first.")
    return recorded_runs_from_disk[0]
