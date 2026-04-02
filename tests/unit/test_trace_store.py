"""Unit tests for TraceStore."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.decisions import (
    Alternative,
    DecisionPoint,
    ExecutionRecord,
    LLMCallRecord,
)
from src.models.enums import AgentPhase, DecisionCategory, RunStatus
from src.models.traces import DecisionTrace, RunMetadata
from src.tracing.store import TraceStore


def _make_trace(run_id: str, status: RunStatus = RunStatus.SUCCESS) -> DecisionTrace:
    """Helper to create a test trace."""
    metadata = RunMetadata(
        run_id=run_id,
        task_description="Test task",
        llm_provider="openrouter",
        temperature=0.0,
        status=status,
        total_cost_usd=0.05,
    )
    dp = DecisionPoint(
        sequence_number=1,
        category=DecisionCategory.DATA_SELECTION,
        phase=AgentPhase.PLANNING,
        question="Which ticker?",
        alternatives=[
            Alternative(value="AAPL", reasoning="reason"),
            Alternative(value="TSLA", reasoning="reason"),
        ],
        chosen="AAPL",
        reasoning="Chose AAPL",
        confidence=0.8,
    )
    return DecisionTrace(
        metadata=metadata,
        decisions=[dp],
        executions=[],
        llm_calls=[],
    )


def _write_trace_files(runs_dir: Path, trace: DecisionTrace) -> None:
    """Write trace and metadata files to the filesystem."""
    run_dir = runs_dir / trace.metadata.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "trace.json").write_text(trace.model_dump_json(indent=2))
    (run_dir / "metadata.json").write_text(
        trace.metadata.model_dump_json(indent=2)
    )


class TestListRunsEmpty:
    """Test listing runs when no runs exist."""

    @pytest.mark.asyncio
    async def test_list_runs_empty(self, tmp_path: Path):
        """No runs should return empty list."""
        store = TraceStore(runs_dir=tmp_path)
        runs = await store.list_runs()
        assert runs == []


class TestListRunsWithData:
    """Test listing runs with data on filesystem."""

    @pytest.mark.asyncio
    async def test_list_runs_with_data(self, tmp_path: Path):
        """Create 3 trace files, verify list returns 3."""
        store = TraceStore(runs_dir=tmp_path)

        for i in range(3):
            trace = _make_trace(f"run-{i:03d}")
            _write_trace_files(tmp_path, trace)

        runs = await store.list_runs()
        assert len(runs) == 3

    @pytest.mark.asyncio
    async def test_list_runs_filter_by_status(self, tmp_path: Path):
        """Filter runs by status."""
        store = TraceStore(runs_dir=tmp_path)

        _write_trace_files(tmp_path, _make_trace("run-001", RunStatus.SUCCESS))
        _write_trace_files(tmp_path, _make_trace("run-002", RunStatus.FAILED))
        _write_trace_files(tmp_path, _make_trace("run-003", RunStatus.SUCCESS))

        success_runs = await store.list_runs(status=RunStatus.SUCCESS)
        assert len(success_runs) == 2

        failed_runs = await store.list_runs(status=RunStatus.FAILED)
        assert len(failed_runs) == 1


class TestGetTraceFromFilesystem:
    """Test reading traces from the filesystem."""

    @pytest.mark.asyncio
    async def test_get_trace_from_filesystem(self, tmp_path: Path):
        """Write a trace.json, verify get_trace reads it."""
        store = TraceStore(runs_dir=tmp_path)

        trace = _make_trace("run-fs-001")
        _write_trace_files(tmp_path, trace)

        loaded = await store.get_trace("run-fs-001")
        assert loaded is not None
        assert loaded.metadata.run_id == "run-fs-001"
        assert len(loaded.decisions) == 1
        assert loaded.decisions[0].chosen == "AAPL"

    @pytest.mark.asyncio
    async def test_get_trace_nonexistent(self, tmp_path: Path):
        """Non-existent run_id returns None."""
        store = TraceStore(runs_dir=tmp_path)
        loaded = await store.get_trace("run-nonexistent")
        assert loaded is None


class TestGetTraceFromCache:
    """Test that cache is used before filesystem."""

    @pytest.mark.asyncio
    async def test_get_trace_from_cache(self, tmp_path: Path):
        """Put trace in cache, verify it's served from cache (not filesystem)."""
        cache = AsyncMock()
        trace = _make_trace("run-cache-001")

        # Cache hit
        cache.get.return_value = trace.model_dump_json()

        store = TraceStore(runs_dir=tmp_path, cache=cache)
        loaded = await store.get_trace("run-cache-001")

        assert loaded is not None
        assert loaded.metadata.run_id == "run-cache-001"
        cache.get.assert_called_once_with("trace:run-cache-001")

    @pytest.mark.asyncio
    async def test_cache_miss_falls_to_filesystem(self, tmp_path: Path):
        """Cache miss should fall through to filesystem."""
        cache = AsyncMock()
        cache.get.return_value = None

        trace = _make_trace("run-cache-002")
        _write_trace_files(tmp_path, trace)

        store = TraceStore(runs_dir=tmp_path, cache=cache)
        loaded = await store.get_trace("run-cache-002")

        assert loaded is not None
        assert loaded.metadata.run_id == "run-cache-002"
        # Should have tried to backfill cache
        cache.set.assert_called_once()


class TestDeleteRun:
    """Test deleting runs."""

    @pytest.mark.asyncio
    async def test_delete_run_from_filesystem(self, tmp_path: Path):
        """Delete should remove run directory."""
        store = TraceStore(runs_dir=tmp_path)

        trace = _make_trace("run-del-001")
        _write_trace_files(tmp_path, trace)

        assert (tmp_path / "run-del-001").exists()

        deleted = await store.delete_run("run-del-001")
        assert deleted is True
        assert not (tmp_path / "run-del-001").exists()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_run(self, tmp_path: Path):
        """Deleting nonexistent run returns False."""
        store = TraceStore(runs_dir=tmp_path)
        deleted = await store.delete_run("run-nonexistent")
        assert deleted is False


class TestGetDecisionsForRun:
    """Test getting decisions for a specific run."""

    @pytest.mark.asyncio
    async def test_get_decisions_for_run(self, tmp_path: Path):
        """Get decisions should return the list from the trace."""
        store = TraceStore(runs_dir=tmp_path)

        trace = _make_trace("run-dec-001")
        _write_trace_files(tmp_path, trace)

        decisions = await store.get_decisions_for_run("run-dec-001")
        assert len(decisions) == 1
        assert decisions[0].chosen == "AAPL"

    @pytest.mark.asyncio
    async def test_get_decisions_nonexistent(self, tmp_path: Path):
        """Nonexistent run should return empty list."""
        store = TraceStore(runs_dir=tmp_path)
        decisions = await store.get_decisions_for_run("run-nonexistent")
        assert decisions == []


class TestListRunsPagination:
    """Test pagination of list_runs."""

    @pytest.mark.asyncio
    async def test_list_runs_limit(self, tmp_path: Path):
        """Limit should restrict number of results."""
        store = TraceStore(runs_dir=tmp_path)

        for i in range(5):
            _write_trace_files(tmp_path, _make_trace(f"run-{i:03d}"))

        runs = await store.list_runs(limit=2)
        assert len(runs) == 2

    @pytest.mark.asyncio
    async def test_list_runs_offset(self, tmp_path: Path):
        """Offset should skip results."""
        store = TraceStore(runs_dir=tmp_path)

        for i in range(5):
            _write_trace_files(tmp_path, _make_trace(f"run-{i:03d}"))

        runs = await store.list_runs(offset=3)
        assert len(runs) == 2
