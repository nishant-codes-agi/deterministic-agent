"""Unit tests for PersistentTracer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.models.decisions import (
    Alternative,
    DecisionPoint,
    ExecutionRecord,
    LLMCallRecord,
)
from src.models.enums import AgentPhase, DecisionCategory, RunStatus
from src.tracing.cost_tracker import CostLimitExceededError, CostTracker
from src.tracing.tracer import PersistentTracer


def _make_decision(seq: int = 1, cost: float = 0.001) -> DecisionPoint:
    """Helper to create a test decision point."""
    return DecisionPoint(
        sequence_number=seq,
        category=DecisionCategory.DATA_SELECTION,
        phase=AgentPhase.PLANNING,
        question=f"Test question {seq}?",
        alternatives=[
            Alternative(value="A", reasoning="reason A"),
            Alternative(value="B", reasoning="reason B"),
        ],
        chosen="A",
        reasoning="Chose A",
        confidence=0.8,
        cost_usd=cost,
    )


def _make_execution() -> ExecutionRecord:
    """Helper to create a test execution record."""
    return ExecutionRecord(
        code_snapshot="print('hello')",
        command="python main.py",
        stdout="hello",
        stderr="",
        exit_code=0,
        duration_ms=100,
    )


def _make_llm_call(cost: float = 0.01) -> LLMCallRecord:
    """Helper to create a test LLM call record."""
    return LLMCallRecord(
        phase=AgentPhase.PLANNING,
        messages=[{"role": "user", "content": "test"}],
        response="test response",
        model="google/gemini-2.5-flash",
        temperature=0.0,
        tokens_in=100,
        tokens_out=50,
        latency_ms=500,
        cost_usd=cost,
    )


class TestIncrementalPersistence:
    """Test that records are incrementally written to JSONL."""

    @pytest.mark.asyncio
    async def test_incremental_persistence(self, tmp_path: Path):
        """Record 3 decisions, read JSONL file, verify 3 lines of valid JSON."""
        tracer = PersistentTracer(
            run_id="test-run-001",
            task="Test task",
            runs_dir=tmp_path,
        )

        for i in range(3):
            await tracer.record_decision(_make_decision(seq=i + 1))

        # Close the file before reading
        tracer._jsonl_file.close()

        jsonl_path = tmp_path / "test-run-001" / "trace.jsonl"
        assert jsonl_path.exists()

        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 3

        for line in lines:
            parsed = json.loads(line)
            assert parsed["type"] == "decision"
            assert "data" in parsed

    @pytest.mark.asyncio
    async def test_mixed_record_types(self, tmp_path: Path):
        """Verify decisions, executions, and LLM calls all appear in JSONL."""
        tracer = PersistentTracer(
            run_id="test-run-002",
            task="Test task",
            runs_dir=tmp_path,
        )

        await tracer.record_decision(_make_decision(seq=1))
        await tracer.record_execution(_make_execution())
        await tracer.record_llm_call(_make_llm_call())

        tracer._jsonl_file.close()

        jsonl_path = tmp_path / "test-run-002" / "trace.jsonl"
        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 3

        types = [json.loads(line)["type"] for line in lines]
        assert types == ["decision", "execution", "llm_call"]


class TestFinalizeCreatesCompleteTrace:
    """Test that finalize() creates proper trace.json and metadata.json."""

    @pytest.mark.asyncio
    async def test_finalize_creates_complete_trace(self, tmp_path: Path):
        """Record decisions + executions + llm_calls, finalize, verify trace.json."""
        tracer = PersistentTracer(
            run_id="test-run-003",
            task="Test task",
            runs_dir=tmp_path,
        )

        await tracer.record_decision(_make_decision(seq=1))
        await tracer.record_decision(_make_decision(seq=2))
        await tracer.record_execution(_make_execution())
        await tracer.record_llm_call(_make_llm_call())

        trace = await tracer.finalize(RunStatus.SUCCESS)

        assert trace.metadata.status == RunStatus.SUCCESS
        assert len(trace.decisions) == 2
        assert len(trace.executions) == 1
        assert len(trace.llm_calls) == 1

        # Verify trace.json exists and is valid
        trace_path = tmp_path / "test-run-003" / "trace.json"
        assert trace_path.exists()
        loaded = json.loads(trace_path.read_text())
        assert loaded["metadata"]["run_id"] == "test-run-003"
        assert loaded["metadata"]["status"] == "success"
        assert len(loaded["decisions"]) == 2

    @pytest.mark.asyncio
    async def test_finalize_writes_metadata(self, tmp_path: Path):
        """Verify metadata.json exists with correct fields."""
        tracer = PersistentTracer(
            run_id="test-run-004",
            task="Test task",
            runs_dir=tmp_path,
        )

        await tracer.record_decision(_make_decision(seq=1))
        await tracer.finalize(RunStatus.SUCCESS)

        metadata_path = tmp_path / "test-run-004" / "metadata.json"
        assert metadata_path.exists()

        meta = json.loads(metadata_path.read_text())
        assert meta["run_id"] == "test-run-004"
        assert meta["task_description"] == "Test task"
        assert meta["status"] == "success"
        assert meta["total_llm_calls"] == 0
        assert meta["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_finalize_with_error(self, tmp_path: Path):
        """Verify error message is captured in trace."""
        tracer = PersistentTracer(
            run_id="test-run-005",
            task="Test task",
            runs_dir=tmp_path,
        )

        trace = await tracer.finalize(RunStatus.FAILED, error="Something broke")
        assert trace.metadata.error == "Something broke"
        assert trace.metadata.status == RunStatus.FAILED


class TestCostTracking:
    """Test cost tracking with the CostTracker."""

    @pytest.mark.asyncio
    async def test_cost_tracking(self):
        """Record LLM calls with known costs, verify total matches."""
        tracker = CostTracker(
            run_id="test-run",
            max_cost_usd=1.0,
        )

        await tracker.add_cost(0.01)
        await tracker.add_cost(0.02)
        await tracker.add_cost(0.005)

        assert abs(tracker.current_cost - 0.035) < 1e-10

    @pytest.mark.asyncio
    async def test_cost_limit_exceeded(self):
        """Set max_cost to $0.001, record an LLM call costing $0.01, verify error."""
        tracker = CostTracker(
            run_id="test-run",
            max_cost_usd=0.001,
        )

        with pytest.raises(CostLimitExceededError) as exc_info:
            await tracker.add_cost(0.01)

        assert exc_info.value.current_cost == 0.01
        assert exc_info.value.max_cost == 0.001

    @pytest.mark.asyncio
    async def test_cost_zero_amount_ignored(self):
        """Zero-cost additions should not change the counter."""
        tracker = CostTracker(
            run_id="test-run",
            max_cost_usd=1.0,
        )

        await tracker.add_cost(0.0)
        await tracker.add_cost(-0.01)
        assert tracker.current_cost == 0.0

    @pytest.mark.asyncio
    async def test_cost_no_limit(self):
        """When max_cost_usd is 0, no limit should be enforced."""
        tracker = CostTracker(
            run_id="test-run",
            max_cost_usd=0.0,
        )

        await tracker.add_cost(100.0)
        assert tracker.current_cost == 100.0


class TestGetDecisions:
    """Test getting decisions from the tracer."""

    @pytest.mark.asyncio
    async def test_get_decisions_returns_copy(self, tmp_path: Path):
        """get_decisions() should return a copy of the list."""
        tracer = PersistentTracer(
            run_id="test-run-006",
            task="Test task",
            runs_dir=tmp_path,
        )

        await tracer.record_decision(_make_decision(seq=1))
        await tracer.record_decision(_make_decision(seq=2))

        decisions = await tracer.get_decisions()
        assert len(decisions) == 2

        # Mutating the returned list should not affect the tracer
        decisions.clear()
        assert len(await tracer.get_decisions()) == 2

        # Clean up
        tracer._jsonl_file.close()
