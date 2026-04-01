"""Unit tests for core data models."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.models.decisions import Alternative, DecisionPoint, ExecutionRecord, LLMCallRecord
from src.models.enums import AgentPhase, DecisionCategory, RunStatus
from src.models.traces import (
    CostEstimate,
    DecisionTrace,
    PathLockConfig,
    RunMetadata,
)


class TestDecisionPointSerialization:
    """Test DecisionPoint serialization round-trip."""

    def test_serialize_deserialize(self):
        dp = DecisionPoint(
            sequence_number=1,
            category=DecisionCategory.DATA_SELECTION,
            phase=AgentPhase.PLANNING,
            question="Which ticker?",
            alternatives=[
                Alternative(value="AAPL", reasoning="Large cap"),
                Alternative(value="TSLA", reasoning="High volatility"),
            ],
            chosen="AAPL",
            reasoning="Most liquid",
            confidence=0.85,
        )
        json_str = dp.model_dump_json()
        restored = DecisionPoint.model_validate_json(json_str)
        assert restored.chosen == dp.chosen
        assert restored.sequence_number == dp.sequence_number
        assert len(restored.alternatives) == 2


class TestDecisionPointValidation:
    """Test DecisionPoint validation rules."""

    def test_requires_min_alternatives(self):
        with pytest.raises(ValidationError):
            DecisionPoint(
                sequence_number=1,
                category=DecisionCategory.DATA_SELECTION,
                phase=AgentPhase.PLANNING,
                question="Which ticker?",
                alternatives=[Alternative(value="AAPL")],  # Only 1, need 2+
                chosen="AAPL",
                reasoning="Only option",
                confidence=0.5,
            )

    def test_confidence_bounds_low(self):
        with pytest.raises(ValidationError):
            DecisionPoint(
                sequence_number=1,
                category=DecisionCategory.DATA_SELECTION,
                phase=AgentPhase.PLANNING,
                question="Which ticker?",
                alternatives=[
                    Alternative(value="AAPL"),
                    Alternative(value="TSLA"),
                ],
                chosen="AAPL",
                reasoning="Test",
                confidence=-0.1,
            )

    def test_confidence_bounds_high(self):
        with pytest.raises(ValidationError):
            DecisionPoint(
                sequence_number=1,
                category=DecisionCategory.DATA_SELECTION,
                phase=AgentPhase.PLANNING,
                question="Which ticker?",
                alternatives=[
                    Alternative(value="AAPL"),
                    Alternative(value="TSLA"),
                ],
                chosen="AAPL",
                reasoning="Test",
                confidence=1.1,
            )


class TestDecisionTraceRoundTrip:
    """Test full trace serialization."""

    def test_full_trace_round_trip(self):
        dp = DecisionPoint(
            sequence_number=1,
            category=DecisionCategory.ALGORITHM_SELECTION,
            phase=AgentPhase.PLANNING,
            question="Which method?",
            alternatives=[
                Alternative(value="IQR"),
                Alternative(value="Z-score"),
            ],
            chosen="IQR",
            reasoning="Robust to outliers",
            confidence=0.9,
        )
        execution = ExecutionRecord(
            code_snapshot="print('hello')",
            command="python main.py",
            stdout="hello",
            exit_code=0,
            duration_ms=150,
        )
        llm_call = LLMCallRecord(
            phase=AgentPhase.PLANNING,
            messages=[{"role": "user", "content": "test"}],
            response="test response",
            model="google/gemini-2.5-flash",
            temperature=0.0,
            tokens_in=100,
            tokens_out=50,
            latency_ms=500,
            cost_usd=0.001,
        )
        metadata = RunMetadata(
            task_description="Test task",
            llm_provider="openrouter",
            temperature=0.0,
        )
        trace = DecisionTrace(
            metadata=metadata,
            decisions=[dp],
            executions=[execution],
            llm_calls=[llm_call],
        )
        json_str = trace.model_dump_json()
        restored = DecisionTrace.model_validate_json(json_str)
        assert len(restored.decisions) == 1
        assert len(restored.executions) == 1
        assert len(restored.llm_calls) == 1
        assert restored.metadata.task_description == "Test task"


class TestPathLockConfig:
    """Test path lock config modes."""

    def test_lock_by_ids(self):
        config = PathLockConfig(
            source_run_id="run-abc123",
            locked_decision_ids=["dp-001", "dp-002"],
        )
        assert config.locked_decision_ids is not None
        assert len(config.locked_decision_ids) == 2

    def test_lock_by_sequence(self):
        config = PathLockConfig(
            source_run_id="run-abc123",
            lock_through_sequence=3,
        )
        assert config.lock_through_sequence == 3

    def test_lock_by_category(self):
        config = PathLockConfig(
            source_run_id="run-abc123",
            lock_categories=[DecisionCategory.DATA_SELECTION],
        )
        assert config.lock_categories is not None
        assert len(config.lock_categories) == 1


class TestRunMetadataDefaults:
    """Test RunMetadata default values."""

    def test_auto_generated_run_id(self):
        meta = RunMetadata(
            task_description="Test",
            llm_provider="openrouter",
            temperature=0.0,
        )
        assert meta.run_id.startswith("run-")
        assert meta.status == RunStatus.PENDING

    def test_status_defaults_to_pending(self):
        meta = RunMetadata(
            task_description="Test",
            llm_provider="openrouter",
            temperature=0.0,
        )
        assert meta.status == RunStatus.PENDING


class TestCostEstimate:
    """Test CostEstimate structure."""

    def test_breakdown_dict(self):
        estimate = CostEstimate(
            model_routing={"planning": "google/gemini-2.5-flash"},
            estimated_tokens_in=1000,
            estimated_tokens_out=500,
            estimated_cost_usd=0.01,
            confidence_low=0.005,
            confidence_high=0.02,
            breakdown={"planning": 0.006, "coding": 0.072},
            cost_by_model={"google/gemini-2.5-flash": 0.006},
        )
        assert "planning" in estimate.breakdown
        assert estimate.estimated_cost_usd == 0.01
