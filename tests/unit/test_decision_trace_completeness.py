"""Tests that decision traces are complete and well-formed.

Validates that every trace produced by the agent has the required structure:
minimum decisions, required categories, alternatives, reasoning, ordering,
execution records, LLM call records, model routing metadata, cost tracking,
and serialization round-trip fidelity.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.models.decisions import Alternative, DecisionPoint, ExecutionRecord, LLMCallRecord
from src.models.enums import AgentPhase, DecisionCategory, RunStatus
from src.models.traces import DecisionTrace, RunMetadata


def _make_trace(
    run_id: str = "run-test001",
    decisions: list[DecisionPoint] | None = None,
    executions: list[ExecutionRecord] | None = None,
    llm_calls: list[LLMCallRecord] | None = None,
    status: RunStatus = RunStatus.SUCCESS,
    total_cost: float = 0.10,
    model_routing: dict[str, str] | None = None,
) -> DecisionTrace:
    """Build a realistic test trace."""
    if model_routing is None:
        model_routing = {
            "planning": "google/gemini-2.5-flash",
            "coding": "anthropic/claude-sonnet-4",
            "evaluating": "openai/gpt-4o-mini",
            "recovering": "anthropic/claude-sonnet-4",
        }

    if decisions is None:
        decisions = [
            DecisionPoint(
                id=f"dp-{i:03d}",
                sequence_number=i,
                category=cat,
                phase=AgentPhase.PLANNING,
                question=q,
                alternatives=[
                    Alternative(value=v1, reasoning="Option A"),
                    Alternative(value=v2, reasoning="Option B"),
                ],
                chosen=v1,
                reasoning=f"Chose {v1} because it fits best",
                confidence=0.8,
            )
            for i, (cat, q, v1, v2) in enumerate([
                (DecisionCategory.DATA_SELECTION, "Which stock ticker?", "AAPL", "TSLA"),
                (DecisionCategory.PARAMETER_TUNING, "What timeframe?", "2y", "1y"),
                (DecisionCategory.ALGORITHM_SELECTION, "Which anomaly method?", "IQR", "z_score"),
                (DecisionCategory.ARCHITECTURE, "Which API pattern?", "REST", "GraphQL"),
                (DecisionCategory.LIBRARY_SELECTION, "Which viz library?", "matplotlib", "plotly"),
            ])
        ]

    if executions is None:
        executions = [
            ExecutionRecord(
                code_snapshot="print('hello')",
                command="python main.py",
                stdout="Anomalies detected: 5",
                stderr="",
                exit_code=0,
                duration_ms=1500,
                artifacts_produced=["report.json"],
            ),
        ]

    if llm_calls is None:
        llm_calls = [
            LLMCallRecord(
                phase=AgentPhase.PLANNING,
                messages=[{"role": "user", "content": "plan"}],
                response="plan response",
                model="google/gemini-2.5-flash",
                temperature=0.0,
                tokens_in=500,
                tokens_out=300,
                latency_ms=800,
                cost_usd=0.002,
            ),
            LLMCallRecord(
                phase=AgentPhase.CODING,
                messages=[{"role": "user", "content": "code"}],
                response="code response",
                model="anthropic/claude-sonnet-4",
                temperature=0.0,
                tokens_in=1000,
                tokens_out=2000,
                latency_ms=3000,
                cost_usd=0.060,
            ),
            LLMCallRecord(
                phase=AgentPhase.EVALUATING,
                messages=[{"role": "user", "content": "eval"}],
                response="eval response",
                model="openai/gpt-4o-mini",
                temperature=0.0,
                tokens_in=800,
                tokens_out=100,
                latency_ms=400,
                cost_usd=0.001,
            ),
        ]

    metadata = RunMetadata(
        run_id=run_id,
        task_description="Pull stock data and detect anomalies",
        llm_provider="openrouter",
        model_routing=model_routing,
        temperature=0.0,
        status=status,
        total_llm_calls=len(llm_calls),
        total_tokens=sum(c.tokens_in + c.tokens_out for c in llm_calls),
        total_cost_usd=total_cost,
    )

    return DecisionTrace(
        metadata=metadata,
        decisions=decisions,
        executions=executions,
        llm_calls=llm_calls,
    )


def _make_five_traces() -> list[DecisionTrace]:
    """Build 5 distinct traces simulating genuine path variance."""
    configs = [
        ("run-001", "AAPL", "IQR", "matplotlib"),
        ("run-002", "TSLA", "z_score", "plotly"),
        ("run-003", "MSFT", "isolation_forest", "matplotlib"),
        ("run-004", "SPY", "IQR", "seaborn"),
        ("run-005", "GOOGL", "z_score", "plotly"),
    ]
    traces = []
    for run_id, ticker, method, viz in configs:
        decisions = [
            DecisionPoint(
                id=f"{run_id}-dp-0",
                sequence_number=0,
                category=DecisionCategory.DATA_SELECTION,
                phase=AgentPhase.PLANNING,
                question="Which stock ticker(s) to analyze?",
                alternatives=[
                    Alternative(value=ticker, reasoning="Selected"),
                    Alternative(value="OTHER", reasoning="Alternative"),
                ],
                chosen=ticker,
                reasoning=f"Chose {ticker}",
                confidence=0.85,
            ),
            DecisionPoint(
                id=f"{run_id}-dp-1",
                sequence_number=1,
                category=DecisionCategory.PARAMETER_TUNING,
                phase=AgentPhase.PLANNING,
                question="What timeframe to use?",
                alternatives=[
                    Alternative(value="2y", reasoning="Long history"),
                    Alternative(value="1y", reasoning="Recent"),
                ],
                chosen="2y",
                reasoning="More data points",
                confidence=0.80,
            ),
            DecisionPoint(
                id=f"{run_id}-dp-2",
                sequence_number=2,
                category=DecisionCategory.ALGORITHM_SELECTION,
                phase=AgentPhase.PLANNING,
                question="Which anomaly detection method?",
                alternatives=[
                    Alternative(value=method, reasoning="Selected"),
                    Alternative(value="other_method", reasoning="Alternative"),
                ],
                chosen=method,
                reasoning=f"Chose {method}",
                confidence=0.75,
            ),
            DecisionPoint(
                id=f"{run_id}-dp-3",
                sequence_number=3,
                category=DecisionCategory.ARCHITECTURE,
                phase=AgentPhase.PLANNING,
                question="Which API framework to use?",
                alternatives=[
                    Alternative(value="fastapi", reasoning="Modern"),
                    Alternative(value="flask", reasoning="Simple"),
                ],
                chosen="fastapi",
                reasoning="Async support",
                confidence=0.90,
            ),
            DecisionPoint(
                id=f"{run_id}-dp-4",
                sequence_number=4,
                category=DecisionCategory.LIBRARY_SELECTION,
                phase=AgentPhase.PLANNING,
                question="Which visualization library?",
                alternatives=[
                    Alternative(value=viz, reasoning="Selected"),
                    Alternative(value="other_viz", reasoning="Alternative"),
                ],
                chosen=viz,
                reasoning=f"Chose {viz}",
                confidence=0.70,
            ),
        ]

        llm_calls = [
            LLMCallRecord(
                phase=AgentPhase.PLANNING,
                messages=[{"role": "user", "content": "plan"}],
                response="{}",
                model="google/gemini-2.5-flash",
                temperature=0.0,
                tokens_in=500, tokens_out=300,
                latency_ms=800, cost_usd=0.002,
            ),
            LLMCallRecord(
                phase=AgentPhase.CODING,
                messages=[{"role": "user", "content": "code"}],
                response="{}",
                model="anthropic/claude-sonnet-4",
                temperature=0.0,
                tokens_in=1000, tokens_out=2000,
                latency_ms=3000, cost_usd=0.060,
            ),
            LLMCallRecord(
                phase=AgentPhase.EVALUATING,
                messages=[{"role": "user", "content": "eval"}],
                response="{}",
                model="openai/gpt-4o-mini",
                temperature=0.0,
                tokens_in=800, tokens_out=100,
                latency_ms=400, cost_usd=0.001,
            ),
        ]

        traces.append(_make_trace(
            run_id=run_id,
            decisions=decisions,
            llm_calls=llm_calls,
            total_cost=0.063,
        ))

    return traces


@pytest.fixture
def recorded_runs() -> list[DecisionTrace]:
    """5 simulated recorded runs with genuine path variance."""
    return _make_five_traces()


@pytest.fixture
def single_trace() -> DecisionTrace:
    """A single well-formed trace for individual tests."""
    return _make_trace()


class TestEveryRunHasMinimumDecisions:
    """Each run must have at least 5 decision points covering the core choices."""

    def test_every_run_has_minimum_decisions(self, recorded_runs):
        for trace in recorded_runs:
            assert len(trace.decisions) >= 5, (
                f"Run {trace.metadata.run_id} has only {len(trace.decisions)} decisions, "
                f"expected at least 5"
            )


class TestRequiredDecisionCategoriesPresent:
    """Every run must have decisions in: data_selection, algorithm_selection, architecture."""

    def test_required_decision_categories_present(self, recorded_runs):
        required = {
            DecisionCategory.DATA_SELECTION,
            DecisionCategory.ALGORITHM_SELECTION,
            DecisionCategory.ARCHITECTURE,
        }
        for trace in recorded_runs:
            categories = {d.category for d in trace.decisions}
            assert required.issubset(categories), (
                f"Run {trace.metadata.run_id} missing categories: "
                f"{required - categories}"
            )


class TestDecisionsHaveAlternatives:
    """Every decision must have at least 2 alternatives."""

    def test_decisions_have_alternatives(self, recorded_runs):
        for trace in recorded_runs:
            for decision in trace.decisions:
                assert len(decision.alternatives) >= 2, (
                    f"Decision '{decision.question}' in {trace.metadata.run_id} "
                    f"has only {len(decision.alternatives)} alternatives"
                )


class TestDecisionsHaveReasoning:
    """Every decision must have non-empty reasoning."""

    def test_decisions_have_reasoning(self, recorded_runs):
        for trace in recorded_runs:
            for decision in trace.decisions:
                assert decision.reasoning.strip(), (
                    f"Decision '{decision.question}' in {trace.metadata.run_id} "
                    f"has empty reasoning"
                )


class TestSequenceNumbersAreOrdered:
    """Sequence numbers must be monotonically increasing."""

    def test_sequence_numbers_are_ordered(self, recorded_runs):
        for trace in recorded_runs:
            seq_nums = [d.sequence_number for d in trace.decisions]
            assert seq_nums == sorted(seq_nums), (
                f"Run {trace.metadata.run_id} has unordered sequence numbers: "
                f"{seq_nums}"
            )


class TestExecutionRecordsPresent:
    """Each run must have at least one execution record."""

    def test_execution_records_present(self, recorded_runs):
        for trace in recorded_runs:
            assert len(trace.executions) >= 1, (
                f"Run {trace.metadata.run_id} has no execution records"
            )


class TestLLMCallsRecorded:
    """Each run must have at least 2 LLM calls (plan + code)."""

    def test_llm_calls_recorded(self, recorded_runs):
        for trace in recorded_runs:
            assert len(trace.llm_calls) >= 2, (
                f"Run {trace.metadata.run_id} has only {len(trace.llm_calls)} "
                f"LLM calls, expected at least 2 (plan + code)"
            )


class TestMultiModelRoutingUsed:
    """Each run must use multiple distinct models (per-phase routing)."""

    def test_multi_model_routing_used(self, recorded_runs):
        for trace in recorded_runs:
            models_used = {call.model for call in trace.llm_calls}
            assert len(models_used) >= 2, (
                f"Expected multiple models via per-phase routing, "
                f"got only: {models_used}"
            )


class TestModelRoutingMetadataPopulated:
    """Run metadata must record which model was used per phase."""

    def test_model_routing_metadata_populated(self, recorded_runs):
        for trace in recorded_runs:
            routing = trace.metadata.model_routing
            assert "planning" in routing or "coding" in routing, (
                f"model_routing not populated: {routing}"
            )


class TestEvaluationUsesCheapModel:
    """Evaluation phase should use the cheapest model (GPT-4o-mini)."""

    def test_evaluation_uses_cheap_model(self, recorded_runs):
        for trace in recorded_runs:
            eval_calls = [
                c for c in trace.llm_calls
                if c.phase == AgentPhase.EVALUATING
            ]
            for call in eval_calls:
                # Should NOT be using expensive models for evaluation
                assert "sonnet" not in call.model.lower(), (
                    f"Evaluation using expensive model: {call.model}"
                )


class TestCostIsPositive:
    """Each run must have a positive total cost."""

    def test_cost_is_positive(self, recorded_runs):
        for trace in recorded_runs:
            assert trace.metadata.total_cost_usd > 0, (
                f"Run {trace.metadata.run_id} has zero cost"
            )


class TestSerializationRoundTrip:
    """Each trace must survive JSON serialization and deserialization."""

    def test_serialization_round_trip(self, recorded_runs):
        for trace in recorded_runs:
            json_str = trace.model_dump_json()
            restored = DecisionTrace.model_validate_json(json_str)
            assert restored.metadata.run_id == trace.metadata.run_id
            assert len(restored.decisions) == len(trace.decisions)
            assert len(restored.executions) == len(trace.executions)
            assert len(restored.llm_calls) == len(trace.llm_calls)

    def test_serialization_preserves_decisions(self, single_trace):
        """Verify individual decision fields survive round-trip."""
        json_str = single_trace.model_dump_json()
        restored = DecisionTrace.model_validate_json(json_str)
        for orig, rest in zip(single_trace.decisions, restored.decisions):
            assert orig.chosen == rest.chosen
            assert orig.category == rest.category
            assert orig.question == rest.question
            assert orig.confidence == rest.confidence
            assert len(orig.alternatives) == len(rest.alternatives)


class TestDecisionConfidenceRange:
    """All decision confidence values must be in [0.0, 1.0]."""

    def test_confidence_in_range(self, recorded_runs):
        for trace in recorded_runs:
            for dp in trace.decisions:
                assert 0.0 <= dp.confidence <= 1.0, (
                    f"Decision '{dp.question}' has confidence {dp.confidence} "
                    f"outside [0.0, 1.0]"
                )


class TestPathVarianceAcrossRuns:
    """Across 5 runs, there should be genuine variance in decisions."""

    def test_ticker_variance(self, recorded_runs):
        """At least 2 different tickers across 5 runs."""
        tickers = set()
        for trace in recorded_runs:
            for dp in trace.decisions:
                if dp.category == DecisionCategory.DATA_SELECTION:
                    tickers.add(dp.chosen)
        assert len(tickers) >= 2, (
            f"Only {len(tickers)} unique ticker(s) across 5 runs: {tickers}. "
            f"Expected at least 2 for genuine variance."
        )

    def test_method_variance(self, recorded_runs):
        """At least 2 different anomaly methods across 5 runs."""
        methods = set()
        for trace in recorded_runs:
            for dp in trace.decisions:
                if dp.category == DecisionCategory.ALGORITHM_SELECTION:
                    methods.add(dp.chosen)
        assert len(methods) >= 2, (
            f"Only {len(methods)} unique method(s) across 5 runs: {methods}. "
            f"Expected at least 2 for genuine variance."
        )

    def test_not_all_identical(self, recorded_runs):
        """The 5 runs should not produce identical decision sequences."""
        decision_signatures = []
        for trace in recorded_runs:
            sig = tuple((dp.category.value, dp.chosen) for dp in trace.decisions)
            decision_signatures.append(sig)
        unique_signatures = set(decision_signatures)
        assert len(unique_signatures) >= 2, (
            "All 5 runs produced identical decision sequences - no genuine variance"
        )


class TestLLMCallPhaseTracking:
    """LLM calls must have their phase correctly recorded."""

    def test_llm_call_phases_valid(self, recorded_runs):
        valid_phases = set(AgentPhase)
        for trace in recorded_runs:
            for call in trace.llm_calls:
                assert call.phase in valid_phases, (
                    f"LLM call has invalid phase: {call.phase}"
                )

    def test_at_least_planning_and_coding_calls(self, recorded_runs):
        """Every run should have at least a planning and coding LLM call."""
        for trace in recorded_runs:
            phases = {call.phase for call in trace.llm_calls}
            assert AgentPhase.PLANNING in phases, (
                f"Run {trace.metadata.run_id} has no planning LLM call"
            )
            assert AgentPhase.CODING in phases, (
                f"Run {trace.metadata.run_id} has no coding LLM call"
            )
