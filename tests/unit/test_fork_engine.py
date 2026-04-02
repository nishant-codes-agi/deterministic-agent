"""Unit tests for ForkEngine and RunComparator (Phase 8)."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional

import pytest

from src.analysis.comparator import RunComparator
from src.forking.engine import ForkEngine
from src.agent.base import CodingAgent, EventHandler
from src.models.comparison import DecisionAlignment, RunComparison
from src.models.decisions import Alternative, DecisionPoint, ExecutionRecord
from src.models.enums import AgentPhase, DecisionCategory, RunStatus
from src.models.traces import DecisionTrace, PathLockConfig, RunMetadata
from src.tracing.store import TraceStore


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_decision(
    seq: int,
    question: str,
    category: DecisionCategory,
    chosen: str,
    dp_id: str | None = None,
    locked: bool = False,
) -> DecisionPoint:
    """Helper to create a test decision point."""
    return DecisionPoint(
        id=dp_id or f"dp-{seq:03d}",
        sequence_number=seq,
        category=category,
        phase=AgentPhase.PLANNING,
        question=question,
        alternatives=[
            Alternative(value=chosen, reasoning="Chosen option"),
            Alternative(value="other", reasoning="Other option"),
        ],
        chosen=chosen,
        reasoning=f"Chose {chosen}",
        confidence=0.8,
        locked=locked,
    )


def _make_trace(
    run_id: str,
    decisions: list[DecisionPoint],
    executions: list[ExecutionRecord] | None = None,
    cost: float = 0.05,
) -> DecisionTrace:
    """Helper to create a test trace."""
    return DecisionTrace(
        metadata=RunMetadata(
            run_id=run_id,
            task_description="Analyze stock data and detect anomalies",
            llm_provider="openrouter",
            model_routing={"planning": "test-model"},
            temperature=0.0,
            status=RunStatus.SUCCESS,
            total_cost_usd=cost,
        ),
        decisions=decisions,
        executions=executions or [],
    )


def _make_source_decisions() -> list[DecisionPoint]:
    """Create a standard set of 5 source decisions."""
    return [
        _make_decision(0, "Which stock ticker?", DecisionCategory.DATA_SELECTION, "AAPL", "dp-001"),
        _make_decision(1, "What timeframe?", DecisionCategory.PARAMETER_TUNING, "2y", "dp-002"),
        _make_decision(2, "Which anomaly method?", DecisionCategory.ALGORITHM_SELECTION, "IQR", "dp-003"),
        _make_decision(3, "Which API framework?", DecisionCategory.LIBRARY_SELECTION, "fastapi", "dp-004"),
        _make_decision(4, "Which visualization library?", DecisionCategory.LIBRARY_SELECTION, "matplotlib", "dp-005"),
    ]


# ── RunComparator Tests ─────────────────────────────────────────────────────


class TestComparisonAlignment:
    """Test that RunComparator aligns decisions correctly across runs."""

    def test_identical_runs_full_match(self):
        """Two traces with identical decisions produce all-match alignment."""
        decisions = _make_source_decisions()
        trace_a = _make_trace("run-a", decisions)
        trace_b = _make_trace("run-b", decisions)

        comparator = RunComparator()
        result = comparator.compare(trace_a, trace_b)

        assert len(result.decision_alignment) == 5
        for alignment in result.decision_alignment:
            assert alignment.match is True

    def test_known_differences_detected(self):
        """Two traces with known differences correctly identify mismatches."""
        decisions_a = _make_source_decisions()
        decisions_b = [
            _make_decision(0, "Which stock ticker?", DecisionCategory.DATA_SELECTION, "TSLA", "dp-b01"),
            _make_decision(1, "What timeframe?", DecisionCategory.PARAMETER_TUNING, "2y", "dp-b02"),
            _make_decision(2, "Which anomaly method?", DecisionCategory.ALGORITHM_SELECTION, "z_score", "dp-b03"),
            _make_decision(3, "Which API framework?", DecisionCategory.LIBRARY_SELECTION, "fastapi", "dp-b04"),
            _make_decision(4, "Which visualization library?", DecisionCategory.LIBRARY_SELECTION, "plotly", "dp-b05"),
        ]
        trace_a = _make_trace("run-a", decisions_a)
        trace_b = _make_trace("run-b", decisions_b)

        comparator = RunComparator()
        result = comparator.compare(trace_a, trace_b)

        assert len(result.decision_alignment) == 5

        # Ticker: AAPL vs TSLA -> mismatch
        ticker = result.decision_alignment[0]
        assert ticker.run_a_choice == "AAPL"
        assert ticker.run_b_choice == "TSLA"
        assert ticker.match is False

        # Timeframe: 2y vs 2y -> match
        timeframe = result.decision_alignment[1]
        assert timeframe.match is True

        # Method: IQR vs z_score -> mismatch
        method = result.decision_alignment[2]
        assert method.match is False

        # Framework: fastapi vs fastapi -> match
        framework = result.decision_alignment[3]
        assert framework.match is True

        # Viz: matplotlib vs plotly -> mismatch
        viz = result.decision_alignment[4]
        assert viz.match is False

    def test_unmatched_decisions_in_b(self):
        """Decisions in B with no match in A are appended to alignment."""
        decisions_a = [
            _make_decision(0, "Which stock ticker?", DecisionCategory.DATA_SELECTION, "AAPL", "dp-a01"),
        ]
        decisions_b = [
            _make_decision(0, "Which stock ticker?", DecisionCategory.DATA_SELECTION, "AAPL", "dp-b01"),
            _make_decision(1, "Extra decision unique to B?", DecisionCategory.ARCHITECTURE, "microservice", "dp-b02"),
        ]
        trace_a = _make_trace("run-a", decisions_a)
        trace_b = _make_trace("run-b", decisions_b)

        comparator = RunComparator()
        result = comparator.compare(trace_a, trace_b)

        # Should have 2 alignments: 1 matched + 1 unmatched from B
        assert len(result.decision_alignment) == 2
        assert result.decision_alignment[0].match is True
        assert result.decision_alignment[1].run_a_choice is None
        assert result.decision_alignment[1].run_b_choice == "microservice"
        assert result.decision_alignment[1].match is False


class TestComparisonVarianceScore:
    """Test variance score computation."""

    def test_all_same_is_zero(self):
        """Identical runs have variance score 0.0."""
        decisions = _make_source_decisions()
        trace_a = _make_trace("run-a", decisions)
        trace_b = _make_trace("run-b", decisions)

        comparator = RunComparator()
        result = comparator.compare(trace_a, trace_b)

        assert result.variance_score == 0.0

    def test_all_different_is_one(self):
        """Completely different runs have variance score 1.0."""
        decisions_a = [
            _make_decision(0, "Which stock ticker?", DecisionCategory.DATA_SELECTION, "AAPL", "dp-a01"),
            _make_decision(1, "What timeframe?", DecisionCategory.PARAMETER_TUNING, "2y", "dp-a02"),
        ]
        decisions_b = [
            _make_decision(0, "Which stock ticker?", DecisionCategory.DATA_SELECTION, "TSLA", "dp-b01"),
            _make_decision(1, "What timeframe?", DecisionCategory.PARAMETER_TUNING, "5y", "dp-b02"),
        ]
        trace_a = _make_trace("run-a", decisions_a)
        trace_b = _make_trace("run-b", decisions_b)

        comparator = RunComparator()
        result = comparator.compare(trace_a, trace_b)

        assert result.variance_score == 1.0

    def test_partial_variance(self):
        """Mixed matches produce a fractional variance score."""
        decisions_a = [
            _make_decision(0, "Which stock ticker?", DecisionCategory.DATA_SELECTION, "AAPL", "dp-a01"),
            _make_decision(1, "What timeframe?", DecisionCategory.PARAMETER_TUNING, "2y", "dp-a02"),
            _make_decision(2, "Which anomaly method?", DecisionCategory.ALGORITHM_SELECTION, "IQR", "dp-a03"),
        ]
        decisions_b = [
            _make_decision(0, "Which stock ticker?", DecisionCategory.DATA_SELECTION, "AAPL", "dp-b01"),
            _make_decision(1, "What timeframe?", DecisionCategory.PARAMETER_TUNING, "5y", "dp-b02"),
            _make_decision(2, "Which anomaly method?", DecisionCategory.ALGORITHM_SELECTION, "IQR", "dp-b03"),
        ]
        trace_a = _make_trace("run-a", decisions_a)
        trace_b = _make_trace("run-b", decisions_b)

        comparator = RunComparator()
        result = comparator.compare(trace_a, trace_b)

        # 2/3 match -> variance = 1/3
        assert abs(result.variance_score - 1.0 / 3.0) < 0.01

    def test_empty_alignments_is_zero(self):
        """Comparing two traces with no decisions gives 0.0."""
        trace_a = _make_trace("run-a", [])
        trace_b = _make_trace("run-b", [])

        comparator = RunComparator()
        result = comparator.compare(trace_a, trace_b)

        assert result.variance_score == 0.0


class TestOutcomeDiff:
    """Test outcome diff computation."""

    def test_cost_diff(self):
        """OutcomeDiff captures cost difference between runs."""
        trace_a = _make_trace("run-a", [], cost=0.05)
        trace_b = _make_trace("run-b", [], cost=0.10)

        comparator = RunComparator()
        result = comparator.compare(trace_a, trace_b)

        assert result.outcome_diff.total_cost_usd["run-a"] == 0.05
        assert result.outcome_diff.total_cost_usd["run-b"] == 0.10

    def test_library_extraction(self):
        """OutcomeDiff extracts libraries from code snapshot."""
        exec_a = ExecutionRecord(
            code_snapshot="import pandas\nimport numpy\nfrom sklearn.ensemble import IsolationForest\nprint('hello')",
            command="python main.py",
            stdout="done",
            stderr="",
            exit_code=0,
            duration_ms=500,
        )
        exec_b = ExecutionRecord(
            code_snapshot="import pandas\nimport matplotlib\nprint('hello')",
            command="python main.py",
            stdout="done",
            stderr="",
            exit_code=0,
            duration_ms=300,
        )
        trace_a = _make_trace("run-a", [], executions=[exec_a])
        trace_b = _make_trace("run-b", [], executions=[exec_b])

        comparator = RunComparator()
        result = comparator.compare(trace_a, trace_b)

        assert "pandas" in result.outcome_diff.libraries_used["run-a"]
        assert "numpy" in result.outcome_diff.libraries_used["run-a"]
        assert "sklearn" in result.outcome_diff.libraries_used["run-a"]
        assert "matplotlib" in result.outcome_diff.libraries_used["run-b"]


# ── ForkEngine Tests ────────────────────────────────────────────────────────


class MockCodingAgent(CodingAgent):
    """Mock agent that returns a trace with customizable decisions.

    When a lock_config is provided with lock_through_sequence, the mock
    simulates the expected fork behavior:
    - Upstream decisions (seq <= lock_through_sequence) are replayed from source
    - The overridden decision uses the new value
    - Downstream decisions can differ
    """

    def __init__(
        self,
        source_decisions: list[DecisionPoint],
        forked_decisions: list[DecisionPoint] | None = None,
    ) -> None:
        self._source_decisions = source_decisions
        self._forked_decisions = forked_decisions

    async def run(
        self,
        task: str,
        lock_config: Optional[PathLockConfig] = None,
        event_handler: Optional[EventHandler] = None,
    ) -> DecisionTrace:
        """Return a trace simulating fork behavior based on lock_config."""
        if self._forked_decisions is not None:
            decisions = self._forked_decisions
        elif lock_config and lock_config.overrides:
            # Simulate fork: keep upstream, apply override, change downstream
            decisions = []

            for d in self._source_decisions:
                if (
                    lock_config.lock_through_sequence is not None
                    and d.sequence_number <= lock_config.lock_through_sequence
                ):
                    # Upstream: locked, same value
                    decisions.append(
                        DecisionPoint(
                            id=f"fork-{d.id}",
                            sequence_number=d.sequence_number,
                            category=d.category,
                            phase=d.phase,
                            question=d.question,
                            alternatives=d.alternatives,
                            chosen=d.chosen,
                            reasoning=d.reasoning,
                            confidence=d.confidence,
                            locked=True,
                        )
                    )
                elif d.id in lock_config.overrides:
                    # Fork point: overridden value
                    decisions.append(
                        DecisionPoint(
                            id=f"fork-{d.id}",
                            sequence_number=d.sequence_number,
                            category=d.category,
                            phase=d.phase,
                            question=d.question,
                            alternatives=d.alternatives,
                            chosen=lock_config.overrides[d.id],
                            reasoning=f"Overridden to {lock_config.overrides[d.id]}",
                            confidence=d.confidence,
                            locked=True,
                        )
                    )
                else:
                    # Downstream: free, potentially different
                    decisions.append(
                        DecisionPoint(
                            id=f"fork-{d.id}",
                            sequence_number=d.sequence_number,
                            category=d.category,
                            phase=d.phase,
                            question=d.question,
                            alternatives=d.alternatives,
                            chosen=d.chosen + "_new",
                            reasoning="Free decision, chose differently",
                            confidence=d.confidence,
                            locked=False,
                        )
                    )
        else:
            decisions = self._source_decisions

        return DecisionTrace(
            metadata=RunMetadata(
                run_id="run-forked",
                task_description=task,
                llm_provider="openrouter",
                model_routing={"planning": "test-model"},
                temperature=0.0,
                status=RunStatus.SUCCESS,
                total_cost_usd=0.05,
            ),
            decisions=decisions,
        )


@pytest.fixture
def tmp_runs_dir():
    """Create a temporary runs directory with a source run."""
    d = tempfile.mkdtemp(prefix="test_fork_runs_")
    runs_dir = Path(d)

    # Write a source run to disk
    source_decisions = _make_source_decisions()
    source_trace = _make_trace("source-run-001", source_decisions)

    run_dir = runs_dir / "source-run-001"
    run_dir.mkdir(parents=True)
    (run_dir / "trace.json").write_text(source_trace.model_dump_json(indent=2))
    (run_dir / "metadata.json").write_text(source_trace.metadata.model_dump_json(indent=2))

    yield runs_dir
    shutil.rmtree(d, ignore_errors=True)


class TestForkPreservesUpstream:
    """Test that forking preserves decisions upstream of the fork point."""

    @pytest.mark.asyncio
    async def test_fork_preserves_upstream(self, tmp_runs_dir):
        """Fork at decision 3, verify decisions 0-1 match source."""
        source_decisions = _make_source_decisions()
        agent = MockCodingAgent(source_decisions)
        store = TraceStore(runs_dir=tmp_runs_dir)
        engine = ForkEngine(agent=agent, trace_store=store)

        forked_trace, comparison = await engine.fork(
            source_run_id="source-run-001",
            decision_id="dp-003",
            new_choice="z_score",
        )

        # Decisions 0 and 1 should be locked and match source
        for dp in forked_trace.decisions:
            if dp.sequence_number < 2:  # upstream of dp-003 (seq 2)
                assert dp.locked is True
                source_dp = source_decisions[dp.sequence_number]
                assert dp.chosen == source_dp.chosen


class TestForkOverridesTarget:
    """Test that the fork-point decision uses the new value."""

    @pytest.mark.asyncio
    async def test_fork_overrides_target(self, tmp_runs_dir):
        """Verify the forked decision uses the override value."""
        source_decisions = _make_source_decisions()
        agent = MockCodingAgent(source_decisions)
        store = TraceStore(runs_dir=tmp_runs_dir)
        engine = ForkEngine(agent=agent, trace_store=store)

        forked_trace, comparison = await engine.fork(
            source_run_id="source-run-001",
            decision_id="dp-003",
            new_choice="z_score",
        )

        # Find the fork-point decision (sequence 2 = "Which anomaly method?")
        fork_dp = next(
            (d for d in forked_trace.decisions if d.sequence_number == 2),
            None,
        )
        assert fork_dp is not None
        assert fork_dp.chosen == "z_score"


class TestForkFreesDownstream:
    """Test that decisions after the fork point can differ."""

    @pytest.mark.asyncio
    async def test_fork_frees_downstream(self, tmp_runs_dir):
        """Verify decisions after fork point are free (not locked)."""
        source_decisions = _make_source_decisions()
        agent = MockCodingAgent(source_decisions)
        store = TraceStore(runs_dir=tmp_runs_dir)
        engine = ForkEngine(agent=agent, trace_store=store)

        forked_trace, comparison = await engine.fork(
            source_run_id="source-run-001",
            decision_id="dp-003",
            new_choice="z_score",
        )

        # Decisions 3 and 4 (seq > 2) should be free (not locked)
        downstream = [
            d for d in forked_trace.decisions if d.sequence_number > 2
        ]
        for dp in downstream:
            assert dp.locked is False

    @pytest.mark.asyncio
    async def test_fork_downstream_can_differ(self, tmp_runs_dir):
        """Verify downstream decisions may have different values."""
        source_decisions = _make_source_decisions()
        agent = MockCodingAgent(source_decisions)
        store = TraceStore(runs_dir=tmp_runs_dir)
        engine = ForkEngine(agent=agent, trace_store=store)

        forked_trace, comparison = await engine.fork(
            source_run_id="source-run-001",
            decision_id="dp-003",
            new_choice="z_score",
        )

        # Downstream decisions got "_new" appended by mock
        downstream = [
            d for d in forked_trace.decisions if d.sequence_number > 2
        ]
        for dp in downstream:
            assert dp.chosen.endswith("_new")


class TestForkMetadata:
    """Test that fork metadata is set correctly."""

    @pytest.mark.asyncio
    async def test_fork_sets_parent_run_id(self, tmp_runs_dir):
        """Forked trace should reference the source run."""
        source_decisions = _make_source_decisions()
        agent = MockCodingAgent(source_decisions)
        store = TraceStore(runs_dir=tmp_runs_dir)
        engine = ForkEngine(agent=agent, trace_store=store)

        forked_trace, _ = await engine.fork(
            source_run_id="source-run-001",
            decision_id="dp-003",
            new_choice="z_score",
        )

        assert forked_trace.metadata.parent_run_id == "source-run-001"
        assert forked_trace.metadata.fork_point == "dp-003"

    @pytest.mark.asyncio
    async def test_fork_returns_comparison(self, tmp_runs_dir):
        """Fork should return a RunComparison object."""
        source_decisions = _make_source_decisions()
        agent = MockCodingAgent(source_decisions)
        store = TraceStore(runs_dir=tmp_runs_dir)
        engine = ForkEngine(agent=agent, trace_store=store)

        _, comparison = await engine.fork(
            source_run_id="source-run-001",
            decision_id="dp-003",
            new_choice="z_score",
        )

        assert isinstance(comparison, RunComparison)
        assert comparison.run_a_id == "source-run-001"
        assert comparison.run_b_id == "run-forked"
        assert len(comparison.decision_alignment) > 0


class TestForkErrors:
    """Test error handling in ForkEngine."""

    @pytest.mark.asyncio
    async def test_fork_missing_source_run(self, tmp_runs_dir):
        """Fork with nonexistent source run raises ValueError."""
        agent = MockCodingAgent([])
        store = TraceStore(runs_dir=tmp_runs_dir)
        engine = ForkEngine(agent=agent, trace_store=store)

        with pytest.raises(ValueError, match="not found"):
            await engine.fork(
                source_run_id="nonexistent-run",
                decision_id="dp-001",
                new_choice="TSLA",
            )

    @pytest.mark.asyncio
    async def test_fork_missing_decision(self, tmp_runs_dir):
        """Fork with nonexistent decision ID raises ValueError."""
        source_decisions = _make_source_decisions()
        agent = MockCodingAgent(source_decisions)
        store = TraceStore(runs_dir=tmp_runs_dir)
        engine = ForkEngine(agent=agent, trace_store=store)

        with pytest.raises(ValueError, match="not found"):
            await engine.fork(
                source_run_id="source-run-001",
                decision_id="dp-999",
                new_choice="TSLA",
            )

    @pytest.mark.asyncio
    async def test_fork_by_sequence_number(self, tmp_runs_dir):
        """Fork using sequence number string instead of decision ID."""
        source_decisions = _make_source_decisions()
        agent = MockCodingAgent(source_decisions)
        store = TraceStore(runs_dir=tmp_runs_dir)
        engine = ForkEngine(agent=agent, trace_store=store)

        forked_trace, comparison = await engine.fork(
            source_run_id="source-run-001",
            decision_id="2",  # sequence number as string
            new_choice="z_score",
        )

        assert forked_trace.metadata.fork_point == "dp-003"
        fork_dp = next(
            (d for d in forked_trace.decisions if d.sequence_number == 2),
            None,
        )
        assert fork_dp is not None
        assert fork_dp.chosen == "z_score"


class TestLockConfigBuild:
    """Test that ForkEngine builds the correct PathLockConfig."""

    @pytest.mark.asyncio
    async def test_lock_config_has_correct_sequence(self, tmp_runs_dir):
        """Lock config should lock through sequence N-1 where N is the fork point."""
        source_decisions = _make_source_decisions()
        captured_lock_config = None

        class CapturingAgent(CodingAgent):
            async def run(self, task, lock_config=None, event_handler=None):
                nonlocal captured_lock_config
                captured_lock_config = lock_config
                return DecisionTrace(
                    metadata=RunMetadata(
                        run_id="run-capture",
                        task_description=task,
                        llm_provider="openrouter",
                        model_routing={},
                        temperature=0.0,
                        status=RunStatus.SUCCESS,
                    ),
                    decisions=source_decisions,
                )

        store = TraceStore(runs_dir=tmp_runs_dir)
        engine = ForkEngine(agent=CapturingAgent(), trace_store=store)

        await engine.fork(
            source_run_id="source-run-001",
            decision_id="dp-003",  # seq 2
            new_choice="z_score",
        )

        assert captured_lock_config is not None
        assert captured_lock_config.source_run_id == "source-run-001"
        assert captured_lock_config.lock_through_sequence == 1  # seq 2 - 1
        assert captured_lock_config.overrides == {"dp-003": "z_score"}
