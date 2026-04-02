"""Unit tests for PathAnalyzer (Phase 9)."""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path

import pytest

from src.analysis.path_analyzer import PathAnalyzer
from src.models.analysis import (
    DecisionVariance,
    OutcomeCorrelation,
    PathAnalysis,
)
from src.models.decisions import Alternative, DecisionPoint, ExecutionRecord
from src.models.enums import AgentPhase, DecisionCategory, RunStatus
from src.models.traces import DecisionTrace, RunMetadata
from src.tracing.store import TraceStore


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_decision(
    seq: int,
    question: str,
    category: DecisionCategory,
    chosen: str,
    dp_id: str | None = None,
) -> DecisionPoint:
    return DecisionPoint(
        id=dp_id or f"dp-{seq:03d}",
        sequence_number=seq,
        category=category,
        phase=AgentPhase.PLANNING,
        question=question,
        alternatives=[
            Alternative(value=chosen, reasoning="Chosen"),
            Alternative(value="other", reasoning="Other"),
        ],
        chosen=chosen,
        reasoning=f"Chose {chosen}",
        confidence=0.8,
    )


def _make_trace(
    run_id: str,
    decisions: list[DecisionPoint],
    status: RunStatus = RunStatus.SUCCESS,
    cost: float = 0.05,
    executions: list[ExecutionRecord] | None = None,
) -> DecisionTrace:
    return DecisionTrace(
        metadata=RunMetadata(
            run_id=run_id,
            task_description="Analyze stock data",
            llm_provider="openrouter",
            model_routing={"planning": "test-model"},
            temperature=0.0,
            status=status,
            total_cost_usd=cost,
        ),
        decisions=decisions,
        executions=executions or [],
    )


def _write_trace_to_disk(runs_dir: Path, trace: DecisionTrace) -> None:
    run_dir = runs_dir / trace.metadata.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "trace.json").write_text(trace.model_dump_json(indent=2))
    (run_dir / "metadata.json").write_text(
        trace.metadata.model_dump_json(indent=2)
    )


@pytest.fixture
def tmp_runs_dir():
    d = tempfile.mkdtemp(prefix="test_analysis_runs_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def _build_five_traces() -> list[DecisionTrace]:
    """Build 5 traces with known variance patterns."""
    traces = []

    # Run 1: AAPL, 2y, IQR, fastapi, matplotlib
    traces.append(_make_trace("run-001", [
        _make_decision(0, "Which stock ticker?", DecisionCategory.DATA_SELECTION, "AAPL"),
        _make_decision(1, "What timeframe?", DecisionCategory.PARAMETER_TUNING, "2y"),
        _make_decision(2, "Which anomaly method?", DecisionCategory.ALGORITHM_SELECTION, "IQR"),
        _make_decision(3, "Which API framework?", DecisionCategory.LIBRARY_SELECTION, "fastapi"),
        _make_decision(4, "Which visualization library?", DecisionCategory.LIBRARY_SELECTION, "matplotlib"),
    ], cost=0.05, executions=[ExecutionRecord(
        code_snapshot="import pandas\nimport numpy\nprint('done')\n" * 10,
        command="python main.py", stdout="done", stderr="", exit_code=0, duration_ms=500,
    )]))

    # Run 2: AAPL, 2y, z_score, fastapi, plotly
    traces.append(_make_trace("run-002", [
        _make_decision(0, "Which stock ticker?", DecisionCategory.DATA_SELECTION, "AAPL"),
        _make_decision(1, "What timeframe?", DecisionCategory.PARAMETER_TUNING, "2y"),
        _make_decision(2, "Which anomaly method?", DecisionCategory.ALGORITHM_SELECTION, "z_score"),
        _make_decision(3, "Which API framework?", DecisionCategory.LIBRARY_SELECTION, "fastapi"),
        _make_decision(4, "Which visualization library?", DecisionCategory.LIBRARY_SELECTION, "plotly"),
    ], cost=0.08, executions=[ExecutionRecord(
        code_snapshot="import pandas\nimport plotly\nprint('done')\n" * 12,
        command="python main.py", stdout="done", stderr="", exit_code=0, duration_ms=600,
    )]))

    # Run 3: TSLA, 1y, isolation_forest, fastapi, matplotlib
    traces.append(_make_trace("run-003", [
        _make_decision(0, "Which stock ticker?", DecisionCategory.DATA_SELECTION, "TSLA"),
        _make_decision(1, "What timeframe?", DecisionCategory.PARAMETER_TUNING, "1y"),
        _make_decision(2, "Which anomaly method?", DecisionCategory.ALGORITHM_SELECTION, "isolation_forest"),
        _make_decision(3, "Which API framework?", DecisionCategory.LIBRARY_SELECTION, "fastapi"),
        _make_decision(4, "Which visualization library?", DecisionCategory.LIBRARY_SELECTION, "matplotlib"),
    ], cost=0.10, executions=[ExecutionRecord(
        code_snapshot="import pandas\nimport sklearn\nprint('done')\n" * 15,
        command="python main.py", stdout="done", stderr="", exit_code=0, duration_ms=800,
    )]))

    # Run 4: TSLA, 2y, IQR, fastapi, seaborn (FAILED)
    traces.append(_make_trace("run-004", [
        _make_decision(0, "Which stock ticker?", DecisionCategory.DATA_SELECTION, "TSLA"),
        _make_decision(1, "What timeframe?", DecisionCategory.PARAMETER_TUNING, "2y"),
        _make_decision(2, "Which anomaly method?", DecisionCategory.ALGORITHM_SELECTION, "IQR"),
        _make_decision(3, "Which API framework?", DecisionCategory.LIBRARY_SELECTION, "fastapi"),
        _make_decision(4, "Which visualization library?", DecisionCategory.LIBRARY_SELECTION, "seaborn"),
    ], status=RunStatus.FAILED, cost=0.04, executions=[ExecutionRecord(
        code_snapshot="import pandas\nimport seaborn\nraise Exception('fail')\n" * 8,
        command="python main.py", stdout="", stderr="Error", exit_code=1, duration_ms=200,
    )]))

    # Run 5: SPY, 1y, z_score, fastapi, matplotlib
    traces.append(_make_trace("run-005", [
        _make_decision(0, "Which stock ticker?", DecisionCategory.DATA_SELECTION, "SPY"),
        _make_decision(1, "What timeframe?", DecisionCategory.PARAMETER_TUNING, "1y"),
        _make_decision(2, "Which anomaly method?", DecisionCategory.ALGORITHM_SELECTION, "z_score"),
        _make_decision(3, "Which API framework?", DecisionCategory.LIBRARY_SELECTION, "fastapi"),
        _make_decision(4, "Which visualization library?", DecisionCategory.LIBRARY_SELECTION, "matplotlib"),
    ], cost=0.06, executions=[ExecutionRecord(
        code_snapshot="import pandas\nimport numpy\nprint('ok')\n" * 11,
        command="python main.py", stdout="ok", stderr="", exit_code=0, duration_ms=450,
    )]))

    return traces


# ── Decision Tree Tests ─────────────────────────────────────────────────────


class TestBuildDecisionTree:
    """Test decision tree construction."""

    def test_tree_has_all_decisions(self):
        """Decision tree includes nodes for each unique question."""
        traces = _build_five_traces()
        analyzer = PathAnalyzer(trace_store=None)  # type: ignore
        tree = analyzer.build_decision_tree(traces)

        # Root has one branch "root" whose children are decision nodes
        assert len(tree.branches) == 1
        children = tree.branches[0].children
        # 5 distinct decision questions
        assert len(children) == 5

    def test_tree_branches_have_correct_counts(self):
        """Each branch shows the correct number of runs."""
        traces = _build_five_traces()
        analyzer = PathAnalyzer(trace_store=None)  # type: ignore
        tree = analyzer.build_decision_tree(traces)

        children = tree.branches[0].children
        # First node = "Which stock ticker?"
        ticker_node = children[0]
        assert ticker_node.question == "Which stock ticker?"

        ticker_choices = {b.choice: b.count for b in ticker_node.branches}
        assert ticker_choices["AAPL"] == 2
        assert ticker_choices["TSLA"] == 2
        assert ticker_choices["SPY"] == 1

    def test_tree_branches_have_run_ids(self):
        """Each branch lists the run IDs that took that path."""
        traces = _build_five_traces()
        analyzer = PathAnalyzer(trace_store=None)  # type: ignore
        tree = analyzer.build_decision_tree(traces)

        children = tree.branches[0].children
        ticker_node = children[0]

        aapl_branch = next(b for b in ticker_node.branches if b.choice == "AAPL")
        assert set(aapl_branch.run_ids) == {"run-001", "run-002"}

    def test_tree_empty_traces(self):
        """Empty traces produce a minimal tree."""
        analyzer = PathAnalyzer(trace_store=None)  # type: ignore
        tree = analyzer.build_decision_tree([])
        assert tree.question == "(no decisions)"

    def test_tree_is_json_serializable(self):
        """The decision tree can be serialized to JSON."""
        traces = _build_five_traces()
        analyzer = PathAnalyzer(trace_store=None)  # type: ignore
        tree = analyzer.build_decision_tree(traces)
        data = tree.model_dump(mode="json")
        assert isinstance(data, dict)
        assert "branches" in data


# ── Variance Tests ──────────────────────────────────────────────────────────


class TestComputeVariance:
    """Test variance computation."""

    def test_uniform_decision_has_zero_entropy(self):
        """A decision where all runs chose the same value has entropy 0."""
        traces = _build_five_traces()
        analyzer = PathAnalyzer(trace_store=None)  # type: ignore
        variances = analyzer.compute_variance_by_decision(traces)

        # "Which API framework?" -> all chose "fastapi"
        framework_var = next(
            v for v in variances if v.question == "Which API framework?"
        )
        assert framework_var.entropy == 0.0
        assert framework_var.unique_choices == ["fastapi"]
        assert framework_var.most_common == "fastapi"
        assert framework_var.samples == 5

    def test_high_variance_has_high_entropy(self):
        """A decision with many unique choices has high entropy."""
        traces = _build_five_traces()
        analyzer = PathAnalyzer(trace_store=None)  # type: ignore
        variances = analyzer.compute_variance_by_decision(traces)

        # "Which anomaly method?" -> IQR(2), z_score(2), isolation_forest(1)
        method_var = next(
            v for v in variances if v.question == "Which anomaly method?"
        )
        assert len(method_var.unique_choices) == 3
        assert method_var.entropy > 1.0  # log2(3) ~ 1.58 for uniform

    def test_sorted_by_entropy_descending(self):
        """Variance list is sorted highest entropy first."""
        traces = _build_five_traces()
        analyzer = PathAnalyzer(trace_store=None)  # type: ignore
        variances = analyzer.compute_variance_by_decision(traces)

        entropies = [v.entropy for v in variances]
        assert entropies == sorted(entropies, reverse=True)

    def test_highest_variance_has_multiple_choices(self):
        """The highest variance decision should have 3 unique choices."""
        traces = _build_five_traces()
        analyzer = PathAnalyzer(trace_store=None)  # type: ignore
        variances = analyzer.compute_variance_by_decision(traces)

        # Ticker, anomaly method, and viz library all have 3 unique choices
        highest = variances[0]
        assert len(highest.unique_choices) == 3
        assert highest.entropy > 1.0

    def test_binary_decision_entropy(self):
        """A 3:2 split should have entropy between 0 and 1."""
        traces = _build_five_traces()
        analyzer = PathAnalyzer(trace_store=None)  # type: ignore
        variances = analyzer.compute_variance_by_decision(traces)

        # "What timeframe?" -> 2y(3), 1y(2)
        timeframe_var = next(
            v for v in variances if v.question == "What timeframe?"
        )
        assert 0 < timeframe_var.entropy < 1.0
        assert timeframe_var.most_common == "2y"
        assert timeframe_var.samples == 5

    def test_empty_traces_returns_empty(self):
        """No traces returns empty variance list."""
        analyzer = PathAnalyzer(trace_store=None)  # type: ignore
        variances = analyzer.compute_variance_by_decision([])
        assert variances == []


# ── Outcome Correlation Tests ───────────────────────────────────────────────


class TestCorrelateWithOutcomes:
    """Test outcome correlation computation."""

    def test_success_rate_computed(self):
        """Success rate is computed per choice."""
        traces = _build_five_traces()
        analyzer = PathAnalyzer(trace_store=None)  # type: ignore
        correlations = analyzer.correlate_with_outcomes(traces)

        # Find "Which stock ticker?"
        ticker_corr = next(
            c for c in correlations if c.question == "Which stock ticker?"
        )
        # AAPL: both runs succeeded -> 1.0
        assert ticker_corr.success_rate_by_choice["AAPL"] == 1.0
        # TSLA: run-003 succeeded, run-004 failed -> 0.5
        assert ticker_corr.success_rate_by_choice["TSLA"] == 0.5
        # SPY: run-005 succeeded -> 1.0
        assert ticker_corr.success_rate_by_choice["SPY"] == 1.0

    def test_cost_computed(self):
        """Average cost is computed per choice."""
        traces = _build_five_traces()
        analyzer = PathAnalyzer(trace_store=None)  # type: ignore
        correlations = analyzer.correlate_with_outcomes(traces)

        ticker_corr = next(
            c for c in correlations if c.question == "Which stock ticker?"
        )
        # AAPL: (0.05 + 0.08) / 2 = 0.065
        assert abs(ticker_corr.avg_cost_by_choice["AAPL"] - 0.065) < 0.001

    def test_code_lines_computed(self):
        """Average code line count is computed per choice."""
        traces = _build_five_traces()
        analyzer = PathAnalyzer(trace_store=None)  # type: ignore
        correlations = analyzer.correlate_with_outcomes(traces)

        ticker_corr = next(
            c for c in correlations if c.question == "Which stock ticker?"
        )
        # AAPL: run-001 has 30 lines, run-002 has 36 lines -> avg 33
        assert ticker_corr.avg_code_lines_by_choice["AAPL"] > 0

    def test_empty_traces_returns_empty(self):
        """No traces returns empty correlations."""
        analyzer = PathAnalyzer(trace_store=None)  # type: ignore
        correlations = analyzer.correlate_with_outcomes([])
        assert correlations == []


# ── Full Analysis Integration ───────────────────────────────────────────────


class TestFullAnalysis:
    """Test the full analyze() method with TraceStore."""

    @pytest.mark.asyncio
    async def test_analyze_all_runs(self, tmp_runs_dir):
        """Analyze all runs from disk."""
        traces = _build_five_traces()
        for t in traces:
            _write_trace_to_disk(tmp_runs_dir, t)

        store = TraceStore(runs_dir=tmp_runs_dir)
        analyzer = PathAnalyzer(trace_store=store)
        result = await analyzer.analyze()

        assert result.total_runs == 5
        assert len(result.variance_by_decision) == 5
        assert len(result.outcome_correlations) == 5
        assert isinstance(result.decision_tree, dict)

    @pytest.mark.asyncio
    async def test_analyze_specific_runs(self, tmp_runs_dir):
        """Analyze only specified run IDs."""
        traces = _build_five_traces()
        for t in traces:
            _write_trace_to_disk(tmp_runs_dir, t)

        store = TraceStore(runs_dir=tmp_runs_dir)
        analyzer = PathAnalyzer(trace_store=store)
        result = await analyzer.analyze(run_ids=["run-001", "run-002"])

        assert result.total_runs == 2

    @pytest.mark.asyncio
    async def test_analyze_empty_dir(self, tmp_runs_dir):
        """Analyze with no runs returns empty analysis."""
        store = TraceStore(runs_dir=tmp_runs_dir)
        analyzer = PathAnalyzer(trace_store=store)
        result = await analyzer.analyze()

        assert result.total_runs == 0
        assert result.variance_by_decision == []
        assert result.outcome_correlations == []

    @pytest.mark.asyncio
    async def test_analyze_missing_run_id_skipped(self, tmp_runs_dir):
        """Missing run IDs are skipped gracefully."""
        traces = _build_five_traces()[:2]
        for t in traces:
            _write_trace_to_disk(tmp_runs_dir, t)

        store = TraceStore(runs_dir=tmp_runs_dir)
        analyzer = PathAnalyzer(trace_store=store)
        result = await analyzer.analyze(run_ids=["run-001", "run-999"])

        assert result.total_runs == 1

    @pytest.mark.asyncio
    async def test_analysis_result_serializable(self, tmp_runs_dir):
        """PathAnalysis can be serialized to JSON."""
        traces = _build_five_traces()
        for t in traces:
            _write_trace_to_disk(tmp_runs_dir, t)

        store = TraceStore(runs_dir=tmp_runs_dir)
        analyzer = PathAnalyzer(trace_store=store)
        result = await analyzer.analyze()

        data = result.model_dump(mode="json")
        assert isinstance(data, dict)
        assert data["total_runs"] == 5
