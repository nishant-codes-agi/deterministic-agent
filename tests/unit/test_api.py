"""Unit tests for FastAPI endpoints (Phase 10)."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.models.decisions import Alternative, DecisionPoint, ExecutionRecord
from src.models.enums import AgentPhase, DecisionCategory, RunStatus
from src.models.traces import DecisionTrace, RunMetadata


# ── Test Fixtures ───────────────────────────────────────────────────────────


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
    run_id: str = "run-test001",
    status: RunStatus = RunStatus.SUCCESS,
    cost: float = 0.05,
) -> DecisionTrace:
    decisions = [
        _make_decision(0, "Which stock ticker?", DecisionCategory.DATA_SELECTION, "AAPL", "dp-001"),
        _make_decision(1, "Which anomaly method?", DecisionCategory.ALGORITHM_SELECTION, "IQR", "dp-002"),
    ]
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
        executions=[
            ExecutionRecord(
                code_snapshot="print('hello')",
                command="python main.py",
                stdout="hello",
                stderr="",
                exit_code=0,
                duration_ms=100,
            )
        ],
    )


def _write_trace(runs_dir: Path, trace: DecisionTrace) -> None:
    run_dir = runs_dir / trace.metadata.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "trace.json").write_text(trace.model_dump_json(indent=2))
    (run_dir / "metadata.json").write_text(
        trace.metadata.model_dump_json(indent=2)
    )


@pytest.fixture
def tmp_runs_dir():
    d = tempfile.mkdtemp(prefix="test_api_runs_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def mock_container(tmp_runs_dir):
    """Create a mock container for API testing."""
    container = MagicMock()
    container.settings = MagicMock()
    container.settings.runs_dir = tmp_runs_dir
    container.settings.llm = MagicMock()
    container.settings.llm.provider = "openrouter"
    container.settings.llm.model_planning = "test/planning"
    container.settings.llm.get_model_for_phase = lambda phase: f"test/{phase.value}"
    container.settings.llm.model_copy = lambda update: container.settings.llm
    container.settings.agent = MagicMock()
    container.settings.agent.max_cost_usd = 2.0
    container.settings.agent.max_iterations = 10
    container.get_cache.return_value = None
    container.get_db_session_factory.return_value = None
    container.get_llm.return_value = MagicMock()
    container.get_sandbox.return_value = MagicMock()
    container.get_redis.return_value = None
    return container


@pytest.fixture
def client(mock_container):
    """Create a test client with mocked container."""
    import src.api as api_module
    api_module._container = mock_container
    return TestClient(api_module.app, raise_server_exceptions=False)


# ── Health & Root Tests ─────────────────────────────────────────────────────


class TestHealthEndpoint:

    def test_health_returns_response(self, client):
        """Health endpoint returns valid JSON."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "services" in data
        assert "version" in data

    def test_root_returns_message(self, client):
        """Root endpoint returns hint message."""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()


# ── Runs List & Get ─────────────────────────────────────────────────────────


class TestListRuns:

    def test_list_runs_empty(self, client):
        """List runs returns empty when no runs exist."""
        response = client.get("/api/v1/runs")
        assert response.status_code == 200
        data = response.json()
        assert data["runs"] == []
        assert data["total"] == 0

    def test_list_runs_with_data(self, client, tmp_runs_dir):
        """List runs returns stored runs."""
        trace = _make_trace("run-abc12345")
        _write_trace(tmp_runs_dir, trace)

        response = client.get("/api/v1/runs")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        assert any(r["run_id"] == "run-abc12345" for r in data["runs"])

    def test_list_runs_with_status_filter(self, client, tmp_runs_dir):
        """List runs filters by status."""
        _write_trace(tmp_runs_dir, _make_trace("run-ok1", RunStatus.SUCCESS))
        _write_trace(tmp_runs_dir, _make_trace("run-fail1", RunStatus.FAILED))

        response = client.get("/api/v1/runs?status=success")
        assert response.status_code == 200
        data = response.json()
        for r in data["runs"]:
            assert r["status"] == "success"

    def test_list_runs_pagination(self, client, tmp_runs_dir):
        """List runs respects limit and offset."""
        for i in range(5):
            _write_trace(tmp_runs_dir, _make_trace(f"run-page{i:03d}"))

        response = client.get("/api/v1/runs?limit=2&offset=0")
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 2
        assert len(data["runs"]) <= 2


class TestGetRun:

    def test_get_run_found(self, client, tmp_runs_dir):
        """Get run metadata for existing run."""
        trace = _make_trace("run-found01")
        _write_trace(tmp_runs_dir, trace)

        response = client.get("/api/v1/runs/run-found01")
        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == "run-found01"
        assert data["status"] == "success"
        assert data["task_description"] == "Analyze stock data"

    def test_get_run_not_found(self, client):
        """Get run returns 404 for missing run."""
        response = client.get("/api/v1/runs/run-nonexistent")
        assert response.status_code == 404


# ── Trace ───────────────────────────────────────────────────────────────────


class TestGetTrace:

    def test_get_trace_found(self, client, tmp_runs_dir):
        """Get full trace for existing run."""
        trace = _make_trace("run-trace01")
        _write_trace(tmp_runs_dir, trace)

        response = client.get("/api/v1/runs/run-trace01/trace")
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["run_id"] == "run-trace01"
        assert len(data["decisions"]) == 2
        assert len(data["executions"]) == 1

    def test_get_trace_not_found(self, client):
        """Get trace returns 404 for missing run."""
        response = client.get("/api/v1/runs/run-missing/trace")
        assert response.status_code == 404


# ── Delete ──────────────────────────────────────────────────────────────────


class TestDeleteRun:

    def test_delete_run(self, client, tmp_runs_dir):
        """Delete existing run removes it."""
        trace = _make_trace("run-delete01")
        _write_trace(tmp_runs_dir, trace)

        response = client.delete("/api/v1/runs/run-delete01")
        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] is True

        # Verify it's gone
        response = client.get("/api/v1/runs/run-delete01")
        assert response.status_code == 404

    def test_delete_run_not_found(self, client):
        """Delete missing run returns 404."""
        response = client.delete("/api/v1/runs/run-ghost")
        assert response.status_code == 404


# ── Compare ─────────────────────────────────────────────────────────────────


class TestCompare:

    def test_compare_two_runs(self, client, tmp_runs_dir):
        """Compare two runs returns alignment and variance."""
        _write_trace(tmp_runs_dir, _make_trace("run-cmp-a"))
        _write_trace(tmp_runs_dir, _make_trace("run-cmp-b"))

        response = client.post(
            "/api/v1/compare",
            json={"run_a_id": "run-cmp-a", "run_b_id": "run-cmp-b"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["run_a_id"] == "run-cmp-a"
        assert data["run_b_id"] == "run-cmp-b"
        assert "decision_alignment" in data
        assert "variance_score" in data
        assert data["variance_score"] == 0.0  # identical runs

    def test_compare_missing_run(self, client, tmp_runs_dir):
        """Compare with missing run returns 404."""
        _write_trace(tmp_runs_dir, _make_trace("run-exists"))

        response = client.post(
            "/api/v1/compare",
            json={"run_a_id": "run-exists", "run_b_id": "run-missing"},
        )
        assert response.status_code == 404


# ── Analysis ────────────────────────────────────────────────────────────────


class TestAnalysis:

    def test_analysis_all_runs(self, client, tmp_runs_dir):
        """Analysis endpoint returns tree and variance data."""
        _write_trace(tmp_runs_dir, _make_trace("run-an-a"))
        _write_trace(tmp_runs_dir, _make_trace("run-an-b"))

        response = client.get("/api/v1/analysis")
        assert response.status_code == 200
        data = response.json()
        assert data["total_runs"] == 2
        assert "decision_tree" in data
        assert "variance_by_decision" in data

    def test_analysis_specific_runs(self, client, tmp_runs_dir):
        """Analysis with specific run IDs."""
        _write_trace(tmp_runs_dir, _make_trace("run-spec-a"))
        _write_trace(tmp_runs_dir, _make_trace("run-spec-b"))
        _write_trace(tmp_runs_dir, _make_trace("run-spec-c"))

        response = client.get("/api/v1/analysis?runs=run-spec-a,run-spec-b")
        assert response.status_code == 200
        data = response.json()
        assert data["total_runs"] == 2

    def test_analysis_empty(self, client):
        """Analysis with no runs returns empty."""
        response = client.get("/api/v1/analysis")
        assert response.status_code == 200
        data = response.json()
        assert data["total_runs"] == 0


# ── Estimate ────────────────────────────────────────────────────────────────


class TestEstimate:

    def test_estimate_default(self, client):
        """Estimate returns cost breakdown."""
        response = client.post(
            "/api/v1/estimate",
            json={"task": "Analyze stock anomalies"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "model_routing" in data
        assert "estimated_cost_usd" in data
        assert data["estimated_cost_usd"] > 0
        assert "breakdown" in data
        assert "confidence_low" in data
        assert "confidence_high" in data
        assert data["confidence_low"] < data["estimated_cost_usd"]
        assert data["confidence_high"] > data["estimated_cost_usd"]


# ── Fork Validation ─────────────────────────────────────────────────────────


class TestForkValidation:

    def test_fork_missing_source(self, client):
        """Fork with missing source run returns 404."""
        response = client.post(
            "/api/v1/runs/run-noexist/fork",
            json={"decision_id": "dp-001", "choice": "TSLA"},
        )
        assert response.status_code == 404

    def test_fork_missing_decision(self, client, tmp_runs_dir):
        """Fork with invalid decision ID returns 404."""
        _write_trace(tmp_runs_dir, _make_trace("run-forktest"))

        response = client.post(
            "/api/v1/runs/run-forktest/fork",
            json={"decision_id": "dp-999", "choice": "TSLA"},
        )
        assert response.status_code == 404

    def test_fork_valid_request_accepted(self, client, tmp_runs_dir):
        """Fork with valid source and decision returns 202-like accepted."""
        _write_trace(tmp_runs_dir, _make_trace("run-forkok"))

        response = client.post(
            "/api/v1/runs/run-forkok/fork",
            json={"decision_id": "dp-001", "choice": "TSLA"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"
        assert "poll_url" in data


# ── Replay Validation ───────────────────────────────────────────────────────


class TestReplayValidation:

    def test_replay_missing_source(self, client):
        """Replay with missing source returns 404."""
        response = client.post(
            "/api/v1/runs/run-noexist/replay",
            json={},
        )
        assert response.status_code == 404

    def test_replay_valid_request_accepted(self, client, tmp_runs_dir):
        """Replay with valid source returns accepted."""
        _write_trace(tmp_runs_dir, _make_trace("run-repok"))

        response = client.post(
            "/api/v1/runs/run-repok/replay",
            json={},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"


# ── Middleware Tests ────────────────────────────────────────────────────────


class TestRequestIDMiddleware:

    def test_response_has_request_id(self, client):
        """Every response should have X-Request-ID header."""
        response = client.get("/health")
        assert "X-Request-ID" in response.headers

    def test_echoes_provided_request_id(self, client):
        """If client sends X-Request-ID, it is echoed back."""
        response = client.get("/health", headers={"X-Request-ID": "my-id-123"})
        assert response.headers["X-Request-ID"] == "my-id-123"


# ── Run Create ──────────────────────────────────────────────────────────────


class TestCreateRun:

    def test_create_run_accepted(self, client):
        """Create run returns accepted with poll URL."""
        response = client.post(
            "/api/v1/runs",
            json={"task": "Analyze stock data"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"
        assert "poll_url" in data
        assert data["poll_url"].startswith("/api/v1/runs/")
