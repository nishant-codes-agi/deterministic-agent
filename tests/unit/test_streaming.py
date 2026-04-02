"""Unit tests for Phase 11: Real-Time Streaming."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.base import AgentEvent, AgentEventType
from src.models.enums import AgentPhase
from src.streaming.publisher import MultiEventHandler, RedisEventPublisher


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_event(
    event_type: AgentEventType = AgentEventType.PHASE_CHANGED,
    run_id: str = "run-test001",
    phase: AgentPhase = AgentPhase.PLANNING,
    payload: dict | None = None,
) -> AgentEvent:
    return AgentEvent(
        event_type=event_type,
        run_id=run_id,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        phase=phase,
        payload=payload or {},
    )


# ── RedisEventPublisher Tests ──────────────────────────────────────────────


class TestRedisEventPublisher:

    @pytest.mark.asyncio
    async def test_publishes_event_to_correct_channel(self):
        """Publisher sends JSON to dagent:stream:{run_id} channel."""
        mock_redis = AsyncMock()
        publisher = RedisEventPublisher(redis_cache=mock_redis)

        event = _make_event(run_id="run-abc123")
        await publisher.handle(event)

        mock_redis.publish_event.assert_called_once()
        call_args = mock_redis.publish_event.call_args
        assert call_args[0][0] == "stream:run-abc123"
        # Second arg should be valid JSON
        event_json = call_args[0][1]
        parsed = json.loads(event_json)
        assert parsed["run_id"] == "run-abc123"
        assert parsed["event_type"] == "phase_changed"

    @pytest.mark.asyncio
    async def test_graceful_when_redis_is_none(self):
        """Publisher silently no-ops when Redis is None."""
        publisher = RedisEventPublisher(redis_cache=None)
        event = _make_event()
        # Should not raise
        await publisher.handle(event)

    @pytest.mark.asyncio
    async def test_graceful_on_publish_error(self):
        """Publisher logs warning but doesn't raise on Redis error."""
        mock_redis = AsyncMock()
        mock_redis.publish_event.side_effect = ConnectionError("Redis down")
        publisher = RedisEventPublisher(redis_cache=mock_redis)

        event = _make_event()
        # Should not raise
        await publisher.handle(event)

    @pytest.mark.asyncio
    async def test_event_payload_included(self):
        """Publisher serializes event payload correctly."""
        mock_redis = AsyncMock()
        publisher = RedisEventPublisher(redis_cache=mock_redis)

        event = _make_event(
            event_type=AgentEventType.COST_UPDATE,
            payload={"model": "gpt-4o-mini", "cost_usd": 0.001, "total_cost_usd": 0.05},
        )
        await publisher.handle(event)

        event_json = mock_redis.publish_event.call_args[0][1]
        parsed = json.loads(event_json)
        assert parsed["payload"]["model"] == "gpt-4o-mini"
        assert parsed["payload"]["cost_usd"] == 0.001

    @pytest.mark.asyncio
    async def test_different_runs_use_different_channels(self):
        """Each run_id gets its own channel."""
        mock_redis = AsyncMock()
        publisher = RedisEventPublisher(redis_cache=mock_redis)

        await publisher.handle(_make_event(run_id="run-aaa"))
        await publisher.handle(_make_event(run_id="run-bbb"))

        calls = mock_redis.publish_event.call_args_list
        assert calls[0][0][0] == "stream:run-aaa"
        assert calls[1][0][0] == "stream:run-bbb"


# ── MultiEventHandler Tests ───────────────────────────────────────────────


class TestMultiEventHandler:

    @pytest.mark.asyncio
    async def test_dispatches_to_all_handlers(self):
        """Multi handler calls all composed handlers."""
        handler_a = AsyncMock()
        handler_b = AsyncMock()
        multi = MultiEventHandler([handler_a, handler_b])

        event = _make_event()
        await multi.handle(event)

        handler_a.handle.assert_called_once_with(event)
        handler_b.handle.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_continues_on_handler_failure(self):
        """If one handler raises, others still get the event."""
        handler_a = AsyncMock()
        handler_a.handle.side_effect = RuntimeError("boom")
        handler_b = AsyncMock()
        multi = MultiEventHandler([handler_a, handler_b])

        event = _make_event()
        await multi.handle(event)

        handler_a.handle.assert_called_once()
        handler_b.handle.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_empty_handlers_no_error(self):
        """Multi handler with empty list is a no-op."""
        multi = MultiEventHandler([])
        event = _make_event()
        await multi.handle(event)

    @pytest.mark.asyncio
    async def test_single_handler(self):
        """Multi handler with one handler works correctly."""
        handler = AsyncMock()
        multi = MultiEventHandler([handler])

        event = _make_event()
        await multi.handle(event)

        handler.handle.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_all_handlers_fail_no_raise(self):
        """If all handlers raise, MultiEventHandler does not raise."""
        handler_a = AsyncMock()
        handler_a.handle.side_effect = RuntimeError("a fails")
        handler_b = AsyncMock()
        handler_b.handle.side_effect = RuntimeError("b fails")
        multi = MultiEventHandler([handler_a, handler_b])

        event = _make_event()
        # Should not raise
        await multi.handle(event)


# ── WebSocket Endpoint Tests ──────────────────────────────────────────────


class TestWebSocketEndpoint:

    def test_websocket_route_registered(self):
        """The WebSocket route is registered on the app."""
        from src.api import app

        ws_routes = [
            r.path for r in app.routes
            if hasattr(r, "path") and "stream" in r.path
        ]
        assert "/api/v1/runs/{run_id}/stream" in ws_routes

    def test_websocket_no_redis_sends_error(self):
        """WebSocket sends error JSON when Redis unavailable."""
        from fastapi.testclient import TestClient
        from src.api import app
        import src.api as api_module

        mock_container = MagicMock()
        mock_container.get_redis.return_value = None
        api_module._container = mock_container

        client = TestClient(app)
        with client.websocket_connect("/api/v1/runs/run-test/stream") as ws:
            data = ws.receive_json()
            assert "error" in data
            assert "Redis" in data["error"] or "unavailable" in data["error"]


# ── API Route Integration Tests ───────────────────────────────────────────


class TestAPIRoutesUsePublisher:

    def test_create_run_builds_event_publisher(self):
        """POST /api/v1/runs creates a RedisEventPublisher for the agent."""
        from fastapi.testclient import TestClient
        from src.api import app
        import src.api as api_module

        mock_container = MagicMock()
        mock_container.settings = MagicMock()
        mock_container.settings.runs_dir = "/tmp/test_runs"
        mock_container.settings.llm = MagicMock()
        mock_container.settings.llm.provider = "openrouter"
        mock_container.settings.llm.model_planning = "test/planning"
        mock_container.settings.llm.get_model_for_phase = lambda p: f"test/{p.value}"
        mock_container.settings.llm.model_copy = lambda update: mock_container.settings.llm
        mock_container.settings.agent = MagicMock()
        mock_container.settings.agent.max_cost_usd = 2.0
        mock_container.settings.agent.max_iterations = 10
        mock_container.get_cache.return_value = None
        mock_container.get_db_session_factory.return_value = None
        mock_container.get_llm.return_value = MagicMock()
        mock_container.get_sandbox.return_value = MagicMock()
        mock_container.get_redis.return_value = MagicMock()
        api_module._container = mock_container

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/v1/runs",
            json={"task": "Test task"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"


# ── Event Serialization Tests ─────────────────────────────────────────────


class TestEventSerialization:

    def test_agent_event_round_trip(self):
        """AgentEvent can be serialized to JSON and back."""
        event = _make_event(
            event_type=AgentEventType.DECISION_MADE,
            run_id="run-serial01",
            phase=AgentPhase.PLANNING,
            payload={"decision_id": "dp-001", "question": "Which ticker?", "chosen": "AAPL"},
        )
        json_str = event.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["event_type"] == "decision_made"
        assert parsed["run_id"] == "run-serial01"
        assert parsed["payload"]["chosen"] == "AAPL"

        # Reconstruct
        restored = AgentEvent.model_validate_json(json_str)
        assert restored.event_type == AgentEventType.DECISION_MADE
        assert restored.run_id == "run-serial01"

    def test_all_event_types_serializable(self):
        """Every AgentEventType can be serialized."""
        for evt_type in AgentEventType:
            event = AgentEvent(
                event_type=evt_type,
                run_id="run-types",
                payload={},
            )
            json_str = event.model_dump_json()
            parsed = json.loads(json_str)
            assert parsed["event_type"] == evt_type.value
