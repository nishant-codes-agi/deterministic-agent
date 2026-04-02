"""WebSocket endpoint for real-time agent event streaming.

Subscribes to a Redis pub/sub channel for a given run and forwards
every event to the connected WebSocket client. Falls back to a
``{"error": ...}`` message if Redis is unavailable.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/api/v1/runs/{run_id}/stream")
async def stream_run(websocket: WebSocket, run_id: str):
    """Stream agent events for a run over WebSocket.

    The client connects, and receives JSON-encoded AgentEvent messages
    in real time until the run completes or the client disconnects.
    """
    await websocket.accept()

    # Get Redis from the DI container
    from src.api import _container

    if _container is None or _container.get_redis() is None:
        await websocket.send_json(
            {"error": "Streaming unavailable — Redis not connected"}
        )
        await websocket.close(code=1011)
        return

    redis_cache = _container.get_redis()
    # Access the underlying aioredis client for pub/sub
    pubsub = redis_cache._client.pubsub()
    channel = f"dagent:stream:{run_id}"

    try:
        await pubsub.subscribe(channel)
        logger.info(f"WebSocket subscribed to {channel}")

        while True:
            message = await pubsub.get_message(
                ignore_subscribe_messages=True, timeout=1.0
            )
            if message and message["type"] == "message":
                data = message["data"]
                # data may be bytes or str depending on decode_responses
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                await websocket.send_text(data)

            # Yield control to allow disconnect detection
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from {channel}")
    except Exception as e:
        logger.warning(f"WebSocket stream error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        try:
            await pubsub.unsubscribe(channel)
            await pubsub.close()
        except Exception:
            pass
