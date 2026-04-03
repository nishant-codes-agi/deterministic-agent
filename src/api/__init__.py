"""FastAPI application entry point.

Includes CORS middleware, request ID middleware, exception handlers
for domain errors, and all API routes.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Global container instance (type is src.dependencies.Container, deferred to avoid import chain)
_container = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown of the dependency container."""
    global _container
    from src.config import get_settings
    from src.dependencies import Container

    settings = get_settings()
    _container = Container(settings)
    await _container.startup()
    logger.info("Application startup complete")
    yield
    if _container:
        await _container.shutdown()
    logger.info("Application shutdown complete")


app = FastAPI(
    title="Deterministic Agent API",
    description=(
        "A coding agent with decision tracing and path locking. "
        "Captures every decision as a traceable, replayable path. "
        "Supports fork-and-explore, cross-run analysis, and cost optimization "
        "via per-phase model routing through OpenRouter."
    ),
    version="0.1.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "ops", "description": "Operational endpoints"},
        {"name": "runs", "description": "Agent run management"},
        {"name": "analysis", "description": "Cross-run analysis and comparison"},
    ],
)


# ── CORS Middleware ──────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request ID Middleware ────────────────────────────────────────────────────


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Adds X-Request-ID header to every request/response."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", uuid4().hex[:12])
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


app.add_middleware(RequestIDMiddleware)


# ── Exception Handlers ──────────────────────────────────────────────────────


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catch-all handler returning RFC 7807 error."""
    logger.exception(f"Unhandled exception: {exc}")
    
    if request.scope.get("type") == "websocket":
        return

    return JSONResponse(
        status_code=500,
        content={
            "type": "about:blank",
            "title": "Internal Server Error",
            "status": 500,
            "detail": str(exc),
        },
    )


try:
    from src.locking.exceptions import PathLockViolationError

    @app.exception_handler(PathLockViolationError)
    async def path_lock_violation_handler(
        request: Request, exc: PathLockViolationError
    ):
        """Handle path lock violation errors."""
        return JSONResponse(
            status_code=409,
            content={
                "type": "urn:deterministic-agent:path-lock-violation",
                "title": "Path Lock Violation",
                "status": 409,
                "detail": str(exc),
                "question": exc.question,
                "expected": exc.expected,
                "got": exc.got,
                "attempts": exc.attempts,
            },
        )
except ImportError:
    pass

try:
    from src.tracing.cost_tracker import CostLimitExceededError

    @app.exception_handler(CostLimitExceededError)
    async def cost_limit_handler(
        request: Request, exc: CostLimitExceededError
    ):
        """Handle cost limit exceeded errors."""
        return JSONResponse(
            status_code=402,
            content={
                "type": "urn:deterministic-agent:cost-limit-exceeded",
                "title": "Cost Limit Exceeded",
                "status": 402,
                "detail": str(exc),
                "current_cost": exc.current_cost,
                "max_cost": exc.max_cost,
            },
        )
except ImportError:
    pass


# ── Health Endpoint ─────────────────────────────────────────────────────────


@app.get("/health", tags=["ops"])
async def health() -> JSONResponse:
    """Health check endpoint.

    Returns the status of the API and its backing services (DB, Redis).
    """
    from src.config import get_settings
    settings = get_settings()
    services: dict[str, Any] = {}

    # Check DB
    if _container and _container.get_db_session_factory() is not None:
        try:
            from sqlalchemy import text

            factory = _container.get_db_session_factory()
            async with factory() as session:
                await session.execute(text("SELECT 1"))
            services["database"] = "ok"
        except Exception as e:
            logger.warning(f"DB health check failed: {e}")
            services["database"] = "unavailable"
    else:
        services["database"] = "unavailable"

    # Check Redis
    if _container and _container.get_redis() is not None:
        try:
            await _container.get_redis().ping()
            services["redis"] = "ok"
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            services["redis"] = "unavailable"
    else:
        services["redis"] = "unavailable"

    all_ok = all(v == "ok" for v in services.values())
    status_code = 200 if all_ok else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if all_ok else "degraded",
            "services": services,
            "version": app.version,
            "env": {
                "llm_provider": settings.llm.provider,
                "llm_model_planning": settings.llm.model_planning,
            },
        },
    )


@app.get("/", include_in_schema=False)
async def root():
    """Redirect hint for the API root."""
    return {"message": "Deterministic Agent API — see /docs or /health"}


# ── Mount Routes ────────────────────────────────────────────────────────────

from src.api.routes import router  # noqa: E402
from src.api.websocket import router as ws_router  # noqa: E402

app.include_router(router)
app.include_router(ws_router)
