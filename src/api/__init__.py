"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.config import get_settings
from src.dependencies import Container

logger = logging.getLogger(__name__)

# Global container instance
_container: Container | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown of the dependency container."""
    global _container
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
    description="A coding agent with decision tracing and path locking",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["ops"])
async def health() -> JSONResponse:
    """Health check endpoint.

    Returns the status of the API and its backing services (DB, Redis).
    """
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
