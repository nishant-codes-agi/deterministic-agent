"""FastAPI routes for the Deterministic Agent API.

Long-running operations (run, replay, fork) use BackgroundTasks and
return immediately with a run_id + status polling URL.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request

from src.api.schemas import (
    AnalysisResponse,
    CompareRequest,
    CompareResponse,
    DeleteResponse,
    EstimateRequest,
    EstimateResponse,
    ForkRequest,
    ReplayRequest,
    RunCreateRequest,
    RunCreateResponse,
    RunListResponse,
    RunMetadataResponse,
    TraceResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


def _get_container(request: Request):
    """Get the DI container from app state."""
    from src.api import _container
    if _container is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return _container


def _build_event_publisher(container):
    """Build a RedisEventPublisher from the container (gracefully None-safe)."""
    from src.streaming.publisher import RedisEventPublisher
    return RedisEventPublisher(redis_cache=container.get_redis())


def _build_agent(container, model_override: Optional[str] = None):
    """Build an AgentRunner from the container."""
    from src.agent.runner import AgentRunner

    settings = container.settings
    llm_config = settings.llm
    if model_override:
        llm_config = llm_config.model_copy(update={"model_override": model_override})

    return AgentRunner(
        llm=container.get_llm(),
        sandbox=container.get_sandbox(),
        config=settings.agent,
        llm_config=llm_config,
        runs_dir=settings.runs_dir,
        cache=container.get_cache(),
        db_session_factory=container.get_db_session_factory(),
    )


def _build_trace_store(container):
    """Build a TraceStore from the container."""
    from src.tracing.store import TraceStore
    settings = container.settings
    return TraceStore(
        runs_dir=settings.runs_dir,
        db_session=None,
        cache=container.get_cache(),
    )


def _meta_to_response(meta) -> RunMetadataResponse:
    """Convert RunMetadata to response schema."""
    return RunMetadataResponse(
        run_id=meta.run_id,
        task_description=meta.task_description,
        status=meta.status.value,
        llm_provider=meta.llm_provider,
        model_routing=meta.model_routing,
        temperature=meta.temperature,
        started_at=meta.started_at,
        completed_at=meta.completed_at,
        parent_run_id=meta.parent_run_id,
        fork_point=meta.fork_point,
        total_llm_calls=meta.total_llm_calls,
        total_tokens=meta.total_tokens,
        total_cost_usd=meta.total_cost_usd,
        error=meta.error,
    )


# ── Runs ────────────────────────────────────────────────────────────────────


@router.post("/runs", response_model=RunCreateResponse, tags=["runs"])
async def create_run(
    body: RunCreateRequest,
    background_tasks: BackgroundTasks,
    request: Request,
):
    """Start a new agent run (background task)."""
    container = _get_container(request)
    settings = container.settings

    if body.max_cost_usd is not None:
        settings.agent.max_cost_usd = body.max_cost_usd
    if body.max_iterations is not None:
        settings.agent.max_iterations = body.max_iterations

    agent = _build_agent(container, model_override=body.model_override)
    event_publisher = _build_event_publisher(container)

    # We don't know the run_id until the agent starts, so we generate one
    from uuid import uuid4
    run_id = f"run-{uuid4().hex[:8]}"

    async def _run_task():
        try:
            await agent.run(task=body.task, event_handler=event_publisher, run_id=run_id)
        except Exception as e:
            logger.exception(f"Background run failed: {e}")

    background_tasks.add_task(_run_task)

    return RunCreateResponse(
        run_id=run_id,
        status="accepted",
        poll_url=f"/api/v1/runs/{run_id}",
    )


@router.get("/runs", response_model=RunListResponse, tags=["runs"])
async def list_runs(
    request: Request,
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List runs with pagination and optional status filter."""
    container = _get_container(request)
    store = _build_trace_store(container)

    from src.models.enums import RunStatus
    status_enum = RunStatus(status) if status else None

    runs = await store.list_runs(status=status_enum, limit=limit, offset=offset)
    return RunListResponse(
        runs=[_meta_to_response(m) for m in runs],
        total=len(runs),
        limit=limit,
        offset=offset,
    )


@router.get("/runs/{run_id}", response_model=RunMetadataResponse, tags=["runs"])
async def get_run(run_id: str, request: Request):
    """Get run metadata."""
    container = _get_container(request)
    store = _build_trace_store(container)

    trace = await store.get_trace(run_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return _meta_to_response(trace.metadata)


@router.get("/runs/{run_id}/trace", response_model=TraceResponse, tags=["runs"])
async def get_trace(run_id: str, request: Request):
    """Get the full decision trace for a run."""
    container = _get_container(request)
    store = _build_trace_store(container)

    trace = await store.get_trace(run_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return TraceResponse(
        metadata=_meta_to_response(trace.metadata),
        decisions=[d.model_dump(mode="json") for d in trace.decisions],
        executions=[e.model_dump(mode="json") for e in trace.executions],
        llm_calls=[c.model_dump(mode="json") for c in trace.llm_calls],
    )


@router.delete("/runs/{run_id}", response_model=DeleteResponse, tags=["runs"])
async def delete_run(run_id: str, request: Request):
    """Delete a run and its artifacts."""
    container = _get_container(request)
    store = _build_trace_store(container)

    deleted = await store.delete_run(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return DeleteResponse(run_id=run_id, deleted=True)


# ── Replay ──────────────────────────────────────────────────────────────────


@router.post(
    "/runs/{run_id}/replay",
    response_model=RunCreateResponse,
    tags=["runs"],
)
async def replay_run(
    run_id: str,
    body: ReplayRequest,
    background_tasks: BackgroundTasks,
    request: Request,
):
    """Replay a run with all decisions locked."""
    container = _get_container(request)
    store = _build_trace_store(container)

    source_trace = await store.get_trace(run_id)
    if not source_trace:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    from src.models.traces import PathLockConfig
    lock_config = PathLockConfig(source_run_id=run_id)
    agent = _build_agent(container, model_override=body.model_override)
    event_publisher = _build_event_publisher(container)

    async def _replay_task():
        try:
            await agent.run(
                task=source_trace.metadata.task_description,
                lock_config=lock_config,
                event_handler=event_publisher,
                run_id=new_run_id,
            )
        except Exception as e:
            logger.exception(f"Background replay failed: {e}")

    background_tasks.add_task(_replay_task)

    from uuid import uuid4
    new_run_id = f"run-{uuid4().hex[:8]}"
    return RunCreateResponse(
        run_id=new_run_id,
        status="accepted",
        poll_url=f"/api/v1/runs/{new_run_id}",
    )


# ── Fork ────────────────────────────────────────────────────────────────────


@router.post(
    "/runs/{run_id}/fork",
    response_model=RunCreateResponse,
    tags=["runs"],
)
async def fork_run(
    run_id: str,
    body: ForkRequest,
    background_tasks: BackgroundTasks,
    request: Request,
):
    """Fork a run from a specific decision point."""
    container = _get_container(request)
    store = _build_trace_store(container)

    # Validate source run exists
    source_trace = await store.get_trace(run_id)
    if not source_trace:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    # Validate decision exists
    found = False
    for dp in source_trace.decisions:
        if dp.id == body.decision_id or str(dp.sequence_number) == body.decision_id:
            found = True
            break
    if not found:
        raise HTTPException(
            status_code=404,
            detail=f"Decision {body.decision_id} not found in run {run_id}",
        )

    agent = _build_agent(container)
    event_publisher = _build_event_publisher(container)

    from src.forking.engine import ForkEngine
    engine = ForkEngine(agent=agent, trace_store=store)

    async def _fork_task():
        try:
            await engine.fork(
                source_run_id=run_id,
                decision_id=body.decision_id,
                new_choice=body.choice,
                event_handler=event_publisher,
                run_id=new_run_id,
            )
        except Exception as e:
            logger.exception(f"Background fork failed: {e}")

    background_tasks.add_task(_fork_task)

    from uuid import uuid4
    new_run_id = f"run-{uuid4().hex[:8]}"
    return RunCreateResponse(
        run_id=new_run_id,
        status="accepted",
        poll_url=f"/api/v1/runs/{new_run_id}",
    )


# ── Compare ─────────────────────────────────────────────────────────────────


@router.post("/compare", response_model=CompareResponse, tags=["analysis"])
async def compare_runs(body: CompareRequest, request: Request):
    """Compare two runs side by side."""
    container = _get_container(request)
    store = _build_trace_store(container)

    trace_a = await store.get_trace(body.run_a_id)
    if not trace_a:
        raise HTTPException(
            status_code=404, detail=f"Run {body.run_a_id} not found"
        )
    trace_b = await store.get_trace(body.run_b_id)
    if not trace_b:
        raise HTTPException(
            status_code=404, detail=f"Run {body.run_b_id} not found"
        )

    from src.analysis.comparator import RunComparator
    comparison = RunComparator().compare(trace_a, trace_b)

    return CompareResponse(
        run_a_id=comparison.run_a_id,
        run_b_id=comparison.run_b_id,
        decision_alignment=[
            a.model_dump(mode="json") for a in comparison.decision_alignment
        ],
        outcome_diff=comparison.outcome_diff.model_dump(mode="json"),
        variance_score=comparison.variance_score,
    )


# ── Analysis ────────────────────────────────────────────────────────────────


@router.get("/analysis", response_model=AnalysisResponse, tags=["analysis"])
async def get_analysis(
    request: Request,
    runs: Optional[str] = Query(
        None, description="Comma-separated run IDs (default: all)"
    ),
):
    """Cross-run path analysis: decision tree, variance, outcome correlation."""
    container = _get_container(request)
    store = _build_trace_store(container)

    from src.analysis.path_analyzer import PathAnalyzer
    analyzer = PathAnalyzer(trace_store=store)

    run_ids = [r.strip() for r in runs.split(",")] if runs else None
    result = await analyzer.analyze(run_ids=run_ids)

    return AnalysisResponse(
        total_runs=result.total_runs,
        decision_tree=result.decision_tree,
        variance_by_decision=[
            v.model_dump(mode="json") for v in result.variance_by_decision
        ],
        outcome_correlations=[
            c.model_dump(mode="json") for c in result.outcome_correlations
        ],
    )


# ── Estimate ────────────────────────────────────────────────────────────────


@router.post("/estimate", response_model=EstimateResponse, tags=["analysis"])
async def estimate_cost(body: EstimateRequest, request: Request):
    """Estimate cost without running."""
    container = _get_container(request)
    settings = container.settings

    from src.models.enums import AgentPhase

    llm_config = settings.llm
    if body.model:
        llm_config = llm_config.model_copy(update={"model_override": body.model})

    phase_estimates = {
        "planning": {"model": llm_config.get_model_for_phase(AgentPhase.PLANNING), "cost": 0.006},
        "coding": {"model": llm_config.get_model_for_phase(AgentPhase.CODING), "cost": 0.072},
        "evaluation": {"model": llm_config.get_model_for_phase(AgentPhase.EVALUATING), "cost": 0.001},
        "recovery": {"model": llm_config.get_model_for_phase(AgentPhase.RECOVERING), "cost": 0.027},
    }

    total = sum(p["cost"] for p in phase_estimates.values())
    routing = {phase: info["model"] for phase, info in phase_estimates.items()}
    breakdown = {phase: info["cost"] for phase, info in phase_estimates.items()}

    return EstimateResponse(
        model_routing=routing,
        estimated_cost_usd=round(total, 4),
        confidence_low=round(total * 0.6, 4),
        confidence_high=round(total * 1.8, 4),
        breakdown=breakdown,
    )
