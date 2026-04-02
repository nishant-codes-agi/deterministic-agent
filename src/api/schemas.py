"""Request/response schemas for the FastAPI endpoints.

Separate from domain models to decouple API contract from internal models.
Error responses follow RFC 7807 (Problem Details for HTTP APIs).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Error Response (RFC 7807) ───────────────────────────────────────────────


class ErrorResponse(BaseModel):
    """RFC 7807 Problem Details error response."""

    type: str = Field(
        default="about:blank",
        description="URI reference identifying the problem type",
    )
    title: str = Field(description="Short human-readable summary")
    status: int = Field(description="HTTP status code")
    detail: Optional[str] = Field(
        default=None,
        description="Human-readable explanation specific to this occurrence",
    )
    instance: Optional[str] = Field(
        default=None,
        description="URI reference identifying the specific occurrence",
    )


# ── Run Schemas ─────────────────────────────────────────────────────────────


class RunCreateRequest(BaseModel):
    """Request to start a new agent run."""

    task: str = Field(description="Natural language task description")
    model_override: Optional[str] = Field(
        default=None,
        description="Override all phases to use one model",
    )
    max_cost_usd: Optional[float] = Field(
        default=None,
        description="Max cost budget in USD",
    )
    max_iterations: Optional[int] = Field(
        default=None,
        description="Max agent loop iterations",
    )


class RunCreateResponse(BaseModel):
    """Response after starting a run (returned immediately)."""

    run_id: str = Field(description="Unique run identifier")
    status: str = Field(default="accepted", description="Current status")
    poll_url: str = Field(description="URL to poll for status updates")


class RunMetadataResponse(BaseModel):
    """Run metadata response."""

    run_id: str
    task_description: str
    status: str
    llm_provider: str
    model_routing: dict[str, str] = Field(default_factory=dict)
    temperature: float
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    parent_run_id: Optional[str] = None
    fork_point: Optional[str] = None
    total_llm_calls: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    error: Optional[str] = None


class RunListResponse(BaseModel):
    """Paginated list of runs."""

    runs: list[RunMetadataResponse]
    total: int
    limit: int
    offset: int


# ── Replay / Fork Schemas ──────────────────────────────────────────────────


class ReplayRequest(BaseModel):
    """Request to replay a run with locked decisions."""

    model_override: Optional[str] = Field(
        default=None,
        description="Override model for replay",
    )


class ForkRequest(BaseModel):
    """Request to fork from a decision point."""

    decision_id: str = Field(
        description="Decision ID or sequence number to override",
    )
    choice: str = Field(description="New choice value for the decision")


# ── Compare Schemas ─────────────────────────────────────────────────────────


class CompareRequest(BaseModel):
    """Request to compare two runs."""

    run_a_id: str = Field(description="First run ID")
    run_b_id: str = Field(description="Second run ID")


class CompareResponse(BaseModel):
    """Comparison result between two runs."""

    run_a_id: str
    run_b_id: str
    decision_alignment: list[dict[str, Any]]
    outcome_diff: dict[str, Any]
    variance_score: float


# ── Estimate Schemas ────────────────────────────────────────────────────────


class EstimateRequest(BaseModel):
    """Request for cost estimation."""

    task: str = Field(description="Task description")
    model: Optional[str] = Field(
        default=None,
        description="Single model override for estimation",
    )


class EstimateResponse(BaseModel):
    """Cost estimate response."""

    model_routing: dict[str, str]
    estimated_cost_usd: float
    confidence_low: float
    confidence_high: float
    breakdown: dict[str, float]


# ── Analysis Schemas ────────────────────────────────────────────────────────


class AnalysisResponse(BaseModel):
    """Cross-run path analysis response."""

    total_runs: int
    decision_tree: dict[str, Any]
    variance_by_decision: list[dict[str, Any]]
    outcome_correlations: list[dict[str, Any]]


# ── Trace Schemas ───────────────────────────────────────────────────────────


class TraceResponse(BaseModel):
    """Full decision trace response."""

    metadata: RunMetadataResponse
    decisions: list[dict[str, Any]]
    executions: list[dict[str, Any]]
    llm_calls: list[dict[str, Any]]


class DeleteResponse(BaseModel):
    """Delete run response."""

    run_id: str
    deleted: bool
