"""Trace and run models for the deterministic agent."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.models.decisions import DecisionPoint, ExecutionRecord, LLMCallRecord
from src.models.enums import DecisionCategory, RunStatus


class PathLockConfig(BaseModel):
    """Configuration for path locking on a replay or fork."""

    model_config = ConfigDict(frozen=True)

    source_run_id: str = Field(description="Run ID to lock from")
    locked_decision_ids: Optional[list[str]] = Field(
        default=None,
        description="Specific decision IDs to lock",
    )
    lock_through_sequence: Optional[int] = Field(
        default=None,
        description="Lock all decisions up to this sequence number",
    )
    lock_categories: Optional[list[DecisionCategory]] = Field(
        default=None,
        description="Lock all decisions in these categories",
    )
    overrides: dict[str, str] = Field(
        default_factory=dict,
        description="Decision ID to new value overrides (for forks)",
    )


class CostEstimate(BaseModel):
    """Predicted cost for a run before execution."""

    model_config = ConfigDict(frozen=True)

    model_routing: dict[str, str] = Field(
        description="Phase to model mapping being estimated",
    )
    estimated_tokens_in: int = Field(description="Estimated input tokens")
    estimated_tokens_out: int = Field(description="Estimated output tokens")
    estimated_cost_usd: float = Field(description="Estimated total cost in USD")
    confidence_low: float = Field(description="Low end of confidence interval")
    confidence_high: float = Field(description="High end of confidence interval")
    breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Cost breakdown by phase",
    )
    cost_by_model: dict[str, float] = Field(
        default_factory=dict,
        description="Cost breakdown by model",
    )


class RunMetadata(BaseModel):
    """Metadata about a single agent run."""

    run_id: str = Field(
        default_factory=lambda: f"run-{uuid4().hex[:8]}",
        description="Unique run ID",
    )
    task_description: str = Field(description="Natural language task description")
    llm_provider: str = Field(description="LLM provider used, e.g. 'openrouter'")
    model_routing: dict[str, str] = Field(
        default_factory=dict,
        description="Map of agent phase to model used in this run",
    )
    temperature: float = Field(description="Temperature setting used")
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the run started",
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="When the run completed",
    )
    status: RunStatus = Field(
        default=RunStatus.PENDING,
        description="Current run status",
    )
    parent_run_id: Optional[str] = Field(
        default=None,
        description="Parent run ID if this is a fork",
    )
    fork_point: Optional[str] = Field(
        default=None,
        description="Decision ID where fork occurred",
    )
    lock_config: Optional[PathLockConfig] = Field(
        default=None,
        description="Path locking configuration if any",
    )
    total_llm_calls: int = Field(default=0, description="Total number of LLM calls")
    total_tokens: int = Field(default=0, description="Total tokens used")
    total_cost_usd: float = Field(default=0.0, description="Total cost in USD")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class DecisionTrace(BaseModel):
    """Complete trace of a single agent run."""

    metadata: RunMetadata = Field(description="Run metadata")
    decisions: list[DecisionPoint] = Field(
        default_factory=list,
        description="All decision points in order",
    )
    executions: list[ExecutionRecord] = Field(
        default_factory=list,
        description="All execution records",
    )
    llm_calls: list[LLMCallRecord] = Field(
        default_factory=list,
        description="All LLM call records",
    )
