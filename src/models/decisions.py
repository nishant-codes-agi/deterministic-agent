"""Decision-related Pydantic models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.models.enums import AgentPhase, DecisionCategory


class Alternative(BaseModel):
    """One option the agent considered at a decision point."""

    model_config = ConfigDict(frozen=True)

    value: str = Field(description="The alternative value, e.g. 'IQR'")
    reasoning: str = Field(default="", description="Why this was considered")


class DecisionPoint(BaseModel):
    """A single branching point in the agent's execution."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(
        default_factory=lambda: f"dp-{uuid4().hex[:8]}",
        description="Unique decision point ID",
    )
    sequence_number: int = Field(description="Ordinal position in the run")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the decision was made",
    )
    category: DecisionCategory = Field(description="Category of this decision")
    phase: AgentPhase = Field(description="Agent phase when decision was made")
    question: str = Field(description="The question being decided")
    alternatives: list[Alternative] = Field(
        min_length=2,
        description="Options considered (minimum 2)",
    )
    chosen: str = Field(description="The chosen alternative value")
    reasoning: str = Field(description="Why this choice was made")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the decision (0.0-1.0)",
    )
    parent_decision_id: Optional[str] = Field(
        default=None,
        description="ID of the parent decision that constrained this one",
    )
    locked: bool = Field(
        default=False,
        description="Whether this decision was forced by path locking",
    )
    cost_usd: float = Field(default=0.0, description="Cost of the LLM call for this decision")
    tokens_in: int = Field(default=0, description="Input tokens used")
    tokens_out: int = Field(default=0, description="Output tokens used")
    cache_hit: bool = Field(default=False, description="Whether this was served from cache")


class ExecutionRecord(BaseModel):
    """Record of a code execution attempt in the sandbox."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(
        default_factory=lambda: f"exec-{uuid4().hex[:8]}",
        description="Unique execution record ID",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the execution occurred",
    )
    code_snapshot: str = Field(description="Full code at this point")
    command: str = Field(description="Command that was executed")
    stdout: str = Field(default="", description="Standard output")
    stderr: str = Field(default="", description="Standard error")
    exit_code: int = Field(description="Process exit code")
    duration_ms: int = Field(description="Execution duration in milliseconds")
    artifacts_produced: list[str] = Field(
        default_factory=list,
        description="File paths of produced artifacts",
    )


class LLMCallRecord(BaseModel):
    """Record of a single LLM API call."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(
        default_factory=lambda: f"llm-{uuid4().hex[:8]}",
        description="Unique LLM call record ID",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the call was made",
    )
    phase: AgentPhase = Field(description="Agent phase for this call")
    messages: list[dict] = Field(description="Full messages array sent to LLM")
    response: str = Field(description="Raw response content")
    structured_output: Optional[dict] = Field(
        default=None,
        description="Parsed structured output if applicable",
    )
    model: str = Field(description="Model used for this call")
    temperature: float = Field(description="Temperature used")
    tokens_in: int = Field(description="Input tokens")
    tokens_out: int = Field(description="Output tokens")
    latency_ms: int = Field(description="Latency in milliseconds")
    cost_usd: float = Field(description="Estimated cost in USD")
    decision_point_id: Optional[str] = Field(
        default=None,
        description="Associated decision point ID if any",
    )
