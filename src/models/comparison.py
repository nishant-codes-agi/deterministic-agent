"""Models for comparing runs."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from src.models.enums import DecisionCategory


class DecisionAlignment(BaseModel):
    """How a decision aligns between two runs."""

    model_config = ConfigDict(frozen=True)

    category: DecisionCategory = Field(description="Decision category")
    question: str = Field(description="Normalized question text")
    run_a_choice: Optional[str] = Field(default=None, description="Choice in run A")
    run_b_choice: Optional[str] = Field(default=None, description="Choice in run B")
    match: bool = Field(description="Whether the choices match")


class OutcomeDiff(BaseModel):
    """Outcome differences between two runs."""

    model_config = ConfigDict(frozen=True)

    code_line_count: dict[str, int] = Field(
        default_factory=dict,
        description="Run ID to line count mapping",
    )
    libraries_used: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Run ID to libraries list mapping",
    )
    execution_time_ms: dict[str, int] = Field(
        default_factory=dict,
        description="Run ID to execution time mapping",
    )
    anomalies_found: dict[str, int] = Field(
        default_factory=dict,
        description="Run ID to anomaly count mapping",
    )
    total_cost_usd: dict[str, float] = Field(
        default_factory=dict,
        description="Run ID to total cost mapping",
    )


class RunComparison(BaseModel):
    """Full comparison between two runs."""

    model_config = ConfigDict(frozen=True)

    run_a_id: str = Field(description="First run ID")
    run_b_id: str = Field(description="Second run ID")
    decision_alignment: list[DecisionAlignment] = Field(
        default_factory=list,
        description="Decision-by-decision alignment",
    )
    outcome_diff: OutcomeDiff = Field(description="Outcome differences")
    variance_score: float = Field(
        ge=0.0, le=1.0,
        description="Overall variance score (0=identical, 1=completely different)",
    )
