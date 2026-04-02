"""Models for cross-run path analysis and visualization."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from src.models.enums import DecisionCategory


class DecisionTreeNode(BaseModel):
    """A node in the merged decision tree across runs."""

    model_config = ConfigDict(frozen=True)

    question: str = Field(description="Decision question")
    category: DecisionCategory = Field(description="Decision category")
    branches: list[DecisionTreeBranch] = Field(
        default_factory=list,
        description="Branches for each unique choice",
    )


class DecisionTreeBranch(BaseModel):
    """A branch in the decision tree, representing one choice."""

    model_config = ConfigDict(frozen=True)

    choice: str = Field(description="The choice value")
    run_ids: list[str] = Field(
        default_factory=list,
        description="Run IDs that took this branch",
    )
    count: int = Field(description="Number of runs taking this branch")
    children: list[DecisionTreeNode] = Field(
        default_factory=list,
        description="Child decision nodes",
    )


# Rebuild models for forward references
DecisionTreeNode.model_rebuild()
DecisionTreeBranch.model_rebuild()


class DecisionVariance(BaseModel):
    """Variance metrics for a single decision across runs."""

    model_config = ConfigDict(frozen=True)

    question: str = Field(description="Decision question text")
    category: DecisionCategory = Field(description="Decision category")
    unique_choices: list[str] = Field(description="All unique choices observed")
    entropy: float = Field(
        ge=0.0,
        description="Shannon entropy (higher = more variance)",
    )
    most_common: str = Field(description="Most frequently chosen value")
    samples: int = Field(description="Number of runs with this decision")


class OutcomeCorrelation(BaseModel):
    """Correlation between a decision's choices and outcome metrics."""

    model_config = ConfigDict(frozen=True)

    question: str = Field(description="Decision question text")
    category: DecisionCategory = Field(description="Decision category")
    success_rate_by_choice: dict[str, float] = Field(
        default_factory=dict,
        description="Choice -> execution success rate (0.0-1.0)",
    )
    avg_cost_by_choice: dict[str, float] = Field(
        default_factory=dict,
        description="Choice -> average total cost in USD",
    )
    avg_code_lines_by_choice: dict[str, float] = Field(
        default_factory=dict,
        description="Choice -> average code line count",
    )


class PathAnalysis(BaseModel):
    """Complete cross-run path analysis result."""

    model_config = ConfigDict(frozen=True)

    total_runs: int = Field(description="Number of runs analyzed")
    decision_tree: dict = Field(
        description="JSON-serializable merged decision tree",
    )
    variance_by_decision: list[DecisionVariance] = Field(
        default_factory=list,
        description="Variance metrics per decision, sorted by entropy descending",
    )
    outcome_correlations: list[OutcomeCorrelation] = Field(
        default_factory=list,
        description="Outcome correlations per decision",
    )
