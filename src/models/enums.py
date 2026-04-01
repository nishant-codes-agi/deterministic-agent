"""Enums used across the deterministic agent system."""

from __future__ import annotations

import enum


class RunStatus(str, enum.Enum):
    """Status of an agent run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"  # Crashed mid-run, trace saved


class AgentPhase(str, enum.Enum):
    """Phases of agent execution."""

    PLANNING = "planning"
    CODING = "coding"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    RECOVERING = "recovering"


class DecisionCategory(str, enum.Enum):
    """Categories of decision points."""

    DATA_SELECTION = "data_selection"
    ALGORITHM_SELECTION = "algorithm_selection"
    ARCHITECTURE = "architecture"
    LIBRARY_SELECTION = "library_selection"
    ERROR_RECOVERY = "error_recovery"
    PARAMETER_TUNING = "parameter_tuning"


class VarianceTier(str, enum.Enum):
    """Variance tiers for decision impact scoring."""

    HIGH = "high"      # 7.5-10: fundamentally changes output
    MEDIUM = "medium"  # 5-7: changes presentation
    LOW = "low"        # 2-3.5: cosmetic code differences
    NOOP = "noop"      # 1-1.5: zero output impact, pin by default
