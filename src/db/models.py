"""SQLAlchemy ORM models for persistent storage."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base, TimestampMixin, UUIDMixin


class RunRecord(Base, UUIDMixin, TimestampMixin):
    """Database record for an agent run."""

    __tablename__ = "runs"

    task_description: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=True)
    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    decision_points: Mapped[list[DecisionPointRecord]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    execution_records: Mapped[list[ExecutionRecordDB]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    llm_calls: Mapped[list[LLMCallRecordDB]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class DecisionPointRecord(Base, UUIDMixin, TimestampMixin):
    """Database record for a decision point."""

    __tablename__ = "decision_points"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("runs.id"),
        nullable=False,
    )
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    phase: Mapped[str] = mapped_column(String(20), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    alternatives: Mapped[dict] = mapped_column(JSON, nullable=False)
    chosen: Mapped[str] = mapped_column(Text, nullable=False)
    reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    locked: Mapped[bool] = mapped_column(Boolean, default=False)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    # Relationship
    run: Mapped[RunRecord] = relationship(back_populates="decision_points")


class ExecutionRecordDB(Base, UUIDMixin, TimestampMixin):
    """Database record for a code execution."""

    __tablename__ = "execution_records"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("runs.id"),
        nullable=False,
    )
    code_snapshot: Mapped[str] = mapped_column(Text, nullable=False)
    command: Mapped[str] = mapped_column(Text, nullable=False)
    stdout: Mapped[str] = mapped_column(Text, default="")
    stderr: Mapped[str] = mapped_column(Text, default="")
    exit_code: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)

    # Relationship
    run: Mapped[RunRecord] = relationship(back_populates="execution_records")


class LLMCallRecordDB(Base, UUIDMixin, TimestampMixin):
    """Database record for an LLM API call."""

    __tablename__ = "llm_calls"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("runs.id"),
        nullable=False,
    )
    phase: Mapped[str] = mapped_column(String(20), nullable=False)
    messages: Mapped[dict] = mapped_column(JSON, nullable=False)
    response: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    tokens_in: Mapped[int] = mapped_column(Integer, nullable=False)
    tokens_out: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False)

    # Relationship
    run: Mapped[RunRecord] = relationship(back_populates="llm_calls")
