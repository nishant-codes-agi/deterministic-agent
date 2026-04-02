"""PersistentTracer - decision tracer with incremental JSONL persistence."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from src.cache.tiered import TieredCache
from src.config import LLMConfig
from src.models.decisions import DecisionPoint, ExecutionRecord, LLMCallRecord
from src.models.enums import RunStatus
from src.models.traces import DecisionTrace, RunMetadata
from src.tracing.base import DecisionTracer
from src.tracing.cost_tracker import CostLimitExceededError, CostTracker

logger = logging.getLogger(__name__)


class PersistentTracer(DecisionTracer):
    """Decision tracer with incremental persistence.

    Writes are append-only to JSONL during the run (crash recovery).
    On finalize(), the complete trace is written to both the database
    and the filesystem as a clean JSON file.
    """

    def __init__(
        self,
        run_id: str,
        task: str,
        runs_dir: Path,
        db_session: Optional[AsyncSession] = None,
        cache: Optional[TieredCache] = None,
        llm_config: Optional[LLMConfig] = None,
        cost_tracker: Optional[CostTracker] = None,
    ) -> None:
        self._run_id = run_id
        self._task = task
        self._runs_dir = runs_dir
        self._db_session = db_session
        self._cache = cache
        self._llm_config = llm_config
        self._cost_tracker = cost_tracker

        # In-memory accumulators
        self._decisions: list[DecisionPoint] = []
        self._executions: list[ExecutionRecord] = []
        self._llm_calls: list[LLMCallRecord] = []
        self._started_at = datetime.now(timezone.utc)

        # Set up run directory and JSONL file
        self._run_dir = self._runs_dir / self._run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = self._run_dir / "trace.jsonl"
        # Open in append mode, line-buffered for crash safety
        self._jsonl_file = open(self._jsonl_path, "a")

    async def record_decision(self, decision: DecisionPoint) -> None:
        """Record a decision point. Appends to JSONL and in-memory list."""
        self._decisions.append(decision)
        self._append_jsonl("decision", decision.model_dump(mode="json"))

        # Update cost tracker
        if self._cost_tracker and decision.cost_usd > 0:
            await self._cost_tracker.add_cost(decision.cost_usd)

    async def record_execution(self, execution: ExecutionRecord) -> None:
        """Record an execution attempt."""
        self._executions.append(execution)
        self._append_jsonl("execution", execution.model_dump(mode="json"))

    async def record_llm_call(self, call: LLMCallRecord) -> None:
        """Record an LLM API call."""
        self._llm_calls.append(call)
        self._append_jsonl("llm_call", call.model_dump(mode="json"))

        # Update cost tracker and check limit
        if self._cost_tracker and call.cost_usd > 0:
            await self._cost_tracker.add_cost(call.cost_usd)

    async def get_decisions(self) -> list[DecisionPoint]:
        """Return current in-memory decisions list."""
        return list(self._decisions)

    async def finalize(
        self, status: RunStatus, error: Optional[str] = None
    ) -> DecisionTrace:
        """Finalize the trace: write trace.json, metadata.json, DB, and cache."""
        # Close JSONL file
        self._jsonl_file.close()

        # Build the complete trace
        trace = self._build_trace(status, error)

        # 1. Write trace.json (pretty-printed, complete)
        trace_json_path = self._run_dir / "trace.json"
        trace_json_path.write_text(
            trace.model_dump_json(indent=2)
        )

        # 2. Write metadata.json (just RunMetadata)
        metadata_path = self._run_dir / "metadata.json"
        metadata_path.write_text(
            trace.metadata.model_dump_json(indent=2)
        )

        # 3. Write to database (if session available)
        if self._db_session:
            try:
                await self._write_to_db(trace)
            except Exception as e:
                logger.warning(f"Failed to write trace to database: {e}")

        # 4. Update cache
        if self._cache:
            try:
                cache_key = f"trace:{self._run_id}"
                await self._cache.set(
                    cache_key,
                    trace.model_dump_json(),
                    persist=True,
                )
            except Exception as e:
                logger.warning(f"Failed to cache trace: {e}")

        return trace

    def _build_trace(
        self, status: RunStatus, error: Optional[str] = None
    ) -> DecisionTrace:
        """Construct the complete DecisionTrace from accumulated data."""
        from src.models.enums import AgentPhase

        # Build model routing map from config
        model_routing: dict[str, str] = {}
        if self._llm_config:
            model_routing = {
                "planning": self._llm_config.get_model_for_phase(AgentPhase.PLANNING),
                "coding": self._llm_config.get_model_for_phase(AgentPhase.CODING),
                "evaluating": self._llm_config.get_model_for_phase(AgentPhase.EVALUATING),
                "recovering": self._llm_config.get_model_for_phase(AgentPhase.RECOVERING),
            }

        total_tokens = sum(c.tokens_in + c.tokens_out for c in self._llm_calls)
        total_cost = sum(c.cost_usd for c in self._llm_calls)

        # Use LLM provider name from config
        provider_name = "openrouter"
        if self._llm_config:
            provider_name = self._llm_config.provider

        temperature = 0.0
        if self._llm_config:
            temperature = self._llm_config.temperature

        metadata = RunMetadata(
            run_id=self._run_id,
            task_description=self._task,
            llm_provider=provider_name,
            model_routing=model_routing,
            temperature=temperature,
            started_at=self._started_at,
            completed_at=datetime.now(timezone.utc),
            status=status,
            total_llm_calls=len(self._llm_calls),
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            error=error,
        )

        return DecisionTrace(
            metadata=metadata,
            decisions=self._decisions,
            executions=self._executions,
            llm_calls=self._llm_calls,
        )

    def _append_jsonl(self, record_type: str, data: dict) -> None:
        """Append a record to the JSONL file, flush immediately for crash safety."""
        line = json.dumps({"type": record_type, "data": data}, default=str)
        self._jsonl_file.write(line + "\n")
        self._jsonl_file.flush()

    async def _write_to_db(self, trace: DecisionTrace) -> None:
        """Write the complete trace to the database."""
        from src.db.models import (
            DecisionPointRecord,
            ExecutionRecordDB,
            LLMCallRecordDB,
            RunRecord,
        )

        run_record = RunRecord(
            task_description=trace.metadata.task_description,
            status=trace.metadata.status.value,
            metadata_json=trace.metadata.model_dump(mode="json"),
            total_cost_usd=trace.metadata.total_cost_usd,
            error=trace.metadata.error,
            completed_at=trace.metadata.completed_at,
        )
        self._db_session.add(run_record)
        await self._db_session.flush()

        # Add decision points
        for dp in trace.decisions:
            dp_record = DecisionPointRecord(
                run_id=run_record.id,
                sequence_number=dp.sequence_number,
                category=dp.category.value,
                phase=dp.phase.value,
                question=dp.question,
                alternatives=[a.model_dump(mode="json") for a in dp.alternatives],
                chosen=dp.chosen,
                reasoning=dp.reasoning,
                confidence=dp.confidence,
                locked=dp.locked,
                cost_usd=dp.cost_usd,
            )
            self._db_session.add(dp_record)

        # Add execution records
        for ex in trace.executions:
            ex_record = ExecutionRecordDB(
                run_id=run_record.id,
                code_snapshot=ex.code_snapshot,
                command=ex.command,
                stdout=ex.stdout,
                stderr=ex.stderr,
                exit_code=ex.exit_code,
                duration_ms=ex.duration_ms,
            )
            self._db_session.add(ex_record)

        # Add LLM call records
        for lc in trace.llm_calls:
            lc_record = LLMCallRecordDB(
                run_id=run_record.id,
                phase=lc.phase.value,
                messages=lc.messages,
                response=lc.response,
                model=lc.model,
                tokens_in=lc.tokens_in,
                tokens_out=lc.tokens_out,
                latency_ms=lc.latency_ms,
                cost_usd=lc.cost_usd,
            )
            self._db_session.add(lc_record)

        await self._db_session.commit()
