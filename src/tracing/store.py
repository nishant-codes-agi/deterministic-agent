"""TraceStore - read-side for traces from cache, database, and filesystem."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.cache.tiered import TieredCache
from src.models.decisions import DecisionPoint
from src.models.enums import RunStatus
from src.models.traces import DecisionTrace, RunMetadata

logger = logging.getLogger(__name__)


class TraceStore:
    """Reads and queries traces from database and filesystem."""

    def __init__(
        self,
        runs_dir: Path,
        db_session: Optional[AsyncSession] = None,
        cache: Optional[TieredCache] = None,
    ) -> None:
        self._runs_dir = runs_dir
        self._db_session = db_session
        self._cache = cache

    async def get_trace(self, run_id: str) -> Optional[DecisionTrace]:
        """Get a complete trace by run ID. Checks cache, then DB, then filesystem."""
        # 1. Check cache
        if self._cache:
            try:
                cached = await self._cache.get(f"trace:{run_id}")
                if cached:
                    logger.debug(f"Trace cache hit: {run_id}")
                    return DecisionTrace.model_validate_json(cached)
            except Exception as e:
                logger.warning(f"Cache read failed for {run_id}: {e}")

        # 2. Check database
        if self._db_session:
            try:
                trace = await self._load_from_db(run_id)
                if trace:
                    # Backfill cache
                    if self._cache:
                        await self._cache.set(
                            f"trace:{run_id}",
                            trace.model_dump_json(),
                        )
                    return trace
            except Exception as e:
                logger.warning(f"DB read failed for {run_id}: {e}")

        # 3. Check filesystem
        trace = self._load_from_filesystem(run_id)
        if trace:
            # Backfill cache
            if self._cache:
                try:
                    await self._cache.set(
                        f"trace:{run_id}",
                        trace.model_dump_json(),
                    )
                except Exception:
                    pass
            return trace

        return None

    async def list_runs(
        self,
        status: Optional[RunStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[RunMetadata]:
        """List run metadata with optional filtering."""
        # Try database first for indexed queries
        if self._db_session:
            try:
                return await self._list_from_db(status, limit, offset)
            except Exception as e:
                logger.warning(f"DB list failed: {e}")

        # Fall back to filesystem
        return self._list_from_filesystem(status, limit, offset)

    async def get_decisions_for_run(self, run_id: str) -> list[DecisionPoint]:
        """Get just the decisions (lighter than full trace)."""
        trace = await self.get_trace(run_id)
        if trace:
            return trace.decisions
        return []

    async def delete_run(self, run_id: str) -> bool:
        """Delete a run's trace, artifacts, and DB records."""
        deleted = False

        # Delete from cache
        if self._cache:
            try:
                await self._cache.delete(f"trace:{run_id}")
            except Exception:
                pass

        # Delete from database
        if self._db_session:
            try:
                from src.db.models import RunRecord
                result = await self._db_session.execute(
                    select(RunRecord).where(
                        RunRecord.task_description.contains(run_id)
                    )
                )
                run = result.scalar_one_or_none()
                if run:
                    await self._db_session.delete(run)
                    await self._db_session.commit()
                    deleted = True
            except Exception as e:
                logger.warning(f"DB delete failed for {run_id}: {e}")

        # Delete from filesystem
        import shutil
        run_dir = self._runs_dir / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)
            deleted = True

        return deleted

    async def search_runs(
        self,
        ticker: Optional[str] = None,
        method: Optional[str] = None,
        min_cost: Optional[float] = None,
        max_cost: Optional[float] = None,
    ) -> list[RunMetadata]:
        """Search runs by decision content."""
        all_runs = await self.list_runs(limit=1000)
        results = []

        for meta in all_runs:
            # Cost filter
            if min_cost is not None and meta.total_cost_usd < min_cost:
                continue
            if max_cost is not None and meta.total_cost_usd > max_cost:
                continue

            # For ticker/method filtering, we need to load the trace
            if ticker or method:
                trace = await self.get_trace(meta.run_id)
                if not trace:
                    continue

                if ticker:
                    found = any(
                        ticker.lower() in dp.chosen.lower()
                        for dp in trace.decisions
                        if dp.category.value == "data_selection"
                    )
                    if not found:
                        continue

                if method:
                    found = any(
                        method.lower() in dp.chosen.lower()
                        for dp in trace.decisions
                        if dp.category.value == "algorithm_selection"
                    )
                    if not found:
                        continue

            results.append(meta)

        return results

    def _load_from_filesystem(self, run_id: str) -> Optional[DecisionTrace]:
        """Load a trace from the filesystem."""
        trace_path = self._runs_dir / run_id / "trace.json"
        if not trace_path.exists():
            return None

        try:
            raw = trace_path.read_text()
            return DecisionTrace.model_validate_json(raw)
        except Exception as e:
            logger.warning(f"Failed to load trace from filesystem: {e}")
            return None

    def _list_from_filesystem(
        self,
        status: Optional[RunStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[RunMetadata]:
        """List runs from filesystem metadata files."""
        if not self._runs_dir.exists():
            return []

        metadata_list: list[RunMetadata] = []
        run_dirs = sorted(
            (d for d in self._runs_dir.iterdir() if d.is_dir()),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )

        for run_dir in run_dirs:
            metadata_path = run_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            try:
                raw = metadata_path.read_text()
                meta = RunMetadata.model_validate_json(raw)
                if status and meta.status != status:
                    continue
                metadata_list.append(meta)
            except Exception as e:
                logger.warning(f"Failed to read metadata from {metadata_path}: {e}")
                continue

        return metadata_list[offset : offset + limit]

    async def _load_from_db(self, run_id: str) -> Optional[DecisionTrace]:
        """Load a trace from the database by matching run_id in metadata_json."""
        from src.db.models import RunRecord
        from sqlalchemy import cast, String

        result = await self._db_session.execute(
            select(RunRecord).where(
                RunRecord.metadata_json["run_id"].astext == run_id
            )
        )
        run = result.scalar_one_or_none()
        if not run:
            return None

        # Reconstruct from DB metadata_json which stores the full RunMetadata
        try:
            metadata = RunMetadata.model_validate(run.metadata_json)
        except Exception:
            return None

        # Load related records
        from src.db.models import DecisionPointRecord, ExecutionRecordDB, LLMCallRecordDB
        from src.models.decisions import Alternative
        from src.models.enums import AgentPhase, DecisionCategory

        dp_result = await self._db_session.execute(
            select(DecisionPointRecord)
            .where(DecisionPointRecord.run_id == run.id)
            .order_by(DecisionPointRecord.sequence_number)
        )
        dp_records = dp_result.scalars().all()

        decisions = []
        for dpr in dp_records:
            alternatives = [
                Alternative(**a) if isinstance(a, dict) else a
                for a in (dpr.alternatives or [])
            ]
            dp = DecisionPoint(
                sequence_number=dpr.sequence_number,
                category=DecisionCategory(dpr.category),
                phase=AgentPhase(dpr.phase),
                question=dpr.question,
                alternatives=alternatives,
                chosen=dpr.chosen,
                reasoning=dpr.reasoning,
                confidence=dpr.confidence,
                locked=dpr.locked,
                cost_usd=dpr.cost_usd,
            )
            decisions.append(dp)

        ex_result = await self._db_session.execute(
            select(ExecutionRecordDB)
            .where(ExecutionRecordDB.run_id == run.id)
            .order_by(ExecutionRecordDB.created_at)
        )
        ex_records = ex_result.scalars().all()

        from src.models.decisions import ExecutionRecord
        executions = [
            ExecutionRecord(
                code_snapshot=ex.code_snapshot,
                command=ex.command,
                stdout=ex.stdout,
                stderr=ex.stderr,
                exit_code=ex.exit_code,
                duration_ms=ex.duration_ms,
            )
            for ex in ex_records
        ]

        lc_result = await self._db_session.execute(
            select(LLMCallRecordDB)
            .where(LLMCallRecordDB.run_id == run.id)
            .order_by(LLMCallRecordDB.created_at)
        )
        lc_records = lc_result.scalars().all()

        from src.models.decisions import LLMCallRecord
        llm_calls = [
            LLMCallRecord(
                phase=AgentPhase(lc.phase),
                messages=lc.messages,
                response=lc.response,
                model=lc.model,
                temperature=0.0,
                tokens_in=lc.tokens_in,
                tokens_out=lc.tokens_out,
                latency_ms=lc.latency_ms,
                cost_usd=lc.cost_usd,
            )
            for lc in lc_records
        ]

        return DecisionTrace(
            metadata=metadata,
            decisions=decisions,
            executions=executions,
            llm_calls=llm_calls,
        )

    async def _list_from_db(
        self,
        status: Optional[RunStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[RunMetadata]:
        """List runs from database."""
        from src.db.models import RunRecord

        query = select(RunRecord).order_by(RunRecord.created_at.desc())
        if status:
            query = query.where(RunRecord.status == status.value)
        query = query.offset(offset).limit(limit)

        result = await self._db_session.execute(query)
        runs = result.scalars().all()

        metadata_list = []
        for run in runs:
            try:
                meta = RunMetadata.model_validate(run.metadata_json)
                metadata_list.append(meta)
            except Exception:
                continue

        return metadata_list
