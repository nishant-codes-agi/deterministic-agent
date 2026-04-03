"""ForkEngine - orchestrates fork-and-explore operations.

A fork takes a source run, locks all decisions upstream of the fork point,
overrides the fork-point decision, and lets downstream decisions run freely.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.agent.base import CodingAgent, EventHandler
from src.analysis.comparator import RunComparator
from src.models.comparison import RunComparison
from src.models.traces import DecisionTrace, PathLockConfig
from src.tracing.store import TraceStore

logger = logging.getLogger(__name__)


class ForkEngine:
    """Orchestrates fork-and-explore operations.

    A fork takes a source run, locks all decisions upstream of the fork point,
    overrides the fork-point decision, and lets downstream decisions run freely.
    """

    def __init__(self, agent: CodingAgent, trace_store: TraceStore) -> None:
        self._agent = agent
        self._store = trace_store
        self._comparator = RunComparator()

    async def fork(
        self,
        source_run_id: str,
        decision_id: str,
        new_choice: str,
        event_handler: Optional[EventHandler] = None,
        run_id: Optional[str] = None,
    ) -> tuple[DecisionTrace, RunComparison]:
        """Fork a run at a specific decision point.

        Locks all decisions upstream of the fork point, overrides the fork-point
        decision with new_choice, and lets downstream decisions run freely.

        Args:
            source_run_id: The run ID to fork from.
            decision_id: Decision ID or sequence number string to override.
            new_choice: The new value for the fork-point decision.
            event_handler: Optional event handler for real-time output.

        Returns:
            Tuple of (new trace, comparison with source run).

        Raises:
            ValueError: If source run or decision not found.
        """
        source = await self._store.get_trace(source_run_id)
        if not source:
            raise ValueError(f"Source run {source_run_id} not found")

        # Find the fork point by ID or sequence number
        fork_decision = None
        for d in source.decisions:
            if d.id == decision_id:
                fork_decision = d
                break
            try:
                if str(d.sequence_number) == decision_id:
                    fork_decision = d
                    break
            except (ValueError, TypeError):
                pass

        if not fork_decision:
            raise ValueError(
                f"Decision {decision_id} not found in run {source_run_id}"
            )

        logger.info(
            f"Forking run {source_run_id} at decision "
            f"{fork_decision.sequence_number} ({fork_decision.question}): "
            f"'{fork_decision.chosen}' -> '{new_choice}'"
        )

        # Build lock config: lock upstream, override fork point, free downstream
        lock_config = PathLockConfig(
            source_run_id=source_run_id,
            lock_through_sequence=fork_decision.sequence_number - 1,
            overrides={fork_decision.id: new_choice},
        )

        # Run the agent with this lock config
        forked_trace = await self._agent.run(
            task=source.metadata.task_description,
            lock_config=lock_config,
            event_handler=event_handler,
            run_id=run_id,
        )

        # Set fork metadata
        forked_trace.metadata.parent_run_id = source_run_id
        forked_trace.metadata.fork_point = fork_decision.id

        # Generate comparison
        comparison = self._comparator.compare(source, forked_trace)

        return forked_trace, comparison
