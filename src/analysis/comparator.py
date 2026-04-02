"""RunComparator - compares two runs to identify decision-level differences."""

from __future__ import annotations

import logging
import re
from typing import Optional

from src.locking.question_matcher import QuestionMatcher
from src.models.comparison import DecisionAlignment, OutcomeDiff, RunComparison
from src.models.decisions import DecisionPoint
from src.models.traces import DecisionTrace

logger = logging.getLogger(__name__)


class RunComparator:
    """Compares two runs to identify decision-level differences."""

    def compare(
        self, run_a: DecisionTrace, run_b: DecisionTrace
    ) -> RunComparison:
        """Align decisions between two runs and compute differences.

        Uses QuestionMatcher to align decisions that may have different IDs
        but answer the same question.

        Args:
            run_a: The first (usually source) run trace.
            run_b: The second (usually forked) run trace.

        Returns:
            A RunComparison with decision alignment, outcome diff,
            and variance score.
        """
        alignments = []
        matched_b_ids: set[str] = set()

        for dec_a in run_a.decisions:
            matched = QuestionMatcher.match(
                dec_a.question,
                dec_a.category,
                dec_a.sequence_number,
                run_b.decisions,
            )
            if matched:
                matched_b_ids.add(matched.id)
                alignments.append(
                    DecisionAlignment(
                        category=dec_a.category,
                        question=dec_a.question,
                        run_a_choice=dec_a.chosen,
                        run_b_choice=matched.chosen,
                        match=self._values_match(dec_a.chosen, matched.chosen),
                    )
                )
            else:
                alignments.append(
                    DecisionAlignment(
                        category=dec_a.category,
                        question=dec_a.question,
                        run_a_choice=dec_a.chosen,
                        run_b_choice=None,
                        match=False,
                    )
                )

        # Add decisions in B that have no match in A
        for dec_b in run_b.decisions:
            if dec_b.id not in matched_b_ids:
                alignments.append(
                    DecisionAlignment(
                        category=dec_b.category,
                        question=dec_b.question,
                        run_a_choice=None,
                        run_b_choice=dec_b.chosen,
                        match=False,
                    )
                )

        outcome_diff = self._compute_outcome_diff(run_a, run_b)
        variance_score = self._compute_variance_score(alignments)

        return RunComparison(
            run_a_id=run_a.metadata.run_id,
            run_b_id=run_b.metadata.run_id,
            decision_alignment=alignments,
            outcome_diff=outcome_diff,
            variance_score=variance_score,
        )

    @staticmethod
    def _values_match(a: str, b: str) -> bool:
        """Check if two decision values match (case-insensitive, normalized)."""
        def normalize(v: str) -> str:
            return re.sub(r"[\s_\-]+", "", v.lower().strip())
        return normalize(a) == normalize(b)

    @staticmethod
    def _compute_variance_score(alignments: list[DecisionAlignment]) -> float:
        """Compute variance score from alignments.

        0.0 = all decisions identical, 1.0 = all different.
        """
        if not alignments:
            return 0.0
        return 1.0 - (sum(1 for a in alignments if a.match) / len(alignments))

    @staticmethod
    def _compute_outcome_diff(
        run_a: DecisionTrace, run_b: DecisionTrace
    ) -> OutcomeDiff:
        """Compute outcome differences between two runs."""
        a_id = run_a.metadata.run_id
        b_id = run_b.metadata.run_id

        # Code line count from last execution
        def _code_lines(trace: DecisionTrace) -> int:
            if trace.executions:
                return len(trace.executions[-1].code_snapshot.splitlines())
            return 0

        # Libraries used: extract import statements from code
        def _libraries(trace: DecisionTrace) -> list[str]:
            libs: set[str] = set()
            if trace.executions:
                code = trace.executions[-1].code_snapshot
                for line in code.splitlines():
                    line = line.strip()
                    if line.startswith("import "):
                        parts = line.split()
                        if len(parts) >= 2:
                            libs.add(parts[1].split(".")[0])
                    elif line.startswith("from "):
                        parts = line.split()
                        if len(parts) >= 2:
                            libs.add(parts[1].split(".")[0])
            return sorted(libs)

        # Execution time from last execution
        def _exec_time(trace: DecisionTrace) -> int:
            if trace.executions:
                return trace.executions[-1].duration_ms
            return 0

        # Anomalies found: count from stdout (heuristic)
        def _anomalies(trace: DecisionTrace) -> int:
            if trace.executions:
                stdout = trace.executions[-1].stdout.lower()
                # Look for common anomaly reporting patterns
                for pattern in [
                    r"found (\d+) anomal",
                    r"(\d+) anomal\w+ detected",
                    r"anomalies:\s*(\d+)",
                ]:
                    match = re.search(pattern, stdout)
                    if match:
                        return int(match.group(1))
            return 0

        return OutcomeDiff(
            code_line_count={a_id: _code_lines(run_a), b_id: _code_lines(run_b)},
            libraries_used={a_id: _libraries(run_a), b_id: _libraries(run_b)},
            execution_time_ms={a_id: _exec_time(run_a), b_id: _exec_time(run_b)},
            anomalies_found={a_id: _anomalies(run_a), b_id: _anomalies(run_b)},
            total_cost_usd={
                a_id: run_a.metadata.total_cost_usd,
                b_id: run_b.metadata.total_cost_usd,
            },
        )
