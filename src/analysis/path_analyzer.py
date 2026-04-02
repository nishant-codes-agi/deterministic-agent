"""PathAnalyzer - cross-run path analysis and visualization data."""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from typing import Optional

from src.locking.question_matcher import QuestionMatcher
from src.models.analysis import (
    DecisionTreeBranch,
    DecisionTreeNode,
    DecisionVariance,
    OutcomeCorrelation,
    PathAnalysis,
)
from src.models.enums import DecisionCategory, RunStatus
from src.models.traces import DecisionTrace
from src.tracing.store import TraceStore

logger = logging.getLogger(__name__)


class PathAnalyzer:
    """Analyzes decision paths across multiple runs.

    Produces a merged decision tree, variance metrics per decision,
    and outcome correlations showing which choices lead to better results.
    """

    def __init__(self, trace_store: TraceStore) -> None:
        self._store = trace_store

    async def analyze(
        self, run_ids: Optional[list[str]] = None
    ) -> PathAnalysis:
        """Analyze all (or specified) runs.

        Args:
            run_ids: Specific run IDs to analyze. If None, analyzes all runs.

        Returns:
            PathAnalysis with decision tree, variance metrics,
            and outcome correlations.
        """
        traces = await self._load_traces(run_ids)
        if not traces:
            return PathAnalysis(
                total_runs=0,
                decision_tree={},
                variance_by_decision=[],
                outcome_correlations=[],
            )

        tree = self.build_decision_tree(traces)
        variance = self.compute_variance_by_decision(traces)
        correlations = self.correlate_with_outcomes(traces)

        return PathAnalysis(
            total_runs=len(traces),
            decision_tree=tree.model_dump(mode="json"),
            variance_by_decision=variance,
            outcome_correlations=correlations,
        )

    async def _load_traces(
        self, run_ids: Optional[list[str]] = None
    ) -> list[DecisionTrace]:
        """Load traces by IDs or all available."""
        if run_ids:
            traces = []
            for rid in run_ids:
                trace = await self._store.get_trace(rid)
                if trace:
                    traces.append(trace)
                else:
                    logger.warning(f"Run {rid} not found, skipping")
            return traces

        # Load all runs
        all_runs = await self._store.list_runs(limit=1000)
        traces = []
        for meta in all_runs:
            trace = await self._store.get_trace(meta.run_id)
            if trace:
                traces.append(trace)
        return traces

    def build_decision_tree(
        self, traces: list[DecisionTrace]
    ) -> DecisionTreeNode:
        """Build a merged decision tree from multiple traces.

        Groups decisions by normalized question, then for each unique choice
        creates a branch showing which runs took that path.
        """
        # Collect all decisions grouped by (category, normalized_question)
        grouped = self._group_decisions(traces)

        if not grouped:
            return DecisionTreeNode(
                question="(no decisions)",
                category=DecisionCategory.DATA_SELECTION,
                branches=[],
            )

        # Build tree: each group becomes a node with choice branches
        # Order by the most common sequence number across traces
        sorted_groups = sorted(grouped.items(), key=lambda kv: kv[1]["avg_seq"])

        # Build as a flat tree (each decision is a top-level node)
        root = DecisionTreeNode(
            question="Decision Tree",
            category=DecisionCategory.DATA_SELECTION,
            branches=[
                DecisionTreeBranch(
                    choice="root",
                    run_ids=[t.metadata.run_id for t in traces],
                    count=len(traces),
                    children=self._build_tree_nodes(sorted_groups),
                )
            ],
        )
        return root

    def _group_decisions(
        self, traces: list[DecisionTrace]
    ) -> dict[tuple[str, str], dict]:
        """Group decisions across traces by normalized (category, question).

        Returns a dict mapping (category, norm_question) to:
        {
            "question": original question text,
            "category": DecisionCategory,
            "choices": {choice_value: [run_ids]},
            "avg_seq": average sequence number,
            "decisions_by_run": {run_id: DecisionPoint},
        }
        """
        grouped: dict[tuple[str, str], dict] = {}

        for trace in traces:
            run_id = trace.metadata.run_id
            for dp in trace.decisions:
                norm_q = QuestionMatcher.normalize(dp.question)
                key = (dp.category.value, norm_q)

                if key not in grouped:
                    grouped[key] = {
                        "question": dp.question,
                        "category": dp.category,
                        "choices": defaultdict(list),
                        "seq_sum": 0,
                        "seq_count": 0,
                        "decisions_by_run": {},
                    }

                grouped[key]["choices"][dp.chosen].append(run_id)
                grouped[key]["seq_sum"] += dp.sequence_number
                grouped[key]["seq_count"] += 1
                grouped[key]["decisions_by_run"][run_id] = dp

        # Compute avg_seq
        for info in grouped.values():
            info["avg_seq"] = (
                info["seq_sum"] / info["seq_count"]
                if info["seq_count"] > 0
                else 0
            )

        return grouped

    @staticmethod
    def _build_tree_nodes(
        sorted_groups: list[tuple[tuple[str, str], dict]],
    ) -> list[DecisionTreeNode]:
        """Build DecisionTreeNode list from sorted groups."""
        nodes = []
        for (_cat, _norm_q), info in sorted_groups:
            branches = []
            for choice, run_ids in sorted(
                info["choices"].items(), key=lambda kv: -len(kv[1])
            ):
                branches.append(
                    DecisionTreeBranch(
                        choice=choice,
                        run_ids=run_ids,
                        count=len(run_ids),
                        children=[],
                    )
                )
            nodes.append(
                DecisionTreeNode(
                    question=info["question"],
                    category=info["category"],
                    branches=branches,
                )
            )
        return nodes

    def compute_variance_by_decision(
        self, traces: list[DecisionTrace]
    ) -> list[DecisionVariance]:
        """Compute variance metrics for each decision across runs.

        For each decision question, computes:
        - unique_choices: all distinct values chosen
        - entropy: Shannon entropy (higher = more variance)
        - most_common: the mode value
        - samples: number of runs with this decision

        Returns list sorted by entropy descending (highest variance first).
        """
        grouped = self._group_decisions(traces)
        variances = []

        for (_cat, _norm_q), info in grouped.items():
            choices_flat = []
            for choice, run_ids in info["choices"].items():
                choices_flat.extend([choice] * len(run_ids))

            counts = Counter(choices_flat)
            total = len(choices_flat)
            unique = sorted(set(choices_flat))

            # Shannon entropy
            entropy = 0.0
            for count in counts.values():
                if count > 0:
                    p = count / total
                    entropy -= p * math.log2(p)

            most_common = counts.most_common(1)[0][0] if counts else ""

            variances.append(
                DecisionVariance(
                    question=info["question"],
                    category=info["category"],
                    unique_choices=unique,
                    entropy=round(entropy, 4),
                    most_common=most_common,
                    samples=total,
                )
            )

        # Sort by entropy descending
        variances.sort(key=lambda v: v.entropy, reverse=True)
        return variances

    def correlate_with_outcomes(
        self, traces: list[DecisionTrace]
    ) -> list[OutcomeCorrelation]:
        """Correlate decision choices with outcome metrics.

        For each decision, computes per-choice:
        - execution_success_rate: fraction of runs with that choice that succeeded
        - avg_cost: average total cost
        - avg_code_lines: average code line count
        """
        grouped = self._group_decisions(traces)

        # Build run_id -> outcome metrics map
        outcomes: dict[str, dict] = {}
        for trace in traces:
            run_id = trace.metadata.run_id
            code_lines = 0
            if trace.executions:
                code_lines = len(
                    trace.executions[-1].code_snapshot.splitlines()
                )

            outcomes[run_id] = {
                "success": trace.metadata.status == RunStatus.SUCCESS,
                "cost": trace.metadata.total_cost_usd,
                "code_lines": code_lines,
            }

        correlations = []
        for (_cat, _norm_q), info in grouped.items():
            success_by_choice: dict[str, list[bool]] = defaultdict(list)
            cost_by_choice: dict[str, list[float]] = defaultdict(list)
            lines_by_choice: dict[str, list[int]] = defaultdict(list)

            for choice, run_ids in info["choices"].items():
                for rid in run_ids:
                    if rid in outcomes:
                        success_by_choice[choice].append(
                            outcomes[rid]["success"]
                        )
                        cost_by_choice[choice].append(outcomes[rid]["cost"])
                        lines_by_choice[choice].append(
                            outcomes[rid]["code_lines"]
                        )

            def _avg(vals: list) -> float:
                return sum(vals) / len(vals) if vals else 0.0

            correlations.append(
                OutcomeCorrelation(
                    question=info["question"],
                    category=info["category"],
                    success_rate_by_choice={
                        c: round(_avg([1.0 if s else 0.0 for s in vals]), 4)
                        for c, vals in success_by_choice.items()
                    },
                    avg_cost_by_choice={
                        c: round(_avg(vals), 6)
                        for c, vals in cost_by_choice.items()
                    },
                    avg_code_lines_by_choice={
                        c: round(_avg(vals), 1)
                        for c, vals in lines_by_choice.items()
                    },
                )
            )

        return correlations
