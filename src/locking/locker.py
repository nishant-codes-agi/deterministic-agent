"""PromptInjectionLocker - path locker that injects constraints into LLM prompts.

Three modes:
1. Full lock: all decisions from source trace are locked
2. Partial lock: only specified decision IDs, sequences, or categories are locked
3. Fork lock: upstream locked, one decision overridden, downstream free
"""

from __future__ import annotations

import logging
from typing import Optional

from src.locking.base import PathLocker
from src.locking.question_matcher import QuestionMatcher
from src.models.decisions import DecisionPoint
from src.models.enums import DecisionCategory
from src.models.traces import DecisionTrace, PathLockConfig

logger = logging.getLogger(__name__)

# Common synonyms for value normalization
_SYNONYMS: dict[str, str] = {
    "iqr": "interquartile_range",
    "interquartile range": "interquartile_range",
    "z-score": "z_score",
    "zscore": "z_score",
    "z score": "z_score",
    "std": "standard_deviation",
    "stdev": "standard_deviation",
    "standard deviation": "standard_deviation",
    "fastapi": "fastapi",
    "fast api": "fastapi",
    "flask": "flask",
    "matplotlib": "matplotlib",
    "plotly": "plotly",
    "seaborn": "seaborn",
}


class PromptInjectionLocker(PathLocker):
    """Path locker that injects constraints into LLM prompts.

    Three modes:
    1. Full lock: all decisions from source trace are locked
    2. Partial lock: only specified decision IDs are locked
    3. Fork lock: upstream locked, one decision overridden, downstream free

    The locking approach:
    1. PROMPT INJECTION: When a decision is locked, the LLM prompt includes
       "You MUST choose {value}" so the LLM reasons with the constraint.
    2. OUTPUT VERIFICATION: After the LLM responds, verify the choice matches.
    3. DIVERGENCE DETECTION: If the locked choice is no longer valid, detect
       and report it.
    """

    def __init__(
        self,
        source_trace: DecisionTrace,
        config: PathLockConfig,
    ) -> None:
        self._source = source_trace
        self._config = config
        self._locked_decisions = self._build_lock_map()
        self._verification_failures: list[dict] = []

    def _build_lock_map(self) -> dict[str, DecisionPoint]:
        """Build a map of question (normalized) -> locked DecisionPoint.

        Uses the PathLockConfig to determine which decisions should be locked:
        - If locked_decision_ids is set: lock only those IDs
        - If lock_through_sequence is set: lock all decisions with
          sequence_number <= N
        - If lock_categories is set: lock all decisions in those categories
        - Apply overrides: if a decision_id is in overrides dict, use the
          override value instead of the original
        - If none of these are set: lock ALL decisions (full replay)
        """
        lock_map: dict[str, DecisionPoint] = {}

        for dp in self._source.decisions:
            should_lock = self._should_lock_decision(dp)
            if not should_lock:
                continue

            # Check for overrides
            override_value = self._config.overrides.get(dp.id)

            key = self._make_lock_key(dp.question, dp.category)

            if override_value is not None:
                # Create a modified decision with the override value
                lock_map[key] = DecisionPoint(
                    id=dp.id,
                    sequence_number=dp.sequence_number,
                    category=dp.category,
                    phase=dp.phase,
                    question=dp.question,
                    alternatives=dp.alternatives,
                    chosen=override_value,
                    reasoning=f"Overridden from '{dp.chosen}' to '{override_value}'",
                    confidence=dp.confidence,
                    locked=True,
                )
            else:
                lock_map[key] = dp

        return lock_map

    def _should_lock_decision(self, dp: DecisionPoint) -> bool:
        """Determine if a specific decision should be locked based on config."""
        has_any_filter = (
            self._config.locked_decision_ids is not None
            or self._config.lock_through_sequence is not None
            or self._config.lock_categories is not None
        )

        # If no filters set, lock ALL decisions (full replay)
        if not has_any_filter:
            return True

        # Check each filter - decision is locked if it matches ANY filter
        if (
            self._config.locked_decision_ids is not None
            and dp.id in self._config.locked_decision_ids
        ):
            return True

        if (
            self._config.lock_through_sequence is not None
            and dp.sequence_number <= self._config.lock_through_sequence
        ):
            return True

        if (
            self._config.lock_categories is not None
            and dp.category in self._config.lock_categories
        ):
            return True

        if (
            self._config.overrides
            and dp.id in self._config.overrides
        ):
            return True

        return False

    @staticmethod
    def _make_lock_key(question: str, category: DecisionCategory) -> str:
        """Create a normalized key for the lock map."""
        return f"{category.value}::{QuestionMatcher.normalize(question)}"

    def is_locked(
        self,
        question: str,
        category: DecisionCategory,
        sequence_number: int,
    ) -> bool:
        """Check if a decision at this point should be locked."""
        # Direct key lookup
        key = self._make_lock_key(question, category)
        if key in self._locked_decisions:
            return True

        # Try fuzzy matching via QuestionMatcher
        matched = QuestionMatcher.match(
            current_question=question,
            current_category=category,
            current_sequence=sequence_number,
            source_decisions=list(self._locked_decisions.values()),
        )
        return matched is not None

    def get_locked_value(
        self,
        question: str,
        category: DecisionCategory,
    ) -> Optional[str]:
        """Get the value to lock to. Returns None if not locked."""
        # Direct key lookup
        key = self._make_lock_key(question, category)
        if key in self._locked_decisions:
            return self._locked_decisions[key].chosen

        # Try fuzzy matching
        matched = QuestionMatcher.match(
            current_question=question,
            current_category=category,
            current_sequence=-1,  # Don't use sequence for value lookup
            source_decisions=list(self._locked_decisions.values()),
        )
        if matched is not None:
            return matched.chosen

        return None

    def get_lock_prompt(self, question: str, locked_value: str) -> str:
        """Generate prompt injection text for a locked decision."""
        return (
            f"\n\n\u26a0\ufe0f  LOCKED DECISION: For the question \"{question}\", "
            f"you MUST choose \"{locked_value}\". "
            f"Do not consider other alternatives for this decision. "
            f"Set locked=true in your response. "
            f"Adapt all subsequent decisions to be compatible with this choice.\n"
        )

    def get_all_lock_prompts(self) -> str:
        """Generate combined prompt injection for ALL locked decisions.

        Used to inject into the planning prompt upfront so the LLM sees
        all constraints at once.
        """
        if not self._locked_decisions:
            return ""

        lines = []
        for dp in self._locked_decisions.values():
            status = "LOCKED"
            lines.append(f"  - {dp.question}: {status} \u2192 {dp.chosen}")

        header = (
            "For each of these decisions, some are LOCKED (must use specified value):\n"
        )
        return header + "\n".join(lines)

    def verify_decision(self, decision: DecisionPoint) -> bool:
        """Verify that a decision matches the locked value.

        Returns True if the decision matches the lock or is unlocked.
        """
        if not decision.locked:
            return True  # Free decision, always valid

        expected = self.get_locked_value(decision.question, decision.category)
        if expected is None:
            return True  # Not in lock map

        # Fuzzy match: normalize both values
        if self._normalize_value(decision.chosen) == self._normalize_value(expected):
            return True

        # Record the verification failure
        self._verification_failures.append({
            "question": decision.question,
            "category": decision.category.value,
            "expected": expected,
            "got": decision.chosen,
            "sequence_number": decision.sequence_number,
        })
        return False

    @property
    def verification_failures(self) -> list[dict]:
        """Return list of verification failures encountered during the run."""
        return list(self._verification_failures)

    @staticmethod
    def _normalize_value(value: str) -> str:
        """Normalize decision values for matching.

        Handles: case, whitespace, common synonyms
        (e.g., 'IQR' == 'interquartile_range').
        """
        normalized = value.lower().strip()

        # Check direct synonyms
        if normalized in _SYNONYMS:
            return _SYNONYMS[normalized]

        # Also check if the value (with underscores/spaces collapsed) matches
        # a synonym's canonical form
        collapsed = normalized.replace("_", " ").replace("-", " ").strip()
        if collapsed in _SYNONYMS:
            return _SYNONYMS[collapsed]

        # Check if value IS a canonical form (reverse lookup)
        canonical_values = set(_SYNONYMS.values())
        if normalized in canonical_values:
            return normalized
        # Try with underscores replaced
        if normalized.replace("_", "") in {v.replace("_", "") for v in canonical_values}:
            # Find the canonical form
            for canon in canonical_values:
                if canon.replace("_", "") == normalized.replace("_", ""):
                    return canon

        # Remove underscores, hyphens, and spaces for comparison
        return normalized.replace("_", "").replace("-", "").replace(" ", "")
