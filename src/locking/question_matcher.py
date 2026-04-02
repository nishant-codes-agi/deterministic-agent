"""Question matching logic for cross-run locking alignment.

Questions from different runs may be phrased differently but mean the same
thing. E.g., "Which ticker to analyze?" vs "What stock ticker should we use?"

Strategy:
1. Exact match on (category, normalized_question)
2. If no exact match, fuzzy match on keywords
3. If still no match, fall back to sequence-number alignment
"""

from __future__ import annotations

import re
from typing import Optional

from src.models.decisions import DecisionPoint
from src.models.enums import DecisionCategory


class QuestionMatcher:
    """Matches decision questions across runs for locking alignment."""

    # Common stop words to ignore during fuzzy matching
    _STOP_WORDS = frozenset({
        "a", "an", "the", "to", "for", "of", "in", "on", "at", "by",
        "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "will", "would", "should", "could",
        "which", "what", "that", "this", "these", "those",
        "we", "you", "i", "it", "they", "use", "choose", "select",
    })

    @staticmethod
    def normalize(text: str) -> str:
        """Normalize a question for comparison.

        Lowercases, strips punctuation, collapses whitespace.
        """
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @classmethod
    def extract_keywords(cls, text: str) -> set[str]:
        """Extract meaningful keywords from a question.

        Also adds stemmed forms (simple suffix stripping) to improve
        matching across singular/plural and verb forms.
        """
        normalized = cls.normalize(text)
        words = set(normalized.split()) - cls._STOP_WORDS

        # Add simple stemmed forms to improve matching
        stemmed = set()
        for w in words:
            stemmed.add(w)
            # Simple plural/suffix stripping
            if w.endswith("s") and len(w) > 3:
                stemmed.add(w[:-1])
            if w.endswith("es") and len(w) > 4:
                stemmed.add(w[:-2])
            if w.endswith("ing") and len(w) > 5:
                stemmed.add(w[:-3])

        return stemmed

    @classmethod
    def match(
        cls,
        current_question: str,
        current_category: DecisionCategory,
        current_sequence: int,
        source_decisions: list[DecisionPoint],
    ) -> Optional[DecisionPoint]:
        """Find the matching decision from source trace.

        Args:
            current_question: The question being asked in the current run.
            current_category: The category of the current decision.
            current_sequence: The sequence number of the current decision.
            source_decisions: Decisions from the source trace to match against.

        Returns:
            The matching DecisionPoint, or None if no match found.
        """
        if not source_decisions:
            return None

        # Strategy 1: Exact match on (category, normalized_question)
        current_norm = cls.normalize(current_question)
        for dp in source_decisions:
            if (
                dp.category == current_category
                and cls.normalize(dp.question) == current_norm
            ):
                return dp

        # Strategy 2: Fuzzy match on keywords within same category
        current_keywords = cls.extract_keywords(current_question)
        if current_keywords:
            best_match: Optional[DecisionPoint] = None
            best_score = 0.0

            for dp in source_decisions:
                if dp.category != current_category:
                    continue

                dp_keywords = cls.extract_keywords(dp.question)
                if not dp_keywords:
                    continue

                # Jaccard similarity
                intersection = current_keywords & dp_keywords
                union = current_keywords | dp_keywords
                score = len(intersection) / len(union) if union else 0.0

                if score > best_score and score >= 0.3:
                    best_score = score
                    best_match = dp

            if best_match is not None:
                return best_match

        # Strategy 3: Fall back to sequence number alignment
        for dp in source_decisions:
            if dp.sequence_number == current_sequence:
                return dp

        return None
