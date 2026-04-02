"""Unit tests for PromptInjectionLocker and QuestionMatcher."""

from __future__ import annotations

import pytest

from src.locking.exceptions import PathLockViolationError, PathLockDivergenceError
from src.locking.locker import PromptInjectionLocker
from src.locking.question_matcher import QuestionMatcher
from src.models.decisions import Alternative, DecisionPoint
from src.models.enums import AgentPhase, DecisionCategory
from src.models.traces import DecisionTrace, PathLockConfig, RunMetadata


def _make_decision(
    seq: int,
    question: str,
    category: DecisionCategory,
    chosen: str,
    dp_id: str | None = None,
    locked: bool = False,
) -> DecisionPoint:
    """Helper to create a test decision point."""
    return DecisionPoint(
        id=dp_id or f"dp-{seq:03d}",
        sequence_number=seq,
        category=category,
        phase=AgentPhase.PLANNING,
        question=question,
        alternatives=[
            Alternative(value=chosen, reasoning="Chosen option"),
            Alternative(value="other", reasoning="Other option"),
        ],
        chosen=chosen,
        reasoning=f"Chose {chosen}",
        confidence=0.8,
        locked=locked,
    )


def _make_trace(decisions: list[DecisionPoint]) -> DecisionTrace:
    """Helper to create a test trace."""
    return DecisionTrace(
        metadata=RunMetadata(
            run_id="source-run-001",
            task_description="Test task",
            llm_provider="openrouter",
            model_routing={"planning": "test-model"},
            temperature=0.0,
        ),
        decisions=decisions,
    )


def _make_source_decisions() -> list[DecisionPoint]:
    """Create a standard set of 5 source decisions."""
    return [
        _make_decision(0, "Which stock ticker(s) to analyze?", DecisionCategory.DATA_SELECTION, "AAPL", "dp-001"),
        _make_decision(1, "What timeframe to use?", DecisionCategory.PARAMETER_TUNING, "2y", "dp-002"),
        _make_decision(2, "Which anomaly detection method?", DecisionCategory.ALGORITHM_SELECTION, "IQR", "dp-003"),
        _make_decision(3, "Which API framework to use?", DecisionCategory.LIBRARY_SELECTION, "fastapi", "dp-004"),
        _make_decision(4, "Which visualization library?", DecisionCategory.LIBRARY_SELECTION, "matplotlib", "dp-005"),
    ]


class TestFullLockAllDecisionsLocked:
    """Test that full lock mode locks all decisions from source trace."""

    def test_full_lock_all_decisions_locked(self):
        """Create a trace with 5 decisions, lock all, verify is_locked returns True for each."""
        decisions = _make_source_decisions()
        trace = _make_trace(decisions)
        config = PathLockConfig(source_run_id="source-run-001")

        locker = PromptInjectionLocker(trace, config)

        for dp in decisions:
            assert locker.is_locked(dp.question, dp.category, dp.sequence_number), (
                f"Decision '{dp.question}' should be locked"
            )

    def test_full_lock_returns_correct_values(self):
        """Verify get_locked_value returns the correct value for each decision."""
        decisions = _make_source_decisions()
        trace = _make_trace(decisions)
        config = PathLockConfig(source_run_id="source-run-001")

        locker = PromptInjectionLocker(trace, config)

        assert locker.get_locked_value("Which stock ticker(s) to analyze?", DecisionCategory.DATA_SELECTION) == "AAPL"
        assert locker.get_locked_value("What timeframe to use?", DecisionCategory.PARAMETER_TUNING) == "2y"
        assert locker.get_locked_value("Which anomaly detection method?", DecisionCategory.ALGORITHM_SELECTION) == "IQR"


class TestPartialLockByIds:
    """Test partial locking by specific decision IDs."""

    def test_partial_lock_by_ids(self):
        """Lock only decision IDs ['dp-001', 'dp-003'], verify dp-002 is not locked."""
        decisions = _make_source_decisions()
        trace = _make_trace(decisions)
        config = PathLockConfig(
            source_run_id="source-run-001",
            locked_decision_ids=["dp-001", "dp-003"],
        )

        locker = PromptInjectionLocker(trace, config)

        # dp-001 should be locked
        assert locker.is_locked(
            "Which stock ticker(s) to analyze?",
            DecisionCategory.DATA_SELECTION, 0,
        )
        # dp-003 should be locked
        assert locker.is_locked(
            "Which anomaly detection method?",
            DecisionCategory.ALGORITHM_SELECTION, 2,
        )
        # dp-002 should NOT be locked
        assert not locker.is_locked(
            "What timeframe to use?",
            DecisionCategory.PARAMETER_TUNING, 1,
        )
        # dp-004 should NOT be locked
        assert not locker.is_locked(
            "Which API framework to use?",
            DecisionCategory.LIBRARY_SELECTION, 3,
        )


class TestPartialLockBySequence:
    """Test partial locking by sequence number."""

    def test_partial_lock_by_sequence(self):
        """Lock through sequence 2, verify decisions 0-2 locked, 3+ not locked."""
        decisions = _make_source_decisions()
        trace = _make_trace(decisions)
        config = PathLockConfig(
            source_run_id="source-run-001",
            lock_through_sequence=2,
        )

        locker = PromptInjectionLocker(trace, config)

        # Decisions 0-2 should be locked
        assert locker.is_locked(decisions[0].question, decisions[0].category, 0)
        assert locker.is_locked(decisions[1].question, decisions[1].category, 1)
        assert locker.is_locked(decisions[2].question, decisions[2].category, 2)

        # Decisions 3+ should NOT be locked
        assert not locker.is_locked(decisions[3].question, decisions[3].category, 3)
        assert not locker.is_locked(decisions[4].question, decisions[4].category, 4)


class TestForkLockWithOverride:
    """Test fork mode: lock all but override one decision."""

    def test_fork_lock_with_override(self):
        """Lock all, override dp-003 with new value, verify override returns the override value."""
        decisions = _make_source_decisions()
        trace = _make_trace(decisions)
        config = PathLockConfig(
            source_run_id="source-run-001",
            overrides={"dp-003": "z_score"},
        )

        locker = PromptInjectionLocker(trace, config)

        # dp-003 should return the override value
        val = locker.get_locked_value(
            "Which anomaly detection method?",
            DecisionCategory.ALGORITHM_SELECTION,
        )
        assert val == "z_score"

        # Other decisions should return original values
        assert locker.get_locked_value(
            "Which stock ticker(s) to analyze?",
            DecisionCategory.DATA_SELECTION,
        ) == "AAPL"


class TestVerifyDecisionMatch:
    """Test decision verification."""

    def test_verify_decision_match(self):
        """Locked decision chosen='AAPL', verify returns True."""
        decisions = _make_source_decisions()
        trace = _make_trace(decisions)
        config = PathLockConfig(source_run_id="source-run-001")
        locker = PromptInjectionLocker(trace, config)

        dp = _make_decision(
            0, "Which stock ticker(s) to analyze?",
            DecisionCategory.DATA_SELECTION, "AAPL", locked=True,
        )
        assert locker.verify_decision(dp)

    def test_verify_decision_mismatch(self):
        """Locked decision chosen='AAPL' but decision has chosen='TSLA', verify returns False."""
        decisions = _make_source_decisions()
        trace = _make_trace(decisions)
        config = PathLockConfig(source_run_id="source-run-001")
        locker = PromptInjectionLocker(trace, config)

        dp = _make_decision(
            0, "Which stock ticker(s) to analyze?",
            DecisionCategory.DATA_SELECTION, "TSLA", locked=True,
        )
        assert not locker.verify_decision(dp)

    def test_verify_unlocked_always_passes(self):
        """Unlocked decision always passes verification."""
        decisions = _make_source_decisions()
        trace = _make_trace(decisions)
        config = PathLockConfig(source_run_id="source-run-001")
        locker = PromptInjectionLocker(trace, config)

        dp = _make_decision(
            0, "Which stock ticker(s) to analyze?",
            DecisionCategory.DATA_SELECTION, "TSLA", locked=False,
        )
        assert locker.verify_decision(dp)

    def test_verification_failures_tracked(self):
        """Verify that failed verifications are tracked."""
        decisions = _make_source_decisions()
        trace = _make_trace(decisions)
        config = PathLockConfig(source_run_id="source-run-001")
        locker = PromptInjectionLocker(trace, config)

        dp = _make_decision(
            0, "Which stock ticker(s) to analyze?",
            DecisionCategory.DATA_SELECTION, "TSLA", locked=True,
        )
        locker.verify_decision(dp)

        assert len(locker.verification_failures) == 1
        assert locker.verification_failures[0]["expected"] == "AAPL"
        assert locker.verification_failures[0]["got"] == "TSLA"


class TestNormalizeHandlesSynonyms:
    """Test value normalization with synonym handling."""

    def test_iqr_matches_interquartile_range(self):
        """'IQR' matches 'interquartile_range'."""
        decisions = [
            _make_decision(0, "Which method?", DecisionCategory.ALGORITHM_SELECTION, "IQR", "dp-001"),
        ]
        trace = _make_trace(decisions)
        config = PathLockConfig(source_run_id="source-run-001")
        locker = PromptInjectionLocker(trace, config)

        dp = _make_decision(
            0, "Which method?",
            DecisionCategory.ALGORITHM_SELECTION, "interquartile_range", locked=True,
        )
        assert locker.verify_decision(dp)

    def test_case_insensitive_match(self):
        """Case differences should not matter."""
        decisions = [
            _make_decision(0, "Which framework?", DecisionCategory.LIBRARY_SELECTION, "FastAPI", "dp-001"),
        ]
        trace = _make_trace(decisions)
        config = PathLockConfig(source_run_id="source-run-001")
        locker = PromptInjectionLocker(trace, config)

        dp = _make_decision(
            0, "Which framework?",
            DecisionCategory.LIBRARY_SELECTION, "fastapi", locked=True,
        )
        assert locker.verify_decision(dp)

    def test_z_score_synonyms(self):
        """'z-score', 'zscore', 'z_score' should all match."""
        decisions = [
            _make_decision(0, "Which method?", DecisionCategory.ALGORITHM_SELECTION, "z-score", "dp-001"),
        ]
        trace = _make_trace(decisions)
        config = PathLockConfig(source_run_id="source-run-001")
        locker = PromptInjectionLocker(trace, config)

        dp = _make_decision(
            0, "Which method?",
            DecisionCategory.ALGORITHM_SELECTION, "z_score", locked=True,
        )
        assert locker.verify_decision(dp)


class TestQuestionMatcherExact:
    """Test QuestionMatcher exact matching."""

    def test_exact_match_by_category_and_question(self):
        """Match by exact (category, normalized_question)."""
        source = _make_source_decisions()
        result = QuestionMatcher.match(
            current_question="Which stock ticker(s) to analyze?",
            current_category=DecisionCategory.DATA_SELECTION,
            current_sequence=0,
            source_decisions=source,
        )
        assert result is not None
        assert result.chosen == "AAPL"

    def test_no_match_wrong_category(self):
        """Same question but wrong category should not match exactly."""
        source = _make_source_decisions()
        result = QuestionMatcher.match(
            current_question="Which stock ticker(s) to analyze?",
            current_category=DecisionCategory.ALGORITHM_SELECTION,
            current_sequence=99,
            source_decisions=source,
        )
        # Should not get an exact match since category differs
        # But might fall back to sequence-number alignment
        # With sequence 99, no fallback either
        assert result is None


class TestQuestionMatcherFuzzy:
    """Test QuestionMatcher fuzzy matching."""

    def test_fuzzy_match_similar_question(self):
        """'Which ticker?' matches 'Which stock ticker(s) to analyze?'."""
        source = _make_source_decisions()
        result = QuestionMatcher.match(
            current_question="What stock ticker should we use?",
            current_category=DecisionCategory.DATA_SELECTION,
            current_sequence=99,  # Wrong sequence to force fuzzy match
            source_decisions=source,
        )
        assert result is not None
        assert result.chosen == "AAPL"

    def test_fuzzy_match_different_phrasing(self):
        """'Which anomaly method to use?' matches 'Which anomaly detection method?'."""
        source = _make_source_decisions()
        result = QuestionMatcher.match(
            current_question="Which anomaly method to apply?",
            current_category=DecisionCategory.ALGORITHM_SELECTION,
            current_sequence=99,
            source_decisions=source,
        )
        assert result is not None
        assert result.chosen == "IQR"


class TestQuestionMatcherSequenceFallback:
    """Test QuestionMatcher sequence number fallback."""

    def test_sequence_fallback(self):
        """No text match, falls back to sequence number alignment."""
        source = _make_source_decisions()
        result = QuestionMatcher.match(
            current_question="Completely unrelated question text here?",
            current_category=DecisionCategory.ARCHITECTURE,  # Different category
            current_sequence=0,  # Matches first decision's sequence
            source_decisions=source,
        )
        assert result is not None
        assert result.sequence_number == 0

    def test_no_match_at_all(self):
        """No text, category, or sequence match returns None."""
        source = _make_source_decisions()
        result = QuestionMatcher.match(
            current_question="Completely unrelated?",
            current_category=DecisionCategory.ARCHITECTURE,
            current_sequence=99,
            source_decisions=source,
        )
        assert result is None


class TestGetAllLockPrompts:
    """Test generation of combined lock prompt text."""

    def test_get_all_lock_prompts_full_lock(self):
        """Full lock should include all decisions in prompt."""
        decisions = _make_source_decisions()
        trace = _make_trace(decisions)
        config = PathLockConfig(source_run_id="source-run-001")
        locker = PromptInjectionLocker(trace, config)

        prompt = locker.get_all_lock_prompts()
        assert "AAPL" in prompt
        assert "2y" in prompt
        assert "IQR" in prompt
        assert "fastapi" in prompt
        assert "matplotlib" in prompt

    def test_get_all_lock_prompts_partial(self):
        """Partial lock should only include locked decisions."""
        decisions = _make_source_decisions()
        trace = _make_trace(decisions)
        config = PathLockConfig(
            source_run_id="source-run-001",
            locked_decision_ids=["dp-001"],
        )
        locker = PromptInjectionLocker(trace, config)

        prompt = locker.get_all_lock_prompts()
        assert "AAPL" in prompt
        # Others should not be in the prompt
        assert "IQR" not in prompt
        assert "matplotlib" not in prompt

    def test_get_all_lock_prompts_empty(self):
        """No locked decisions should return empty string."""
        trace = _make_trace([])
        config = PathLockConfig(source_run_id="source-run-001")
        locker = PromptInjectionLocker(trace, config)

        assert locker.get_all_lock_prompts() == ""


class TestGetLockPrompt:
    """Test individual lock prompt generation."""

    def test_lock_prompt_contains_question_and_value(self):
        """Lock prompt should contain the question and locked value."""
        decisions = _make_source_decisions()
        trace = _make_trace(decisions)
        config = PathLockConfig(source_run_id="source-run-001")
        locker = PromptInjectionLocker(trace, config)

        prompt = locker.get_lock_prompt("Which ticker?", "AAPL")
        assert "Which ticker?" in prompt
        assert "AAPL" in prompt
        assert "MUST" in prompt
        assert "locked=true" in prompt


class TestPathLockExceptions:
    """Test exception classes."""

    def test_violation_error(self):
        """PathLockViolationError captures question, expected, got, attempts."""
        err = PathLockViolationError(
            question="Which ticker?",
            expected="AAPL",
            got="TSLA",
            attempts=3,
        )
        assert err.question == "Which ticker?"
        assert err.expected == "AAPL"
        assert err.got == "TSLA"
        assert err.attempts == 3
        assert "AAPL" in str(err)
        assert "TSLA" in str(err)

    def test_divergence_error(self):
        """PathLockDivergenceError captures question, locked_value, reason."""
        err = PathLockDivergenceError(
            question="Which library?",
            locked_value="deprecated_lib",
            reason="Library no longer available",
        )
        assert err.question == "Which library?"
        assert err.locked_value == "deprecated_lib"
        assert err.reason == "Library no longer available"
        assert "deprecated_lib" in str(err)


class TestLockByCategories:
    """Test locking by decision categories."""

    def test_lock_by_categories(self):
        """Lock only LIBRARY_SELECTION decisions."""
        decisions = _make_source_decisions()
        trace = _make_trace(decisions)
        config = PathLockConfig(
            source_run_id="source-run-001",
            lock_categories=[DecisionCategory.LIBRARY_SELECTION],
        )
        locker = PromptInjectionLocker(trace, config)

        # Library decisions should be locked
        assert locker.is_locked(
            "Which API framework to use?",
            DecisionCategory.LIBRARY_SELECTION, 3,
        )
        assert locker.is_locked(
            "Which visualization library?",
            DecisionCategory.LIBRARY_SELECTION, 4,
        )

        # Non-library decisions should not be locked
        assert not locker.is_locked(
            "Which stock ticker(s) to analyze?",
            DecisionCategory.DATA_SELECTION, 0,
        )
        assert not locker.is_locked(
            "Which anomaly detection method?",
            DecisionCategory.ALGORITHM_SELECTION, 2,
        )
