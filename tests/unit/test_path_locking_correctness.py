"""Tests that path locking produces correct behavior.

Validates full lock replay, partial lock boundaries, fork override,
lock verification mismatch detection, and lock prompt injection into
the LLM messages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock

import pytest

from src.agent.runner import AgentRunner
from src.config import AgentConfig, LLMConfig
from src.llm.base import LLMMessage, LLMProvider, LLMResponse
from src.locking.locker import PromptInjectionLocker
from src.models.decisions import Alternative, DecisionPoint
from src.models.enums import AgentPhase, DecisionCategory, RunStatus
from src.models.traces import DecisionTrace, PathLockConfig, RunMetadata
from src.sandbox.base import ExecutionResult, SandboxProvider


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_decision(
    seq: int,
    question: str,
    category: DecisionCategory,
    chosen: str,
    dp_id: str | None = None,
    locked: bool = False,
) -> DecisionPoint:
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


def _make_source_trace(run_id: str = "source-run") -> DecisionTrace:
    """Create a source trace with 5 standard decisions."""
    decisions = [
        _make_decision(0, "Which stock ticker(s) to analyze?", DecisionCategory.DATA_SELECTION, "AAPL", "dp-001"),
        _make_decision(1, "What timeframe to use?", DecisionCategory.PARAMETER_TUNING, "2y", "dp-002"),
        _make_decision(2, "Which anomaly detection method?", DecisionCategory.ALGORITHM_SELECTION, "IQR", "dp-003"),
        _make_decision(3, "Which API framework to use?", DecisionCategory.LIBRARY_SELECTION, "fastapi", "dp-004"),
        _make_decision(4, "Which visualization library?", DecisionCategory.LIBRARY_SELECTION, "matplotlib", "dp-005"),
    ]
    return DecisionTrace(
        metadata=RunMetadata(
            run_id=run_id,
            task_description="Stock anomaly detection task",
            llm_provider="openrouter",
            model_routing={"planning": "test/model"},
            temperature=0.0,
            status=RunStatus.SUCCESS,
        ),
        decisions=decisions,
    )


def _make_planning_response(decisions: list[dict]) -> str:
    return json.dumps({
        "decisions": decisions,
        "implementation_plan": "Step 1: fetch. Step 2: detect. Step 3: report.",
    })


def _make_coding_response() -> str:
    return json.dumps({
        "files": {"main.py": "print('hello')"},
        "requirements": [],
        "entry_point": "main.py",
    })


def _make_eval_success() -> str:
    return json.dumps({
        "status": "success",
        "assessment": "Code ran successfully",
        "recovery_strategy": "",
        "next_action": "done",
    })


def _standard_decisions_dict(locked: bool = False) -> list[dict]:
    """Standard decisions as dicts for mock LLM responses."""
    return [
        {
            "question": "Which stock ticker(s) to analyze?",
            "category": "data_selection",
            "alternatives": [
                {"value": "AAPL", "reasoning": "Liquid stock"},
                {"value": "TSLA", "reasoning": "Volatile"},
            ],
            "chosen": "AAPL",
            "reasoning": "Most reliable data",
            "confidence": 0.85,
            "locked": locked,
        },
        {
            "question": "What timeframe to use?",
            "category": "parameter_tuning",
            "alternatives": [
                {"value": "2y", "reasoning": "More data"},
                {"value": "1y", "reasoning": "Recent"},
            ],
            "chosen": "2y",
            "reasoning": "More data points",
            "confidence": 0.80,
            "locked": locked,
        },
        {
            "question": "Which anomaly detection method?",
            "category": "algorithm_selection",
            "alternatives": [
                {"value": "IQR", "reasoning": "Simple, robust"},
                {"value": "z_score", "reasoning": "Statistical"},
            ],
            "chosen": "IQR",
            "reasoning": "Most robust",
            "confidence": 0.75,
            "locked": locked,
        },
    ]


class MockLLMProvider(LLMProvider):
    """Mock LLM that returns predetermined responses and captures messages."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self.captured_messages: list[list[LLMMessage]] = []

    async def complete(
        self,
        messages: list[LLMMessage],
        model: str = "",
        response_format: Optional[type] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        self.captured_messages.append(messages)
        idx = min(self._call_count, len(self._responses) - 1)
        content = self._responses[idx]
        self._call_count += 1

        structured = None
        try:
            structured = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            pass

        return LLMResponse(
            content=content,
            model=model or "test-model",
            tokens_in=100,
            tokens_out=50,
            latency_ms=100,
            cost_usd=0.001,
            structured=structured,
        )

    async def estimate_cost(self, messages: list[LLMMessage], model: str = "") -> float:
        return 0.001

    @property
    def provider_name(self) -> str:
        return "mock"


class MockSandbox(SandboxProvider):
    """Mock sandbox that always succeeds."""

    async def setup_workspace(self, run_id: str) -> Path:
        import tempfile
        p = Path(tempfile.mkdtemp()) / run_id / "workspace"
        p.mkdir(parents=True, exist_ok=True)
        return p

    async def execute(self, command: str, cwd: Optional[Path] = None, timeout: int = 120, env=None) -> ExecutionResult:
        return ExecutionResult(
            stdout="Anomalies detected: 5", stderr="",
            exit_code=0, duration_ms=1000, artifacts=[], timed_out=False,
        )

    async def install_packages(self, packages: list[str], cwd: Path) -> ExecutionResult:
        return ExecutionResult(
            stdout="", stderr="", exit_code=0,
            duration_ms=0, artifacts=[], timed_out=False,
        )

    async def cleanup_workspace(self, run_id: str) -> None:
        pass


def _make_agent_config() -> AgentConfig:
    return AgentConfig(max_iterations=3, execution_timeout=60, max_cost_usd=1.0)


def _make_llm_config() -> LLMConfig:
    return LLMConfig(
        provider="openrouter",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
        max_tokens=4096,
        model_planning="test/planning-model",
        model_coding="test/coding-model",
        model_evaluation="test/eval-model",
        model_recovery="test/recovery-model",
        model_fallback="test/fallback-model",
    )


# ── Test Classes ──────────────────────────────────────────────────────────────


class TestFullLockAllDecisionsMatch:
    """Replaying a run with full lock produces identical decisions."""

    @pytest.mark.asyncio
    async def test_full_lock_all_decisions_match(self, tmp_path: Path):
        """Run agent once, lock all decisions, replay - verify all decisions match."""
        # Step 1: Initial run
        responses = [
            _make_planning_response(_standard_decisions_dict()),
            _make_coding_response(),
            _make_eval_success(),
        ]
        llm = MockLLMProvider(responses)
        sandbox = MockSandbox()
        agent = AgentRunner(
            llm=llm, sandbox=sandbox,
            config=_make_agent_config(), llm_config=_make_llm_config(),
            runs_dir=tmp_path,
        )

        original = await agent.run("Build a stock anomaly detector")
        assert original.metadata.status == RunStatus.SUCCESS
        assert len(original.decisions) >= 3

        # Step 2: Replay with full lock
        replay_llm = MockLLMProvider([
            _make_planning_response(_standard_decisions_dict(locked=True)),
            _make_coding_response(),
            _make_eval_success(),
        ])
        replay_agent = AgentRunner(
            llm=replay_llm, sandbox=sandbox,
            config=_make_agent_config(), llm_config=_make_llm_config(),
            runs_dir=tmp_path,
        )

        lock_config = PathLockConfig(source_run_id=original.metadata.run_id)
        replayed = await replay_agent.run(
            "Build a stock anomaly detector",
            lock_config=lock_config,
        )

        assert replayed.metadata.status == RunStatus.SUCCESS

        # Verify all decisions match (excluding error_recovery)
        for orig, replay in zip(original.decisions, replayed.decisions):
            if orig.category != DecisionCategory.ERROR_RECOVERY:
                assert replay.chosen == orig.chosen, (
                    f"Mismatch at '{orig.question}': "
                    f"expected {orig.chosen}, got {replay.chosen}"
                )


class TestPartialLockRespectsBoundaries:
    """Locking decisions 1-3 doesn't constrain decisions 4+."""

    @pytest.mark.asyncio
    async def test_partial_lock_respects_boundaries(self, tmp_path: Path):
        # Initial run with 3 decisions
        responses = [
            _make_planning_response(_standard_decisions_dict()),
            _make_coding_response(),
            _make_eval_success(),
        ]
        llm = MockLLMProvider(responses)
        sandbox = MockSandbox()
        agent = AgentRunner(
            llm=llm, sandbox=sandbox,
            config=_make_agent_config(), llm_config=_make_llm_config(),
            runs_dir=tmp_path,
        )

        original = await agent.run("Build a stock anomaly detector")

        # Replay with only first 2 locked, third free
        varied_decisions = _standard_decisions_dict(locked=True)[:2] + [
            {
                "question": "Which anomaly detection method?",
                "category": "algorithm_selection",
                "alternatives": [
                    {"value": "IQR", "reasoning": "Simple"},
                    {"value": "z_score", "reasoning": "Statistical"},
                ],
                "chosen": "z_score",  # Different from original IQR
                "reasoning": "Better for normal distributions",
                "confidence": 0.70,
            },
        ]

        replay_llm = MockLLMProvider([
            _make_planning_response(varied_decisions),
            _make_coding_response(),
            _make_eval_success(),
        ])
        replay_agent = AgentRunner(
            llm=replay_llm, sandbox=sandbox,
            config=_make_agent_config(), llm_config=_make_llm_config(),
            runs_dir=tmp_path,
        )

        lock_config = PathLockConfig(
            source_run_id=original.metadata.run_id,
            lock_through_sequence=1,
        )

        replayed = await replay_agent.run(
            "Build a stock anomaly detector",
            lock_config=lock_config,
        )

        assert replayed.metadata.status == RunStatus.SUCCESS

        # First two decisions should match
        assert original.decisions[0].chosen == replayed.decisions[0].chosen
        assert original.decisions[1].chosen == replayed.decisions[1].chosen

        # Third decision can differ
        if len(replayed.decisions) >= 3:
            assert replayed.decisions[2].chosen == "z_score"


class TestLockVerificationDetectsMismatch:
    """When LLM ignores a lock, verification catches it."""

    @pytest.mark.asyncio
    async def test_lock_verification_detects_mismatch(self, tmp_path: Path):
        # Initial run
        responses = [
            _make_planning_response(_standard_decisions_dict()),
            _make_coding_response(),
            _make_eval_success(),
        ]
        llm = MockLLMProvider(responses)
        sandbox = MockSandbox()
        agent = AgentRunner(
            llm=llm, sandbox=sandbox,
            config=_make_agent_config(), llm_config=_make_llm_config(),
            runs_dir=tmp_path,
        )

        original = await agent.run("Build a stock anomaly detector")

        # Replay with LLM that ignores the lock (picks TSLA instead of AAPL)
        violating = _standard_decisions_dict(locked=True)
        violating[0]["chosen"] = "TSLA"  # Violates lock

        violating_response = _make_planning_response(violating)
        replay_llm = MockLLMProvider([
            violating_response, violating_response,
            violating_response, violating_response,
            _make_coding_response(), _make_eval_success(),
        ])
        replay_agent = AgentRunner(
            llm=replay_llm, sandbox=sandbox,
            config=_make_agent_config(), llm_config=_make_llm_config(),
            runs_dir=tmp_path,
        )

        lock_config = PathLockConfig(source_run_id=original.metadata.run_id)
        trace = await replay_agent.run(
            "Build a stock anomaly detector",
            lock_config=lock_config,
        )

        # Should fail due to lock violation
        assert trace.metadata.status == RunStatus.FAILED
        assert "violation" in trace.metadata.error.lower()


class TestForkOverridesSingleDecision:
    """Forking changes one decision while locking others."""

    @pytest.mark.asyncio
    async def test_fork_overrides_single_decision(self, tmp_path: Path):
        # Initial run
        responses = [
            _make_planning_response(_standard_decisions_dict()),
            _make_coding_response(),
            _make_eval_success(),
        ]
        llm = MockLLMProvider(responses)
        sandbox = MockSandbox()
        agent = AgentRunner(
            llm=llm, sandbox=sandbox,
            config=_make_agent_config(), llm_config=_make_llm_config(),
            runs_dir=tmp_path,
        )

        original = await agent.run("Build a stock anomaly detector")

        # Find dp-002 (the anomaly method decision) by sequence
        target_dp = original.decisions[2]  # IQR

        # Fork: override IQR -> z_score, lock everything else
        forked_decisions = _standard_decisions_dict(locked=True)
        forked_decisions[2]["chosen"] = "z_score"  # Override

        replay_llm = MockLLMProvider([
            _make_planning_response(forked_decisions),
            _make_coding_response(),
            _make_eval_success(),
        ])
        replay_agent = AgentRunner(
            llm=replay_llm, sandbox=sandbox,
            config=_make_agent_config(), llm_config=_make_llm_config(),
            runs_dir=tmp_path,
        )

        lock_config = PathLockConfig(
            source_run_id=original.metadata.run_id,
            overrides={target_dp.id: "z_score"},
        )

        forked = await replay_agent.run(
            "Build a stock anomaly detector",
            lock_config=lock_config,
        )

        assert forked.metadata.status == RunStatus.SUCCESS

        # The overridden decision should have the new value
        if len(forked.decisions) >= 3:
            assert forked.decisions[2].chosen == "z_score"

        # Other decisions should still match the original
        assert forked.decisions[0].chosen == original.decisions[0].chosen
        assert forked.decisions[1].chosen == original.decisions[1].chosen


class TestLockPromptIsInjected:
    """Verify the LLM prompt contains lock instructions."""

    @pytest.mark.asyncio
    async def test_lock_prompt_is_injected(self, tmp_path: Path):
        # Initial run
        responses = [
            _make_planning_response(_standard_decisions_dict()),
            _make_coding_response(),
            _make_eval_success(),
        ]
        llm = MockLLMProvider(responses)
        sandbox = MockSandbox()
        agent = AgentRunner(
            llm=llm, sandbox=sandbox,
            config=_make_agent_config(), llm_config=_make_llm_config(),
            runs_dir=tmp_path,
        )

        original = await agent.run("Build a stock anomaly detector")

        # Replay with captured messages
        replay_llm = MockLLMProvider([
            _make_planning_response(_standard_decisions_dict(locked=True)),
            _make_coding_response(),
            _make_eval_success(),
        ])
        replay_agent = AgentRunner(
            llm=replay_llm, sandbox=sandbox,
            config=_make_agent_config(), llm_config=_make_llm_config(),
            runs_dir=tmp_path,
        )

        lock_config = PathLockConfig(source_run_id=original.metadata.run_id)
        await replay_agent.run(
            "Build a stock anomaly detector",
            lock_config=lock_config,
        )

        # Check that the planning prompt (first LLM call) contains lock instructions
        planning_messages = replay_llm.captured_messages[0]
        all_content = " ".join(m.content for m in planning_messages)

        # Should contain the locked values
        assert "AAPL" in all_content, "Lock prompt should contain locked value 'AAPL'"
        assert "MUST" in all_content, "Lock prompt should contain directive 'MUST'"


class TestLockConfigCapturedInMetadata:
    """Lock config should be persisted in the replayed trace's metadata."""

    @pytest.mark.asyncio
    async def test_lock_config_captured_in_metadata(self, tmp_path: Path):
        # Initial run
        responses = [
            _make_planning_response(_standard_decisions_dict()),
            _make_coding_response(),
            _make_eval_success(),
        ]
        llm = MockLLMProvider(responses)
        sandbox = MockSandbox()
        agent = AgentRunner(
            llm=llm, sandbox=sandbox,
            config=_make_agent_config(), llm_config=_make_llm_config(),
            runs_dir=tmp_path,
        )

        original = await agent.run("Build a stock anomaly detector")

        # Replay
        replay_llm = MockLLMProvider([
            _make_planning_response(_standard_decisions_dict(locked=True)),
            _make_coding_response(),
            _make_eval_success(),
        ])
        replay_agent = AgentRunner(
            llm=replay_llm, sandbox=sandbox,
            config=_make_agent_config(), llm_config=_make_llm_config(),
            runs_dir=tmp_path,
        )

        lock_config = PathLockConfig(source_run_id=original.metadata.run_id)
        replayed = await replay_agent.run(
            "Build a stock anomaly detector",
            lock_config=lock_config,
        )

        assert replayed.metadata.lock_config is not None
        assert replayed.metadata.lock_config.source_run_id == original.metadata.run_id
        assert replayed.metadata.parent_run_id == original.metadata.run_id


class TestLockerUnitBehavior:
    """Unit tests for PromptInjectionLocker without running the full agent."""

    def test_full_lock_all_locked(self):
        """Full lock: all decisions from source should be locked."""
        trace = _make_source_trace()
        config = PathLockConfig(source_run_id="source-run")
        locker = PromptInjectionLocker(trace, config)

        for dp in trace.decisions:
            assert locker.is_locked(dp.question, dp.category, dp.sequence_number)

    def test_partial_lock_by_sequence(self):
        """Lock through sequence 2: decisions 0-2 locked, 3+ free."""
        trace = _make_source_trace()
        config = PathLockConfig(
            source_run_id="source-run",
            lock_through_sequence=2,
        )
        locker = PromptInjectionLocker(trace, config)

        assert locker.is_locked(trace.decisions[0].question, trace.decisions[0].category, 0)
        assert locker.is_locked(trace.decisions[1].question, trace.decisions[1].category, 1)
        assert locker.is_locked(trace.decisions[2].question, trace.decisions[2].category, 2)
        assert not locker.is_locked(trace.decisions[3].question, trace.decisions[3].category, 3)
        assert not locker.is_locked(trace.decisions[4].question, trace.decisions[4].category, 4)

    def test_fork_override_returns_new_value(self):
        """Override dp-003 with z_score: get_locked_value returns z_score."""
        trace = _make_source_trace()
        config = PathLockConfig(
            source_run_id="source-run",
            overrides={"dp-003": "z_score"},
        )
        locker = PromptInjectionLocker(trace, config)

        val = locker.get_locked_value(
            "Which anomaly detection method?",
            DecisionCategory.ALGORITHM_SELECTION,
        )
        assert val == "z_score"

    def test_verify_match_returns_true(self):
        """Verify a decision that matches the lock returns True."""
        trace = _make_source_trace()
        config = PathLockConfig(source_run_id="source-run")
        locker = PromptInjectionLocker(trace, config)

        dp = _make_decision(
            0, "Which stock ticker(s) to analyze?",
            DecisionCategory.DATA_SELECTION, "AAPL", locked=True,
        )
        assert locker.verify_decision(dp)

    def test_verify_mismatch_returns_false(self):
        """Verify a decision that doesn't match the lock returns False."""
        trace = _make_source_trace()
        config = PathLockConfig(source_run_id="source-run")
        locker = PromptInjectionLocker(trace, config)

        dp = _make_decision(
            0, "Which stock ticker(s) to analyze?",
            DecisionCategory.DATA_SELECTION, "TSLA", locked=True,
        )
        assert not locker.verify_decision(dp)
        assert len(locker.verification_failures) == 1

    def test_lock_prompt_contains_all_locked_values(self):
        """get_all_lock_prompts() should include all locked decision values."""
        trace = _make_source_trace()
        config = PathLockConfig(source_run_id="source-run")
        locker = PromptInjectionLocker(trace, config)

        prompt = locker.get_all_lock_prompts()
        assert "AAPL" in prompt
        assert "2y" in prompt
        assert "IQR" in prompt
        assert "fastapi" in prompt
        assert "matplotlib" in prompt
