"""Integration tests for path locking with the full agent loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.runner import AgentRunner
from src.config import AgentConfig, LLMConfig
from src.locking.exceptions import PathLockViolationError
from src.locking.locker import PromptInjectionLocker
from src.llm.base import LLMMessage, LLMProvider, LLMResponse
from src.models.decisions import Alternative, DecisionPoint
from src.models.enums import AgentPhase, DecisionCategory, RunStatus
from src.models.traces import DecisionTrace, PathLockConfig, RunMetadata
from src.sandbox.base import ExecutionResult, SandboxProvider


# --- Test fixtures ---


def _make_planning_response(decisions: list[dict]) -> str:
    """Build a planning JSON response string."""
    return json.dumps({
        "decisions": decisions,
        "implementation_plan": "Step 1: fetch data. Step 2: detect anomalies.",
    })


def _make_coding_response() -> str:
    """Build a coding JSON response string."""
    return json.dumps({
        "files": {
            "main.py": "import yfinance as yf\nprint('hello')\n",
            "requirements.txt": "yfinance\n",
        },
        "requirements": ["yfinance"],
        "entry_point": "main.py",
    })


def _make_eval_success() -> str:
    """Build a successful evaluation response."""
    return json.dumps({
        "status": "success",
        "assessment": "Code ran successfully",
        "recovery_strategy": "",
        "next_action": "done",
    })


def _standard_decisions() -> list[dict]:
    """Standard planning decisions for the stock task."""
    return [
        {
            "question": "Which stock ticker(s) to analyze?",
            "category": "data_selection",
            "alternatives": [
                {"value": "AAPL", "reasoning": "Liquid stock"},
                {"value": "TSLA", "reasoning": "High volatility"},
            ],
            "chosen": "AAPL",
            "reasoning": "Most reliable data",
            "confidence": 0.85,
        },
        {
            "question": "What timeframe to use?",
            "category": "parameter_tuning",
            "alternatives": [
                {"value": "2y", "reasoning": "Good history"},
                {"value": "1y", "reasoning": "More recent"},
            ],
            "chosen": "2y",
            "reasoning": "More data points",
            "confidence": 0.80,
        },
        {
            "question": "Which anomaly detection method?",
            "category": "algorithm_selection",
            "alternatives": [
                {"value": "IQR", "reasoning": "Simple, robust"},
                {"value": "z_score", "reasoning": "Statistical"},
            ],
            "chosen": "IQR",
            "reasoning": "Most robust for financial data",
            "confidence": 0.75,
        },
    ]


def _locked_decisions() -> list[dict]:
    """Same decisions but with locked=true markers."""
    decs = _standard_decisions()
    for d in decs:
        d["locked"] = True
    return decs


class MockLLMProvider(LLMProvider):
    """Mock LLM provider that returns predetermined responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    async def complete(
        self,
        messages: list[LLMMessage],
        model: str = "",
        response_format: Optional[type] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        idx = min(self._call_count, len(self._responses) - 1)
        content = self._responses[idx]
        self._call_count += 1

        # Try to parse as JSON for structured output
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

    async def estimate_cost(
        self, messages: list[LLMMessage], model: str = ""
    ) -> float:
        return 0.001

    @property
    def provider_name(self) -> str:
        return "mock"


class MockSandbox(SandboxProvider):
    """Mock sandbox that returns success."""

    async def setup_workspace(self, run_id: str) -> Path:
        import tempfile
        p = Path(tempfile.mkdtemp()) / run_id / "workspace"
        p.mkdir(parents=True, exist_ok=True)
        return p

    async def execute(
        self, command: str, cwd: Optional[Path] = None, timeout: int = 120
    ) -> ExecutionResult:
        return ExecutionResult(
            stdout="Anomalies detected: 5\nReport saved.",
            stderr="",
            exit_code=0,
            duration_ms=1000,
            artifacts=[],
            timed_out=False,
        )

    async def install_packages(
        self, packages: list[str], cwd: Path
    ) -> ExecutionResult:
        return ExecutionResult(
            stdout="", stderr="", exit_code=0,
            duration_ms=0, artifacts=[], timed_out=False,
        )

    async def cleanup_workspace(self, run_id: str) -> None:
        pass


def _make_agent_config() -> AgentConfig:
    return AgentConfig(
        max_iterations=3,
        execution_timeout=60,
        max_cost_usd=1.0,
    )


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


# --- Integration Tests ---


class TestLockedReplayProducesSameDecisions:
    """Test that replaying with all decisions locked produces the same decisions."""

    @pytest.mark.asyncio
    async def test_locked_replay_produces_same_decisions(self, tmp_path: Path):
        """Run agent once, lock all decisions, replay - verify all decisions match."""
        # Step 1: Do an initial run
        responses = [
            _make_planning_response(_standard_decisions()),
            _make_coding_response(),
            _make_eval_success(),
        ]
        llm = MockLLMProvider(responses)
        sandbox = MockSandbox()

        agent = AgentRunner(
            llm=llm,
            sandbox=sandbox,
            config=_make_agent_config(),
            llm_config=_make_llm_config(),
            runs_dir=tmp_path,
        )

        original_trace = await agent.run("Build a stock anomaly detector")
        assert original_trace.metadata.status == RunStatus.SUCCESS
        assert len(original_trace.decisions) >= 3

        # Step 2: Replay with all decisions locked
        # The mock LLM returns the same decisions (simulating locked behavior)
        replay_responses = [
            _make_planning_response(_locked_decisions()),
            _make_coding_response(),
            _make_eval_success(),
        ]
        replay_llm = MockLLMProvider(replay_responses)

        replay_agent = AgentRunner(
            llm=replay_llm,
            sandbox=sandbox,
            config=_make_agent_config(),
            llm_config=_make_llm_config(),
            runs_dir=tmp_path,
        )

        lock_config = PathLockConfig(
            source_run_id=original_trace.metadata.run_id,
        )

        replay_trace = await replay_agent.run(
            "Build a stock anomaly detector",
            lock_config=lock_config,
        )

        assert replay_trace.metadata.status == RunStatus.SUCCESS

        # Verify all decisions match
        for orig, locked in zip(
            original_trace.decisions, replay_trace.decisions
        ):
            assert orig.chosen == locked.chosen, (
                f"Decision mismatch: {orig.question} - "
                f"original={orig.chosen}, replay={locked.chosen}"
            )


class TestPartialLockFreeDecisionsMayVary:
    """Test that with partial locking, free decisions can differ."""

    @pytest.mark.asyncio
    async def test_partial_lock_free_decisions_may_vary(self, tmp_path: Path):
        """Lock first 2 decisions, verify decisions 3+ can differ between runs."""
        # Initial run
        responses = [
            _make_planning_response(_standard_decisions()),
            _make_coding_response(),
            _make_eval_success(),
        ]
        llm = MockLLMProvider(responses)
        sandbox = MockSandbox()

        agent = AgentRunner(
            llm=llm,
            sandbox=sandbox,
            config=_make_agent_config(),
            llm_config=_make_llm_config(),
            runs_dir=tmp_path,
        )

        original_trace = await agent.run("Build a stock anomaly detector")

        # Replay with only first 2 locked, third decision different
        varied_decisions = _locked_decisions()[:2] + [
            {
                "question": "Which anomaly detection method?",
                "category": "algorithm_selection",
                "alternatives": [
                    {"value": "IQR", "reasoning": "Simple"},
                    {"value": "z_score", "reasoning": "Statistical"},
                ],
                "chosen": "z_score",  # Different from original!
                "reasoning": "Better for normal distributions",
                "confidence": 0.70,
            },
        ]

        replay_responses = [
            _make_planning_response(varied_decisions),
            _make_coding_response(),
            _make_eval_success(),
        ]
        replay_llm = MockLLMProvider(replay_responses)

        replay_agent = AgentRunner(
            llm=replay_llm,
            sandbox=sandbox,
            config=_make_agent_config(),
            llm_config=_make_llm_config(),
            runs_dir=tmp_path,
        )

        # Lock only through sequence 1 (first two decisions)
        lock_config = PathLockConfig(
            source_run_id=original_trace.metadata.run_id,
            lock_through_sequence=1,
        )

        replay_trace = await replay_agent.run(
            "Build a stock anomaly detector",
            lock_config=lock_config,
        )

        assert replay_trace.metadata.status == RunStatus.SUCCESS

        # First two decisions should match
        assert original_trace.decisions[0].chosen == replay_trace.decisions[0].chosen
        assert original_trace.decisions[1].chosen == replay_trace.decisions[1].chosen

        # Third decision can differ
        if len(replay_trace.decisions) >= 3:
            assert replay_trace.decisions[2].chosen == "z_score"


class TestLockViolationRaises:
    """Test that PathLockViolationError is raised after max retries."""

    @pytest.mark.asyncio
    async def test_lock_violation_raises(self, tmp_path: Path):
        """Mock LLM that ignores lock, verify PathLockViolationError after 3 retries."""
        # Initial run
        responses = [
            _make_planning_response(_standard_decisions()),
            _make_coding_response(),
            _make_eval_success(),
        ]
        llm = MockLLMProvider(responses)
        sandbox = MockSandbox()

        agent = AgentRunner(
            llm=llm,
            sandbox=sandbox,
            config=_make_agent_config(),
            llm_config=_make_llm_config(),
            runs_dir=tmp_path,
        )

        original_trace = await agent.run("Build a stock anomaly detector")

        # Replay: LLM ignores the lock and always picks TSLA instead of AAPL
        violating_decisions = [
            {
                "question": "Which stock ticker(s) to analyze?",
                "category": "data_selection",
                "alternatives": [
                    {"value": "AAPL", "reasoning": "Liquid"},
                    {"value": "TSLA", "reasoning": "Volatile"},
                ],
                "chosen": "TSLA",  # Violates lock (should be AAPL)
                "reasoning": "Prefer volatility",
                "confidence": 0.9,
                "locked": True,
            },
            {
                "question": "What timeframe to use?",
                "category": "parameter_tuning",
                "alternatives": [
                    {"value": "2y", "reasoning": "Good history"},
                    {"value": "1y", "reasoning": "More recent"},
                ],
                "chosen": "2y",
                "reasoning": "More data",
                "confidence": 0.8,
                "locked": True,
            },
            {
                "question": "Which anomaly detection method?",
                "category": "algorithm_selection",
                "alternatives": [
                    {"value": "IQR", "reasoning": "Simple"},
                    {"value": "z_score", "reasoning": "Statistical"},
                ],
                "chosen": "IQR",
                "reasoning": "Robust",
                "confidence": 0.75,
                "locked": True,
            },
        ]

        # Provide enough responses for 4 attempts (1 initial + 3 retries)
        violating_response = _make_planning_response(violating_decisions)
        replay_responses = [
            violating_response,
            violating_response,
            violating_response,
            violating_response,
            _make_coding_response(),
            _make_eval_success(),
        ]
        replay_llm = MockLLMProvider(replay_responses)

        replay_agent = AgentRunner(
            llm=replay_llm,
            sandbox=sandbox,
            config=_make_agent_config(),
            llm_config=_make_llm_config(),
            runs_dir=tmp_path,
        )

        lock_config = PathLockConfig(
            source_run_id=original_trace.metadata.run_id,
        )

        # The violation should be caught and result in FAILED status
        # (runner catches PathLockViolationError and finalizes with FAILED)
        trace = await replay_agent.run(
            "Build a stock anomaly detector",
            lock_config=lock_config,
        )
        assert trace.metadata.status == RunStatus.FAILED
        assert "violation" in trace.metadata.error.lower()


class TestLockConfigInMetadata:
    """Test that lock_config is captured in the trace metadata."""

    @pytest.mark.asyncio
    async def test_lock_config_captured(self, tmp_path: Path):
        """Verify lock_config is stored in trace metadata for replayed runs."""
        # Initial run
        responses = [
            _make_planning_response(_standard_decisions()),
            _make_coding_response(),
            _make_eval_success(),
        ]
        llm = MockLLMProvider(responses)
        sandbox = MockSandbox()

        agent = AgentRunner(
            llm=llm,
            sandbox=sandbox,
            config=_make_agent_config(),
            llm_config=_make_llm_config(),
            runs_dir=tmp_path,
        )

        original_trace = await agent.run("Build a stock anomaly detector")

        # Replay
        replay_responses = [
            _make_planning_response(_locked_decisions()),
            _make_coding_response(),
            _make_eval_success(),
        ]
        replay_llm = MockLLMProvider(replay_responses)
        replay_agent = AgentRunner(
            llm=replay_llm,
            sandbox=sandbox,
            config=_make_agent_config(),
            llm_config=_make_llm_config(),
            runs_dir=tmp_path,
        )

        lock_config = PathLockConfig(
            source_run_id=original_trace.metadata.run_id,
        )

        replay_trace = await replay_agent.run(
            "Build a stock anomaly detector",
            lock_config=lock_config,
        )

        assert replay_trace.metadata.lock_config is not None
        assert replay_trace.metadata.lock_config.source_run_id == original_trace.metadata.run_id
        assert replay_trace.metadata.parent_run_id == original_trace.metadata.run_id
