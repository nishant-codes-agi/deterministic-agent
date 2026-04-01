"""Integration tests for AgentRunner with mocked LLM.

These tests verify the full agent loop works end-to-end by mocking the LLM
to return predefined planning/coding/evaluation responses.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock

import pytest

from src.agent.runner import AgentRunner
from src.config import AgentConfig, LLMConfig
from src.llm.base import LLMMessage, LLMProvider, LLMResponse
from src.models.enums import RunStatus
from src.sandbox.subprocess_sandbox import SubprocessSandbox


# --- Fixtures ---


@pytest.fixture
def tmp_runs_dir():
    """Create a temporary runs directory for tests."""
    d = tempfile.mkdtemp(prefix="test_runs_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def llm_config():
    return LLMConfig(
        provider="openrouter",
        api_key="test-key",
        model_planning="test/planning-model",
        model_coding="test/coding-model",
        model_evaluation="test/eval-model",
        model_recovery="test/recovery-model",
    )


@pytest.fixture
def agent_config():
    return AgentConfig(
        max_iterations=3,
        execution_timeout=30,
        max_cost_usd=1.00,
    )


# --- Mock LLM Responses ---

PLANNING_RESPONSE = json.dumps({
    "decisions": [
        {
            "question": "Which stock ticker?",
            "category": "data_selection",
            "alternatives": [
                {"value": "AAPL", "reasoning": "Stable blue chip"},
                {"value": "TSLA", "reasoning": "High volatility"},
            ],
            "chosen": "AAPL",
            "reasoning": "Reliable data for testing",
            "confidence": 0.85,
        },
        {
            "question": "Which anomaly method?",
            "category": "algorithm_selection",
            "alternatives": [
                {"value": "IQR", "reasoning": "Simple and robust"},
                {"value": "Z-score", "reasoning": "Statistical standard"},
            ],
            "chosen": "IQR",
            "reasoning": "Works well with financial data",
            "confidence": 0.8,
        },
    ],
    "implementation_plan": "1. Fetch AAPL data\n2. Compute IQR\n3. Flag anomalies",
})

CODING_RESPONSE_SUCCESS = json.dumps({
    "files": {
        "main.py": (
            "import json\n"
            "print('Fetching stock data...')\n"
            "anomalies = [{'date': '2024-01-15', 'value': 195.2}]\n"
            "print(f'Found {len(anomalies)} anomalies')\n"
            "with open('report.json', 'w') as f:\n"
            "    json.dump({'anomalies': anomalies, 'ticker': 'AAPL'}, f)\n"
            "print('Report generated successfully')\n"
        ),
        "requirements.txt": "",
    },
    "requirements": [],
    "entry_point": "main.py",
})

CODING_RESPONSE_FAILING = json.dumps({
    "files": {
        "main.py": (
            "import nonexistent_module\n"
            "print('This will fail')\n"
        ),
        "requirements.txt": "",
    },
    "requirements": [],
    "entry_point": "main.py",
})

EVAL_SUCCESS = json.dumps({
    "status": "success",
    "assessment": "Code ran successfully, anomalies detected, report generated",
    "recovery_strategy": "",
    "next_action": "done",
})

EVAL_FAILURE = json.dumps({
    "status": "failed",
    "assessment": "Import error: nonexistent_module not found",
    "recovery_strategy": "Remove bad import and use standard libraries",
    "next_action": "fix",
})

EVAL_ABORT = json.dumps({
    "status": "failed",
    "assessment": "Fundamental issue, cannot proceed",
    "recovery_strategy": "",
    "next_action": "abort",
})

RECOVERY_RESPONSE = json.dumps({
    "files": {
        "main.py": (
            "import json\n"
            "print('Fixed: Fetching stock data...')\n"
            "anomalies = [{'date': '2024-01-15', 'value': 195.2}]\n"
            "print(f'Found {len(anomalies)} anomalies')\n"
            "with open('report.json', 'w') as f:\n"
            "    json.dump({'anomalies': anomalies}, f)\n"
            "print('Report generated successfully')\n"
        ),
        "requirements.txt": "",
    },
    "requirements": [],
    "decisions": [
        {
            "question": "Recovery approach?",
            "category": "error_recovery",
            "alternatives": [
                {"value": "remove_import", "reasoning": "Not needed"},
                {"value": "install_package", "reasoning": "Add dependency"},
            ],
            "chosen": "remove_import",
            "reasoning": "Module was unnecessary",
            "confidence": 0.95,
        }
    ],
})


def make_llm_response(content: str, model: str = "test/model") -> LLMResponse:
    return LLMResponse(
        content=content,
        structured=None,
        tokens_in=100,
        tokens_out=200,
        latency_ms=50,
        model=model,
        cost_usd=0.001,
    )


# --- Tests ---


@pytest.mark.asyncio
async def test_agent_run_with_mock_llm(tmp_runs_dir, llm_config, agent_config):
    """Full agent loop: plan -> code -> execute -> evaluate (success)."""
    mock_llm = AsyncMock(spec=LLMProvider)
    mock_llm.provider_name = "mock"
    mock_llm.complete = AsyncMock(side_effect=[
        make_llm_response(PLANNING_RESPONSE, "test/planning-model"),
        make_llm_response(CODING_RESPONSE_SUCCESS, "test/coding-model"),
        make_llm_response(EVAL_SUCCESS, "test/eval-model"),
    ])

    sandbox = SubprocessSandbox(runs_dir=tmp_runs_dir)
    agent = AgentRunner(
        llm=mock_llm,
        sandbox=sandbox,
        config=agent_config,
        llm_config=llm_config,
    )

    trace = await agent.run(
        "Pull stock data and detect anomalies"
    )

    assert trace.metadata.status == RunStatus.SUCCESS
    assert len(trace.decisions) >= 2  # At least the planning decisions
    assert len(trace.llm_calls) == 3  # plan + code + eval
    assert len(trace.executions) == 1
    assert trace.executions[0].exit_code == 0

    # Verify decisions captured
    questions = [d.question for d in trace.decisions]
    assert "Which stock ticker?" in questions
    assert "Which anomaly method?" in questions

    # Verify model routing recorded
    assert trace.metadata.model_routing["planning"] == "test/planning-model"
    assert trace.metadata.model_routing["coding"] == "test/coding-model"

    # Verify files were written to workspace
    workspace = tmp_runs_dir / trace.metadata.run_id / "workspace"
    assert (workspace / "main.py").exists()


@pytest.mark.asyncio
async def test_agent_handles_execution_failure(tmp_runs_dir, llm_config, agent_config):
    """Agent handles code that fails, recovers, and succeeds on second try."""
    mock_llm = AsyncMock(spec=LLMProvider)
    mock_llm.provider_name = "mock"
    mock_llm.complete = AsyncMock(side_effect=[
        make_llm_response(PLANNING_RESPONSE, "test/planning-model"),
        # First iteration: bad code
        make_llm_response(CODING_RESPONSE_FAILING, "test/coding-model"),
        make_llm_response(EVAL_FAILURE, "test/eval-model"),
        make_llm_response(RECOVERY_RESPONSE, "test/recovery-model"),
        # Second iteration: fixed code (from recovery)
        make_llm_response(CODING_RESPONSE_SUCCESS, "test/coding-model"),
        make_llm_response(EVAL_SUCCESS, "test/eval-model"),
    ])

    sandbox = SubprocessSandbox(runs_dir=tmp_runs_dir)
    agent = AgentRunner(
        llm=mock_llm,
        sandbox=sandbox,
        config=agent_config,
        llm_config=llm_config,
    )

    trace = await agent.run("Pull stock data and detect anomalies")

    assert trace.metadata.status == RunStatus.SUCCESS
    # Should have planning decisions + recovery decision
    assert len(trace.decisions) >= 3
    assert len(trace.executions) >= 2  # Failed + successful


@pytest.mark.asyncio
async def test_agent_max_iterations(tmp_runs_dir, llm_config, agent_config):
    """Agent stops at max_iterations and returns FAILED."""
    agent_config.max_iterations = 2

    # LLM always returns failing code and "fix" evaluation
    mock_llm = AsyncMock(spec=LLMProvider)
    mock_llm.provider_name = "mock"

    responses = [
        make_llm_response(PLANNING_RESPONSE, "test/planning-model"),
    ]
    # Each iteration: code + eval(fail) + recovery
    for _ in range(agent_config.max_iterations):
        responses.extend([
            make_llm_response(CODING_RESPONSE_FAILING, "test/coding-model"),
            make_llm_response(EVAL_FAILURE, "test/eval-model"),
            make_llm_response(RECOVERY_RESPONSE, "test/recovery-model"),
        ])

    mock_llm.complete = AsyncMock(side_effect=responses)

    sandbox = SubprocessSandbox(runs_dir=tmp_runs_dir)
    agent = AgentRunner(
        llm=mock_llm,
        sandbox=sandbox,
        config=agent_config,
        llm_config=llm_config,
    )

    trace = await agent.run("Pull stock data and detect anomalies")

    assert trace.metadata.status == RunStatus.FAILED
    assert trace.metadata.error == "Max iterations exceeded"
