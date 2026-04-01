"""Tests for src/agent/parsers.py"""

from __future__ import annotations

import json

import pytest

from src.agent.parsers import (
    ParseError,
    parse_coding_response,
    parse_evaluation_response,
    parse_json_from_llm,
    parse_planning_response,
    parse_recovery_response,
)


# --- parse_json_from_llm ---


class TestParseJsonFromLlm:
    def test_bare_json(self):
        raw = '{"key": "value", "number": 42}'
        result = parse_json_from_llm(raw)
        assert result == {"key": "value", "number": 42}

    def test_fenced_json(self):
        raw = '```json\n{"greeting": "hello"}\n```'
        result = parse_json_from_llm(raw)
        assert result == {"greeting": "hello"}

    def test_fenced_without_json_tag(self):
        raw = '```\n{"greeting": "hello"}\n```'
        result = parse_json_from_llm(raw)
        assert result == {"greeting": "hello"}

    def test_with_preamble(self):
        raw = "Here's the plan:\n```json\n{\"decisions\": []}\n```"
        result = parse_json_from_llm(raw)
        assert result == {"decisions": []}

    def test_invalid_raises_parse_error(self):
        raw = "This is not JSON at all, just plain text"
        with pytest.raises(ParseError) as exc_info:
            parse_json_from_llm(raw)
        assert exc_info.value.raw_text == raw

    def test_nested_json(self):
        raw = '{"outer": {"inner": [1, 2, 3]}}'
        result = parse_json_from_llm(raw)
        assert result == {"outer": {"inner": [1, 2, 3]}}

    def test_json_with_surrounding_text(self):
        raw = 'Sure! Here is your JSON:\n\n{"status": "ok"}\n\nHope that helps!'
        result = parse_json_from_llm(raw)
        assert result == {"status": "ok"}


# --- parse_planning_response ---


class TestParsePlanningResponse:
    def test_full_planning_response(self):
        response = {
            "decisions": [
                {
                    "question": "Which stock ticker?",
                    "category": "data_selection",
                    "alternatives": [
                        {"value": "AAPL", "reasoning": "Stable"},
                        {"value": "TSLA", "reasoning": "Volatile"},
                    ],
                    "chosen": "TSLA",
                    "reasoning": "More anomalies",
                    "confidence": 0.8,
                },
                {
                    "question": "Which anomaly method?",
                    "category": "algorithm_selection",
                    "alternatives": [
                        {"value": "IQR", "reasoning": "Simple"},
                        {"value": "Z-score", "reasoning": "Statistical"},
                    ],
                    "chosen": "IQR",
                    "reasoning": "Robust to outliers",
                    "confidence": 0.7,
                },
            ],
            "implementation_plan": "Step 1: fetch data...",
        }

        decisions, plan = parse_planning_response(response)

        assert len(decisions) == 2
        assert decisions[0].question == "Which stock ticker?"
        assert decisions[0].chosen == "TSLA"
        assert decisions[0].confidence == 0.8
        assert decisions[0].sequence_number == 0
        assert decisions[1].sequence_number == 1
        assert plan == "Step 1: fetch data..."

    def test_missing_fields_uses_defaults(self):
        response = {
            "decisions": [
                {
                    "question": "Pick a thing",
                    "alternatives": [
                        {"value": "A"},
                        {"value": "B"},
                    ],
                    "chosen": "A",
                }
            ]
        }

        decisions, plan = parse_planning_response(response)

        assert len(decisions) == 1
        assert decisions[0].confidence == 0.5  # default
        assert plan == ""

    def test_empty_decisions(self):
        response = {"decisions": [], "implementation_plan": "Do stuff"}
        decisions, plan = parse_planning_response(response)
        assert len(decisions) == 0
        assert plan == "Do stuff"

    def test_insufficient_alternatives_padded(self):
        response = {
            "decisions": [
                {
                    "question": "Pick one",
                    "category": "architecture",
                    "alternatives": [{"value": "only_one"}],
                    "chosen": "only_one",
                    "confidence": 0.9,
                }
            ]
        }

        decisions, _ = parse_planning_response(response)
        assert len(decisions) == 1
        assert len(decisions[0].alternatives) >= 2

    def test_sequence_start_offset(self):
        response = {
            "decisions": [
                {
                    "question": "Q",
                    "alternatives": [{"value": "A"}, {"value": "B"}],
                    "chosen": "A",
                }
            ]
        }

        decisions, _ = parse_planning_response(response, sequence_start=5)
        assert decisions[0].sequence_number == 5


# --- parse_coding_response ---


class TestParseCodingResponse:
    def test_normal_response(self):
        response = {
            "files": {
                "main.py": "print('hello')",
                "requirements.txt": "pandas\n",
            },
            "requirements": ["pandas", "yfinance"],
        }

        files, reqs = parse_coding_response(response)
        assert "main.py" in files
        assert files["main.py"] == "print('hello')"
        assert reqs == ["pandas", "yfinance"]

    def test_missing_files(self):
        response = {"requirements": ["pandas"]}
        files, reqs = parse_coding_response(response)
        assert files == {}
        assert reqs == ["pandas"]

    def test_missing_requirements(self):
        response = {"files": {"main.py": "code"}}
        files, reqs = parse_coding_response(response)
        assert files == {"main.py": "code"}
        assert reqs == []


# --- parse_evaluation_response ---


class TestParseEvaluationResponse:
    def test_success(self):
        response = {
            "status": "success",
            "assessment": "All good",
            "recovery_strategy": "",
            "next_action": "done",
        }
        result = parse_evaluation_response(response)
        assert result["status"] == "success"
        assert result["next_action"] == "done"

    def test_failed(self):
        response = {
            "status": "failed",
            "assessment": "Import error",
            "recovery_strategy": "Fix the import",
            "next_action": "fix",
        }
        result = parse_evaluation_response(response)
        assert result["status"] == "failed"
        assert result["next_action"] == "fix"

    def test_invalid_status_defaults(self):
        response = {
            "status": "garbage",
            "next_action": "whatever",
        }
        result = parse_evaluation_response(response)
        assert result["status"] == "failed"
        assert result["next_action"] == "fix"

    def test_done_sets_success(self):
        response = {
            "status": "partial",
            "next_action": "done",
        }
        result = parse_evaluation_response(response)
        assert result["status"] == "success"


# --- parse_recovery_response ---


class TestParseRecoveryResponse:
    def test_with_decisions(self):
        response = {
            "files": {"main.py": "fixed code"},
            "requirements": ["pandas"],
            "decisions": [
                {
                    "question": "Recovery approach?",
                    "category": "error_recovery",
                    "alternatives": [
                        {"value": "retry", "reasoning": "Transient"},
                        {"value": "rewrite", "reasoning": "Fundamental"},
                    ],
                    "chosen": "rewrite",
                    "reasoning": "Bug in logic",
                    "confidence": 0.9,
                }
            ],
        }

        files, reqs, decisions = parse_recovery_response(response)
        assert files == {"main.py": "fixed code"}
        assert reqs == ["pandas"]
        assert len(decisions) == 1
        assert decisions[0].chosen == "rewrite"

    def test_without_decisions(self):
        response = {
            "files": {"main.py": "fixed code"},
            "requirements": [],
        }

        files, reqs, decisions = parse_recovery_response(response)
        assert files == {"main.py": "fixed code"}
        assert len(decisions) == 0
