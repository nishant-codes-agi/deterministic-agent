"""Response parsing utilities for LLM outputs.

Defensively handles common LLM deviations: markdown fences, preamble text,
missing fields, extra fields, and wrong types. Each parser uses sensible
defaults rather than crashing.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from uuid import uuid4

from src.models.decisions import DecisionPoint
from src.models.enums import AgentPhase, DecisionCategory

logger = logging.getLogger(__name__)


class ParseError(Exception):
    """Raised when LLM output cannot be parsed.

    Attributes:
        raw_text: The original text that failed to parse.
    """

    def __init__(self, message: str, raw_text: str = "") -> None:
        super().__init__(message)
        self.raw_text = raw_text


def parse_json_from_llm(text: str) -> dict:
    """Extract JSON from LLM response that may include markdown or preamble.

    Handles:
    - Bare JSON string
    - ```json ... ``` fenced blocks
    - Mixed text with JSON block ("Here's the plan:\\n```json...")
    - Multiple JSON blocks (takes the first valid one)

    Raises:
        ParseError: If no valid JSON can be extracted.
    """
    text = text.strip()

    # Try bare JSON first
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try ```json ... ``` fenced blocks
    matches = re.findall(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    for match in matches:
        match = match.strip()
        if match.startswith("{"):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    # Try to find any JSON object in the text
    # Use a greedy search for the outermost braces
    brace_depth = 0
    start_idx = None
    for i, char in enumerate(text):
        if char == "{":
            if brace_depth == 0:
                start_idx = i
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
            if brace_depth == 0 and start_idx is not None:
                candidate = text[start_idx : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    start_idx = None
                    continue

    raise ParseError(
        "Could not extract valid JSON from LLM response",
        raw_text=text,
    )


def _normalize_category(raw: str) -> DecisionCategory:
    """Normalize a category string to a DecisionCategory enum value."""
    raw = raw.strip().lower().replace(" ", "_").replace("-", "_")
    try:
        return DecisionCategory(raw)
    except ValueError:
        # Try fuzzy matching
        for cat in DecisionCategory:
            if raw in cat.value or cat.value in raw:
                return cat
        logger.warning(f"Unknown decision category '{raw}', defaulting to architecture")
        return DecisionCategory.ARCHITECTURE


def parse_planning_response(
    response: dict, sequence_start: int = 0
) -> tuple[list[DecisionPoint], str]:
    """Convert raw planning JSON into DecisionPoint models + plan text.

    Handles missing fields, wrong types, and extra fields gracefully.

    Args:
        response: Parsed JSON dict from the planning LLM response.
        sequence_start: Starting sequence number for decision IDs.

    Returns:
        Tuple of (list of DecisionPoint models, implementation plan text).
    """
    decisions: list[DecisionPoint] = []
    raw_decisions = response.get("decisions", [])

    if not isinstance(raw_decisions, list):
        logger.warning("'decisions' field is not a list, wrapping in list")
        raw_decisions = [raw_decisions] if raw_decisions else []

    for i, raw in enumerate(raw_decisions):
        if not isinstance(raw, dict):
            logger.warning(f"Decision {i} is not a dict, skipping")
            continue

        # Parse alternatives
        raw_alts = raw.get("alternatives", [])
        if not isinstance(raw_alts, list):
            raw_alts = []

        alternatives = []
        for alt in raw_alts:
            if isinstance(alt, dict):
                alternatives.append({
                    "value": str(alt.get("value", "unknown")),
                    "reasoning": str(alt.get("reasoning", "")),
                })
            elif isinstance(alt, str):
                alternatives.append({"value": alt, "reasoning": ""})

        # Ensure at least 2 alternatives
        chosen = str(raw.get("chosen", ""))
        if len(alternatives) < 2:
            if chosen and not any(a["value"] == chosen for a in alternatives):
                alternatives.append({"value": chosen, "reasoning": "Selected option"})
            while len(alternatives) < 2:
                alternatives.append({"value": "other", "reasoning": "Default alternative"})

        # Parse confidence
        confidence = raw.get("confidence", 0.5)
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.5

        try:
            dp = DecisionPoint(
                id=f"dp-{uuid4().hex[:8]}",
                sequence_number=sequence_start + i,
                timestamp=datetime.now(timezone.utc),
                category=_normalize_category(str(raw.get("category", "architecture"))),
                phase=AgentPhase.PLANNING,
                question=str(raw.get("question", f"Decision {i}")),
                alternatives=alternatives,
                chosen=chosen or str(alternatives[0]["value"]),
                reasoning=str(raw.get("reasoning", "")),
                confidence=confidence,
                locked=bool(raw.get("locked", False)),
            )
            decisions.append(dp)
        except Exception as e:
            logger.warning(f"Failed to create DecisionPoint {i}: {e}")
            continue

    plan_text = str(response.get("implementation_plan", ""))

    return decisions, plan_text


def parse_coding_response(response: dict) -> tuple[dict[str, str], list[str]]:
    """Extract file contents and requirements from coding response.

    Args:
        response: Parsed JSON dict from the coding LLM response.

    Returns:
        Tuple of (files dict mapping filename->content, requirements list).
    """
    files = response.get("files", {})
    if not isinstance(files, dict):
        logger.warning("'files' field is not a dict")
        files = {}

    # Ensure all values are strings
    clean_files: dict[str, str] = {}
    for name, content in files.items():
        clean_files[str(name)] = str(content)

    requirements = response.get("requirements", [])
    if not isinstance(requirements, list):
        requirements = []
    requirements = [str(r) for r in requirements]

    return clean_files, requirements


def parse_evaluation_response(response: dict) -> dict:
    """Validate and return the assessment dict.

    Args:
        response: Parsed JSON dict from the evaluation LLM response.

    Returns:
        Validated assessment dict with status, assessment, recovery_strategy, next_action.
    """
    valid_statuses = {"success", "partial", "failed"}
    valid_actions = {"fix", "retry", "abort", "done"}

    status = str(response.get("status", "failed")).lower()
    if status not in valid_statuses:
        status = "failed"

    next_action = str(response.get("next_action", "fix")).lower()
    if next_action not in valid_actions:
        next_action = "fix"

    # Map "done" to success status for consistency
    if next_action == "done":
        status = "success"

    return {
        "status": status,
        "assessment": str(response.get("assessment", "No assessment provided")),
        "recovery_strategy": str(response.get("recovery_strategy", "")),
        "next_action": next_action,
    }


def parse_recovery_response(
    response: dict, sequence_start: int = 0
) -> tuple[dict[str, str], list[str], list[DecisionPoint]]:
    """Parse recovery response into files, requirements, and optional new decisions.

    Args:
        response: Parsed JSON dict from the recovery LLM response.
        sequence_start: Starting sequence number for any new decisions.

    Returns:
        Tuple of (files dict, requirements list, new decisions list).
    """
    files, requirements = parse_coding_response(response)

    # Parse optional recovery decisions
    decisions: list[DecisionPoint] = []
    raw_decisions = response.get("decisions", [])
    if isinstance(raw_decisions, list):
        for i, raw in enumerate(raw_decisions):
            if not isinstance(raw, dict):
                continue

            raw_alts = raw.get("alternatives", [])
            alternatives = []
            for alt in raw_alts if isinstance(raw_alts, list) else []:
                if isinstance(alt, dict):
                    alternatives.append({
                        "value": str(alt.get("value", "unknown")),
                        "reasoning": str(alt.get("reasoning", "")),
                    })

            chosen = str(raw.get("chosen", ""))
            while len(alternatives) < 2:
                alternatives.append({"value": "other", "reasoning": "Default alternative"})

            confidence = raw.get("confidence", 0.5)
            try:
                confidence = max(0.0, min(1.0, float(confidence)))
            except (ValueError, TypeError):
                confidence = 0.5

            try:
                dp = DecisionPoint(
                    id=f"dp-{uuid4().hex[:8]}",
                    sequence_number=sequence_start + i,
                    timestamp=datetime.now(timezone.utc),
                    category=_normalize_category(
                        str(raw.get("category", "error_recovery"))
                    ),
                    phase=AgentPhase.RECOVERING,
                    question=str(raw.get("question", f"Recovery decision {i}")),
                    alternatives=alternatives,
                    chosen=chosen or str(alternatives[0]["value"]),
                    reasoning=str(raw.get("reasoning", "")),
                    confidence=confidence,
                )
                decisions.append(dp)
            except Exception as e:
                logger.warning(f"Failed to create recovery DecisionPoint: {e}")

    return files, requirements, decisions
