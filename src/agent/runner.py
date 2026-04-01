"""AgentRunner - the coding agent state machine.

Implements the Plan -> Code -> Execute -> Evaluate loop with per-phase
model routing via OpenRouter. This is the heart of the system.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from src.agent.base import (
    AgentEvent,
    AgentEventType,
    CodingAgent,
    EventHandler,
)
from src.agent.parsers import (
    ParseError,
    parse_coding_response,
    parse_evaluation_response,
    parse_json_from_llm,
    parse_planning_response,
    parse_recovery_response,
)
from src.agent.prompts import (
    SYSTEM_PROMPT,
    coding_prompt,
    evaluation_prompt,
    lock_decision_prompt,
    planning_prompt,
    recovery_prompt,
)
from src.config import AgentConfig, LLMConfig
from src.llm.base import LLMMessage, LLMProvider, LLMResponse
from src.models.decisions import (
    DecisionPoint,
    ExecutionRecord,
    LLMCallRecord,
)
from src.models.enums import AgentPhase, RunStatus
from src.models.traces import DecisionTrace, PathLockConfig, RunMetadata
from src.sandbox.base import SandboxProvider

logger = logging.getLogger(__name__)


class AgentRunner(CodingAgent):
    """Coding agent that uses per-phase model routing for cost optimization.

    Each phase uses the optimal model for its task:
    - Planning: Gemini 2.5 Flash (structured reasoning, cheap)
    - Coding: Claude Sonnet 4 (highest first-pass code quality)
    - Evaluation: GPT-4o-mini (cheap classification)
    - Recovery: Claude Sonnet 4 (code comprehension)

    All models are accessed through a single OpenRouter provider instance.
    The `model` parameter on _call_llm() handles the routing.
    """

    def __init__(
        self,
        llm: LLMProvider,
        sandbox: SandboxProvider,
        config: AgentConfig,
        llm_config: LLMConfig,
    ) -> None:
        self._llm = llm
        self._sandbox = sandbox
        self._config = config
        self._llm_config = llm_config
        self._event_handler: Optional[EventHandler] = None

    def _get_model(self, phase: AgentPhase) -> str:
        """Get the configured model for a given agent phase."""
        return self._llm_config.get_model_for_phase(phase)

    async def run(
        self,
        task: str,
        lock_config: Optional[PathLockConfig] = None,
        event_handler: Optional[EventHandler] = None,
    ) -> DecisionTrace:
        """Execute the agent on a task.

        Runs the full Plan -> Code -> Execute -> Evaluate loop, collecting
        decisions, execution records, and LLM call records along the way.
        """
        self._event_handler = event_handler
        run_id = f"run-{uuid4().hex[:8]}"
        workspace = await self._sandbox.setup_workspace(run_id)

        # Accumulators for trace data (replaced by PersistentTracer in Phase 4)
        decisions: list[DecisionPoint] = []
        executions: list[ExecutionRecord] = []
        llm_calls: list[LLMCallRecord] = []

        try:
            # Phase 1: PLANNING (uses Gemini 2.5 Flash)
            await self._emit(AgentEvent(
                event_type=AgentEventType.PHASE_CHANGED,
                run_id=run_id,
                phase=AgentPhase.PLANNING,
            ))
            plan, plan_decisions = await self._plan(
                task, run_id, decisions, llm_calls, lock_config
            )
            decisions.extend(plan_decisions)

            # Track code state across iterations
            code_files: dict[str, str] = {}
            requirements: list[str] = []

            # Phase 2-4: CODE -> EXECUTE -> EVALUATE loop
            for iteration in range(self._config.max_iterations):
                # CODING (uses Claude Sonnet 4)
                await self._emit(AgentEvent(
                    event_type=AgentEventType.PHASE_CHANGED,
                    run_id=run_id,
                    phase=AgentPhase.CODING,
                ))

                previous_error = ""
                if executions and executions[-1].exit_code != 0:
                    previous_error = executions[-1].stderr

                code_files, requirements = await self._code(
                    task, plan, decisions, llm_calls, run_id,
                    iteration=iteration,
                    previous_error=previous_error,
                )

                # Write files to workspace
                await self._write_files(workspace, code_files)

                # Install dependencies
                if requirements:
                    await self._sandbox.install_packages(requirements, workspace)

                # EXECUTING (no LLM call - pure subprocess)
                await self._emit(AgentEvent(
                    event_type=AgentEventType.PHASE_CHANGED,
                    run_id=run_id,
                    phase=AgentPhase.EXECUTING,
                ))
                exec_result = await self._execute(
                    workspace, code_files, run_id, executions
                )

                # EVALUATING (uses GPT-4o-mini)
                await self._emit(AgentEvent(
                    event_type=AgentEventType.PHASE_CHANGED,
                    run_id=run_id,
                    phase=AgentPhase.EVALUATING,
                ))
                assessment = await self._evaluate(
                    exec_result, code_files, task, decisions, llm_calls, run_id
                )

                if assessment["status"] == "success":
                    return self._build_trace(
                        run_id, task, decisions, executions,
                        llm_calls, RunStatus.SUCCESS,
                    )

                if assessment["next_action"] == "abort":
                    return self._build_trace(
                        run_id, task, decisions, executions,
                        llm_calls, RunStatus.FAILED,
                        error=assessment["assessment"],
                    )

                if assessment["next_action"] == "retry":
                    # Just loop again with the same code
                    continue

                # RECOVERY (uses Claude Sonnet 4)
                await self._emit(AgentEvent(
                    event_type=AgentEventType.PHASE_CHANGED,
                    run_id=run_id,
                    phase=AgentPhase.RECOVERING,
                ))
                plan, recovery_decisions = await self._recover(
                    task, plan, assessment, code_files,
                    decisions, executions, llm_calls, run_id,
                )
                decisions.extend(recovery_decisions)

            # Max iterations exceeded
            return self._build_trace(
                run_id, task, decisions, executions,
                llm_calls, RunStatus.FAILED,
                error="Max iterations exceeded",
            )

        except Exception as e:
            logger.exception(f"Agent run {run_id} crashed: {e}")
            return self._build_trace(
                run_id, task, decisions, executions,
                llm_calls, RunStatus.PARTIAL,
                error=str(e),
            )
        finally:
            # Don't cleanup workspace - keep artifacts for inspection
            pass

    async def _plan(
        self,
        task: str,
        run_id: str,
        decisions: list[DecisionPoint],
        llm_calls: list[LLMCallRecord],
        lock_config: Optional[PathLockConfig] = None,
    ) -> tuple[str, list[DecisionPoint]]:
        """Planning phase: produce structured plan with decisions.

        Uses Gemini 2.5 Flash for structured reasoning.
        """
        # Build locked decisions string if path locking is active
        locked_str = ""
        if lock_config and lock_config.source_run_id:
            # For now, lock injection is handled by the caller
            # Phase 5 will implement full path locking
            locked_str = self._build_lock_injections(lock_config, decisions)

        prompt_text = planning_prompt(task, locked_decisions=locked_str)
        messages = [
            LLMMessage(role="system", content=SYSTEM_PROMPT),
            LLMMessage(role="user", content=prompt_text),
        ]

        response = await self._call_llm(
            messages, AgentPhase.PLANNING, llm_calls, run_id,
        )

        # Parse the planning response
        try:
            parsed = parse_json_from_llm(response.content)
        except ParseError as e:
            logger.warning(f"Failed to parse planning response: {e}")
            # Return empty plan with no decisions
            return "Failed to parse planning response", []

        plan_decisions, plan_text = parse_planning_response(
            parsed, sequence_start=len(decisions),
        )

        # Emit decision events
        for dp in plan_decisions:
            await self._emit(AgentEvent(
                event_type=AgentEventType.DECISION_MADE,
                run_id=run_id,
                phase=AgentPhase.PLANNING,
                payload={"decision_id": dp.id, "question": dp.question, "chosen": dp.chosen},
            ))

        return plan_text, plan_decisions

    async def _code(
        self,
        task: str,
        plan: str,
        decisions: list[DecisionPoint],
        llm_calls: list[LLMCallRecord],
        run_id: str,
        iteration: int = 0,
        previous_error: str = "",
    ) -> tuple[dict[str, str], list[str]]:
        """Coding phase: generate complete Python code.

        Uses Claude Sonnet 4 for highest first-pass code quality.
        """
        decisions_summary = self._format_decisions(decisions)
        prompt_text = coding_prompt(
            task, plan, decisions_summary,
            iteration=iteration,
            previous_error=previous_error,
        )

        messages = [
            LLMMessage(role="system", content=SYSTEM_PROMPT),
            LLMMessage(role="user", content=prompt_text),
        ]

        response = await self._call_llm(
            messages, AgentPhase.CODING, llm_calls, run_id,
        )

        try:
            parsed = parse_json_from_llm(response.content)
        except ParseError as e:
            logger.warning(f"Failed to parse coding response: {e}")
            # Try to extract code from raw text as fallback
            return {"main.py": response.content}, []

        code_files, requirements = parse_coding_response(parsed)

        await self._emit(AgentEvent(
            event_type=AgentEventType.CODE_WRITTEN,
            run_id=run_id,
            phase=AgentPhase.CODING,
            payload={"files": list(code_files.keys()), "iteration": iteration},
        ))

        return code_files, requirements

    async def _write_files(
        self, workspace: Path, code_files: dict[str, str]
    ) -> None:
        """Write code files to the workspace directory."""
        for filename, content in code_files.items():
            filepath = workspace / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(content)
            logger.debug(f"Wrote {filepath}")

    async def _execute(
        self,
        workspace: Path,
        code_files: dict[str, str],
        run_id: str,
        executions: list[ExecutionRecord],
    ) -> ExecutionRecord:
        """Execute the generated code in the sandbox.

        Detects the entry point (main.py or file with __name__ == '__main__').
        """
        await self._emit(AgentEvent(
            event_type=AgentEventType.EXECUTION_STARTED,
            run_id=run_id,
            phase=AgentPhase.EXECUTING,
        ))

        # Find entry point
        entry_point = self._find_entry_point(code_files)
        command = f"python {entry_point}"

        result = await self._sandbox.execute(
            command,
            cwd=workspace,
            timeout=self._config.execution_timeout,
        )

        # Build code snapshot for the execution record
        code_snapshot = "\n\n".join(
            f"# --- {name} ---\n{content}"
            for name, content in code_files.items()
        )

        exec_record = ExecutionRecord(
            code_snapshot=code_snapshot,
            command=command,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            duration_ms=result.duration_ms,
            artifacts_produced=[str(a) for a in result.artifacts],
        )
        executions.append(exec_record)

        await self._emit(AgentEvent(
            event_type=AgentEventType.EXECUTION_COMPLETED,
            run_id=run_id,
            phase=AgentPhase.EXECUTING,
            payload={
                "exit_code": result.exit_code,
                "duration_ms": result.duration_ms,
                "timed_out": result.timed_out,
            },
        ))

        return exec_record

    async def _evaluate(
        self,
        exec_result: ExecutionRecord,
        code_files: dict[str, str],
        task: str,
        decisions: list[DecisionPoint],
        llm_calls: list[LLMCallRecord],
        run_id: str,
    ) -> dict:
        """Evaluation phase: assess execution results.

        Uses GPT-4o-mini as a cheap JSON classifier.
        """
        prompt_text = evaluation_prompt(
            task=task,
            stdout=exec_result.stdout,
            stderr=exec_result.stderr,
            exit_code=exec_result.exit_code,
            code_files=code_files,
        )

        messages = [
            LLMMessage(role="system", content="You are a code evaluation assistant. Respond only with valid JSON."),
            LLMMessage(role="user", content=prompt_text),
        ]

        response = await self._call_llm(
            messages, AgentPhase.EVALUATING, llm_calls, run_id,
        )

        try:
            parsed = parse_json_from_llm(response.content)
        except ParseError:
            # If we can't parse, check exit code as fallback
            if exec_result.exit_code == 0:
                return {
                    "status": "success",
                    "assessment": "Code executed successfully (exit code 0)",
                    "recovery_strategy": "",
                    "next_action": "done",
                }
            return {
                "status": "failed",
                "assessment": f"Code failed with exit code {exec_result.exit_code}",
                "recovery_strategy": "Fix the error based on stderr output",
                "next_action": "fix",
            }

        return parse_evaluation_response(parsed)

    async def _recover(
        self,
        task: str,
        plan: str,
        assessment: dict,
        code_files: dict[str, str],
        decisions: list[DecisionPoint],
        executions: list[ExecutionRecord],
        llm_calls: list[LLMCallRecord],
        run_id: str,
    ) -> tuple[str, list[DecisionPoint]]:
        """Recovery phase: fix failing code.

        Uses Claude Sonnet 4 for code comprehension and debugging.
        Returns updated plan text and any new recovery decisions.
        """
        await self._emit(AgentEvent(
            event_type=AgentEventType.RECOVERY_STARTED,
            run_id=run_id,
            phase=AgentPhase.RECOVERING,
        ))

        last_error = ""
        if executions:
            last_exec = executions[-1]
            last_error = last_exec.stderr or last_exec.stdout

        prompt_text = recovery_prompt(
            task=task,
            plan=plan,
            code_files=code_files,
            error=last_error,
            assessment=assessment["assessment"],
            recovery_strategy=assessment["recovery_strategy"],
        )

        messages = [
            LLMMessage(role="system", content=SYSTEM_PROMPT),
            LLMMessage(role="user", content=prompt_text),
        ]

        response = await self._call_llm(
            messages, AgentPhase.RECOVERING, llm_calls, run_id,
        )

        try:
            parsed = parse_json_from_llm(response.content)
        except ParseError as e:
            logger.warning(f"Failed to parse recovery response: {e}")
            return plan, []

        files, requirements, recovery_decisions = parse_recovery_response(
            parsed, sequence_start=len(decisions),
        )

        # Update code_files in-place with the recovered code
        code_files.clear()
        code_files.update(files)

        # Emit decision events for recovery decisions
        for dp in recovery_decisions:
            await self._emit(AgentEvent(
                event_type=AgentEventType.DECISION_MADE,
                run_id=run_id,
                phase=AgentPhase.RECOVERING,
                payload={"decision_id": dp.id, "question": dp.question, "chosen": dp.chosen},
            ))

        return plan, recovery_decisions

    async def _call_llm(
        self,
        messages: list[LLMMessage],
        phase: AgentPhase,
        llm_calls: list[LLMCallRecord],
        run_id: str,
        response_format: Optional[type] = None,
    ) -> LLMResponse:
        """Central LLM call method with per-phase model routing.

        Records every call for the decision trace.
        """
        model = self._get_model(phase)
        response = await self._llm.complete(
            messages=messages,
            model=model,
            response_format=response_format,
        )

        # Record LLM call
        record = LLMCallRecord(
            phase=phase,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            response=response.content,
            structured_output=response.structured,
            model=response.model,
            temperature=self._llm_config.temperature,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            latency_ms=response.latency_ms,
            cost_usd=response.cost_usd,
        )
        llm_calls.append(record)

        await self._emit(AgentEvent(
            event_type=AgentEventType.COST_UPDATE,
            run_id=run_id,
            phase=phase,
            payload={
                "model": response.model,
                "cost_usd": response.cost_usd,
                "tokens_in": response.tokens_in,
                "tokens_out": response.tokens_out,
                "total_cost_usd": sum(c.cost_usd for c in llm_calls),
            },
        ))

        return response

    def _build_trace(
        self,
        run_id: str,
        task: str,
        decisions: list[DecisionPoint],
        executions: list[ExecutionRecord],
        llm_calls: list[LLMCallRecord],
        status: RunStatus,
        error: Optional[str] = None,
    ) -> DecisionTrace:
        """Construct the complete DecisionTrace from accumulated data."""
        # Build model routing map
        model_routing = {
            "planning": self._get_model(AgentPhase.PLANNING),
            "coding": self._get_model(AgentPhase.CODING),
            "evaluating": self._get_model(AgentPhase.EVALUATING),
            "recovering": self._get_model(AgentPhase.RECOVERING),
        }

        total_tokens = sum(c.tokens_in + c.tokens_out for c in llm_calls)
        total_cost = sum(c.cost_usd for c in llm_calls)

        metadata = RunMetadata(
            run_id=run_id,
            task_description=task,
            llm_provider=self._llm.provider_name,
            model_routing=model_routing,
            temperature=self._llm_config.temperature,
            status=status,
            completed_at=datetime.now(timezone.utc),
            total_llm_calls=len(llm_calls),
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            error=error,
        )

        return DecisionTrace(
            metadata=metadata,
            decisions=decisions,
            executions=executions,
            llm_calls=llm_calls,
        )

    async def _emit(self, event: AgentEvent) -> None:
        """Emit an event if handler is set."""
        if self._event_handler:
            try:
                await self._event_handler.handle(event)
            except Exception as e:
                logger.warning(f"Event handler error: {e}")

    @staticmethod
    def _find_entry_point(code_files: dict[str, str]) -> str:
        """Find the entry point file from the code files.

        Looks for main.py first, then any file with `if __name__`.
        """
        if "main.py" in code_files:
            return "main.py"

        for name, content in code_files.items():
            if name.endswith(".py") and "__name__" in content:
                return name

        # Fallback: first .py file
        for name in code_files:
            if name.endswith(".py"):
                return name

        return "main.py"

    @staticmethod
    def _format_decisions(decisions: list[DecisionPoint]) -> str:
        """Format decisions as a readable summary for prompts."""
        if not decisions:
            return "No decisions made yet."

        lines = []
        for dp in decisions:
            lock_marker = " [LOCKED]" if dp.locked else ""
            lines.append(
                f"- {dp.category.value}: {dp.question} -> {dp.chosen}{lock_marker} "
                f"(confidence: {dp.confidence:.2f})"
            )
        return "\n".join(lines)

    @staticmethod
    def _build_lock_injections(
        lock_config: PathLockConfig,
        existing_decisions: list[DecisionPoint],
    ) -> str:
        """Build lock injection strings from a PathLockConfig.

        This is a placeholder for Phase 5's full path locking implementation.
        """
        # For now, just return empty - Phase 5 will implement this
        return ""
