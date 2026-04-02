"""Prompt templates for the coding agent.

This file is the SINGLE SOURCE OF TRUTH for every prompt sent to the LLM.
Each prompt is a function that takes context parameters and returns a formatted string.
"""

from __future__ import annotations


SYSTEM_PROMPT = """\
You are a coding agent that builds complete, working Python solutions.

Your role:
1. Analyze the task and make explicit decisions at every branching point
2. Consider at least 2 alternatives for each decision
3. Output structured JSON for decisions so they can be traced and replayed
4. Write complete, production-ready Python code
5. Handle errors gracefully

For every decision you make, you MUST output structured reasoning with:
- The question being decided
- At least 2 alternatives considered (with reasoning for each)
- Your chosen option
- Why you chose it
- Your confidence level (0.0-1.0)

Decision categories: data_selection, algorithm_selection, architecture, \
library_selection, error_recovery, parameter_tuning
"""


def planning_prompt(task: str, locked_decisions: str = "") -> str:
    """Build the planning prompt.

    Instructs the LLM to produce a structured plan with explicit decisions
    as JSON. Each decision must have alternatives, a chosen value, reasoning,
    and confidence.
    """
    lock_section = ""
    if locked_decisions:
        lock_section = f"""
LOCKED DECISIONS (you MUST follow these exactly):
{locked_decisions}

For each locked decision above, use the specified value. Set locked=true.
Do not consider alternatives for locked decisions.
"""

    return f"""\
Given the following task, produce a structured implementation plan as JSON.

TASK: {task}

{lock_section}

Identify the key decisions required by THIS specific task. For each decision area \
(e.g., data source, algorithm, libraries, output format, error handling), you must:
- Frame it as a clear question
- Consider at least 2 concrete alternatives with reasoning for each
- Pick the best option and explain why
- Assign a confidence score (0.0–1.0)

Typical decision categories: data_selection, algorithm_selection, architecture, \
library_selection, error_recovery, parameter_tuning

Respond with ONLY valid JSON in this exact format (no preamble, no markdown fences):
{{
  "decisions": [
    {{
      "question": "Which data source to use?",
      "category": "data_selection",
      "alternatives": [
        {{"value": "yfinance", "reasoning": "Free, no API key needed, well-maintained"}},
        {{"value": "alpha_vantage", "reasoning": "More reliable but requires API key"}}
      ],
      "chosen": "yfinance",
      "reasoning": "Zero-friction setup, sufficient for the task",
      "confidence": 0.9
    }}
  ],
  "implementation_plan": "Step-by-step plan for coding the solution..."
}}"""


def coding_prompt(
    task: str,
    plan: str,
    decisions_summary: str,
    iteration: int = 0,
    previous_error: str = "",
) -> str:
    """Build the coding prompt.

    Instructs the LLM to write complete Python code based on the plan and decisions.
    """
    error_context = ""
    if previous_error:
        error_context = f"""
PREVIOUS ATTEMPT FAILED with this error:
{previous_error}

Fix the issues and produce corrected code. Do NOT repeat the same mistakes.
"""

    return f"""\
Write complete Python code to implement the following task.

TASK: {task}

PLAN:
{plan}

DECISIONS MADE:
{decisions_summary}

{error_context}

REQUIREMENTS:
- Write complete, runnable Python code (not pseudocode)
- Pull real data from yfinance (no mocked datasets)
- Detect anomalies using the chosen method
- Generate a summary report
- Expose results through a simple API endpoint
- Include proper error handling
- Include a requirements.txt with all dependencies
- The main entry point should be main.py with an `if __name__ == "__main__"` block

This is iteration {iteration} of the code-execute-evaluate loop.

Respond with ONLY valid JSON in this exact format:
{{
  "files": {{
    "main.py": "# Full content of main.py\\n...",
    "requirements.txt": "yfinance\\npandas\\n..."
  }},
  "requirements": ["yfinance", "pandas", "numpy"],
  "entry_point": "main.py"
}}"""


def evaluation_prompt(
    task: str,
    stdout: str,
    stderr: str,
    exit_code: int,
    code_files: dict[str, str],
) -> str:
    """Build the evaluation prompt.

    Instructs the LLM to assess execution results and determine next action.
    """
    # Truncate long outputs to stay within token limits
    max_output = 3000
    if len(stdout) > max_output:
        stdout = stdout[:max_output] + "\n... [truncated]"
    if len(stderr) > max_output:
        stderr = stderr[:max_output] + "\n... [truncated]"

    file_list = "\n".join(f"  - {name}" for name in code_files.keys())

    return f"""\
Evaluate the execution results of the generated code.

TASK: {task}

FILES WRITTEN:
{file_list}

EXECUTION RESULTS:
- Exit code: {exit_code}
- stdout:
{stdout}

- stderr:
{stderr}

Assess:
1. Did the code run successfully (exit_code == 0)?
2. Did it produce the expected outputs (data fetched, anomalies detected, report generated)?
3. What errors occurred and what recovery strategy should be used?

Respond with ONLY valid JSON in this exact format:
{{
  "status": "success" or "partial" or "failed",
  "assessment": "What happened and why",
  "recovery_strategy": "How to fix if not success (empty string if success)",
  "next_action": "fix" or "retry" or "abort" or "done"
}}

Rules:
- "done" means code ran perfectly, all outputs produced
- "fix" means there are fixable errors, provide recovery_strategy
- "retry" means transient error (network timeout), try again as-is
- "abort" means fundamental issue that cannot be fixed (e.g., impossible task)"""


def recovery_prompt(
    task: str,
    plan: str,
    code_files: dict[str, str],
    error: str,
    assessment: str,
    recovery_strategy: str,
) -> str:
    """Build the recovery prompt.

    Instructs the LLM to fix the failing code based on the error and assessment.
    """
    # Include the full code of files for context
    code_context = ""
    for name, content in code_files.items():
        code_context += f"\n--- {name} ---\n{content}\n"

    return f"""\
The previous code execution failed. Fix the code based on the error analysis.

TASK: {task}

PLAN:
{plan}

CURRENT CODE:
{code_context}

ERROR:
{error}

ASSESSMENT:
{assessment}

RECOVERY STRATEGY:
{recovery_strategy}

Fix the code and return the corrected version. You may also make new decisions
about error recovery approaches (category: error_recovery).

Respond with ONLY valid JSON in this exact format:
{{
  "files": {{
    "main.py": "# Fixed content of main.py\\n...",
    "requirements.txt": "yfinance\\npandas\\n..."
  }},
  "requirements": ["yfinance", "pandas", "numpy"],
  "entry_point": "main.py",
  "decisions": [
    {{
      "question": "Which recovery approach to use?",
      "category": "error_recovery",
      "alternatives": [
        {{"value": "option_a", "reasoning": "..."}},
        {{"value": "option_b", "reasoning": "..."}}
      ],
      "chosen": "option_a",
      "reasoning": "...",
      "confidence": 0.8
    }}
  ]
}}

The "decisions" array is optional - include it only if you made new decisions during recovery."""


def lock_decision_prompt(question: str, locked_value: str) -> str:
    """Build a lock injection for a single decision.

    This is injected into the planning prompt to force a specific decision value.
    """
    return (
        f"For \"{question}\", you MUST choose {locked_value}. "
        f"Do not consider other options. Set locked=true in your response."
    )
