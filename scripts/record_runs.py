"""Record 5 genuine agent runs to demonstrate path variance.

Runs the agent 5 times on the SAME task with NO locking.
Each run should produce a genuinely different solution.
After running, produces a variance report: runs/variance_report.json
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.runner import AgentRunner
from src.config import get_settings
from src.llm.factory import create_llm_provider
from src.models.enums import RunStatus
from src.sandbox.subprocess_sandbox import SubprocessSandbox
from src.tracing.store import TraceStore

TASK = """\
Pull stock data from yfinance — you choose which ticker(s), timeframe, and \
indicators. Detect anomalous trading days using a statistical method of your \
choosing. Generate a summary report with visualizations saved to anomalies.png \
and a JSON report anomaly_report.json. Expose the results through a simple \
FastAPI endpoint. Write everything in a single main.py with a clear \
if __name__ == "__main__" block.\
"""

NUM_RUNS = 5

# Use varied temperatures per run to encourage genuinely different decision paths.
# The first run uses 0.0 (baseline), then increasing temperatures for more variance.
RUN_TEMPERATURES = [0.0, 0.4, 0.7, 0.9, 1.0]


def _shannon_entropy(values: list[str]) -> float:
    """Compute Shannon entropy for a list of categorical values."""
    if not values:
        return 0.0
    total = len(values)
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 4)


def _most_common(values: list[str]) -> str:
    """Return the most common value in a list."""
    if not values:
        return ""
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get)


async def record_five_runs() -> list:
    """Execute 5 agent runs and return their traces."""
    settings = get_settings()
    sandbox = SubprocessSandbox(runs_dir=settings.runs_dir)

    traces = []
    for i in range(NUM_RUNS):
        # Use a different temperature for each run to encourage decision variance
        temp = RUN_TEMPERATURES[i] if i < len(RUN_TEMPERATURES) else 0.7
        llm_config = settings.llm.model_copy(update={"temperature": temp})
        llm = create_llm_provider(llm_config)

        agent = AgentRunner(
            llm=llm,
            sandbox=sandbox,
            config=settings.agent,
            llm_config=llm_config,
            runs_dir=settings.runs_dir,
        )

        print(f"\n{'=' * 60}")
        print(f"RUN {i + 1}/{NUM_RUNS}  (temperature={temp})")
        print(f"{'=' * 60}\n")

        run_id = None
        try:
            trace = await agent.run(TASK)
            run_id = trace.metadata.run_id
        except Exception as e:
            print(f"Run {i + 1} failed with exception: {e}")
            continue
        finally:
            # Always clean up the workspace to free disk space
            if run_id:
                await sandbox.cleanup_workspace(run_id)

        # Print summary
        print(f"Run ID: {trace.metadata.run_id}")
        print(f"Status: {trace.metadata.status.value}")
        print(f"Models used: {trace.metadata.model_routing}")
        print(f"Decisions:")
        for d in trace.decisions:
            print(f"  [{d.category.value}] {d.question} -> {d.chosen}")
        print(f"Cost: ${trace.metadata.total_cost_usd:.4f}")

        # Per-model cost breakdown
        model_costs: dict[str, float] = {}
        for call in trace.llm_calls:
            model_costs[call.model] = model_costs.get(call.model, 0) + call.cost_usd
        for model, cost in sorted(model_costs.items()):
            print(f"  {model}: ${cost:.6f}")
        print()

        traces.append(trace)

    return traces


def build_variance_report(traces: list, settings) -> dict:
    """Build a variance report from the recorded traces."""
    if not traces:
        return {"error": "No successful runs to analyze"}

    # Collect decisions by normalized question
    decisions_by_question: dict[str, list[dict]] = defaultdict(list)
    for trace in traces:
        for dp in trace.decisions:
            key = dp.question.lower().strip()
            decisions_by_question[key].append({
                "run_id": trace.metadata.run_id,
                "category": dp.category.value,
                "chosen": dp.chosen,
            })

    # Compute variance for each decision question
    decision_variance = {}
    for question, entries in decisions_by_question.items():
        values = [e["chosen"] for e in entries]
        unique = sorted(set(values))
        category = entries[0]["category"]

        # Use a friendlier key name
        friendly_key = category
        if "ticker" in question:
            friendly_key = "ticker_selection"
        elif "method" in question or "anomaly" in question:
            friendly_key = "anomaly_method"
        elif "framework" in question or "api" in question:
            friendly_key = "api_framework"
        elif "timeframe" in question or "period" in question:
            friendly_key = "timeframe"
        elif "visualization" in question or "library" in question:
            friendly_key = "visualization_library"

        decision_variance[friendly_key] = {
            "question": question,
            "unique_values": len(unique),
            "values": unique,
            "entropy": _shannon_entropy(values),
            "most_common": _most_common(values),
            "samples": len(values),
        }

    # Aggregate costs
    total_cost = sum(t.metadata.total_cost_usd for t in traces)
    avg_cost = total_cost / len(traces) if traces else 0

    cost_by_model: dict[str, float] = {}
    cost_by_phase: dict[str, float] = {}
    for trace in traces:
        for call in trace.llm_calls:
            cost_by_model[call.model] = cost_by_model.get(call.model, 0) + call.cost_usd
            cost_by_phase[call.phase.value] = cost_by_phase.get(call.phase.value, 0) + call.cost_usd

    # Collect unique tickers, methods, frameworks
    unique_tickers = set()
    unique_methods = set()
    unique_frameworks = set()
    for trace in traces:
        for dp in trace.decisions:
            cat = dp.category.value
            val = dp.chosen.lower()
            if cat == "data_selection":
                unique_tickers.add(dp.chosen)
            elif cat == "algorithm_selection":
                unique_methods.add(dp.chosen)
            elif cat == "library_selection" and ("api" in dp.question.lower() or "framework" in dp.question.lower()):
                unique_frameworks.add(dp.chosen)

    # Get model routing from settings
    model_routing = {
        "planning": settings.llm.model_planning,
        "coding": settings.llm.model_coding,
        "evaluation": settings.llm.model_evaluation,
        "recovery": settings.llm.model_recovery,
    }
    if settings.llm.model_override:
        model_routing = {k: settings.llm.model_override for k in model_routing}

    # Per-run summaries
    run_summaries = []
    for trace in traces:
        models_used = set()
        for call in trace.llm_calls:
            models_used.add(call.model)

        # Find key decisions
        ticker = None
        method = None
        for dp in trace.decisions:
            if dp.category.value == "data_selection" and ticker is None:
                ticker = dp.chosen
            elif dp.category.value == "algorithm_selection" and method is None:
                method = dp.chosen

        iterations = len(trace.executions)

        run_summaries.append({
            "run_id": trace.metadata.run_id,
            "status": trace.metadata.status.value,
            "ticker": ticker,
            "method": method,
            "cost_usd": round(trace.metadata.total_cost_usd, 6),
            "models_used": len(models_used),
            "iterations": iterations,
            "decisions_count": len(trace.decisions),
        })

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_runs": len(traces),
        "successful_runs": sum(1 for t in traces if t.metadata.status == RunStatus.SUCCESS),
        "llm_strategy": "multi-model via OpenRouter",
        "model_routing": model_routing,
        "unique_tickers": sorted(unique_tickers),
        "unique_methods": sorted(unique_methods),
        "unique_frameworks": sorted(unique_frameworks),
        "decision_variance": decision_variance,
        "cost_summary": {
            "total_all_runs_usd": round(total_cost, 6),
            "avg_per_run_usd": round(avg_cost, 6),
            "cost_by_model": {k: round(v, 6) for k, v in sorted(cost_by_model.items())},
            "cost_by_phase": {k: round(v, 6) for k, v in sorted(cost_by_phase.items())},
        },
        "runs": run_summaries,
    }

    return report


async def main() -> None:
    """Record 5 runs and produce a variance report."""
    settings = get_settings()

    # Ensure runs directory exists
    runs_dir = Path(settings.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RECORDING 5 AGENT RUNS FOR VARIANCE ANALYSIS")
    print("=" * 60)
    print(f"\nTask: {TASK[:100]}...")
    print(f"Runs directory: {runs_dir}")
    print(f"LLM Strategy: multi-model via OpenRouter")
    print(f"  Planning:   {settings.llm.model_planning}")
    print(f"  Coding:     {settings.llm.model_coding}")
    print(f"  Evaluation: {settings.llm.model_evaluation}")
    print(f"  Recovery:   {settings.llm.model_recovery}")
    if settings.llm.model_override:
        print(f"  OVERRIDE:   {settings.llm.model_override}")
    print()

    # Record runs
    traces = await record_five_runs()

    if not traces:
        print("\nERROR: No successful runs recorded.")
        sys.exit(1)

    # Build and write variance report
    report = build_variance_report(traces, settings)
    report_path = runs_dir / "variance_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nVariance report written to: {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("VARIANCE ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\nTotal runs: {report['total_runs']}")
    print(f"Successful: {report['successful_runs']}")
    print(f"Unique tickers: {report['unique_tickers']}")
    print(f"Unique methods: {report['unique_methods']}")
    print(f"Unique frameworks: {report['unique_frameworks']}")

    print(f"\nDecision Variance:")
    for key, var in report["decision_variance"].items():
        indicator = "HIGH" if var["entropy"] > 1.0 else "MEDIUM" if var["entropy"] > 0.5 else "LOW"
        print(f"  {key}: {var['unique_values']} unique values, entropy={var['entropy']} [{indicator}]")
        print(f"    Values: {var['values']}")

    print(f"\nCost Summary:")
    print(f"  Total: ${report['cost_summary']['total_all_runs_usd']:.4f}")
    print(f"  Avg per run: ${report['cost_summary']['avg_per_run_usd']:.4f}")
    print(f"  By model:")
    for model, cost in report["cost_summary"]["cost_by_model"].items():
        print(f"    {model}: ${cost:.6f}")
    print(f"  By phase:")
    for phase, cost in report["cost_summary"]["cost_by_phase"].items():
        print(f"    {phase}: ${cost:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
