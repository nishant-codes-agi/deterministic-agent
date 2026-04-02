"""CLI interface for the Deterministic Agent.

Commands: run, replay, trace, list, fork, compare, estimate.
Uses Click for commands and Rich for pretty output.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

console = Console()


def _run_async(coro):
    """Run an async coroutine from synchronous Click commands."""
    return asyncio.run(coro)


def _load_task(task: Optional[str], task_file: Optional[Path]) -> str:
    """Load task description from argument or file."""
    if task_file:
        return task_file.read_text().strip()
    if task:
        return task
    console.print("[red]Error: Provide a task description or --task-file[/red]")
    raise SystemExit(1)


def _get_settings():
    """Get application settings."""
    from src.config import get_settings
    return get_settings()


def _build_agent(settings, model_override: Optional[str] = None):
    """Build an AgentRunner with the given settings."""
    from src.agent.runner import AgentRunner
    from src.llm.factory import create_llm_provider
    from src.sandbox.subprocess_sandbox import SubprocessSandbox

    llm_config = settings.llm
    if model_override:
        llm_config = llm_config.model_copy(update={"model_override": model_override})

    llm = create_llm_provider(llm_config)
    sandbox = SubprocessSandbox(runs_dir=settings.runs_dir)

    return AgentRunner(
        llm=llm,
        sandbox=sandbox,
        config=settings.agent,
        llm_config=llm_config,
        runs_dir=settings.runs_dir,
    )


def _build_trace_store(settings):
    """Build a TraceStore."""
    from src.tracing.store import TraceStore
    return TraceStore(runs_dir=settings.runs_dir)


def _show_cost_estimate(settings, model_override: Optional[str] = None):
    """Show a cost estimate panel."""
    from src.config import LLMConfig
    from src.models.enums import AgentPhase

    llm_config = settings.llm
    if model_override:
        llm_config = llm_config.model_copy(update={"model_override": model_override})

    # Per-phase cost estimates (rough averages based on typical token usage)
    phase_estimates = {
        "Planning": {
            "model": llm_config.get_model_for_phase(AgentPhase.PLANNING),
            "cost": 0.006,
        },
        "Coding": {
            "model": llm_config.get_model_for_phase(AgentPhase.CODING),
            "cost": 0.072,
        },
        "Evaluation": {
            "model": llm_config.get_model_for_phase(AgentPhase.EVALUATING),
            "cost": 0.001,
        },
        "Recovery": {
            "model": llm_config.get_model_for_phase(AgentPhase.RECOVERING),
            "cost": 0.027,
        },
    }

    total = sum(p["cost"] for p in phase_estimates.values())
    low = total * 0.6
    high = total * 1.8

    lines = []
    strategy = "Single model" if llm_config.model_override else "Multi-model via OpenRouter"
    lines.append(f"Strategy: {strategy}\n")
    for phase_name, info in phase_estimates.items():
        lines.append(f"{phase_name + ':':14s} {info['model']:40s} ${info['cost']:.3f}")
    lines.append("─" * 60)
    lines.append(f"Estimated total: ${total:.3f} (range: ${low:.2f}-${high:.2f})")
    lines.append(f"Budget: ${settings.agent.max_cost_usd:.2f}")

    console.print(Panel("\n".join(lines), title="Cost Estimate", border_style="cyan"))
    return total


def _render_trace_tree(trace) -> Tree:
    """Render a decision trace as a Rich Tree."""
    tree = Tree(
        f"[bold]Run {trace.metadata.run_id}[/bold] "
        f"({trace.metadata.status.value})"
    )

    for dp in trace.decisions:
        lock_style = "green" if dp.locked else "yellow"
        lock_label = " [LOCKED]" if dp.locked else ""
        branch = tree.add(
            f"[{lock_style}]{dp.category.value}[/{lock_style}]: "
            f"{dp.question}{lock_label}"
        )
        branch.add(f"[bold]Chosen:[/bold] {dp.chosen}")
        branch.add(f"[dim]Confidence: {dp.confidence:.2f}[/dim]")
        if dp.alternatives:
            alts = branch.add("[dim]Alternatives:[/dim]")
            for alt in dp.alternatives:
                alts.add(f"{alt.value}: {alt.reasoning[:80]}")

    return tree


def _render_trace_table(trace) -> Table:
    """Render decisions as a Rich Table."""
    table = Table(title=f"Decisions for {trace.metadata.run_id}")
    table.add_column("#", style="dim", width=3)
    table.add_column("Category", style="cyan")
    table.add_column("Question")
    table.add_column("Chosen", style="bold")
    table.add_column("Confidence", justify="right")
    table.add_column("Locked", justify="center")

    for dp in trace.decisions:
        locked_str = "[green]Yes[/green]" if dp.locked else "[yellow]No[/yellow]"
        table.add_row(
            str(dp.sequence_number),
            dp.category.value,
            dp.question,
            dp.chosen,
            f"{dp.confidence:.2f}",
            locked_str,
        )

    return table


def _render_run_table(runs) -> Table:
    """Render a list of RunMetadata as a Rich Table."""
    table = Table(title="Recorded Runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Status")
    table.add_column("Decisions", justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_column("Models", style="dim")
    table.add_column("Started", style="dim")

    for meta in runs:
        status_style = {
            "success": "green",
            "failed": "red",
            "partial": "yellow",
            "pending": "dim",
            "running": "blue",
        }.get(meta.status.value, "white")

        # Count distinct models from model_routing
        models = set(meta.model_routing.values()) if meta.model_routing else set()
        model_count = f"{len(models)} model{'s' if len(models) != 1 else ''}"

        table.add_row(
            meta.run_id,
            f"[{status_style}]{meta.status.value}[/{status_style}]",
            str(meta.total_llm_calls),
            f"${meta.total_cost_usd:.4f}",
            model_count,
            meta.started_at.strftime("%Y-%m-%d %H:%M") if meta.started_at else "?",
        )

    return table


# ── CLI Group ────────────────────────────────────────────────────────────────


@click.group()
def cli():
    """The Deterministic Agent — capture, replay, and explore decision paths."""
    pass


# ── run ──────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("task", required=False)
@click.option("--task-file", "-f", type=click.Path(exists=True), help="Read task from file")
@click.option("--model", "-m", default=None, help="LLM model override")
@click.option("--max-cost", type=float, default=None, help="Max cost in USD")
@click.option("--max-iterations", type=int, default=None, help="Max agent iterations")
@click.option("--pin-low", is_flag=True, help="Auto-pin LOW/NOOP variance decisions")
@click.option("--no-confirm", is_flag=True, help="Skip cost confirmation")
@click.option("--verbose", "-v", is_flag=True, help="Stream decisions in real-time")
def run(task, task_file, model, max_cost, max_iterations, pin_low, no_confirm, verbose):
    """Run the agent on a task."""
    task_text = _load_task(task, Path(task_file) if task_file else None)
    settings = _get_settings()

    if max_cost is not None:
        settings.agent.max_cost_usd = max_cost
    if max_iterations is not None:
        settings.agent.max_iterations = max_iterations

    # Show cost estimate
    _show_cost_estimate(settings, model_override=model)

    if not no_confirm:
        if not click.confirm("\nProceed with this run?", default=True):
            console.print("[dim]Aborted.[/dim]")
            raise SystemExit(0)

    agent = _build_agent(settings, model_override=model)

    # Build event handler for verbose mode
    event_handler = None
    if verbose:
        from src.cli.event_handler import RichEventHandler
        event_handler = RichEventHandler()

    async def _run():
        return await agent.run(
            task=task_text,
            event_handler=event_handler,
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        disable=verbose,  # disable progress bar in verbose mode
    ) as progress:
        if not verbose:
            progress.add_task("Running agent...", total=None)
        trace = _run_async(_run())

    # Show summary
    console.print()
    console.print(_render_trace_table(trace))

    # Summary panel
    status_color = "green" if trace.metadata.status.value == "success" else "red"
    console.print(
        Panel(
            f"[bold]Run ID:[/bold] {trace.metadata.run_id}\n"
            f"[bold]Status:[/bold] [{status_color}]{trace.metadata.status.value}[/{status_color}]\n"
            f"[bold]Decisions:[/bold] {len(trace.decisions)}\n"
            f"[bold]Cost:[/bold] ${trace.metadata.total_cost_usd:.4f}\n"
            f"[bold]LLM Calls:[/bold] {trace.metadata.total_llm_calls}",
            title="Run Complete",
            border_style=status_color,
        )
    )

    console.print(f"\n[bold cyan]Run ID: {trace.metadata.run_id}[/bold cyan]")


# ── replay ───────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("run_id")
@click.option("--verify", is_flag=True, help="Verify output matches original")
@click.option("--verbose", "-v", is_flag=True, help="Stream decisions in real-time")
def replay(run_id, verify, verbose):
    """Replay a previous run with locked decisions."""
    settings = _get_settings()
    store = _build_trace_store(settings)

    async def _replay():
        from src.models.traces import PathLockConfig

        # 1. Load source trace
        source_trace = await store.get_trace(run_id)
        if not source_trace:
            console.print(f"[red]Run {run_id} not found[/red]")
            raise SystemExit(1)

        # 2. Show what will be locked
        console.print(f"[bold]Replaying run {run_id}[/bold]")
        console.print(f"Task: {source_trace.metadata.task_description[:100]}")
        console.print(f"Locking {len(source_trace.decisions)} decisions:\n")

        for dp in source_trace.decisions:
            console.print(
                f"  [green]LOCK[/green] [{dp.category.value}] "
                f"{dp.question} -> [bold]{dp.chosen}[/bold]"
            )

        console.print()

        # 3. Run agent with full lock
        lock_config = PathLockConfig(source_run_id=run_id)
        agent = _build_agent(settings)

        event_handler = None
        if verbose:
            from src.cli.event_handler import RichEventHandler
            event_handler = RichEventHandler()

        trace = await agent.run(
            task=source_trace.metadata.task_description,
            lock_config=lock_config,
            event_handler=event_handler,
        )

        # 4. If --verify, diff the decisions
        if verify:
            console.print("\n[bold]Verification Results:[/bold]")
            mismatches = 0
            for orig, locked in zip(
                source_trace.decisions, trace.decisions
            ):
                if orig.chosen == locked.chosen:
                    console.print(
                        f"  [green]✓[/green] {orig.question}: "
                        f"{orig.chosen} -> {locked.chosen}"
                    )
                else:
                    mismatches += 1
                    console.print(
                        f"  [red]✗[/red] {orig.question}: "
                        f"{orig.chosen} -> {locked.chosen}"
                    )

            if mismatches == 0:
                console.print("\n[green]All decisions matched![/green]")
            else:
                console.print(
                    f"\n[red]{mismatches} decision(s) diverged[/red]"
                )

        return trace

    trace = _run_async(_replay())

    console.print(
        f"\n[bold cyan]Replay Run ID: {trace.metadata.run_id}[/bold cyan]"
    )


# ── trace ────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("run_id")
@click.option(
    "--format", "fmt",
    type=click.Choice(["tree", "json", "table"]),
    default="tree",
)
@click.option("--decisions-only", is_flag=True)
def trace(run_id, fmt, decisions_only):
    """View the decision trace for a run."""
    settings = _get_settings()
    store = _build_trace_store(settings)

    async def _trace():
        return await store.get_trace(run_id)

    t = _run_async(_trace())
    if not t:
        console.print(f"[red]Run {run_id} not found[/red]")
        raise SystemExit(1)

    if fmt == "json":
        if decisions_only:
            data = [dp.model_dump(mode="json") for dp in t.decisions]
        else:
            data = t.model_dump(mode="json")
        console.print_json(json.dumps(data, indent=2, default=str))

    elif fmt == "table":
        console.print(_render_trace_table(t))

    else:  # tree
        console.print(_render_trace_tree(t))

    # Show cost summary
    if not decisions_only:
        console.print(
            f"\n[dim]Total cost: ${t.metadata.total_cost_usd:.4f} | "
            f"LLM calls: {t.metadata.total_llm_calls} | "
            f"Status: {t.metadata.status.value}[/dim]"
        )


# ── list ─────────────────────────────────────────────────────────────────────


@cli.command(name="list")
@click.option(
    "--status",
    type=click.Choice(["success", "failed", "partial"]),
)
@click.option("--limit", "-n", type=int, default=20)
def list_runs(status, limit):
    """List recorded runs."""
    from src.models.enums import RunStatus

    settings = _get_settings()
    store = _build_trace_store(settings)

    status_enum = RunStatus(status) if status else None

    async def _list():
        return await store.list_runs(status=status_enum, limit=limit)

    runs = _run_async(_list())

    if not runs:
        console.print("[dim]No runs found.[/dim]")
        return

    console.print(_render_run_table(runs))


# ── fork ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("run_id")
@click.option("--decision", "-d", required=True, help="Decision ID or sequence number to override")
@click.option("--choice", "-c", required=True, help="New choice value")
@click.option("--compare", is_flag=True, help="Auto-compare with parent after completion")
@click.option("--verbose", "-v", is_flag=True, help="Stream decisions in real-time")
def fork(run_id, decision, choice, compare, verbose):
    """Fork from a decision point with a different choice."""
    settings = _get_settings()
    store = _build_trace_store(settings)

    async def _fork():
        from src.models.traces import PathLockConfig

        source_trace = await store.get_trace(run_id)
        if not source_trace:
            console.print(f"[red]Run {run_id} not found[/red]")
            raise SystemExit(1)

        # Find the decision to override (by ID or sequence number)
        target_dp = None
        try:
            seq = int(decision)
            for dp in source_trace.decisions:
                if dp.sequence_number == seq:
                    target_dp = dp
                    break
        except ValueError:
            for dp in source_trace.decisions:
                if dp.id == decision:
                    target_dp = dp
                    break

        if not target_dp:
            console.print(f"[red]Decision '{decision}' not found in run {run_id}[/red]")
            console.print("Available decisions:")
            for dp in source_trace.decisions:
                console.print(f"  {dp.sequence_number}: {dp.id} - {dp.question} -> {dp.chosen}")
            raise SystemExit(1)

        console.print(f"[bold]Forking run {run_id}[/bold]")
        console.print(
            f"Overriding decision {target_dp.sequence_number}: "
            f"{target_dp.question}"
        )
        console.print(
            f"  Original: [red]{target_dp.chosen}[/red] -> "
            f"New: [green]{choice}[/green]"
        )
        console.print()

        # Build lock config with override
        lock_config = PathLockConfig(
            source_run_id=run_id,
            overrides={target_dp.id: choice},
        )

        agent = _build_agent(settings)

        event_handler = None
        if verbose:
            from src.cli.event_handler import RichEventHandler
            event_handler = RichEventHandler()

        trace = await agent.run(
            task=source_trace.metadata.task_description,
            lock_config=lock_config,
            event_handler=event_handler,
        )

        return source_trace, trace

    source_trace, forked_trace = _run_async(_fork())

    console.print(
        f"\n[bold cyan]Forked Run ID: {forked_trace.metadata.run_id}[/bold cyan]"
    )

    if compare:
        _show_comparison(source_trace, forked_trace)


# ── compare ──────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("run_id_1")
@click.argument("run_id_2")
@click.option(
    "--format", "fmt",
    type=click.Choice(["table", "json"]),
    default="table",
)
def compare(run_id_1, run_id_2, fmt):
    """Compare two runs side by side."""
    settings = _get_settings()
    store = _build_trace_store(settings)

    async def _compare():
        t1 = await store.get_trace(run_id_1)
        t2 = await store.get_trace(run_id_2)
        if not t1:
            console.print(f"[red]Run {run_id_1} not found[/red]")
            raise SystemExit(1)
        if not t2:
            console.print(f"[red]Run {run_id_2} not found[/red]")
            raise SystemExit(1)
        return t1, t2

    t1, t2 = _run_async(_compare())

    if fmt == "json":
        data = _build_comparison_data(t1, t2)
        console.print_json(json.dumps(data, indent=2, default=str))
    else:
        _show_comparison(t1, t2)


def _show_comparison(t1, t2):
    """Show a side-by-side comparison of two traces."""
    table = Table(title=f"Comparison: {t1.metadata.run_id} vs {t2.metadata.run_id}")
    table.add_column("Category", style="cyan")
    table.add_column("Question")
    table.add_column(f"{t1.metadata.run_id}", style="bold")
    table.add_column(f"{t2.metadata.run_id}", style="bold")
    table.add_column("Match", justify="center")

    # Match decisions by category + question
    d1_map = {(dp.category.value, dp.question): dp for dp in t1.decisions}
    d2_map = {(dp.category.value, dp.question): dp for dp in t2.decisions}

    all_keys = list(dict.fromkeys(list(d1_map.keys()) + list(d2_map.keys())))

    for key in all_keys:
        dp1 = d1_map.get(key)
        dp2 = d2_map.get(key)
        cat, question = key
        val1 = dp1.chosen if dp1 else "[dim]—[/dim]"
        val2 = dp2.chosen if dp2 else "[dim]—[/dim]"
        match = "[green]=[/green]" if dp1 and dp2 and dp1.chosen == dp2.chosen else "[red]≠[/red]"
        table.add_row(cat, question, val1, val2, match)

    console.print(table)

    # Cost comparison
    console.print(
        f"\n[dim]Cost: {t1.metadata.run_id}=${t1.metadata.total_cost_usd:.4f} | "
        f"{t2.metadata.run_id}=${t2.metadata.total_cost_usd:.4f}[/dim]"
    )


def _build_comparison_data(t1, t2) -> dict:
    """Build a JSON-serializable comparison between two traces."""
    d1_map = {(dp.category.value, dp.question): dp for dp in t1.decisions}
    d2_map = {(dp.category.value, dp.question): dp for dp in t2.decisions}
    all_keys = list(dict.fromkeys(list(d1_map.keys()) + list(d2_map.keys())))

    comparisons = []
    for key in all_keys:
        dp1 = d1_map.get(key)
        dp2 = d2_map.get(key)
        cat, question = key
        comparisons.append({
            "category": cat,
            "question": question,
            "run_1": dp1.chosen if dp1 else None,
            "run_2": dp2.chosen if dp2 else None,
            "match": dp1 is not None and dp2 is not None and dp1.chosen == dp2.chosen,
        })

    return {
        "run_1": t1.metadata.run_id,
        "run_2": t2.metadata.run_id,
        "decisions": comparisons,
        "cost_1": t1.metadata.total_cost_usd,
        "cost_2": t2.metadata.total_cost_usd,
    }


# ── estimate ─────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("task", required=False)
@click.option("--task-file", "-f", type=click.Path(exists=True))
@click.option("--single-model", "-m", default=None, help="Override all phases to use one model")
@click.option("--compare-strategies", is_flag=True, help="Compare multi-model vs single-model cost")
def estimate(task, task_file, single_model, compare_strategies):
    """Estimate cost without running. Shows per-phase model routing and breakdown."""
    settings = _get_settings()

    # Multi-model estimate (default)
    _show_cost_estimate(settings)

    if compare_strategies:
        # Also show single-model estimate for comparison
        console.print()
        compare_model = single_model or "anthropic/claude-sonnet-4"
        console.print(f"[bold]Single-model comparison ({compare_model}):[/bold]")
        _show_cost_estimate(settings, model_override=compare_model)

        console.print(
            "\n[dim]Multi-model routing typically saves 60-70% on costs "
            "by using cheaper models for planning and evaluation.[/dim]"
        )
    elif single_model:
        console.print()
        _show_cost_estimate(settings, model_override=single_model)
