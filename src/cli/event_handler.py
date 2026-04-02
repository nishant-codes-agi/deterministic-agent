"""Rich-based event handler for verbose CLI output."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from src.agent.base import AgentEvent, AgentEventType
from src.models.enums import AgentPhase

console = Console()

# Phase colors
_PHASE_COLORS = {
    AgentPhase.PLANNING: "blue",
    AgentPhase.CODING: "cyan",
    AgentPhase.EXECUTING: "yellow",
    AgentPhase.EVALUATING: "magenta",
    AgentPhase.RECOVERING: "red",
}


class RichEventHandler:
    """Handles agent events and prints them to the console with Rich formatting."""

    async def handle(self, event: AgentEvent) -> None:
        """Handle an agent event by printing it with Rich formatting."""
        match event.event_type:
            case AgentEventType.PHASE_CHANGED:
                phase = event.phase
                color = _PHASE_COLORS.get(phase, "white")
                console.print(f"\n[bold {color}]>>> {phase.value.upper()}[/bold {color}]")

            case AgentEventType.DECISION_MADE:
                p = event.payload
                locked = "[green][LOCKED][/green] " if p.get("locked") else ""
                console.print(
                    f"  {locked}[dim]Decision:[/dim] {p.get('question', '?')} "
                    f"-> [bold]{p.get('chosen', '?')}[/bold]"
                )

            case AgentEventType.CODE_WRITTEN:
                files = event.payload.get("files", [])
                iteration = event.payload.get("iteration", 0)
                console.print(
                    f"  [dim]Code written (iteration {iteration}):[/dim] "
                    f"{', '.join(files)}"
                )

            case AgentEventType.EXECUTION_STARTED:
                console.print("  [dim]Executing code...[/dim]")

            case AgentEventType.EXECUTION_COMPLETED:
                p = event.payload
                exit_code = p.get("exit_code", -1)
                duration = p.get("duration_ms", 0)
                if exit_code == 0:
                    console.print(
                        f"  [green]Execution passed[/green] "
                        f"({duration}ms)"
                    )
                else:
                    console.print(
                        f"  [red]Execution failed[/red] "
                        f"(exit code {exit_code}, {duration}ms)"
                    )
                    if p.get("timed_out"):
                        console.print("  [red]Timed out![/red]")

            case AgentEventType.ERROR_HIT:
                console.print(
                    f"  [red]Error:[/red] {event.payload.get('error', 'unknown')}"
                )

            case AgentEventType.RECOVERY_STARTED:
                console.print("  [yellow]Starting recovery...[/yellow]")

            case AgentEventType.COST_UPDATE:
                p = event.payload
                model = p.get("model", "?")
                cost = p.get("cost_usd", 0)
                total = p.get("total_cost_usd", 0)
                console.print(
                    f"  [dim]Cost: ${cost:.4f} ({model}) | "
                    f"Total: ${total:.4f}[/dim]"
                )
