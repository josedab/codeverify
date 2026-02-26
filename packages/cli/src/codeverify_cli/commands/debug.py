"""CodeVerify CLI - Debug command."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from codeverify_cli.commands import console


@click.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--function", "-fn", help="Specific function to debug")
@click.option("--interactive", "-i", is_flag=True, help="Interactive step-through mode")
@click.pass_context
def debug(ctx: click.Context, file: str, function: str | None, interactive: bool) -> None:
    """Debug verification for a file.

    Shows step-by-step verification trace to understand
    how formal verification analyzes your code.

    Examples:

        codeverify debug src/math.py
        codeverify debug src/math.py --function calculate
        codeverify debug src/math.py -i
    """
    from codeverify_verifier import VerificationDebugger

    verbose = ctx.obj.get("verbose", False)

    console.print(
        Panel.fit(
            "[bold blue]CodeVerify[/bold blue] - Verification Debugger",
            subtitle="Step-by-Step Analysis",
        )
    )

    file_path = Path(file)
    code = file_path.read_text()

    debugger = VerificationDebugger()

    console.print(f"Analyzing: [cyan]{file_path}[/cyan]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Running verification...", total=None)
        result = asyncio.run(debugger.trace(code))

    # Display steps
    steps = result.get("steps", [])

    if not steps:
        console.print("[yellow]No verification steps generated[/yellow]")
        console.print(f"Result: {result.get('result', 'unknown')}")
        return

    console.print(f"[bold]Verification Result:[/bold] {result.get('result', 'unknown')}\n")

    for i, step in enumerate(steps, 1):
        status = step.get("status", "pending")
        status_icon = {
            "passed": "[green]✓[/green]",
            "failed": "[red]✗[/red]",
            "pending": "[yellow]○[/yellow]",
            "skipped": "[dim]○[/dim]",
        }.get(status, "○")

        console.print(f"{status_icon} [bold]Step {i}:[/bold] {step.get('title', 'Unknown')}")

        if step.get("description"):
            console.print(f"   {step['description']}")

        if step.get("constraint") and verbose:
            console.print(f"   [dim]Constraint: {step['constraint']}[/dim]")

        if step.get("model") and status == "failed":
            console.print(f"   [red]Counterexample: {step['model']}[/red]")

        if interactive:
            if not click.confirm("Continue?", default=True):
                break

    # Show counterexample if verification failed
    if result.get("result") == "unverified" and result.get("counterexample"):
        console.print("\n[bold red]Counterexample Found:[/bold red]")
        for var, val in result["counterexample"].items():
            console.print(f"  {var} = {val}")
