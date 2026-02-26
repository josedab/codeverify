"""CodeVerify CLI - Offline mode commands."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from codeverify_cli.commands import console


@click.group()
def offline():
    """Offline/air-gapped mode commands.

    Run analysis without cloud connectivity using local models.
    """
    pass


@offline.command("status")
@click.pass_context
def offline_status(ctx: click.Context) -> None:
    """Check offline mode readiness.

    Shows availability of local components (Z3, Ollama, cached models).

    Examples:

        codeverify offline status
    """
    from codeverify_core import get_offline_manager

    console.print(
        Panel.fit(
            "[bold blue]CodeVerify[/bold blue] - Offline Mode Status",
        )
    )

    manager = get_offline_manager()
    status = asyncio.run(manager.check_offline_readiness())

    # Show readiness
    ready_icon = "✓" if status.get("ready") else "✗"
    ready_color = "green" if status.get("ready") else "red"
    console.print(
        f"[{ready_color}]{ready_icon} Overall Ready: {status.get('ready')}[/{ready_color}]"
    )

    console.print("\n[bold]Components:[/bold]")

    # Z3
    z3_icon = "✓" if status.get("z3_available") else "✗"
    z3_color = "green" if status.get("z3_available") else "yellow"
    console.print(f"  [{z3_color}]{z3_icon} Z3 Solver[/{z3_color}]")

    # Ollama
    ollama_icon = "✓" if status.get("ollama_available") else "✗"
    ollama_color = "green" if status.get("ollama_available") else "yellow"
    console.print(f"  [{ollama_color}]{ollama_icon} Ollama LLM[/{ollama_color}]")

    # Models
    models = status.get("models_available", [])
    if models:
        console.print("\n[bold]Available Models:[/bold]")
        for model in models:
            console.print(f"  • {model}")
    else:
        console.print("\n[yellow]No local models cached[/yellow]")
        console.print("[dim]Run 'codeverify offline setup' to download models[/dim]")


@offline.command("setup")
@click.option("--model", default="codellama:7b-instruct", help="Model to download")
@click.option("--force", is_flag=True, help="Force re-download")
@click.pass_context
def offline_setup(ctx: click.Context, model: str, force: bool) -> None:
    """Setup offline mode (download models).

    Downloads required models for offline analysis.

    Examples:

        codeverify offline setup
        codeverify offline setup --model llama3.2:1b
    """
    from codeverify_core import get_offline_manager

    console.print(
        Panel.fit(
            "[bold blue]CodeVerify[/bold blue] - Offline Mode Setup",
        )
    )

    manager = get_offline_manager()

    console.print(f"[bold]Downloading model: {model}[/bold]")
    console.print("[dim]This may take several minutes...[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Downloading {model}...", total=None)

        try:
            result = asyncio.run(manager.download_model(model, force=force))

            if result.get("success"):
                progress.update(task, description=f"✓ Downloaded {model}")
                console.print(f"\n[green]✓ Model {model} ready for offline use[/green]")
            else:
                console.print(f"\n[red]✗ Failed to download: {result.get('error')}[/red]")

        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            console.print("[dim]Make sure Ollama is installed and running[/dim]")


@offline.command("analyze")
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["rich", "json"]),
    default="rich",
    help="Output format",
)
@click.pass_context
def offline_analyze(ctx: click.Context, path: str, output_format: str) -> None:
    """Run offline analysis (no cloud required).

    Uses local Z3 and Ollama for analysis.

    Examples:

        codeverify offline analyze src/
        codeverify offline analyze file.py -f json
    """
    from codeverify_core import get_offline_manager

    verbose = ctx.obj.get("verbose", False)

    console.print(
        Panel.fit(
            "[bold blue]CodeVerify[/bold blue] - Offline Analysis", subtitle="Air-Gapped Mode"
        )
    )

    manager = get_offline_manager()

    # Check readiness
    status = asyncio.run(manager.check_offline_readiness())
    if not status.get("z3_available"):
        console.print("[yellow]⚠ Z3 not available - limited analysis[/yellow]")

    path_obj = Path(path)

    if path_obj.is_file():
        files = [path_obj]
    else:
        files = list(path_obj.rglob("*.py"))[:20]

    if not files:
        console.print("[yellow]No Python files found[/yellow]")
        return

    all_findings = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Analyzing...", total=len(files))

        for file in files:
            try:
                code = file.read_text()
                result = asyncio.run(
                    manager.analyze_code_offline(
                        code,
                        language="python",
                        include_llm_analysis=status.get("ollama_available", False),
                    )
                )

                for finding in result.findings:
                    finding["file"] = str(file)
                    all_findings.append(finding)

                progress.advance(task)
            except Exception as e:
                if verbose:
                    console.print(f"[dim]Error in {file}: {e}[/dim]")

    if output_format == "json":
        import json

        click.echo(
            json.dumps(
                {
                    "offline_mode": True,
                    "findings": all_findings,
                    "capabilities_used": result.capabilities_used if "result" in dir() else [],
                },
                indent=2,
                default=str,
            )
        )
        return

    console.print("\n[bold]Offline Analysis Complete[/bold]")
    console.print(
        f"[dim]Capabilities: Z3={status.get('z3_available')}, LLM={status.get('ollama_available')}[/dim]\n"
    )

    if not all_findings:
        console.print("[green]✓ No issues found[/green]")
        return

    console.print(f"Found [bold]{len(all_findings)}[/bold] issues:\n")

    for finding in all_findings[:20]:  # Limit output
        severity = finding.get("severity", "medium")
        color = {"critical": "red bold", "high": "red", "medium": "yellow", "low": "blue"}.get(
            severity, "white"
        )
        console.print(
            f"[{color}]• {finding.get('category', 'unknown')}[/{color}] - {finding.get('file')}:{finding.get('line', '?')}"
        )
        console.print(f"  {finding.get('message', '')}")
