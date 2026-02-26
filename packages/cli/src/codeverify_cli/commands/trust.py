"""CodeVerify CLI - Trust score command."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from codeverify_cli.commands import console


@click.command("trust-score")
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
def trust_score(ctx: click.Context, path: str, output_format: str) -> None:
    """Calculate trust score for code.

    Analyzes code to determine its trustworthiness, particularly for
    AI-generated code. Shows risk level and recommendations.

    Examples:

        codeverify trust-score src/module.py
        codeverify trust-score . -f json
    """
    from codeverify_agents import TrustScoreAgent

    verbose = ctx.obj.get("verbose", False)

    console.print(
        Panel.fit(
            "[bold blue]CodeVerify[/bold blue] - Trust Score Analysis",
            subtitle="AI Detection & Risk Assessment",
        )
    )

    path_obj = Path(path)

    if path_obj.is_file():
        files = [path_obj]
    else:
        files = (
            list(path_obj.rglob("*.py"))
            + list(path_obj.rglob("*.ts"))
            + list(path_obj.rglob("*.js"))
        )
        files = files[:20]  # Limit for CLI

    if not files:
        console.print("[yellow]No supported files found[/yellow]")
        return

    agent = TrustScoreAgent()
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Analyzing trust scores...", total=len(files))

        for file in files:
            try:
                code = file.read_text()
                result = asyncio.run(agent.analyze(code, {"file_path": str(file)}))
                results.append(
                    {
                        "file": str(file),
                        "score": result.score,
                        "risk_level": result.risk_level,
                        "ai_probability": result.ai_probability,
                        "recommendations": result.recommendations[:3],
                    }
                )
                progress.advance(task)
            except Exception as e:
                if verbose:
                    console.print(f"[dim]Skipped {file}: {e}[/dim]")

    if output_format == "json":
        import json

        click.echo(json.dumps(results, indent=2))
        return

    # Rich output
    table = Table(title="Trust Score Results")
    table.add_column("File", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Risk", justify="center")
    table.add_column("AI Prob", justify="right")

    for r in results:
        score = r["score"]
        risk = r["risk_level"]

        # Color code
        if score >= 80:
            score_str = f"[green]{score:.0f}[/green]"
        elif score >= 60:
            score_str = f"[yellow]{score:.0f}[/yellow]"
        else:
            score_str = f"[red]{score:.0f}[/red]"

        risk_colors = {"low": "green", "medium": "yellow", "high": "red", "critical": "red bold"}
        risk_str = f"[{risk_colors.get(risk, 'white')}]{risk}[/{risk_colors.get(risk, 'white')}]"

        table.add_row(str(Path(r["file"]).name), score_str, risk_str, f"{r['ai_probability']:.0f}%")

    console.print(table)

    # Summary
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0
    console.print(f"\n[bold]Average Trust Score:[/bold] {avg_score:.1f}/100")
