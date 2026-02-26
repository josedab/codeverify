"""CodeVerify CLI - Team report command."""

from __future__ import annotations

from pathlib import Path

import click
from rich.panel import Panel

from codeverify_cli.commands import console


@click.command("team-report")
@click.option("--output", "-o", type=click.Path(), help="Output file (markdown)")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["rich", "markdown", "json"]),
    default="rich",
    help="Output format",
)
@click.pass_context
def team_report(ctx: click.Context, output: str | None, output_format: str) -> None:
    """Generate team learning report.

    Examples:

        codeverify team-report
        codeverify team-report -o report.md -f markdown
    """
    from codeverify_agents.team_learning import TeamLearningAgent

    console.print(
        Panel.fit(
            "[bold blue]CodeVerify[/bold blue] - Team Learning", subtitle="Organization Insights"
        )
    )

    agent = TeamLearningAgent()
    report = agent.generate_org_health_report()

    if output_format == "markdown" or output:
        md = agent.export_report_markdown(report)
        if output:
            Path(output).write_text(md)
            console.print(f"[green]Report saved to {output}[/green]")
        else:
            console.print(md)
    elif output_format == "json":
        import json

        data = {
            "total_findings": report.total_findings,
            "total_prs": report.total_prs_analyzed,
            "trend": report.trend_vs_last_period.value
            if hasattr(report.trend_vs_last_period, "value")
            else str(report.trend_vs_last_period),
            "patterns": len(report.systemic_patterns),
            "recommendations": len(report.training_recommendations),
        }
        click.echo(json.dumps(data, indent=2))
    else:
        console.print(f"Total findings: [bold]{report.total_findings}[/bold]")
        console.print(f"PRs analyzed: {report.total_prs_analyzed}")
        console.print(f"Systemic patterns: {len(report.systemic_patterns)}")
        console.print(f"Training recommendations: {len(report.training_recommendations)}")

        if report.teams_needing_attention:
            console.print("\n[yellow]Teams needing attention:[/yellow]")
            for team in report.teams_needing_attention:
                console.print(f"  • {team}")
