"""CodeVerify CLI - Telemetry commands."""

from __future__ import annotations

import click
from rich.panel import Panel
from rich.table import Table

from codeverify_cli.commands import console


@click.group()
def telemetry():
    """View verification telemetry and ROI reports."""
    pass


@telemetry.command("report")
@click.option("--days", type=int, default=30, help="Report period in days")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["rich", "json", "markdown"]),
    default="rich",
)
def telemetry_report(days: int, output_format: str) -> None:
    """Generate an ROI report for verification activities."""
    import json as json_mod

    from codeverify_core.telemetry import CostEstimator, ROIDashboard, TelemetryCollector

    collector = TelemetryCollector()
    estimator = CostEstimator()
    dashboard = ROIDashboard(collector=collector, cost_estimator=estimator)
    report = dashboard.generate_report(period_days=days)

    if output_format == "json":
        click.echo(
            json_mod.dumps(
                {
                    "period_days": days,
                    "total_findings": report.total_findings,
                    "bugs_prevented": report.bugs_prevented,
                    "estimated_cost_saved": report.estimated_cost_saved,
                    "fix_rate": report.fix_rate,
                    "roi_multiplier": report.roi_multiplier,
                },
                indent=2,
            )
        )
        return

    if output_format == "markdown":
        summary = dashboard.generate_executive_summary(report)
        click.echo(summary)
        return

    console.print(Panel.fit(f"[bold blue]ROI Report[/bold blue] - Last {days} Days"))

    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")
    table.add_row("Total Findings", str(report.total_findings))
    table.add_row("Bugs Prevented", str(report.bugs_prevented))
    table.add_row("Cost Saved", f"${report.estimated_cost_saved:,.2f}")
    table.add_row("Fix Rate", f"{report.fix_rate:.1%}")
    table.add_row("Dev Hours Saved", f"{report.developer_hours_saved:.0f}")
    table.add_row("ROI Multiplier", f"{report.roi_multiplier:.1f}x")
    console.print(table)
