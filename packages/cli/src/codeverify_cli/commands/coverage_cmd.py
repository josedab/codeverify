"""CodeVerify CLI - Coverage commands."""

from __future__ import annotations

from pathlib import Path

import click
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from codeverify_cli.commands import console


@click.group()
def coverage():
    """Proof coverage dashboard.

    Track verification coverage across your codebase.
    """
    pass


@coverage.command("show")
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
def coverage_show(ctx: click.Context, path: str, output_format: str) -> None:
    """Show proof coverage for code.

    Displays verification coverage metrics including line, function,
    and file coverage.

    Examples:

        codeverify coverage show src/
        codeverify coverage show . -f json
    """
    from codeverify_core import ProofCoverageCalculator

    verbose = ctx.obj.get("verbose", False)

    console.print(
        Panel.fit(
            "[bold blue]CodeVerify[/bold blue] - Proof Coverage", subtitle="Verification Metrics"
        )
    )

    path_obj = Path(path)
    calculator = ProofCoverageCalculator()

    if path_obj.is_file():
        files = [path_obj]
    else:
        files = list(path_obj.rglob("*.py"))[:50]

    if not files:
        console.print("[yellow]No Python files found[/yellow]")
        return

    # Calculate coverage for each file
    file_coverages = []
    total_lines = 0
    verified_lines = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Calculating coverage...", total=len(files))

        for file in files:
            try:
                content = file.read_text()
                # Get any existing verifications (would come from database in production)
                verifications = []  # Placeholder - would load from storage

                file_coverage = calculator.calculate_file_coverage(
                    str(file),
                    content,
                    verifications,
                )
                file_coverages.append(file_coverage)
                total_lines += file_coverage.total_lines
                verified_lines += file_coverage.verified_lines

                progress.advance(task)
            except Exception as e:
                if verbose:
                    console.print(f"[dim]Error in {file}: {e}[/dim]")

    # Calculate overall percentage
    overall_pct = (verified_lines / total_lines * 100) if total_lines > 0 else 0

    if output_format == "json":
        import json

        click.echo(
            json.dumps(
                {
                    "total_files": len(file_coverages),
                    "total_lines": total_lines,
                    "verified_lines": verified_lines,
                    "overall_percentage": overall_pct,
                    "files": [
                        {
                            "path": fc.file_path,
                            "total_lines": fc.total_lines,
                            "verified_lines": fc.verified_lines,
                            "status": fc.status.value,
                        }
                        for fc in file_coverages
                    ],
                },
                indent=2,
            )
        )
        return

    # Rich output
    console.print(f"\n[bold]Overall Coverage: {overall_pct:.1f}%[/bold]")
    console.print(_coverage_bar(overall_pct))
    console.print(f"\nFiles: {len(file_coverages)} | Lines: {verified_lines}/{total_lines}\n")

    # Show file breakdown
    table = Table(title="File Coverage")
    table.add_column("File", style="cyan")
    table.add_column("Lines", justify="right")
    table.add_column("Verified", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Status")

    for fc in sorted(
        file_coverages, key=lambda x: x.verified_lines / max(x.total_lines, 1), reverse=True
    )[:15]:
        pct = (fc.verified_lines / fc.total_lines * 100) if fc.total_lines > 0 else 0

        if pct >= 80:
            pct_str = f"[green]{pct:.0f}%[/green]"
            status = "[green]✓[/green]"
        elif pct >= 50:
            pct_str = f"[yellow]{pct:.0f}%[/yellow]"
            status = "[yellow]~[/yellow]"
        else:
            pct_str = f"[red]{pct:.0f}%[/red]"
            status = "[red]✗[/red]"

        # Truncate long paths
        file_path = fc.file_path
        if len(file_path) > 40:
            file_path = "..." + file_path[-37:]

        table.add_row(
            file_path,
            str(fc.total_lines),
            str(fc.verified_lines),
            pct_str,
            status,
        )

    console.print(table)


@coverage.command("report")
@click.argument("repository", default=".")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["html", "json", "markdown"]),
    default="html",
    help="Report format",
)
@click.pass_context
def coverage_report(ctx: click.Context, repository: str, output: str, output_format: str) -> None:
    """Generate proof coverage report.

    Creates a detailed coverage report with trends and heatmaps.

    Examples:

        codeverify coverage report --output coverage.html
        codeverify coverage report -f json -o coverage.json
    """
    from codeverify_core import get_proof_coverage_dashboard

    console.print(
        Panel.fit(
            "[bold blue]CodeVerify[/bold blue] - Coverage Report",
        )
    )

    dashboard = get_proof_coverage_dashboard()

    # Generate report
    report = dashboard.export_report(repository, format=output_format)

    if output:
        output_path = Path(output)
        if isinstance(report, dict):
            import json

            output_path.write_text(json.dumps(report, indent=2, default=str))
        else:
            output_path.write_text(report)
        console.print(f"[green]✓ Report saved to {output}[/green]")
    else:
        if output_format == "json":
            import json

            click.echo(json.dumps(report, indent=2, default=str))
        else:
            click.echo(report)


@coverage.command("trends")
@click.argument("repository", default=".")
@click.option("--days", type=int, default=30, help="Number of days to show")
@click.pass_context
def coverage_trends(ctx: click.Context, repository: str, days: int) -> None:
    """Show coverage trends over time.

    Examples:

        codeverify coverage trends --days 90
    """
    from codeverify_core import get_proof_coverage_dashboard

    console.print(
        Panel.fit(
            "[bold blue]CodeVerify[/bold blue] - Coverage Trends",
        )
    )

    dashboard = get_proof_coverage_dashboard()
    trends = dashboard.get_trends(repository, days=days)

    if not trends:
        console.print("[yellow]No trend data available yet[/yellow]")
        console.print("[dim]Trends are calculated from verification history[/dim]")
        return

    console.print(f"\n[bold]Coverage over last {days} days:[/bold]\n")

    # Simple ASCII chart
    for trend in trends[-10:]:  # Show last 10 data points
        pct = trend.coverage_percentage
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        date_str = trend.date.strftime("%Y-%m-%d")
        console.print(f"{date_str} [{_pct_color(pct)}]{bar}[/] {pct:.1f}%")


def _coverage_bar(pct: float, width: int = 30) -> str:
    """Generate a coverage progress bar."""
    filled = int(pct / 100 * width)
    empty = width - filled
    color = _pct_color(pct)
    return f"[{color}]{'█' * filled}{'░' * empty}[/] {pct:.1f}%"


def _pct_color(pct: float) -> str:
    """Get color for percentage."""
    if pct >= 80:
        return "green"
    elif pct >= 50:
        return "yellow"
    return "red"
