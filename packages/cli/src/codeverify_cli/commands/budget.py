"""CodeVerify CLI - Budget command."""

from __future__ import annotations

from pathlib import Path

import click
from rich.panel import Panel
from rich.table import Table

from codeverify_cli.commands import console


@click.command("budget")
@click.argument("action", type=click.Choice(["estimate", "report"]))
@click.argument("files", nargs=-1)
@click.option("--tier", type=click.Choice(["free", "standard", "premium"]), default="standard")
@click.option("--max-cost", type=float, default=5.0, help="Max cost per PR ($)")
@click.pass_context
def budget(ctx: click.Context, action: str, files: tuple, tier: str, max_cost: float) -> None:
    """Manage verification budget and costs.

    Examples:

        codeverify budget estimate src/*.py
        codeverify budget estimate src/ --tier premium
        codeverify budget report
    """
    from codeverify_core.budget_optimizer import Budget, RiskFactors, VerificationBudgetOptimizer

    console.print(
        Panel.fit(
            "[bold blue]CodeVerify[/bold blue] - Budget Optimizer", subtitle="Cost Management"
        )
    )

    optimizer = VerificationBudgetOptimizer()
    budget_obj = Budget(tier=tier, max_cost_per_pr=max_cost)

    if action == "estimate":
        if not files:
            console.print("[red]Please provide files to estimate[/red]")
            return

        file_list = []
        for pattern in files:
            p = Path(pattern)
            if p.is_file():
                file_list.append(p)
            elif p.is_dir():
                file_list.extend(p.rglob("*.py"))
                file_list.extend(p.rglob("*.ts"))
                file_list.extend(p.rglob("*.js"))

        console.print(f"Estimating for [bold]{len(file_list)}[/bold] files (tier: {tier})")

        file_infos = []
        for f in file_list[:20]:  # Limit for demo
            lines = len(f.read_text().splitlines())
            file_infos.append(
                {
                    "file_path": str(f),
                    "size_lines": lines,
                    "factors": RiskFactors(file_complexity=min(lines / 500, 1.0)),
                }
            )

        result = optimizer.optimize_batch(file_infos, budget_obj)

        table = Table(title="Verification Plan")
        table.add_column("File", style="cyan")
        table.add_column("Depth")
        table.add_column("Risk")
        table.add_column("Cost")

        for d in result.decisions:
            risk_color = (
                "red" if d.risk_score > 0.7 else "yellow" if d.risk_score > 0.3 else "green"
            )
            table.add_row(
                Path(d.file_path).name,
                d.depth.value,
                f"[{risk_color}]{d.risk_score:.2f}[/{risk_color}]",
                f"${d.estimated_cost:.3f}",
            )

        console.print(table)
        console.print(f"\n[bold]Total estimated cost:[/bold] ${result.total_estimated_cost:.2f}")
        console.print(f"[bold]Budget utilization:[/bold] {result.budget_utilization:.1%}")

    else:  # report
        report = optimizer.get_usage_report()
        console.print(f"Total cost: ${report['total_cost']:.2f}")
        console.print(f"Total files: {report['total_files']}")
