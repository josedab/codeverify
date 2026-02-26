"""CodeVerify CLI - Monorepo commands."""

from __future__ import annotations

from pathlib import Path

import click
from rich.panel import Panel
from rich.table import Table

from codeverify_cli.commands import console


@click.group()
def monorepo() -> None:
    """Monorepo intelligence commands.

    Analyze monorepo structure, dependencies, and affected packages.
    """
    pass


@monorepo.command("analyze")
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["rich", "json", "dot"]),
    default="rich",
    help="Output format",
)
@click.pass_context
def monorepo_analyze(ctx: click.Context, path: str, output_format: str) -> None:
    """Analyze monorepo structure and dependencies.

    Examples:

        codeverify monorepo analyze              # Current directory
        codeverify monorepo analyze ./my-repo   # Specific path
        codeverify monorepo analyze -f json     # JSON output
    """
    from codeverify_core.monorepo import MonorepoAnalyzer, MonorepoType

    console.print(
        Panel.fit(
            "[bold blue]CodeVerify[/bold blue] - Monorepo Analysis",
            subtitle="Dependency Intelligence",
        )
    )

    path_obj = Path(path)
    analyzer = MonorepoAnalyzer(path_obj)

    if analyzer.monorepo_type == MonorepoType.NONE:
        console.print("[yellow]No monorepo detected.[/yellow]")
        console.print("Supported: Nx, Turborepo, Lerna, pnpm, Yarn workspaces")
        return

    console.print(f"[green]Detected:[/green] {analyzer.monorepo_type.value}")

    packages = analyzer.discover_packages()
    graph = analyzer.build_dependency_graph()
    cycles = graph.detect_cycles()

    if output_format == "json":
        import json

        data = {
            "type": analyzer.monorepo_type.value,
            "packages": [
                {"name": p.name, "path": str(p.path), "version": p.version} for p in packages
            ],
            "edges": graph.edges,
            "cycles": cycles,
        }
        click.echo(json.dumps(data, indent=2))
    else:
        table = Table(title="Packages")
        table.add_column("Name", style="cyan")
        table.add_column("Version")
        table.add_column("Path")
        table.add_column("Dependencies")

        for pkg in packages:
            deps = len(graph.edges.get(pkg.name, []))
            table.add_row(
                pkg.name, pkg.version or "-", str(pkg.path.relative_to(path_obj)), str(deps)
            )

        console.print(table)

        if cycles:
            console.print(f"\n[red]⚠ Circular dependencies detected: {len(cycles)}[/red]")
            for cycle in cycles[:3]:
                console.print(f"  [dim]→ {' → '.join(cycle)}[/dim]")
        else:
            console.print("\n[green]✓ No circular dependencies[/green]")


@monorepo.command("affected")
@click.argument("files", nargs=-1, required=True)
@click.option("--path", "-p", type=click.Path(exists=True), default=".", help="Monorepo root")
@click.pass_context
def monorepo_affected(ctx: click.Context, files: tuple, path: str) -> None:
    """Get packages affected by changed files.

    Examples:

        codeverify monorepo affected packages/core/src/index.ts
        codeverify monorepo affected $(git diff --name-only HEAD~1)
    """
    from codeverify_core.monorepo import MonorepoAnalyzer

    analyzer = MonorepoAnalyzer(Path(path))
    affected = analyzer.get_affected_packages(list(files))

    if affected:
        console.print("[bold]Affected packages:[/bold]")
        for pkg in affected:
            console.print(f"  • {pkg}")
    else:
        console.print("[dim]No packages affected[/dim]")
