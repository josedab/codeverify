"""CodeVerify CLI - Impact analysis commands."""

from __future__ import annotations

import click
from rich.panel import Panel

from codeverify_cli.commands import console


@click.group()
def impact():
    """Cross-repository impact analysis."""
    pass


@impact.command("analyze")
@click.argument("repo_name")
@click.argument("changed_files", nargs=-1)
@click.option(
    "--format", "-f", "output_format", type=click.Choice(["rich", "json"]), default="rich"
)
def impact_analyze(repo_name: str, changed_files: tuple[str, ...], output_format: str) -> None:
    """Analyze cross-repository impact of changes.

    Example:
        codeverify impact analyze my-lib src/api.py src/models.py
    """
    import json as json_mod

    from codeverify_core.impact_analysis import CrossRepoImpactAnalyzer

    analyzer = CrossRepoImpactAnalyzer()

    if output_format == "json":
        click.echo(
            json_mod.dumps(
                {
                    "repo": repo_name,
                    "changed_files": list(changed_files),
                    "blast_radius": 0,
                    "impacted_repos": [],
                    "note": "Register repositories first with 'impact register'",
                },
                indent=2,
            )
        )
        return

    console.print(Panel.fit(f"[bold blue]Impact Analysis[/bold blue] - {repo_name}"))
    console.print(f"Changed files: {len(changed_files)}")
    console.print(
        "[dim]Register repositories with 'codeverify impact register' to track dependencies[/dim]"
    )
