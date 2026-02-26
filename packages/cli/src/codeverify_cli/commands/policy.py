"""CodeVerify CLI - Policy commands."""

from __future__ import annotations

from pathlib import Path

import click
from rich.panel import Panel
from rich.table import Table

from codeverify_cli.commands import console


@click.group()
def policy():
    """Manage verification policies."""
    pass


@policy.command("list-built-in")
def policy_list_built_in() -> None:
    """List all built-in policy templates."""
    from codeverify_core.policy_engine import BUILT_IN_POLICIES

    table = Table(title="Built-in Policies")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Action")
    table.add_column("Depth")
    table.add_column("Priority", justify="right")

    for p in BUILT_IN_POLICIES:
        table.add_row(p.id, p.name, p.action.value, p.verification_depth or "-", str(p.priority))
    console.print(table)


@policy.command("evaluate")
@click.argument("policy_file", type=click.Path(exists=True))
@click.argument("file_path")
@click.option(
    "--format", "-f", "output_format", type=click.Choice(["rich", "json"]), default="rich"
)
def policy_evaluate(policy_file: str, file_path: str, output_format: str) -> None:
    """Evaluate a policy against a file."""
    import json as json_mod

    from codeverify_core.policy_engine import PolicyEngine

    policy_content = Path(policy_file).read_text()
    engine = PolicyEngine()
    policy_set = engine.load_from_yaml(policy_content)

    context = {"file_path": file_path}
    results = engine.evaluate(policy_set, context)
    matched = [r for r in results if r.matched]

    if output_format == "json":
        click.echo(
            json_mod.dumps(
                [
                    {
                        "rule_id": r.rule_id,
                        "action": r.action.value,
                        "matched": r.matched,
                        "reason": r.reason,
                    }
                    for r in results
                ],
                indent=2,
            )
        )
        return

    console.print(Panel.fit(f"[bold blue]Policy Evaluation[/bold blue] - {file_path}"))
    for r in results:
        icon = "[green]MATCH[/green]" if r.matched else "[dim]no match[/dim]"
        console.print(f"  {icon} {r.rule_name} -> {r.action.value}")
