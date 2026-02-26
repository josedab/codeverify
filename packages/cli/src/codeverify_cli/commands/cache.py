"""CodeVerify CLI - Cache commands."""

from __future__ import annotations

import click
from rich.table import Table

from codeverify_cli.commands import console


@click.group()
def cache():
    """Manage the incremental verification cache."""
    pass


@cache.command("stats")
@click.option(
    "--format", "-f", "output_format", type=click.Choice(["rich", "json"]), default="rich"
)
def cache_stats(output_format: str) -> None:
    """Show verification cache statistics."""
    import json as json_mod

    from codeverify_core.verification_cache import VerificationCache

    vc = VerificationCache()
    stats = vc.get_stats()

    if output_format == "json":
        click.echo(json_mod.dumps(stats, indent=2))
        return

    table = Table(title="Verification Cache Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")
    for key, value in stats.items():
        table.add_row(key.replace("_", " ").title(), str(value))
    console.print(table)


@cache.command("clear")
@click.confirmation_option(prompt="Clear all cached verification results?")
def cache_clear() -> None:
    """Clear all cached verification results."""
    from codeverify_core.verification_cache import VerificationCache

    vc = VerificationCache()
    vc.clear()
    console.print("[green]Cache cleared successfully[/green]")
