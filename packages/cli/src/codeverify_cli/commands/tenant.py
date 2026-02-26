"""CodeVerify CLI - Tenant commands."""

from __future__ import annotations

import click
from rich.table import Table

from codeverify_cli.commands import console


@click.group()
def tenant():
    """Manage SaaS tenants and usage."""
    pass


@tenant.command("list")
@click.option(
    "--format", "-f", "output_format", type=click.Choice(["rich", "json"]), default="rich"
)
def tenant_list(output_format: str) -> None:
    """List all tenants."""
    import json as json_mod

    from codeverify_core.multi_tenancy import TenantManager

    mgr = TenantManager()
    tenants = mgr.list_tenants()

    if output_format == "json":
        click.echo(
            json_mod.dumps(
                [{"id": t.id, "name": t.name, "tier": t.tier.value} for t in tenants], indent=2
            )
        )
        return

    if not tenants:
        console.print("[yellow]No tenants configured[/yellow]")
        return

    table = Table(title="Tenants")
    table.add_column("ID", style="dim")
    table.add_column("Name", style="cyan")
    table.add_column("Tier", style="bold")
    for t in tenants:
        table.add_row(t.id[:8], t.name, t.tier.value)
    console.print(table)


@tenant.command("create")
@click.argument("name")
@click.argument("slug")
@click.option("--tier", type=click.Choice(["free", "pro", "enterprise"]), default="free")
def tenant_create(name: str, slug: str, tier: str) -> None:
    """Create a new tenant."""
    from codeverify_core.multi_tenancy import TenantManager, TenantTier

    mgr = TenantManager()
    t = mgr.create_tenant(name, slug, TenantTier(tier))
    console.print(
        f"[green]Tenant created:[/green] {t.name} (ID: {t.id[:8]}..., Tier: {t.tier.value})"
    )
