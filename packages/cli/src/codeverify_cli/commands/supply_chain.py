"""CodeVerify CLI - Supply chain commands."""

from __future__ import annotations

from pathlib import Path

import click
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from codeverify_cli.commands import console


@click.group("supply-chain")
def supply_chain() -> None:
    """Supply chain verification and dependency analysis."""


@supply_chain.command("scan")
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option(
    "--format", "-f", "output_format", type=click.Choice(["rich", "json"]), default="rich"
)
@click.pass_context
def supply_chain_scan(ctx: click.Context, path: str, output_format: str) -> None:
    """Scan dependencies for vulnerabilities and supply chain threats."""
    from codeverify_core.supply_chain_verification import SupplyChainVerifier

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
    ) as progress:
        progress.add_task("Scanning dependencies...", total=None)

        verifier = SupplyChainVerifier()
        result = verifier.verify(path)

    if output_format == "json":
        import json

        console.print(json.dumps(result, indent=2, default=str))
        return

    table = Table(title="Supply Chain Scan Results")
    table.add_column("Package", style="cyan")
    table.add_column("Threat", style="red")
    table.add_column("Severity", style="yellow")
    table.add_column("Details")

    threats = result.get("threats", [])
    if not threats:
        console.print("[green]✓ No supply chain threats detected[/green]")
        return

    for threat in threats:
        table.add_row(
            str(threat.get("package", "unknown")),
            str(threat.get("threat_type", "unknown")),
            str(threat.get("severity", "unknown")),
            str(threat.get("description", "")),
        )
    console.print(table)


@supply_chain.command("sbom")
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--output", "-o", type=click.Path(), help="Output file for SBOM")
@click.option(
    "--format", "-f", "sbom_format", type=click.Choice(["cyclonedx", "spdx"]), default="cyclonedx"
)
def supply_chain_sbom(path: str, output: str | None, sbom_format: str) -> None:
    """Generate Software Bill of Materials (SBOM)."""
    from codeverify_core.sbom import SBOMFormat, SBOMGenerator

    generator = SBOMGenerator()
    fmt = SBOMFormat.CYCLONEDX if sbom_format == "cyclonedx" else SBOMFormat.SPDX
    sbom = generator.generate(path, fmt)

    import json

    sbom_json = json.dumps(sbom.to_dict(), indent=2, default=str)

    if output:
        Path(output).write_text(sbom_json)
        console.print(f"[green]SBOM written to {output}[/green]")
    else:
        console.print(sbom_json)
