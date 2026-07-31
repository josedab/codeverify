"""CodeVerify CLI - Self-healing commands."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from codeverify_cli.commands import console


@click.group()
def heal():
    """Self-healing code suggestions.

    Automatically detect and fix code issues with verified corrections.
    """
    pass


@heal.command("analyze")
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--verify", is_flag=True, help="Verify fixes with Z3 before suggesting")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["rich", "json"]),
    default="rich",
    help="Output format",
)
@click.pass_context
def heal_analyze(ctx: click.Context, path: str, verify: bool, output_format: str) -> None:
    """Analyze code for self-healing opportunities.

    Scans code for issues that can be automatically fixed with
    verified corrections.

    Examples:

        codeverify heal analyze src/
        codeverify heal analyze file.py --verify
    """
    from codeverify_agents import SelfHealingAgent

    verbose = ctx.obj.get("verbose", False)

    console.print(
        Panel.fit(
            "[bold blue]CodeVerify[/bold blue] - Self-Healing Analysis",
            subtitle="Verified Code Fixes",
        )
    )

    path_obj = Path(path)

    files = [path_obj] if path_obj.is_file() else list(path_obj.rglob("*.py"))[:20]  # Limit for CLI

    if not files:
        console.print("[yellow]No Python files found[/yellow]")
        return

    agent = SelfHealingAgent()
    all_fixes = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Analyzing...", total=len(files))

        for file in files:
            try:
                code = file.read_text()
                fixes = asyncio.run(
                    agent.analyze_and_suggest_fixes(
                        code,
                        str(file),
                        verify_fixes=verify,
                    )
                )
                for fix in fixes:
                    fix["file"] = str(file)
                    all_fixes.append(fix)
                progress.advance(task)
            except Exception as e:
                if verbose:
                    console.print(f"[dim]Error in {file}: {e}[/dim]")

    if output_format == "json":
        import json

        click.echo(json.dumps(all_fixes, indent=2, default=str))
        return

    if not all_fixes:
        console.print("[green]✓ No fixable issues found[/green]")
        return

    console.print(f"\nFound [bold]{len(all_fixes)}[/bold] fixable issues:\n")

    for i, fix in enumerate(all_fixes, 1):
        verified = "✓" if fix.get("verified") else "?"
        console.print(f"[bold]{i}. [{verified}] {fix.get('category', 'unknown')}[/bold]")
        console.print(f"   File: {fix.get('file')}:{fix.get('line')}")
        console.print(f"   Issue: {fix.get('issue')}")
        if fix.get("fix"):
            console.print(f"   [green]Fix: {fix.get('fix')[:80]}...[/green]")
        if fix.get("confidence"):
            console.print(f"   Confidence: {fix.get('confidence'):.0%}")
        console.print()


@heal.command("apply")
@click.argument("path", type=click.Path(exists=True))
@click.option("--all", "apply_all", is_flag=True, help="Apply all fixes without prompting")
@click.option("--category", type=str, help="Only apply fixes of this category")
@click.option("--min-confidence", type=float, default=0.8, help="Minimum confidence threshold")
@click.pass_context
def heal_apply(
    ctx: click.Context, path: str, apply_all: bool, category: str, min_confidence: float
) -> None:
    """Apply self-healing fixes to code.

    Examples:

        codeverify heal apply src/module.py --all
        codeverify heal apply . --category null_safety --min-confidence 0.9
    """
    from codeverify_agents import SelfHealingAgent

    console.print("[bold]Scanning for fixable issues...[/bold]")

    path_obj = Path(path)
    agent = SelfHealingAgent()

    files = [path_obj] if path_obj.is_file() else list(path_obj.rglob("*.py"))[:20]

    applied = 0
    skipped = 0

    for file in files:
        try:
            code = file.read_text()
            fixes = asyncio.run(agent.analyze_and_suggest_fixes(code, str(file), verify_fixes=True))

            for fix in fixes:
                # Filter by category
                if category and fix.get("category") != category:
                    continue

                # Filter by confidence
                if fix.get("confidence", 0) < min_confidence:
                    skipped += 1
                    continue

                if not apply_all:
                    console.print(
                        f"\n[bold]{fix.get('category')}[/bold] in {file}:{fix.get('line')}"
                    )
                    console.print(f"Issue: {fix.get('issue')}")
                    console.print(f"Fix: {fix.get('fix')}")
                    if not click.confirm("Apply this fix?"):
                        skipped += 1
                        continue

                # Apply fix
                new_code = asyncio.run(agent.apply_fix(code, fix))
                file.write_text(new_code)
                code = new_code  # Update for subsequent fixes
                applied += 1
                console.print(f"[green]✓ Applied fix to {file}:{fix.get('line')}[/green]")

        except Exception as e:
            console.print(f"[red]Error processing {file}: {e}[/red]")

    console.print(f"\n[bold]Summary:[/bold] Applied {applied} fixes, skipped {skipped}")
