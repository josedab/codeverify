"""CodeVerify CLI - Rules command."""

from __future__ import annotations

from pathlib import Path

import click
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from codeverify_cli.commands import console


@click.command()
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--rule", "-r", multiple=True, help="Rule IDs to apply")
@click.option("--all-rules", is_flag=True, help="Apply all built-in rules")
@click.option("--custom", "-c", type=click.Path(exists=True), help="Custom rules file")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["rich", "json"]),
    default="rich",
    help="Output format",
)
@click.pass_context
def rules(
    ctx: click.Context,
    path: str,
    rule: tuple,
    all_rules: bool,
    custom: str | None,
    output_format: str,
) -> None:
    """Evaluate custom rules against code.

    Run pattern-based, AST, or semantic rules to find specific issues.

    Examples:

        codeverify rules src/ --all-rules
        codeverify rules src/ -r no-print -r no-eval
        codeverify rules src/ --custom my-rules.yml
    """
    from codeverify_core.rules import CustomRule, RuleEvaluator, get_builtin_rules

    verbose = ctx.obj.get("verbose", False)

    console.print(
        Panel.fit(
            "[bold blue]CodeVerify[/bold blue] - Custom Rules", subtitle="Pattern & AST Analysis"
        )
    )

    # Load rules
    evaluator = RuleEvaluator()
    rules_to_apply = []

    if all_rules or not rule:
        rules_to_apply = get_builtin_rules()
        console.print(f"[dim]Using {len(rules_to_apply)} built-in rules[/dim]")

    if rule:
        builtin = {r.id: r for r in get_builtin_rules()}
        for r_id in rule:
            if r_id in builtin:
                rules_to_apply.append(builtin[r_id])
            else:
                console.print(f"[yellow]Warning: Rule '{r_id}' not found[/yellow]")

    if custom:
        import yaml

        with open(custom) as f:
            custom_rules = yaml.safe_load(f)
            for r in custom_rules.get("rules", []):
                from codeverify_core.rules import RuleType

                rules_to_apply.append(
                    CustomRule(
                        id=r["id"],
                        name=r["name"],
                        description=r.get("description", ""),
                        type=RuleType(r.get("type", "pattern")),
                        pattern=r.get("pattern"),
                        severity=r.get("severity", "warning"),
                        message=r.get("message", "Rule violation"),
                    )
                )

    # Find files
    path_obj = Path(path)
    if path_obj.is_file():
        files = [path_obj]
    else:
        files = (
            list(path_obj.rglob("*.py"))
            + list(path_obj.rglob("*.ts"))
            + list(path_obj.rglob("*.js"))
        )

    # Evaluate
    all_violations = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Evaluating rules...", total=len(files))

        for file in files:
            try:
                code = file.read_text()
                for r in rules_to_apply:
                    violations = evaluator.evaluate(r, code)
                    for v in violations:
                        v.file_path = str(file)
                        all_violations.append(v)
                progress.advance(task)
            except Exception as e:
                if verbose:
                    console.print(f"[dim]Error processing {file}: {e}[/dim]")

    if output_format == "json":
        import json

        click.echo(
            json.dumps(
                [
                    {
                        "rule_id": v.rule_id,
                        "file": v.file_path,
                        "line": v.line,
                        "message": v.message,
                        "severity": v.severity,
                    }
                    for v in all_violations
                ],
                indent=2,
            )
        )
        return

    # Rich output
    if not all_violations:
        console.print("[green]✓ No rule violations found![/green]")
        return

    table = Table(title=f"Found {len(all_violations)} Violations")
    table.add_column("Rule", style="cyan")
    table.add_column("File", style="dim")
    table.add_column("Line", justify="right")
    table.add_column("Message")

    for v in all_violations[:50]:  # Limit output
        table.add_row(
            v.rule_id,
            str(Path(v.file_path).name) if v.file_path else "-",
            str(v.line) if v.line else "-",
            v.message[:60] + "..." if len(v.message) > 60 else v.message,
        )

    console.print(table)

    if len(all_violations) > 50:
        console.print(f"[dim]... and {len(all_violations) - 50} more[/dim]")
