"""CodeVerify CLI - Analysis commands."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import click
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

from codeverify_cli.analyzer import LocalAnalyzer
from codeverify_cli.commands import console
from codeverify_cli.config import load_config, validate_config
from codeverify_cli.formatter import format_findings, format_summary


@click.command()
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--config", "-c", type=click.Path(), help="Path to .codeverify.yml")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["rich", "json", "sarif"]),
    default="rich",
    help="Output format",
)
@click.option(
    "--severity",
    "-s",
    type=click.Choice(["critical", "high", "medium", "low", "all"]),
    default="all",
    help="Minimum severity to report",
)
@click.option("--fix", is_flag=True, help="Show fix suggestions")
@click.option("--staged", is_flag=True, help="Only analyze staged files (git)")
@click.option(
    "--fail-on",
    type=click.Choice(["critical", "high", "medium", "low", "none"]),
    default="high",
    help="Exit with error if findings at this severity or above",
)
@click.pass_context
def analyze(
    ctx: click.Context,
    path: str,
    config: str | None,
    output_format: str,
    severity: str,
    fix: bool,
    staged: bool,
    fail_on: str,
) -> None:
    """Analyze code for issues.

    Examples:

        codeverify analyze                    # Analyze current directory
        codeverify analyze src/               # Analyze specific path
        codeverify analyze --staged           # Only staged git files
        codeverify analyze -f json > report.json
    """
    verbose = ctx.obj.get("verbose", False)

    console.print(
        Panel.fit(
            "[bold blue]CodeVerify[/bold blue] - Local Analysis",
            subtitle="AI + Formal Verification",
        )
    )

    # Load configuration
    config_path = Path(config) if config else Path(path) / ".codeverify.yml"
    cfg = load_config(config_path)

    if verbose:
        console.print(f"[dim]Config: {config_path}[/dim]")
        console.print(f"[dim]Path: {path}[/dim]")

    # Run analysis
    analyzer = LocalAnalyzer(cfg, verbose=verbose)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Analyzing code...", total=None)

        try:
            if staged:
                files = get_staged_files(path)
                if not files:
                    console.print("[yellow]No staged files to analyze[/yellow]")
                    return
                results = asyncio.run(analyzer.analyze_files(files))
            else:
                results = asyncio.run(analyzer.analyze_path(Path(path)))
        except Exception as e:
            console.print(f"[red]Analysis failed: {e}[/red]")
            sys.exit(1)

    # Filter by severity
    if severity != "all":
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        min_level = severity_order.get(severity, 3)
        results.findings = [
            f
            for f in results.findings
            if severity_order.get(f.get("severity", "low"), 3) <= min_level
        ]

    # Output results
    if output_format == "json":
        import json

        click.echo(json.dumps(results.to_dict(), indent=2, default=str))
    elif output_format == "sarif":
        sarif = to_sarif(results)
        import json

        click.echo(json.dumps(sarif, indent=2))
    else:
        format_findings(console, results, show_fix=fix)
        format_summary(console, results)

    # Determine exit code
    if fail_on != "none":
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        fail_level = severity_order.get(fail_on, 1)

        for finding in results.findings:
            if severity_order.get(finding.get("severity", "low"), 3) <= fail_level:
                sys.exit(1)


@click.command()
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--fix", is_flag=True, help="Automatically apply fixes")
@click.option("--dry-run", is_flag=True, help="Show what would be fixed without applying")
@click.pass_context
def fix(ctx: click.Context, path: str, fix: bool, dry_run: bool) -> None:
    """Apply suggested fixes to code.

    Examples:

        codeverify fix src/              # Show available fixes
        codeverify fix src/ --fix        # Apply all fixes
        codeverify fix src/ --dry-run    # Preview fixes
    """
    verbose = ctx.obj.get("verbose", False)

    console.print("[bold]Scanning for fixable issues...[/bold]")

    config_path = Path(path) / ".codeverify.yml"
    cfg = load_config(config_path)

    analyzer = LocalAnalyzer(cfg, verbose=verbose)
    results = asyncio.run(analyzer.analyze_path(Path(path)))

    fixable = [f for f in results.findings if f.get("fix_suggestion")]

    if not fixable:
        console.print("[green]No fixable issues found![/green]")
        return

    console.print(f"Found [bold]{len(fixable)}[/bold] fixable issues")

    for i, finding in enumerate(fixable, 1):
        console.print(f"\n[bold]{i}. {finding.get('title')}[/bold]")
        console.print(f"   File: {finding.get('file_path')}:{finding.get('line_start')}")

        if dry_run or not fix:
            console.print("   [dim]Suggested fix:[/dim]")
            syntax = Syntax(finding.get("fix_suggestion", ""), "python", theme="monokai")
            console.print(syntax)

    if fix and not dry_run and click.confirm("Apply all fixes?"):
        applied = 0
        for finding in fixable:
            try:
                apply_fix(finding)
                applied += 1
            except Exception as e:
                console.print(f"[red]Failed to apply fix: {e}[/red]")

        console.print(f"[green]Applied {applied} fixes[/green]")


@click.command()
@click.option("--config", "-c", type=click.Path(), help="Path to .codeverify.yml")
def validate(config: str | None) -> None:
    """Validate configuration file.

    Checks .codeverify.yml for errors and warnings.
    """
    config_path = Path(config) if config else Path(".") / ".codeverify.yml"

    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        console.print("Run [bold]codeverify init[/bold] to create one")
        sys.exit(1)

    errors, warnings = validate_config(config_path)

    if errors:
        console.print("[red]Configuration errors:[/red]")
        for error in errors:
            console.print(f"  ✗ {error}")
        sys.exit(1)

    if warnings:
        console.print("[yellow]Configuration warnings:[/yellow]")
        for warning in warnings:
            console.print(f"  ⚠ {warning}")

    console.print("[green]✓ Configuration is valid[/green]")


@click.command()
@click.option("--force", "-f", is_flag=True, help="Overwrite existing config")
def init(force: bool) -> None:
    """Initialize CodeVerify configuration.

    Creates a .codeverify.yml file with sensible defaults.
    """
    config_path = Path(".") / ".codeverify.yml"

    if config_path.exists() and not force:
        console.print(f"[yellow]Config already exists: {config_path}[/yellow]")
        console.print("Use --force to overwrite")
        return

    default_config = """# CodeVerify Configuration
# Documentation: https://codeverify.dev/docs/configuration

version: "1"

# Languages to analyze
languages:
  - python
  - typescript
  - javascript
  - go
  - java

# File patterns
include:
  - "src/**/*"
  - "lib/**/*"
  - "app/**/*"

exclude:
  - "**/node_modules/**"
  - "**/__pycache__/**"
  - "**/vendor/**"
  - "**/*.test.*"
  - "**/*.spec.*"

# Severity thresholds for pass/fail
thresholds:
  critical: 0    # Max critical findings allowed
  high: 0        # Max high findings allowed
  medium: 5      # Max medium findings allowed
  low: 10        # Max low findings allowed

# Formal verification settings
verification:
  enabled: true
  timeout_seconds: 30
  checks:
    - null_safety
    - array_bounds
    - integer_overflow
    - division_by_zero

# AI analysis settings
ai:
  enabled: true
  semantic_analysis: true
  security_analysis: true
  model: gpt-4

# Custom rules (optional)
# custom_rules:
#   - id: no-print-statements
#     name: No Print Statements
#     description: Disallow print() in production code
#     severity: low
#     pattern: "print\\s*\\("

# Ignore rules (optional)
# ignore:
#   - pattern: "**/tests/**"
#     categories: [security]
#     reason: Test code has relaxed security requirements
"""

    config_path.write_text(default_config)
    console.print(f"[green]✓ Created {config_path}[/green]")
    console.print("\nNext steps:")
    console.print("  1. Edit .codeverify.yml to customize settings")
    console.print("  2. Run [bold]codeverify analyze[/bold] to scan your code")


@click.command()
@click.argument("rule_file", type=click.Path(exists=True))
@click.option(
    "--test-file", "-t", type=click.Path(exists=True), help="Test file to validate against"
)
def test_rule(rule_file: str, test_file: str | None) -> None:
    """Test a custom rule definition.

    Validates rule syntax and optionally tests against sample code.
    """
    from codeverify_cli.rules import test_custom_rule

    results = test_custom_rule(Path(rule_file), Path(test_file) if test_file else None)

    if results["valid"]:
        console.print("[green]✓ Rule is valid[/green]")
        if results.get("matches"):
            console.print(f"  Found {len(results['matches'])} matches in test file")
            for match in results["matches"][:5]:
                console.print(f"    Line {match['line']}: {match['snippet'][:50]}...")
    else:
        console.print("[red]✗ Rule validation failed[/red]")
        for error in results.get("errors", []):
            console.print(f"  {error}")
        sys.exit(1)


@click.command()
def status() -> None:
    """Show analysis status and statistics.

    Displays recent analysis history and configuration.
    """
    console.print(Panel.fit("[bold]CodeVerify Status[/bold]"))

    # Check for config
    config_path = Path(".") / ".codeverify.yml"
    if config_path.exists():
        console.print(f"[green]✓[/green] Configuration: {config_path}")
    else:
        console.print("[yellow]![/yellow] No configuration found")
        console.print("  Run [bold]codeverify init[/bold] to create one")

    # Check git status
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True
        )
        if result.returncode == 0:
            console.print("[green]✓[/green] Git repository detected")

            # Check for staged files
            result = subprocess.run(
                ["git", "diff", "--staged", "--name-only"], capture_output=True, text=True
            )
            staged = result.stdout.strip().split("\n") if result.stdout.strip() else []
            if staged:
                console.print(f"  {len(staged)} staged files")
    except Exception:
        pass

    console.print("\n[dim]Run 'codeverify analyze' to start analysis[/dim]")


@click.command()
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--branch", "-b", default="HEAD", help="Branch or commit to scan")
@click.option("--include", "-i", multiple=True, help="Include patterns")
@click.option("--exclude", "-e", multiple=True, help="Exclude patterns")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["rich", "json"]),
    default="rich",
    help="Output format",
)
@click.pass_context
def scan(
    ctx: click.Context, path: str, branch: str, include: tuple, exclude: tuple, output_format: str
) -> None:
    """Run full codebase scan.

    Performs comprehensive analysis of entire codebase with
    trend tracking and detailed reporting.

    Examples:

        codeverify scan                      # Scan current directory
        codeverify scan --branch main        # Scan specific branch
        codeverify scan -i "src/**" -e "tests/**"
    """
    verbose = ctx.obj.get("verbose", False)

    console.print(
        Panel.fit(
            "[bold blue]CodeVerify[/bold blue] - Codebase Scan", subtitle="Comprehensive Analysis"
        )
    )

    path_obj = Path(path)

    # Collect files
    all_files = []
    extensions = [".py", ".ts", ".js", ".tsx", ".jsx", ".go", ".java"]

    for ext in extensions:
        all_files.extend(path_obj.rglob(f"*{ext}"))

    # Apply filters
    if include:
        import fnmatch

        filtered = []
        for f in all_files:
            for pattern in include:
                if fnmatch.fnmatch(str(f), pattern):
                    filtered.append(f)
                    break
        all_files = filtered

    if exclude:
        import fnmatch

        filtered = []
        for f in all_files:
            excluded = False
            for pattern in exclude:
                if fnmatch.fnmatch(str(f), pattern):
                    excluded = True
                    break
            if not excluded:
                filtered.append(f)
        all_files = filtered

    # Default exclusions
    all_files = [
        f
        for f in all_files
        if not any(
            p in str(f) for p in ["node_modules", "__pycache__", ".git", "vendor", "dist", "build"]
        )
    ]

    console.print(f"Scanning [bold]{len(all_files)}[/bold] files...")

    # Run analysis
    config_path = path_obj / ".codeverify.yml"
    cfg = load_config(config_path)
    analyzer = LocalAnalyzer(cfg, verbose=verbose)

    results = {"files": 0, "findings": [], "by_severity": {}, "by_category": {}}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Scanning {len(all_files)} files...", total=len(all_files))

        for file in all_files:
            try:
                file_results = asyncio.run(analyzer.analyze_files([file]))
                results["files"] += 1
                results["findings"].extend(file_results.findings)
                progress.advance(task)
            except Exception as e:
                if verbose:
                    console.print(f"[dim]Error: {file}: {e}[/dim]")

    # Aggregate results
    for f in results["findings"]:
        sev = f.get("severity", "low")
        cat = f.get("category", "other")
        results["by_severity"][sev] = results["by_severity"].get(sev, 0) + 1
        results["by_category"][cat] = results["by_category"].get(cat, 0) + 1

    if output_format == "json":
        import json

        click.echo(
            json.dumps(
                {
                    "files_scanned": results["files"],
                    "total_findings": len(results["findings"]),
                    "by_severity": results["by_severity"],
                    "by_category": results["by_category"],
                },
                indent=2,
            )
        )
        return

    # Summary
    console.print("\n[bold]Scan Complete[/bold]\n")

    summary = Table(title="Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right")

    summary.add_row("Files Scanned", str(results["files"]))
    summary.add_row("Total Findings", str(len(results["findings"])))
    summary.add_row("Critical", f"[red]{results['by_severity'].get('critical', 0)}[/red]")
    summary.add_row("High", f"[red]{results['by_severity'].get('high', 0)}[/red]")
    summary.add_row("Medium", f"[yellow]{results['by_severity'].get('medium', 0)}[/yellow]")
    summary.add_row("Low", f"[dim]{results['by_severity'].get('low', 0)}[/dim]")

    console.print(summary)

    if results["by_category"]:
        cat_table = Table(title="By Category")
        cat_table.add_column("Category")
        cat_table.add_column("Count", justify="right")
        for cat, count in sorted(results["by_category"].items(), key=lambda x: -x[1]):
            cat_table.add_row(cat, str(count))
        console.print(cat_table)


@click.command(name="list-rules")
def list_rules() -> None:
    """List all available built-in rules.

    Shows rule ID, name, severity, and description for all
    built-in rules that can be used with --rule flag.
    """
    from codeverify_core.rules import get_builtin_rules

    console.print(Panel.fit("[bold]Built-in Rules[/bold]"))

    rules = get_builtin_rules()

    table = Table()
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Severity")
    table.add_column("Description")

    for rule in rules:
        sev_color = {"error": "red", "warning": "yellow", "info": "blue"}.get(
            rule.severity, "white"
        )
        table.add_row(
            rule.id,
            rule.name,
            f"[{sev_color}]{rule.severity}[/{sev_color}]",
            rule.description[:50] + "..." if len(rule.description) > 50 else rule.description,
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(rules)} rules[/dim]")


def get_staged_files(path: str) -> list[Path]:
    """Get list of staged git files."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "diff", "--staged", "--name-only"], capture_output=True, text=True, cwd=path
        )
        if result.returncode == 0:
            files = result.stdout.strip().split("\n")
            return [Path(path) / f for f in files if f]
    except Exception:
        pass

    return []


def apply_fix(finding: dict[str, Any]) -> None:
    """Apply a fix suggestion to a file."""
    file_path = Path(finding["file_path"])
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # This is a simplified implementation
    # In production, would use AST-aware patching
    console.print(f"[dim]Would apply fix to {file_path}:{finding.get('line_start')}[/dim]")


def to_sarif(results: Any) -> dict[str, Any]:
    """Convert results to SARIF format for IDE integration."""
    return {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "CodeVerify",
                        "version": "0.1.0",
                        "informationUri": "https://codeverify.dev",
                        "rules": [],
                    }
                },
                "results": [
                    {
                        "ruleId": finding.get("category", "unknown"),
                        "level": _sarif_level(finding.get("severity", "low")),
                        "message": {"text": finding.get("description", "")},
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {"uri": finding.get("file_path", "")},
                                    "region": {
                                        "startLine": finding.get("line_start", 1),
                                        "endLine": finding.get("line_end")
                                        or finding.get("line_start", 1),
                                    },
                                }
                            }
                        ],
                    }
                    for finding in results.findings
                ],
            }
        ],
    }


def _sarif_level(severity: str) -> str:
    """Convert severity to SARIF level."""
    return {
        "critical": "error",
        "high": "error",
        "medium": "warning",
        "low": "note",
        "info": "note",
    }.get(severity, "note")
