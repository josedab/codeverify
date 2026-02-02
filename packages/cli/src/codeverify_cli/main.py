"""CodeVerify CLI - Main entry point."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.syntax import Syntax
from rich import print as rprint

from codeverify_cli.analyzer import LocalAnalyzer
from codeverify_cli.config import load_config, validate_config
from codeverify_cli.formatter import format_findings, format_summary

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="codeverify")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """CodeVerify - AI-powered code analysis with formal verification.
    
    Run local analysis on your code before pushing to catch bugs,
    security issues, and verify correctness.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@cli.command()
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--config", "-c", type=click.Path(), help="Path to .codeverify.yml")
@click.option("--format", "-f", "output_format", 
              type=click.Choice(["rich", "json", "sarif"]), 
              default="rich", help="Output format")
@click.option("--severity", "-s", 
              type=click.Choice(["critical", "high", "medium", "low", "all"]),
              default="all", help="Minimum severity to report")
@click.option("--fix", is_flag=True, help="Show fix suggestions")
@click.option("--staged", is_flag=True, help="Only analyze staged files (git)")
@click.option("--fail-on", type=click.Choice(["critical", "high", "medium", "low", "none"]),
              default="high", help="Exit with error if findings at this severity or above")
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
    
    console.print(Panel.fit(
        "[bold blue]CodeVerify[/bold blue] - Local Analysis",
        subtitle="AI + Formal Verification"
    ))
    
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
        task = progress.add_task("Analyzing code...", total=None)
        
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
            f for f in results.findings
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


@cli.command()
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
    
    if fix and not dry_run:
        if click.confirm("Apply all fixes?"):
            applied = 0
            for finding in fixable:
                try:
                    apply_fix(finding)
                    applied += 1
                except Exception as e:
                    console.print(f"[red]Failed to apply fix: {e}[/red]")
            
            console.print(f"[green]Applied {applied} fixes[/green]")


@cli.command()
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


@cli.command()
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


@cli.command()
@click.argument("rule_file", type=click.Path(exists=True))
@click.option("--test-file", "-t", type=click.Path(exists=True), help="Test file to validate against")
def test_rule(rule_file: str, test_file: str | None) -> None:
    """Test a custom rule definition.
    
    Validates rule syntax and optionally tests against sample code.
    """
    from codeverify_cli.rules import test_custom_rule
    
    results = test_custom_rule(Path(rule_file), Path(test_file) if test_file else None)
    
    if results["valid"]:
        console.print(f"[green]✓ Rule is valid[/green]")
        if results.get("matches"):
            console.print(f"  Found {len(results['matches'])} matches in test file")
            for match in results["matches"][:5]:
                console.print(f"    Line {match['line']}: {match['snippet'][:50]}...")
    else:
        console.print(f"[red]✗ Rule validation failed[/red]")
        for error in results.get("errors", []):
            console.print(f"  {error}")
        sys.exit(1)


@cli.command()
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
        console.print(f"[yellow]![/yellow] No configuration found")
        console.print("  Run [bold]codeverify init[/bold] to create one")
    
    # Check git status
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            console.print("[green]✓[/green] Git repository detected")
            
            # Check for staged files
            result = subprocess.run(
                ["git", "diff", "--staged", "--name-only"],
                capture_output=True, text=True
            )
            staged = result.stdout.strip().split("\n") if result.stdout.strip() else []
            if staged:
                console.print(f"  {len(staged)} staged files")
    except Exception:
        pass
    
    console.print("\n[dim]Run 'codeverify analyze' to start analysis[/dim]")


def get_staged_files(path: str) -> list[Path]:
    """Get list of staged git files."""
    import subprocess
    
    try:
        result = subprocess.run(
            ["git", "diff", "--staged", "--name-only"],
            capture_output=True, text=True, cwd=path
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
        "runs": [{
            "tool": {
                "driver": {
                    "name": "CodeVerify",
                    "version": "0.1.0",
                    "informationUri": "https://codeverify.dev",
                    "rules": []
                }
            },
            "results": [
                {
                    "ruleId": finding.get("category", "unknown"),
                    "level": _sarif_level(finding.get("severity", "low")),
                    "message": {"text": finding.get("description", "")},
                    "locations": [{
                        "physicalLocation": {
                            "artifactLocation": {"uri": finding.get("file_path", "")},
                            "region": {
                                "startLine": finding.get("line_start", 1),
                                "endLine": finding.get("line_end") or finding.get("line_start", 1),
                            }
                        }
                    }]
                }
                for finding in results.findings
            ]
        }]
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


# ============================================================================
# New Feature Commands
# ============================================================================


@cli.command()
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--format", "-f", "output_format",
              type=click.Choice(["rich", "json"]),
              default="rich", help="Output format")
@click.pass_context
def trust_score(ctx: click.Context, path: str, output_format: str) -> None:
    """Calculate trust score for code.
    
    Analyzes code to determine its trustworthiness, particularly for
    AI-generated code. Shows risk level and recommendations.
    
    Examples:
    
        codeverify trust-score src/module.py
        codeverify trust-score . -f json
    """
    from codeverify_agents import TrustScoreAgent
    
    verbose = ctx.obj.get("verbose", False)
    
    console.print(Panel.fit(
        "[bold blue]CodeVerify[/bold blue] - Trust Score Analysis",
        subtitle="AI Detection & Risk Assessment"
    ))
    
    path_obj = Path(path)
    
    if path_obj.is_file():
        files = [path_obj]
    else:
        files = list(path_obj.rglob("*.py")) + list(path_obj.rglob("*.ts")) + list(path_obj.rglob("*.js"))
        files = files[:20]  # Limit for CLI
    
    if not files:
        console.print("[yellow]No supported files found[/yellow]")
        return
    
    agent = TrustScoreAgent()
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Analyzing trust scores...", total=len(files))
        
        for file in files:
            try:
                code = file.read_text()
                result = asyncio.run(agent.analyze(code, {"file_path": str(file)}))
                results.append({
                    "file": str(file),
                    "score": result.score,
                    "risk_level": result.risk_level,
                    "ai_probability": result.ai_probability,
                    "recommendations": result.recommendations[:3],
                })
                progress.advance(task)
            except Exception as e:
                if verbose:
                    console.print(f"[dim]Skipped {file}: {e}[/dim]")
    
    if output_format == "json":
        import json
        click.echo(json.dumps(results, indent=2))
        return
    
    # Rich output
    table = Table(title="Trust Score Results")
    table.add_column("File", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Risk", justify="center")
    table.add_column("AI Prob", justify="right")
    
    for r in results:
        score = r["score"]
        risk = r["risk_level"]
        
        # Color code
        if score >= 80:
            score_str = f"[green]{score:.0f}[/green]"
        elif score >= 60:
            score_str = f"[yellow]{score:.0f}[/yellow]"
        else:
            score_str = f"[red]{score:.0f}[/red]"
        
        risk_colors = {"low": "green", "medium": "yellow", "high": "red", "critical": "red bold"}
        risk_str = f"[{risk_colors.get(risk, 'white')}]{risk}[/{risk_colors.get(risk, 'white')}]"
        
        table.add_row(
            str(Path(r["file"]).name),
            score_str,
            risk_str,
            f"{r['ai_probability']:.0f}%"
        )
    
    console.print(table)
    
    # Summary
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0
    console.print(f"\n[bold]Average Trust Score:[/bold] {avg_score:.1f}/100")


@cli.command()
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--rule", "-r", multiple=True, help="Rule IDs to apply")
@click.option("--all-rules", is_flag=True, help="Apply all built-in rules")
@click.option("--custom", "-c", type=click.Path(exists=True), help="Custom rules file")
@click.option("--format", "-f", "output_format",
              type=click.Choice(["rich", "json"]),
              default="rich", help="Output format")
@click.pass_context
def rules(ctx: click.Context, path: str, rule: tuple, all_rules: bool, 
          custom: str | None, output_format: str) -> None:
    """Evaluate custom rules against code.
    
    Run pattern-based, AST, or semantic rules to find specific issues.
    
    Examples:
    
        codeverify rules src/ --all-rules
        codeverify rules src/ -r no-print -r no-eval
        codeverify rules src/ --custom my-rules.yml
    """
    from codeverify_core.rules import RuleEvaluator, get_builtin_rules, CustomRule
    
    verbose = ctx.obj.get("verbose", False)
    
    console.print(Panel.fit(
        "[bold blue]CodeVerify[/bold blue] - Custom Rules",
        subtitle="Pattern & AST Analysis"
    ))
    
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
                rules_to_apply.append(CustomRule(
                    id=r["id"],
                    name=r["name"],
                    description=r.get("description", ""),
                    type=RuleType(r.get("type", "pattern")),
                    pattern=r.get("pattern"),
                    severity=r.get("severity", "warning"),
                    message=r.get("message", "Rule violation"),
                ))
    
    # Find files
    path_obj = Path(path)
    if path_obj.is_file():
        files = [path_obj]
    else:
        files = list(path_obj.rglob("*.py")) + list(path_obj.rglob("*.ts")) + list(path_obj.rglob("*.js"))
    
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
        click.echo(json.dumps([{
            "rule_id": v.rule_id,
            "file": v.file_path,
            "line": v.line,
            "message": v.message,
            "severity": v.severity,
        } for v in all_violations], indent=2))
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


@cli.command()
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--branch", "-b", default="HEAD", help="Branch or commit to scan")
@click.option("--include", "-i", multiple=True, help="Include patterns")
@click.option("--exclude", "-e", multiple=True, help="Exclude patterns")
@click.option("--format", "-f", "output_format",
              type=click.Choice(["rich", "json"]),
              default="rich", help="Output format")
@click.pass_context
def scan(ctx: click.Context, path: str, branch: str, include: tuple,
         exclude: tuple, output_format: str) -> None:
    """Run full codebase scan.
    
    Performs comprehensive analysis of entire codebase with
    trend tracking and detailed reporting.
    
    Examples:
    
        codeverify scan                      # Scan current directory
        codeverify scan --branch main        # Scan specific branch
        codeverify scan -i "src/**" -e "tests/**"
    """
    verbose = ctx.obj.get("verbose", False)
    
    console.print(Panel.fit(
        "[bold blue]CodeVerify[/bold blue] - Codebase Scan",
        subtitle="Comprehensive Analysis"
    ))
    
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
    all_files = [f for f in all_files if not any(
        p in str(f) for p in ["node_modules", "__pycache__", ".git", "vendor", "dist", "build"]
    )]
    
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
        click.echo(json.dumps({
            "files_scanned": results["files"],
            "total_findings": len(results["findings"]),
            "by_severity": results["by_severity"],
            "by_category": results["by_category"],
        }, indent=2))
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


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--function", "-fn", help="Specific function to debug")
@click.option("--interactive", "-i", is_flag=True, help="Interactive step-through mode")
@click.pass_context
def debug(ctx: click.Context, file: str, function: str | None, interactive: bool) -> None:
    """Debug verification for a file.
    
    Shows step-by-step verification trace to understand
    how formal verification analyzes your code.
    
    Examples:
    
        codeverify debug src/math.py
        codeverify debug src/math.py --function calculate
        codeverify debug src/math.py -i
    """
    from codeverify_verifier import VerificationDebugger
    
    verbose = ctx.obj.get("verbose", False)
    
    console.print(Panel.fit(
        "[bold blue]CodeVerify[/bold blue] - Verification Debugger",
        subtitle="Step-by-Step Analysis"
    ))
    
    file_path = Path(file)
    code = file_path.read_text()
    
    debugger = VerificationDebugger()
    
    console.print(f"Analyzing: [cyan]{file_path}[/cyan]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Running verification...", total=None)
        result = asyncio.run(debugger.trace(code))
    
    # Display steps
    steps = result.get("steps", [])
    
    if not steps:
        console.print("[yellow]No verification steps generated[/yellow]")
        console.print(f"Result: {result.get('result', 'unknown')}")
        return
    
    console.print(f"[bold]Verification Result:[/bold] {result.get('result', 'unknown')}\n")
    
    for i, step in enumerate(steps, 1):
        status = step.get("status", "pending")
        status_icon = {
            "passed": "[green]✓[/green]",
            "failed": "[red]✗[/red]",
            "pending": "[yellow]○[/yellow]",
            "skipped": "[dim]○[/dim]",
        }.get(status, "○")
        
        console.print(f"{status_icon} [bold]Step {i}:[/bold] {step.get('title', 'Unknown')}")
        
        if step.get("description"):
            console.print(f"   {step['description']}")
        
        if step.get("constraint") and verbose:
            console.print(f"   [dim]Constraint: {step['constraint']}[/dim]")
        
        if step.get("model") and status == "failed":
            console.print(f"   [red]Counterexample: {step['model']}[/red]")
        
        if interactive:
            if not click.confirm("Continue?", default=True):
                break
    
    # Show counterexample if verification failed
    if result.get("result") == "unverified" and result.get("counterexample"):
        console.print("\n[bold red]Counterexample Found:[/bold red]")
        for var, val in result["counterexample"].items():
            console.print(f"  {var} = {val}")


@cli.command(name="list-rules")
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
        sev_color = {"error": "red", "warning": "yellow", "info": "blue"}.get(rule.severity, "white")
        table.add_row(
            rule.id,
            rule.name,
            f"[{sev_color}]{rule.severity}[/{sev_color}]",
            rule.description[:50] + "..." if len(rule.description) > 50 else rule.description,
        )
    
    console.print(table)
    console.print(f"\n[dim]Total: {len(rules)} rules[/dim]")


# ============================================
# Next-Gen Feature CLI Commands
# ============================================

@cli.group()
def monorepo() -> None:
    """Monorepo intelligence commands.
    
    Analyze monorepo structure, dependencies, and affected packages.
    """
    pass


@monorepo.command("analyze")
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--format", "-f", "output_format",
              type=click.Choice(["rich", "json", "dot"]),
              default="rich", help="Output format")
@click.pass_context
def monorepo_analyze(ctx: click.Context, path: str, output_format: str) -> None:
    """Analyze monorepo structure and dependencies.
    
    Examples:
    
        codeverify monorepo analyze              # Current directory
        codeverify monorepo analyze ./my-repo   # Specific path
        codeverify monorepo analyze -f json     # JSON output
    """
    from codeverify_core.monorepo import MonorepoAnalyzer, MonorepoType
    
    console.print(Panel.fit(
        "[bold blue]CodeVerify[/bold blue] - Monorepo Analysis",
        subtitle="Dependency Intelligence"
    ))
    
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
            "packages": [{"name": p.name, "path": str(p.path), "version": p.version} for p in packages],
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
            table.add_row(pkg.name, pkg.version or "-", str(pkg.path.relative_to(path_obj)), str(deps))
        
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


@cli.command("generate-tests")
@click.argument("file", type=click.Path(exists=True))
@click.option("--framework", "-f",
              type=click.Choice(["pytest", "unittest", "jest", "vitest", "go"]),
              default=None, help="Test framework (auto-detected if not specified)")
@click.option("--output", "-o", type=click.Path(), help="Output file")
@click.pass_context
def generate_tests(ctx: click.Context, file: str, framework: str | None, output: str | None) -> None:
    """Generate regression tests from verification counterexamples.
    
    Examples:
    
        codeverify generate-tests src/math.py
        codeverify generate-tests src/utils.ts -f jest
        codeverify generate-tests src/calc.py -o tests/test_calc.py
    """
    from codeverify_agents.test_generator import TestGeneratorAgent, TestFramework
    
    console.print(Panel.fit(
        "[bold blue]CodeVerify[/bold blue] - Test Generator",
        subtitle="Counterexample → Tests"
    ))
    
    file_path = Path(file)
    agent = TestGeneratorAgent()
    
    # Detect language and framework
    ext = file_path.suffix
    lang_map = {".py": "python", ".ts": "typescript", ".js": "javascript", ".go": "go"}
    language = lang_map.get(ext, "python")
    
    if framework:
        fw = TestFramework(framework if framework != "go" else "go_test")
    else:
        fw = agent._select_framework(language)
    
    console.print(f"Language: [cyan]{language}[/cyan]")
    console.print(f"Framework: [cyan]{fw.value}[/cyan]")
    
    # Note: In real implementation, this would analyze the file and generate tests
    console.print("\n[dim]Analyzing file for verification counterexamples...[/dim]")
    console.print("[yellow]Note: Run verification first to generate counterexamples[/yellow]")


@cli.command("attest")
@click.argument("target")
@click.option("--verify", "-v", is_flag=True, help="Verify existing attestation")
@click.option("--output", "-o", type=click.Path(), help="Output file for attestation")
@click.pass_context
def attest(ctx: click.Context, target: str, verify: bool, output: str | None) -> None:
    """Create or verify verification attestations.
    
    Examples:
    
        codeverify attest PR#123                 # Create attestation for PR
        codeverify attest commit:abc123         # Attest specific commit
        codeverify attest --verify att.json    # Verify attestation
    """
    from codeverify_core.proof_carrying import ProofCarryingManager
    
    console.print(Panel.fit(
        "[bold blue]CodeVerify[/bold blue] - Proof Attestation",
        subtitle="Cryptographic Verification"
    ))
    
    # Get secret key from environment
    secret_key = os.environ.get("CODEVERIFY_ATTESTATION_KEY", "development-key")
    manager = ProofCarryingManager(secret_key=secret_key)
    
    if verify:
        # Verify existing attestation
        try:
            with open(target, "r") as f:
                content = f.read()
            attestation = manager.extract_from_commit_message(content)
            if attestation and manager.verify_attestation(attestation):
                console.print("[green]✓ Attestation valid[/green]")
                console.print(f"  Code hash: {attestation.proof.code_hash}")
                console.print(f"  Type: {attestation.proof.verification_type}")
                console.print(f"  Result: {attestation.proof.result}")
            else:
                console.print("[red]✗ Attestation invalid or not found[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    else:
        # Create new attestation
        console.print(f"Creating attestation for: [cyan]{target}[/cyan]")
        console.print("[yellow]Note: Run verification first, then attest results[/yellow]")


@cli.command("invariants")
@click.argument("spec", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Choice(["z3", "smt", "both"]), default="z3")
@click.pass_context
def invariants(ctx: click.Context, spec: str, output: str) -> None:
    """Compile natural language invariants to Z3 assertions.
    
    Examples:
    
        codeverify invariants specs/balance.txt
        codeverify invariants specs/user.md -o smt
    """
    from codeverify_agents.nl_invariants import NaturalLanguageInvariantsAgent
    
    console.print(Panel.fit(
        "[bold blue]CodeVerify[/bold blue] - NL Invariants",
        subtitle="English → Z3"
    ))
    
    spec_path = Path(spec)
    content = spec_path.read_text()
    
    agent = NaturalLanguageInvariantsAgent()
    
    console.print(f"[dim]Processing {spec_path.name}...[/dim]")
    
    for line in content.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        
        console.print(f"\n[cyan]Input:[/cyan] {line}")
        constraints = agent._parse_constraints(line)
        
        for c in constraints:
            z3_code = agent._to_z3(c)
            console.print(f"  [green]Z3:[/green] {z3_code}")


@cli.command("semantic-diff")
@click.argument("old_file", type=click.Path(exists=True))
@click.argument("new_file", type=click.Path(exists=True))
@click.option("--format", "-f", "output_format",
              type=click.Choice(["rich", "mermaid", "dot", "html"]),
              default="rich", help="Output format")
@click.option("--output", "-o", type=click.Path(), help="Output file")
@click.pass_context
def semantic_diff(ctx: click.Context, old_file: str, new_file: str,
                  output_format: str, output: str | None) -> None:
    """Visualize behavioral changes between two versions.
    
    Examples:
    
        codeverify semantic-diff old.py new.py
        codeverify semantic-diff v1.ts v2.ts -f mermaid
        codeverify semantic-diff main.py feature.py -f html -o diff.html
    """
    from codeverify_agents.semantic_diff import SemanticDiffAgent, ChangeType
    
    console.print(Panel.fit(
        "[bold blue]CodeVerify[/bold blue] - Semantic Diff",
        subtitle="Behavioral Analysis"
    ))
    
    old_path = Path(old_file)
    new_path = Path(new_file)
    
    old_code = old_path.read_text()
    new_code = new_path.read_text()
    
    # Detect language
    ext = old_path.suffix
    lang_map = {".py": "python", ".ts": "typescript", ".js": "javascript", ".go": "go"}
    language = lang_map.get(ext, "python")
    
    agent = SemanticDiffAgent()
    
    console.print(f"Comparing: [cyan]{old_path.name}[/cyan] → [cyan]{new_path.name}[/cyan]")
    console.print(f"Language: {language}")
    
    changes = agent._detect_signature_changes(old_code, new_code, language)
    
    if output_format == "mermaid":
        diagram = agent._to_mermaid(changes)
        if output:
            Path(output).write_text(diagram)
            console.print(f"[green]Saved to {output}[/green]")
        else:
            console.print(diagram)
    elif output_format == "dot":
        diagram = agent._to_dot(changes)
        if output:
            Path(output).write_text(diagram)
            console.print(f"[green]Saved to {output}[/green]")
        else:
            console.print(diagram)
    else:
        if changes:
            table = Table(title="Behavioral Changes")
            table.add_column("Type", style="cyan")
            table.add_column("Location")
            table.add_column("Change")
            table.add_column("Impact")
            
            for change in changes:
                impact_color = "red" if "breaking" in change.impact.lower() else "yellow"
                table.add_row(
                    change.change_type.value,
                    change.location,
                    f"{change.old_behavior[:20]}... → {change.new_behavior[:20]}...",
                    f"[{impact_color}]{change.impact}[/{impact_color}]"
                )
            
            console.print(table)
        else:
            console.print("[green]No behavioral changes detected[/green]")


@cli.command("budget")
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
    from codeverify_core.budget_optimizer import (
        VerificationBudgetOptimizer, Budget, RiskFactors
    )
    
    console.print(Panel.fit(
        "[bold blue]CodeVerify[/bold blue] - Budget Optimizer",
        subtitle="Cost Management"
    ))
    
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
            file_infos.append({
                "file_path": str(f),
                "size_lines": lines,
                "factors": RiskFactors(file_complexity=min(lines / 500, 1.0)),
            })
        
        result = optimizer.optimize_batch(file_infos, budget_obj)
        
        table = Table(title="Verification Plan")
        table.add_column("File", style="cyan")
        table.add_column("Depth")
        table.add_column("Risk")
        table.add_column("Cost")
        
        for d in result.decisions:
            risk_color = "red" if d.risk_score > 0.7 else "yellow" if d.risk_score > 0.3 else "green"
            table.add_row(
                Path(d.file_path).name,
                d.depth.value,
                f"[{risk_color}]{d.risk_score:.2f}[/{risk_color}]",
                f"${d.estimated_cost:.3f}"
            )
        
        console.print(table)
        console.print(f"\n[bold]Total estimated cost:[/bold] ${result.total_estimated_cost:.2f}")
        console.print(f"[bold]Budget utilization:[/bold] {result.budget_utilization:.1%}")
        
    else:  # report
        report = optimizer.get_usage_report()
        console.print(f"Total cost: ${report['total_cost']:.2f}")
        console.print(f"Total files: {report['total_files']}")


@cli.command("team-report")
@click.option("--output", "-o", type=click.Path(), help="Output file (markdown)")
@click.option("--format", "-f", "output_format",
              type=click.Choice(["rich", "markdown", "json"]),
              default="rich", help="Output format")
@click.pass_context
def team_report(ctx: click.Context, output: str | None, output_format: str) -> None:
    """Generate team learning report.
    
    Examples:
    
        codeverify team-report
        codeverify team-report -o report.md -f markdown
    """
    from codeverify_agents.team_learning import TeamLearningAgent
    
    console.print(Panel.fit(
        "[bold blue]CodeVerify[/bold blue] - Team Learning",
        subtitle="Organization Insights"
    ))
    
    agent = TeamLearningAgent()
    report = agent.generate_org_health_report()
    
    if output_format == "markdown" or output:
        md = agent.export_report_markdown(report)
        if output:
            Path(output).write_text(md)
            console.print(f"[green]Report saved to {output}[/green]")
        else:
            console.print(md)
    elif output_format == "json":
        import json
        data = {
            "total_findings": report.total_findings,
            "total_prs": report.total_prs_analyzed,
            "trend": report.trend_vs_last_period.value if hasattr(report.trend_vs_last_period, 'value') else str(report.trend_vs_last_period),
            "patterns": len(report.systemic_patterns),
            "recommendations": len(report.training_recommendations),
        }
        click.echo(json.dumps(data, indent=2))
    else:
        console.print(f"Total findings: [bold]{report.total_findings}[/bold]")
        console.print(f"PRs analyzed: {report.total_prs_analyzed}")
        console.print(f"Systemic patterns: {len(report.systemic_patterns)}")
        console.print(f"Training recommendations: {len(report.training_recommendations)}")
        
        if report.teams_needing_attention:
            console.print(f"\n[yellow]Teams needing attention:[/yellow]")
            for team in report.teams_needing_attention:
                console.print(f"  • {team}")


@cli.group()
def ramp() -> None:
    """Gradual verification ramp commands.
    
    Manage warnings-only onboarding mode for repositories.
    """
    pass


@ramp.command("start")
@click.argument("repository")
@click.option("--baseline-days", type=int, default=7, help="Days for baseline collection")
@click.option("--observation-days", type=int, default=14, help="Days for observation period")
@click.option("--transition-days", type=int, default=14, help="Days for transition period")
@click.pass_context
def ramp_start(ctx: click.Context, repository: str, baseline_days: int,
               observation_days: int, transition_days: int) -> None:
    """Start verification ramp for a repository.
    
    Examples:
    
        codeverify ramp start myorg/myrepo
        codeverify ramp start myrepo --baseline-days 3 --observation-days 7
    """
    from codeverify_core.gradual_ramp import GradualVerificationRamp, RampSchedule
    
    console.print(Panel.fit(
        "[bold blue]CodeVerify[/bold blue] - Gradual Ramp",
        subtitle="Onboarding Mode"
    ))
    
    schedule = RampSchedule(
        baseline_days=baseline_days,
        observation_days=observation_days,
        transition_days=transition_days,
    )
    
    ramp_manager = GradualVerificationRamp(default_schedule=schedule)
    state = ramp_manager.start_ramp(repository)
    
    console.print(f"[green]✓ Ramp started for {repository}[/green]")
    console.print(f"\nSchedule:")
    console.print(f"  Baseline: {baseline_days} days")
    console.print(f"  Observation: {observation_days} days")
    console.print(f"  Transition: {transition_days} days")
    console.print(f"\nTotal: {baseline_days + observation_days + transition_days} days until full enforcement")


@ramp.command("status")
@click.argument("repository")
@click.pass_context
def ramp_status(ctx: click.Context, repository: str) -> None:
    """Show ramp status for a repository.
    
    Examples:
    
        codeverify ramp status myorg/myrepo
    """
    from codeverify_core.gradual_ramp import GradualVerificationRamp
    
    ramp_manager = GradualVerificationRamp()
    progress = ramp_manager.get_progress_report(repository)
    
    if not progress:
        console.print(f"[yellow]No ramp found for {repository}[/yellow]")
        console.print("Use 'codeverify ramp start' to begin")
        return
    
    console.print(Panel.fit(
        f"[bold blue]Ramp Status:[/bold blue] {repository}",
    ))
    
    phase_colors = {
        "baseline": "blue",
        "observation": "yellow",
        "transition": "orange3",
        "enforcing": "green",
    }
    phase = progress.current_phase.value
    color = phase_colors.get(phase, "white")
    
    console.print(f"Phase: [{color}]{phase.upper()}[/{color}]")
    console.print(f"Enforcement: {progress.enforcement_level.value}")
    console.print(f"Progress: {progress.percent_complete:.1f}%")
    console.print(f"Days elapsed: {progress.days_elapsed}")
    console.print(f"Days remaining: {progress.days_remaining}")
    console.print(f"\nNext milestone: {progress.next_milestone}")


@ramp.command("pause")
@click.argument("repository")
@click.pass_context
def ramp_pause(ctx: click.Context, repository: str) -> None:
    """Pause ramp for a repository."""
    from codeverify_core.gradual_ramp import GradualVerificationRamp
    
    ramp_manager = GradualVerificationRamp()
    if ramp_manager.pause_ramp(repository):
        console.print(f"[yellow]⏸ Ramp paused for {repository}[/yellow]")
    else:
        console.print(f"[red]No ramp found for {repository}[/red]")


@ramp.command("resume")
@click.argument("repository")
@click.pass_context
def ramp_resume(ctx: click.Context, repository: str) -> None:
    """Resume paused ramp for a repository."""
    from codeverify_core.gradual_ramp import GradualVerificationRamp
    
    ramp_manager = GradualVerificationRamp()
    if ramp_manager.resume_ramp(repository):
        console.print(f"[green]▶ Ramp resumed for {repository}[/green]")
    else:
        console.print(f"[red]No ramp found for {repository}[/red]")


@ramp.command("end")
@click.argument("repository")
@click.option("--confirm", is_flag=True, help="Skip confirmation")
@click.pass_context
def ramp_end(ctx: click.Context, repository: str, confirm: bool) -> None:
    """End ramp and enable full enforcement."""
    from codeverify_core.gradual_ramp import GradualVerificationRamp
    
    if not confirm:
        if not click.confirm(f"End ramp and enable full enforcement for {repository}?"):
            return
    
    ramp_manager = GradualVerificationRamp()
    if ramp_manager.end_ramp(repository):
        console.print(f"[green]✓ Full enforcement enabled for {repository}[/green]")
    else:
        console.print(f"[red]No ramp found for {repository}[/red]")


# ============================================================================
# Self-Healing Commands
# ============================================================================


@cli.group()
def heal():
    """Self-healing code suggestions.
    
    Automatically detect and fix code issues with verified corrections.
    """
    pass


@heal.command("analyze")
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--verify", is_flag=True, help="Verify fixes with Z3 before suggesting")
@click.option("--format", "-f", "output_format",
              type=click.Choice(["rich", "json"]),
              default="rich", help="Output format")
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
    
    console.print(Panel.fit(
        "[bold blue]CodeVerify[/bold blue] - Self-Healing Analysis",
        subtitle="Verified Code Fixes"
    ))
    
    path_obj = Path(path)
    
    if path_obj.is_file():
        files = [path_obj]
    else:
        files = list(path_obj.rglob("*.py"))[:20]  # Limit for CLI
    
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
                fixes = asyncio.run(agent.analyze_and_suggest_fixes(
                    code,
                    str(file),
                    verify_fixes=verify,
                ))
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
def heal_apply(ctx: click.Context, path: str, apply_all: bool, category: str, min_confidence: float) -> None:
    """Apply self-healing fixes to code.
    
    Examples:
    
        codeverify heal apply src/module.py --all
        codeverify heal apply . --category null_safety --min-confidence 0.9
    """
    from codeverify_agents import SelfHealingAgent
    
    console.print("[bold]Scanning for fixable issues...[/bold]")
    
    path_obj = Path(path)
    agent = SelfHealingAgent()
    
    if path_obj.is_file():
        files = [path_obj]
    else:
        files = list(path_obj.rglob("*.py"))[:20]
    
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
                    console.print(f"\n[bold]{fix.get('category')}[/bold] in {file}:{fix.get('line')}")
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


# ============================================================================
# Offline Mode Commands
# ============================================================================


@cli.group()
def offline():
    """Offline/air-gapped mode commands.
    
    Run analysis without cloud connectivity using local models.
    """
    pass


@offline.command("status")
@click.pass_context
def offline_status(ctx: click.Context) -> None:
    """Check offline mode readiness.
    
    Shows availability of local components (Z3, Ollama, cached models).
    
    Examples:
    
        codeverify offline status
    """
    from codeverify_core import get_offline_manager
    
    console.print(Panel.fit(
        "[bold blue]CodeVerify[/bold blue] - Offline Mode Status",
    ))
    
    manager = get_offline_manager()
    status = asyncio.run(manager.check_offline_readiness())
    
    # Show readiness
    ready_icon = "✓" if status.get("ready") else "✗"
    ready_color = "green" if status.get("ready") else "red"
    console.print(f"[{ready_color}]{ready_icon} Overall Ready: {status.get('ready')}[/{ready_color}]")
    
    console.print("\n[bold]Components:[/bold]")
    
    # Z3
    z3_icon = "✓" if status.get("z3_available") else "✗"
    z3_color = "green" if status.get("z3_available") else "yellow"
    console.print(f"  [{z3_color}]{z3_icon} Z3 Solver[/{z3_color}]")
    
    # Ollama
    ollama_icon = "✓" if status.get("ollama_available") else "✗"
    ollama_color = "green" if status.get("ollama_available") else "yellow"
    console.print(f"  [{ollama_color}]{ollama_icon} Ollama LLM[/{ollama_color}]")
    
    # Models
    models = status.get("models_available", [])
    if models:
        console.print(f"\n[bold]Available Models:[/bold]")
        for model in models:
            console.print(f"  • {model}")
    else:
        console.print("\n[yellow]No local models cached[/yellow]")
        console.print("[dim]Run 'codeverify offline setup' to download models[/dim]")


@offline.command("setup")
@click.option("--model", default="codellama:7b-instruct", help="Model to download")
@click.option("--force", is_flag=True, help="Force re-download")
@click.pass_context
def offline_setup(ctx: click.Context, model: str, force: bool) -> None:
    """Setup offline mode (download models).
    
    Downloads required models for offline analysis.
    
    Examples:
    
        codeverify offline setup
        codeverify offline setup --model llama3.2:1b
    """
    from codeverify_core import get_offline_manager
    
    console.print(Panel.fit(
        "[bold blue]CodeVerify[/bold blue] - Offline Mode Setup",
    ))
    
    manager = get_offline_manager()
    
    console.print(f"[bold]Downloading model: {model}[/bold]")
    console.print("[dim]This may take several minutes...[/dim]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Downloading {model}...", total=None)
        
        try:
            result = asyncio.run(manager.download_model(model, force=force))
            
            if result.get("success"):
                progress.update(task, description=f"✓ Downloaded {model}")
                console.print(f"\n[green]✓ Model {model} ready for offline use[/green]")
            else:
                console.print(f"\n[red]✗ Failed to download: {result.get('error')}[/red]")
                
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            console.print("[dim]Make sure Ollama is installed and running[/dim]")


@offline.command("analyze")
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--format", "-f", "output_format",
              type=click.Choice(["rich", "json"]),
              default="rich", help="Output format")
@click.pass_context
def offline_analyze(ctx: click.Context, path: str, output_format: str) -> None:
    """Run offline analysis (no cloud required).
    
    Uses local Z3 and Ollama for analysis.
    
    Examples:
    
        codeverify offline analyze src/
        codeverify offline analyze file.py -f json
    """
    from codeverify_core import get_offline_manager
    
    verbose = ctx.obj.get("verbose", False)
    
    console.print(Panel.fit(
        "[bold blue]CodeVerify[/bold blue] - Offline Analysis",
        subtitle="Air-Gapped Mode"
    ))
    
    manager = get_offline_manager()
    
    # Check readiness
    status = asyncio.run(manager.check_offline_readiness())
    if not status.get("z3_available"):
        console.print("[yellow]⚠ Z3 not available - limited analysis[/yellow]")
    
    path_obj = Path(path)
    
    if path_obj.is_file():
        files = [path_obj]
    else:
        files = list(path_obj.rglob("*.py"))[:20]
    
    if not files:
        console.print("[yellow]No Python files found[/yellow]")
        return
    
    all_findings = []
    
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
                result = asyncio.run(manager.analyze_code_offline(
                    code,
                    language="python",
                    include_llm_analysis=status.get("ollama_available", False),
                ))
                
                for finding in result.findings:
                    finding["file"] = str(file)
                    all_findings.append(finding)
                
                progress.advance(task)
            except Exception as e:
                if verbose:
                    console.print(f"[dim]Error in {file}: {e}[/dim]")
    
    if output_format == "json":
        import json
        click.echo(json.dumps({
            "offline_mode": True,
            "findings": all_findings,
            "capabilities_used": result.capabilities_used if 'result' in dir() else [],
        }, indent=2, default=str))
        return
    
    console.print(f"\n[bold]Offline Analysis Complete[/bold]")
    console.print(f"[dim]Capabilities: Z3={status.get('z3_available')}, LLM={status.get('ollama_available')}[/dim]\n")
    
    if not all_findings:
        console.print("[green]✓ No issues found[/green]")
        return
    
    console.print(f"Found [bold]{len(all_findings)}[/bold] issues:\n")
    
    for finding in all_findings[:20]:  # Limit output
        severity = finding.get("severity", "medium")
        color = {"critical": "red bold", "high": "red", "medium": "yellow", "low": "blue"}.get(severity, "white")
        console.print(f"[{color}]• {finding.get('category', 'unknown')}[/{color}] - {finding.get('file')}:{finding.get('line', '?')}")
        console.print(f"  {finding.get('message', '')}")


# ============================================================================
# Proof Coverage Commands
# ============================================================================


@cli.group()
def coverage():
    """Proof coverage dashboard.
    
    Track verification coverage across your codebase.
    """
    pass


@coverage.command("show")
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--format", "-f", "output_format",
              type=click.Choice(["rich", "json"]),
              default="rich", help="Output format")
@click.pass_context
def coverage_show(ctx: click.Context, path: str, output_format: str) -> None:
    """Show proof coverage for code.
    
    Displays verification coverage metrics including line, function,
    and file coverage.
    
    Examples:
    
        codeverify coverage show src/
        codeverify coverage show . -f json
    """
    from codeverify_core import get_proof_coverage_dashboard, ProofCoverageCalculator
    
    verbose = ctx.obj.get("verbose", False)
    
    console.print(Panel.fit(
        "[bold blue]CodeVerify[/bold blue] - Proof Coverage",
        subtitle="Verification Metrics"
    ))
    
    path_obj = Path(path)
    calculator = ProofCoverageCalculator()
    
    if path_obj.is_file():
        files = [path_obj]
    else:
        files = list(path_obj.rglob("*.py"))[:50]
    
    if not files:
        console.print("[yellow]No Python files found[/yellow]")
        return
    
    # Calculate coverage for each file
    file_coverages = []
    total_lines = 0
    verified_lines = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Calculating coverage...", total=len(files))
        
        for file in files:
            try:
                content = file.read_text()
                # Get any existing verifications (would come from database in production)
                verifications = []  # Placeholder - would load from storage
                
                file_coverage = calculator.calculate_file_coverage(
                    str(file),
                    content,
                    verifications,
                )
                file_coverages.append(file_coverage)
                total_lines += file_coverage.total_lines
                verified_lines += file_coverage.verified_lines
                
                progress.advance(task)
            except Exception as e:
                if verbose:
                    console.print(f"[dim]Error in {file}: {e}[/dim]")
    
    # Calculate overall percentage
    overall_pct = (verified_lines / total_lines * 100) if total_lines > 0 else 0
    
    if output_format == "json":
        import json
        click.echo(json.dumps({
            "total_files": len(file_coverages),
            "total_lines": total_lines,
            "verified_lines": verified_lines,
            "overall_percentage": overall_pct,
            "files": [
                {
                    "path": fc.file_path,
                    "total_lines": fc.total_lines,
                    "verified_lines": fc.verified_lines,
                    "status": fc.status.value,
                }
                for fc in file_coverages
            ]
        }, indent=2))
        return
    
    # Rich output
    console.print(f"\n[bold]Overall Coverage: {overall_pct:.1f}%[/bold]")
    console.print(_coverage_bar(overall_pct))
    console.print(f"\nFiles: {len(file_coverages)} | Lines: {verified_lines}/{total_lines}\n")
    
    # Show file breakdown
    table = Table(title="File Coverage")
    table.add_column("File", style="cyan")
    table.add_column("Lines", justify="right")
    table.add_column("Verified", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Status")
    
    for fc in sorted(file_coverages, key=lambda x: x.verified_lines / max(x.total_lines, 1), reverse=True)[:15]:
        pct = (fc.verified_lines / fc.total_lines * 100) if fc.total_lines > 0 else 0
        
        if pct >= 80:
            pct_str = f"[green]{pct:.0f}%[/green]"
            status = "[green]✓[/green]"
        elif pct >= 50:
            pct_str = f"[yellow]{pct:.0f}%[/yellow]"
            status = "[yellow]~[/yellow]"
        else:
            pct_str = f"[red]{pct:.0f}%[/red]"
            status = "[red]✗[/red]"
        
        # Truncate long paths
        file_path = fc.file_path
        if len(file_path) > 40:
            file_path = "..." + file_path[-37:]
        
        table.add_row(
            file_path,
            str(fc.total_lines),
            str(fc.verified_lines),
            pct_str,
            status,
        )
    
    console.print(table)


@coverage.command("report")
@click.argument("repository", default=".")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--format", "-f", "output_format",
              type=click.Choice(["html", "json", "markdown"]),
              default="html", help="Report format")
@click.pass_context
def coverage_report(ctx: click.Context, repository: str, output: str, output_format: str) -> None:
    """Generate proof coverage report.
    
    Creates a detailed coverage report with trends and heatmaps.
    
    Examples:
    
        codeverify coverage report --output coverage.html
        codeverify coverage report -f json -o coverage.json
    """
    from codeverify_core import get_proof_coverage_dashboard
    
    console.print(Panel.fit(
        "[bold blue]CodeVerify[/bold blue] - Coverage Report",
    ))
    
    dashboard = get_proof_coverage_dashboard()
    
    # Generate report
    report = dashboard.export_report(repository, format=output_format)
    
    if output:
        output_path = Path(output)
        if isinstance(report, dict):
            import json
            output_path.write_text(json.dumps(report, indent=2, default=str))
        else:
            output_path.write_text(report)
        console.print(f"[green]✓ Report saved to {output}[/green]")
    else:
        if output_format == "json":
            import json
            click.echo(json.dumps(report, indent=2, default=str))
        else:
            click.echo(report)


@coverage.command("trends")
@click.argument("repository", default=".")
@click.option("--days", type=int, default=30, help="Number of days to show")
@click.pass_context
def coverage_trends(ctx: click.Context, repository: str, days: int) -> None:
    """Show coverage trends over time.
    
    Examples:
    
        codeverify coverage trends --days 90
    """
    from codeverify_core import get_proof_coverage_dashboard
    
    console.print(Panel.fit(
        "[bold blue]CodeVerify[/bold blue] - Coverage Trends",
    ))
    
    dashboard = get_proof_coverage_dashboard()
    trends = dashboard.get_trends(repository, days=days)
    
    if not trends:
        console.print("[yellow]No trend data available yet[/yellow]")
        console.print("[dim]Trends are calculated from verification history[/dim]")
        return
    
    console.print(f"\n[bold]Coverage over last {days} days:[/bold]\n")
    
    # Simple ASCII chart
    for trend in trends[-10:]:  # Show last 10 data points
        pct = trend.coverage_percentage
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        date_str = trend.date.strftime("%Y-%m-%d")
        console.print(f"{date_str} [{_pct_color(pct)}]{bar}[/] {pct:.1f}%")


def _coverage_bar(pct: float, width: int = 30) -> str:
    """Generate a coverage progress bar."""
    filled = int(pct / 100 * width)
    empty = width - filled
    color = _pct_color(pct)
    return f"[{color}]{'█' * filled}{'░' * empty}[/] {pct:.1f}%"


def _pct_color(pct: float) -> str:
    """Get color for percentage."""
    if pct >= 80:
        return "green"
    elif pct >= 50:
        return "yellow"
    return "red"


if __name__ == "__main__":
    cli()
