"""Output formatting for CLI."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax


def format_findings(console: Console, results: Any, show_fix: bool = False) -> None:
    """Format and display findings."""
    if not results.findings:
        console.print("\n[green]✓ No issues found![/green]")
        return
    
    severity_emoji = {
        "critical": "🔴",
        "high": "🟠",
        "medium": "🟡",
        "low": "🔵",
        "info": "⚪",
    }
    
    severity_style = {
        "critical": "bold red",
        "high": "red",
        "medium": "yellow",
        "low": "blue",
        "info": "dim",
    }
    
    console.print(f"\n[bold]Found {len(results.findings)} issue(s):[/bold]\n")
    
    # Group by file
    by_file: dict[str, list[dict[str, Any]]] = {}
    for finding in results.findings:
        file_path = finding.get("file_path", "unknown")
        if file_path not in by_file:
            by_file[file_path] = []
        by_file[file_path].append(finding)
    
    for file_path, findings in by_file.items():
        console.print(f"[bold]{file_path}[/bold]")
        
        for finding in findings:
            severity = finding.get("severity", "low")
            emoji = severity_emoji.get(severity, "⚪")
            style = severity_style.get(severity, "")
            
            line = finding.get("line_start", "?")
            title = finding.get("title", "Issue")
            
            console.print(f"  {emoji} [{style}]{title}[/{style}] (line {line})")
            
            if description := finding.get("description"):
                console.print(f"     [dim]{description[:100]}[/dim]")
            
            if show_fix and (fix := finding.get("fix_suggestion")):
                console.print("     [bold green]Suggested fix:[/bold green]")
                # Detect language from file extension
                lang = "python"
                if file_path.endswith((".ts", ".tsx", ".js", ".jsx")):
                    lang = "typescript"
                elif file_path.endswith(".go"):
                    lang = "go"
                elif file_path.endswith(".java"):
                    lang = "java"
                
                syntax = Syntax(fix, lang, theme="monokai", line_numbers=False)
                console.print(syntax)
            
            console.print()
        
        console.print()


def format_summary(console: Console, results: Any) -> None:
    """Format and display analysis summary."""
    summary = results.to_dict().get("summary", {})
    
    table = Table(title="Analysis Summary", show_header=True, header_style="bold")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")
    
    table.add_row("Files Analyzed", str(results.files_analyzed))
    table.add_row("Functions Found", str(results.functions_found))
    table.add_row("Classes Found", str(results.classes_found))
    table.add_row("Duration", f"{results.duration_ms:.0f}ms")
    table.add_row("", "")
    table.add_row("[red]Critical[/red]", str(summary.get("critical", 0)))
    table.add_row("[red]High[/red]", str(summary.get("high", 0)))
    table.add_row("[yellow]Medium[/yellow]", str(summary.get("medium", 0)))
    table.add_row("[blue]Low[/blue]", str(summary.get("low", 0)))
    table.add_row("[bold]Total[/bold]", f"[bold]{summary.get('total', 0)}[/bold]")
    
    console.print(table)
    
    # Pass/fail status
    critical = summary.get("critical", 0)
    high = summary.get("high", 0)
    
    if critical > 0 or high > 0:
        console.print("\n[red]❌ Analysis failed - critical/high issues found[/red]")
    else:
        console.print("\n[green]✓ Analysis passed[/green]")


def format_diff(console: Console, original: str, fixed: str, file_path: str) -> None:
    """Format a diff between original and fixed code."""
    import difflib
    
    original_lines = original.splitlines(keepends=True)
    fixed_lines = fixed.splitlines(keepends=True)
    
    diff = difflib.unified_diff(
        original_lines,
        fixed_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
    )
    
    diff_text = "".join(diff)
    
    if diff_text:
        syntax = Syntax(diff_text, "diff", theme="monokai")
        console.print(syntax)
    else:
        console.print("[dim]No changes[/dim]")
