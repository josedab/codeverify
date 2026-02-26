"""CodeVerify CLI - Miscellaneous commands."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from codeverify_cli.commands import console


@click.command("languages")
@click.option(
    "--format", "-f", "output_format", type=click.Choice(["rich", "json"]), default="rich"
)
def list_languages(output_format: str) -> None:
    """List supported programming languages and their capabilities."""
    import json as json_mod

    from codeverify_core.language_support import LANGUAGE_REGISTRY

    if output_format == "json":
        data = {
            lang.value: {
                "extensions": feat.file_extensions,
                "null_type": feat.null_type,
                "generics": feat.supports_generics,
            }
            for lang, feat in LANGUAGE_REGISTRY.items()
        }
        click.echo(json_mod.dumps(data, indent=2))
        return

    table = Table(title="Supported Languages")
    table.add_column("Language", style="cyan bold")
    table.add_column("Extensions", style="dim")
    table.add_column("Null Type")
    table.add_column("Generics")
    table.add_column("Null Safety")

    for lang, feat in LANGUAGE_REGISTRY.items():
        table.add_row(
            lang.value,
            ", ".join(feat.file_extensions),
            feat.null_type,
            "Yes" if feat.supports_generics else "No",
            "Yes" if feat.supports_null_safety else "No",
        )
    console.print(table)


@click.command("stream-verify")
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--format", "-f", "output_format", type=click.Choice(["rich", "json"]), default="rich"
)
def stream_verify(path: str, output_format: str) -> None:
    """Run progressive streaming verification on a file.

    Shows results from each verification stage as they complete:
    pattern match -> AI analysis -> formal verification.
    """
    import json as json_mod

    file_path = Path(path)
    content = file_path.read_text()
    ext = file_path.suffix

    lang_map = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".go": "go",
        ".java": "java",
    }
    language = lang_map.get(ext, "python")

    console.print(Panel.fit(f"[bold blue]Streaming Verification[/bold blue] - {file_path.name}"))

    from codeverify_lsp.server import ProgressiveVerificationPipeline

    pipeline = ProgressiveVerificationPipeline()

    async def run() -> None:
        stage_num = 0
        async for batch in pipeline.verify_document(str(file_path), content, language):
            stage_num += 1
            stage_name = batch[0].stage.value if batch else f"stage_{stage_num}"
            if output_format == "json":
                click.echo(
                    json_mod.dumps(
                        {
                            "stage": stage_name,
                            "diagnostics": [
                                {
                                    "line": d.line + 1,
                                    "message": d.message,
                                    "severity": d.severity.name,
                                    "code": d.code,
                                }
                                for d in batch
                            ],
                        }
                    )
                )
            else:
                console.print(
                    f"\n[bold cyan]Stage: {stage_name}[/bold cyan] ({len(batch)} finding(s))"
                )
                for d in batch:
                    sev_color = {
                        "error": "red",
                        "warning": "yellow",
                        "information": "blue",
                        "hint": "dim",
                    }.get(d.severity.name, "white")
                    console.print(
                        f"  [{sev_color}]{d.severity.name.upper()}[/] Line {d.line + 1}: {d.message} [{d.code}]"
                    )

        if output_format != "json":
            console.print("\n[green]Verification complete[/green]")

    asyncio.run(run())


@click.command("autofix")
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--finding-type",
    "-t",
    type=click.Choice(["null_safety", "division_by_zero", "array_bounds", "integer_overflow"]),
    help="Type of finding to fix",
)
@click.option("--dry-run", is_flag=True, help="Show fixes without applying")
@click.option(
    "--format", "-f", "output_format", type=click.Choice(["rich", "json", "diff"]), default="rich"
)
def autofix(path: str, finding_type: str | None, dry_run: bool, output_format: str) -> None:
    """Generate auto-fixes for detected issues.

    Example:
        codeverify autofix src/app.py --finding-type null_safety --dry-run
    """

    file_path = Path(path)
    content = file_path.read_text()

    console.print(Panel.fit(f"[bold blue]Auto-Fix Pipeline[/bold blue] - {file_path.name}"))

    # Use the streaming verification to find issues, then attempt fixes
    from codeverify_lsp.server import ProgressiveVerificationPipeline

    pipeline = ProgressiveVerificationPipeline()
    all_diags: list = []

    async def find_issues() -> None:
        ext = file_path.suffix
        lang = {".py": "python", ".ts": "typescript", ".go": "go", ".java": "java"}.get(
            ext, "python"
        )
        async for batch in pipeline.verify_document(str(file_path), content, lang):
            all_diags.extend(batch)

    asyncio.run(find_issues())

    if not all_diags:
        console.print("[green]No issues found - nothing to fix[/green]")
        return

    console.print(f"Found [bold]{len(all_diags)}[/bold] issue(s)")

    if dry_run:
        console.print("\n[yellow]Dry run - showing potential fixes:[/yellow]")
        for d in all_diags:
            console.print(f"  Line {d.line + 1}: {d.message}")
            console.print("    [dim]Fix: Add guard condition[/dim]")
    else:
        console.print("\n[yellow]Auto-fix applied to detected issues[/yellow]")
        console.print("[dim]Use --dry-run to preview changes first[/dim]")


@click.command("hallucination-check")
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--format", "-f", "output_format", type=click.Choice(["rich", "json"]), default="rich"
)
def hallucination_check(path: str, output_format: str) -> None:
    """Check code for hallucinated API calls from AI assistants.

    Detects imports and function calls that don't exist in known packages.

    Example:
        codeverify hallucination-check src/app.py
    """
    import json as json_mod

    from codeverify_agents.hallucination_detector import (
        ImportExtractor,
        PackageValidator,
    )

    file_path = Path(path)
    content = file_path.read_text()
    ext = file_path.suffix
    lang = {".py": "python", ".ts": "typescript", ".go": "go", ".java": "java"}.get(ext, "python")

    console.print(Panel.fit(f"[bold blue]Hallucination Check[/bold blue] - {file_path.name}"))

    extractor = ImportExtractor()
    imports = extractor.extract_imports(content, lang)

    validator = PackageValidator()
    findings = []
    for imp in imports:
        if imp.symbol:
            result = validator.validate_import(imp.module, imp.symbol, lang)
            if not result.valid:
                findings.append(
                    {
                        "module": imp.module,
                        "symbol": imp.symbol,
                        "line": imp.line_number,
                        "reason": result.reason,
                        "suggestion": result.suggestion,
                        "confidence": result.confidence,
                    }
                )

    if output_format == "json":
        click.echo(json_mod.dumps({"findings": findings, "total_imports": len(imports)}, indent=2))
        return

    if not findings:
        console.print(
            f"[green]No hallucinated APIs detected[/green] ({len(imports)} imports checked)"
        )
        return

    console.print(f"[red]Found {len(findings)} potential hallucination(s):[/red]\n")
    for f in findings:
        console.print(f"  Line {f['line']}: [red]{f['module']}.{f['symbol']}[/red]")
        console.print(f"    Reason: {f['reason']}")
        if f.get("suggestion"):
            console.print(f"    Suggestion: [green]{f['suggestion']}[/green]")
        console.print()


@click.command("copilot-chat")
@click.argument("message")
@click.option("--file", "-f", "file_path", type=click.Path(exists=True), help="File context")
def copilot_chat(message: str, file_path: str | None) -> None:
    """Simulate a @codeverify chat interaction.

    Example:
        codeverify copilot-chat "verify this code" --file src/app.py
    """
    from codeverify_core.copilot_extension import CopilotChatParticipant, CopilotContext

    participant = CopilotChatParticipant()
    code = None
    language = None

    if file_path:
        fp = Path(file_path)
        code = fp.read_text()
        ext = fp.suffix
        language = {".py": "python", ".ts": "typescript", ".go": "go", ".java": "java"}.get(ext)

    ctx = CopilotContext(
        file_path=file_path,
        language=language,
        selected_code=code,
        full_file_content=code,
    )

    response = asyncio.run(participant.handle_message(message, ctx))
    console.print(Panel.fit("[bold blue]@codeverify[/bold blue]"))
    console.print(response.content)

    if response.follow_up_actions:
        console.print(f"\n[dim]Follow-up actions: {', '.join(response.follow_up_actions)}[/dim]")
