"""CodeVerify CLI - Generate commands."""

from __future__ import annotations

import os
from pathlib import Path

import click
from rich.panel import Panel
from rich.table import Table

from codeverify_cli.commands import console


@click.command("generate-tests")
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--framework",
    "-f",
    type=click.Choice(["pytest", "unittest", "jest", "vitest", "go"]),
    default=None,
    help="Test framework (auto-detected if not specified)",
)
@click.option("--output", "-o", type=click.Path(), help="Output file")
@click.pass_context
def generate_tests(
    ctx: click.Context, file: str, framework: str | None, output: str | None
) -> None:
    """Generate regression tests from verification counterexamples.

    Examples:

        codeverify generate-tests src/math.py
        codeverify generate-tests src/utils.ts -f jest
        codeverify generate-tests src/calc.py -o tests/test_calc.py
    """
    from codeverify_agents.test_generator import TestFramework, TestGeneratorAgent

    console.print(
        Panel.fit(
            "[bold blue]CodeVerify[/bold blue] - Test Generator", subtitle="Counterexample → Tests"
        )
    )

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


@click.command("attest")
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

    console.print(
        Panel.fit(
            "[bold blue]CodeVerify[/bold blue] - Proof Attestation",
            subtitle="Cryptographic Verification",
        )
    )

    # Get secret key from environment
    secret_key = os.environ.get("CODEVERIFY_ATTESTATION_KEY", "development-key")
    manager = ProofCarryingManager(secret_key=secret_key)

    if verify:
        # Verify existing attestation
        try:
            with open(target) as f:
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


@click.command("invariants")
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

    console.print(
        Panel.fit("[bold blue]CodeVerify[/bold blue] - NL Invariants", subtitle="English → Z3")
    )

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


@click.command("semantic-diff")
@click.argument("old_file", type=click.Path(exists=True))
@click.argument("new_file", type=click.Path(exists=True))
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["rich", "mermaid", "dot", "html"]),
    default="rich",
    help="Output format",
)
@click.option("--output", "-o", type=click.Path(), help="Output file")
@click.pass_context
def semantic_diff(
    ctx: click.Context, old_file: str, new_file: str, output_format: str, output: str | None
) -> None:
    """Visualize behavioral changes between two versions.

    Examples:

        codeverify semantic-diff old.py new.py
        codeverify semantic-diff v1.ts v2.ts -f mermaid
        codeverify semantic-diff main.py feature.py -f html -o diff.html
    """
    from codeverify_agents.semantic_diff import SemanticDiffAgent

    console.print(
        Panel.fit(
            "[bold blue]CodeVerify[/bold blue] - Semantic Diff", subtitle="Behavioral Analysis"
        )
    )

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
                    f"[{impact_color}]{change.impact}[/{impact_color}]",
                )

            console.print(table)
        else:
            console.print("[green]No behavioral changes detected[/green]")
