"""CodeVerify CLI - Knowledge graph commands."""

from __future__ import annotations

import click
from rich.table import Table

from codeverify_cli.commands import console


@click.group("knowledge-graph")
def knowledge_graph_cli() -> None:
    """Organization knowledge graph operations."""


@knowledge_graph_cli.command("stats")
@click.option(
    "--format", "-f", "output_format", type=click.Choice(["rich", "json"]), default="rich"
)
def kg_stats(output_format: str) -> None:
    """Show knowledge graph statistics."""
    from codeverify_core.knowledge_graph import get_knowledge_graph

    graph = get_knowledge_graph()
    stats = graph.get_stats()

    if output_format == "json":
        import json

        console.print(json.dumps(stats, indent=2))
        return

    table = Table(title="Knowledge Graph Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Nodes", str(stats["total_nodes"]))
    table.add_row("Total Edges", str(stats["total_edges"]))
    table.add_row("Max Nodes", str(stats["max_nodes"]))
    for ntype, count in stats.get("node_types", {}).items():
        table.add_row(f"  {ntype} nodes", str(count))
    for etype, count in stats.get("edge_types", {}).items():
        table.add_row(f"  {etype} edges", str(count))
    console.print(table)


@knowledge_graph_cli.command("search")
@click.argument("pattern_hash")
@click.option("--threshold", "-t", type=float, default=0.7, help="Similarity threshold")
def kg_search(pattern_hash: str, threshold: float) -> None:
    """Search for similar proofs in the knowledge graph."""
    from codeverify_core.knowledge_graph import get_knowledge_graph

    graph = get_knowledge_graph()
    results = graph.find_similar_proofs(pattern_hash, threshold)

    if not results:
        console.print("[yellow]No similar proofs found[/yellow]")
        return

    table = Table(title=f"Similar Proofs (threshold={threshold})")
    table.add_column("Proof ID", style="cyan")
    table.add_column("Label")
    table.add_column("Similarity", style="green")
    for node, score in results:
        table.add_row(node.id, node.label, f"{score:.2%}")
    console.print(table)
