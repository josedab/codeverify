"""Organization Knowledge Graph - proof and pattern relationship tracking.

Maintains a graph of proofs, patterns, functions, files, teams, and
repositories.  Enables cross-project proof reuse, pattern discovery, and
team-level analytics by modelling relationships such as PROVES, CONTAINS,
DEPENDS_ON, SIMILAR_TO, and REUSES_PROOF.
"""

from __future__ import annotations

import hashlib
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

# =============================================================================
# Enums
# =============================================================================


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph."""

    PROOF = "proof"
    PATTERN = "pattern"
    FUNCTION = "function"
    FILE = "file"
    TEAM = "team"
    REPOSITORY = "repository"
    BUG_CLASS = "bug_class"


class EdgeType(str, Enum):
    """Types of edges in the knowledge graph."""

    PROVES = "proves"
    CONTAINS = "contains"
    DEPENDS_ON = "depends_on"
    AUTHORED_BY = "authored_by"
    SIMILAR_TO = "similar_to"
    FIXES = "fixes"
    REUSES_PROOF = "reuses_proof"


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class GraphNode:
    """A node in the knowledge graph."""

    id: str
    node_type: NodeType
    label: str
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class GraphEdge:
    """A directed edge in the knowledge graph."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeGraphConfig:
    """Configuration for the knowledge graph."""

    max_nodes: int = 100_000
    similarity_threshold: float = 0.7
    enable_cross_project: bool = True


# =============================================================================
# Knowledge Graph
# =============================================================================


class KnowledgeGraph:
    """In-memory directed graph of verification artefacts.

    Stores nodes (proofs, patterns, functions, …) and weighted directed
    edges.  Supports neighbour queries, similarity look-ups, subgraph
    extraction, and basic statistics.
    """

    def __init__(self, config: KnowledgeGraphConfig | None = None) -> None:
        """Initialise the knowledge graph."""
        self.config = config or KnowledgeGraphConfig()

        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []

        # Adjacency lists for fast look-ups
        self._outgoing: dict[str, list[GraphEdge]] = defaultdict(list)
        self._incoming: dict[str, list[GraphEdge]] = defaultdict(list)

        # Secondary indices
        self._nodes_by_type: dict[NodeType, list[str]] = defaultdict(list)
        self._pattern_hashes: dict[str, str] = {}  # hash → node_id

    # -- mutations -----------------------------------------------------------

    def add_node(self, node: GraphNode) -> str:
        """Add a node to the graph. Returns the node id."""
        if len(self._nodes) >= self.config.max_nodes:
            raise ValueError(f"Maximum node count ({self.config.max_nodes}) reached.")

        self._nodes[node.id] = node
        self._nodes_by_type[node.node_type].append(node.id)

        # Index pattern hash when present
        pattern_hash = node.properties.get("pattern_hash")
        if pattern_hash:
            self._pattern_hashes[pattern_hash] = node.id

        return node.id

    def add_edge(self, edge: GraphEdge) -> None:
        """Add a directed edge to the graph."""
        if edge.source_id not in self._nodes:
            raise KeyError(f"Source node {edge.source_id!r} not found.")
        if edge.target_id not in self._nodes:
            raise KeyError(f"Target node {edge.target_id!r} not found.")

        self._edges.append(edge)
        self._outgoing[edge.source_id].append(edge)
        self._incoming[edge.target_id].append(edge)

    # -- queries -------------------------------------------------------------

    def get_node(self, node_id: str) -> GraphNode | None:
        """Return the node with *node_id*, or ``None``."""
        return self._nodes.get(node_id)

    def get_neighbors(
        self,
        node_id: str,
        edge_type: EdgeType | None = None,
    ) -> list[GraphNode]:
        """Return neighbours reachable from *node_id*.

        Optionally filtered by *edge_type*.
        """
        result: list[GraphNode] = []
        for edge in self._outgoing.get(node_id, []):
            if edge_type is not None and edge.edge_type != edge_type:
                continue
            neighbor = self._nodes.get(edge.target_id)
            if neighbor is not None:
                result.append(neighbor)
        return result

    def find_similar_proofs(
        self,
        pattern_hash: str,
        threshold: float | None = None,
    ) -> list[tuple[GraphNode, float]]:
        """Find proof nodes similar to *pattern_hash*.

        Similarity is computed as the ratio of shared hex-digit prefixes.
        Returns ``(node, similarity)`` pairs above *threshold*, sorted
        descending by similarity.
        """
        if threshold is None:
            threshold = self.config.similarity_threshold

        results: list[tuple[GraphNode, float]] = []
        proof_ids = self._nodes_by_type.get(NodeType.PROOF, [])

        for nid in proof_ids:
            node = self._nodes[nid]
            node_hash = node.properties.get("pattern_hash", "")
            if not node_hash:
                continue

            sim = _hash_similarity(pattern_hash, node_hash)
            if sim >= threshold:
                results.append((node, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def find_patterns_for_team(self, team_id: str) -> list[GraphNode]:
        """Return all pattern nodes authored by *team_id*."""
        team_node = self._nodes.get(team_id)
        if team_node is None:
            return []

        patterns: list[GraphNode] = []
        for edge in self._incoming.get(team_id, []):
            if edge.edge_type == EdgeType.AUTHORED_BY:
                source = self._nodes.get(edge.source_id)
                if source is not None and source.node_type == NodeType.PATTERN:
                    patterns.append(source)
        return patterns

    def get_subgraph(
        self,
        center_node_id: str,
        depth: int = 2,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Extract a subgraph around *center_node_id* up to *depth* hops."""
        visited_ids: set[str] = set()
        edge_set: list[GraphEdge] = []
        frontier: set[str] = {center_node_id}

        for _ in range(depth):
            next_frontier: set[str] = set()
            for nid in frontier:
                if nid in visited_ids:
                    continue
                visited_ids.add(nid)
                for edge in self._outgoing.get(nid, []):
                    edge_set.append(edge)
                    next_frontier.add(edge.target_id)
                for edge in self._incoming.get(nid, []):
                    edge_set.append(edge)
                    next_frontier.add(edge.source_id)
            frontier = next_frontier - visited_ids

        # Include last-frontier nodes
        visited_ids.update(frontier)

        nodes = [self._nodes[nid] for nid in visited_ids if nid in self._nodes]
        return nodes, edge_set

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics about the graph."""
        type_counts: dict[str, int] = {}
        for nt, ids in self._nodes_by_type.items():
            type_counts[nt.value] = len(ids)

        edge_type_counts: dict[str, int] = defaultdict(int)
        for edge in self._edges:
            edge_type_counts[edge.edge_type.value] += 1

        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "node_type_counts": type_counts,
            "edge_type_counts": dict(edge_type_counts),
            "pattern_hashes_indexed": len(self._pattern_hashes),
        }


# =============================================================================
# Proof Reuse Engine
# =============================================================================


class ProofReuseEngine:
    """Suggests and tracks reuse of existing proofs for new functions.

    Uses the knowledge graph to find proofs whose pattern hash is similar
    to new code, and records whether reuse succeeded.
    """

    def __init__(self, graph: KnowledgeGraph) -> None:
        """Initialise with an existing knowledge graph."""
        self._graph = graph
        self._usage_log: list[dict[str, Any]] = []

    def suggest_proof(
        self,
        code_hash: str,
        function_signature: str,
    ) -> list[GraphNode]:
        """Suggest existing proofs that may apply to *code_hash*.

        Combines pattern-hash similarity with a simple signature keyword
        heuristic to rank suggestions.
        """
        candidates = self._graph.find_similar_proofs(code_hash)
        if not candidates:
            return []

        sig_tokens = set(function_signature.lower().replace("(", " ").replace(")", " ").split())

        ranked: list[tuple[GraphNode, float]] = []
        for node, sim in candidates:
            label_tokens = set(node.label.lower().split())
            keyword_bonus = len(sig_tokens & label_tokens) * 0.05
            ranked.append((node, sim + keyword_bonus))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in ranked]

    def record_proof_usage(
        self,
        proof_id: str,
        function_id: str,
        success: bool,
    ) -> None:
        """Record that *proof_id* was reused for *function_id*."""
        self._usage_log.append(
            {
                "proof_id": proof_id,
                "function_id": function_id,
                "success": success,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        # Add a REUSES_PROOF edge on success
        if success and self._graph.get_node(proof_id) and self._graph.get_node(function_id):
            edge = GraphEdge(
                source_id=function_id,
                target_id=proof_id,
                edge_type=EdgeType.REUSES_PROOF,
                weight=1.0,
                properties={"recorded_at": datetime.now(UTC).isoformat()},
            )
            self._graph.add_edge(edge)

    def get_reuse_stats(self) -> dict[str, Any]:
        """Return statistics about proof reuse."""
        total = len(self._usage_log)
        successes = sum(1 for entry in self._usage_log if entry["success"])
        return {
            "total_reuse_attempts": total,
            "successful_reuses": successes,
            "reuse_success_rate": successes / max(total, 1),
        }


# =============================================================================
# Knowledge Ingester
# =============================================================================


class KnowledgeIngester:
    """Ingests verification results into the knowledge graph.

    Creates nodes for repositories, files, functions, proofs, and the
    corresponding edges.
    """

    def __init__(self, graph: KnowledgeGraph) -> None:
        """Initialise with an existing knowledge graph."""
        self._graph = graph

    def ingest_verification_result(
        self,
        repo_id: str,
        file_path: str,
        function_name: str,
        findings: list[dict[str, Any]],
        proofs: list[dict[str, Any]],
    ) -> list[str]:
        """Ingest a verification result and return created node ids.

        Creates (or reuses) nodes for the repository, file, function,
        each finding's bug class, and each proof.  Connects them with
        appropriate edges.
        """
        created_ids: list[str] = []

        # -- repository node --------------------------------------------------
        repo_node = self._graph.get_node(repo_id)
        if repo_node is None:
            repo_node = GraphNode(
                id=repo_id,
                node_type=NodeType.REPOSITORY,
                label=repo_id,
            )
            self._graph.add_node(repo_node)
            created_ids.append(repo_id)

        # -- file node --------------------------------------------------------
        file_id = _deterministic_id(f"file:{repo_id}:{file_path}")
        if self._graph.get_node(file_id) is None:
            file_node = GraphNode(
                id=file_id,
                node_type=NodeType.FILE,
                label=file_path,
                properties={"repo_id": repo_id},
            )
            self._graph.add_node(file_node)
            created_ids.append(file_id)
            self._graph.add_edge(
                GraphEdge(
                    source_id=repo_id,
                    target_id=file_id,
                    edge_type=EdgeType.CONTAINS,
                )
            )

        # -- function node ----------------------------------------------------
        func_id = _deterministic_id(f"func:{repo_id}:{file_path}:{function_name}")
        if self._graph.get_node(func_id) is None:
            func_node = GraphNode(
                id=func_id,
                node_type=NodeType.FUNCTION,
                label=function_name,
                properties={"file_path": file_path, "repo_id": repo_id},
            )
            self._graph.add_node(func_node)
            created_ids.append(func_id)
            self._graph.add_edge(
                GraphEdge(
                    source_id=file_id,
                    target_id=func_id,
                    edge_type=EdgeType.CONTAINS,
                )
            )

        # -- findings (bug-class nodes) ---------------------------------------
        for finding in findings:
            category = finding.get("category", "unknown")
            bc_id = _deterministic_id(f"bugclass:{category}")
            if self._graph.get_node(bc_id) is None:
                bc_node = GraphNode(
                    id=bc_id,
                    node_type=NodeType.BUG_CLASS,
                    label=category,
                    properties={"description": finding.get("message", "")},
                )
                self._graph.add_node(bc_node)
                created_ids.append(bc_id)

            self._graph.add_edge(
                GraphEdge(
                    source_id=func_id,
                    target_id=bc_id,
                    edge_type=EdgeType.DEPENDS_ON,
                    properties={"severity": finding.get("severity", "info")},
                )
            )

        # -- proof nodes ------------------------------------------------------
        for proof in proofs:
            proof_id = str(uuid.uuid4())
            pattern_hash = proof.get(
                "pattern_hash",
                hashlib.sha256(f"{function_name}:{proof}".encode()).hexdigest(),
            )
            proof_node = GraphNode(
                id=proof_id,
                node_type=NodeType.PROOF,
                label=proof.get("label", f"proof-{function_name}"),
                properties={
                    "pattern_hash": pattern_hash,
                    "status": proof.get("status", "verified"),
                    "function": function_name,
                    "file_path": file_path,
                },
            )
            self._graph.add_node(proof_node)
            created_ids.append(proof_id)

            self._graph.add_edge(
                GraphEdge(
                    source_id=proof_id,
                    target_id=func_id,
                    edge_type=EdgeType.PROVES,
                )
            )

        return created_ids


# =============================================================================
# Helpers
# =============================================================================


def _hash_similarity(a: str, b: str) -> float:
    """Compute a simple similarity between two hex-digest strings.

    Uses the ratio of matching characters in the shorter prefix.
    """
    min_len = min(len(a), len(b))
    if min_len == 0:
        return 0.0
    matches = sum(1 for x, y in zip(a, b, strict=False) if x == y)
    return matches / min_len


def _deterministic_id(raw: str) -> str:
    """Create a deterministic UUID-style id from a raw string."""
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


# =============================================================================
# Module Singletons
# =============================================================================


_knowledge_graph: KnowledgeGraph | None = None


def get_knowledge_graph(
    config: KnowledgeGraphConfig | None = None,
) -> KnowledgeGraph:
    """Get the global knowledge graph instance."""
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = KnowledgeGraph(config=config)
    return _knowledge_graph


def reset_knowledge_graph() -> None:
    """Reset the global knowledge graph (mainly for testing)."""
    global _knowledge_graph
    _knowledge_graph = None
