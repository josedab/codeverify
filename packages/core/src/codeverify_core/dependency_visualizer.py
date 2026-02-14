"""Cross-Repo Dependency Visualizer.

Interactive graph generation, query API, and visualization export for
multi-repository dependency analysis.  Supports DOT, Mermaid, D3 JSON,
and Cytoscape export formats with multiple layout algorithms.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()

# ── Enums ────────────────────────────────────────────────────────────────────


class NodeType(str, Enum):
    """Classification of dependency graph nodes."""

    REPOSITORY = "repository"
    PACKAGE = "package"
    MODULE = "module"
    FUNCTION = "function"
    CLASS = "class"
    INTERFACE = "interface"


class EdgeType(str, Enum):
    """Classification of dependency graph edges."""

    IMPORTS = "imports"
    DEPENDS_ON = "depends_on"
    EXTENDS = "extends"
    IMPLEMENTS = "implements"
    CALLS = "calls"
    PUBLISHES = "publishes"
    CONSUMES = "consumes"


class LayoutAlgorithm(str, Enum):
    """Graph layout algorithms for positioning nodes."""

    FORCE_DIRECTED = "force_directed"
    HIERARCHICAL = "hierarchical"
    CIRCULAR = "circular"
    TREE = "tree"
    GRID = "grid"


class ExportFormat(str, Enum):
    """Supported export formats for dependency graphs."""

    DOT = "dot"
    MERMAID = "mermaid"
    D3_JSON = "d3_json"
    SVG_DATA = "svg_data"
    CYTOSCAPE = "cytoscape"


# ── Data Models ──────────────────────────────────────────────────────────────


@dataclass
class GraphNode:
    """A node in the dependency graph."""

    id: str
    name: str
    node_type: NodeType
    repo: str | None = None
    version: str | None = None
    health_score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    position_x: float = 0.0
    position_y: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise the node to a plain dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "node_type": self.node_type.value,
            "repo": self.repo,
            "version": self.version,
            "health_score": self.health_score,
            "metadata": self.metadata,
            "position": {"x": self.position_x, "y": self.position_y},
        }


@dataclass
class GraphEdge:
    """A directed edge in the dependency graph."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the edge to a plain dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "label": self.label,
            "metadata": self.metadata,
        }


@dataclass
class DependencyGraph:
    """Container for a full dependency graph."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the graph to a plain dictionary."""
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "metadata": self.metadata,
        }

    def get_node(self, node_id: str) -> GraphNode | None:
        """Look up a node by its ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None


@dataclass
class GraphQuery:
    """Parameters for querying a dependency graph."""

    root_node_id: str | None = None
    max_depth: int = 3
    node_types: list[NodeType] = field(default_factory=list)
    edge_types: list[EdgeType] = field(default_factory=list)
    include_transitive: bool = True
    filter_health_below: float | None = None


@dataclass
class ImpactPath:
    """A dependency path between two nodes with risk scoring."""

    source: str
    target: str
    path: list[str] = field(default_factory=list)
    edge_types: list[EdgeType] = field(default_factory=list)
    total_weight: float = 0.0
    risk_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise the impact path to a plain dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "path": self.path,
            "edge_types": [et.value for et in self.edge_types],
            "total_weight": self.total_weight,
            "risk_score": self.risk_score,
        }


@dataclass
class ClusterInfo:
    """Information about a cluster of tightly-coupled nodes."""

    id: str
    name: str
    node_ids: list[str] = field(default_factory=list)
    interconnectedness: float = 0.0
    external_deps: int = 0
    internal_deps: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise the cluster info to a plain dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "node_ids": self.node_ids,
            "interconnectedness": self.interconnectedness,
            "external_deps": self.external_deps,
            "internal_deps": self.internal_deps,
        }


@dataclass
class DependencyReport:
    """Full dependency analysis report."""

    total_nodes: int = 0
    total_edges: int = 0
    max_depth: int = 0
    orphan_nodes: list[str] = field(default_factory=list)
    circular_deps: list[list[str]] = field(default_factory=list)
    clusters: list[ClusterInfo] = field(default_factory=list)
    critical_paths: list[ImpactPath] = field(default_factory=list)
    health_summary: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a plain dictionary."""
        return {
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "max_depth": self.max_depth,
            "orphan_nodes": self.orphan_nodes,
            "circular_deps": self.circular_deps,
            "clusters": [c.to_dict() for c in self.clusters],
            "critical_paths": [p.to_dict() for p in self.critical_paths],
            "health_summary": self.health_summary,
        }


# ── Graph Builder ────────────────────────────────────────────────────────────


class GraphBuilder:
    """Build dependency graphs from repository analysis."""

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []

    # -- mutations -----------------------------------------------------------

    def add_node(
        self,
        name: str,
        node_type: NodeType,
        repo: str | None = None,
        **kwargs: Any,
    ) -> GraphNode:
        """Create and register a graph node, returning it."""
        node_id = kwargs.pop("id", None) or str(uuid.uuid4())
        node = GraphNode(
            id=node_id,
            name=name,
            node_type=node_type,
            repo=repo,
            **kwargs,
        )
        self._nodes[node.id] = node
        logger.debug("node_added", node_id=node.id, name=name, node_type=node_type.value)
        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        **kwargs: Any,
    ) -> GraphEdge:
        """Create and register a graph edge, returning it."""
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            **kwargs,
        )
        self._edges.append(edge)
        logger.debug(
            "edge_added",
            source=source_id,
            target=target_id,
            edge_type=edge_type.value,
        )
        return edge

    # -- manifest parsers ----------------------------------------------------

    def from_package_json(self, package_data: dict, repo_name: str) -> DependencyGraph:
        """Build a dependency graph from parsed ``package.json`` data."""
        pkg_name = package_data.get("name", repo_name)
        root = self.add_node(pkg_name, NodeType.PACKAGE, repo=repo_name)

        section_map: dict[str, EdgeType] = {
            "dependencies": EdgeType.DEPENDS_ON,
            "devDependencies": EdgeType.DEPENDS_ON,
            "peerDependencies": EdgeType.DEPENDS_ON,
        }
        for section, edge_type in section_map.items():
            for dep_name, dep_version in package_data.get(section, {}).items():
                dep_node = self.add_node(
                    dep_name,
                    NodeType.PACKAGE,
                    version=str(dep_version),
                )
                self.add_edge(root.id, dep_node.id, edge_type, label=section)

        logger.info("graph_from_package_json", repo=repo_name, nodes=len(self._nodes))
        return self.build()

    def from_requirements(self, requirements_text: str, repo_name: str) -> DependencyGraph:
        """Build a dependency graph from a ``requirements.txt`` file."""
        root = self.add_node(repo_name, NodeType.REPOSITORY, repo=repo_name)

        for line in requirements_text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            match = re.match(r"^([A-Za-z0-9_.-]+)\s*([><=!~].*)?$", line)
            if match:
                dep_name = match.group(1)
                dep_version = (match.group(2) or "").strip()
                dep_node = self.add_node(
                    dep_name,
                    NodeType.PACKAGE,
                    version=dep_version or None,
                )
                self.add_edge(root.id, dep_node.id, EdgeType.DEPENDS_ON)

        logger.info("graph_from_requirements", repo=repo_name, nodes=len(self._nodes))
        return self.build()

    def from_import_analysis(self, imports: list[dict]) -> DependencyGraph:
        """Build a graph from import analysis results.

        Each entry in *imports* should have ``source``, ``target``, and
        optionally ``edge_type`` (defaults to ``IMPORTS``).
        """
        seen: set[str] = set()
        for imp in imports:
            source_name = imp.get("source", "")
            target_name = imp.get("target", "")
            if not source_name or not target_name:
                continue

            if source_name not in seen:
                self.add_node(source_name, NodeType.MODULE, id=source_name)
                seen.add(source_name)
            if target_name not in seen:
                self.add_node(target_name, NodeType.MODULE, id=target_name)
                seen.add(target_name)

            edge_type_str = imp.get("edge_type", "imports")
            try:
                edge_type = EdgeType(edge_type_str)
            except ValueError:
                edge_type = EdgeType.IMPORTS

            self.add_edge(source_name, target_name, edge_type)

        logger.info("graph_from_imports", total_imports=len(imports))
        return self.build()

    # -- build ---------------------------------------------------------------

    def build(self) -> DependencyGraph:
        """Return the assembled :class:`DependencyGraph`."""
        graph = DependencyGraph(
            nodes=list(self._nodes.values()),
            edges=list(self._edges),
            metadata={"node_count": len(self._nodes), "edge_count": len(self._edges)},
        )
        logger.info("graph_built", nodes=len(graph.nodes), edges=len(graph.edges))
        return graph

    def merge_graphs(self, graphs: list[DependencyGraph]) -> DependencyGraph:
        """Merge multiple dependency graphs into a single unified graph."""
        merged_nodes: dict[str, GraphNode] = {}
        merged_edges: list[GraphEdge] = []

        for graph in graphs:
            for node in graph.nodes:
                if node.id not in merged_nodes:
                    merged_nodes[node.id] = node
            merged_edges.extend(graph.edges)

        # Deduplicate edges
        seen_edges: set[tuple[str, str, str]] = set()
        unique_edges: list[GraphEdge] = []
        for edge in merged_edges:
            key = (edge.source_id, edge.target_id, edge.edge_type.value)
            if key not in seen_edges:
                seen_edges.add(key)
                unique_edges.append(edge)

        result = DependencyGraph(
            nodes=list(merged_nodes.values()),
            edges=unique_edges,
            metadata={
                "merged_from": len(graphs),
                "node_count": len(merged_nodes),
                "edge_count": len(unique_edges),
            },
        )
        logger.info(
            "graphs_merged",
            source_graphs=len(graphs),
            nodes=len(result.nodes),
            edges=len(result.edges),
        )
        return result


# ── Graph Analyzer ───────────────────────────────────────────────────────────


class GraphAnalyzer:
    """Analyze dependency graph for patterns, cycles, and risks."""

    def __init__(self) -> None:
        self._adjacency: dict[str, list[tuple[str, EdgeType, float]]] = defaultdict(list)
        self._reverse_adj: dict[str, list[tuple[str, EdgeType, float]]] = defaultdict(list)

    # -- helpers -------------------------------------------------------------

    def _build_adjacency(self, graph: DependencyGraph) -> None:
        """Populate adjacency maps from *graph*."""
        self._adjacency.clear()
        self._reverse_adj.clear()
        for edge in graph.edges:
            self._adjacency[edge.source_id].append(
                (edge.target_id, edge.edge_type, edge.weight)
            )
            self._reverse_adj[edge.target_id].append(
                (edge.source_id, edge.edge_type, edge.weight)
            )

    # -- cycle detection (Tarjan's SCC) --------------------------------------

    def find_cycles(self, graph: DependencyGraph) -> list[list[str]]:
        """Detect circular dependencies using Tarjan's algorithm.

        Returns a list of strongly-connected components with size > 1.
        """
        self._build_adjacency(graph)
        sccs = self._tarjan_scc(graph)
        cycles = [scc for scc in sccs if len(scc) > 1]
        logger.info("cycles_detected", count=len(cycles))
        return cycles

    def _tarjan_scc(self, graph: DependencyGraph) -> list[list[str]]:
        """Tarjan's strongly-connected components algorithm."""
        index_counter = [0]
        stack: list[str] = []
        on_stack: set[str] = set()
        indices: dict[str, int] = {}
        lowlinks: dict[str, int] = {}
        result: list[list[str]] = []

        node_ids = {n.id for n in graph.nodes}

        def strongconnect(node_id: str) -> None:
            indices[node_id] = index_counter[0]
            lowlinks[node_id] = index_counter[0]
            index_counter[0] += 1
            stack.append(node_id)
            on_stack.add(node_id)

            for neighbor_id, _, _ in self._adjacency.get(node_id, []):
                if neighbor_id not in node_ids:
                    continue
                if neighbor_id not in indices:
                    strongconnect(neighbor_id)
                    lowlinks[node_id] = min(lowlinks[node_id], lowlinks[neighbor_id])
                elif neighbor_id in on_stack:
                    lowlinks[node_id] = min(lowlinks[node_id], indices[neighbor_id])

            if lowlinks[node_id] == indices[node_id]:
                component: list[str] = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    component.append(w)
                    if w == node_id:
                        break
                result.append(component)

        for nid in node_ids:
            if nid not in indices:
                strongconnect(nid)

        return result

    # -- path finding --------------------------------------------------------

    def find_critical_path(
        self,
        graph: DependencyGraph,
        source_id: str,
        target_id: str,
    ) -> ImpactPath | None:
        """Find the highest-weight path between *source_id* and *target_id*.

        Uses BFS to find the shortest path, then scores it.
        """
        self._build_adjacency(graph)

        # BFS for shortest path
        visited: set[str] = set()
        queue: deque[tuple[str, list[str], list[EdgeType], float]] = deque()
        queue.append((source_id, [source_id], [], 0.0))
        visited.add(source_id)

        while queue:
            current, path, edge_types, total_weight = queue.popleft()
            if current == target_id:
                node_map = {n.id: n for n in graph.nodes}
                risk = self._calculate_path_risk(path, node_map)
                return ImpactPath(
                    source=source_id,
                    target=target_id,
                    path=path,
                    edge_types=edge_types,
                    total_weight=total_weight,
                    risk_score=risk,
                )

            for neighbor_id, edge_type, weight in self._adjacency.get(current, []):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((
                        neighbor_id,
                        path + [neighbor_id],
                        edge_types + [edge_type],
                        total_weight + weight,
                    ))

        return None

    def _calculate_path_risk(
        self,
        path: list[str],
        node_map: dict[str, GraphNode],
    ) -> float:
        """Compute risk score for a path based on node health and path length."""
        if not path:
            return 0.0
        health_scores = [
            node_map[nid].health_score for nid in path if nid in node_map
        ]
        if not health_scores:
            return 0.0
        avg_health = sum(health_scores) / len(health_scores)
        # Longer paths with lower health are riskier
        return round((1.0 - avg_health) * len(path) * 10, 2)

    # -- impact radius -------------------------------------------------------

    def calculate_impact_radius(
        self,
        graph: DependencyGraph,
        node_id: str,
        max_depth: int = 5,
    ) -> list[str]:
        """Return all node IDs reachable from *node_id* within *max_depth*."""
        self._build_adjacency(graph)
        reachable = self._bfs_reachable(graph, node_id, max_depth)
        logger.info(
            "impact_radius_calculated",
            node_id=node_id,
            reachable_count=len(reachable),
        )
        return reachable

    def _bfs_reachable(
        self,
        graph: DependencyGraph,
        start_id: str,
        max_depth: int,
    ) -> list[str]:
        """BFS traversal returning all reachable node IDs within *max_depth*."""
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque()
        queue.append((start_id, 0))
        visited.add(start_id)
        result: list[str] = []

        while queue:
            current, depth = queue.popleft()
            if depth > 0:
                result.append(current)
            if depth >= max_depth:
                continue
            for neighbor_id, _, _ in self._adjacency.get(current, []):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, depth + 1))

        return result

    # -- orphan detection ----------------------------------------------------

    def find_orphans(self, graph: DependencyGraph) -> list[str]:
        """Return node IDs that have no incoming or outgoing edges."""
        self._build_adjacency(graph)
        connected: set[str] = set()
        for edge in graph.edges:
            connected.add(edge.source_id)
            connected.add(edge.target_id)
        orphans = [n.id for n in graph.nodes if n.id not in connected]
        logger.info("orphans_found", count=len(orphans))
        return orphans

    # -- cluster analysis ----------------------------------------------------

    def cluster_analysis(self, graph: DependencyGraph) -> list[ClusterInfo]:
        """Identify tightly-coupled clusters via strongly-connected components."""
        self._build_adjacency(graph)
        sccs = self._tarjan_scc(graph)
        all_node_ids = {n.id for n in graph.nodes}
        clusters: list[ClusterInfo] = []

        for idx, scc in enumerate(sccs):
            scc_set = set(scc)
            internal = 0
            external = 0
            for nid in scc:
                for neighbor_id, _, _ in self._adjacency.get(nid, []):
                    if neighbor_id in scc_set:
                        internal += 1
                    elif neighbor_id in all_node_ids:
                        external += 1

            max_possible = len(scc) * (len(scc) - 1) if len(scc) > 1 else 1
            interconnectedness = internal / max_possible if max_possible > 0 else 0.0

            clusters.append(
                ClusterInfo(
                    id=f"cluster-{idx}",
                    name=f"Cluster {idx}",
                    node_ids=scc,
                    interconnectedness=round(interconnectedness, 3),
                    external_deps=external,
                    internal_deps=internal,
                )
            )

        logger.info("clusters_identified", count=len(clusters))
        return clusters

    # -- depth calculation ---------------------------------------------------

    def get_depth(self, graph: DependencyGraph, root_id: str | None = None) -> int:
        """Compute the maximum depth of the dependency graph.

        If *root_id* is provided, compute from that node; otherwise find the
        global max depth across all roots (nodes with no incoming edges).
        """
        self._build_adjacency(graph)

        if root_id is not None:
            return self._depth_from(root_id)

        # Find root nodes (no incoming edges)
        has_incoming = {e.target_id for e in graph.edges}
        roots = [n.id for n in graph.nodes if n.id not in has_incoming]
        if not roots:
            roots = [graph.nodes[0].id] if graph.nodes else []

        return max((self._depth_from(r) for r in roots), default=0)

    def _depth_from(self, start_id: str) -> int:
        """BFS-based depth calculation from a single node."""
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque()
        queue.append((start_id, 0))
        visited.add(start_id)
        max_d = 0

        while queue:
            current, depth = queue.popleft()
            max_d = max(max_d, depth)
            for neighbor_id, _, _ in self._adjacency.get(current, []):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, depth + 1))

        return max_d

    # -- report generation ---------------------------------------------------

    def generate_report(self, graph: DependencyGraph) -> DependencyReport:
        """Produce a comprehensive :class:`DependencyReport` for *graph*."""
        cycles = self.find_cycles(graph)
        orphans = self.find_orphans(graph)
        clusters = self.cluster_analysis(graph)
        depth = self.get_depth(graph)

        # Health summary
        health_buckets: dict[str, int] = {
            "healthy": 0,
            "degraded": 0,
            "critical": 0,
        }
        for node in graph.nodes:
            if node.health_score >= 0.8:
                health_buckets["healthy"] += 1
            elif node.health_score >= 0.4:
                health_buckets["degraded"] += 1
            else:
                health_buckets["critical"] += 1

        # Critical paths: between all root → leaf pairs (limit to avoid explosion)
        has_incoming = {e.target_id for e in graph.edges}
        has_outgoing = {e.source_id for e in graph.edges}
        roots = [n.id for n in graph.nodes if n.id not in has_incoming and n.id in has_outgoing]
        leaves = [n.id for n in graph.nodes if n.id not in has_outgoing and n.id in has_incoming]

        critical_paths: list[ImpactPath] = []
        for root in roots[:5]:
            for leaf in leaves[:5]:
                path = self.find_critical_path(graph, root, leaf)
                if path is not None:
                    critical_paths.append(path)

        report = DependencyReport(
            total_nodes=len(graph.nodes),
            total_edges=len(graph.edges),
            max_depth=depth,
            orphan_nodes=orphans,
            circular_deps=cycles,
            clusters=clusters,
            critical_paths=critical_paths,
            health_summary=health_buckets,
        )
        logger.info(
            "report_generated",
            nodes=report.total_nodes,
            edges=report.total_edges,
            cycles=len(cycles),
        )
        return report


# ── Graph Query Engine ───────────────────────────────────────────────────────


class GraphQueryEngine:
    """Query dependency graphs with filters and traversals."""

    def __init__(self, graph: DependencyGraph) -> None:
        self._graph = graph
        self._node_map: dict[str, GraphNode] = {n.id: n for n in graph.nodes}
        self._outgoing: dict[str, list[GraphEdge]] = defaultdict(list)
        self._incoming: dict[str, list[GraphEdge]] = defaultdict(list)
        for edge in graph.edges:
            self._outgoing[edge.source_id].append(edge)
            self._incoming[edge.target_id].append(edge)

    # -- query ---------------------------------------------------------------

    def query(self, query: GraphQuery) -> DependencyGraph:
        """Execute a :class:`GraphQuery` and return the matching sub-graph."""
        if query.root_node_id is None:
            nodes = list(self._graph.nodes)
        elif query.include_transitive:
            reachable = self._bfs_ids(query.root_node_id, query.max_depth)
            reachable.add(query.root_node_id)
            nodes = [n for n in self._graph.nodes if n.id in reachable]
        else:
            # Direct neighbours only
            direct = {query.root_node_id}
            for edge in self._outgoing.get(query.root_node_id, []):
                direct.add(edge.target_id)
            nodes = [n for n in self._graph.nodes if n.id in direct]

        # Apply type filter
        if query.node_types:
            type_set = set(query.node_types)
            nodes = [n for n in nodes if n.node_type in type_set]

        # Apply health filter
        if query.filter_health_below is not None:
            nodes = [n for n in nodes if n.health_score < query.filter_health_below]

        node_ids = {n.id for n in nodes}

        # Collect edges within the filtered node set
        edges = [
            e
            for e in self._graph.edges
            if e.source_id in node_ids and e.target_id in node_ids
        ]

        # Apply edge type filter
        if query.edge_types:
            edge_type_set = set(query.edge_types)
            edges = [e for e in edges if e.edge_type in edge_type_set]

        return DependencyGraph(nodes=nodes, edges=edges)

    # -- path finding --------------------------------------------------------

    def find_paths(
        self,
        from_id: str,
        to_id: str,
        max_depth: int = 10,
    ) -> list[ImpactPath]:
        """Find all paths between *from_id* and *to_id* up to *max_depth*."""
        results: list[ImpactPath] = []
        self._dfs_paths(from_id, to_id, max_depth, [from_id], [], 0.0, results)
        return results

    def _dfs_paths(
        self,
        current: str,
        target: str,
        max_depth: int,
        path: list[str],
        edge_types: list[EdgeType],
        weight: float,
        results: list[ImpactPath],
    ) -> None:
        """Recursive DFS collecting all paths to *target*."""
        if current == target and len(path) > 1:
            risk = self._path_risk(path)
            results.append(
                ImpactPath(
                    source=path[0],
                    target=target,
                    path=list(path),
                    edge_types=list(edge_types),
                    total_weight=weight,
                    risk_score=risk,
                )
            )
            return

        if len(path) > max_depth:
            return

        for edge in self._outgoing.get(current, []):
            if edge.target_id not in path:
                path.append(edge.target_id)
                edge_types.append(edge.edge_type)
                self._dfs_paths(
                    edge.target_id, target, max_depth,
                    path, edge_types, weight + edge.weight, results,
                )
                path.pop()
                edge_types.pop()

    def _path_risk(self, path: list[str]) -> float:
        """Compute risk score for a path."""
        scores = [
            self._node_map[nid].health_score
            for nid in path
            if nid in self._node_map
        ]
        if not scores:
            return 0.0
        avg_health = sum(scores) / len(scores)
        return round((1.0 - avg_health) * len(path) * 10, 2)

    # -- dependents / dependencies ------------------------------------------

    def get_dependents(
        self,
        node_id: str,
        direct_only: bool = False,
    ) -> list[GraphNode]:
        """Return nodes that depend on *node_id* (have edges pointing to it)."""
        if direct_only:
            ids = {e.source_id for e in self._incoming.get(node_id, [])}
        else:
            ids = self._bfs_reverse(node_id)
        return [self._node_map[nid] for nid in ids if nid in self._node_map]

    def get_dependencies(
        self,
        node_id: str,
        direct_only: bool = False,
    ) -> list[GraphNode]:
        """Return nodes that *node_id* depends on (outgoing edges)."""
        if direct_only:
            ids = {e.target_id for e in self._outgoing.get(node_id, [])}
        else:
            ids = self._bfs_ids(node_id, max_depth=50)
        return [self._node_map[nid] for nid in ids if nid in self._node_map]

    def search_nodes(
        self,
        pattern: str,
        node_types: list[NodeType] | None = None,
    ) -> list[GraphNode]:
        """Search nodes whose name matches *pattern* (case-insensitive regex)."""
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            logger.warning("invalid_search_pattern", pattern=pattern)
            return []

        results: list[GraphNode] = []
        for node in self._graph.nodes:
            if node_types and node.node_type not in node_types:
                continue
            if regex.search(node.name):
                results.append(node)
        return results

    # -- internal BFS helpers ------------------------------------------------

    def _bfs_ids(self, start_id: str, max_depth: int) -> set[str]:
        """Forward BFS returning reachable node IDs."""
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(start_id, 0)])
        visited.add(start_id)

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for edge in self._outgoing.get(current, []):
                if edge.target_id not in visited:
                    visited.add(edge.target_id)
                    queue.append((edge.target_id, depth + 1))

        visited.discard(start_id)
        return visited

    def _bfs_reverse(self, start_id: str) -> set[str]:
        """Reverse BFS returning all nodes that transitively depend on *start_id*."""
        visited: set[str] = set()
        queue: deque[str] = deque([start_id])
        visited.add(start_id)

        while queue:
            current = queue.popleft()
            for edge in self._incoming.get(current, []):
                if edge.source_id not in visited:
                    visited.add(edge.source_id)
                    queue.append(edge.source_id)

        visited.discard(start_id)
        return visited


# ── Graph Exporter ───────────────────────────────────────────────────────────


_DOT_SHAPES: dict[NodeType, str] = {
    NodeType.REPOSITORY: "folder",
    NodeType.PACKAGE: "box3d",
    NodeType.MODULE: "component",
    NodeType.FUNCTION: "ellipse",
    NodeType.CLASS: "record",
    NodeType.INTERFACE: "diamond",
}

_DOT_EDGE_STYLES: dict[EdgeType, str] = {
    EdgeType.IMPORTS: "solid",
    EdgeType.DEPENDS_ON: "solid",
    EdgeType.EXTENDS: "dashed",
    EdgeType.IMPLEMENTS: "dotted",
    EdgeType.CALLS: "bold",
    EdgeType.PUBLISHES: "solid",
    EdgeType.CONSUMES: "dashed",
}

_MERMAID_ARROWS: dict[EdgeType, str] = {
    EdgeType.IMPORTS: "-->",
    EdgeType.DEPENDS_ON: "-->",
    EdgeType.EXTENDS: "-.->",
    EdgeType.IMPLEMENTS: "-.->",
    EdgeType.CALLS: "==>",
    EdgeType.PUBLISHES: "-->",
    EdgeType.CONSUMES: "-.->",
}


class GraphExporter:
    """Export dependency graphs to various visualization formats."""

    def __init__(self) -> None:
        pass

    # -- public API ----------------------------------------------------------

    def export(
        self,
        graph: DependencyGraph,
        format: ExportFormat,
        layout: LayoutAlgorithm = LayoutAlgorithm.FORCE_DIRECTED,
    ) -> str:
        """Export *graph* to the requested *format* after applying *layout*."""
        laid_out = self._apply_layout(graph, layout)

        exporters: dict[ExportFormat, Any] = {
            ExportFormat.DOT: self.to_dot,
            ExportFormat.MERMAID: self.to_mermaid,
            ExportFormat.D3_JSON: self.to_d3_json,
            ExportFormat.CYTOSCAPE: self.to_cytoscape,
            ExportFormat.SVG_DATA: self.to_dot,  # fallback to DOT
        }
        exporter = exporters.get(format, self.to_mermaid)
        result = exporter(laid_out)
        logger.info("graph_exported", format=format.value, nodes=len(graph.nodes))
        return result

    # -- DOT format ----------------------------------------------------------

    def to_dot(self, graph: DependencyGraph) -> str:
        """Export to Graphviz DOT format."""
        lines: list[str] = ["digraph DependencyGraph {"]
        lines.append('  rankdir=LR;')
        lines.append('  node [fontname="Helvetica", fontsize=10];')
        lines.append('  edge [fontname="Helvetica", fontsize=8];')
        lines.append("")

        for node in graph.nodes:
            shape = _DOT_SHAPES.get(node.node_type, "ellipse")
            label = node.name.replace('"', '\\"')
            color = self._health_color_dot(node.health_score)
            attrs = f'label="{label}", shape={shape}, color="{color}"'
            if node.repo:
                attrs += f', tooltip="{node.repo}"'
            safe_id = self._dot_safe_id(node.id)
            lines.append(f'  {safe_id} [{attrs}];')

        lines.append("")

        for edge in graph.edges:
            style = _DOT_EDGE_STYLES.get(edge.edge_type, "solid")
            src = self._dot_safe_id(edge.source_id)
            tgt = self._dot_safe_id(edge.target_id)
            edge_label = edge.label or edge.edge_type.value
            attrs = f'label="{edge_label}", style={style}'
            if edge.weight != 1.0:
                attrs += f", penwidth={max(0.5, edge.weight * 2)}"
            lines.append(f"  {src} -> {tgt} [{attrs}];")

        lines.append("}")
        return "\n".join(lines)

    # -- Mermaid format ------------------------------------------------------

    def to_mermaid(self, graph: DependencyGraph) -> str:
        """Export to Mermaid flowchart syntax."""
        lines: list[str] = ["graph LR"]

        # Node definitions
        for node in graph.nodes:
            safe_id = self._mermaid_safe_id(node.id)
            display = node.name
            if node.version:
                display += f" v{node.version}"

            shape_open, shape_close = self._mermaid_shape(node.node_type)
            lines.append(f"  {safe_id}{shape_open}{display}{shape_close}")

        lines.append("")

        # Edge definitions
        for edge in graph.edges:
            src = self._mermaid_safe_id(edge.source_id)
            tgt = self._mermaid_safe_id(edge.target_id)
            arrow = _MERMAID_ARROWS.get(edge.edge_type, "-->")
            label = edge.label or edge.edge_type.value
            lines.append(f"  {src} {arrow}|{label}| {tgt}")

        return "\n".join(lines)

    # -- D3 JSON format ------------------------------------------------------

    def to_d3_json(self, graph: DependencyGraph) -> str:
        """Export to D3.js force-directed graph JSON."""
        nodes: list[dict[str, Any]] = []
        for node in graph.nodes:
            nodes.append({
                "id": node.id,
                "name": node.name,
                "group": node.node_type.value,
                "repo": node.repo,
                "version": node.version,
                "health": node.health_score,
                "x": node.position_x,
                "y": node.position_y,
                "radius": max(5, node.health_score * 20),
            })

        links: list[dict[str, Any]] = []
        for edge in graph.edges:
            links.append({
                "source": edge.source_id,
                "target": edge.target_id,
                "type": edge.edge_type.value,
                "weight": edge.weight,
                "label": edge.label or edge.edge_type.value,
            })

        d3_data: dict[str, Any] = {
            "nodes": nodes,
            "links": links,
            "metadata": graph.metadata,
        }
        return json.dumps(d3_data, indent=2)

    # -- Cytoscape JSON format -----------------------------------------------

    def to_cytoscape(self, graph: DependencyGraph) -> str:
        """Export to Cytoscape.js JSON format."""
        elements: list[dict[str, Any]] = []

        for node in graph.nodes:
            elements.append({
                "group": "nodes",
                "data": {
                    "id": node.id,
                    "label": node.name,
                    "type": node.node_type.value,
                    "repo": node.repo,
                    "version": node.version,
                    "health": node.health_score,
                },
                "position": {
                    "x": node.position_x,
                    "y": node.position_y,
                },
            })

        for idx, edge in enumerate(graph.edges):
            elements.append({
                "group": "edges",
                "data": {
                    "id": f"e{idx}",
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "type": edge.edge_type.value,
                    "weight": edge.weight,
                    "label": edge.label or edge.edge_type.value,
                },
            })

        cyto_data: dict[str, Any] = {
            "elements": elements,
            "style": self._cytoscape_default_style(),
            "layout": {"name": "cose"},
        }
        return json.dumps(cyto_data, indent=2)

    # -- layout algorithms ---------------------------------------------------

    def _apply_layout(
        self,
        graph: DependencyGraph,
        layout: LayoutAlgorithm,
    ) -> DependencyGraph:
        """Position nodes according to *layout* and return the updated graph."""
        n = len(graph.nodes)
        if n == 0:
            return graph

        if layout == LayoutAlgorithm.CIRCULAR:
            for i, node in enumerate(graph.nodes):
                angle = 2 * math.pi * i / n
                node.position_x = round(300 + 250 * math.cos(angle), 2)
                node.position_y = round(300 + 250 * math.sin(angle), 2)

        elif layout == LayoutAlgorithm.GRID:
            cols = max(1, int(math.ceil(math.sqrt(n))))
            for i, node in enumerate(graph.nodes):
                node.position_x = float((i % cols) * 150)
                node.position_y = float((i // cols) * 150)

        elif layout == LayoutAlgorithm.HIERARCHICAL:
            self._layout_hierarchical(graph)

        elif layout == LayoutAlgorithm.TREE:
            self._layout_hierarchical(graph)

        else:  # FORCE_DIRECTED — simple spring-based approximation
            self._layout_force_directed(graph)

        return graph

    def _layout_hierarchical(self, graph: DependencyGraph) -> None:
        """Assign positions based on topological layers."""
        # Compute in-degree for each node
        in_degree: dict[str, int] = defaultdict(int)
        outgoing: dict[str, list[str]] = defaultdict(list)
        node_ids = {n.id for n in graph.nodes}

        for edge in graph.edges:
            if edge.target_id in node_ids:
                in_degree[edge.target_id] += 1
            outgoing[edge.source_id].append(edge.target_id)

        # Kahn's algorithm for topological ordering / layer assignment
        layers: dict[str, int] = {}
        queue: deque[str] = deque()
        remaining_in: dict[str, int] = {}

        for node in graph.nodes:
            remaining_in[node.id] = in_degree.get(node.id, 0)
            if remaining_in[node.id] == 0:
                queue.append(node.id)
                layers[node.id] = 0

        while queue:
            nid = queue.popleft()
            for target in outgoing.get(nid, []):
                if target not in node_ids:
                    continue
                remaining_in[target] -= 1
                layers[target] = max(layers.get(target, 0), layers[nid] + 1)
                if remaining_in[target] == 0:
                    queue.append(target)

        # Assign positions not yet placed (cycles)
        max_layer = max(layers.values(), default=0)
        for node in graph.nodes:
            if node.id not in layers:
                max_layer += 1
                layers[node.id] = max_layer

        # Group by layer and spread horizontally
        layer_groups: dict[int, list[GraphNode]] = defaultdict(list)
        for node in graph.nodes:
            layer_groups[layers[node.id]].append(node)

        for layer_idx, nodes_in_layer in layer_groups.items():
            for col, node in enumerate(nodes_in_layer):
                node.position_x = float(col * 200)
                node.position_y = float(layer_idx * 150)

    def _layout_force_directed(self, graph: DependencyGraph) -> None:
        """Simple force-directed layout (Fruchterman-Reingold approximation)."""
        n = len(graph.nodes)
        if n == 0:
            return

        area = 600.0 * 600.0
        k = math.sqrt(area / n)
        iterations = min(50, n * 5)

        # Initialise positions in a circle
        for i, node in enumerate(graph.nodes):
            angle = 2 * math.pi * i / n
            node.position_x = 300 + 200 * math.cos(angle)
            node.position_y = 300 + 200 * math.sin(angle)

        node_map = {n.id: n for n in graph.nodes}

        for _ in range(iterations):
            disp: dict[str, tuple[float, float]] = {n.id: (0.0, 0.0) for n in graph.nodes}

            # Repulsive forces between all node pairs
            for i, u in enumerate(graph.nodes):
                for v in graph.nodes[i + 1:]:
                    dx = u.position_x - v.position_x
                    dy = u.position_y - v.position_y
                    dist = max(math.sqrt(dx * dx + dy * dy), 0.01)
                    force = (k * k) / dist
                    fx = (dx / dist) * force
                    fy = (dy / dist) * force
                    disp[u.id] = (disp[u.id][0] + fx, disp[u.id][1] + fy)
                    disp[v.id] = (disp[v.id][0] - fx, disp[v.id][1] - fy)

            # Attractive forces along edges
            for edge in graph.edges:
                u = node_map.get(edge.source_id)
                v = node_map.get(edge.target_id)
                if u is None or v is None:
                    continue
                dx = u.position_x - v.position_x
                dy = u.position_y - v.position_y
                dist = max(math.sqrt(dx * dx + dy * dy), 0.01)
                force = (dist * dist) / k
                fx = (dx / dist) * force
                fy = (dy / dist) * force
                disp[u.id] = (disp[u.id][0] - fx, disp[u.id][1] - fy)
                disp[v.id] = (disp[v.id][0] + fx, disp[v.id][1] + fy)

            # Apply displacements with temperature cooling
            temperature = max(1.0, 100.0 * (1.0 - _ / iterations))
            for node in graph.nodes:
                dx, dy = disp[node.id]
                dist = max(math.sqrt(dx * dx + dy * dy), 0.01)
                node.position_x += (dx / dist) * min(abs(dx), temperature)
                node.position_y += (dy / dist) * min(abs(dy), temperature)
                # Clamp to canvas
                node.position_x = max(0, min(600, node.position_x))
                node.position_y = max(0, min(600, node.position_y))

        # Round final positions
        for node in graph.nodes:
            node.position_x = round(node.position_x, 2)
            node.position_y = round(node.position_y, 2)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _dot_safe_id(node_id: str) -> str:
        """Make a node ID safe for DOT format."""
        safe = re.sub(r"[^a-zA-Z0-9_]", "_", node_id)
        if safe[0:1].isdigit():
            safe = f"n_{safe}"
        return safe

    @staticmethod
    def _mermaid_safe_id(node_id: str) -> str:
        """Make a node ID safe for Mermaid syntax."""
        return re.sub(r"[^a-zA-Z0-9_]", "_", node_id)

    @staticmethod
    def _mermaid_shape(node_type: NodeType) -> tuple[str, str]:
        """Return Mermaid shape delimiters for *node_type*."""
        shapes: dict[NodeType, tuple[str, str]] = {
            NodeType.REPOSITORY: ("[[", "]]"),
            NodeType.PACKAGE: ("[", "]"),
            NodeType.MODULE: ("(", ")"),
            NodeType.FUNCTION: ("([", "])"),
            NodeType.CLASS: ("{", "}"),
            NodeType.INTERFACE: ("{{", "}}"),
        }
        return shapes.get(node_type, ("[", "]"))

    @staticmethod
    def _health_color_dot(health: float) -> str:
        """Map health score 0..1 to a DOT-compatible colour."""
        if health >= 0.8:
            return "green"
        if health >= 0.5:
            return "orange"
        return "red"

    @staticmethod
    def _cytoscape_default_style() -> list[dict[str, Any]]:
        """Minimal default Cytoscape stylesheet."""
        return [
            {
                "selector": "node",
                "style": {
                    "label": "data(label)",
                    "background-color": "#4a90d9",
                    "text-valign": "center",
                    "font-size": 10,
                },
            },
            {
                "selector": "edge",
                "style": {
                    "label": "data(label)",
                    "curve-style": "bezier",
                    "target-arrow-shape": "triangle",
                    "font-size": 8,
                },
            },
        ]


# ── Dependency Visualizer (Orchestrator) ─────────────────────────────────────


class DependencyVisualizer:
    """Main orchestrator for dependency visualization.

    Combines graph building, analysis, querying, and export into a
    single high-level API.
    """

    def __init__(self) -> None:
        self._builder = GraphBuilder()
        self._analyzer = GraphAnalyzer()
        self._exporter = GraphExporter()

    def build_graph(self, repos: list[dict]) -> DependencyGraph:
        """Build a unified graph from multiple repository descriptors.

        Each entry in *repos* should contain ``name`` and one of
        ``package_json``, ``requirements``, or ``imports``.
        """
        graphs: list[DependencyGraph] = []

        for repo in repos:
            name = repo.get("name", "unknown")
            builder = GraphBuilder()

            if "package_json" in repo:
                g = builder.from_package_json(repo["package_json"], name)
                graphs.append(g)
            elif "requirements" in repo:
                g = builder.from_requirements(repo["requirements"], name)
                graphs.append(g)
            elif "imports" in repo:
                g = builder.from_import_analysis(repo["imports"])
                graphs.append(g)
            else:
                logger.warning("repo_no_manifest", repo=name)

        if not graphs:
            return DependencyGraph()

        merged = self._builder.merge_graphs(graphs)
        logger.info("dependency_graph_built", repos=len(repos), nodes=len(merged.nodes))
        return merged

    def analyze(self, graph: DependencyGraph) -> DependencyReport:
        """Run full analysis on *graph* and return a :class:`DependencyReport`."""
        return self._analyzer.generate_report(graph)

    def visualize(
        self,
        graph: DependencyGraph,
        format: ExportFormat = ExportFormat.MERMAID,
    ) -> str:
        """Export *graph* to the requested visualization format."""
        return self._exporter.export(graph, format)

    def query(self, graph: DependencyGraph, query: GraphQuery) -> DependencyGraph:
        """Query *graph* using the provided :class:`GraphQuery` parameters."""
        engine = GraphQueryEngine(graph)
        return engine.query(query)

    def impact_analysis(self, graph: DependencyGraph, node_id: str) -> dict[str, Any]:
        """Compute impact analysis for a single node.

        Returns a dictionary with ``affected_nodes``, ``depth``,
        ``risk_score``, and ``cycle_member`` fields.
        """
        affected = self._analyzer.calculate_impact_radius(graph, node_id)
        cycles = self._analyzer.find_cycles(graph)
        in_cycle = any(node_id in cycle for cycle in cycles)

        node = graph.get_node(node_id)
        health = node.health_score if node else 1.0
        risk = round((1.0 - health) * len(affected) * 5, 2)

        result: dict[str, Any] = {
            "node_id": node_id,
            "affected_nodes": affected,
            "affected_count": len(affected),
            "depth": self._analyzer.get_depth(graph, root_id=node_id),
            "risk_score": risk,
            "cycle_member": in_cycle,
        }
        logger.info(
            "impact_analysis_complete",
            node_id=node_id,
            affected=len(affected),
            risk=risk,
        )
        return result
