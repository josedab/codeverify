"""Tests for Knowledge Graph module."""

import pytest

from codeverify_core.knowledge_graph import (
    EdgeType,
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
    KnowledgeGraphConfig,
    KnowledgeIngester,
    NodeType,
    ProofReuseEngine,
    get_knowledge_graph,
    reset_knowledge_graph,
)


class TestNodeType:
    """Tests for NodeType enum."""

    def test_all_types_exist(self):
        """All expected node types exist."""
        assert NodeType.PROOF.value == "proof"
        assert NodeType.PATTERN.value == "pattern"
        assert NodeType.FUNCTION.value == "function"
        assert NodeType.FILE.value == "file"
        assert NodeType.TEAM.value == "team"
        assert NodeType.REPOSITORY.value == "repository"
        assert NodeType.BUG_CLASS.value == "bug_class"


class TestEdgeType:
    """Tests for EdgeType enum."""

    def test_all_types_exist(self):
        """All expected edge types exist."""
        assert EdgeType.PROVES.value == "proves"
        assert EdgeType.CONTAINS.value == "contains"
        assert EdgeType.DEPENDS_ON.value == "depends_on"
        assert EdgeType.SIMILAR_TO.value == "similar_to"
        assert EdgeType.REUSES_PROOF.value == "reuses_proof"


class TestGraphNode:
    """Tests for GraphNode dataclass."""

    def test_creation_with_properties(self):
        """Can create a GraphNode with properties."""
        node = GraphNode(
            id="n1",
            node_type=NodeType.PROOF,
            label="test proof",
            properties={"pattern_hash": "abc123"},
        )
        assert node.id == "n1"
        assert node.node_type == NodeType.PROOF
        assert node.properties["pattern_hash"] == "abc123"


class TestGraphEdge:
    """Tests for GraphEdge dataclass."""

    def test_creation(self):
        """Can create a GraphEdge."""
        edge = GraphEdge(
            source_id="n1",
            target_id="n2",
            edge_type=EdgeType.PROVES,
            weight=0.9,
        )
        assert edge.source_id == "n1"
        assert edge.edge_type == EdgeType.PROVES
        assert edge.weight == 0.9


class TestKnowledgeGraph:
    """Tests for KnowledgeGraph."""

    def _make_graph(self) -> KnowledgeGraph:
        return KnowledgeGraph()

    def test_add_node_and_get_node(self):
        """Can add and retrieve a node."""
        g = self._make_graph()
        node = GraphNode(id="n1", node_type=NodeType.FILE, label="main.py")
        g.add_node(node)
        assert g.get_node("n1") is node
        assert g.get_node("missing") is None

    def test_add_edge_and_get_neighbors(self):
        """Can add an edge and retrieve neighbours."""
        g = self._make_graph()
        g.add_node(GraphNode(id="a", node_type=NodeType.FILE, label="a"))
        g.add_node(GraphNode(id="b", node_type=NodeType.FUNCTION, label="b"))
        g.add_edge(GraphEdge(source_id="a", target_id="b", edge_type=EdgeType.CONTAINS))
        neighbors = g.get_neighbors("a")
        assert len(neighbors) == 1
        assert neighbors[0].id == "b"

    def test_get_neighbors_with_edge_type_filter(self):
        """get_neighbors filters by edge_type."""
        g = self._make_graph()
        g.add_node(GraphNode(id="a", node_type=NodeType.FILE, label="a"))
        g.add_node(GraphNode(id="b", node_type=NodeType.FUNCTION, label="b"))
        g.add_node(GraphNode(id="c", node_type=NodeType.PROOF, label="c"))
        g.add_edge(GraphEdge(source_id="a", target_id="b", edge_type=EdgeType.CONTAINS))
        g.add_edge(GraphEdge(source_id="a", target_id="c", edge_type=EdgeType.PROVES))

        contains_only = g.get_neighbors("a", edge_type=EdgeType.CONTAINS)
        assert len(contains_only) == 1
        assert contains_only[0].id == "b"

    def test_find_similar_proofs_returns_matches(self):
        """find_similar_proofs returns matches above threshold."""
        g = self._make_graph()
        g.add_node(GraphNode(
            id="p1", node_type=NodeType.PROOF, label="proof",
            properties={"pattern_hash": "aabbccdd"},
        ))
        results = g.find_similar_proofs("aabbccdd", threshold=0.5)
        assert len(results) >= 1
        assert results[0][1] >= 0.5

    def test_find_similar_proofs_empty_for_no_matches(self):
        """find_similar_proofs returns empty for no matches."""
        g = self._make_graph()
        g.add_node(GraphNode(
            id="p1", node_type=NodeType.PROOF, label="proof",
            properties={"pattern_hash": "aabbccdd"},
        ))
        results = g.find_similar_proofs("zzzzzzzz", threshold=0.99)
        assert results == []

    def test_get_subgraph_returns_correct_depth(self):
        """get_subgraph returns nodes within depth."""
        g = self._make_graph()
        g.add_node(GraphNode(id="a", node_type=NodeType.FILE, label="a"))
        g.add_node(GraphNode(id="b", node_type=NodeType.FUNCTION, label="b"))
        g.add_node(GraphNode(id="c", node_type=NodeType.PROOF, label="c"))
        g.add_edge(GraphEdge(source_id="a", target_id="b", edge_type=EdgeType.CONTAINS))
        g.add_edge(GraphEdge(source_id="b", target_id="c", edge_type=EdgeType.PROVES))

        nodes, edges = g.get_subgraph("a", depth=1)
        node_ids = {n.id for n in nodes}
        assert "a" in node_ids
        assert "b" in node_ids

    def test_get_stats(self):
        """get_stats returns summary statistics."""
        g = self._make_graph()
        g.add_node(GraphNode(id="n1", node_type=NodeType.FILE, label="f"))
        g.add_node(GraphNode(id="n2", node_type=NodeType.FUNCTION, label="fn"))
        g.add_edge(GraphEdge(source_id="n1", target_id="n2", edge_type=EdgeType.CONTAINS))
        stats = g.get_stats()
        assert stats["total_nodes"] == 2
        assert stats["total_edges"] == 1
        assert "node_type_counts" in stats


class TestProofReuseEngine:
    """Tests for ProofReuseEngine."""

    def _make_engine(self) -> tuple[KnowledgeGraph, ProofReuseEngine]:
        g = KnowledgeGraph()
        g.add_node(GraphNode(
            id="proof1", node_type=NodeType.PROOF, label="null check proof",
            properties={"pattern_hash": "aabbccdd11223344"},
        ))
        return g, ProofReuseEngine(g)

    def test_suggest_proof(self):
        """suggest_proof returns relevant proofs."""
        g, engine = self._make_engine()
        suggestions = engine.suggest_proof("aabbccdd11223344", "def check_null(x)")
        assert len(suggestions) >= 1

    def test_record_proof_usage(self):
        """record_proof_usage logs the usage."""
        g, engine = self._make_engine()
        g.add_node(GraphNode(id="func1", node_type=NodeType.FUNCTION, label="fn"))
        engine.record_proof_usage("proof1", "func1", success=True)
        stats = engine.get_reuse_stats()
        assert stats["total_reuse_attempts"] == 1
        assert stats["successful_reuses"] == 1

    def test_get_reuse_stats(self):
        """get_reuse_stats returns correct statistics."""
        _, engine = self._make_engine()
        stats = engine.get_reuse_stats()
        assert stats["total_reuse_attempts"] == 0
        assert stats["reuse_success_rate"] == 0.0


class TestKnowledgeIngester:
    """Tests for KnowledgeIngester."""

    def test_ingest_verification_result(self):
        """ingest_verification_result creates nodes and edges."""
        g = KnowledgeGraph()
        ingester = KnowledgeIngester(g)
        created_ids = ingester.ingest_verification_result(
            repo_id="my-repo",
            file_path="src/main.py",
            function_name="process",
            findings=[{"category": "null_deref", "message": "Possible null"}],
            proofs=[{"label": "null-check-proof", "status": "verified"}],
        )
        assert len(created_ids) >= 3  # repo + file + function + bugclass + proof
        stats = g.get_stats()
        assert stats["total_nodes"] >= 4
        assert stats["total_edges"] >= 3


class TestKnowledgeGraphSingletons:
    """Tests for module-level singletons."""

    def test_get_and_reset_knowledge_graph(self):
        """get/reset knowledge graph singletons."""
        reset_knowledge_graph()
        g1 = get_knowledge_graph()
        g2 = get_knowledge_graph()
        assert g1 is g2
        reset_knowledge_graph()
        g3 = get_knowledge_graph()
        assert g3 is not g1
        reset_knowledge_graph()


class TestKnowledgeGraphEdgeCases:
    """Edge case tests for KnowledgeGraph."""

    def test_max_nodes_triggers_error(self):
        """Exceeding max_nodes raises ValueError."""
        config = KnowledgeGraphConfig(max_nodes=2)
        g = KnowledgeGraph(config=config)
        g.add_node(GraphNode(id="a", node_type=NodeType.FILE, label="a"))
        g.add_node(GraphNode(id="b", node_type=NodeType.FILE, label="b"))
        with pytest.raises(ValueError, match="Maximum node count"):
            g.add_node(GraphNode(id="c", node_type=NodeType.FILE, label="c"))
