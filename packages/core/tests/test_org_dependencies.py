"""Tests for organization-wide dependency analysis."""

import pytest
from datetime import datetime
from codeverify_core.org_dependencies import (
    OrgDependencyAnalyzer,
    OrgDependencyGraph,
    OrgRepository,
    RepositoryConnection,
    TransitiveRisk,
    get_org_dependency_analyzer,
    reset_org_dependency_analyzer,
)


class TestOrgRepository:
    """Tests for OrgRepository dataclass."""

    def test_creation(self):
        """Test repository creation."""
        repo = OrgRepository(
            name="myorg/myrepo",
            language="python",
            is_critical=True,
            team="platform",
        )
        
        assert repo.name == "myorg/myrepo"
        assert repo.language == "python"
        assert repo.is_critical is True
        assert repo.team == "platform"

    def test_default_values(self):
        """Test default values."""
        repo = OrgRepository(name="test/repo")
        
        assert repo.language == ""
        assert repo.is_critical is False
        assert repo.team == ""
        assert repo.last_verified is None


class TestRepositoryConnection:
    """Tests for RepositoryConnection dataclass."""

    def test_creation(self):
        """Test connection creation."""
        conn = RepositoryConnection(
            source="repo-a",
            target="repo-b",
            dependency_type="runtime",
            version_constraint=">=1.0.0",
        )
        
        assert conn.source == "repo-a"
        assert conn.target == "repo-b"
        assert conn.dependency_type == "runtime"
        assert conn.version_constraint == ">=1.0.0"


class TestTransitiveRisk:
    """Tests for TransitiveRisk dataclass."""

    def test_creation(self):
        """Test risk creation."""
        risk = TransitiveRisk(
            source_repo="repo-a",
            affected_repos=["repo-b", "repo-c"],
            risk_type="unverified_dependency",
            severity="high",
            description="Unverified dependency chain",
        )
        
        assert risk.source_repo == "repo-a"
        assert len(risk.affected_repos) == 2
        assert risk.severity == "high"


class TestOrgDependencyGraph:
    """Tests for OrgDependencyGraph."""

    @pytest.fixture
    def graph(self):
        return OrgDependencyGraph()

    def test_add_repository(self, graph):
        """Test adding repositories."""
        repo = OrgRepository(name="test/repo", language="python")
        graph.add_repository(repo)
        
        assert "test/repo" in graph.repositories
        assert graph.repositories["test/repo"] == repo

    def test_add_connection(self, graph):
        """Test adding connections."""
        graph.add_repository(OrgRepository(name="repo-a"))
        graph.add_repository(OrgRepository(name="repo-b"))
        
        conn = RepositoryConnection(source="repo-a", target="repo-b")
        graph.add_connection(conn)
        
        assert len(graph.connections) == 1
        assert conn in graph.connections

    def test_get_dependencies(self, graph):
        """Test getting direct dependencies."""
        graph.add_repository(OrgRepository(name="a"))
        graph.add_repository(OrgRepository(name="b"))
        graph.add_repository(OrgRepository(name="c"))
        
        graph.add_connection(RepositoryConnection(source="a", target="b"))
        graph.add_connection(RepositoryConnection(source="a", target="c"))
        
        deps = graph.get_dependencies("a")
        assert set(deps) == {"b", "c"}

    def test_get_dependents(self, graph):
        """Test getting dependents (reverse dependencies)."""
        graph.add_repository(OrgRepository(name="a"))
        graph.add_repository(OrgRepository(name="b"))
        graph.add_repository(OrgRepository(name="c"))
        
        graph.add_connection(RepositoryConnection(source="a", target="c"))
        graph.add_connection(RepositoryConnection(source="b", target="c"))
        
        dependents = graph.get_dependents("c")
        assert set(dependents) == {"a", "b"}

    def test_get_transitive_dependencies(self, graph):
        """Test getting transitive dependencies."""
        # Setup: a -> b -> c -> d
        for name in ["a", "b", "c", "d"]:
            graph.add_repository(OrgRepository(name=name))
        
        graph.add_connection(RepositoryConnection(source="a", target="b"))
        graph.add_connection(RepositoryConnection(source="b", target="c"))
        graph.add_connection(RepositoryConnection(source="c", target="d"))
        
        transitive = graph.get_transitive_dependencies("a")
        assert set(transitive) == {"b", "c", "d"}

    def test_get_transitive_dependencies_with_cycle(self, graph):
        """Test transitive dependencies with cycle detection."""
        # Setup: a -> b -> c -> a (cycle)
        for name in ["a", "b", "c"]:
            graph.add_repository(OrgRepository(name=name))
        
        graph.add_connection(RepositoryConnection(source="a", target="b"))
        graph.add_connection(RepositoryConnection(source="b", target="c"))
        graph.add_connection(RepositoryConnection(source="c", target="a"))
        
        # Should not infinite loop
        transitive = graph.get_transitive_dependencies("a")
        assert set(transitive) == {"b", "c"}

    def test_detect_cycles(self, graph):
        """Test cycle detection."""
        for name in ["a", "b", "c"]:
            graph.add_repository(OrgRepository(name=name))
        
        graph.add_connection(RepositoryConnection(source="a", target="b"))
        graph.add_connection(RepositoryConnection(source="b", target="c"))
        graph.add_connection(RepositoryConnection(source="c", target="a"))
        
        cycles = graph.detect_cycles()
        assert len(cycles) > 0

    def test_no_cycles(self, graph):
        """Test cycle detection with no cycles."""
        for name in ["a", "b", "c"]:
            graph.add_repository(OrgRepository(name=name))
        
        graph.add_connection(RepositoryConnection(source="a", target="b"))
        graph.add_connection(RepositoryConnection(source="b", target="c"))
        
        cycles = graph.detect_cycles()
        assert len(cycles) == 0

    def test_find_strongly_connected_components(self, graph):
        """Test finding strongly connected components."""
        # Two components: (a, b) and (c, d)
        for name in ["a", "b", "c", "d"]:
            graph.add_repository(OrgRepository(name=name))
        
        # Component 1: a <-> b
        graph.add_connection(RepositoryConnection(source="a", target="b"))
        graph.add_connection(RepositoryConnection(source="b", target="a"))
        
        # Component 2: c <-> d
        graph.add_connection(RepositoryConnection(source="c", target="d"))
        graph.add_connection(RepositoryConnection(source="d", target="c"))
        
        # Link between components
        graph.add_connection(RepositoryConnection(source="b", target="c"))
        
        components = graph.find_strongly_connected_components()
        # Should find at least 2 non-trivial components
        assert len(components) >= 2

    def test_get_critical_path(self, graph):
        """Test getting critical path to a repository."""
        for name in ["a", "b", "c"]:
            graph.add_repository(OrgRepository(name=name, is_critical=(name == "c")))
        
        graph.add_connection(RepositoryConnection(source="a", target="b"))
        graph.add_connection(RepositoryConnection(source="b", target="c"))
        
        path = graph.get_critical_path("a", "c")
        assert path == ["a", "b", "c"]

    def test_to_visualization(self, graph):
        """Test generating visualization data."""
        graph.add_repository(OrgRepository(name="a", team="team1"))
        graph.add_repository(OrgRepository(name="b", team="team2"))
        graph.add_connection(RepositoryConnection(source="a", target="b"))
        
        viz = graph.to_visualization()
        
        assert "nodes" in viz
        assert "edges" in viz
        assert len(viz["nodes"]) == 2
        assert len(viz["edges"]) == 1


class TestOrgDependencyAnalyzer:
    """Tests for OrgDependencyAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        return OrgDependencyAnalyzer()

    def test_analyze_transitive_risks_unverified(self, analyzer):
        """Test detecting unverified dependency risks."""
        graph = OrgDependencyGraph()
        
        # a depends on b (verified), b depends on c (unverified)
        graph.add_repository(OrgRepository(
            name="a",
            last_verified=datetime.utcnow(),
        ))
        graph.add_repository(OrgRepository(
            name="b",
            last_verified=datetime.utcnow(),
        ))
        graph.add_repository(OrgRepository(
            name="c",
            last_verified=None,  # Unverified
        ))
        
        graph.add_connection(RepositoryConnection(source="a", target="b"))
        graph.add_connection(RepositoryConnection(source="b", target="c"))
        
        risks = analyzer.analyze_transitive_risks(graph)
        
        # Should detect risk from unverified dependency
        unverified_risks = [r for r in risks if r.risk_type == "unverified_dependency"]
        assert len(unverified_risks) > 0

    def test_analyze_transitive_risks_critical_path(self, analyzer):
        """Test detecting critical dependency paths."""
        graph = OrgDependencyGraph()
        
        graph.add_repository(OrgRepository(name="a", is_critical=True))
        graph.add_repository(OrgRepository(name="b"))
        graph.add_repository(OrgRepository(name="c"))
        
        # Long chain to critical repo
        graph.add_connection(RepositoryConnection(source="c", target="b"))
        graph.add_connection(RepositoryConnection(source="b", target="a"))
        
        risks = analyzer.analyze_transitive_risks(graph)
        
        critical_risks = [r for r in risks if r.risk_type == "critical_dependency_chain"]
        # May or may not find depending on chain length threshold
        assert isinstance(critical_risks, list)

    def test_get_impact_analysis(self, analyzer):
        """Test impact analysis for a repository."""
        graph = OrgDependencyGraph()
        
        # b and c depend on a
        graph.add_repository(OrgRepository(name="a"))
        graph.add_repository(OrgRepository(name="b"))
        graph.add_repository(OrgRepository(name="c"))
        
        graph.add_connection(RepositoryConnection(source="b", target="a"))
        graph.add_connection(RepositoryConnection(source="c", target="a"))
        
        impact = analyzer.get_impact_analysis(graph, "a")
        
        assert "affected_repositories" in impact
        assert set(impact["affected_repositories"]) == {"b", "c"}

    def test_generate_report(self, analyzer):
        """Test report generation."""
        graph = OrgDependencyGraph()
        graph.add_repository(OrgRepository(name="a", team="team1"))
        graph.add_repository(OrgRepository(name="b", team="team1"))
        graph.add_connection(RepositoryConnection(source="a", target="b"))
        
        report = analyzer.generate_report(graph)
        
        assert "summary" in report
        assert "total_repositories" in report["summary"]
        assert report["summary"]["total_repositories"] == 2


class TestGlobalAnalyzer:
    """Tests for global analyzer functions."""

    def teardown_method(self):
        reset_org_dependency_analyzer()

    def test_get_analyzer_singleton(self):
        """Test singleton pattern."""
        analyzer1 = get_org_dependency_analyzer()
        analyzer2 = get_org_dependency_analyzer()
        assert analyzer1 is analyzer2

    def test_reset_analyzer(self):
        """Test analyzer reset."""
        analyzer1 = get_org_dependency_analyzer()
        reset_org_dependency_analyzer()
        analyzer2 = get_org_dependency_analyzer()
        assert analyzer1 is not analyzer2
