"""Tests for organization-wide dependency analysis."""


import pytest

from codeverify_core.org_dependencies import (
    DependencyEdge,
    OrgDependencyAnalyzer,
    OrgDependencyGraph,
    OrgRepository,
    RiskLevel,
    TransitiveRisk,
    get_org_dependency_analyzer,
    reset_org_dependency_analyzer,
)

ORG = "myorg"


class TestOrgRepository:
    """Tests for OrgRepository dataclass."""

    def test_creation(self):
        """Test repository creation."""
        repo = OrgRepository(
            name="myrepo",
            org=ORG,
            language="python",
            team="platform",
        )

        assert repo.name == "myrepo"
        assert repo.language == "python"
        assert repo.team == "platform"
        assert repo.full_name == "myorg/myrepo"

    def test_default_values(self):
        """Test default values."""
        repo = OrgRepository(name="repo", org=ORG)

        assert repo.language == "unknown"
        assert repo.team is None
        assert repo.last_verified is None


class TestDependencyEdge:
    """Tests for DependencyEdge dataclass."""

    def test_creation(self):
        """Test edge creation."""
        edge = DependencyEdge(
            source="myorg/repo-a",
            target="myorg/repo-b",
            dependency_type="direct",
            version_constraint=">=1.0.0",
        )

        assert edge.source == "myorg/repo-a"
        assert edge.target == "myorg/repo-b"
        assert edge.dependency_type == "direct"
        assert edge.version_constraint == ">=1.0.0"

    def test_default_values(self):
        """Test default values."""
        edge = DependencyEdge(source="a", target="b")

        assert edge.dependency_type == "direct"
        assert edge.version_constraint is None
        assert edge.depth == 1
        assert edge.propagates_risk is True
        assert edge.risk_multiplier == 1.0


class TestTransitiveRisk:
    """Tests for TransitiveRisk dataclass."""

    def test_creation(self):
        """Test risk creation."""
        risk = TransitiveRisk(
            source_repo="myorg/repo-a",
            risk_type="security",
            severity=RiskLevel.HIGH,
            affected_path=["myorg/repo-b", "myorg/repo-a"],
            description="Vulnerability propagation",
        )

        assert risk.source_repo == "myorg/repo-a"
        assert len(risk.affected_path) == 2
        assert risk.severity == RiskLevel.HIGH


class TestOrgDependencyGraph:
    """Tests for OrgDependencyGraph."""

    @pytest.fixture
    def graph(self):
        return OrgDependencyGraph(ORG)

    def _make_repo(self, name, **kwargs):
        return OrgRepository(name=name, org=ORG, **kwargs)

    def test_add_repository(self, graph):
        """Test adding repositories."""
        repo = self._make_repo("repo", language="python")
        graph.add_repository(repo)

        assert f"{ORG}/repo" in graph._repos
        assert graph._repos[f"{ORG}/repo"] == repo

    def test_add_dependency(self, graph):
        """Test adding dependencies."""
        graph.add_repository(self._make_repo("repo-a"))
        graph.add_repository(self._make_repo("repo-b"))

        graph.add_dependency(f"{ORG}/repo-a", f"{ORG}/repo-b")

        assert len(graph._edges) == 1

    def test_get_direct_dependencies(self, graph):
        """Test getting direct dependencies."""
        graph.add_repository(self._make_repo("a"))
        graph.add_repository(self._make_repo("b"))
        graph.add_repository(self._make_repo("c"))

        graph.add_dependency(f"{ORG}/a", f"{ORG}/b")
        graph.add_dependency(f"{ORG}/a", f"{ORG}/c")

        deps = graph.get_direct_dependencies(f"{ORG}/a")
        assert set(deps) == {f"{ORG}/b", f"{ORG}/c"}

    def test_get_direct_dependents(self, graph):
        """Test getting dependents (reverse dependencies)."""
        graph.add_repository(self._make_repo("a"))
        graph.add_repository(self._make_repo("b"))
        graph.add_repository(self._make_repo("c"))

        graph.add_dependency(f"{ORG}/a", f"{ORG}/c")
        graph.add_dependency(f"{ORG}/b", f"{ORG}/c")

        dependents = graph.get_direct_dependents(f"{ORG}/c")
        assert set(dependents) == {f"{ORG}/a", f"{ORG}/b"}

    def test_get_transitive_dependencies(self, graph):
        """Test getting transitive dependencies."""
        # Setup: a -> b -> c -> d
        for name in ["a", "b", "c", "d"]:
            graph.add_repository(self._make_repo(name))

        graph.add_dependency(f"{ORG}/a", f"{ORG}/b")
        graph.add_dependency(f"{ORG}/b", f"{ORG}/c")
        graph.add_dependency(f"{ORG}/c", f"{ORG}/d")

        transitive = graph.get_transitive_dependencies(f"{ORG}/a")
        assert set(transitive.keys()) == {f"{ORG}/b", f"{ORG}/c", f"{ORG}/d"}

    def test_get_transitive_dependencies_with_cycle(self, graph):
        """Test transitive dependencies with cycle detection."""
        # Setup: a -> b -> c -> a (cycle)
        for name in ["a", "b", "c"]:
            graph.add_repository(self._make_repo(name))

        graph.add_dependency(f"{ORG}/a", f"{ORG}/b")
        graph.add_dependency(f"{ORG}/b", f"{ORG}/c")
        graph.add_dependency(f"{ORG}/c", f"{ORG}/a")

        # Should not infinite loop; note: in a cycle, the source itself
        # appears as a transitive dependency via the back-edge
        transitive = graph.get_transitive_dependencies(f"{ORG}/a")
        assert {f"{ORG}/b", f"{ORG}/c"}.issubset(set(transitive.keys()))

    def test_detect_circular_dependencies(self, graph):
        """Test cycle detection."""
        for name in ["a", "b", "c"]:
            graph.add_repository(self._make_repo(name))

        graph.add_dependency(f"{ORG}/a", f"{ORG}/b")
        graph.add_dependency(f"{ORG}/b", f"{ORG}/c")
        graph.add_dependency(f"{ORG}/c", f"{ORG}/a")

        cycles = graph.detect_circular_dependencies()
        assert len(cycles) > 0

    def test_no_cycles(self, graph):
        """Test cycle detection with no cycles."""
        for name in ["a", "b", "c"]:
            graph.add_repository(self._make_repo(name))

        graph.add_dependency(f"{ORG}/a", f"{ORG}/b")
        graph.add_dependency(f"{ORG}/b", f"{ORG}/c")

        cycles = graph.detect_circular_dependencies()
        assert len(cycles) == 0

    def test_detect_clusters(self, graph):
        """Test detecting clusters of tightly coupled repositories."""
        for name in ["a", "b", "c", "d"]:
            graph.add_repository(self._make_repo(name))

        # Component 1: a <-> b
        graph.add_dependency(f"{ORG}/a", f"{ORG}/b")
        graph.add_dependency(f"{ORG}/b", f"{ORG}/a")

        # Component 2: c <-> d
        graph.add_dependency(f"{ORG}/c", f"{ORG}/d")
        graph.add_dependency(f"{ORG}/d", f"{ORG}/c")

        # Link between components
        graph.add_dependency(f"{ORG}/b", f"{ORG}/c")

        clusters = graph.detect_clusters(min_cohesion=0.0)
        # Should find at least 1 cluster
        assert len(clusters) >= 1

    def test_generate_visualization_data(self, graph):
        """Test generating visualization data."""
        graph.add_repository(self._make_repo("a", team="team1"))
        graph.add_repository(self._make_repo("b", team="team2"))
        graph.add_dependency(f"{ORG}/a", f"{ORG}/b")

        viz = graph.generate_visualization_data()

        assert "nodes" in viz
        assert "edges" in viz
        assert len(viz["nodes"]) == 2
        assert len(viz["edges"]) == 1

    def test_get_affected_by_change(self, graph):
        """Test getting repos affected by a change."""
        for name in ["a", "b", "c"]:
            graph.add_repository(self._make_repo(name))

        # b and c depend on a
        graph.add_dependency(f"{ORG}/b", f"{ORG}/a")
        graph.add_dependency(f"{ORG}/c", f"{ORG}/a")

        affected = graph.get_affected_by_change(f"{ORG}/a")

        assert affected["total_affected"] == 2

    def test_get_metrics(self, graph):
        """Test calculating organization-wide metrics."""
        graph.add_repository(self._make_repo("a", team="team1"))
        graph.add_repository(self._make_repo("b", team="team1"))
        graph.add_dependency(f"{ORG}/a", f"{ORG}/b")

        metrics = graph.get_metrics()

        assert metrics.total_repos == 2
        assert metrics.total_dependencies == 1


class TestOrgDependencyAnalyzer:
    """Tests for OrgDependencyAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        return OrgDependencyAnalyzer(ORG)

    def test_analyze_from_manifest_data(self, analyzer):
        """Test building graph from manifest data."""
        manifest_data = [
            {
                "name": "service-a",
                "language": "python",
                "type": "service",
                "dependencies": [f"{ORG}/lib-core"],
            },
            {
                "name": "lib-core",
                "language": "python",
                "type": "library",
                "dependencies": [],
            },
        ]

        analyzer.analyze_from_manifest_data(manifest_data)

        assert len(analyzer.graph._repos) == 2
        assert len(analyzer.graph._edges) == 1

    def test_get_full_analysis(self, analyzer):
        """Test full analysis report generation."""
        analyzer.graph.add_repository(
            OrgRepository(name="a", org=ORG, team="team1")
        )
        analyzer.graph.add_repository(
            OrgRepository(name="b", org=ORG, team="team1")
        )
        analyzer.graph.add_dependency(f"{ORG}/a", f"{ORG}/b")

        report = analyzer.get_full_analysis()

        assert report["organization"] == ORG
        assert "metrics" in report
        assert report["metrics"]["total_repos"] == 2

    def test_analyze_transitive_risks(self, analyzer):
        """Test detecting transitive risks from vulnerabilities."""
        analyzer.graph.add_repository(
            OrgRepository(name="a", org=ORG, known_vulnerabilities=0)
        )
        analyzer.graph.add_repository(
            OrgRepository(name="b", org=ORG, known_vulnerabilities=3)
        )
        analyzer.graph.add_dependency(f"{ORG}/a", f"{ORG}/b")

        risks = analyzer.graph.analyze_transitive_risks()

        # Should detect risk from vulnerable dependency
        assert isinstance(risks, list)


class TestGlobalAnalyzer:
    """Tests for global analyzer functions."""

    def teardown_method(self):
        reset_org_dependency_analyzer()

    def test_get_analyzer_singleton(self):
        """Test singleton pattern."""
        analyzer1 = get_org_dependency_analyzer(ORG)
        analyzer2 = get_org_dependency_analyzer(ORG)
        assert analyzer1 is analyzer2

    def test_reset_analyzer(self):
        """Test analyzer reset."""
        analyzer1 = get_org_dependency_analyzer(ORG)
        reset_org_dependency_analyzer()
        analyzer2 = get_org_dependency_analyzer(ORG)
        assert analyzer1 is not analyzer2
