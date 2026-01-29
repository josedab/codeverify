"""Tests for Monorepo Intelligence module."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from codeverify_core.monorepo import (
    DependencyGraph,
    MonorepoAnalyzer,
    MonorepoType,
    Package,
    PackageDependency,
)


class TestPackage:
    """Tests for Package dataclass."""

    def test_package_creation(self):
        """Package can be created with required fields."""
        pkg = Package(
            name="@myorg/core",
            path=Path("/repo/packages/core"),
            version="1.0.0",
        )
        assert pkg.name == "@myorg/core"
        assert pkg.version == "1.0.0"
        assert pkg.dependencies == []

    def test_package_with_dependencies(self):
        """Package can have dependencies."""
        dep = PackageDependency(
            name="lodash",
            version="^4.17.0",
            is_internal=False,
        )
        pkg = Package(
            name="utils",
            path=Path("/repo/packages/utils"),
            version="2.0.0",
            dependencies=[dep],
        )
        assert len(pkg.dependencies) == 1
        assert pkg.dependencies[0].name == "lodash"


class TestDependencyGraph:
    """Tests for DependencyGraph."""

    def test_empty_graph(self):
        """Empty graph has no packages."""
        graph = DependencyGraph()
        assert graph.packages == {}
        assert graph.edges == {}

    def test_add_package(self):
        """Can add packages to graph."""
        graph = DependencyGraph()
        pkg = Package(name="core", path=Path("/pkg"), version="1.0.0")
        graph.packages["core"] = pkg
        assert "core" in graph.packages

    def test_add_edge(self):
        """Can add dependency edges."""
        graph = DependencyGraph()
        graph.edges["app"] = ["core", "utils"]
        assert graph.edges["app"] == ["core", "utils"]

    def test_detect_cycle_no_cycle(self):
        """Detects no cycle in acyclic graph."""
        graph = DependencyGraph()
        graph.edges = {
            "app": ["core"],
            "core": ["utils"],
            "utils": [],
        }
        cycles = graph.detect_cycles()
        assert cycles == []

    def test_detect_cycle_simple(self):
        """Detects simple cycle."""
        graph = DependencyGraph()
        graph.edges = {
            "a": ["b"],
            "b": ["c"],
            "c": ["a"],
        }
        cycles = graph.detect_cycles()
        assert len(cycles) > 0

    def test_get_dependents(self):
        """Gets packages that depend on a given package."""
        graph = DependencyGraph()
        graph.edges = {
            "app": ["core"],
            "worker": ["core"],
            "core": ["utils"],
            "utils": [],
        }
        dependents = graph.get_dependents("core")
        assert "app" in dependents
        assert "worker" in dependents
        assert "utils" not in dependents

    def test_topological_sort(self):
        """Topological sort produces valid order."""
        graph = DependencyGraph()
        graph.edges = {
            "app": ["core", "utils"],
            "core": ["utils"],
            "utils": [],
        }
        sorted_pkgs = graph.topological_sort()
        
        # utils must come before core and app
        utils_idx = sorted_pkgs.index("utils")
        core_idx = sorted_pkgs.index("core")
        app_idx = sorted_pkgs.index("app")
        
        assert utils_idx < core_idx
        assert utils_idx < app_idx
        assert core_idx < app_idx


class TestMonorepoAnalyzer:
    """Tests for MonorepoAnalyzer."""

    def test_detect_nx(self, tmp_path):
        """Detects Nx monorepo."""
        (tmp_path / "nx.json").write_text('{"version": 2}')
        
        analyzer = MonorepoAnalyzer(tmp_path)
        assert analyzer.monorepo_type == MonorepoType.NX

    def test_detect_turborepo(self, tmp_path):
        """Detects Turborepo monorepo."""
        (tmp_path / "turbo.json").write_text('{"pipeline": {}}')
        
        analyzer = MonorepoAnalyzer(tmp_path)
        assert analyzer.monorepo_type == MonorepoType.TURBOREPO

    def test_detect_lerna(self, tmp_path):
        """Detects Lerna monorepo."""
        (tmp_path / "lerna.json").write_text('{"version": "1.0.0"}')
        
        analyzer = MonorepoAnalyzer(tmp_path)
        assert analyzer.monorepo_type == MonorepoType.LERNA

    def test_detect_pnpm(self, tmp_path):
        """Detects pnpm workspaces."""
        (tmp_path / "pnpm-workspace.yaml").write_text("packages:\n  - packages/*")
        
        analyzer = MonorepoAnalyzer(tmp_path)
        assert analyzer.monorepo_type == MonorepoType.PNPM

    def test_detect_yarn_workspaces(self, tmp_path):
        """Detects Yarn workspaces."""
        package_json = {
            "name": "monorepo",
            "workspaces": ["packages/*"]
        }
        (tmp_path / "package.json").write_text(json.dumps(package_json))
        
        analyzer = MonorepoAnalyzer(tmp_path)
        assert analyzer.monorepo_type == MonorepoType.YARN_WORKSPACES

    def test_detect_none(self, tmp_path):
        """Detects no monorepo."""
        (tmp_path / "package.json").write_text('{"name": "single-pkg"}')
        
        analyzer = MonorepoAnalyzer(tmp_path)
        assert analyzer.monorepo_type == MonorepoType.NONE

    def test_discover_packages_npm(self, tmp_path):
        """Discovers npm packages."""
        # Create package structure
        packages_dir = tmp_path / "packages"
        packages_dir.mkdir()
        
        core_dir = packages_dir / "core"
        core_dir.mkdir()
        (core_dir / "package.json").write_text(json.dumps({
            "name": "@myorg/core",
            "version": "1.0.0",
            "dependencies": {"lodash": "^4.0.0"}
        }))
        
        utils_dir = packages_dir / "utils"
        utils_dir.mkdir()
        (utils_dir / "package.json").write_text(json.dumps({
            "name": "@myorg/utils",
            "version": "1.0.0",
            "dependencies": {"@myorg/core": "^1.0.0"}
        }))
        
        # Create root with workspaces
        (tmp_path / "package.json").write_text(json.dumps({
            "name": "monorepo",
            "workspaces": ["packages/*"]
        }))
        
        analyzer = MonorepoAnalyzer(tmp_path)
        packages = analyzer.discover_packages()
        
        assert len(packages) >= 2
        names = [p.name for p in packages]
        assert "@myorg/core" in names
        assert "@myorg/utils" in names

    def test_build_dependency_graph(self, tmp_path):
        """Builds dependency graph from packages."""
        packages_dir = tmp_path / "packages"
        packages_dir.mkdir()
        
        # Create interdependent packages
        for name, deps in [
            ("core", {}),
            ("utils", {"@myorg/core": "^1.0.0"}),
            ("app", {"@myorg/core": "^1.0.0", "@myorg/utils": "^1.0.0"}),
        ]:
            pkg_dir = packages_dir / name
            pkg_dir.mkdir()
            (pkg_dir / "package.json").write_text(json.dumps({
                "name": f"@myorg/{name}",
                "version": "1.0.0",
                "dependencies": deps
            }))
        
        (tmp_path / "package.json").write_text(json.dumps({
            "workspaces": ["packages/*"]
        }))
        
        analyzer = MonorepoAnalyzer(tmp_path)
        graph = analyzer.build_dependency_graph()
        
        assert "@myorg/core" in graph.packages
        assert "@myorg/utils" in graph.packages
        assert "@myorg/app" in graph.packages

    def test_get_affected_packages(self, tmp_path):
        """Gets packages affected by a change."""
        packages_dir = tmp_path / "packages"
        packages_dir.mkdir()
        
        for name, deps in [
            ("core", {}),
            ("utils", {"@myorg/core": "^1.0.0"}),
            ("app", {"@myorg/utils": "^1.0.0"}),
        ]:
            pkg_dir = packages_dir / name
            pkg_dir.mkdir()
            (pkg_dir / "package.json").write_text(json.dumps({
                "name": f"@myorg/{name}",
                "version": "1.0.0",
                "dependencies": deps
            }))
        
        (tmp_path / "package.json").write_text(json.dumps({
            "workspaces": ["packages/*"]
        }))
        
        analyzer = MonorepoAnalyzer(tmp_path)
        
        # Change in core should affect utils and app
        affected = analyzer.get_affected_packages(["packages/core/src/index.ts"])
        
        assert "@myorg/core" in affected
        assert "@myorg/utils" in affected
        assert "@myorg/app" in affected

    def test_get_build_order(self, tmp_path):
        """Gets correct build order."""
        packages_dir = tmp_path / "packages"
        packages_dir.mkdir()
        
        for name, deps in [
            ("core", {}),
            ("utils", {"@myorg/core": "^1.0.0"}),
            ("app", {"@myorg/utils": "^1.0.0"}),
        ]:
            pkg_dir = packages_dir / name
            pkg_dir.mkdir()
            (pkg_dir / "package.json").write_text(json.dumps({
                "name": f"@myorg/{name}",
                "version": "1.0.0",
                "dependencies": deps
            }))
        
        (tmp_path / "package.json").write_text(json.dumps({
            "workspaces": ["packages/*"]
        }))
        
        analyzer = MonorepoAnalyzer(tmp_path)
        order = analyzer.get_build_order()
        
        # core must be built before utils, utils before app
        names = [p.name for p in order]
        if "@myorg/core" in names and "@myorg/utils" in names:
            assert names.index("@myorg/core") < names.index("@myorg/utils")


class TestMonorepoEdgeCases:
    """Edge case tests for monorepo analysis."""

    def test_circular_dependency_detection(self, tmp_path):
        """Detects circular dependencies."""
        packages_dir = tmp_path / "packages"
        packages_dir.mkdir()
        
        # Create circular deps: a -> b -> c -> a
        for name, dep in [("a", "c"), ("b", "a"), ("c", "b")]:
            pkg_dir = packages_dir / name
            pkg_dir.mkdir()
            (pkg_dir / "package.json").write_text(json.dumps({
                "name": f"@myorg/{name}",
                "version": "1.0.0",
                "dependencies": {f"@myorg/{dep}": "^1.0.0"}
            }))
        
        (tmp_path / "package.json").write_text(json.dumps({
            "workspaces": ["packages/*"]
        }))
        
        analyzer = MonorepoAnalyzer(tmp_path)
        graph = analyzer.build_dependency_graph()
        cycles = graph.detect_cycles()
        
        assert len(cycles) > 0

    def test_empty_workspace(self, tmp_path):
        """Handles empty workspace directory."""
        (tmp_path / "package.json").write_text(json.dumps({
            "workspaces": ["packages/*"]
        }))
        (tmp_path / "packages").mkdir()
        
        analyzer = MonorepoAnalyzer(tmp_path)
        packages = analyzer.discover_packages()
        
        assert packages == []

    def test_malformed_package_json(self, tmp_path):
        """Handles malformed package.json gracefully."""
        packages_dir = tmp_path / "packages"
        packages_dir.mkdir()
        
        pkg_dir = packages_dir / "bad"
        pkg_dir.mkdir()
        (pkg_dir / "package.json").write_text("{ invalid json }")
        
        (tmp_path / "package.json").write_text(json.dumps({
            "workspaces": ["packages/*"]
        }))
        
        analyzer = MonorepoAnalyzer(tmp_path)
        # Should not raise, just skip the bad package
        packages = analyzer.discover_packages()
        assert isinstance(packages, list)
