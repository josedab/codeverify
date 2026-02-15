"""Tests for Monorepo Intelligence module."""

import json
from pathlib import Path

import pytest

from codeverify_core.monorepo import (
    DependencyEdge,
    MonorepoAnalyzer,
    PackageInfo,
    WorkspaceType,
)


class TestPackageInfo:
    """Tests for PackageInfo dataclass."""

    def test_package_creation(self):
        """PackageInfo can be created with required fields."""
        pkg = PackageInfo(
            name="@myorg/core",
            path=Path("/repo/packages/core"),
            version="1.0.0",
        )
        assert pkg.name == "@myorg/core"
        assert pkg.version == "1.0.0"
        assert pkg.dependencies == []

    def test_package_with_dependencies(self):
        """PackageInfo can have string dependencies."""
        pkg = PackageInfo(
            name="utils",
            path=Path("/repo/packages/utils"),
            version="2.0.0",
            dependencies=["lodash", "express"],
        )
        assert len(pkg.dependencies) == 2
        assert pkg.dependencies[0] == "lodash"

    def test_package_defaults(self):
        """PackageInfo has correct defaults."""
        pkg = PackageInfo(name="test", path=Path("/pkg"))
        assert pkg.version is None
        assert pkg.dependencies == []
        assert pkg.dev_dependencies == []
        assert pkg.peer_dependencies == []
        assert pkg.exports == []
        assert pkg.entry_points == []
        assert pkg.language == "unknown"


class TestDependencyEdge:
    """Tests for DependencyEdge dataclass."""

    def test_edge_creation(self):
        """DependencyEdge can be created with required fields."""
        edge = DependencyEdge(source="app", target="core")
        assert edge.source == "app"
        assert edge.target == "core"
        assert edge.dep_type == "runtime"

    def test_edge_with_dep_type(self):
        """DependencyEdge can have custom dep_type."""
        edge = DependencyEdge(source="app", target="core", dep_type="dev")
        assert edge.dep_type == "dev"


class TestMonorepoAnalyzer:
    """Tests for MonorepoAnalyzer."""

    def test_detect_nx(self, tmp_path):
        """Detects Nx monorepo."""
        (tmp_path / "nx.json").write_text('{"version": 2}')

        analyzer = MonorepoAnalyzer(tmp_path)
        assert analyzer._detect_workspace_type() == WorkspaceType.NX

    def test_detect_turborepo(self, tmp_path):
        """Detects Turborepo monorepo."""
        (tmp_path / "turbo.json").write_text('{"pipeline": {}}')

        analyzer = MonorepoAnalyzer(tmp_path)
        assert analyzer._detect_workspace_type() == WorkspaceType.TURBOREPO

    def test_detect_lerna(self, tmp_path):
        """Detects Lerna monorepo."""
        (tmp_path / "lerna.json").write_text('{"version": "1.0.0"}')

        analyzer = MonorepoAnalyzer(tmp_path)
        assert analyzer._detect_workspace_type() == WorkspaceType.LERNA

    def test_detect_pnpm(self, tmp_path):
        """Detects pnpm workspaces."""
        (tmp_path / "pnpm-workspace.yaml").write_text("packages:\n  - packages/*")

        analyzer = MonorepoAnalyzer(tmp_path)
        assert analyzer._detect_workspace_type() == WorkspaceType.PNPM

    def test_detect_yarn_workspaces(self, tmp_path):
        """Detects Yarn workspaces."""
        package_json = {"name": "monorepo", "workspaces": ["packages/*"]}
        (tmp_path / "package.json").write_text(json.dumps(package_json))

        analyzer = MonorepoAnalyzer(tmp_path)
        assert analyzer._detect_workspace_type() == WorkspaceType.YARN

    def test_detect_unknown(self, tmp_path):
        """Detects unknown workspace type."""
        (tmp_path / "package.json").write_text('{"name": "single-pkg"}')

        analyzer = MonorepoAnalyzer(tmp_path)
        assert analyzer._detect_workspace_type() == WorkspaceType.UNKNOWN

    @pytest.mark.asyncio
    async def test_discover_packages_npm(self, tmp_path):
        """Discovers npm packages."""
        # Create package structure
        packages_dir = tmp_path / "packages"
        packages_dir.mkdir()

        core_dir = packages_dir / "core"
        core_dir.mkdir()
        (core_dir / "package.json").write_text(
            json.dumps(
                {"name": "@myorg/core", "version": "1.0.0", "dependencies": {"lodash": "^4.0.0"}}
            )
        )

        utils_dir = packages_dir / "utils"
        utils_dir.mkdir()
        (utils_dir / "package.json").write_text(
            json.dumps(
                {
                    "name": "@myorg/utils",
                    "version": "1.0.0",
                    "dependencies": {"@myorg/core": "^1.0.0"},
                }
            )
        )

        # Create root with workspaces
        (tmp_path / "package.json").write_text(
            json.dumps({"name": "monorepo", "workspaces": ["packages/*"]})
        )

        analyzer = MonorepoAnalyzer(tmp_path)
        workspace_type = analyzer._detect_workspace_type()
        packages = await analyzer._discover_packages(workspace_type)

        assert len(packages) >= 2
        names = [p.name for p in packages]
        assert "@myorg/core" in names
        assert "@myorg/utils" in names

    @pytest.mark.asyncio
    async def test_build_dependency_graph(self, tmp_path):
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
            (pkg_dir / "package.json").write_text(
                json.dumps({"name": f"@myorg/{name}", "version": "1.0.0", "dependencies": deps})
            )

        (tmp_path / "package.json").write_text(json.dumps({"workspaces": ["packages/*"]}))

        analyzer = MonorepoAnalyzer(tmp_path)
        workspace_type = analyzer._detect_workspace_type()
        packages = await analyzer._discover_packages(workspace_type)
        analyzer._build_dependency_graph(packages)

        assert "@myorg/core" in analyzer._packages
        assert "@myorg/utils" in analyzer._packages
        assert "@myorg/app" in analyzer._packages

    @pytest.mark.asyncio
    async def test_get_dependents(self, tmp_path):
        """Gets packages that depend on a given package."""
        packages_dir = tmp_path / "packages"
        packages_dir.mkdir()

        for name, deps in [
            ("core", {}),
            ("utils", {"@myorg/core": "^1.0.0"}),
            ("app", {"@myorg/utils": "^1.0.0"}),
        ]:
            pkg_dir = packages_dir / name
            pkg_dir.mkdir()
            (pkg_dir / "package.json").write_text(
                json.dumps({"name": f"@myorg/{name}", "version": "1.0.0", "dependencies": deps})
            )

        (tmp_path / "package.json").write_text(json.dumps({"workspaces": ["packages/*"]}))

        analyzer = MonorepoAnalyzer(tmp_path)
        workspace_type = analyzer._detect_workspace_type()
        packages = await analyzer._discover_packages(workspace_type)
        analyzer._build_dependency_graph(packages)

        dependents = analyzer.get_dependents("@myorg/core")
        assert "@myorg/utils" in dependents

    @pytest.mark.asyncio
    async def test_get_transitive_dependents(self, tmp_path):
        """Gets transitively affected packages."""
        packages_dir = tmp_path / "packages"
        packages_dir.mkdir()

        for name, deps in [
            ("core", {}),
            ("utils", {"@myorg/core": "^1.0.0"}),
            ("app", {"@myorg/utils": "^1.0.0"}),
        ]:
            pkg_dir = packages_dir / name
            pkg_dir.mkdir()
            (pkg_dir / "package.json").write_text(
                json.dumps({"name": f"@myorg/{name}", "version": "1.0.0", "dependencies": deps})
            )

        (tmp_path / "package.json").write_text(json.dumps({"workspaces": ["packages/*"]}))

        analyzer = MonorepoAnalyzer(tmp_path)
        workspace_type = analyzer._detect_workspace_type()
        packages = await analyzer._discover_packages(workspace_type)
        analyzer._build_dependency_graph(packages)

        # Change in core should transitively affect utils and app
        affected = analyzer.get_transitive_dependents("@myorg/core")
        assert "@myorg/utils" in affected
        assert "@myorg/app" in affected


class TestMonorepoEdgeCases:
    """Edge case tests for monorepo analysis."""

    @pytest.mark.asyncio
    async def test_circular_dependency_detection(self, tmp_path):
        """Detects circular dependencies."""
        packages_dir = tmp_path / "packages"
        packages_dir.mkdir()

        # Create circular deps: a -> c, b -> a, c -> b
        for name, dep in [("a", "c"), ("b", "a"), ("c", "b")]:
            pkg_dir = packages_dir / name
            pkg_dir.mkdir()
            (pkg_dir / "package.json").write_text(
                json.dumps(
                    {
                        "name": f"@myorg/{name}",
                        "version": "1.0.0",
                        "dependencies": {f"@myorg/{dep}": "^1.0.0"},
                    }
                )
            )

        (tmp_path / "package.json").write_text(json.dumps({"workspaces": ["packages/*"]}))

        analyzer = MonorepoAnalyzer(tmp_path)
        workspace_type = analyzer._detect_workspace_type()
        packages = await analyzer._discover_packages(workspace_type)
        analyzer._build_dependency_graph(packages)
        cycles = analyzer._detect_cycles()

        assert len(cycles) > 0

    @pytest.mark.asyncio
    async def test_empty_workspace(self, tmp_path):
        """Handles empty workspace directory."""
        (tmp_path / "package.json").write_text(json.dumps({"workspaces": ["packages/*"]}))
        (tmp_path / "packages").mkdir()

        analyzer = MonorepoAnalyzer(tmp_path)
        workspace_type = analyzer._detect_workspace_type()
        packages = await analyzer._discover_packages(workspace_type)

        assert packages == []

    @pytest.mark.asyncio
    async def test_malformed_package_json(self, tmp_path):
        """Handles malformed package.json gracefully."""
        packages_dir = tmp_path / "packages"
        packages_dir.mkdir()

        pkg_dir = packages_dir / "bad"
        pkg_dir.mkdir()
        (pkg_dir / "package.json").write_text("{ invalid json }")

        (tmp_path / "package.json").write_text(json.dumps({"workspaces": ["packages/*"]}))

        analyzer = MonorepoAnalyzer(tmp_path)
        workspace_type = analyzer._detect_workspace_type()
        # Should not raise, just skip the bad package
        packages = await analyzer._discover_packages(workspace_type)
        assert isinstance(packages, list)
