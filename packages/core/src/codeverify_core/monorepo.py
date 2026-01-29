"""Monorepo Intelligence - Cross-package dependency analysis for monorepos.

Supports Nx, Turborepo, Lerna, pnpm workspaces, and Yarn workspaces.
Provides cross-boundary verification and impact radius mapping.
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


class WorkspaceType(str, Enum):
    """Supported monorepo workspace types."""
    NX = "nx"
    TURBOREPO = "turborepo"
    LERNA = "lerna"
    PNPM = "pnpm"
    YARN = "yarn"
    NPM = "npm"
    PYTHON_MONOREPO = "python_monorepo"
    UNKNOWN = "unknown"


@dataclass
class PackageInfo:
    """Information about a package in the monorepo."""
    name: str
    path: Path
    version: str | None = None
    dependencies: list[str] = field(default_factory=list)
    dev_dependencies: list[str] = field(default_factory=list)
    peer_dependencies: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    language: str = "unknown"


@dataclass
class DependencyEdge:
    """An edge in the dependency graph."""
    source: str  # Package name
    target: str  # Dependency package name
    dep_type: str = "runtime"  # runtime, dev, peer


@dataclass
class InterfaceContract:
    """Contract for a cross-package interface."""
    package: str
    name: str
    signature: str
    parameters: list[dict[str, Any]] = field(default_factory=list)
    return_type: str | None = None
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    source_file: str | None = None
    line_number: int | None = None


@dataclass
class ImpactAnalysis:
    """Analysis of impact from a change."""
    changed_package: str
    changed_files: list[str]
    directly_affected: list[str]  # Packages directly depending on changed package
    transitively_affected: list[str]  # All packages affected through deps
    affected_contracts: list[InterfaceContract] = field(default_factory=list)
    risk_level: str = "low"  # low, medium, high, critical
    recommendations: list[str] = field(default_factory=list)


@dataclass
class MonorepoAnalysis:
    """Complete monorepo analysis result."""
    workspace_type: WorkspaceType
    root_path: Path
    packages: list[PackageInfo]
    dependency_graph: list[DependencyEdge]
    interfaces: list[InterfaceContract]
    cycles: list[list[str]]  # Circular dependency chains
    metadata: dict[str, Any] = field(default_factory=dict)


class MonorepoAnalyzer:
    """
    Analyzer for monorepo structures.
    
    Detects workspace configuration, builds dependency graphs,
    and identifies cross-package interfaces.
    """

    def __init__(self, repo_root: Path | str) -> None:
        """Initialize analyzer with repository root path."""
        self.repo_root = Path(repo_root)
        self._packages: dict[str, PackageInfo] = {}
        self._dependency_graph: list[DependencyEdge] = []

    async def analyze(self) -> MonorepoAnalysis:
        """Perform complete monorepo analysis."""
        logger.info("Starting monorepo analysis", root=str(self.repo_root))
        
        # Detect workspace type
        workspace_type = self._detect_workspace_type()
        logger.info("Detected workspace type", type=workspace_type.value)
        
        # Discover packages based on workspace type
        packages = await self._discover_packages(workspace_type)
        
        # Build dependency graph
        self._build_dependency_graph(packages)
        
        # Extract interfaces
        interfaces = await self._extract_interfaces(packages)
        
        # Detect cycles
        cycles = self._detect_cycles()
        
        return MonorepoAnalysis(
            workspace_type=workspace_type,
            root_path=self.repo_root,
            packages=packages,
            dependency_graph=self._dependency_graph,
            interfaces=interfaces,
            cycles=cycles,
        )

    def _detect_workspace_type(self) -> WorkspaceType:
        """Detect the type of monorepo workspace."""
        # Check for Nx
        if (self.repo_root / "nx.json").exists():
            return WorkspaceType.NX
        
        # Check for Turborepo
        if (self.repo_root / "turbo.json").exists():
            return WorkspaceType.TURBOREPO
        
        # Check for Lerna
        if (self.repo_root / "lerna.json").exists():
            return WorkspaceType.LERNA
        
        # Check for pnpm workspaces
        if (self.repo_root / "pnpm-workspace.yaml").exists():
            return WorkspaceType.PNPM
        
        # Check for Yarn workspaces in package.json
        pkg_json = self.repo_root / "package.json"
        if pkg_json.exists():
            try:
                data = json.loads(pkg_json.read_text())
                if "workspaces" in data:
                    return WorkspaceType.YARN
            except (json.JSONDecodeError, OSError):
                pass
        
        # Check for Python monorepo (pyproject.toml with packages)
        pyproject = self.repo_root / "pyproject.toml"
        if pyproject.exists():
            content = pyproject.read_text()
            if "[tool.hatch" in content or "packages" in content:
                return WorkspaceType.PYTHON_MONOREPO
        
        return WorkspaceType.UNKNOWN

    async def _discover_packages(self, workspace_type: WorkspaceType) -> list[PackageInfo]:
        """Discover packages in the monorepo."""
        packages: list[PackageInfo] = []
        
        if workspace_type == WorkspaceType.NX:
            packages = await self._discover_nx_packages()
        elif workspace_type == WorkspaceType.TURBOREPO:
            packages = await self._discover_turbo_packages()
        elif workspace_type == WorkspaceType.LERNA:
            packages = await self._discover_lerna_packages()
        elif workspace_type in (WorkspaceType.PNPM, WorkspaceType.YARN, WorkspaceType.NPM):
            packages = await self._discover_npm_workspace_packages()
        elif workspace_type == WorkspaceType.PYTHON_MONOREPO:
            packages = await self._discover_python_packages()
        else:
            packages = await self._discover_generic_packages()
        
        # Store for later reference
        for pkg in packages:
            self._packages[pkg.name] = pkg
        
        logger.info("Discovered packages", count=len(packages))
        return packages

    async def _discover_nx_packages(self) -> list[PackageInfo]:
        """Discover packages in an Nx workspace."""
        packages: list[PackageInfo] = []
        
        # Check nx.json for project locations
        nx_json = self.repo_root / "nx.json"
        if nx_json.exists():
            try:
                config = json.loads(nx_json.read_text())
                # Nx can define workspaceLayout
                layout = config.get("workspaceLayout", {})
                apps_dir = layout.get("appsDir", "apps")
                libs_dir = layout.get("libsDir", "libs")
                
                # Scan apps and libs directories
                for dir_name in [apps_dir, libs_dir]:
                    dir_path = self.repo_root / dir_name
                    if dir_path.exists():
                        for subdir in dir_path.iterdir():
                            if subdir.is_dir():
                                pkg = await self._parse_package_dir(subdir)
                                if pkg:
                                    packages.append(pkg)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to parse nx.json", error=str(e))
        
        # Also check project.json files
        for project_json in self.repo_root.rglob("project.json"):
            if "node_modules" in str(project_json):
                continue
            pkg_dir = project_json.parent
            pkg = await self._parse_package_dir(pkg_dir)
            if pkg and pkg.name not in [p.name for p in packages]:
                packages.append(pkg)
        
        return packages

    async def _discover_turbo_packages(self) -> list[PackageInfo]:
        """Discover packages in a Turborepo workspace."""
        return await self._discover_npm_workspace_packages()

    async def _discover_lerna_packages(self) -> list[PackageInfo]:
        """Discover packages in a Lerna workspace."""
        packages: list[PackageInfo] = []
        
        lerna_json = self.repo_root / "lerna.json"
        if lerna_json.exists():
            try:
                config = json.loads(lerna_json.read_text())
                pkg_patterns = config.get("packages", ["packages/*"])
                
                for pattern in pkg_patterns:
                    # Convert glob to directory search
                    base_dir = pattern.rstrip("/*")
                    search_dir = self.repo_root / base_dir
                    
                    if search_dir.exists():
                        for subdir in search_dir.iterdir():
                            if subdir.is_dir():
                                pkg = await self._parse_package_dir(subdir)
                                if pkg:
                                    packages.append(pkg)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to parse lerna.json", error=str(e))
        
        return packages

    async def _discover_npm_workspace_packages(self) -> list[PackageInfo]:
        """Discover packages in npm/yarn/pnpm workspaces."""
        packages: list[PackageInfo] = []
        
        # Read workspaces from package.json
        pkg_json = self.repo_root / "package.json"
        if pkg_json.exists():
            try:
                data = json.loads(pkg_json.read_text())
                workspaces = data.get("workspaces", [])
                
                # Handle both array and object format
                if isinstance(workspaces, dict):
                    workspaces = workspaces.get("packages", [])
                
                for pattern in workspaces:
                    # Convert glob pattern to search
                    if pattern.endswith("/*"):
                        base_dir = pattern[:-2]
                        search_dir = self.repo_root / base_dir
                        
                        if search_dir.exists():
                            for subdir in search_dir.iterdir():
                                if subdir.is_dir():
                                    pkg = await self._parse_package_dir(subdir)
                                    if pkg:
                                        packages.append(pkg)
                    else:
                        pkg_dir = self.repo_root / pattern
                        if pkg_dir.exists():
                            pkg = await self._parse_package_dir(pkg_dir)
                            if pkg:
                                packages.append(pkg)
                                
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to parse package.json", error=str(e))
        
        return packages

    async def _discover_python_packages(self) -> list[PackageInfo]:
        """Discover packages in a Python monorepo."""
        packages: list[PackageInfo] = []
        
        # Look for packages directory structure
        for subdir in ["packages", "libs", "apps"]:
            pkg_dir = self.repo_root / subdir
            if pkg_dir.exists():
                for child in pkg_dir.iterdir():
                    if child.is_dir():
                        pkg = await self._parse_python_package(child)
                        if pkg:
                            packages.append(pkg)
        
        return packages

    async def _discover_generic_packages(self) -> list[PackageInfo]:
        """Generic package discovery for unknown workspace types."""
        packages: list[PackageInfo] = []
        
        # Look for common patterns
        for subdir in ["packages", "libs", "apps", "modules"]:
            pkg_dir = self.repo_root / subdir
            if pkg_dir.exists():
                for child in pkg_dir.iterdir():
                    if child.is_dir():
                        pkg = await self._parse_package_dir(child)
                        if pkg:
                            packages.append(pkg)
        
        return packages

    async def _parse_package_dir(self, pkg_dir: Path) -> PackageInfo | None:
        """Parse a package directory for package information."""
        pkg_json = pkg_dir / "package.json"
        
        if pkg_json.exists():
            try:
                data = json.loads(pkg_json.read_text())
                return PackageInfo(
                    name=data.get("name", pkg_dir.name),
                    path=pkg_dir,
                    version=data.get("version"),
                    dependencies=list(data.get("dependencies", {}).keys()),
                    dev_dependencies=list(data.get("devDependencies", {}).keys()),
                    peer_dependencies=list(data.get("peerDependencies", {}).keys()),
                    exports=self._extract_exports(data),
                    entry_points=self._extract_entry_points(data),
                    language="typescript" if (pkg_dir / "tsconfig.json").exists() else "javascript",
                )
            except (json.JSONDecodeError, OSError):
                pass
        
        # Try Python package
        return await self._parse_python_package(pkg_dir)

    async def _parse_python_package(self, pkg_dir: Path) -> PackageInfo | None:
        """Parse a Python package directory."""
        pyproject = pkg_dir / "pyproject.toml"
        setup_py = pkg_dir / "setup.py"
        
        if pyproject.exists():
            try:
                content = pyproject.read_text()
                name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', content)
                version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
                
                deps = []
                deps_match = re.search(r'dependencies\s*=\s*\[(.*?)\]', content, re.DOTALL)
                if deps_match:
                    deps = re.findall(r'["\']([^"\'>=<\[]+)', deps_match.group(1))
                
                return PackageInfo(
                    name=name_match.group(1) if name_match else pkg_dir.name,
                    path=pkg_dir,
                    version=version_match.group(1) if version_match else None,
                    dependencies=deps,
                    language="python",
                )
            except OSError:
                pass
        elif setup_py.exists():
            return PackageInfo(
                name=pkg_dir.name,
                path=pkg_dir,
                language="python",
            )
        
        return None

    def _extract_exports(self, pkg_data: dict) -> list[str]:
        """Extract export paths from package.json."""
        exports = []
        
        # Standard exports field
        if "exports" in pkg_data:
            exp = pkg_data["exports"]
            if isinstance(exp, str):
                exports.append(exp)
            elif isinstance(exp, dict):
                exports.extend(exp.keys())
        
        # Main entry
        if "main" in pkg_data:
            exports.append(pkg_data["main"])
        
        return exports

    def _extract_entry_points(self, pkg_data: dict) -> list[str]:
        """Extract entry points from package.json."""
        entry_points = []
        
        for key in ["main", "module", "types", "typings"]:
            if key in pkg_data:
                entry_points.append(pkg_data[key])
        
        return entry_points

    def _build_dependency_graph(self, packages: list[PackageInfo]) -> None:
        """Build the dependency graph from discovered packages."""
        self._dependency_graph = []
        
        pkg_names = {pkg.name for pkg in packages}
        
        for pkg in packages:
            # Runtime dependencies
            for dep in pkg.dependencies:
                if dep in pkg_names:
                    self._dependency_graph.append(DependencyEdge(
                        source=pkg.name,
                        target=dep,
                        dep_type="runtime",
                    ))
            
            # Dev dependencies
            for dep in pkg.dev_dependencies:
                if dep in pkg_names:
                    self._dependency_graph.append(DependencyEdge(
                        source=pkg.name,
                        target=dep,
                        dep_type="dev",
                    ))
            
            # Peer dependencies
            for dep in pkg.peer_dependencies:
                if dep in pkg_names:
                    self._dependency_graph.append(DependencyEdge(
                        source=pkg.name,
                        target=dep,
                        dep_type="peer",
                    ))

    async def _extract_interfaces(self, packages: list[PackageInfo]) -> list[InterfaceContract]:
        """Extract interface contracts from packages."""
        interfaces: list[InterfaceContract] = []
        
        for pkg in packages:
            if pkg.language in ("typescript", "javascript"):
                pkg_interfaces = await self._extract_ts_interfaces(pkg)
            elif pkg.language == "python":
                pkg_interfaces = await self._extract_python_interfaces(pkg)
            else:
                pkg_interfaces = []
            
            interfaces.extend(pkg_interfaces)
        
        return interfaces

    async def _extract_ts_interfaces(self, pkg: PackageInfo) -> list[InterfaceContract]:
        """Extract interfaces from TypeScript packages."""
        interfaces: list[InterfaceContract] = []
        
        # Look for index.ts or main entry point
        for pattern in ["src/index.ts", "index.ts", "lib/index.ts"]:
            entry = pkg.path / pattern
            if entry.exists():
                content = entry.read_text()
                interfaces.extend(self._parse_ts_exports(pkg.name, content, str(entry)))
                break
        
        # Also check d.ts files
        for dts in pkg.path.rglob("*.d.ts"):
            if "node_modules" in str(dts):
                continue
            content = dts.read_text()
            interfaces.extend(self._parse_ts_exports(pkg.name, content, str(dts)))
        
        return interfaces

    def _parse_ts_exports(self, pkg_name: str, content: str, source_file: str) -> list[InterfaceContract]:
        """Parse TypeScript content for exported interfaces."""
        interfaces: list[InterfaceContract] = []
        
        # Match exported functions
        func_pattern = r'export\s+(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)\s*(?::\s*([^{;]+))?'
        for match in re.finditer(func_pattern, content):
            name = match.group(1)
            params_str = match.group(2)
            return_type = match.group(3).strip() if match.group(3) else None
            
            params = self._parse_ts_params(params_str)
            
            interfaces.append(InterfaceContract(
                package=pkg_name,
                name=name,
                signature=f"function {name}({params_str}): {return_type or 'void'}",
                parameters=params,
                return_type=return_type,
                source_file=source_file,
            ))
        
        # Match exported interfaces
        interface_pattern = r'export\s+interface\s+(\w+)\s*\{([^}]*)\}'
        for match in re.finditer(interface_pattern, content, re.DOTALL):
            name = match.group(1)
            body = match.group(2)
            
            interfaces.append(InterfaceContract(
                package=pkg_name,
                name=name,
                signature=f"interface {name}",
                source_file=source_file,
            ))
        
        return interfaces

    def _parse_ts_params(self, params_str: str) -> list[dict[str, Any]]:
        """Parse TypeScript function parameters."""
        params = []
        if not params_str.strip():
            return params
        
        for param in params_str.split(","):
            param = param.strip()
            if ":" in param:
                name, type_str = param.split(":", 1)
                params.append({
                    "name": name.strip().lstrip("?"),
                    "type": type_str.strip(),
                    "optional": "?" in name,
                })
            elif param:
                params.append({"name": param, "type": "any", "optional": False})
        
        return params

    async def _extract_python_interfaces(self, pkg: PackageInfo) -> list[InterfaceContract]:
        """Extract interfaces from Python packages."""
        interfaces: list[InterfaceContract] = []
        
        # Look for __init__.py
        for pattern in ["src/*/__init__.py", "__init__.py"]:
            for init_file in pkg.path.glob(pattern):
                content = init_file.read_text()
                interfaces.extend(self._parse_python_exports(pkg.name, content, str(init_file)))
        
        return interfaces

    def _parse_python_exports(self, pkg_name: str, content: str, source_file: str) -> list[InterfaceContract]:
        """Parse Python content for exported functions/classes."""
        interfaces: list[InterfaceContract] = []
        
        # Match function definitions
        func_pattern = r'^(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*([^:]+))?:'
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            name = match.group(1)
            if name.startswith("_"):  # Skip private
                continue
            
            params_str = match.group(2)
            return_type = match.group(3).strip() if match.group(3) else None
            
            interfaces.append(InterfaceContract(
                package=pkg_name,
                name=name,
                signature=f"def {name}({params_str}) -> {return_type or 'None'}",
                return_type=return_type,
                source_file=source_file,
            ))
        
        # Match class definitions
        class_pattern = r'^class\s+(\w+)\s*(?:\([^)]*\))?\s*:'
        for match in re.finditer(class_pattern, content, re.MULTILINE):
            name = match.group(1)
            if name.startswith("_"):
                continue
            
            interfaces.append(InterfaceContract(
                package=pkg_name,
                name=name,
                signature=f"class {name}",
                source_file=source_file,
            ))
        
        return interfaces

    def _detect_cycles(self) -> list[list[str]]:
        """Detect circular dependencies in the dependency graph."""
        cycles: list[list[str]] = []
        
        # Build adjacency list
        adj: dict[str, list[str]] = {}
        for edge in self._dependency_graph:
            if edge.source not in adj:
                adj[edge.source] = []
            adj[edge.source].append(edge.target)
        
        # DFS-based cycle detection
        visited: set[str] = set()
        rec_stack: set[str] = set()
        
        def dfs(node: str, path: list[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    if cycle not in cycles:
                        cycles.append(cycle)
            
            rec_stack.remove(node)
        
        for pkg in self._packages:
            if pkg not in visited:
                dfs(pkg, [])
        
        return cycles

    def get_dependents(self, package_name: str) -> list[str]:
        """Get all packages that depend on the given package."""
        dependents = []
        for edge in self._dependency_graph:
            if edge.target == package_name:
                dependents.append(edge.source)
        return dependents

    def get_dependencies(self, package_name: str) -> list[str]:
        """Get all packages that the given package depends on."""
        dependencies = []
        for edge in self._dependency_graph:
            if edge.source == package_name:
                dependencies.append(edge.target)
        return dependencies

    def get_transitive_dependents(self, package_name: str) -> list[str]:
        """Get all packages transitively affected by changes to the given package."""
        affected: set[str] = set()
        queue = [package_name]
        
        while queue:
            current = queue.pop(0)
            for edge in self._dependency_graph:
                if edge.target == current and edge.source not in affected:
                    affected.add(edge.source)
                    queue.append(edge.source)
        
        return list(affected)

    async def analyze_impact(
        self,
        changed_package: str,
        changed_files: list[str],
    ) -> ImpactAnalysis:
        """Analyze the impact of changes to a package."""
        directly_affected = self.get_dependents(changed_package)
        transitively_affected = self.get_transitive_dependents(changed_package)
        
        # Find affected contracts
        affected_contracts = []
        pkg = self._packages.get(changed_package)
        if pkg:
            # Check if changed files include interface definitions
            for contract in await self._extract_interfaces([pkg]):
                if contract.source_file and any(
                    cf in contract.source_file for cf in changed_files
                ):
                    affected_contracts.append(contract)
        
        # Determine risk level
        risk_level = self._calculate_impact_risk(
            len(directly_affected),
            len(transitively_affected),
            len(affected_contracts),
        )
        
        # Generate recommendations
        recommendations = self._generate_impact_recommendations(
            changed_package,
            directly_affected,
            affected_contracts,
            risk_level,
        )
        
        return ImpactAnalysis(
            changed_package=changed_package,
            changed_files=changed_files,
            directly_affected=directly_affected,
            transitively_affected=transitively_affected,
            affected_contracts=affected_contracts,
            risk_level=risk_level,
            recommendations=recommendations,
        )

    def _calculate_impact_risk(
        self,
        direct_count: int,
        transitive_count: int,
        contract_count: int,
    ) -> str:
        """Calculate risk level based on impact metrics."""
        score = direct_count * 2 + transitive_count + contract_count * 3
        
        if score >= 15:
            return "critical"
        elif score >= 10:
            return "high"
        elif score >= 5:
            return "medium"
        else:
            return "low"

    def _generate_impact_recommendations(
        self,
        changed_package: str,
        directly_affected: list[str],
        affected_contracts: list[InterfaceContract],
        risk_level: str,
    ) -> list[str]:
        """Generate recommendations based on impact analysis."""
        recommendations = []
        
        if risk_level in ("critical", "high"):
            recommendations.append(
                f"High impact change: {len(directly_affected)} packages directly affected"
            )
            recommendations.append(
                "Consider adding integration tests across affected packages"
            )
        
        if affected_contracts:
            recommendations.append(
                f"{len(affected_contracts)} public interfaces modified - verify backwards compatibility"
            )
        
        if directly_affected:
            recommendations.append(
                f"Run tests for: {', '.join(directly_affected[:5])}"
                + (f" and {len(directly_affected) - 5} more" if len(directly_affected) > 5 else "")
            )
        
        return recommendations

    def to_mermaid_graph(self) -> str:
        """Generate a Mermaid diagram of the dependency graph."""
        lines = ["graph TD"]
        
        for edge in self._dependency_graph:
            style = "-->" if edge.dep_type == "runtime" else "-.->||"
            lines.append(f"    {edge.source.replace('@', '_').replace('/', '_')} {style} {edge.target.replace('@', '_').replace('/', '_')}")
        
        return "\n".join(lines)
