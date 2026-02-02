"""Organization-wide Dependency Graph Analysis.

Extends cross-repository analysis with:
1. Organization-wide dependency tracking
2. Transitive risk detection
3. Dependency health scoring
4. Security propagation analysis
5. Visualization data generation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import structlog

logger = structlog.get_logger()


class RiskLevel(str, Enum):
    """Risk level for dependencies."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class DependencyHealth(str, Enum):
    """Health status of a dependency relationship."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    AT_RISK = "at_risk"
    BROKEN = "broken"
    UNKNOWN = "unknown"


@dataclass
class OrgRepository:
    """A repository within an organization."""
    name: str
    org: str
    description: str = ""
    language: str = "unknown"
    repo_type: str = "library"  # library, service, application, package
    visibility: str = "internal"  # public, internal, private
    team: str | None = None
    
    # Dependency metadata
    direct_dependencies: list[str] = field(default_factory=list)
    dev_dependencies: list[str] = field(default_factory=list)
    
    # Health and risk
    last_verified: datetime | None = None
    verification_status: str = "unknown"  # passed, failed, partial, unknown
    known_vulnerabilities: int = 0
    risk_score: float = 0.0
    
    # Metrics
    stars: int = 0
    open_issues: int = 0
    last_commit: datetime | None = None
    contributors: int = 0

    @property
    def full_name(self) -> str:
        """Get full repository name."""
        return f"{self.org}/{self.name}"


@dataclass
class DependencyEdge:
    """An edge in the dependency graph."""
    source: str  # Full repo name
    target: str  # Full repo name
    dependency_type: str = "direct"  # direct, dev, peer, optional, transitive
    version_constraint: str | None = None
    depth: int = 1  # 1 = direct, 2+ = transitive
    
    # Risk propagation
    propagates_risk: bool = True
    risk_multiplier: float = 1.0


@dataclass
class TransitiveRisk:
    """Risk propagated through the dependency chain."""
    source_repo: str
    vulnerability_id: str | None = None
    risk_type: str = "security"  # security, stability, maintenance, license
    severity: RiskLevel = RiskLevel.MEDIUM
    affected_path: list[str] = field(default_factory=list)
    description: str = ""
    remediation: str | None = None


@dataclass
class DependencyCluster:
    """A cluster of related repositories."""
    cluster_id: str
    name: str
    repos: list[str] = field(default_factory=list)
    central_repo: str | None = None
    team: str | None = None
    cohesion_score: float = 0.0  # How tightly coupled


@dataclass
class OrgDependencyMetrics:
    """Organization-wide dependency metrics."""
    total_repos: int = 0
    total_dependencies: int = 0
    total_transitive: int = 0
    avg_dependency_depth: float = 0.0
    max_dependency_depth: int = 0
    circular_dependencies: int = 0
    orphan_repos: int = 0
    
    # Risk metrics
    repos_with_critical_risk: int = 0
    repos_with_high_risk: int = 0
    total_vulnerabilities: int = 0
    avg_risk_score: float = 0.0
    
    # Health metrics
    healthy_repos: int = 0
    degraded_repos: int = 0
    at_risk_repos: int = 0


class OrgDependencyGraph:
    """Organization-wide dependency graph with risk analysis."""

    def __init__(self, org: str) -> None:
        self.org = org
        self._repos: dict[str, OrgRepository] = {}
        self._edges: list[DependencyEdge] = []
        self._transitive_risks: list[TransitiveRisk] = []
        self._clusters: list[DependencyCluster] = []
        self._adjacency: dict[str, set[str]] = {}  # repo -> dependents
        self._reverse_adjacency: dict[str, set[str]] = {}  # repo -> dependencies

    def add_repository(self, repo: OrgRepository) -> None:
        """Add a repository to the graph."""
        self._repos[repo.full_name] = repo
        if repo.full_name not in self._adjacency:
            self._adjacency[repo.full_name] = set()
        if repo.full_name not in self._reverse_adjacency:
            self._reverse_adjacency[repo.full_name] = set()

    def add_dependency(
        self,
        source: str,
        target: str,
        dependency_type: str = "direct",
        version_constraint: str | None = None,
    ) -> None:
        """Add a dependency edge."""
        edge = DependencyEdge(
            source=source,
            target=target,
            dependency_type=dependency_type,
            version_constraint=version_constraint,
        )
        self._edges.append(edge)
        
        # Update adjacency
        if source not in self._adjacency:
            self._adjacency[source] = set()
        if target not in self._reverse_adjacency:
            self._reverse_adjacency[target] = set()
        
        self._adjacency[source].add(target)
        self._reverse_adjacency[target].add(source)

    def get_direct_dependencies(self, repo: str) -> list[str]:
        """Get direct dependencies of a repository."""
        return list(self._adjacency.get(repo, set()))

    def get_direct_dependents(self, repo: str) -> list[str]:
        """Get repositories that directly depend on this one."""
        return list(self._reverse_adjacency.get(repo, set()))

    def get_transitive_dependencies(
        self,
        repo: str,
        max_depth: int = 10,
    ) -> dict[str, int]:
        """Get all transitive dependencies with their depth."""
        result: dict[str, int] = {}
        visited: set[str] = set()
        
        def dfs(current: str, depth: int) -> None:
            if depth > max_depth or current in visited:
                return
            visited.add(current)
            
            for dep in self._adjacency.get(current, set()):
                if dep not in result or result[dep] > depth:
                    result[dep] = depth
                dfs(dep, depth + 1)
        
        dfs(repo, 1)
        return result

    def get_transitive_dependents(
        self,
        repo: str,
        max_depth: int = 10,
    ) -> dict[str, int]:
        """Get all repositories transitively depending on this one."""
        result: dict[str, int] = {}
        visited: set[str] = set()
        
        def dfs(current: str, depth: int) -> None:
            if depth > max_depth or current in visited:
                return
            visited.add(current)
            
            for dependent in self._reverse_adjacency.get(current, set()):
                if dependent not in result or result[dependent] > depth:
                    result[dependent] = depth
                dfs(dependent, depth + 1)
        
        dfs(repo, 1)
        return result

    def detect_circular_dependencies(self) -> list[list[str]]:
        """Detect circular dependencies in the graph."""
        cycles: list[list[str]] = []
        visited: set[str] = set()
        rec_stack: set[str] = set()
        path: list[str] = []
        
        def dfs(node: str) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self._adjacency.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    if cycle not in cycles:
                        cycles.append(cycle)
            
            path.pop()
            rec_stack.remove(node)
        
        for node in self._repos:
            if node not in visited:
                dfs(node)
        
        return cycles

    def analyze_transitive_risks(self) -> list[TransitiveRisk]:
        """Analyze risks that propagate through dependencies."""
        risks: list[TransitiveRisk] = []
        
        # Find repos with vulnerabilities
        vulnerable_repos = [
            repo for repo in self._repos.values()
            if repo.known_vulnerabilities > 0
        ]
        
        for vuln_repo in vulnerable_repos:
            # Find all repos affected by this vulnerability
            dependents = self.get_transitive_dependents(vuln_repo.full_name)
            
            for dependent, depth in dependents.items():
                # Calculate risk severity based on depth and vulnerability count
                if depth == 1:
                    severity = RiskLevel.HIGH if vuln_repo.known_vulnerabilities > 5 else RiskLevel.MEDIUM
                elif depth == 2:
                    severity = RiskLevel.MEDIUM
                else:
                    severity = RiskLevel.LOW
                
                # Build the path
                path = self._find_path(dependent, vuln_repo.full_name)
                
                risk = TransitiveRisk(
                    source_repo=vuln_repo.full_name,
                    risk_type="security",
                    severity=severity,
                    affected_path=path,
                    description=(
                        f"{vuln_repo.full_name} has {vuln_repo.known_vulnerabilities} "
                        f"vulnerabilities affecting {dependent} (depth: {depth})"
                    ),
                    remediation=f"Update {vuln_repo.full_name} or find alternative",
                )
                risks.append(risk)
        
        self._transitive_risks = risks
        return risks

    def _find_path(self, source: str, target: str) -> list[str]:
        """Find a path from source to target in the dependency graph."""
        if source == target:
            return [source]
        
        visited: set[str] = set()
        queue: list[tuple[str, list[str]]] = [(source, [source])]
        
        while queue:
            node, path = queue.pop(0)
            
            if node in visited:
                continue
            visited.add(node)
            
            for neighbor in self._adjacency.get(node, set()):
                if neighbor == target:
                    return path + [neighbor]
                queue.append((neighbor, path + [neighbor]))
        
        return []

    def calculate_risk_scores(self) -> None:
        """Calculate risk scores for all repositories."""
        # First pass: direct risk
        for repo in self._repos.values():
            direct_risk = repo.known_vulnerabilities * 10
            
            # Adjust for maintenance indicators
            if repo.last_commit:
                days_since_commit = (datetime.utcnow() - repo.last_commit).days
                if days_since_commit > 365:
                    direct_risk += 20  # Unmaintained
                elif days_since_commit > 180:
                    direct_risk += 10
            
            # Adjust for open issues
            if repo.open_issues > 100:
                direct_risk += 10
            elif repo.open_issues > 50:
                direct_risk += 5
            
            repo.risk_score = min(100.0, direct_risk)
        
        # Second pass: propagate risk from dependencies
        for repo in self._repos.values():
            deps = self.get_transitive_dependencies(repo.full_name)
            inherited_risk = 0.0
            
            for dep, depth in deps.items():
                if dep in self._repos:
                    dep_repo = self._repos[dep]
                    # Risk decreases with depth
                    inherited_risk += dep_repo.risk_score / (depth * 2)
            
            # Cap inherited risk contribution
            repo.risk_score = min(100.0, repo.risk_score + min(inherited_risk, 30))

    def detect_clusters(self, min_cohesion: float = 0.3) -> list[DependencyCluster]:
        """Detect clusters of tightly coupled repositories."""
        clusters: list[DependencyCluster] = []
        visited: set[str] = set()
        
        for repo_name in self._repos:
            if repo_name in visited:
                continue
            
            # Find strongly connected component
            component = self._find_connected_component(repo_name)
            visited.update(component)
            
            if len(component) >= 2:
                # Calculate cohesion (internal edges / possible edges)
                internal_edges = 0
                for r in component:
                    for dep in self._adjacency.get(r, set()):
                        if dep in component:
                            internal_edges += 1
                
                max_edges = len(component) * (len(component) - 1)
                cohesion = internal_edges / max_edges if max_edges > 0 else 0
                
                if cohesion >= min_cohesion:
                    # Find central repo (most connections)
                    central = max(
                        component,
                        key=lambda r: len(self._adjacency.get(r, set()) & component) +
                                     len(self._reverse_adjacency.get(r, set()) & component)
                    )
                    
                    cluster = DependencyCluster(
                        cluster_id=f"cluster-{len(clusters)}",
                        name=f"Cluster around {central.split('/')[-1]}",
                        repos=list(component),
                        central_repo=central,
                        cohesion_score=cohesion,
                    )
                    clusters.append(cluster)
        
        self._clusters = clusters
        return clusters

    def _find_connected_component(self, start: str) -> set[str]:
        """Find connected component containing start node (ignoring direction)."""
        component: set[str] = set()
        queue = [start]
        
        while queue:
            node = queue.pop(0)
            if node in component:
                continue
            component.add(node)
            
            # Add neighbors in both directions
            for neighbor in self._adjacency.get(node, set()):
                queue.append(neighbor)
            for neighbor in self._reverse_adjacency.get(node, set()):
                queue.append(neighbor)
        
        return component

    def get_metrics(self) -> OrgDependencyMetrics:
        """Calculate organization-wide metrics."""
        metrics = OrgDependencyMetrics()
        
        metrics.total_repos = len(self._repos)
        metrics.total_dependencies = len(self._edges)
        
        # Calculate transitive dependencies count and depth
        total_transitive = 0
        max_depth = 0
        depths = []
        
        for repo in self._repos:
            trans_deps = self.get_transitive_dependencies(repo)
            transitive_count = sum(1 for d in trans_deps.values() if d > 1)
            total_transitive += transitive_count
            
            if trans_deps:
                max_repo_depth = max(trans_deps.values())
                max_depth = max(max_depth, max_repo_depth)
                depths.append(max_repo_depth)
        
        metrics.total_transitive = total_transitive
        metrics.max_dependency_depth = max_depth
        metrics.avg_dependency_depth = sum(depths) / len(depths) if depths else 0
        
        # Circular dependencies
        cycles = self.detect_circular_dependencies()
        metrics.circular_dependencies = len(cycles)
        
        # Orphan repos (no dependencies and no dependents)
        orphans = 0
        for repo in self._repos:
            if not self._adjacency.get(repo) and not self._reverse_adjacency.get(repo):
                orphans += 1
        metrics.orphan_repos = orphans
        
        # Risk metrics
        self.calculate_risk_scores()
        risk_scores = [r.risk_score for r in self._repos.values()]
        
        metrics.repos_with_critical_risk = sum(1 for s in risk_scores if s >= 80)
        metrics.repos_with_high_risk = sum(1 for s in risk_scores if 60 <= s < 80)
        metrics.total_vulnerabilities = sum(r.known_vulnerabilities for r in self._repos.values())
        metrics.avg_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0
        
        # Health metrics (based on verification status)
        for repo in self._repos.values():
            if repo.verification_status == "passed":
                metrics.healthy_repos += 1
            elif repo.verification_status == "partial":
                metrics.degraded_repos += 1
            elif repo.verification_status == "failed":
                metrics.at_risk_repos += 1
        
        return metrics

    def get_affected_by_change(
        self,
        repo: str,
        change_severity: str = "minor",
    ) -> dict[str, Any]:
        """Get all repos affected by a change in the given repo."""
        dependents = self.get_transitive_dependents(repo)
        
        affected = {
            "directly_affected": [],
            "transitively_affected": [],
            "total_affected": len(dependents),
            "risk_assessment": [],
        }
        
        for dependent, depth in dependents.items():
            impact = {
                "repo": dependent,
                "depth": depth,
                "path": self._find_path(dependent, repo),
            }
            
            if depth == 1:
                affected["directly_affected"].append(impact)
            else:
                affected["transitively_affected"].append(impact)
            
            # Risk assessment
            if change_severity == "breaking":
                risk_level = "high" if depth == 1 else "medium"
            elif change_severity == "minor":
                risk_level = "low"
            else:
                risk_level = "medium"
            
            affected["risk_assessment"].append({
                "repo": dependent,
                "risk_level": risk_level,
                "recommendation": (
                    f"Test {dependent} after deploying changes to {repo}"
                    if depth == 1
                    else f"Monitor {dependent} for issues"
                ),
            })
        
        return affected

    def generate_visualization_data(self) -> dict[str, Any]:
        """Generate data for dependency graph visualization."""
        nodes = []
        edges = []
        
        # Generate nodes
        for repo in self._repos.values():
            node = {
                "id": repo.full_name,
                "label": repo.name,
                "group": repo.team or "default",
                "risk_score": repo.risk_score,
                "type": repo.repo_type,
                "language": repo.language,
                "size": 10 + len(self._reverse_adjacency.get(repo.full_name, set())) * 2,
            }
            
            # Color based on risk
            if repo.risk_score >= 80:
                node["color"] = "#e74c3c"  # Red
            elif repo.risk_score >= 60:
                node["color"] = "#f39c12"  # Orange
            elif repo.risk_score >= 40:
                node["color"] = "#f1c40f"  # Yellow
            else:
                node["color"] = "#2ecc71"  # Green
            
            nodes.append(node)
        
        # Generate edges
        for edge in self._edges:
            edges.append({
                "from": edge.source,
                "to": edge.target,
                "type": edge.dependency_type,
                "depth": edge.depth,
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "clusters": [
                {
                    "id": c.cluster_id,
                    "name": c.name,
                    "repos": c.repos,
                    "cohesion": c.cohesion_score,
                }
                for c in self._clusters
            ],
            "metrics": self.get_metrics().__dict__,
        }

    def export_to_dot(self) -> str:
        """Export graph to DOT format for Graphviz."""
        lines = [f'digraph "{self.org}_dependencies" {{']
        lines.append('  rankdir=LR;')
        lines.append('  node [shape=box, style=filled];')
        lines.append('')
        
        # Add nodes with colors
        for repo in self._repos.values():
            if repo.risk_score >= 80:
                color = "#ffcccc"
            elif repo.risk_score >= 60:
                color = "#ffe6cc"
            elif repo.risk_score >= 40:
                color = "#ffffcc"
            else:
                color = "#ccffcc"
            
            label = f"{repo.name}\\n({repo.risk_score:.0f})"
            lines.append(f'  "{repo.full_name}" [label="{label}", fillcolor="{color}"];')
        
        lines.append('')
        
        # Add edges
        for edge in self._edges:
            style = "solid" if edge.dependency_type == "direct" else "dashed"
            lines.append(f'  "{edge.source}" -> "{edge.target}" [style={style}];')
        
        lines.append('}')
        return '\n'.join(lines)


class OrgDependencyAnalyzer:
    """High-level analyzer for organization dependencies."""

    def __init__(self, org: str) -> None:
        self.org = org
        self.graph = OrgDependencyGraph(org)

    def analyze_from_manifest_data(
        self,
        manifest_data: list[dict[str, Any]],
    ) -> None:
        """Build graph from manifest data (package.json, requirements.txt, etc.)."""
        for manifest in manifest_data:
            repo = OrgRepository(
                name=manifest.get("name", "unknown"),
                org=self.org,
                language=manifest.get("language", "unknown"),
                repo_type=manifest.get("type", "library"),
                direct_dependencies=manifest.get("dependencies", []),
                dev_dependencies=manifest.get("devDependencies", []),
                known_vulnerabilities=manifest.get("vulnerabilities", 0),
            )
            self.graph.add_repository(repo)
            
            # Add dependency edges
            for dep in repo.direct_dependencies:
                self.graph.add_dependency(repo.full_name, dep, "direct")
            for dep in repo.dev_dependencies:
                self.graph.add_dependency(repo.full_name, dep, "dev")

    def get_full_analysis(self) -> dict[str, Any]:
        """Get complete organization analysis."""
        # Run all analyses
        metrics = self.graph.get_metrics()
        risks = self.graph.analyze_transitive_risks()
        clusters = self.graph.detect_clusters()
        cycles = self.graph.detect_circular_dependencies()
        
        return {
            "organization": self.org,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics.__dict__,
            "transitive_risks": [
                {
                    "source": r.source_repo,
                    "severity": r.severity.value,
                    "affected_path": r.affected_path,
                    "description": r.description,
                }
                for r in risks
            ],
            "clusters": [
                {
                    "name": c.name,
                    "repos": c.repos,
                    "central_repo": c.central_repo,
                    "cohesion": c.cohesion_score,
                }
                for c in clusters
            ],
            "circular_dependencies": cycles,
            "recommendations": self._generate_recommendations(metrics, risks, cycles),
        }

    def _generate_recommendations(
        self,
        metrics: OrgDependencyMetrics,
        risks: list[TransitiveRisk],
        cycles: list[list[str]],
    ) -> list[dict[str, str]]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if metrics.circular_dependencies > 0:
            recommendations.append({
                "priority": "high",
                "category": "architecture",
                "title": "Resolve circular dependencies",
                "description": (
                    f"Found {metrics.circular_dependencies} circular dependency chains. "
                    "These can cause build issues and make changes risky."
                ),
            })
        
        if metrics.repos_with_critical_risk > 0:
            recommendations.append({
                "priority": "critical",
                "category": "security",
                "title": "Address critical security risks",
                "description": (
                    f"{metrics.repos_with_critical_risk} repositories have critical risk scores. "
                    "Review and update vulnerable dependencies immediately."
                ),
            })
        
        if metrics.avg_dependency_depth > 5:
            recommendations.append({
                "priority": "medium",
                "category": "architecture",
                "title": "Reduce dependency depth",
                "description": (
                    f"Average dependency depth is {metrics.avg_dependency_depth:.1f}. "
                    "Deep dependency chains increase risk and build times."
                ),
            })
        
        critical_risks = [r for r in risks if r.severity == RiskLevel.CRITICAL]
        if critical_risks:
            recommendations.append({
                "priority": "critical",
                "category": "security",
                "title": "Critical transitive vulnerabilities",
                "description": (
                    f"{len(critical_risks)} critical vulnerabilities propagating through dependencies. "
                    "Update affected packages to patch security issues."
                ),
            })
        
        if metrics.orphan_repos > 0:
            recommendations.append({
                "priority": "low",
                "category": "maintenance",
                "title": "Review orphan repositories",
                "description": (
                    f"{metrics.orphan_repos} repositories have no dependencies and no dependents. "
                    "Consider archiving unused repositories."
                ),
            })
        
        return recommendations


# Global analyzer instance
_org_analyzer: OrgDependencyAnalyzer | None = None


def get_org_dependency_analyzer(org: str) -> OrgDependencyAnalyzer:
    """Get or create the org dependency analyzer."""
    global _org_analyzer
    if _org_analyzer is None or _org_analyzer.org != org:
        _org_analyzer = OrgDependencyAnalyzer(org)
    return _org_analyzer
