"""Cross-Repository Blast Radius Analysis (v1.2.0 Enhanced).

Organization-wide dependency graph with advanced change impact propagation,
blast radius scoring, Mermaid visualization, and affected team alerting.

Builds on the existing blast_radius module with:
- Full org-wide dependency graph with cycle detection
- BFS transitive dependent discovery with path tracking
- Impact level calculation based on change type, distance, and repo criticality
- Blast radius score (0-10) and recommendations
- Mermaid diagram generation for PR comments
"""

from __future__ import annotations

import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class CrossRepoDependencyType(str, Enum):
    """Type of cross-repository dependency."""

    PACKAGE = "package"
    API_CONTRACT = "api_contract"
    SHARED_LIBRARY = "shared_library"
    DATABASE_SCHEMA = "database_schema"
    MESSAGE_SCHEMA = "message_schema"
    SUBMODULE = "submodule"
    RUNTIME = "runtime"


class CrossRepoChangeImpact(str, Enum):
    """Impact classification for cross-repo changes."""

    BREAKING = "breaking"
    COMPATIBLE = "compatible"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


class CrossRepoImpactLevel(str, Enum):
    """Severity of impact on a dependent repository."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass
class OrgRepository:
    """Repository in the organization graph."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    owner: str = ""
    language: str = ""
    team: str = ""
    is_critical: bool = False
    deployment_tier: str = "standard"
    tags: list[str] = field(default_factory=list)


@dataclass
class CrossRepoDependency:
    """Dependency edge between repositories."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_repo: str = ""
    target_repo: str = ""
    dep_type: CrossRepoDependencyType = CrossRepoDependencyType.PACKAGE
    package_name: str = ""
    version_constraint: str = ""
    is_direct: bool = True


@dataclass
class CrossRepoChange:
    """A code change to analyze across repositories."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    repository: str = ""
    files_changed: list[str] = field(default_factory=list)
    apis_changed: list[str] = field(default_factory=list)
    schemas_changed: list[str] = field(default_factory=list)
    impact_type: CrossRepoChangeImpact = CrossRepoChangeImpact.UNKNOWN
    commit_sha: str = ""
    author: str = ""


@dataclass
class CrossRepoImpactedRepo:
    """A repository impacted by a cross-repo change."""

    repository: str = ""
    impact_level: CrossRepoImpactLevel = CrossRepoImpactLevel.NONE
    path: list[str] = field(default_factory=list)
    distance: int = 0
    team: str = ""
    affected_apis: list[str] = field(default_factory=list)
    needs_testing: bool = False
    needs_deployment: bool = False


@dataclass
class CrossRepoBlastReport:
    """Complete cross-repo blast radius report."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    change: CrossRepoChange | None = None
    impacted_repos: list[CrossRepoImpactedRepo] = field(default_factory=list)
    total_repos: int = 0
    total_impacted: int = 0
    critical_count: int = 0
    high_count: int = 0
    teams_affected: list[str] = field(default_factory=list)
    blast_score: float = 0.0
    recommendations: list[str] = field(default_factory=list)
    analysis_time_ms: int = 0
    mermaid_diagram: str = ""

    @property
    def impact_pct(self) -> float:
        return (self.total_impacted / self.total_repos * 100) if self.total_repos > 0 else 0.0

    @property
    def summary(self) -> str:
        return (
            f"Blast radius: {self.total_impacted}/{self.total_repos} repos "
            f"({self.impact_pct:.0f}%), score {self.blast_score:.1f}/10, "
            f"{len(self.teams_affected)} team(s)"
        )


class OrgDependencyGraph:
    """Organization-wide repository dependency graph."""

    def __init__(self) -> None:
        self.repos: dict[str, OrgRepository] = {}
        self.edges: list[CrossRepoDependency] = []
        self._fwd: dict[str, list[CrossRepoDependency]] = defaultdict(list)
        self._rev: dict[str, list[CrossRepoDependency]] = defaultdict(list)

    def add_repo(self, repo: OrgRepository) -> None:
        self.repos[repo.name] = repo

    def add_edge(self, dep: CrossRepoDependency) -> None:
        self.edges.append(dep)
        self._fwd[dep.source_repo].append(dep)
        self._rev[dep.target_repo].append(dep)

    def get_dependents(self, repo: str) -> list[tuple[str, CrossRepoDependency]]:
        return [(d.source_repo, d) for d in self._rev.get(repo, [])]

    def get_transitive_dependents(
        self, repo: str, max_depth: int = 10
    ) -> list[tuple[str, int, list[str]]]:
        visited: set[str] = {repo}
        queue: deque[tuple[str, int, list[str]]] = deque([(repo, 0, [repo])])
        results: list[tuple[str, int, list[str]]] = []

        while queue:
            current, depth, path = queue.popleft()
            if depth > 0:
                results.append((current, depth, path))
            if depth >= max_depth:
                continue
            for dep_name, _ in self.get_dependents(current):
                if dep_name not in visited:
                    visited.add(dep_name)
                    queue.append((dep_name, depth + 1, path + [dep_name]))

        return results

    def detect_cycles(self) -> list[list[str]]:
        cycles: list[list[str]] = []
        visited: set[str] = set()
        stack: set[str] = set()

        def dfs(node: str, path: list[str]) -> None:
            visited.add(node)
            stack.add(node)
            for dep_name, _ in [(d.target_repo, d) for d in self._fwd.get(node, [])]:
                if dep_name not in visited:
                    dfs(dep_name, path + [dep_name])
                elif dep_name in stack:
                    idx = path.index(dep_name) if dep_name in path else -1
                    if idx >= 0:
                        cycles.append(path[idx:] + [dep_name])
            stack.discard(node)

        for name in self.repos:
            if name not in visited:
                dfs(name, [name])
        return cycles


class CrossRepoBlastAnalyzer:
    """Analyzes cross-repository blast radius for code changes."""

    def __init__(self) -> None:
        self.graph = OrgDependencyGraph()
        self.reports: dict[str, CrossRepoBlastReport] = {}

    def add_repository(
        self,
        name: str,
        owner: str = "",
        team: str = "",
        language: str = "",
        is_critical: bool = False,
    ) -> OrgRepository:
        repo = OrgRepository(
            name=name, owner=owner, team=team,
            language=language, is_critical=is_critical,
        )
        self.graph.add_repo(repo)
        return repo

    def add_dependency(
        self,
        source: str,
        target: str,
        dep_type: CrossRepoDependencyType = CrossRepoDependencyType.PACKAGE,
        package_name: str = "",
    ) -> CrossRepoDependency:
        dep = CrossRepoDependency(
            source_repo=source, target_repo=target,
            dep_type=dep_type, package_name=package_name,
        )
        self.graph.add_edge(dep)
        return dep

    def analyze(
        self, change: CrossRepoChange, max_depth: int = 5
    ) -> CrossRepoBlastReport:
        import time
        start = time.monotonic()

        dependents = self.graph.get_transitive_dependents(change.repository, max_depth)
        impacted: list[CrossRepoImpactedRepo] = []

        for repo_name, distance, path in dependents:
            repo = self.graph.repos.get(repo_name)
            level = self._calc_impact(change, repo, distance)
            if level == CrossRepoImpactLevel.NONE:
                continue
            impacted.append(CrossRepoImpactedRepo(
                repository=repo_name,
                impact_level=level,
                path=path,
                distance=distance,
                team=repo.team if repo else "",
                affected_apis=change.apis_changed,
                needs_testing=level in (CrossRepoImpactLevel.CRITICAL, CrossRepoImpactLevel.HIGH),
                needs_deployment=level == CrossRepoImpactLevel.CRITICAL,
            ))

        impacted.sort(key=lambda r: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(r.impact_level.value, 4))

        report = CrossRepoBlastReport(
            change=change,
            impacted_repos=impacted,
            total_repos=len(self.graph.repos),
            total_impacted=len(impacted),
            critical_count=sum(1 for r in impacted if r.impact_level == CrossRepoImpactLevel.CRITICAL),
            high_count=sum(1 for r in impacted if r.impact_level == CrossRepoImpactLevel.HIGH),
            teams_affected=list({r.team for r in impacted if r.team}),
            analysis_time_ms=int((time.monotonic() - start) * 1000),
        )

        report.blast_score = self._calc_score(report)
        report.recommendations = self._gen_recs(report)
        report.mermaid_diagram = self._render_mermaid(report)
        self.reports[report.id] = report
        return report

    def _calc_impact(
        self, change: CrossRepoChange, repo: OrgRepository | None, dist: int
    ) -> CrossRepoImpactLevel:
        if change.impact_type == CrossRepoChangeImpact.INTERNAL:
            return CrossRepoImpactLevel.NONE
        if change.impact_type == CrossRepoChangeImpact.BREAKING:
            if dist == 1:
                return CrossRepoImpactLevel.CRITICAL if (repo and repo.is_critical) else CrossRepoImpactLevel.HIGH
            if dist == 2:
                return CrossRepoImpactLevel.HIGH if (repo and repo.is_critical) else CrossRepoImpactLevel.MEDIUM
            return CrossRepoImpactLevel.LOW
        if dist <= 2:
            return CrossRepoImpactLevel.MEDIUM
        return CrossRepoImpactLevel.LOW

    def _calc_score(self, report: CrossRepoBlastReport) -> float:
        if report.total_repos == 0:
            return 0.0
        pct = report.total_impacted / report.total_repos
        return min(10.0, pct * 4 + report.critical_count * 2.0 + report.high_count + len(report.teams_affected) * 0.5)

    def _gen_recs(self, report: CrossRepoBlastReport) -> list[str]:
        recs = []
        if report.critical_count:
            recs.append(f"⚠️ {report.critical_count} critical impact(s) — coordinate deployment")
        if report.blast_score > 7:
            recs.append("Consider smaller PRs to reduce blast radius")
        if len(report.teams_affected) > 2:
            recs.append(f"Notify teams: {', '.join(report.teams_affected[:3])}")
        testing = [r for r in report.impacted_repos if r.needs_testing]
        if testing:
            recs.append(f"Run integration tests for {len(testing)} affected repo(s)")
        if not recs:
            recs.append("Low impact — safe to proceed")
        return recs

    def _render_mermaid(self, report: CrossRepoBlastReport) -> str:
        lines = ["graph LR"]
        if not report.change:
            return "\n".join(lines)
        src = report.change.repository.replace("-", "_")
        lines.append(f'    {src}["{report.change.repository} 🔴"]')
        lines.append(f"    style {src} fill:#fee2e2,stroke:#ef4444")
        for imp in report.impacted_repos:
            n = imp.repository.replace("-", "_")
            icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(imp.impact_level.value, "⚪")
            lines.append(f'    {n}["{imp.repository} {icon}"]')
            if len(imp.path) >= 2:
                prev = imp.path[-2].replace("-", "_")
                lines.append(f"    {prev} --> {n}")
        return "\n".join(lines)


# ─── Singleton ──────────────────────────────────────────────────────────

_cross_repo_analyzer: CrossRepoBlastAnalyzer | None = None


def get_cross_repo_blast_analyzer() -> CrossRepoBlastAnalyzer:
    global _cross_repo_analyzer
    if _cross_repo_analyzer is None:
        _cross_repo_analyzer = CrossRepoBlastAnalyzer()
    return _cross_repo_analyzer


def reset_cross_repo_blast_analyzer() -> None:
    global _cross_repo_analyzer
    _cross_repo_analyzer = None
