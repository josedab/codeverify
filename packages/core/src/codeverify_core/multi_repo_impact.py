"""Multi-Repository Impact Analysis.

Org-wide repository indexing, blast radius calculation, team notifications,
and migration planning for cross-repository changes.
"""

from __future__ import annotations

import re
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()

# =============================================================================
# Enums
# =============================================================================

class IndexStatus(str, Enum):
    PENDING = "pending"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"
    STALE = "stale"

class ChangeScope(str, Enum):
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    PACKAGE = "package"
    API = "api"
    SCHEMA = "schema"

class NotificationUrgency(str, Enum):
    IMMEDIATE = "immediate"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    DIGEST = "digest"

class MigrationPhase(str, Enum):
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    TESTING = "testing"
    ROLLOUT = "rollout"
    COMPLETED = "completed"

# =============================================================================
# Data Models
# =============================================================================

@dataclass
class RepositoryIndex:
    repo_name: str
    owner: str
    last_indexed: datetime | None
    status: IndexStatus
    exported_symbols: list[str]
    imported_symbols: list[str]
    api_endpoints: list[str]
    dependencies: list[str]
    language: str
    file_count: int = 0
    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_name": self.repo_name,
            "owner": self.owner,
            "last_indexed": self.last_indexed.isoformat() if self.last_indexed else None,
            "status": self.status.value,
            "exported_symbols": self.exported_symbols,
            "imported_symbols": self.imported_symbols,
            "api_endpoints": self.api_endpoints,
            "dependencies": self.dependencies,
            "language": self.language,
            "file_count": self.file_count,
        }
@dataclass
class OrgDependencyGraph:
    org_name: str
    repositories: list[RepositoryIndex]
    edges: list[dict[str, str]]
    created_at: datetime
    total_repos: int
    indexed_repos: int
    def to_dict(self) -> dict[str, Any]:
        return {
            "org_name": self.org_name,
            "repositories": [r.to_dict() for r in self.repositories],
            "edges": self.edges,
            "created_at": self.created_at.isoformat(),
            "total_repos": self.total_repos,
            "indexed_repos": self.indexed_repos,
        }
@dataclass
class BlastRadiusResult:
    change_repo: str
    change_description: str
    change_scope: ChangeScope
    affected_repos: list[str]
    affected_teams: list[str]
    affected_services: list[str]
    risk_score: float
    direct_impacts: int
    transitive_impacts: int
    breaking_changes: list[dict[str, Any]]
    def to_dict(self) -> dict[str, Any]:
        return {
            "change_repo": self.change_repo,
            "change_description": self.change_description,
            "change_scope": self.change_scope.value,
            "affected_repos": self.affected_repos,
            "affected_teams": self.affected_teams,
            "affected_services": self.affected_services,
            "risk_score": self.risk_score,
            "direct_impacts": self.direct_impacts,
            "transitive_impacts": self.transitive_impacts,
            "breaking_changes": self.breaking_changes,
        }
@dataclass
class TeamNotification:
    team: str
    repo: str
    urgency: NotificationUrgency
    message: str
    change_description: str
    required_action: str | None = None
    deadline: datetime | None = None
    def to_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "repo": self.repo,
            "urgency": self.urgency.value,
            "message": self.message,
            "change_description": self.change_description,
            "required_action": self.required_action,
            "deadline": self.deadline.isoformat() if self.deadline else None,
        }
@dataclass
class MigrationPlan:
    id: str
    name: str
    description: str
    phase: MigrationPhase
    source_repo: str
    affected_repos: list[str]
    stages: list[dict[str, Any]]
    current_stage: int = 0
    estimated_effort_hours: float = 0
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "phase": self.phase.value,
            "source_repo": self.source_repo,
            "affected_repos": self.affected_repos,
            "stages": self.stages,
            "current_stage": self.current_stage,
            "estimated_effort_hours": self.estimated_effort_hours,
        }
@dataclass
class MultiRepoImpactReport:
    org_name: str
    analysis_date: datetime
    repos_analyzed: int
    blast_radius: BlastRadiusResult
    notifications: list[TeamNotification]
    migration_plan: MigrationPlan | None = None
    risk_heatmap: dict[str, float] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    def to_dict(self) -> dict[str, Any]:
        return {
            "org_name": self.org_name,
            "analysis_date": self.analysis_date.isoformat(),
            "repos_analyzed": self.repos_analyzed,
            "blast_radius": self.blast_radius.to_dict(),
            "notifications": [n.to_dict() for n in self.notifications],
            "migration_plan": self.migration_plan.to_dict() if self.migration_plan else None,
            "risk_heatmap": self.risk_heatmap,
            "recommendations": self.recommendations,
        }

# =============================================================================
# Repository Indexer
# =============================================================================

class OrgRepositoryIndexer:

    def __init__(self) -> None:
        self._indexes: dict[str, RepositoryIndex] = {}

    def index_repository(self, repo_name: str, owner: str, code_files: dict[str, str]) -> RepositoryIndex:
        language = self._detect_language(code_files)
        exported: list[str] = []
        imported: list[str] = []
        endpoints: list[str] = []
        for path, content in code_files.items():
            lang = self._lang_for_file(path, language)
            exported.extend(self._extract_exports(content, lang))
            imported.extend(self._extract_imports(content, lang))
            endpoints.extend(self._extract_api_endpoints(content, lang))
        own_exports = set(exported)
        dependencies = sorted({
            sym.split(".")[0]
            for sym in imported
            if sym.split(".")[0] not in own_exports
        })
        index = RepositoryIndex(
            repo_name=repo_name, owner=owner,
            last_indexed=datetime.now(timezone.utc), status=IndexStatus.INDEXED,
            exported_symbols=sorted(set(exported)),
            imported_symbols=sorted(set(imported)),
            api_endpoints=sorted(set(endpoints)),
            dependencies=dependencies, language=language,
            file_count=len(code_files),
        )
        self._indexes[f"{owner}/{repo_name}"] = index
        logger.info("repository_indexed", repo=repo_name, exports=len(index.exported_symbols))
        return index

    def build_org_graph(self, repositories: list[RepositoryIndex]) -> OrgDependencyGraph:
        export_map: dict[str, str] = {}
        for repo in repositories:
            full = f"{repo.owner}/{repo.repo_name}"
            for sym in repo.exported_symbols:
                export_map[sym] = full
        edges: list[dict[str, str]] = []
        for repo in repositories:
            full = f"{repo.owner}/{repo.repo_name}"
            for sym in repo.imported_symbols:
                provider = export_map.get(sym)
                if provider and provider != full:
                    edge = {"source": full, "target": provider, "symbol": sym}
                    if edge not in edges:
                        edges.append(edge)
        indexed_count = sum(1 for r in repositories if r.status == IndexStatus.INDEXED)
        graph = OrgDependencyGraph(
            org_name=repositories[0].owner if repositories else "",
            repositories=repositories, edges=edges,
            created_at=datetime.now(timezone.utc),
            total_repos=len(repositories), indexed_repos=indexed_count,
        )
        logger.info("org_graph_built", repos=graph.total_repos, edges=len(edges))
        return graph

    def refresh_index(self, repo_index: RepositoryIndex, updated_files: dict[str, str]) -> RepositoryIndex:
        new_exported = list(repo_index.exported_symbols)
        new_imported = list(repo_index.imported_symbols)
        new_endpoints = list(repo_index.api_endpoints)
        for path, content in updated_files.items():
            lang = self._lang_for_file(path, repo_index.language)
            new_exported.extend(self._extract_exports(content, lang))
            new_imported.extend(self._extract_imports(content, lang))
            new_endpoints.extend(self._extract_api_endpoints(content, lang))
        repo_index.exported_symbols = sorted(set(new_exported))
        repo_index.imported_symbols = sorted(set(new_imported))
        repo_index.api_endpoints = sorted(set(new_endpoints))
        repo_index.last_indexed = datetime.now(timezone.utc)
        repo_index.status = IndexStatus.INDEXED
        repo_index.file_count += len(updated_files)
        return repo_index

    def _extract_exports(self, code: str, language: str) -> list[str]:
        symbols: list[str] = []
        if language == "python":
            for m in re.finditer(r"^def\s+([A-Za-z_]\w*)\s*\(", code, re.MULTILINE):
                if not m.group(1).startswith("_"):
                    symbols.append(m.group(1))
            for m in re.finditer(r"^class\s+([A-Za-z_]\w*)", code, re.MULTILINE):
                symbols.append(m.group(1))
            all_match = re.search(r"__all__\s*=\s*\[([^\]]*)\]", code, re.DOTALL)
            if all_match:
                for item in re.finditer(r"""['"](\w+)['"]""", all_match.group(1)):
                    symbols.append(item.group(1))
        elif language in ("javascript", "typescript"):
            for m in re.finditer(r"export\s+(?:default\s+)?(?:function|class|const|let|var)\s+(\w+)", code):
                symbols.append(m.group(1))
            for m in re.finditer(r"export\s*\{([^}]+)\}", code):
                for name in m.group(1).split(","):
                    clean = name.strip().split(" as ")[0].strip()
                    if clean:
                        symbols.append(clean)
        elif language == "go":
            for m in re.finditer(r"^func\s+(?:\([^)]*\)\s+)?([A-Z]\w*)\s*\(", code, re.MULTILINE):
                symbols.append(m.group(1))
            for m in re.finditer(r"^type\s+([A-Z]\w*)\s+", code, re.MULTILINE):
                symbols.append(m.group(1))
        elif language == "java":
            for m in re.finditer(r"public\s+(?:static\s+)?(?:class|interface|enum)\s+(\w+)", code):
                symbols.append(m.group(1))
        return symbols

    def _extract_imports(self, code: str, language: str) -> list[str]:
        imports: list[str] = []
        if language == "python":
            for m in re.finditer(r"^\s*import\s+([\w.]+)", code, re.MULTILINE):
                imports.append(m.group(1).split(".")[0])
            for m in re.finditer(r"^\s*from\s+([\w.]+)\s+import", code, re.MULTILINE):
                imports.append(m.group(1).split(".")[0])
        elif language in ("javascript", "typescript"):
            for m in re.finditer(r"""import\s+.*?\s+from\s+['"]([^'"]+)['"]""", code):
                imports.append(m.group(1))
            for m in re.finditer(r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)""", code):
                imports.append(m.group(1))
        elif language == "go":
            for m in re.finditer(r"""^\s*import\s+"([^"]+)"$""", code, re.MULTILINE):
                imports.append(m.group(1).split("/")[-1])
            block = re.search(r"import\s*\((.*?)\)", code, re.DOTALL)
            if block:
                for m in re.finditer(r'"([^"]+)"', block.group(1)):
                    imports.append(m.group(1).split("/")[-1])
        elif language == "java":
            for m in re.finditer(r"^\s*import\s+([\w.]+)", code, re.MULTILINE):
                parts = m.group(1).split(".")
                if len(parts) >= 2:
                    imports.append(parts[-1])
        return imports

    def _extract_api_endpoints(self, code: str, language: str) -> list[str]:
        endpoints: list[str] = []
        patterns = [
            r"""@(?:app|router|blueprint)\.\s*(?:get|post|put|delete|patch|route)\s*\(\s*['"]([^'"]+)['"]""",
            r"""(?:app|router)\.\s*(?:get|post|put|delete|patch)\s*\(\s*['"]([^'"]+)['"]""",
            r"""@(?:GetMapping|PostMapping|PutMapping|DeleteMapping|RequestMapping)\s*\(\s*(?:value\s*=\s*)?['"]([^'"]+)['"]""",
            r"""(?:HandleFunc|Handle)\s*\(\s*['"]([^'"]+)['"]""",
        ]
        for pat in patterns:
            for m in re.finditer(pat, code):
                endpoints.append(m.group(1))
        return endpoints
    @staticmethod
    def _detect_language(code_files: dict[str, str]) -> str:
        ext_lang = {".py": "python", ".js": "javascript", ".ts": "typescript", ".go": "go", ".java": "java"}
        counts: dict[str, int] = {}
        for path in code_files:
            for ext, lang in ext_lang.items():
                if path.endswith(ext):
                    counts[lang] = counts.get(lang, 0) + 1
                    break
        return max(counts, key=counts.get) if counts else "unknown"  # type: ignore[arg-type]
    @staticmethod
    def _lang_for_file(path: str, default: str) -> str:
        for ext, lang in {".py": "python", ".js": "javascript", ".ts": "typescript", ".go": "go", ".java": "java"}.items():
            if path.endswith(ext):
                return lang
        return default

# =============================================================================
# Blast Radius Calculator
# =============================================================================

class BlastRadiusCalculator:

    _SCOPE_WEIGHTS: dict[ChangeScope, float] = {
        ChangeScope.FUNCTION: 1.0, ChangeScope.CLASS: 2.0,
        ChangeScope.MODULE: 3.0, ChangeScope.PACKAGE: 4.0,
        ChangeScope.API: 5.0, ChangeScope.SCHEMA: 5.0,
    }

    def __init__(self) -> None:
        self._scope_weights = dict(self._SCOPE_WEIGHTS)

    def calculate(self, graph: OrgDependencyGraph, changed_repo: str, changed_symbols: list[str]) -> BlastRadiusResult:
        direct_deps = self._find_direct_dependents(graph, changed_repo, changed_symbols)
        transitive_deps = self._find_transitive_dependents(graph, direct_deps)
        breaking = self._detect_breaking_changes(changed_symbols, graph)
        all_affected = sorted(set(direct_deps + transitive_deps))
        scope = self._infer_scope(changed_symbols, graph, changed_repo)
        affected_teams: list[str] = []
        affected_services: list[str] = []
        for repo_idx in graph.repositories:
            full_name = f"{repo_idx.owner}/{repo_idx.repo_name}"
            if full_name in all_affected:
                affected_services.append(repo_idx.repo_name)
                affected_teams.append(f"team-{repo_idx.repo_name}")
        blast = BlastRadiusResult(
            change_repo=changed_repo,
            change_description=f"Changed symbols: {', '.join(changed_symbols[:5])}",
            change_scope=scope, affected_repos=all_affected,
            affected_teams=affected_teams, affected_services=affected_services,
            risk_score=0.0, direct_impacts=len(direct_deps),
            transitive_impacts=len(transitive_deps), breaking_changes=breaking,
        )
        blast.risk_score = self._calculate_risk_score(blast)
        logger.info("blast_radius_calculated", repo=changed_repo, affected=len(all_affected), risk=blast.risk_score)
        return blast

    def _find_direct_dependents(self, graph: OrgDependencyGraph, repo: str, symbols: list[str]) -> list[str]:
        dependents: list[str] = []
        symbol_set = set(symbols)
        for edge in graph.edges:
            if edge["target"] == repo and edge.get("symbol") in symbol_set:
                if edge["source"] not in dependents:
                    dependents.append(edge["source"])
        # Fallback: check imported symbols when no edge matches
        if not dependents:
            for repo_idx in graph.repositories:
                full = f"{repo_idx.owner}/{repo_idx.repo_name}"
                if full != repo and symbol_set & set(repo_idx.imported_symbols):
                    dependents.append(full)
        return dependents

    def _find_transitive_dependents(self, graph: OrgDependencyGraph, direct_deps: list[str]) -> list[str]:
        adjacency: dict[str, list[str]] = {}
        for edge in graph.edges:
            adjacency.setdefault(edge["target"], []).append(edge["source"])
        transitive: list[str] = []
        visited: set[str] = set(direct_deps)
        queue: deque[str] = deque(direct_deps)
        while queue:
            current = queue.popleft()
            for dep in adjacency.get(current, []):
                if dep not in visited:
                    visited.add(dep)
                    transitive.append(dep)
                    queue.append(dep)
        return transitive

    def _detect_breaking_changes(self, changed_symbols: list[str], dep_graph: OrgDependencyGraph) -> list[dict[str, Any]]:
        consumed: dict[str, list[str]] = {}
        for repo_idx in dep_graph.repositories:
            full = f"{repo_idx.owner}/{repo_idx.repo_name}"
            for sym in repo_idx.imported_symbols:
                consumed.setdefault(sym, []).append(full)
        breaking: list[dict[str, Any]] = []
        for sym in changed_symbols:
            consumers = consumed.get(sym, [])
            if consumers:
                breaking.append({
                    "symbol": sym, "consumers": consumers,
                    "consumer_count": len(consumers),
                    "severity": "high" if len(consumers) > 2 else "medium",
                })
        return breaking

    def _calculate_risk_score(self, blast: BlastRadiusResult) -> float:
        scope_weight = self._scope_weights.get(blast.change_scope, 1.0)
        base = min(blast.direct_impacts * 10 + blast.transitive_impacts * 5, 60)
        breaking_penalty = min(len(blast.breaking_changes) * 8, 25)
        scope_factor = scope_weight / 5.0
        raw = (base + breaking_penalty) * (0.5 + 0.5 * scope_factor)
        return round(min(raw, 100.0), 1)
    @staticmethod
    def _infer_scope(symbols: list[str], graph: OrgDependencyGraph, repo: str) -> ChangeScope:
        for repo_idx in graph.repositories:
            if f"{repo_idx.owner}/{repo_idx.repo_name}" == repo:
                for sym in symbols:
                    if sym.startswith("/") or sym in repo_idx.api_endpoints:
                        return ChangeScope.API
                break
        for sym in symbols:
            if sym and sym[0].isupper() and not sym.isupper():
                return ChangeScope.CLASS
        return ChangeScope.MODULE if len(symbols) > 5 else ChangeScope.FUNCTION

# =============================================================================
# Team Notifier
# =============================================================================

class TeamNotifier:

    def __init__(self) -> None:
        self._thresholds = {"immediate": 80.0, "high": 50.0, "normal": 20.0, "low": 5.0}

    def generate_notifications(self, blast: BlastRadiusResult, team_mapping: dict[str, str]) -> list[TeamNotification]:
        urgency = self._determine_urgency(blast.risk_score, blast.breaking_changes)
        team_repos: dict[str, list[str]] = {}
        for repo in blast.affected_repos:
            team = team_mapping.get(repo, team_mapping.get(repo.split("/")[-1], ""))
            if team:
                team_repos.setdefault(team, []).append(repo)
        breaking_consumers = {c for bc in blast.breaking_changes for c in bc.get("consumers", [])}
        notifications: list[TeamNotification] = []
        for team, repos in team_repos.items():
            has_breaking = any(r in breaking_consumers for r in repos)
            notif_urgency = NotificationUrgency.IMMEDIATE if has_breaking and urgency == NotificationUrgency.HIGH else urgency
            repo_list = ", ".join(r.split("/")[-1] for r in repos)
            message = f"Upstream change in {blast.change_repo} affects: {repo_list}. Risk: {blast.risk_score}/100."
            required_action = "Review breaking changes and update affected imports." if has_breaking else None
            notifications.append(TeamNotification(
                team=team, repo=blast.change_repo, urgency=notif_urgency,
                message=message, change_description=blast.change_description,
                required_action=required_action,
            ))
        logger.info("notifications_generated", count=len(notifications), urgency=urgency.value)
        return notifications

    def _determine_urgency(self, risk_score: float, breaking_changes: list[dict[str, Any]]) -> NotificationUrgency:
        if breaking_changes and risk_score >= self._thresholds["immediate"]:
            return NotificationUrgency.IMMEDIATE
        if risk_score >= self._thresholds["high"]:
            return NotificationUrgency.HIGH
        if risk_score >= self._thresholds["normal"]:
            return NotificationUrgency.NORMAL
        if risk_score >= self._thresholds["low"]:
            return NotificationUrgency.LOW
        return NotificationUrgency.DIGEST

# =============================================================================
# Migration Planner
# =============================================================================

class MigrationPlanner:

    _EFFORT_PER_BREAKING = 4.0  # hours
    _EFFORT_PER_REPO = 2.0  # hours

    def __init__(self) -> None:
        self._effort_per_breaking = self._EFFORT_PER_BREAKING
        self._effort_per_repo = self._EFFORT_PER_REPO

    def create_plan(self, blast: BlastRadiusResult, description: str) -> MigrationPlan:
        stages = self._generate_stages(blast.affected_repos, blast.breaking_changes)
        effort = len(blast.affected_repos) * self._effort_per_repo + len(blast.breaking_changes) * self._effort_per_breaking
        plan = MigrationPlan(
            id=str(uuid.uuid4()), name=f"Migration: {blast.change_repo}",
            description=description, phase=MigrationPhase.PLANNING,
            source_repo=blast.change_repo, affected_repos=blast.affected_repos,
            stages=stages, current_stage=0, estimated_effort_hours=round(effort, 1),
        )
        logger.info("migration_plan_created", plan_id=plan.id, stages=len(stages))
        return plan

    def _generate_stages(self, affected_repos: list[str], breaking_changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        breaking_repos: set[str] = set()
        for bc in breaking_changes:
            breaking_repos.update(bc.get("consumers", []))
        priority = [r for r in affected_repos if r in breaking_repos]
        remaining = [r for r in affected_repos if r not in breaking_repos]
        stages: list[dict[str, Any]] = [
            {"stage": 1, "name": "Publish source changes", "action": "release", "repos": []},
        ]
        for i in range(0, len(priority), 3):
            stages.append({
                "stage": len(stages) + 1, "name": f"Priority batch {i // 3 + 1}",
                "action": "update", "repos": priority[i : i + 3],
            })
        for i in range(0, len(remaining), 5):
            stages.append({
                "stage": len(stages) + 1, "name": f"Batch {i // 5 + 1}",
                "action": "update", "repos": remaining[i : i + 5],
            })
        stages.append({
            "stage": len(stages) + 1, "name": "Validation",
            "action": "validate", "repos": affected_repos,
        })
        return stages

    def advance_stage(self, plan: MigrationPlan) -> MigrationPlan:
        if plan.current_stage >= len(plan.stages) - 1:
            plan.phase = MigrationPhase.COMPLETED
            return plan
        plan.current_stage += 1
        action = plan.stages[plan.current_stage].get("action")
        if action == "validate":
            plan.phase = MigrationPhase.TESTING
        elif plan.current_stage >= len(plan.stages) - 2:
            plan.phase = MigrationPhase.ROLLOUT
        else:
            plan.phase = MigrationPhase.IN_PROGRESS
        logger.info("migration_advanced", plan_id=plan.id, stage=plan.current_stage, phase=plan.phase.value)
        return plan

# =============================================================================
# Multi-Repo Impact Analyzer (Orchestrator)
# =============================================================================

class MultiRepoImpactAnalyzer:

    def __init__(self) -> None:
        self._calculator = BlastRadiusCalculator()
        self._notifier = TeamNotifier()
        self._planner = MigrationPlanner()

    def analyze_change(
        self, org_name: str, graph: OrgDependencyGraph, changed_repo: str,
        changed_symbols: list[str], team_mapping: dict[str, str] | None = None,
    ) -> MultiRepoImpactReport:
        logger.info("multi_repo_analysis_started", org=org_name, repo=changed_repo)
        blast = self._calculator.calculate(graph, changed_repo, changed_symbols)
        notifications: list[TeamNotification] = []
        if team_mapping:
            notifications = self._notifier.generate_notifications(blast, team_mapping)
        migration_plan: MigrationPlan | None = None
        if blast.breaking_changes:
            migration_plan = self._planner.create_plan(blast, f"Migration for changes in {changed_repo}")
        heatmap = self.generate_risk_heatmap(graph)
        recommendations = self._generate_recommendations(blast, notifications)
        report = MultiRepoImpactReport(
            org_name=org_name, analysis_date=datetime.now(timezone.utc),
            repos_analyzed=graph.total_repos, blast_radius=blast,
            notifications=notifications, migration_plan=migration_plan,
            risk_heatmap=heatmap, recommendations=recommendations,
        )
        logger.info("multi_repo_analysis_complete", org=org_name, affected=len(blast.affected_repos), risk=blast.risk_score)
        return report

    def generate_risk_heatmap(self, graph: OrgDependencyGraph) -> dict[str, float]:
        dependents_map: dict[str, set[str]] = {}
        for edge in graph.edges:
            dependents_map.setdefault(edge["target"], set()).add(edge["source"])
        heatmap: dict[str, float] = {}
        for repo_idx in graph.repositories:
            full = f"{repo_idx.owner}/{repo_idx.repo_name}"
            # BFS transitive dependents
            visited: set[str] = set()
            queue: deque[str] = deque([full])
            while queue:
                current = queue.popleft()
                for dep in dependents_map.get(current, set()):
                    if dep not in visited:
                        visited.add(dep)
                        queue.append(dep)
            direct = len(dependents_map.get(full, set()))
            risk = direct * 15.0 + len(visited) * 8.0 + len(repo_idx.exported_symbols) * 0.5 + len(repo_idx.api_endpoints) * 2.0
            heatmap[full] = round(min(risk, 100.0), 1)
        return heatmap

    def get_most_critical_repos(self, graph: OrgDependencyGraph, top_n: int = 10) -> list[tuple[str, float]]:
        heatmap = self.generate_risk_heatmap(graph)
        return sorted(heatmap.items(), key=lambda x: x[1], reverse=True)[:top_n]
    @staticmethod
    def _generate_recommendations(blast: BlastRadiusResult, notifications: list[TeamNotification]) -> list[str]:
        recs: list[str] = []
        if blast.risk_score >= 80:
            recs.append("CRITICAL: Very high blast radius. Coordinate with all affected teams before merging.")
        elif blast.risk_score >= 50:
            recs.append("HIGH: Significant downstream impact. Run integration tests and notify teams.")
        if blast.breaking_changes:
            recs.append(f"{len(blast.breaking_changes)} breaking change(s). Consider a deprecation period or versioned release.")
        if blast.transitive_impacts > 0:
            recs.append(f"{blast.transitive_impacts} transitive dependent(s). Verify indirect consumers are unaffected.")
        immediate = sum(1 for n in notifications if n.urgency == NotificationUrgency.IMMEDIATE)
        if immediate:
            recs.append(f"{immediate} team(s) require immediate notification before proceeding.")
        if not blast.affected_repos:
            recs.append("No downstream repositories affected. Standard code review is sufficient.")
        return recs
