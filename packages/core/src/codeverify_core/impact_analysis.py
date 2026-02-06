"""Cross-Repository Impact Analysis.

Analyze how changes in one repository propagate across an organization's
dependency graph.  Supports package.json, pyproject.toml, go.mod, pom.xml,
and requirements.txt manifests.
"""

from __future__ import annotations

import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()

# ── Enums ────────────────────────────────────────────────────────────────────


class DependencyType(str, Enum):
    """Classification of how one package depends on another."""
    DIRECT = "direct"
    DEV = "dev"
    PEER = "peer"
    TRANSITIVE = "transitive"


class ImpactSeverity(str, Enum):
    """Severity of impact a change has on downstream consumers."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ── Data Models ──────────────────────────────────────────────────────────────


@dataclass
class PackageReference:
    """A package declared in a manifest file."""
    name: str
    version: str
    source_file: str  # e.g. "package.json", "pyproject.toml", "go.mod"
    dep_type: DependencyType


@dataclass
class RepositoryNode:
    """A node in the cross-repository dependency graph."""
    repo_id: str
    name: str
    org: str
    default_branch: str = "main"
    dependencies: list[PackageReference] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)  # repo IDs
    last_indexed: float = field(default_factory=time.time)


@dataclass
class ImpactedFunction:
    """A function identified as impacted by an upstream change."""
    file_path: str
    function_name: str
    dependency_chain: list[str] = field(default_factory=list)
    impact_reason: str = ""


@dataclass
class ImpactReport:
    """Complete report of cross-repository impact for a set of changes."""
    source_repo: str
    changed_files: list[str] = field(default_factory=list)
    impacted_repos: list[str] = field(default_factory=list)
    impacted_functions: list[ImpactedFunction] = field(default_factory=list)
    severity: ImpactSeverity = ImpactSeverity.NONE
    blast_radius: int = 0
    generated_at: float = field(default_factory=time.time)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a plain dictionary."""
        return {
            "source_repo": self.source_repo,
            "changed_files": self.changed_files,
            "impacted_repos": self.impacted_repos,
            "impacted_functions": [
                {
                    "file_path": f.file_path,
                    "function_name": f.function_name,
                    "dependency_chain": f.dependency_chain,
                    "impact_reason": f.impact_reason,
                }
                for f in self.impacted_functions
            ],
            "severity": self.severity.value,
            "blast_radius": self.blast_radius,
            "generated_at": self.generated_at,
            "recommendations": self.recommendations,
        }


# ── Dependency Graph Builder ─────────────────────────────────────────────────


class DependencyGraphBuilder:
    """Parse dependency manifests into :class:`PackageReference` lists."""

    def parse_package_json(self, content: str) -> list[PackageReference]:
        """Parse an npm ``package.json`` file."""
        refs: list[PackageReference] = []
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.warning("package_json_parse_failed", error=str(exc))
            return refs
        section_map: dict[str, DependencyType] = {
            "dependencies": DependencyType.DIRECT,
            "devDependencies": DependencyType.DEV,
            "peerDependencies": DependencyType.PEER,
        }
        for section, dep_type in section_map.items():
            for name, version in data.get(section, {}).items():
                refs.append(PackageReference(
                    name=name, version=self._strip_semver_prefix(version),
                    source_file="package.json", dep_type=dep_type,
                ))
        logger.debug("parsed_package_json", total=len(refs))
        return refs

    def parse_pyproject_toml(self, content: str) -> list[PackageReference]:
        """Parse a Python ``pyproject.toml`` (PEP 621 and Poetry)."""
        refs: list[PackageReference] = []
        # PEP 621: dependencies = ["requests>=2.28", ...]
        pep621 = re.search(
            r"(?:^|\n)\[project\].*?dependencies\s*=\s*\[(.*?)\]", content, re.DOTALL)
        if pep621:
            for item in re.finditer(r'"([^"]+)"', pep621.group(1)):
                name, ver = self._split_python_spec(item.group(1))
                refs.append(PackageReference(name=name, version=ver,
                    source_file="pyproject.toml", dep_type=DependencyType.DIRECT))
        # Optional dependencies (treated as dev)
        opt = re.search(r"\[project\.optional-dependencies\]", content)
        if opt:
            section = content[opt.end():]
            nxt = re.search(r"\n\[", section)
            section = section[:nxt.start()] if nxt else section
            for item in re.finditer(r'"([^"]+)"', section):
                name, ver = self._split_python_spec(item.group(1))
                refs.append(PackageReference(name=name, version=ver,
                    source_file="pyproject.toml", dep_type=DependencyType.DEV))
        # Poetry: [tool.poetry.dependencies]
        poetry = re.search(
            r"\[tool\.poetry\.dependencies\](.*?)(?:\n\[|\Z)", content, re.DOTALL)
        if poetry:
            for m in re.finditer(r'^(\S+)\s*=\s*"([^"]*)"', poetry.group(1), re.MULTILINE):
                if m.group(1).lower() != "python":
                    refs.append(PackageReference(name=m.group(1), version=m.group(2),
                        source_file="pyproject.toml", dep_type=DependencyType.DIRECT))
        # Poetry dev dependencies
        pdev = re.search(
            r"\[tool\.poetry\.group\.dev\.dependencies\](.*?)(?:\n\[|\Z)", content, re.DOTALL)
        if pdev:
            for m in re.finditer(r'^(\S+)\s*=\s*"([^"]*)"', pdev.group(1), re.MULTILINE):
                refs.append(PackageReference(name=m.group(1), version=m.group(2),
                    source_file="pyproject.toml", dep_type=DependencyType.DEV))
        logger.debug("parsed_pyproject_toml", total=len(refs))
        return refs

    def parse_go_mod(self, content: str) -> list[PackageReference]:
        """Parse a Go ``go.mod`` file."""
        refs: list[PackageReference] = []
        # Single-line: require github.com/foo/bar v1.2.3
        for m in re.finditer(r"^require\s+(\S+)\s+(\S+)", content, re.MULTILINE):
            refs.append(PackageReference(name=m.group(1), version=m.group(2),
                source_file="go.mod", dep_type=DependencyType.DIRECT))
        # Block: require ( ... )
        block = re.search(r"require\s*\((.*?)\)", content, re.DOTALL)
        if block:
            for line in block.group(1).strip().splitlines():
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    dt = DependencyType.TRANSITIVE if "// indirect" in line else DependencyType.DIRECT
                    refs.append(PackageReference(name=parts[0], version=parts[1],
                        source_file="go.mod", dep_type=dt))
        logger.debug("parsed_go_mod", total=len(refs))
        return refs

    def parse_pom_xml(self, content: str) -> list[PackageReference]:
        """Parse a Maven ``pom.xml`` (regex-based, no XML library needed)."""
        refs: list[PackageReference] = []
        pat = re.compile(
            r"<dependency>\s*<groupId>([^<]+)</groupId>\s*"
            r"<artifactId>([^<]+)</artifactId>\s*"
            r"(?:<version>([^<]+)</version>\s*)?(?:<scope>([^<]+)</scope>\s*)?",
            re.DOTALL)
        for m in pat.finditer(content):
            gid, aid = m.group(1).strip(), m.group(2).strip()
            ver = (m.group(3) or "").strip() or "managed"
            scope = (m.group(4) or "").strip().lower()
            dt = (DependencyType.DEV if scope == "test"
                  else DependencyType.PEER if scope == "provided"
                  else DependencyType.DIRECT)
            refs.append(PackageReference(name=f"{gid}:{aid}", version=ver,
                source_file="pom.xml", dep_type=dt))
        logger.debug("parsed_pom_xml", total=len(refs))
        return refs

    def parse_requirements_txt(self, content: str) -> list[PackageReference]:
        """Parse a Python ``requirements.txt`` file."""
        refs: list[PackageReference] = []
        for raw in content.splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            if " #" in line:
                line = line[:line.index(" #")].strip()
            m = re.match(r"([A-Za-z0-9_][A-Za-z0-9._-]*(?:\[[^\]]*\])?)\s*([><=!~]+.+)?", line)
            if m:
                name = re.sub(r"\[.*?\]", "", m.group(1)).strip()
                refs.append(PackageReference(name=name, version=(m.group(2) or "*").strip(),
                    source_file="requirements.txt", dep_type=DependencyType.DIRECT))
        logger.debug("parsed_requirements_txt", total=len(refs))
        return refs

    # ── helpers ──

    @staticmethod
    def _strip_semver_prefix(version: str) -> str:
        return version.lstrip("^~>=<")

    @staticmethod
    def _split_python_spec(spec: str) -> tuple[str, str]:
        """Split ``'requests>=2.28'`` into ``('requests', '>=2.28')``."""
        parts = re.split(r"([><=!~]+)", spec, maxsplit=1)
        name = parts[0].strip()
        return name, (spec[len(name):].strip() if len(parts) > 1 else "*")


# ── Cross-Repository Impact Analyzer ─────────────────────────────────────────


class CrossRepoImpactAnalyzer:
    """Analyze how changes in one repository cascade across dependents.

    Usage::

        analyzer = CrossRepoImpactAnalyzer()
        analyzer.register_repository(node_a)
        analyzer.register_repository(node_b)
        report = analyzer.analyze_impact("org/repo-a", ["src/lib.py"])
    """

    def __init__(self) -> None:
        self._repos: dict[str, RepositoryNode] = {}

    def register_repository(self, repo: RepositoryNode) -> None:
        """Register (or replace) a repository node."""
        if repo.repo_id in self._repos:
            logger.warning("repository_replaced", repo_id=repo.repo_id)
        self._repos[repo.repo_id] = repo
        logger.debug("repository_registered", repo_id=repo.repo_id)

    def build_dependency_graph(self) -> dict[str, list[str]]:
        """Build an adjacency list mapping each repo to its dependents.

        Cross-references dependency package names against registered
        repository names and also honours explicitly declared dependents.
        """
        graph: dict[str, list[str]] = {rid: [] for rid in self._repos}

        # Map package name -> repo_id for name-based resolution
        name_to_repo: dict[str, str] = {}
        for rid, node in self._repos.items():
            name_to_repo[node.name] = rid

        # Edges from declared dependencies
        for rid, node in self._repos.items():
            for dep_ref in node.dependencies:
                target = name_to_repo.get(dep_ref.name)
                if target and target != rid and rid not in graph[target]:
                    graph[target].append(rid)

        # Edges from explicit dependents
        for rid, node in self._repos.items():
            for dep_id in node.dependents:
                if dep_id in self._repos and dep_id not in graph.get(rid, []):
                    graph.setdefault(rid, []).append(dep_id)

        logger.info(
            "dependency_graph_built",
            repos=len(self._repos),
            edges=sum(len(v) for v in graph.values()),
        )
        return graph

    def analyze_impact(self, repo_id: str, changed_files: list[str]) -> ImpactReport:
        """Produce a full impact report for changes in *repo_id*."""
        if repo_id not in self._repos:
            logger.error("unknown_repository", repo_id=repo_id)
            return ImpactReport(source_repo=repo_id, changed_files=changed_files)

        graph = self.build_dependency_graph()
        downstream = self._collect_downstream(repo_id, graph)
        blast_radius = len(downstream)
        impacted_fns = self._trace_impacted_functions(repo_id, changed_files, downstream)
        severity = self._calculate_severity(blast_radius, changed_files)

        report = ImpactReport(
            source_repo=repo_id,
            changed_files=changed_files,
            impacted_repos=downstream,
            impacted_functions=impacted_fns,
            severity=severity,
            blast_radius=blast_radius,
            generated_at=time.time(),
        )
        report.recommendations = self._generate_recommendations(report)

        logger.info(
            "impact_analysis_complete",
            source=repo_id, blast_radius=blast_radius, severity=severity.value,
        )
        return report

    def get_blast_radius(self, repo_id: str) -> int:
        """Return the number of downstream repositories affected."""
        if repo_id not in self._repos:
            return 0
        return len(self._collect_downstream(repo_id, self.build_dependency_graph()))

    def get_downstream_repos(self, repo_id: str) -> list[RepositoryNode]:
        """Return all repository nodes downstream of *repo_id*."""
        if repo_id not in self._repos:
            return []
        ids = self._collect_downstream(repo_id, self.build_dependency_graph())
        return [self._repos[rid] for rid in ids if rid in self._repos]

    def get_dependency_path(self, from_repo: str, to_repo: str) -> list[str] | None:
        """BFS shortest path from *from_repo* to *to_repo*, or ``None``."""
        if from_repo not in self._repos or to_repo not in self._repos:
            return None
        if from_repo == to_repo:
            return [from_repo]

        graph = self.build_dependency_graph()
        visited: set[str] = set()
        queue: deque[tuple[str, list[str]]] = deque([(from_repo, [from_repo])])

        while queue:
            current, path = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            for neighbor in graph.get(current, []):
                new_path = path + [neighbor]
                if neighbor == to_repo:
                    return new_path
                queue.append((neighbor, new_path))
        return None

    # ── internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _collect_downstream(repo_id: str, graph: dict[str, list[str]]) -> list[str]:
        """BFS collecting all transitive dependents of *repo_id*."""
        visited: set[str] = set()
        queue: deque[str] = deque(graph.get(repo_id, []))
        result: list[str] = []
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            result.append(current)
            for child in graph.get(current, []):
                if child not in visited:
                    queue.append(child)
        return result

    def _trace_impacted_functions(
        self,
        source_repo_id: str,
        changed_files: list[str],
        downstream_ids: list[str],
    ) -> list[ImpactedFunction]:
        """Heuristically identify functions impacted via dependency chains."""
        impacted: list[ImpactedFunction] = []
        for changed_file in changed_files:
            symbol = self._infer_symbol(changed_file)
            if not symbol:
                continue
            for downstream_id in downstream_ids:
                chain = self.get_dependency_path(source_repo_id, downstream_id)
                impacted.append(ImpactedFunction(
                    file_path=changed_file,
                    function_name=symbol,
                    dependency_chain=chain or [source_repo_id, downstream_id],
                    impact_reason=(
                        f"Change to '{symbol}' in {source_repo_id} "
                        f"may affect {downstream_id} via dependency chain"
                    ),
                ))
        return impacted

    @staticmethod
    def _infer_symbol(file_path: str) -> str:
        """Derive a probable exported symbol name from a file path."""
        cleaned = file_path
        for prefix in ("src/", "lib/", "pkg/", "internal/", "cmd/"):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
                break
        cleaned = re.sub(r"\.[^.]+$", "", cleaned)
        parts = [p for p in cleaned.split("/") if p and p != "index"]
        return parts[-1] if parts else ""

    @staticmethod
    def _calculate_severity(blast_radius: int, changed_files: list[str]) -> ImpactSeverity:
        """Determine severity from blast radius and nature of changed files."""
        if blast_radius == 0:
            return ImpactSeverity.NONE

        high_impact = ("api", "schema", "proto", "graphql", "openapi", "swagger", "grpc", "interface", "types")
        hi_count = sum(1 for f in changed_files if any(p in f.lower() for p in high_impact))

        score = blast_radius + hi_count * 3
        if len(changed_files) > 20:
            score += 10
        elif len(changed_files) > 10:
            score += 5

        if score >= 20:
            return ImpactSeverity.CRITICAL
        if score >= 10:
            return ImpactSeverity.HIGH
        if score >= 4:
            return ImpactSeverity.MEDIUM
        return ImpactSeverity.LOW

    @staticmethod
    def _generate_recommendations(report: ImpactReport) -> list[str]:
        """Generate actionable recommendations from the impact report."""
        recs: list[str] = []

        if report.severity == ImpactSeverity.CRITICAL:
            recs.append(
                "CRITICAL: This change impacts many downstream repositories. "
                "Coordinate a rollout plan with affected teams before merging."
            )
        elif report.severity == ImpactSeverity.HIGH:
            recs.append(
                "HIGH: Significant downstream impact detected. Notify owners "
                "of affected repositories and run integration tests first."
            )

        if report.blast_radius > 0:
            repos_str = ", ".join(report.impacted_repos[:5])
            suffix = f" and {report.blast_radius - 5} more" if report.blast_radius > 5 else ""
            recs.append(
                f"Run integration tests for the {report.blast_radius} affected "
                f"downstream repo(s): {repos_str}{suffix}."
            )

        api_kw = ("api", "schema", "proto", "graphql", "openapi")
        if any(any(kw in f.lower() for kw in api_kw) for f in report.changed_files):
            recs.append(
                "API or schema files were modified. Verify backward "
                "compatibility and consider a versioned release."
            )

        cfg_ext = (".json", ".yaml", ".yml", ".toml", ".ini", ".cfg")
        if any(f.endswith(ext) for f in report.changed_files for ext in cfg_ext):
            recs.append(
                "Configuration files were changed. Verify that downstream "
                "consumers do not rely on the previous configuration values."
            )

        if not report.impacted_repos:
            recs.append("No downstream repositories affected. Standard review is sufficient.")

        if report.severity in (ImpactSeverity.NONE, ImpactSeverity.LOW):
            recs.append("Impact is minimal. Standard code review is sufficient.")

        return recs
