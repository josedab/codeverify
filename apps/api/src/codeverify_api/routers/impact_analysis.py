"""Cross-repository impact analysis API router."""

from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

router = APIRouter()


class PackageReference(BaseModel):
    name: str
    version: str
    source_file: str
    dep_type: str = Field(description="Dependency type: direct, dev, peer, transitive")


class RepositoryRegistration(BaseModel):
    repo_id: str
    name: str
    org: str
    default_branch: str = "main"
    dependencies: list[PackageReference] = Field(default_factory=list)


class RepositoryInfo(BaseModel):
    repo_id: str
    name: str
    org: str
    default_branch: str
    dependency_count: int
    dependent_count: int


class ImpactAnalysisRequest(BaseModel):
    repo_id: str = Field(description="Repository where changes were made")
    changed_files: list[str] = Field(description="List of changed file paths")


class ImpactedRepo(BaseModel):
    repo_id: str
    name: str
    org: str
    dependency_chain: list[str]
    impact_reason: str


class ImpactReport(BaseModel):
    source_repo: str
    changed_files: list[str]
    impacted_repos: list[ImpactedRepo]
    blast_radius: int
    severity: str = Field(description="Impact severity: none, low, medium, high, critical")
    recommendations: list[str]


class DependencyParseRequest(BaseModel):
    content: str = Field(description="File content to parse")
    file_type: str = Field(description="File type: package.json, pyproject.toml, go.mod, requirements.txt, pom.xml")


# In-memory repository graph
_repos: dict[str, dict[str, Any]] = {}
_dependency_graph: dict[str, list[str]] = {}  # repo_id -> [dependent_repo_ids]


def _rebuild_graph() -> None:
    """Rebuild the dependency graph from registered repos."""
    global _dependency_graph
    _dependency_graph.clear()

    # Build package -> repo mapping
    package_to_repo: dict[str, str] = {}
    for repo_id, repo in _repos.items():
        package_to_repo[repo["name"]] = repo_id

    # Build dependency edges
    for repo_id, repo in _repos.items():
        for dep in repo.get("dependencies", []):
            dep_repo_id = package_to_repo.get(dep["name"])
            if dep_repo_id and dep_repo_id != repo_id:
                _dependency_graph.setdefault(dep_repo_id, [])
                if repo_id not in _dependency_graph[dep_repo_id]:
                    _dependency_graph[dep_repo_id].append(repo_id)


def _get_all_dependents(repo_id: str, visited: set[str] | None = None) -> list[str]:
    """Recursively get all downstream dependents."""
    if visited is None:
        visited = set()
    if repo_id in visited:
        return []
    visited.add(repo_id)

    direct = _dependency_graph.get(repo_id, [])
    all_deps = list(direct)
    for dep in direct:
        transitive = _get_all_dependents(dep, visited)
        for t in transitive:
            if t not in all_deps:
                all_deps.append(t)
    return all_deps


def _calculate_severity(blast_radius: int, changed_files: list[str]) -> str:
    """Calculate impact severity based on blast radius and change scope."""
    has_security_files = any(
        any(kw in f.lower() for kw in ("auth", "security", "crypto", "permission", "secret"))
        for f in changed_files
    )
    has_api_files = any(
        any(kw in f.lower() for kw in ("api", "router", "endpoint", "handler", "schema"))
        for f in changed_files
    )

    if blast_radius == 0:
        return "none"
    elif blast_radius <= 2 and not has_security_files:
        return "low"
    elif blast_radius <= 5 or has_api_files:
        return "medium"
    elif blast_radius <= 15 or has_security_files:
        return "high"
    else:
        return "critical"


@router.post("/repos", response_model=RepositoryInfo, status_code=status.HTTP_201_CREATED)
async def register_repository(request: RepositoryRegistration) -> RepositoryInfo:
    """Register a repository in the dependency graph."""
    _repos[request.repo_id] = {
        "repo_id": request.repo_id,
        "name": request.name,
        "org": request.org,
        "default_branch": request.default_branch,
        "dependencies": [d.model_dump() for d in request.dependencies],
    }
    _rebuild_graph()

    return RepositoryInfo(
        repo_id=request.repo_id,
        name=request.name,
        org=request.org,
        default_branch=request.default_branch,
        dependency_count=len(request.dependencies),
        dependent_count=len(_dependency_graph.get(request.repo_id, [])),
    )


@router.get("/repos", response_model=list[RepositoryInfo])
async def list_repositories(
    org: str | None = Query(default=None, description="Filter by organization"),
) -> list[RepositoryInfo]:
    """List all registered repositories."""
    repos = list(_repos.values())
    if org:
        repos = [r for r in repos if r["org"] == org]

    return [
        RepositoryInfo(
            repo_id=r["repo_id"],
            name=r["name"],
            org=r["org"],
            default_branch=r["default_branch"],
            dependency_count=len(r.get("dependencies", [])),
            dependent_count=len(_dependency_graph.get(r["repo_id"], [])),
        )
        for r in repos
    ]


@router.post("/analyze", response_model=ImpactReport)
async def analyze_impact(request: ImpactAnalysisRequest) -> ImpactReport:
    """Analyze the cross-repository impact of changes."""
    if request.repo_id not in _repos:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository '{request.repo_id}' not registered",
        )

    dependents = _get_all_dependents(request.repo_id)
    blast_radius = len(dependents)
    severity = _calculate_severity(blast_radius, request.changed_files)

    impacted = []
    for dep_id in dependents:
        repo = _repos.get(dep_id, {})
        impacted.append(ImpactedRepo(
            repo_id=dep_id,
            name=repo.get("name", dep_id),
            org=repo.get("org", ""),
            dependency_chain=[request.repo_id, dep_id],
            impact_reason=f"Depends on {_repos[request.repo_id]['name']}",
        ))

    recommendations = []
    if severity in ("high", "critical"):
        recommendations.append("Run full verification on all impacted repositories")
        recommendations.append("Notify downstream repository maintainers")
    if severity == "critical":
        recommendations.append("Consider a staged rollout to minimize blast radius")
        recommendations.append("Create a coordinated release plan across affected repos")
    if any("api" in f.lower() for f in request.changed_files):
        recommendations.append("Check for API contract breaking changes")
    if not recommendations:
        recommendations.append("No special action required - low impact change")

    return ImpactReport(
        source_repo=request.repo_id,
        changed_files=request.changed_files,
        impacted_repos=impacted,
        blast_radius=blast_radius,
        severity=severity,
        recommendations=recommendations,
    )


@router.get("/repos/{repo_id}/blast-radius")
async def get_blast_radius(repo_id: str) -> dict[str, Any]:
    """Get the blast radius of a repository."""
    if repo_id not in _repos:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Repository not found")

    dependents = _get_all_dependents(repo_id)
    direct = _dependency_graph.get(repo_id, [])

    return {
        "repo_id": repo_id,
        "name": _repos[repo_id]["name"],
        "direct_dependents": len(direct),
        "transitive_dependents": len(dependents) - len(direct),
        "total_blast_radius": len(dependents),
        "dependent_repos": [
            {"repo_id": d, "name": _repos.get(d, {}).get("name", d)}
            for d in dependents
        ],
    }


@router.get("/graph")
async def get_dependency_graph() -> dict[str, Any]:
    """Get the full dependency graph."""
    nodes = [
        {"id": r["repo_id"], "name": r["name"], "org": r["org"]}
        for r in _repos.values()
    ]
    edges = []
    for source, targets in _dependency_graph.items():
        for target in targets:
            edges.append({"from": source, "to": target})

    return {
        "nodes": nodes,
        "edges": edges,
        "total_repos": len(nodes),
        "total_edges": len(edges),
    }


@router.post("/parse-dependencies", response_model=list[PackageReference])
async def parse_dependencies(request: DependencyParseRequest) -> list[PackageReference]:
    """Parse dependencies from a manifest file."""
    import json
    import re

    deps: list[PackageReference] = []

    if request.file_type == "package.json":
        try:
            data = json.loads(request.content)
            for name, version in data.get("dependencies", {}).items():
                deps.append(PackageReference(name=name, version=version, source_file="package.json", dep_type="direct"))
            for name, version in data.get("devDependencies", {}).items():
                deps.append(PackageReference(name=name, version=version, source_file="package.json", dep_type="dev"))
            for name, version in data.get("peerDependencies", {}).items():
                deps.append(PackageReference(name=name, version=version, source_file="package.json", dep_type="peer"))
        except json.JSONDecodeError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON")

    elif request.file_type == "requirements.txt":
        for line in request.content.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("-"):
                match = re.match(r"^([\w\-_.]+)(?:\[.*?\])?(?:([><=!~]+)(.+))?", line)
                if match:
                    name = match.group(1)
                    version = (match.group(2) or "") + (match.group(3) or "")
                    deps.append(PackageReference(name=name, version=version or "*", source_file="requirements.txt", dep_type="direct"))

    elif request.file_type == "go.mod":
        in_require = False
        for line in request.content.splitlines():
            stripped = line.strip()
            if stripped.startswith("require ("):
                in_require = True
                continue
            if stripped == ")":
                in_require = False
                continue
            if in_require or stripped.startswith("require "):
                parts = stripped.replace("require ", "").strip().split()
                if len(parts) >= 2:
                    deps.append(PackageReference(name=parts[0], version=parts[1], source_file="go.mod", dep_type="direct"))

    return deps
