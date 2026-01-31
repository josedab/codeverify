"""
Cross-Repository Dependency Verification API Router

Provides REST API endpoints for cross-repo verification:
- Dependency graph management
- Contract registration
- Impact analysis
- Breaking change detection
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


router = APIRouter(prefix="/api/v1/cross-repo", tags=["cross-repo"])


# =============================================================================
# Request/Response Models
# =============================================================================

class RepositoryModel(BaseModel):
    """Repository model."""
    name: str = Field(..., description="Repository name")
    owner: str = Field(..., description="Repository owner")
    url: Optional[str] = Field(None, description="Repository URL")
    default_branch: str = Field("main", description="Default branch")
    language: str = Field("unknown", description="Primary language")
    repo_type: str = Field("library", description="Repository type")


class DependencyModel(BaseModel):
    """Dependency model."""
    source: str = Field(..., description="Source repository (depends on target)")
    target: str = Field(..., description="Target repository (being depended on)")
    dependency_type: str = Field("direct", description="Type of dependency")
    version_constraint: Optional[str] = Field(None, description="Version constraint")
    contracts: List[str] = Field(default_factory=list, description="Contracts involved")


class ContractModel(BaseModel):
    """Contract model."""
    name: str = Field(..., description="Contract name")
    owner_repo: str = Field(..., description="Repository that owns this contract")
    contract_type: str = Field(
        "function",
        description="Contract type: function, class, api_endpoint, event, schema"
    )
    signature: Dict[str, Any] = Field(
        default_factory=dict, description="Contract signature/schema"
    )
    version: str = Field("1.0.0", description="Contract version")
    stability: str = Field("stable", description="Stability: stable, beta, deprecated")


class ImpactAnalysisRequest(BaseModel):
    """Request for impact analysis."""
    repo: str = Field(..., description="Repository being changed")
    old_contracts: Dict[str, Dict[str, Any]] = Field(
        ..., description="Old contract versions"
    )
    new_contracts: Dict[str, Dict[str, Any]] = Field(
        ..., description="New contract versions"
    )


class DependencyDetectionRequest(BaseModel):
    """Request for dependency detection."""
    code: str = Field(..., description="Source code to analyze")
    language: str = Field("python", description="Programming language")
    package_config: Optional[str] = Field(
        None, description="Package config (requirements.txt, package.json)"
    )


# =============================================================================
# In-Memory State (would use database in production)
# =============================================================================

# Dependency graph storage
_repositories: Dict[str, Dict[str, Any]] = {}
_dependencies: List[Dict[str, Any]] = []
_contracts: Dict[str, Dict[str, Any]] = {}

# Graph adjacency lists
_dependents: Dict[str, set] = {}  # repo -> repos that depend on it
_dep_on: Dict[str, set] = {}  # repo -> repos it depends on


# =============================================================================
# API Endpoints
# =============================================================================

@router.post(
    "/repositories",
    summary="Register Repository",
    description="Register a repository in the dependency graph"
)
async def register_repository(request: RepositoryModel) -> Dict[str, Any]:
    """Register a repository in the dependency graph."""
    full_name = f"{request.owner}/{request.name}"
    
    _repositories[full_name] = {
        "name": request.name,
        "owner": request.owner,
        "url": request.url,
        "default_branch": request.default_branch,
        "language": request.language,
        "repo_type": request.repo_type,
        "created_at": time.time(),
        "exported_contracts": [],
        "imported_contracts": [],
    }
    
    if full_name not in _dependents:
        _dependents[full_name] = set()
    if full_name not in _dep_on:
        _dep_on[full_name] = set()
    
    return {
        "registered": True,
        "repository": full_name,
    }


@router.get(
    "/repositories",
    summary="List Repositories",
    description="List all registered repositories"
)
async def list_repositories() -> Dict[str, Any]:
    """List all registered repositories."""
    return {
        "repositories": list(_repositories.values()),
        "total": len(_repositories),
    }


@router.get(
    "/repositories/{owner}/{name}",
    summary="Get Repository",
    description="Get repository details"
)
async def get_repository(owner: str, name: str) -> Dict[str, Any]:
    """Get repository details."""
    full_name = f"{owner}/{name}"
    
    if full_name not in _repositories:
        raise HTTPException(status_code=404, detail="Repository not found")
    
    repo = _repositories[full_name]
    
    return {
        "repository": repo,
        "dependents": list(_dependents.get(full_name, set())),
        "dependencies": list(_dep_on.get(full_name, set())),
    }


@router.delete(
    "/repositories/{owner}/{name}",
    summary="Remove Repository",
    description="Remove a repository from the graph"
)
async def remove_repository(owner: str, name: str) -> Dict[str, Any]:
    """Remove a repository from the graph."""
    full_name = f"{owner}/{name}"
    
    if full_name not in _repositories:
        raise HTTPException(status_code=404, detail="Repository not found")
    
    del _repositories[full_name]
    
    # Clean up dependencies
    global _dependencies
    _dependencies = [
        d for d in _dependencies
        if d["source"] != full_name and d["target"] != full_name
    ]
    
    # Clean up adjacency lists
    _dependents.pop(full_name, None)
    _dep_on.pop(full_name, None)
    
    for deps in _dependents.values():
        deps.discard(full_name)
    for deps in _dep_on.values():
        deps.discard(full_name)
    
    return {"removed": True, "repository": full_name}


@router.post(
    "/dependencies",
    summary="Add Dependency",
    description="Add a dependency between repositories"
)
async def add_dependency(request: DependencyModel) -> Dict[str, Any]:
    """Add a dependency between repositories."""
    if request.source not in _repositories:
        raise HTTPException(
            status_code=404,
            detail=f"Source repository not found: {request.source}"
        )
    
    if request.target not in _repositories:
        raise HTTPException(
            status_code=404,
            detail=f"Target repository not found: {request.target}"
        )
    
    dep = {
        "source": request.source,
        "target": request.target,
        "dependency_type": request.dependency_type,
        "version_constraint": request.version_constraint,
        "contracts": request.contracts,
        "created_at": time.time(),
    }
    
    _dependencies.append(dep)
    
    # Update adjacency lists
    if request.target not in _dependents:
        _dependents[request.target] = set()
    _dependents[request.target].add(request.source)
    
    if request.source not in _dep_on:
        _dep_on[request.source] = set()
    _dep_on[request.source].add(request.target)
    
    return {
        "added": True,
        "dependency": dep,
    }


@router.get(
    "/dependencies",
    summary="List Dependencies",
    description="List all dependencies"
)
async def list_dependencies() -> Dict[str, Any]:
    """List all dependencies."""
    return {
        "dependencies": _dependencies,
        "total": len(_dependencies),
    }


@router.get(
    "/dependencies/{owner}/{name}/dependents",
    summary="Get Dependents",
    description="Get repositories that depend on this repository"
)
async def get_dependents(
    owner: str,
    name: str,
    recursive: bool = False,
) -> Dict[str, Any]:
    """Get repositories that depend on this repository."""
    full_name = f"{owner}/{name}"
    
    if full_name not in _repositories:
        raise HTTPException(status_code=404, detail="Repository not found")
    
    if not recursive:
        dependents = list(_dependents.get(full_name, set()))
    else:
        # BFS for transitive dependents
        dependents_set: set = set()
        queue = list(_dependents.get(full_name, set()))
        
        while queue:
            current = queue.pop(0)
            if current in dependents_set:
                continue
            
            dependents_set.add(current)
            
            if current in _dependents:
                for dep in _dependents[current]:
                    if dep not in dependents_set:
                        queue.append(dep)
        
        dependents = list(dependents_set)
    
    return {
        "repository": full_name,
        "dependents": dependents,
        "count": len(dependents),
        "recursive": recursive,
    }


@router.post(
    "/contracts",
    summary="Register Contract",
    description="Register a contract in the graph"
)
async def register_contract(request: ContractModel) -> Dict[str, Any]:
    """Register a contract in the graph."""
    contract_id = hashlib.sha256(
        f"{request.owner_repo}:{request.name}:{request.version}".encode()
    ).hexdigest()[:16]
    
    _contracts[contract_id] = {
        "contract_id": contract_id,
        "name": request.name,
        "owner_repo": request.owner_repo,
        "contract_type": request.contract_type,
        "signature": request.signature,
        "version": request.version,
        "stability": request.stability,
        "consumers": [],
        "created_at": time.time(),
    }
    
    # Update repository's exported contracts
    if request.owner_repo in _repositories:
        if contract_id not in _repositories[request.owner_repo]["exported_contracts"]:
            _repositories[request.owner_repo]["exported_contracts"].append(contract_id)
    
    return {
        "registered": True,
        "contract_id": contract_id,
    }


@router.get(
    "/contracts",
    summary="List Contracts",
    description="List all contracts"
)
async def list_contracts(
    owner_repo: Optional[str] = None,
    contract_type: Optional[str] = None,
) -> Dict[str, Any]:
    """List all contracts."""
    contracts = list(_contracts.values())
    
    if owner_repo:
        contracts = [c for c in contracts if c["owner_repo"] == owner_repo]
    
    if contract_type:
        contracts = [c for c in contracts if c["contract_type"] == contract_type]
    
    return {
        "contracts": contracts,
        "total": len(contracts),
    }


@router.get(
    "/contracts/{contract_id}",
    summary="Get Contract",
    description="Get contract details"
)
async def get_contract(contract_id: str) -> Dict[str, Any]:
    """Get contract details."""
    if contract_id not in _contracts:
        raise HTTPException(status_code=404, detail="Contract not found")
    
    return {"contract": _contracts[contract_id]}


@router.post(
    "/contracts/{contract_id}/consume",
    summary="Register Consumer",
    description="Register a repository as consuming this contract"
)
async def register_consumer(
    contract_id: str,
    consumer_repo: str,
) -> Dict[str, Any]:
    """Register a repository as consuming a contract."""
    if contract_id not in _contracts:
        raise HTTPException(status_code=404, detail="Contract not found")
    
    if consumer_repo not in _repositories:
        raise HTTPException(status_code=404, detail="Consumer repository not found")
    
    if consumer_repo not in _contracts[contract_id]["consumers"]:
        _contracts[contract_id]["consumers"].append(consumer_repo)
    
    if contract_id not in _repositories[consumer_repo]["imported_contracts"]:
        _repositories[consumer_repo]["imported_contracts"].append(contract_id)
    
    return {
        "registered": True,
        "contract_id": contract_id,
        "consumer": consumer_repo,
    }


@router.post(
    "/analyze/impact",
    summary="Analyze Impact",
    description="Analyze impact of contract changes"
)
async def analyze_impact(request: ImpactAnalysisRequest) -> Dict[str, Any]:
    """
    Analyze the impact of contract changes.
    
    Returns affected repositories and breaking changes.
    """
    if request.repo not in _repositories:
        raise HTTPException(status_code=404, detail="Repository not found")
    
    contract_changes = []
    breaking_changes = 0
    
    # Find all changed contract IDs
    all_ids = set(request.old_contracts.keys()) | set(request.new_contracts.keys())
    
    for contract_id in all_ids:
        old = request.old_contracts.get(contract_id)
        new = request.new_contracts.get(contract_id)
        
        if old and new:
            # Modified
            change = _compare_contracts(old, new)
            change["contract_id"] = contract_id
            change["change_type"] = "modified"
            contract_changes.append(change)
            if change["breaking"]:
                breaking_changes += 1
        
        elif old and not new:
            # Removed
            contract_changes.append({
                "contract_id": contract_id,
                "change_type": "removed",
                "breaking": True,
                "impact": "major_breaking",
                "description": f"Contract removed",
            })
            breaking_changes += 1
        
        elif new and not old:
            # Added
            contract_changes.append({
                "contract_id": contract_id,
                "change_type": "added",
                "breaking": False,
                "impact": "compatible",
                "description": "New contract added",
            })
    
    # Find affected repositories
    affected_repos = []
    dependents = _get_transitive_dependents(request.repo)
    
    for dependent in dependents:
        repo_data = _repositories.get(dependent)
        if repo_data:
            for change in contract_changes:
                if change["contract_id"] in repo_data["imported_contracts"]:
                    if dependent not in affected_repos:
                        affected_repos.append(dependent)
    
    # Generate recommendations
    recommendations = []
    if breaking_changes > 0:
        recommendations.append(
            f"This change includes {breaking_changes} breaking change(s)"
        )
        if affected_repos:
            recommendations.append(
                f"Affected repositories: {', '.join(affected_repos)}"
            )
        recommendations.append(
            "Consider creating a new major version or deprecation period"
        )
    
    return {
        "source_repo": request.repo,
        "contract_changes": contract_changes,
        "affected_repos": affected_repos,
        "total_affected": len(affected_repos),
        "breaking_changes": breaking_changes,
        "recommendations": recommendations,
    }


@router.post(
    "/analyze/upgrade-plan",
    summary="Generate Upgrade Plan",
    description="Generate an upgrade plan for affected repositories"
)
async def generate_upgrade_plan(
    request: ImpactAnalysisRequest,
) -> Dict[str, Any]:
    """Generate an upgrade plan for affected repositories."""
    # First analyze impact
    impact = await analyze_impact(request)
    
    # Sort affected repos by dependency order
    affected = set(impact["affected_repos"])
    sorted_repos = _topological_sort()
    ordered_affected = [r for r in sorted_repos if r in affected]
    
    steps = []
    for i, repo in enumerate(ordered_affected):
        step = {
            "order": i + 1,
            "repository": repo,
            "actions": [],
        }
        
        repo_data = _repositories.get(repo)
        if repo_data:
            for change in impact["contract_changes"]:
                if change["contract_id"] in repo_data["imported_contracts"]:
                    if change["breaking"]:
                        step["actions"].append({
                            "type": "update_code",
                            "contract_id": change["contract_id"],
                            "description": change["description"],
                        })
        
        if step["actions"]:
            steps.append(step)
    
    return {
        "source_repo": request.repo,
        "steps": steps,
        "total_steps": len(steps),
        "breaking_changes": impact["breaking_changes"],
    }


@router.post(
    "/detect",
    summary="Detect Dependencies",
    description="Detect dependencies from code"
)
async def detect_dependencies(
    request: DependencyDetectionRequest,
) -> Dict[str, Any]:
    """Detect dependencies from code."""
    import re
    
    dependencies = []
    
    if request.language == "python":
        # From imports
        import_pattern = r'^\s*(?:from|import)\s+(\w+)'
        for match in re.finditer(import_pattern, request.code, re.MULTILINE):
            pkg = match.group(1)
            # Filter standard library
            stdlib = [
                "os", "sys", "re", "json", "time", "typing", "datetime",
                "collections", "functools", "itertools", "pathlib",
                "dataclasses", "enum", "abc", "asyncio", "unittest",
            ]
            if pkg not in stdlib:
                dependencies.append({
                    "name": pkg,
                    "type": "import",
                })
        
        # From requirements.txt
        if request.package_config:
            for line in request.package_config.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    pkg = re.split(r'[=<>!\[\]]', line)[0]
                    if pkg:
                        dependencies.append({
                            "name": pkg,
                            "type": "requirement",
                        })
    
    elif request.language in ("typescript", "javascript"):
        # From imports
        import_pattern = r'(?:import|require)\s*\([\'"]([^\'"]+)[\'"]\)'
        for match in re.finditer(import_pattern, request.code):
            pkg = match.group(1)
            if not pkg.startswith('.'):
                dependencies.append({
                    "name": pkg.split('/')[0],
                    "type": "import",
                })
        
        # From package.json
        if request.package_config:
            try:
                import json
                pkg_json = json.loads(request.package_config)
                for dep_type in ["dependencies", "devDependencies"]:
                    for pkg in pkg_json.get(dep_type, {}).keys():
                        dependencies.append({
                            "name": pkg,
                            "type": dep_type,
                        })
            except json.JSONDecodeError:
                pass
    
    # Detect API dependencies
    api_patterns = [
        r'(?:fetch|axios|request)\s*\(\s*[\'"]([^\'"]+)[\'"]',
        r'(?:get|post|put|delete)\s*\(\s*[\'"]([^\'"]+)[\'"]',
    ]
    
    for pattern in api_patterns:
        for match in re.finditer(pattern, request.code, re.IGNORECASE):
            url = match.group(1)
            dependencies.append({
                "name": url,
                "type": "api",
            })
    
    # Deduplicate
    seen = set()
    unique_deps = []
    for dep in dependencies:
        key = f"{dep['name']}:{dep['type']}"
        if key not in seen:
            seen.add(key)
            unique_deps.append(dep)
    
    return {
        "dependencies": unique_deps,
        "total": len(unique_deps),
    }


@router.get(
    "/graph",
    summary="Get Dependency Graph",
    description="Get the full dependency graph"
)
async def get_graph() -> Dict[str, Any]:
    """Get the full dependency graph."""
    return {
        "repositories": _repositories,
        "dependencies": _dependencies,
        "contracts": _contracts,
        "stats": {
            "repository_count": len(_repositories),
            "dependency_count": len(_dependencies),
            "contract_count": len(_contracts),
        },
    }


@router.get(
    "/graph/cycles",
    summary="Detect Cycles",
    description="Detect cycles in the dependency graph"
)
async def detect_cycles() -> Dict[str, Any]:
    """Detect cycles in the dependency graph."""
    cycles = _detect_cycles()
    
    return {
        "cycles": cycles,
        "has_cycles": len(cycles) > 0,
        "cycle_count": len(cycles),
    }


# =============================================================================
# Helper Functions
# =============================================================================

def _get_transitive_dependents(repo: str) -> List[str]:
    """Get transitive dependents of a repository."""
    result: set = set()
    queue = list(_dependents.get(repo, set()))
    
    while queue:
        current = queue.pop(0)
        if current in result:
            continue
        
        result.add(current)
        
        if current in _dependents:
            for dep in _dependents[current]:
                if dep not in result:
                    queue.append(dep)
    
    return list(result)


def _topological_sort() -> List[str]:
    """Topological sort of repositories."""
    in_degree: Dict[str, int] = {}
    for repo in _repositories:
        in_degree[repo] = len(_dep_on.get(repo, set()))
    
    queue = [r for r, d in in_degree.items() if d == 0]
    result = []
    
    while queue:
        current = queue.pop(0)
        result.append(current)
        
        for dependent in _dependents.get(current, set()):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)
    
    return result


def _detect_cycles() -> List[List[str]]:
    """Detect cycles in the dependency graph."""
    cycles: List[List[str]] = []
    visited: set = set()
    rec_stack: set = set()
    
    def dfs(node: str, path: List[str]) -> None:
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        for neighbor in _dep_on.get(node, set()):
            if neighbor not in visited:
                dfs(neighbor, path.copy())
            elif neighbor in rec_stack:
                cycle_start = path.index(neighbor)
                cycles.append(path[cycle_start:] + [neighbor])
        
        rec_stack.remove(node)
    
    for repo in _repositories:
        if repo not in visited:
            dfs(repo, [])
    
    return cycles


def _compare_contracts(
    old: Dict[str, Any],
    new: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare two contract versions for breaking changes."""
    result = {
        "breaking": False,
        "impact": "compatible",
        "description": "",
    }
    
    breaking_reasons = []
    contract_type = old.get("contract_type", "function")
    
    if contract_type == "function":
        # Check return type
        if old.get("signature", {}).get("return_type") != new.get("signature", {}).get("return_type"):
            if old.get("signature", {}).get("return_type"):
                breaking_reasons.append("Return type changed")
        
        # Check parameters
        old_params = {
            p["name"]: p
            for p in old.get("signature", {}).get("parameters", [])
        }
        new_params = {
            p["name"]: p
            for p in new.get("signature", {}).get("parameters", [])
        }
        
        # Removed parameters
        for name in old_params:
            if name not in new_params:
                breaking_reasons.append(f"Parameter '{name}' removed")
        
        # Added required parameters
        for name, param in new_params.items():
            if name not in old_params:
                if not param.get("optional", False):
                    breaking_reasons.append(f"Required parameter '{name}' added")
    
    elif contract_type == "api_endpoint":
        # Check path and method
        old_sig = old.get("signature", {})
        new_sig = new.get("signature", {})
        
        if old_sig.get("path") != new_sig.get("path"):
            breaking_reasons.append("Endpoint path changed")
        
        if old_sig.get("method") != new_sig.get("method"):
            breaking_reasons.append("HTTP method changed")
    
    if breaking_reasons:
        result["breaking"] = True
        result["impact"] = "major_breaking"
        result["description"] = "; ".join(breaking_reasons)
    
    return result
