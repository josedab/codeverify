"""
Cross-Repository Dependency Verification

Verify that code changes don't break contracts with dependent repositories:
- Dependency graph analysis
- Contract verification across repos
- Impact analysis for changes
- Breaking change detection

Essential for microservices and monorepo architectures.
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# Data Models
# =============================================================================

class DependencyType(str, Enum):
    """Types of dependencies between repositories."""
    DIRECT = "direct"           # Direct import/require
    TRANSITIVE = "transitive"   # Indirect through another dep
    API = "api"                 # API/HTTP dependency
    EVENT = "event"             # Event/message bus
    DATABASE = "database"       # Shared database schema
    FILE = "file"               # Shared file/config


class ChangeImpact(str, Enum):
    """Impact level of a change on dependents."""
    NONE = "none"
    COMPATIBLE = "compatible"       # Backward compatible
    MINOR_BREAKING = "minor_breaking"  # Might break some uses
    MAJOR_BREAKING = "major_breaking"  # Will definitely break


@dataclass
class Repository:
    """Represents a repository in the dependency graph."""
    
    name: str
    owner: str
    url: Optional[str] = None
    default_branch: str = "main"
    
    # Language and type
    language: str = "unknown"
    repo_type: str = "library"  # library, service, app
    
    # Contracts exported by this repo
    exported_contracts: List[str] = field(default_factory=list)
    
    # Contracts consumed from other repos
    imported_contracts: List[str] = field(default_factory=list)
    
    def full_name(self) -> str:
        """Get full repository name."""
        return f"{self.owner}/{self.name}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "owner": self.owner,
            "url": self.url,
            "default_branch": self.default_branch,
            "language": self.language,
            "repo_type": self.repo_type,
            "exported_contracts": self.exported_contracts,
            "imported_contracts": self.imported_contracts,
        }


@dataclass
class Dependency:
    """A dependency between two repositories."""
    
    source: str  # Repo that depends
    target: str  # Repo being depended on
    
    dependency_type: DependencyType
    version_constraint: Optional[str] = None
    
    # Contracts involved in this dependency
    contracts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "dependency_type": self.dependency_type.value,
            "version_constraint": self.version_constraint,
            "contracts": self.contracts,
        }


@dataclass
class Contract:
    """A contract (interface) exported by a repository."""
    
    contract_id: str
    name: str
    owner_repo: str
    
    # Type of contract
    contract_type: str  # "function", "class", "api_endpoint", "event", "schema"
    
    # Signature/schema
    signature: Dict[str, Any] = field(default_factory=dict)
    
    # Version
    version: str = "1.0.0"
    
    # Stability
    stability: str = "stable"  # "stable", "beta", "deprecated"
    
    # Consumers
    consumers: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "contract_id": self.contract_id,
            "name": self.name,
            "owner_repo": self.owner_repo,
            "contract_type": self.contract_type,
            "signature": self.signature,
            "version": self.version,
            "stability": self.stability,
            "consumers": self.consumers,
        }


@dataclass
class ContractChange:
    """A change to a contract."""
    
    contract_id: str
    change_type: str  # "added", "removed", "modified"
    
    # Old and new signatures for comparison
    old_signature: Optional[Dict[str, Any]] = None
    new_signature: Optional[Dict[str, Any]] = None
    
    # Analysis
    breaking: bool = False
    impact: ChangeImpact = ChangeImpact.NONE
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "contract_id": self.contract_id,
            "change_type": self.change_type,
            "old_signature": self.old_signature,
            "new_signature": self.new_signature,
            "breaking": self.breaking,
            "impact": self.impact.value,
            "description": self.description,
        }


@dataclass
class ImpactAnalysis:
    """Results of impact analysis for a change."""
    
    source_repo: str
    affected_repos: List[str] = field(default_factory=list)
    contract_changes: List[ContractChange] = field(default_factory=list)
    
    # Summary
    total_affected: int = 0
    breaking_changes: int = 0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_repo": self.source_repo,
            "affected_repos": self.affected_repos,
            "contract_changes": [c.to_dict() for c in self.contract_changes],
            "total_affected": self.total_affected,
            "breaking_changes": self.breaking_changes,
            "recommendations": self.recommendations,
        }


# =============================================================================
# Dependency Graph
# =============================================================================

class DependencyGraph:
    """
    Manages the dependency graph between repositories.
    
    Supports:
    - Adding/removing repositories and dependencies
    - Finding dependents and dependencies
    - Topological sorting
    - Cycle detection
    """
    
    def __init__(self):
        self.repositories: Dict[str, Repository] = {}
        self.dependencies: List[Dependency] = []
        self.contracts: Dict[str, Contract] = {}
        
        # Graph adjacency lists
        self._dependents: Dict[str, Set[str]] = {}  # repo -> repos that depend on it
        self._dependencies: Dict[str, Set[str]] = {}  # repo -> repos it depends on
    
    def add_repository(self, repo: Repository) -> None:
        """Add a repository to the graph."""
        full_name = repo.full_name()
        self.repositories[full_name] = repo
        
        if full_name not in self._dependents:
            self._dependents[full_name] = set()
        if full_name not in self._dependencies:
            self._dependencies[full_name] = set()
    
    def add_dependency(self, dep: Dependency) -> None:
        """Add a dependency to the graph."""
        self.dependencies.append(dep)
        
        # Update adjacency lists
        if dep.target not in self._dependents:
            self._dependents[dep.target] = set()
        self._dependents[dep.target].add(dep.source)
        
        if dep.source not in self._dependencies:
            self._dependencies[dep.source] = set()
        self._dependencies[dep.source].add(dep.target)
    
    def add_contract(self, contract: Contract) -> None:
        """Add a contract to the graph."""
        self.contracts[contract.contract_id] = contract
    
    def get_dependents(self, repo: str, recursive: bool = False) -> Set[str]:
        """
        Get repositories that depend on the given repository.
        
        If recursive=True, includes transitive dependents.
        """
        if repo not in self._dependents:
            return set()
        
        if not recursive:
            return self._dependents[repo].copy()
        
        # BFS for transitive dependents
        result: Set[str] = set()
        queue = list(self._dependents[repo])
        
        while queue:
            current = queue.pop(0)
            if current in result:
                continue
            
            result.add(current)
            
            if current in self._dependents:
                for dep in self._dependents[current]:
                    if dep not in result:
                        queue.append(dep)
        
        return result
    
    def get_dependencies(self, repo: str, recursive: bool = False) -> Set[str]:
        """
        Get repositories that the given repository depends on.
        
        If recursive=True, includes transitive dependencies.
        """
        if repo not in self._dependencies:
            return set()
        
        if not recursive:
            return self._dependencies[repo].copy()
        
        # BFS for transitive dependencies
        result: Set[str] = set()
        queue = list(self._dependencies[repo])
        
        while queue:
            current = queue.pop(0)
            if current in result:
                continue
            
            result.add(current)
            
            if current in self._dependencies:
                for dep in self._dependencies[current]:
                    if dep not in result:
                        queue.append(dep)
        
        return result
    
    def detect_cycles(self) -> List[List[str]]:
        """Detect cycles in the dependency graph."""
        cycles: List[List[str]] = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        
        def dfs(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self._dependencies.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])
            
            rec_stack.remove(node)
        
        for repo in self.repositories:
            if repo not in visited:
                dfs(repo, [])
        
        return cycles
    
    def topological_sort(self) -> List[str]:
        """
        Return repositories in topological order.
        
        Repos with no dependencies come first.
        """
        in_degree: Dict[str, int] = {}
        for repo in self.repositories:
            in_degree[repo] = len(self._dependencies.get(repo, set()))
        
        # Start with repos that have no dependencies
        queue = [r for r, d in in_degree.items() if d == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for dependent in self._dependents.get(current, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Export graph as dictionary."""
        return {
            "repositories": {
                k: v.to_dict() for k, v in self.repositories.items()
            },
            "dependencies": [d.to_dict() for d in self.dependencies],
            "contracts": {
                k: v.to_dict() for k, v in self.contracts.items()
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DependencyGraph:
        """Import graph from dictionary."""
        graph = cls()
        
        for repo_data in data.get("repositories", {}).values():
            repo = Repository(
                name=repo_data["name"],
                owner=repo_data["owner"],
                url=repo_data.get("url"),
                default_branch=repo_data.get("default_branch", "main"),
                language=repo_data.get("language", "unknown"),
                repo_type=repo_data.get("repo_type", "library"),
            )
            graph.add_repository(repo)
        
        for dep_data in data.get("dependencies", []):
            dep = Dependency(
                source=dep_data["source"],
                target=dep_data["target"],
                dependency_type=DependencyType(dep_data["dependency_type"]),
                version_constraint=dep_data.get("version_constraint"),
                contracts=dep_data.get("contracts", []),
            )
            graph.add_dependency(dep)
        
        return graph


# =============================================================================
# Contract Analyzer
# =============================================================================

class ContractAnalyzer:
    """Analyzes contracts for breaking changes."""
    
    def compare_contracts(
        self,
        old_contract: Contract,
        new_contract: Contract,
    ) -> ContractChange:
        """Compare two versions of a contract."""
        change = ContractChange(
            contract_id=old_contract.contract_id,
            change_type="modified",
            old_signature=old_contract.signature,
            new_signature=new_contract.signature,
        )
        
        # Analyze based on contract type
        if old_contract.contract_type == "function":
            self._analyze_function_change(change, old_contract, new_contract)
        elif old_contract.contract_type == "class":
            self._analyze_class_change(change, old_contract, new_contract)
        elif old_contract.contract_type == "api_endpoint":
            self._analyze_api_change(change, old_contract, new_contract)
        elif old_contract.contract_type == "event":
            self._analyze_event_change(change, old_contract, new_contract)
        elif old_contract.contract_type == "schema":
            self._analyze_schema_change(change, old_contract, new_contract)
        
        return change
    
    def _analyze_function_change(
        self,
        change: ContractChange,
        old: Contract,
        new: Contract,
    ) -> None:
        """Analyze changes to a function contract."""
        old_sig = old.signature
        new_sig = new.signature
        
        breaking_reasons = []
        
        # Check return type changes
        if old_sig.get("return_type") != new_sig.get("return_type"):
            if old_sig.get("return_type"):
                breaking_reasons.append(
                    f"Return type changed from {old_sig.get('return_type')} "
                    f"to {new_sig.get('return_type')}"
                )
        
        # Check parameter changes
        old_params = {p["name"]: p for p in old_sig.get("parameters", [])}
        new_params = {p["name"]: p for p in new_sig.get("parameters", [])}
        
        # Removed parameters (breaking)
        for name in old_params:
            if name not in new_params:
                breaking_reasons.append(f"Parameter '{name}' removed")
        
        # Changed parameter types (breaking)
        for name, old_param in old_params.items():
            if name in new_params:
                new_param = new_params[name]
                if old_param.get("type") != new_param.get("type"):
                    breaking_reasons.append(
                        f"Parameter '{name}' type changed from "
                        f"{old_param.get('type')} to {new_param.get('type')}"
                    )
        
        # Added required parameters (breaking)
        for name, new_param in new_params.items():
            if name not in old_params:
                if not new_param.get("optional", False):
                    breaking_reasons.append(
                        f"Required parameter '{name}' added"
                    )
        
        if breaking_reasons:
            change.breaking = True
            change.impact = ChangeImpact.MAJOR_BREAKING
            change.description = "; ".join(breaking_reasons)
        else:
            change.impact = ChangeImpact.COMPATIBLE
            change.description = "Compatible changes only"
    
    def _analyze_class_change(
        self,
        change: ContractChange,
        old: Contract,
        new: Contract,
    ) -> None:
        """Analyze changes to a class contract."""
        old_sig = old.signature
        new_sig = new.signature
        
        breaking_reasons = []
        
        # Check removed methods
        old_methods = set(old_sig.get("methods", {}).keys())
        new_methods = set(new_sig.get("methods", {}).keys())
        
        removed_methods = old_methods - new_methods
        if removed_methods:
            breaking_reasons.append(
                f"Methods removed: {', '.join(removed_methods)}"
            )
        
        # Check removed attributes
        old_attrs = set(old_sig.get("attributes", {}).keys())
        new_attrs = set(new_sig.get("attributes", {}).keys())
        
        removed_attrs = old_attrs - new_attrs
        if removed_attrs:
            breaking_reasons.append(
                f"Attributes removed: {', '.join(removed_attrs)}"
            )
        
        if breaking_reasons:
            change.breaking = True
            change.impact = ChangeImpact.MAJOR_BREAKING
            change.description = "; ".join(breaking_reasons)
        else:
            change.impact = ChangeImpact.COMPATIBLE
    
    def _analyze_api_change(
        self,
        change: ContractChange,
        old: Contract,
        new: Contract,
    ) -> None:
        """Analyze changes to an API endpoint contract."""
        old_sig = old.signature
        new_sig = new.signature
        
        breaking_reasons = []
        
        # Path changed (breaking)
        if old_sig.get("path") != new_sig.get("path"):
            breaking_reasons.append(
                f"Path changed from {old_sig.get('path')} to {new_sig.get('path')}"
            )
        
        # Method changed (breaking)
        if old_sig.get("method") != new_sig.get("method"):
            breaking_reasons.append(
                f"HTTP method changed from {old_sig.get('method')} "
                f"to {new_sig.get('method')}"
            )
        
        # Required query params removed (breaking)
        old_params = set(old_sig.get("query_params", {}).keys())
        new_params = set(new_sig.get("query_params", {}).keys())
        
        # Actually, removing required params is okay
        # Adding required params is breaking
        for param in new_params - old_params:
            param_def = new_sig.get("query_params", {}).get(param, {})
            if param_def.get("required", False):
                breaking_reasons.append(f"Required param '{param}' added")
        
        # Response schema changed
        old_response = old_sig.get("response_schema", {})
        new_response = new_sig.get("response_schema", {})
        
        if old_response and new_response:
            # Check for removed fields
            old_fields = set(old_response.get("properties", {}).keys())
            new_fields = set(new_response.get("properties", {}).keys())
            
            removed_fields = old_fields - new_fields
            if removed_fields:
                breaking_reasons.append(
                    f"Response fields removed: {', '.join(removed_fields)}"
                )
        
        if breaking_reasons:
            change.breaking = True
            change.impact = ChangeImpact.MAJOR_BREAKING
            change.description = "; ".join(breaking_reasons)
        else:
            change.impact = ChangeImpact.COMPATIBLE
    
    def _analyze_event_change(
        self,
        change: ContractChange,
        old: Contract,
        new: Contract,
    ) -> None:
        """Analyze changes to an event contract."""
        old_sig = old.signature
        new_sig = new.signature
        
        breaking_reasons = []
        
        # Event name changed (breaking)
        if old_sig.get("event_name") != new_sig.get("event_name"):
            breaking_reasons.append(
                f"Event name changed from {old_sig.get('event_name')} "
                f"to {new_sig.get('event_name')}"
            )
        
        # Required payload fields removed (breaking)
        old_fields = set(old_sig.get("payload_schema", {}).get("required", []))
        new_fields = set(new_sig.get("payload_schema", {}).get("required", []))
        
        removed_fields = old_fields - new_fields
        if removed_fields:
            change.impact = ChangeImpact.MINOR_BREAKING
            change.description = f"Required fields removed: {', '.join(removed_fields)}"
        
        # Added required fields (breaking)
        added_fields = new_fields - old_fields
        if added_fields:
            breaking_reasons.append(
                f"Required fields added: {', '.join(added_fields)}"
            )
        
        if breaking_reasons:
            change.breaking = True
            change.impact = ChangeImpact.MAJOR_BREAKING
            change.description = "; ".join(breaking_reasons)
    
    def _analyze_schema_change(
        self,
        change: ContractChange,
        old: Contract,
        new: Contract,
    ) -> None:
        """Analyze changes to a database schema contract."""
        old_sig = old.signature
        new_sig = new.signature
        
        breaking_reasons = []
        
        # Check column changes
        old_columns = {c["name"]: c for c in old_sig.get("columns", [])}
        new_columns = {c["name"]: c for c in new_sig.get("columns", [])}
        
        # Removed columns (breaking)
        removed = set(old_columns.keys()) - set(new_columns.keys())
        if removed:
            breaking_reasons.append(f"Columns removed: {', '.join(removed)}")
        
        # Type changes (breaking)
        for name, old_col in old_columns.items():
            if name in new_columns:
                new_col = new_columns[name]
                if old_col.get("type") != new_col.get("type"):
                    breaking_reasons.append(
                        f"Column '{name}' type changed from "
                        f"{old_col.get('type')} to {new_col.get('type')}"
                    )
        
        if breaking_reasons:
            change.breaking = True
            change.impact = ChangeImpact.MAJOR_BREAKING
            change.description = "; ".join(breaking_reasons)
        else:
            change.impact = ChangeImpact.COMPATIBLE


# =============================================================================
# Impact Analyzer
# =============================================================================

class ImpactAnalyzer:
    """Analyzes the impact of changes across repositories."""
    
    def __init__(self, graph: DependencyGraph):
        self.graph = graph
        self.contract_analyzer = ContractAnalyzer()
    
    def analyze_changes(
        self,
        repo: str,
        old_contracts: Dict[str, Contract],
        new_contracts: Dict[str, Contract],
    ) -> ImpactAnalysis:
        """
        Analyze the impact of contract changes in a repository.
        
        Returns affected repositories and breaking changes.
        """
        analysis = ImpactAnalysis(source_repo=repo)
        
        # Find all contract changes
        all_contract_ids = set(old_contracts.keys()) | set(new_contracts.keys())
        
        for contract_id in all_contract_ids:
            old = old_contracts.get(contract_id)
            new = new_contracts.get(contract_id)
            
            if old and new:
                # Modified
                change = self.contract_analyzer.compare_contracts(old, new)
                if change.impact != ChangeImpact.NONE:
                    analysis.contract_changes.append(change)
                    if change.breaking:
                        analysis.breaking_changes += 1
            
            elif old and not new:
                # Removed
                change = ContractChange(
                    contract_id=contract_id,
                    change_type="removed",
                    old_signature=old.signature,
                    breaking=True,
                    impact=ChangeImpact.MAJOR_BREAKING,
                    description=f"Contract '{old.name}' removed",
                )
                analysis.contract_changes.append(change)
                analysis.breaking_changes += 1
            
            elif new and not old:
                # Added (not breaking)
                change = ContractChange(
                    contract_id=contract_id,
                    change_type="added",
                    new_signature=new.signature,
                    breaking=False,
                    impact=ChangeImpact.COMPATIBLE,
                    description=f"Contract '{new.name}' added",
                )
                analysis.contract_changes.append(change)
        
        # Find affected repositories
        dependents = self.graph.get_dependents(repo, recursive=True)
        
        for dependent in dependents:
            dep_repo = self.graph.repositories.get(dependent)
            if dep_repo:
                # Check if this repo uses any changed contracts
                for change in analysis.contract_changes:
                    if change.contract_id in dep_repo.imported_contracts:
                        if dependent not in analysis.affected_repos:
                            analysis.affected_repos.append(dependent)
        
        analysis.total_affected = len(analysis.affected_repos)
        
        # Generate recommendations
        if analysis.breaking_changes > 0:
            analysis.recommendations.append(
                f"This change includes {analysis.breaking_changes} breaking changes"
            )
            analysis.recommendations.append(
                f"Affected repositories: {', '.join(analysis.affected_repos)}"
            )
            analysis.recommendations.append(
                "Consider creating a new major version or deprecation period"
            )
        
        return analysis
    
    def generate_upgrade_plan(
        self,
        analysis: ImpactAnalysis,
    ) -> Dict[str, Any]:
        """
        Generate an upgrade plan for affected repositories.
        
        Returns ordered list of repositories to update and actions needed.
        """
        plan = {
            "source_repo": analysis.source_repo,
            "steps": [],
            "total_steps": 0,
        }
        
        # Sort affected repos by dependency order
        affected = set(analysis.affected_repos)
        sorted_repos = [
            r for r in self.graph.topological_sort()
            if r in affected
        ]
        
        for i, repo in enumerate(sorted_repos):
            step = {
                "order": i + 1,
                "repository": repo,
                "actions": [],
            }
            
            # Find which contracts this repo needs to update
            repo_data = self.graph.repositories.get(repo)
            if repo_data:
                for change in analysis.contract_changes:
                    if change.contract_id in repo_data.imported_contracts:
                        if change.breaking:
                            step["actions"].append({
                                "type": "update_code",
                                "contract": change.contract_id,
                                "description": change.description,
                            })
            
            if step["actions"]:
                plan["steps"].append(step)
        
        plan["total_steps"] = len(plan["steps"])
        return plan


# =============================================================================
# Dependency Detector
# =============================================================================

class DependencyDetector:
    """Detects dependencies from code and configuration."""
    
    def detect_python_dependencies(self, code: str, requirements: str = "") -> List[str]:
        """Detect Python package dependencies."""
        deps = []
        
        # From import statements
        import_pattern = r'^\s*(?:from|import)\s+(\w+)'
        import re
        for match in re.finditer(import_pattern, code, re.MULTILINE):
            pkg = match.group(1)
            if pkg not in ["os", "sys", "re", "json", "time", "typing"]:
                deps.append(pkg)
        
        # From requirements.txt
        if requirements:
            for line in requirements.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Handle package[extras]==version format
                    pkg = re.split(r'[=<>!\[\]]', line)[0]
                    if pkg:
                        deps.append(pkg)
        
        return list(set(deps))
    
    def detect_npm_dependencies(self, package_json: Dict[str, Any]) -> List[str]:
        """Detect npm package dependencies."""
        deps = []
        
        for dep_type in ["dependencies", "devDependencies", "peerDependencies"]:
            deps.extend(package_json.get(dep_type, {}).keys())
        
        return list(set(deps))
    
    def detect_api_dependencies(self, code: str) -> List[Dict[str, Any]]:
        """Detect API endpoint dependencies from code."""
        deps = []
        import re
        
        # Look for HTTP client calls
        patterns = [
            r'(?:fetch|axios|request)\s*\(\s*[\'"]([^\'"]+)[\'"]',
            r'(?:get|post|put|delete|patch)\s*\(\s*[\'"]([^\'"]+)[\'"]',
            r'httpClient\.\w+\s*\(\s*[\'"]([^\'"]+)[\'"]',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, code, re.IGNORECASE):
                url = match.group(1)
                deps.append({
                    "type": "api",
                    "url": url,
                })
        
        return deps
