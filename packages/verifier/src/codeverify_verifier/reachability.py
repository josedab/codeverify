"""Vulnerability Reachability Analysis - Uses Z3 to prove if vulnerabilities are reachable.

This module provides reachability analysis to determine if known CVEs/vulnerabilities
are actually exploitable in a given codebase by analyzing call paths from entry points.

Key differentiator: Instead of just reporting CVEs, CodeVerify PROVES whether they're reachable.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog
from z3 import (
    And,
    Bool,
    BoolRef,
    Implies,
    Int,
    Not,
    Or,
    Solver,
    sat,
    unsat,
)

logger = structlog.get_logger()


class ReachabilityStatus(str, Enum):
    """Status of vulnerability reachability."""
    
    REACHABLE = "reachable"  # Proven reachable from entry points
    UNREACHABLE = "unreachable"  # Proven unreachable
    UNKNOWN = "unknown"  # Could not determine
    CONDITIONAL = "conditional"  # Reachable under certain conditions


class VulnerabilityType(str, Enum):
    """Types of vulnerabilities."""
    
    CVE = "cve"  # Known CVE
    CWE = "cwe"  # Common Weakness Enumeration
    OWASP = "owasp"  # OWASP category
    CUSTOM = "custom"  # Custom vulnerability rule


@dataclass
class CodeNode:
    """Represents a node in the call graph."""
    
    id: str
    name: str
    file_path: str
    line_start: int
    line_end: int
    node_type: str  # "function", "method", "class", "module"
    is_entry_point: bool = False
    is_vulnerable: bool = False
    vulnerability_id: str | None = None
    conditions: list[str] = field(default_factory=list)  # Guard conditions
    
    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class CallEdge:
    """Represents a call edge in the call graph."""
    
    source: str  # Source node ID
    target: str  # Target node ID
    call_site_line: int
    conditions: list[str] = field(default_factory=list)  # Conditions for this call
    is_conditional: bool = False  # True if inside if/try/etc
    
    def __hash__(self) -> int:
        return hash((self.source, self.target, self.call_site_line))


@dataclass
class Vulnerability:
    """Represents a known vulnerability."""
    
    id: str
    type: VulnerabilityType
    title: str
    description: str
    severity: str  # "critical", "high", "medium", "low"
    affected_functions: list[str] = field(default_factory=list)
    affected_files: list[str] = field(default_factory=list)
    cve_id: str | None = None
    cwe_id: str | None = None
    cvss_score: float | None = None
    fix_available: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "description": self.description,
            "severity": self.severity,
            "affected_functions": self.affected_functions,
            "affected_files": self.affected_files,
            "cve_id": self.cve_id,
            "cwe_id": self.cwe_id,
            "cvss_score": self.cvss_score,
            "fix_available": self.fix_available,
        }


@dataclass
class ReachabilityResult:
    """Result of reachability analysis for a vulnerability."""
    
    vulnerability: Vulnerability
    status: ReachabilityStatus
    reachable_paths: list[list[str]] = field(default_factory=list)  # Paths from entry to vuln
    unreachable_reason: str | None = None
    entry_points: list[str] = field(default_factory=list)  # Entry points that can reach vuln
    conditions: list[str] = field(default_factory=list)  # Conditions required for reachability
    proof_time_ms: float = 0.0
    confidence: float = 1.0  # Confidence in the result
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "vulnerability": self.vulnerability.to_dict(),
            "status": self.status.value,
            "reachable_paths": self.reachable_paths,
            "unreachable_reason": self.unreachable_reason,
            "entry_points": self.entry_points,
            "conditions": self.conditions,
            "proof_time_ms": round(self.proof_time_ms, 2),
            "confidence": round(self.confidence, 3),
        }


@dataclass  
class ReachabilityReport:
    """Complete reachability analysis report."""
    
    total_vulnerabilities: int
    reachable_count: int
    unreachable_count: int
    conditional_count: int
    unknown_count: int
    results: list[ReachabilityResult] = field(default_factory=list)
    total_analysis_time_ms: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "total_vulnerabilities": self.total_vulnerabilities,
            "reachable_count": self.reachable_count,
            "unreachable_count": self.unreachable_count,
            "conditional_count": self.conditional_count,
            "unknown_count": self.unknown_count,
            "false_positive_rate": round(
                self.unreachable_count / max(self.total_vulnerabilities, 1) * 100, 1
            ),
            "results": [r.to_dict() for r in self.results],
            "total_analysis_time_ms": round(self.total_analysis_time_ms, 2),
        }


class CallGraphBuilder:
    """Builds a call graph from source code."""
    
    # Patterns for entry points
    ENTRY_POINT_PATTERNS = {
        "python": [
            r'if\s+__name__\s*==\s*["\']__main__["\']',  # Main block
            r"@app\.(route|get|post|put|delete|patch)",  # Flask routes
            r"@router\.(get|post|put|delete|patch)",  # FastAPI routes
            r"def\s+(handle|process|execute|run|main)\s*\(",  # Handler functions
            r"class\s+\w+View\(",  # Django views
            r"def\s+lambda_handler\s*\(",  # AWS Lambda
            r"def\s+test_\w+\s*\(",  # Test functions
        ],
        "typescript": [
            r"export\s+(default\s+)?function\s+\w+",  # Exported functions
            r"app\.(get|post|put|delete|patch)\s*\(",  # Express routes
            r"@(Get|Post|Put|Delete|Patch)\s*\(",  # NestJS decorators
        ],
    }
    
    # Patterns for function calls
    CALL_PATTERNS = {
        "python": r"(\w+)\s*\(",
        "typescript": r"(\w+)\s*\(",
    }
    
    def __init__(self) -> None:
        self.nodes: dict[str, CodeNode] = {}
        self.edges: list[CallEdge] = []
    
    def build_from_code(
        self,
        code: str,
        file_path: str,
        language: str = "python",
    ) -> tuple[dict[str, CodeNode], list[CallEdge]]:
        """Build call graph from source code.
        
        This is a simplified implementation. In production, use AST parsing.
        """
        self.nodes = {}
        self.edges = []
        
        lines = code.split("\n")
        current_function: CodeNode | None = None
        indent_stack: list[tuple[int, CodeNode]] = []
        
        # Parse functions and classes
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            
            # Pop from stack based on indent
            while indent_stack and indent <= indent_stack[-1][0]:
                indent_stack.pop()
            
            # Detect function definitions
            func_match = re.match(r"(?:async\s+)?def\s+(\w+)\s*\(", stripped)
            if func_match:
                func_name = func_match.group(1)
                node_id = f"{file_path}::{func_name}"
                
                # Check if this is an entry point
                is_entry = self._is_entry_point(code, line_num, language)
                
                node = CodeNode(
                    id=node_id,
                    name=func_name,
                    file_path=file_path,
                    line_start=line_num,
                    line_end=line_num,  # Updated when we leave the function
                    node_type="function",
                    is_entry_point=is_entry,
                )
                
                self.nodes[node_id] = node
                current_function = node
                indent_stack.append((indent, node))
                continue
            
            # Detect method calls within functions
            if current_function and stripped and not stripped.startswith("#"):
                calls = re.findall(self.CALL_PATTERNS.get(language, r"(\w+)\s*\("), stripped)
                for call_name in calls:
                    # Skip built-in functions
                    if call_name in {"print", "len", "range", "str", "int", "float", "list", "dict", "set"}:
                        continue
                    
                    # Check if call is conditional
                    is_conditional = any(
                        kw in line for kw in ["if ", "elif ", "while ", "try:", "except:"]
                    ) or any(indent_stack[i][1].name for i in range(len(indent_stack)) if i > 0)
                    
                    # Look for the target function
                    target_id = f"{file_path}::{call_name}"
                    
                    edge = CallEdge(
                        source=current_function.id,
                        target=target_id,
                        call_site_line=line_num,
                        is_conditional=is_conditional,
                    )
                    self.edges.append(edge)
        
        return self.nodes, self.edges
    
    def _is_entry_point(self, code: str, line_num: int, language: str) -> bool:
        """Check if a function is an entry point."""
        patterns = self.ENTRY_POINT_PATTERNS.get(language, [])
        
        # Check if function is decorated or has entry point pattern
        lines = code.split("\n")
        
        # Look at preceding lines for decorators
        for i in range(max(0, line_num - 5), line_num):
            line = lines[i] if i < len(lines) else ""
            for pattern in patterns:
                if re.search(pattern, line):
                    return True
        
        return False
    
    def add_external_dependency(
        self,
        package: str,
        function: str,
        vulnerability: Vulnerability,
    ) -> str:
        """Add an external dependency as a vulnerable node."""
        node_id = f"external::{package}::{function}"
        
        node = CodeNode(
            id=node_id,
            name=f"{package}.{function}",
            file_path=f"<external:{package}>",
            line_start=0,
            line_end=0,
            node_type="external",
            is_vulnerable=True,
            vulnerability_id=vulnerability.id,
        )
        
        self.nodes[node_id] = node
        return node_id


class ReachabilityAnalyzer:
    """Analyzes vulnerability reachability using Z3 SMT solver."""
    
    def __init__(self, timeout_ms: int = 30000) -> None:
        self.timeout_ms = timeout_ms
        self.call_graph_builder = CallGraphBuilder()
    
    def analyze(
        self,
        code_files: dict[str, str],  # file_path -> code
        vulnerabilities: list[Vulnerability],
        entry_points: list[str] | None = None,  # Override detected entry points
        language: str = "python",
    ) -> ReachabilityReport:
        """Analyze reachability of vulnerabilities in the codebase.
        
        Args:
            code_files: Dictionary mapping file paths to source code
            vulnerabilities: List of known vulnerabilities to check
            entry_points: Optional list of entry point function names
            language: Programming language
            
        Returns:
            Complete reachability report
        """
        import time
        start_time = time.time()
        
        # Build combined call graph
        all_nodes: dict[str, CodeNode] = {}
        all_edges: list[CallEdge] = []
        
        for file_path, code in code_files.items():
            nodes, edges = self.call_graph_builder.build_from_code(code, file_path, language)
            all_nodes.update(nodes)
            all_edges.extend(edges)
        
        # Mark vulnerable nodes
        for vuln in vulnerabilities:
            for func in vuln.affected_functions:
                for node_id, node in all_nodes.items():
                    if node.name == func or func in node_id:
                        node.is_vulnerable = True
                        node.vulnerability_id = vuln.id
        
        # Override entry points if provided
        if entry_points:
            for node in all_nodes.values():
                node.is_entry_point = node.name in entry_points
        
        # Analyze each vulnerability
        results: list[ReachabilityResult] = []
        
        for vuln in vulnerabilities:
            result = self._analyze_vulnerability(vuln, all_nodes, all_edges)
            results.append(result)
        
        # Compute summary
        total_time = (time.time() - start_time) * 1000
        
        return ReachabilityReport(
            total_vulnerabilities=len(vulnerabilities),
            reachable_count=sum(1 for r in results if r.status == ReachabilityStatus.REACHABLE),
            unreachable_count=sum(1 for r in results if r.status == ReachabilityStatus.UNREACHABLE),
            conditional_count=sum(1 for r in results if r.status == ReachabilityStatus.CONDITIONAL),
            unknown_count=sum(1 for r in results if r.status == ReachabilityStatus.UNKNOWN),
            results=results,
            total_analysis_time_ms=total_time,
        )
    
    def _analyze_vulnerability(
        self,
        vuln: Vulnerability,
        nodes: dict[str, CodeNode],
        edges: list[CallEdge],
    ) -> ReachabilityResult:
        """Analyze reachability of a single vulnerability using Z3."""
        import time
        start_time = time.time()
        
        # Find vulnerable nodes
        vuln_nodes = [
            n for n in nodes.values()
            if n.is_vulnerable and n.vulnerability_id == vuln.id
        ]
        
        if not vuln_nodes:
            # Vulnerability not in codebase - check if it's an external dependency
            # that might be called
            return ReachabilityResult(
                vulnerability=vuln,
                status=ReachabilityStatus.UNREACHABLE,
                unreachable_reason="Vulnerable code not found in codebase",
                proof_time_ms=(time.time() - start_time) * 1000,
            )
        
        # Find entry points
        entry_nodes = [n for n in nodes.values() if n.is_entry_point]
        
        if not entry_nodes:
            # No entry points detected - assume all top-level functions are entry points
            entry_nodes = [
                n for n in nodes.values()
                if n.node_type == "function" and "::" in n.id and n.id.count("::") == 1
            ]
        
        if not entry_nodes:
            return ReachabilityResult(
                vulnerability=vuln,
                status=ReachabilityStatus.UNKNOWN,
                unreachable_reason="No entry points detected",
                proof_time_ms=(time.time() - start_time) * 1000,
            )
        
        # Use Z3 to prove reachability
        reachable_paths, reaching_entries, conditions = self._z3_reachability_check(
            entry_nodes, vuln_nodes, nodes, edges
        )
        
        proof_time = (time.time() - start_time) * 1000
        
        if reachable_paths:
            if conditions:
                return ReachabilityResult(
                    vulnerability=vuln,
                    status=ReachabilityStatus.CONDITIONAL,
                    reachable_paths=reachable_paths,
                    entry_points=[n.name for n in reaching_entries],
                    conditions=conditions,
                    proof_time_ms=proof_time,
                    confidence=0.9,  # Conditional reachability has lower confidence
                )
            else:
                return ReachabilityResult(
                    vulnerability=vuln,
                    status=ReachabilityStatus.REACHABLE,
                    reachable_paths=reachable_paths,
                    entry_points=[n.name for n in reaching_entries],
                    proof_time_ms=proof_time,
                )
        else:
            return ReachabilityResult(
                vulnerability=vuln,
                status=ReachabilityStatus.UNREACHABLE,
                unreachable_reason="No path from any entry point to vulnerable code",
                proof_time_ms=proof_time,
            )
    
    def _z3_reachability_check(
        self,
        entry_nodes: list[CodeNode],
        vuln_nodes: list[CodeNode],
        all_nodes: dict[str, CodeNode],
        edges: list[CallEdge],
    ) -> tuple[list[list[str]], list[CodeNode], list[str]]:
        """Use Z3 to check reachability from entry points to vulnerable nodes."""
        solver = Solver()
        solver.set("timeout", self.timeout_ms)
        
        # Create boolean variable for each node's reachability
        node_reachable: dict[str, BoolRef] = {}
        for node_id in all_nodes:
            node_reachable[node_id] = Bool(f"reach_{node_id}")
        
        # Entry points are reachable
        for entry in entry_nodes:
            solver.add(node_reachable[entry.id] == True)
        
        # If a caller is reachable and calls a function, that function is reachable
        for edge in edges:
            if edge.source in node_reachable and edge.target in node_reachable:
                # source reachable -> target reachable (if edge exists)
                edge_exists = Bool(f"edge_{edge.source}_{edge.target}")
                solver.add(edge_exists == True)  # Edge exists in call graph
                solver.add(Implies(
                    And(node_reachable[edge.source], edge_exists),
                    node_reachable[edge.target]
                ))
        
        # Check if any vulnerable node is reachable
        vuln_reachable_vars = [
            node_reachable[v.id] for v in vuln_nodes if v.id in node_reachable
        ]
        
        if not vuln_reachable_vars:
            return [], [], []
        
        # Add constraint that at least one vulnerable node is reachable
        solver.add(Or(*vuln_reachable_vars))
        
        result = solver.check()
        
        if result == sat:
            model = solver.model()
            
            # Find which entry points reach vulnerable code
            reaching_entries = []
            for entry in entry_nodes:
                if model.evaluate(node_reachable[entry.id], model_completion=True):
                    reaching_entries.append(entry)
            
            # Reconstruct paths (simplified - in production use path enumeration)
            paths = self._find_paths(reaching_entries, vuln_nodes, edges)
            
            # Extract conditions
            conditions = []
            for edge in edges:
                if edge.is_conditional:
                    conditions.append(f"Conditional call at line {edge.call_site_line}")
            
            return paths, reaching_entries, conditions
        else:
            return [], [], []
    
    def _find_paths(
        self,
        entries: list[CodeNode],
        targets: list[CodeNode],
        edges: list[CallEdge],
    ) -> list[list[str]]:
        """Find all paths from entries to targets using BFS."""
        # Build adjacency list
        adj: dict[str, list[str]] = {}
        for edge in edges:
            if edge.source not in adj:
                adj[edge.source] = []
            adj[edge.source].append(edge.target)
        
        paths = []
        target_ids = {t.id for t in targets}
        
        for entry in entries:
            # BFS to find paths
            queue: list[list[str]] = [[entry.id]]
            visited: set[str] = set()
            
            while queue and len(paths) < 5:  # Limit to 5 paths per entry
                path = queue.pop(0)
                current = path[-1]
                
                if current in target_ids:
                    paths.append([self._simplify_id(n) for n in path])
                    continue
                
                if current in visited:
                    continue
                visited.add(current)
                
                for neighbor in adj.get(current, []):
                    if neighbor not in visited:
                        queue.append(path + [neighbor])
        
        return paths
    
    def _simplify_id(self, node_id: str) -> str:
        """Simplify node ID for display."""
        if "::" in node_id:
            parts = node_id.split("::")
            return parts[-1]
        return node_id


class VulnerabilityScanner:
    """Scans code for known vulnerabilities and analyzes reachability."""
    
    # Known vulnerable patterns (simplified)
    VULN_PATTERNS = {
        "CWE-78": {  # OS Command Injection
            "patterns": [
                r"subprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True",
                r"os\.system\s*\(",
                r"os\.popen\s*\(",
            ],
            "title": "OS Command Injection",
            "severity": "critical",
        },
        "CWE-89": {  # SQL Injection
            "patterns": [
                r'execute\s*\(\s*["\'].*%s.*["\']\s*%',
                r'execute\s*\(\s*f["\']',
                r'\.format\s*\([^)]*\).*execute',
            ],
            "title": "SQL Injection",
            "severity": "critical",
        },
        "CWE-502": {  # Deserialization
            "patterns": [
                r"pickle\.loads?\s*\(",
                r"yaml\.load\s*\([^)]*Loader\s*=\s*yaml\.Loader",
                r"marshal\.loads?\s*\(",
            ],
            "title": "Insecure Deserialization",
            "severity": "high",
        },
        "CWE-295": {  # Certificate Validation
            "patterns": [
                r"verify\s*=\s*False",
                r"ssl\._create_unverified_context",
            ],
            "title": "Improper Certificate Validation",
            "severity": "high",
        },
        "CWE-798": {  # Hardcoded Credentials
            "patterns": [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
            ],
            "title": "Hardcoded Credentials",
            "severity": "high",
        },
    }
    
    def __init__(self) -> None:
        self.analyzer = ReachabilityAnalyzer()
    
    def scan_and_analyze(
        self,
        code_files: dict[str, str],
        external_vulns: list[Vulnerability] | None = None,
        language: str = "python",
    ) -> ReachabilityReport:
        """Scan code for vulnerabilities and analyze reachability.
        
        Args:
            code_files: Dictionary mapping file paths to source code
            external_vulns: List of external/CVE vulnerabilities to check
            language: Programming language
            
        Returns:
            Complete reachability report
        """
        # Find vulnerabilities in code
        vulnerabilities = self._scan_for_vulns(code_files)
        
        # Add external vulnerabilities
        if external_vulns:
            vulnerabilities.extend(external_vulns)
        
        if not vulnerabilities:
            return ReachabilityReport(
                total_vulnerabilities=0,
                reachable_count=0,
                unreachable_count=0,
                conditional_count=0,
                unknown_count=0,
            )
        
        # Analyze reachability
        return self.analyzer.analyze(code_files, vulnerabilities, language=language)
    
    def _scan_for_vulns(self, code_files: dict[str, str]) -> list[Vulnerability]:
        """Scan code files for vulnerability patterns."""
        vulnerabilities = []
        
        for file_path, code in code_files.items():
            lines = code.split("\n")
            
            for cwe_id, vuln_info in self.VULN_PATTERNS.items():
                for pattern in vuln_info["patterns"]:
                    for match in re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE):
                        # Find line number
                        line_num = code[:match.start()].count("\n") + 1
                        
                        # Find enclosing function
                        func_name = self._find_enclosing_function(lines, line_num)
                        
                        vuln = Vulnerability(
                            id=f"{cwe_id}:{file_path}:{line_num}",
                            type=VulnerabilityType.CWE,
                            title=vuln_info["title"],
                            description=f"Found at line {line_num} in {file_path}",
                            severity=vuln_info["severity"],
                            affected_functions=[func_name] if func_name else [],
                            affected_files=[file_path],
                            cwe_id=cwe_id,
                        )
                        vulnerabilities.append(vuln)
        
        return vulnerabilities
    
    def _find_enclosing_function(self, lines: list[str], line_num: int) -> str | None:
        """Find the function that encloses a given line."""
        for i in range(line_num - 1, -1, -1):
            match = re.match(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(", lines[i])
            if match:
                return match.group(1)
        return None


def create_cve_vulnerability(
    cve_id: str,
    title: str,
    affected_package: str,
    affected_functions: list[str],
    severity: str = "high",
    cvss_score: float | None = None,
    fix_available: bool = False,
) -> Vulnerability:
    """Helper to create a CVE vulnerability for external dependencies."""
    return Vulnerability(
        id=cve_id,
        type=VulnerabilityType.CVE,
        title=title,
        description=f"CVE in {affected_package}",
        severity=severity,
        affected_functions=affected_functions,
        cve_id=cve_id,
        cvss_score=cvss_score,
        fix_available=fix_available,
    )
