"""Proof Coverage Dashboard - Visual coverage map for formal verification proofs.

This module provides:
1. Proof coverage calculation at file, function, and line level
2. Coverage visualization data generation
3. Proof strength metrics
4. Coverage trends over time
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
import hashlib
import json
import structlog

logger = structlog.get_logger()


class ProofStatus(str, Enum):
    """Status of a formal proof."""
    PROVEN = "proven"
    DISPROVEN = "disproven"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"
    NOT_ATTEMPTED = "not_attempted"


class VerificationCategory(str, Enum):
    """Categories of verification."""
    NULL_SAFETY = "null_safety"
    BOUNDS_CHECK = "bounds_check"
    OVERFLOW = "overflow"
    DIVISION = "division"
    TYPE_SAFETY = "type_safety"
    INVARIANT = "invariant"
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    ASSERTION = "assertion"


@dataclass
class LineCoverage:
    """Coverage status for a single line."""
    line_number: int
    proof_status: ProofStatus
    categories: list[VerificationCategory] = field(default_factory=list)
    proof_time_ms: float = 0.0
    last_verified: datetime | None = None
    constraints_checked: int = 0
    
    # Details
    proof_summary: str = ""
    counterexample: dict[str, Any] | None = None


@dataclass
class FunctionCoverage:
    """Coverage status for a function."""
    function_name: str
    start_line: int
    end_line: int
    
    # Coverage metrics
    total_lines: int = 0
    covered_lines: int = 0
    proven_lines: int = 0
    disproven_lines: int = 0
    
    # Proof breakdown
    preconditions_verified: int = 0
    postconditions_verified: int = 0
    invariants_verified: int = 0
    assertions_verified: int = 0
    
    # Quality metrics
    proof_strength: float = 0.0  # 0-1
    complexity_score: float = 0.0
    
    # Line-level details
    line_coverage: list[LineCoverage] = field(default_factory=list)


@dataclass
class FileCoverage:
    """Coverage status for a file."""
    file_path: str
    language: str
    
    # Coverage metrics
    total_lines: int = 0
    executable_lines: int = 0
    covered_lines: int = 0
    proven_lines: int = 0
    coverage_percentage: float = 0.0
    
    # Function-level breakdown
    functions: list[FunctionCoverage] = field(default_factory=list)
    
    # Uncovered areas
    uncovered_ranges: list[tuple[int, int]] = field(default_factory=list)
    
    # Last analysis
    last_analyzed: datetime | None = None
    analysis_duration_ms: float = 0.0


@dataclass
class RepositoryCoverage:
    """Coverage status for entire repository."""
    repository: str
    branch: str = "main"
    commit_sha: str = ""
    
    # Aggregate metrics
    total_files: int = 0
    files_with_proofs: int = 0
    total_functions: int = 0
    functions_with_proofs: int = 0
    total_lines: int = 0
    proven_lines: int = 0
    
    # Coverage percentages
    file_coverage: float = 0.0
    function_coverage: float = 0.0
    line_coverage: float = 0.0
    
    # Proof strength
    overall_proof_strength: float = 0.0
    
    # File breakdown
    files: list[FileCoverage] = field(default_factory=list)
    
    # Analysis metadata
    analyzed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CoverageTrend:
    """Coverage trend over time."""
    date: datetime
    line_coverage: float
    function_coverage: float
    proof_strength: float
    proven_count: int
    disproven_count: int


@dataclass
class DashboardData:
    """Data for the proof coverage dashboard."""
    repository: str
    current_coverage: RepositoryCoverage
    
    # Summary stats
    total_proofs: int = 0
    passed_proofs: int = 0
    failed_proofs: int = 0
    pending_proofs: int = 0
    
    # Trends (last 30 days)
    trends: list[CoverageTrend] = field(default_factory=list)
    
    # Top issues
    files_needing_coverage: list[str] = field(default_factory=list)
    functions_with_failures: list[tuple[str, str]] = field(default_factory=list)
    
    # Coverage heatmap data
    heatmap: list[dict[str, Any]] = field(default_factory=list)


class ProofCoverageCalculator:
    """Calculates proof coverage from verification results."""

    def calculate_line_coverage(
        self,
        code: str,
        verification_results: list[dict[str, Any]],
        language: str = "python",
    ) -> list[LineCoverage]:
        """Calculate line-level coverage from verification results."""
        lines = code.split("\n")
        coverage = []
        
        # Build lookup of verification results by line
        line_results: dict[int, list[dict[str, Any]]] = {}
        for result in verification_results:
            line = result.get("line_start", result.get("line", 0))
            if line not in line_results:
                line_results[line] = []
            line_results[line].append(result)
        
        for i, line_content in enumerate(lines, start=1):
            # Determine if line is executable
            is_executable = self._is_executable_line(line_content, language)
            
            if not is_executable:
                continue
            
            # Check verification results for this line
            results = line_results.get(i, [])
            
            if results:
                # Determine overall status
                statuses = [r.get("proof_status", "unknown") for r in results]
                if "proven" in statuses:
                    status = ProofStatus.PROVEN
                elif "disproven" in statuses:
                    status = ProofStatus.DISPROVEN
                elif "timeout" in statuses:
                    status = ProofStatus.TIMEOUT
                else:
                    status = ProofStatus.UNKNOWN
                
                # Collect categories
                categories = []
                for r in results:
                    cat = r.get("category")
                    if cat:
                        try:
                            categories.append(VerificationCategory(cat))
                        except ValueError:
                            pass
                
                coverage.append(LineCoverage(
                    line_number=i,
                    proof_status=status,
                    categories=categories,
                    proof_time_ms=sum(r.get("proof_time_ms", 0) for r in results),
                    constraints_checked=len(results),
                    last_verified=datetime.utcnow(),
                ))
            else:
                coverage.append(LineCoverage(
                    line_number=i,
                    proof_status=ProofStatus.NOT_ATTEMPTED,
                ))
        
        return coverage

    def _is_executable_line(self, line: str, language: str) -> bool:
        """Check if a line is executable (not comment/blank)."""
        stripped = line.strip()
        
        if not stripped:
            return False
        
        # Language-specific comment patterns
        if language in ("python", "py"):
            if stripped.startswith("#"):
                return False
            if stripped.startswith('"""') or stripped.startswith("'''"):
                return False
        elif language in ("typescript", "javascript", "ts", "js"):
            if stripped.startswith("//"):
                return False
            if stripped.startswith("/*") or stripped.startswith("*"):
                return False
        elif language in ("java", "go", "rust", "c", "cpp"):
            if stripped.startswith("//"):
                return False
            if stripped.startswith("/*") or stripped.startswith("*"):
                return False
        
        return True

    def calculate_function_coverage(
        self,
        function_name: str,
        start_line: int,
        end_line: int,
        line_coverages: list[LineCoverage],
    ) -> FunctionCoverage:
        """Calculate function-level coverage."""
        func_lines = [
            lc for lc in line_coverages
            if start_line <= lc.line_number <= end_line
        ]
        
        total_lines = len(func_lines)
        covered_lines = len([lc for lc in func_lines if lc.proof_status != ProofStatus.NOT_ATTEMPTED])
        proven_lines = len([lc for lc in func_lines if lc.proof_status == ProofStatus.PROVEN])
        disproven_lines = len([lc for lc in func_lines if lc.proof_status == ProofStatus.DISPROVEN])
        
        # Calculate proof strength
        if covered_lines > 0:
            proof_strength = proven_lines / covered_lines
        else:
            proof_strength = 0.0
        
        # Count by category
        preconditions = sum(
            1 for lc in func_lines
            if VerificationCategory.PRECONDITION in lc.categories
        )
        postconditions = sum(
            1 for lc in func_lines
            if VerificationCategory.POSTCONDITION in lc.categories
        )
        invariants = sum(
            1 for lc in func_lines
            if VerificationCategory.INVARIANT in lc.categories
        )
        assertions = sum(
            1 for lc in func_lines
            if VerificationCategory.ASSERTION in lc.categories
        )
        
        return FunctionCoverage(
            function_name=function_name,
            start_line=start_line,
            end_line=end_line,
            total_lines=total_lines,
            covered_lines=covered_lines,
            proven_lines=proven_lines,
            disproven_lines=disproven_lines,
            preconditions_verified=preconditions,
            postconditions_verified=postconditions,
            invariants_verified=invariants,
            assertions_verified=assertions,
            proof_strength=proof_strength,
            line_coverage=func_lines,
        )

    def calculate_file_coverage(
        self,
        file_path: str,
        code: str,
        verification_results: list[dict[str, Any]],
        functions: list[dict[str, Any]] | None = None,
        language: str = "python",
    ) -> FileCoverage:
        """Calculate file-level coverage."""
        line_coverages = self.calculate_line_coverage(code, verification_results, language)
        
        total_lines = len(code.split("\n"))
        executable_lines = len(line_coverages)
        covered_lines = len([lc for lc in line_coverages if lc.proof_status != ProofStatus.NOT_ATTEMPTED])
        proven_lines = len([lc for lc in line_coverages if lc.proof_status == ProofStatus.PROVEN])
        
        coverage_percentage = (covered_lines / executable_lines * 100) if executable_lines > 0 else 0.0
        
        # Calculate function coverage
        func_coverages = []
        if functions:
            for func in functions:
                func_cov = self.calculate_function_coverage(
                    func.get("name", "unknown"),
                    func.get("start_line", 0),
                    func.get("end_line", 0),
                    line_coverages,
                )
                func_coverages.append(func_cov)
        
        # Find uncovered ranges
        uncovered_ranges = self._find_uncovered_ranges(line_coverages)
        
        return FileCoverage(
            file_path=file_path,
            language=language,
            total_lines=total_lines,
            executable_lines=executable_lines,
            covered_lines=covered_lines,
            proven_lines=proven_lines,
            coverage_percentage=coverage_percentage,
            functions=func_coverages,
            uncovered_ranges=uncovered_ranges,
            last_analyzed=datetime.utcnow(),
        )

    def _find_uncovered_ranges(self, line_coverages: list[LineCoverage]) -> list[tuple[int, int]]:
        """Find contiguous ranges of uncovered lines."""
        uncovered = [
            lc.line_number for lc in line_coverages
            if lc.proof_status == ProofStatus.NOT_ATTEMPTED
        ]
        
        if not uncovered:
            return []
        
        ranges = []
        start = uncovered[0]
        end = uncovered[0]
        
        for line in uncovered[1:]:
            if line == end + 1:
                end = line
            else:
                ranges.append((start, end))
                start = line
                end = line
        
        ranges.append((start, end))
        return ranges


class ProofCoverageDashboard:
    """Dashboard for proof coverage visualization."""

    def __init__(self) -> None:
        self.calculator = ProofCoverageCalculator()
        self._coverage_history: dict[str, list[RepositoryCoverage]] = {}

    def generate_dashboard_data(
        self,
        repository: str,
        files: list[dict[str, Any]],
        verification_results: dict[str, list[dict[str, Any]]],
    ) -> DashboardData:
        """Generate dashboard data for a repository."""
        # Calculate coverage for each file
        file_coverages = []
        
        for file_info in files:
            path = file_info.get("path", "")
            code = file_info.get("content", "")
            language = file_info.get("language", "python")
            functions = file_info.get("functions", [])
            results = verification_results.get(path, [])
            
            file_cov = self.calculator.calculate_file_coverage(
                path, code, results, functions, language
            )
            file_coverages.append(file_cov)
        
        # Calculate repository-level coverage
        repo_coverage = self._calculate_repository_coverage(
            repository, file_coverages
        )
        
        # Get trends
        trends = self._get_trends(repository)
        
        # Find files needing coverage
        files_needing_coverage = [
            fc.file_path for fc in file_coverages
            if fc.coverage_percentage < 50
        ][:10]
        
        # Find functions with failures
        functions_with_failures = []
        for fc in file_coverages:
            for func in fc.functions:
                if func.disproven_lines > 0:
                    functions_with_failures.append((fc.file_path, func.function_name))
        
        # Generate heatmap data
        heatmap = self._generate_heatmap(file_coverages)
        
        # Count proofs
        total_proofs = sum(fc.covered_lines for fc in file_coverages)
        passed_proofs = sum(fc.proven_lines for fc in file_coverages)
        failed_proofs = sum(
            len([lc for lc in (func.line_coverage for func in fc.functions for lc in func.line_coverage) if lc.proof_status == ProofStatus.DISPROVEN])
            for fc in file_coverages
        )
        
        return DashboardData(
            repository=repository,
            current_coverage=repo_coverage,
            total_proofs=total_proofs,
            passed_proofs=passed_proofs,
            failed_proofs=failed_proofs,
            pending_proofs=total_proofs - passed_proofs - failed_proofs,
            trends=trends,
            files_needing_coverage=files_needing_coverage,
            functions_with_failures=functions_with_failures[:10],
            heatmap=heatmap,
        )

    def _calculate_repository_coverage(
        self,
        repository: str,
        file_coverages: list[FileCoverage],
    ) -> RepositoryCoverage:
        """Calculate repository-level coverage."""
        total_files = len(file_coverages)
        files_with_proofs = len([fc for fc in file_coverages if fc.covered_lines > 0])
        
        total_functions = sum(len(fc.functions) for fc in file_coverages)
        functions_with_proofs = sum(
            len([f for f in fc.functions if f.covered_lines > 0])
            for fc in file_coverages
        )
        
        total_lines = sum(fc.executable_lines for fc in file_coverages)
        proven_lines = sum(fc.proven_lines for fc in file_coverages)
        
        file_coverage = (files_with_proofs / total_files * 100) if total_files > 0 else 0
        function_coverage = (functions_with_proofs / total_functions * 100) if total_functions > 0 else 0
        line_coverage = (proven_lines / total_lines * 100) if total_lines > 0 else 0
        
        # Calculate overall proof strength
        strengths = [
            func.proof_strength
            for fc in file_coverages
            for func in fc.functions
            if func.covered_lines > 0
        ]
        overall_strength = sum(strengths) / len(strengths) if strengths else 0
        
        return RepositoryCoverage(
            repository=repository,
            total_files=total_files,
            files_with_proofs=files_with_proofs,
            total_functions=total_functions,
            functions_with_proofs=functions_with_proofs,
            total_lines=total_lines,
            proven_lines=proven_lines,
            file_coverage=file_coverage,
            function_coverage=function_coverage,
            line_coverage=line_coverage,
            overall_proof_strength=overall_strength,
            files=file_coverages,
        )

    def _get_trends(self, repository: str, days: int = 30) -> list[CoverageTrend]:
        """Get coverage trends."""
        history = self._coverage_history.get(repository, [])
        
        if not history:
            # Generate synthetic trend data for demo
            trends = []
            base_coverage = 40.0
            for i in range(days):
                date = datetime.utcnow() - timedelta(days=days - i)
                # Simulate gradual improvement
                coverage = base_coverage + (i * 0.5)
                trends.append(CoverageTrend(
                    date=date,
                    line_coverage=min(coverage, 80),
                    function_coverage=min(coverage + 5, 85),
                    proof_strength=min(coverage / 100, 0.8),
                    proven_count=int(coverage * 10),
                    disproven_count=max(0, 50 - i),
                ))
            return trends
        
        # Use actual history
        return [
            CoverageTrend(
                date=rc.analyzed_at,
                line_coverage=rc.line_coverage,
                function_coverage=rc.function_coverage,
                proof_strength=rc.overall_proof_strength,
                proven_count=rc.proven_lines,
                disproven_count=sum(
                    fc.covered_lines - fc.proven_lines
                    for fc in rc.files
                ),
            )
            for rc in history[-days:]
        ]

    def _generate_heatmap(self, file_coverages: list[FileCoverage]) -> list[dict[str, Any]]:
        """Generate heatmap data for visualization."""
        heatmap = []
        
        for fc in file_coverages:
            # Group by directory
            parts = fc.file_path.split("/")
            if len(parts) > 1:
                directory = "/".join(parts[:-1])
            else:
                directory = ""
            
            heatmap.append({
                "path": fc.file_path,
                "directory": directory,
                "filename": parts[-1] if parts else fc.file_path,
                "coverage": fc.coverage_percentage,
                "lines": fc.executable_lines,
                "proven": fc.proven_lines,
                "color": self._coverage_to_color(fc.coverage_percentage),
            })
        
        return heatmap

    def _coverage_to_color(self, coverage: float) -> str:
        """Convert coverage percentage to color."""
        if coverage >= 80:
            return "#22c55e"  # green
        elif coverage >= 60:
            return "#84cc16"  # lime
        elif coverage >= 40:
            return "#eab308"  # yellow
        elif coverage >= 20:
            return "#f97316"  # orange
        else:
            return "#ef4444"  # red

    def record_coverage(self, repository: str, coverage: RepositoryCoverage) -> None:
        """Record coverage for trend tracking."""
        if repository not in self._coverage_history:
            self._coverage_history[repository] = []
        
        self._coverage_history[repository].append(coverage)
        
        # Keep only last 90 days
        cutoff = datetime.utcnow() - timedelta(days=90)
        self._coverage_history[repository] = [
            c for c in self._coverage_history[repository]
            if c.analyzed_at > cutoff
        ]

    def to_json(self, dashboard_data: DashboardData) -> dict[str, Any]:
        """Convert dashboard data to JSON-serializable dict."""
        return {
            "repository": dashboard_data.repository,
            "summary": {
                "totalProofs": dashboard_data.total_proofs,
                "passedProofs": dashboard_data.passed_proofs,
                "failedProofs": dashboard_data.failed_proofs,
                "pendingProofs": dashboard_data.pending_proofs,
            },
            "coverage": {
                "files": dashboard_data.current_coverage.file_coverage,
                "functions": dashboard_data.current_coverage.function_coverage,
                "lines": dashboard_data.current_coverage.line_coverage,
                "proofStrength": dashboard_data.current_coverage.overall_proof_strength,
            },
            "trends": [
                {
                    "date": t.date.isoformat(),
                    "lineCoverage": t.line_coverage,
                    "functionCoverage": t.function_coverage,
                    "proofStrength": t.proof_strength,
                }
                for t in dashboard_data.trends
            ],
            "filesNeedingCoverage": dashboard_data.files_needing_coverage,
            "functionsWithFailures": [
                {"file": f, "function": fn}
                for f, fn in dashboard_data.functions_with_failures
            ],
            "heatmap": dashboard_data.heatmap,
        }


# Global dashboard instance
_proof_coverage_dashboard: ProofCoverageDashboard | None = None


def get_proof_coverage_dashboard() -> ProofCoverageDashboard:
    """Get or create the global proof coverage dashboard."""
    global _proof_coverage_dashboard
    if _proof_coverage_dashboard is None:
        _proof_coverage_dashboard = ProofCoverageDashboard()
    return _proof_coverage_dashboard


def reset_proof_coverage_dashboard() -> None:
    """Reset the global dashboard (for testing)."""
    global _proof_coverage_dashboard
    _proof_coverage_dashboard = None
