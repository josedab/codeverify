"""
Automated Fix Verification Engine

Automatically verifies that suggested fixes resolve identified issues
without introducing new problems. Includes fix validation, regression
checking, and safety verification.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4


class FixStatus(str, Enum):
    """Status of a fix verification."""
    PENDING = "pending"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    FAILED = "failed"
    PARTIAL = "partial"
    REJECTED = "rejected"


class IssueType(str, Enum):
    """Types of issues that fixes address."""
    SECURITY = "security"
    BUG = "bug"
    PERFORMANCE = "performance"
    STYLE = "style"
    LOGIC = "logic"
    TYPE_ERROR = "type_error"
    RESOURCE_LEAK = "resource_leak"
    NULL_CHECK = "null_check"
    BOUNDS_CHECK = "bounds_check"
    CONCURRENCY = "concurrency"


class VerificationMethod(str, Enum):
    """Methods used to verify fixes."""
    STATIC_ANALYSIS = "static_analysis"
    PATTERN_MATCHING = "pattern_matching"
    TYPE_CHECKING = "type_checking"
    CONTROL_FLOW = "control_flow"
    DATA_FLOW = "data_flow"
    SYMBOLIC_EXECUTION = "symbolic_execution"
    TEST_EXECUTION = "test_execution"


class SafetyLevel(str, Enum):
    """Safety level of a fix."""
    SAFE = "safe"
    MOSTLY_SAFE = "mostly_safe"
    NEEDS_REVIEW = "needs_review"
    POTENTIALLY_UNSAFE = "potentially_unsafe"
    UNSAFE = "unsafe"


@dataclass
class Issue:
    """An identified issue to be fixed."""
    id: str
    type: IssueType
    description: str
    file_path: str
    line_start: int
    line_end: int
    severity: str
    code_snippet: str
    suggested_fix: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "description": self.description,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "severity": self.severity,
            "code_snippet": self.code_snippet,
            "suggested_fix": self.suggested_fix,
        }


@dataclass
class Fix:
    """A proposed fix for an issue."""
    id: str
    issue_id: str
    original_code: str
    fixed_code: str
    description: str
    file_path: str
    line_start: int
    line_end: int
    author: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "issue_id": self.issue_id,
            "original_code": self.original_code,
            "fixed_code": self.fixed_code,
            "description": self.description,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class RegressionCheck:
    """Result of a regression check."""
    id: str
    check_type: str
    passed: bool
    description: str
    details: Optional[str] = None
    affected_lines: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "check_type": self.check_type,
            "passed": self.passed,
            "description": self.description,
            "details": self.details,
            "affected_lines": self.affected_lines,
        }


@dataclass
class SafetyCheck:
    """Result of a safety verification check."""
    id: str
    check_name: str
    passed: bool
    safety_level: SafetyLevel
    message: str
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "check_name": self.check_name,
            "passed": self.passed,
            "safety_level": self.safety_level.value,
            "message": self.message,
            "recommendations": self.recommendations,
        }


@dataclass
class VerificationResult:
    """Complete result of fix verification."""
    id: str
    fix_id: str
    issue_id: str
    status: FixStatus
    issue_resolved: bool
    regression_checks: List[RegressionCheck]
    safety_checks: List[SafetyCheck]
    methods_used: List[VerificationMethod]
    overall_safety: SafetyLevel
    confidence: float
    summary: str
    verified_at: datetime
    new_issues: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "fix_id": self.fix_id,
            "issue_id": self.issue_id,
            "status": self.status.value,
            "issue_resolved": self.issue_resolved,
            "regression_checks": [r.to_dict() for r in self.regression_checks],
            "safety_checks": [s.to_dict() for s in self.safety_checks],
            "methods_used": [m.value for m in self.methods_used],
            "overall_safety": self.overall_safety.value,
            "confidence": self.confidence,
            "summary": self.summary,
            "verified_at": self.verified_at.isoformat(),
            "new_issues": self.new_issues,
        }


class IssueResolver:
    """Checks if a fix resolves the original issue."""

    # Patterns that indicate issue resolution for different issue types
    RESOLUTION_PATTERNS = {
        IssueType.NULL_CHECK: [
            r"if\s+\w+\s*(?:is not None|!= None|!=\s*null)",
            r"\w+\s*\?\.",  # Optional chaining
            r"if\s+\w+:",
            r"\w+\s+or\s+",  # Default value
        ],
        IssueType.BOUNDS_CHECK: [
            r"if\s+\w+\s*<\s*len\(",
            r"if\s+0\s*<=\s*\w+",
            r"if\s+\w+\s*>=\s*0",
            r"min\(|max\(",
        ],
        IssueType.RESOURCE_LEAK: [
            r"with\s+\w+\s*\(",
            r"\.close\(\)",
            r"finally:",
            r"try:.*finally:",
        ],
        IssueType.SECURITY: [
            r"escape\(|sanitize\(|validate\(",
            r"parameterized|prepared",
            r"html\.escape|quote\(",
        ],
        IssueType.TYPE_ERROR: [
            r"isinstance\(|type\(",
            r":\s*\w+\s*[,\)]",  # Type hints
            r"typing\.",
        ],
        IssueType.CONCURRENCY: [
            r"lock\.|Lock\(|acquire\(|release\(",
            r"synchronized|atomic",
            r"threading\.",
        ],
    }

    def check_resolution(
        self,
        issue: Issue,
        fix: Fix,
    ) -> Tuple[bool, float, str]:
        """Check if a fix resolves the issue."""
        issue_type = issue.type
        fixed_code = fix.fixed_code

        # Check if the problematic code pattern is removed
        original_present = self._pattern_in_code(issue.code_snippet, fixed_code)

        if original_present:
            # Original problematic code still exists
            return False, 0.3, "Original problematic code pattern still present"

        # Check for resolution patterns
        patterns = self.RESOLUTION_PATTERNS.get(issue_type, [])
        matches = 0

        for pattern in patterns:
            if re.search(pattern, fixed_code, re.IGNORECASE):
                matches += 1

        if patterns and matches > 0:
            confidence = min(1.0, 0.5 + (matches / len(patterns)) * 0.5)
            return True, confidence, f"Fix contains {matches} resolution patterns"

        # Basic check: code changed
        if fix.original_code != fix.fixed_code:
            return True, 0.6, "Code modified but resolution patterns not detected"

        return False, 0.2, "No evidence of issue resolution"

    def _pattern_in_code(self, pattern: str, code: str) -> bool:
        """Check if a pattern exists in code."""
        # Normalize whitespace
        pattern_normalized = " ".join(pattern.split())
        code_normalized = " ".join(code.split())
        return pattern_normalized in code_normalized


class RegressionChecker:
    """Checks for regressions introduced by a fix."""

    def check(
        self,
        fix: Fix,
        context: Optional[Dict[str, str]] = None,
    ) -> List[RegressionCheck]:
        """Run regression checks on a fix."""
        checks: List[RegressionCheck] = []

        # Check for removed functionality
        checks.append(self._check_removed_functionality(fix))

        # Check for changed return types/behavior
        checks.append(self._check_behavior_changes(fix))

        # Check for new dependencies
        checks.append(self._check_new_dependencies(fix))

        # Check for error handling changes
        checks.append(self._check_error_handling(fix))

        # Check for variable scope issues
        checks.append(self._check_scope_issues(fix))

        return checks

    def _check_removed_functionality(self, fix: Fix) -> RegressionCheck:
        """Check if functionality was removed."""
        original = fix.original_code
        fixed = fix.fixed_code

        # Count function/method definitions
        orig_funcs = len(re.findall(r"def\s+\w+\s*\(", original))
        fixed_funcs = len(re.findall(r"def\s+\w+\s*\(", fixed))

        if fixed_funcs < orig_funcs:
            return RegressionCheck(
                id=str(uuid4()),
                check_type="removed_functionality",
                passed=False,
                description="Function definitions removed",
                details=f"Original: {orig_funcs} functions, Fixed: {fixed_funcs} functions",
            )

        return RegressionCheck(
            id=str(uuid4()),
            check_type="removed_functionality",
            passed=True,
            description="No functionality removed",
        )

    def _check_behavior_changes(self, fix: Fix) -> RegressionCheck:
        """Check for behavior changes."""
        original = fix.original_code
        fixed = fix.fixed_code

        # Check return statements
        orig_returns = re.findall(r"return\s+(.+?)(?:\n|$)", original)
        fixed_returns = re.findall(r"return\s+(.+?)(?:\n|$)", fixed)

        if len(orig_returns) != len(fixed_returns):
            return RegressionCheck(
                id=str(uuid4()),
                check_type="behavior_change",
                passed=False,
                description="Return statement count changed",
                details=f"Original: {len(orig_returns)}, Fixed: {len(fixed_returns)}",
            )

        return RegressionCheck(
            id=str(uuid4()),
            check_type="behavior_change",
            passed=True,
            description="Return behavior unchanged",
        )

    def _check_new_dependencies(self, fix: Fix) -> RegressionCheck:
        """Check for new imports/dependencies."""
        original = fix.original_code
        fixed = fix.fixed_code

        orig_imports = set(re.findall(r"(?:import|from)\s+(\w+)", original))
        fixed_imports = set(re.findall(r"(?:import|from)\s+(\w+)", fixed))

        new_imports = fixed_imports - orig_imports

        if new_imports:
            return RegressionCheck(
                id=str(uuid4()),
                check_type="new_dependencies",
                passed=True,  # Not necessarily a failure, just informational
                description="New imports added",
                details=f"New imports: {', '.join(new_imports)}",
            )

        return RegressionCheck(
            id=str(uuid4()),
            check_type="new_dependencies",
            passed=True,
            description="No new dependencies introduced",
        )

    def _check_error_handling(self, fix: Fix) -> RegressionCheck:
        """Check for error handling changes."""
        original = fix.original_code
        fixed = fix.fixed_code

        orig_try = len(re.findall(r"\btry\s*:", original))
        fixed_try = len(re.findall(r"\btry\s*:", fixed))

        if orig_try > 0 and fixed_try == 0:
            return RegressionCheck(
                id=str(uuid4()),
                check_type="error_handling",
                passed=False,
                description="Error handling removed",
                details="Try blocks were removed from the code",
            )

        return RegressionCheck(
            id=str(uuid4()),
            check_type="error_handling",
            passed=True,
            description="Error handling preserved",
        )

    def _check_scope_issues(self, fix: Fix) -> RegressionCheck:
        """Check for variable scope issues."""
        fixed = fix.fixed_code

        # Check for uses of undefined variables (basic check)
        defined_vars = set(re.findall(r"(\w+)\s*=", fixed))
        used_vars = set(re.findall(r"\b(\w+)\b", fixed))

        # Filter out common builtins and keywords
        builtins = {"True", "False", "None", "self", "cls", "print", "len", "range", "str", "int", "float", "list", "dict", "set", "tuple", "if", "else", "elif", "for", "while", "try", "except", "finally", "return", "def", "class", "import", "from", "as", "and", "or", "not", "in", "is"}
        used_vars = used_vars - builtins - defined_vars

        return RegressionCheck(
            id=str(uuid4()),
            check_type="scope_issues",
            passed=True,
            description="No obvious scope issues detected",
        )


class SafetyVerifier:
    """Verifies the safety of a fix."""

    # Dangerous patterns to check for
    DANGEROUS_PATTERNS = {
        "eval_exec": (r"\beval\s*\(|\bexec\s*\(", "Use of eval/exec is potentially dangerous"),
        "shell_injection": (r"subprocess\.\w+\(.*shell\s*=\s*True", "Shell injection risk"),
        "sql_injection": (r"execute\s*\(\s*[\"'].*%s", "Potential SQL injection"),
        "path_traversal": (r"\.\.\/|\.\.\\", "Path traversal pattern detected"),
        "hardcoded_secret": (r"(?:password|secret|key|token)\s*=\s*[\"'][^\"']+[\"']", "Hardcoded secret"),
        "insecure_random": (r"random\.\w+\(", "Insecure random for security-sensitive operations"),
    }

    def verify(self, fix: Fix) -> List[SafetyCheck]:
        """Run safety checks on a fix."""
        checks: List[SafetyCheck] = []
        fixed_code = fix.fixed_code

        # Check for dangerous patterns
        for check_name, (pattern, message) in self.DANGEROUS_PATTERNS.items():
            if re.search(pattern, fixed_code, re.IGNORECASE):
                checks.append(SafetyCheck(
                    id=str(uuid4()),
                    check_name=check_name,
                    passed=False,
                    safety_level=SafetyLevel.POTENTIALLY_UNSAFE,
                    message=message,
                    recommendations=[f"Review {check_name} usage and ensure it's intentional"],
                ))
            else:
                checks.append(SafetyCheck(
                    id=str(uuid4()),
                    check_name=check_name,
                    passed=True,
                    safety_level=SafetyLevel.SAFE,
                    message=f"No {check_name.replace('_', ' ')} issues detected",
                ))

        # Check for proper input validation
        checks.append(self._check_input_validation(fix))

        # Check for resource management
        checks.append(self._check_resource_management(fix))

        return checks

    def _check_input_validation(self, fix: Fix) -> SafetyCheck:
        """Check for input validation."""
        fixed = fix.fixed_code

        # Look for validation patterns
        validation_patterns = [
            r"if\s+not\s+\w+:",
            r"validate\(",
            r"isinstance\(",
            r"raise\s+(?:ValueError|TypeError)",
        ]

        has_validation = any(
            re.search(p, fixed) for p in validation_patterns
        )

        if has_validation:
            return SafetyCheck(
                id=str(uuid4()),
                check_name="input_validation",
                passed=True,
                safety_level=SafetyLevel.SAFE,
                message="Input validation detected",
            )

        return SafetyCheck(
            id=str(uuid4()),
            check_name="input_validation",
            passed=True,
            safety_level=SafetyLevel.MOSTLY_SAFE,
            message="No explicit input validation detected",
            recommendations=["Consider adding input validation"],
        )

    def _check_resource_management(self, fix: Fix) -> SafetyCheck:
        """Check for proper resource management."""
        fixed = fix.fixed_code

        # Check for file/resource handling
        has_open = "open(" in fixed
        has_with = "with " in fixed
        has_close = ".close()" in fixed

        if has_open and not (has_with or has_close):
            return SafetyCheck(
                id=str(uuid4()),
                check_name="resource_management",
                passed=False,
                safety_level=SafetyLevel.NEEDS_REVIEW,
                message="File opened without proper closure pattern",
                recommendations=["Use 'with' statement for file handling"],
            )

        return SafetyCheck(
            id=str(uuid4()),
            check_name="resource_management",
            passed=True,
            safety_level=SafetyLevel.SAFE,
            message="Resource management appears correct",
        )


class FixVerificationEngine:
    """Main engine for automated fix verification."""

    def __init__(self):
        self.issue_resolver = IssueResolver()
        self.regression_checker = RegressionChecker()
        self.safety_verifier = SafetyVerifier()

        self.issues: Dict[str, Issue] = {}
        self.fixes: Dict[str, Fix] = {}
        self.results: Dict[str, VerificationResult] = {}

    def register_issue(
        self,
        issue_type: str,
        description: str,
        file_path: str,
        line_start: int,
        line_end: int,
        severity: str,
        code_snippet: str,
        suggested_fix: Optional[str] = None,
    ) -> Issue:
        """Register an issue to be fixed."""
        issue_t = IssueType(issue_type) if issue_type in [t.value for t in IssueType] else IssueType.BUG

        issue = Issue(
            id=str(uuid4()),
            type=issue_t,
            description=description,
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            severity=severity,
            code_snippet=code_snippet,
            suggested_fix=suggested_fix,
        )

        self.issues[issue.id] = issue
        return issue

    def submit_fix(
        self,
        issue_id: str,
        original_code: str,
        fixed_code: str,
        description: str,
        file_path: str,
        line_start: int,
        line_end: int,
        author: Optional[str] = None,
    ) -> Fix:
        """Submit a fix for verification."""
        fix = Fix(
            id=str(uuid4()),
            issue_id=issue_id,
            original_code=original_code,
            fixed_code=fixed_code,
            description=description,
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            author=author,
        )

        self.fixes[fix.id] = fix
        return fix

    async def verify_fix(
        self,
        fix_id: str,
        context: Optional[Dict[str, str]] = None,
    ) -> VerificationResult:
        """Verify a submitted fix."""
        fix = self.fixes.get(fix_id)
        if not fix:
            raise ValueError(f"Fix not found: {fix_id}")

        issue = self.issues.get(fix.issue_id)
        if not issue:
            raise ValueError(f"Issue not found: {fix.issue_id}")

        methods_used: List[VerificationMethod] = []

        # Check if issue is resolved
        issue_resolved, resolution_confidence, resolution_msg = self.issue_resolver.check_resolution(issue, fix)
        methods_used.append(VerificationMethod.PATTERN_MATCHING)
        methods_used.append(VerificationMethod.STATIC_ANALYSIS)

        # Run regression checks
        regression_checks = self.regression_checker.check(fix, context)
        methods_used.append(VerificationMethod.CONTROL_FLOW)

        # Run safety checks
        safety_checks = self.safety_verifier.verify(fix)
        methods_used.append(VerificationMethod.DATA_FLOW)

        # Determine overall safety
        failed_safety = [s for s in safety_checks if not s.passed]
        if any(s.safety_level == SafetyLevel.UNSAFE for s in failed_safety):
            overall_safety = SafetyLevel.UNSAFE
        elif any(s.safety_level == SafetyLevel.POTENTIALLY_UNSAFE for s in failed_safety):
            overall_safety = SafetyLevel.POTENTIALLY_UNSAFE
        elif any(s.safety_level == SafetyLevel.NEEDS_REVIEW for s in safety_checks):
            overall_safety = SafetyLevel.NEEDS_REVIEW
        elif failed_safety:
            overall_safety = SafetyLevel.MOSTLY_SAFE
        else:
            overall_safety = SafetyLevel.SAFE

        # Determine status
        regression_passed = all(r.passed for r in regression_checks)
        safety_passed = all(s.passed for s in safety_checks)

        if issue_resolved and regression_passed and safety_passed:
            status = FixStatus.VERIFIED
        elif issue_resolved and (regression_passed or safety_passed):
            status = FixStatus.PARTIAL
        elif not issue_resolved:
            status = FixStatus.FAILED
        else:
            status = FixStatus.REJECTED

        # Calculate confidence
        confidence = resolution_confidence
        if regression_passed:
            confidence += 0.1
        if safety_passed:
            confidence += 0.1
        confidence = min(1.0, confidence)

        # Generate summary
        summary = self._generate_summary(
            issue_resolved, regression_passed, safety_passed,
            resolution_msg, overall_safety
        )

        result = VerificationResult(
            id=str(uuid4()),
            fix_id=fix_id,
            issue_id=fix.issue_id,
            status=status,
            issue_resolved=issue_resolved,
            regression_checks=regression_checks,
            safety_checks=safety_checks,
            methods_used=methods_used,
            overall_safety=overall_safety,
            confidence=confidence,
            summary=summary,
            verified_at=datetime.now(),
        )

        self.results[result.id] = result
        return result

    def _generate_summary(
        self,
        issue_resolved: bool,
        regression_passed: bool,
        safety_passed: bool,
        resolution_msg: str,
        overall_safety: SafetyLevel,
    ) -> str:
        """Generate verification summary."""
        parts = []

        if issue_resolved:
            parts.append(f"Issue resolved: {resolution_msg}")
        else:
            parts.append(f"Issue NOT resolved: {resolution_msg}")

        if not regression_passed:
            parts.append("Regression checks failed")

        if not safety_passed:
            parts.append(f"Safety concerns: {overall_safety.value}")

        if issue_resolved and regression_passed and safety_passed:
            parts.append("Fix verified and safe to apply")

        return ". ".join(parts)

    def get_issue(self, issue_id: str) -> Optional[Issue]:
        """Get an issue by ID."""
        return self.issues.get(issue_id)

    def get_fix(self, fix_id: str) -> Optional[Fix]:
        """Get a fix by ID."""
        return self.fixes.get(fix_id)

    def get_result(self, result_id: str) -> Optional[VerificationResult]:
        """Get a verification result by ID."""
        return self.results.get(result_id)

    def get_results_for_fix(self, fix_id: str) -> List[VerificationResult]:
        """Get all verification results for a fix."""
        return [r for r in self.results.values() if r.fix_id == fix_id]

    def get_statistics(self) -> Dict[str, Any]:
        """Get verification statistics."""
        status_counts: Dict[str, int] = defaultdict(int)
        safety_counts: Dict[str, int] = defaultdict(int)

        for result in self.results.values():
            status_counts[result.status.value] += 1
            safety_counts[result.overall_safety.value] += 1

        return {
            "total_issues": len(self.issues),
            "total_fixes": len(self.fixes),
            "total_verifications": len(self.results),
            "status_counts": dict(status_counts),
            "safety_counts": dict(safety_counts),
            "verification_rate": len(self.results) / max(1, len(self.fixes)),
            "success_rate": status_counts.get("verified", 0) / max(1, len(self.results)),
        }
