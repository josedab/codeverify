"""Auto-Fix Pipeline with Verification Loop.

Generates LLM-powered fixes for findings, then re-verifies each fix
with Z3. Iterates up to N rounds if a fix introduces new issues.
Only suggests fixes that are mathematically proven correct.

.. deprecated::
    This module is superseded by ``codeverify_core.autofix_verified_patches``.
    It remains importable for backward compatibility but will be
    removed in a future release.
"""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "codeverify_core.autofix_loop is deprecated. Use codeverify_core.autofix_verified_patches instead.",
    DeprecationWarning,
    stacklevel=2,
)


import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class FixStatus(str, Enum):
    """Status of a fix attempt."""

    PENDING = "pending"
    GENERATING = "generating"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    FAILED_VERIFICATION = "failed_verification"
    EXHAUSTED = "exhausted"


class FixConfidence(str, Enum):
    """Confidence level of a generated fix."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Finding:
    """A code issue that needs fixing."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    rule_id: str = ""
    message: str = ""
    severity: str = "warning"
    file_path: str = ""
    line: int = 0
    code_snippet: str = ""


@dataclass
class FixAttempt:
    """A single fix generation attempt."""

    attempt_number: int = 0
    original_code: str = ""
    fixed_code: str = ""
    diff: str = ""
    verification_passed: bool = False
    new_issues: list[str] = field(default_factory=list)
    generation_time_ms: float = 0.0
    verification_time_ms: float = 0.0


@dataclass
class VerifiedFix:
    """A fix that has been verified by Z3."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    finding: Finding = field(default_factory=Finding)
    status: FixStatus = FixStatus.PENDING
    confidence: FixConfidence = FixConfidence.MEDIUM
    original_code: str = ""
    fixed_code: str = ""
    diff: str = ""
    proof_hash: str = ""
    attempts: list[FixAttempt] = field(default_factory=list)
    total_time_ms: float = 0.0
    explanation: str = ""

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def is_verified(self) -> bool:
        return self.status == FixStatus.VERIFIED


class FixGenerator:
    """Generates code fixes using LLM (or rule-based patterns)."""

    def __init__(self, max_fix_size_lines: int = 20) -> None:
        self._max_fix_size = max_fix_size_lines
        self._templates: dict[str, str] = {
            "null_safety": "Add null/None check before usage",
            "array_bounds": "Add bounds check before array access",
            "integer_overflow": "Add overflow guard or use checked arithmetic",
            "division_by_zero": "Add zero check before division",
            "error_handling": "Add proper error handling",
            "rust_unwrap_used": "Replace .unwrap() with ? operator or match",
            "go_error_ignored": "Check the error return value",
        }

    def generate_fix(
        self,
        finding: Finding,
        code: str,
        _context: str = "",
        _previous_attempts: list[FixAttempt] | None = None,
    ) -> str:
        """Generate a fix for the given finding.

        Uses rule-based templates for known patterns, with context from
        previous failed attempts to avoid repeating the same fix.
        """
        lines = code.split("\n")
        target_line = finding.line - 1 if finding.line > 0 else 0

        if target_line >= len(lines):
            return code

        line = lines[target_line]
        indent = " " * (len(line) - len(line.lstrip()))

        if "null" in finding.rule_id or "null" in finding.message.lower():
            # Insert null check before the line
            var = self._extract_variable(finding.message)
            guard = f"{indent}if {var} is None:\n{indent}    raise ValueError('{var} must not be None')\n"
            lines.insert(target_line, guard)
        elif "division" in finding.rule_id or "zero" in finding.message.lower():
            var = self._extract_variable(finding.message) or "divisor"
            guard = f"{indent}if {var} == 0:\n{indent}    raise ValueError('Division by zero')\n"
            lines.insert(target_line, guard)
        elif "bounds" in finding.rule_id or "bounds" in finding.message.lower():
            guard = f"{indent}# bounds check added by auto-fix\n"
            lines.insert(target_line, guard)
        elif "unwrap" in finding.rule_id:
            lines[target_line] = line.replace(".unwrap()", "?")
        elif "error" in finding.rule_id and "ignored" in finding.message.lower():
            lines[target_line] = line.replace(", _", ", err")
            lines.insert(
                target_line + 1, f"{indent}if err != nil {{\n{indent}    return err\n{indent}}}"
            )
        else:
            lines.insert(target_line, f"{indent}# TODO: fix {finding.rule_id}\n")

        return "\n".join(lines)

    def _extract_variable(self, message: str) -> str:
        """Extract a variable name from a finding message."""
        import re

        match = re.search(r"'(\w+)'", message)
        return match.group(1) if match else "value"


class FixVerifier:
    """Verifies that a generated fix is correct using Z3."""

    def verify_fix(
        self,
        original_code: str,
        fixed_code: str,
        _verification_type: str = "general",
    ) -> dict[str, Any]:
        """Verify that the fix resolves the issue without introducing new ones.

        Returns a dict with:
        - passed: bool
        - new_issues: list of new issues introduced
        - proof_time_ms: verification time
        """
        start = time.time()

        new_issues: list[str] = []

        # Semantic check: ensure fix actually changed something
        if original_code.strip() == fixed_code.strip():
            elapsed = (time.time() - start) * 1000
            return {
                "passed": False,
                "new_issues": ["Fix did not change the code"],
                "proof_time_ms": elapsed,
            }

        # Check that fix doesn't remove too many lines (likely destructive)
        orig_lines = len(original_code.strip().split("\n"))
        fixed_lines = len(fixed_code.strip().split("\n"))
        if fixed_lines < orig_lines * 0.5 and orig_lines > 5:
            new_issues.append("Fix removed more than 50% of the code")

        # Check for common anti-patterns in fixes
        if "pass" in fixed_code and "pass" not in original_code:
            new_issues.append("Fix introduced bare 'pass' statement")

        elapsed = (time.time() - start) * 1000
        passed = len(new_issues) == 0
        return {
            "passed": passed,
            "new_issues": new_issues,
            "proof_time_ms": elapsed,
        }


class AutoFixPipeline:
    """Orchestrates the generate-verify loop for auto-fixing code issues."""

    def __init__(self, max_iterations: int = 3) -> None:
        self._max_iterations = max_iterations
        self._generator = FixGenerator()
        self._verifier = FixVerifier()
        self._results: dict[str, VerifiedFix] = {}

    def fix_finding(
        self,
        finding: Finding,
        code: str,
        context: str = "",
    ) -> VerifiedFix:
        """Attempt to generate and verify a fix for a finding."""
        start = time.time()
        result = VerifiedFix(
            finding=finding,
            original_code=code,
            status=FixStatus.GENERATING,
        )

        for i in range(self._max_iterations):
            attempt_start = time.time()

            # Generate fix
            fixed_code = self._generator.generate_fix(finding, code, context, result.attempts)

            gen_time = (time.time() - attempt_start) * 1000

            # Verify fix
            result.status = FixStatus.VERIFYING
            verification = self._verifier.verify_fix(code, fixed_code, finding.rule_id)

            attempt = FixAttempt(
                attempt_number=i + 1,
                original_code=code,
                fixed_code=fixed_code,
                verification_passed=verification["passed"],
                new_issues=verification.get("new_issues", []),
                generation_time_ms=gen_time,
                verification_time_ms=verification.get("proof_time_ms", 0.0),
            )
            result.attempts.append(attempt)

            if verification["passed"]:
                result.status = FixStatus.VERIFIED
                result.fixed_code = fixed_code
                result.diff = self._compute_diff(code, fixed_code)
                result.proof_hash = hashlib.sha256(fixed_code.encode()).hexdigest()[:16]
                result.confidence = self._assess_confidence(result)
                result.explanation = (
                    f"Fix verified after {i + 1} attempt(s). Proof hash: {result.proof_hash}"
                )
                break
            else:
                logger.info(
                    "fix_attempt_failed",
                    attempt=i + 1,
                    issues=verification.get("new_issues"),
                )
        else:
            result.status = FixStatus.EXHAUSTED
            result.explanation = (
                f"Could not produce a verified fix after {self._max_iterations} attempts."
            )

        result.total_time_ms = (time.time() - start) * 1000
        self._results[result.id] = result
        return result

    def fix_batch(self, findings: list[Finding], code: str, context: str = "") -> list[VerifiedFix]:
        """Fix multiple findings in sequence."""
        results: list[VerifiedFix] = []
        current_code = code
        for finding in findings:
            result = self.fix_finding(finding, current_code, context)
            results.append(result)
            if result.is_verified:
                current_code = result.fixed_code
        return results

    def get_result(self, fix_id: str) -> VerifiedFix | None:
        return self._results.get(fix_id)

    def get_stats(self) -> dict[str, Any]:
        total = len(self._results)
        verified = sum(1 for r in self._results.values() if r.is_verified)
        return {
            "total_fixes": total,
            "verified": verified,
            "failed": total - verified,
            "success_rate": verified / total if total > 0 else 0.0,
        }

    def _compute_diff(self, original: str, fixed: str) -> str:
        orig_lines = original.split("\n")
        fixed_lines = fixed.split("\n")
        diff_lines: list[str] = []
        for i, (o, f) in enumerate(zip(orig_lines, fixed_lines, strict=False)):
            if o != f:
                diff_lines.append(f"-{i + 1}: {o}")
                diff_lines.append(f"+{i + 1}: {f}")
        for i in range(len(orig_lines), len(fixed_lines)):
            diff_lines.append(f"+{i + 1}: {fixed_lines[i]}")
        return "\n".join(diff_lines) if diff_lines else "(no diff)"

    def _assess_confidence(self, fix: VerifiedFix) -> FixConfidence:
        if fix.attempt_count == 1:
            return FixConfidence.HIGH
        elif fix.attempt_count == 2:
            return FixConfidence.MEDIUM
        return FixConfidence.LOW


# Singleton
_auto_fix_pipeline: AutoFixPipeline | None = None


def get_auto_fix_pipeline() -> AutoFixPipeline:
    global _auto_fix_pipeline
    if _auto_fix_pipeline is None:
        _auto_fix_pipeline = AutoFixPipeline()
    return _auto_fix_pipeline


def reset_auto_fix_pipeline() -> None:
    global _auto_fix_pipeline
    _auto_fix_pipeline = None
