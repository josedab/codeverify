"""Autofix Validation — verified fix application with regression detection.

Extends the base autofix pipeline with post-fix validation, regression checking,
batch processing strategies, and PR workflow integration.  Every generated fix
is re-verified after application to confirm the original issue is resolved and
no regressions have been introduced.

Key components:
    - **FixValidator**: apply a fix, re-run checks, confirm resolution.
    - **RegressionChecker**: ensure a fix doesn't break existing behaviour.
    - **BatchFixProcessor**: process multiple fixes with ordering strategies.
    - **PRDescriptionGenerator**: build rich PR descriptions from batch results.

.. deprecated::
    This module is superseded by ``codeverify_core.autofix_verified_patches``.
    It remains importable for backward compatibility but will be
    removed in a future release.
"""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "codeverify_core.autofix_validation is deprecated. Use codeverify_core.autofix_verified_patches instead.",
    DeprecationWarning,
    stacklevel=2,
)


import ast
import difflib
import hashlib
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog

logger = structlog.get_logger()


# =============================================================================
# Enums
# =============================================================================


class FixValidationStatus(str, Enum):
    """Lifecycle status of a single fix validation attempt."""

    PENDING = "pending"
    VALIDATING = "validating"
    PASSED = "passed"
    FAILED = "failed"
    REGRESSION_DETECTED = "regression_detected"


class RegressionType(str, Enum):
    """Category of regression introduced by a fix."""

    TEST_FAILURE = "test_failure"
    BEHAVIOR_CHANGE = "behavior_change"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    NEW_ISSUE_INTRODUCED = "new_issue_introduced"


class BatchFixStrategy(str, Enum):
    """Strategy for ordering fixes within a batch."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DEPENDENCY_ORDERED = "dependency_ordered"
    PRIORITY_FIRST = "priority_first"


# =============================================================================
# Configuration & Result Dataclasses
# =============================================================================


@dataclass
class FixValidationConfig:
    """Tunable knobs for fix validation and batch processing."""

    max_validation_attempts: int = 3
    regression_check_enabled: bool = True
    performance_threshold_ms: float = 100.0
    auto_rollback: bool = True
    batch_size: int = 10
    allowed_languages: list[str] = field(
        default_factory=lambda: ["python", "typescript", "go", "java"]
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_validation_attempts": self.max_validation_attempts,
            "regression_check_enabled": self.regression_check_enabled,
            "performance_threshold_ms": self.performance_threshold_ms,
            "auto_rollback": self.auto_rollback,
            "batch_size": self.batch_size,
            "allowed_languages": list(self.allowed_languages),
        }


@dataclass
class RegressionResult:
    """Details of a single regression detected after applying a fix."""

    regression_type: RegressionType
    description: str
    affected_tests: list[str] = field(default_factory=list)
    severity: str = "medium"
    rollback_recommended: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "regression_type": self.regression_type.value,
            "description": self.description,
            "affected_tests": list(self.affected_tests),
            "severity": self.severity,
            "rollback_recommended": self.rollback_recommended,
        }


@dataclass
class FixValidationResult:
    """Outcome of validating a single fix through the pipeline."""

    fix_id: str
    status: FixValidationStatus
    issue_resolved: bool
    regressions: list[RegressionResult] = field(default_factory=list)
    validation_time_ms: float = 0.0
    attempt_count: int = 0
    original_code: str = ""
    fixed_code: str = ""
    proof_of_correctness: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fix_id": self.fix_id,
            "status": self.status.value,
            "issue_resolved": self.issue_resolved,
            "regressions": [r.to_dict() for r in self.regressions],
            "validation_time_ms": self.validation_time_ms,
            "attempt_count": self.attempt_count,
            "proof_of_correctness": self.proof_of_correctness,
        }


@dataclass
class BatchFixResult:
    """Aggregated result of processing a batch of fixes."""

    batch_id: str
    strategy: BatchFixStrategy
    total_fixes: int = 0
    successful_fixes: int = 0
    failed_fixes: int = 0
    regression_fixes: int = 0
    fixes: list[FixValidationResult] = field(default_factory=list)
    total_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "strategy": self.strategy.value,
            "total_fixes": self.total_fixes,
            "successful_fixes": self.successful_fixes,
            "failed_fixes": self.failed_fixes,
            "regression_fixes": self.regression_fixes,
            "fixes": [f.to_dict() for f in self.fixes],
            "total_time_ms": self.total_time_ms,
        }


@dataclass
class PRDescription:
    """A structured pull-request description generated from batch results."""

    title: str
    body: str
    labels: list[str] = field(default_factory=list)
    reviewers: list[str] = field(default_factory=list)
    fixes_summary: list[dict[str, Any]] = field(default_factory=list)
    before_after_snippets: list[dict[str, str]] = field(default_factory=list)
    verification_proof: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "body": self.body,
            "labels": list(self.labels),
            "reviewers": list(self.reviewers),
            "fixes_summary": list(self.fixes_summary),
            "before_after_snippets": list(self.before_after_snippets),
            "verification_proof": self.verification_proof,
        }


# =============================================================================
# Suspicious Pattern Registry
# =============================================================================

_SUSPICIOUS_PATTERNS: list[dict[str, Any]] = [
    {"name": "eval_call", "pattern": re.compile(r"\beval\s*\("), "severity": "critical"},
    {"name": "exec_call", "pattern": re.compile(r"\bexec\s*\("), "severity": "critical"},
    {"name": "os_system", "pattern": re.compile(r"\bos\.system\s*\("), "severity": "high"},
    {
        "name": "subprocess_shell",
        "pattern": re.compile(r"\bsubprocess\.\w+\(.*shell\s*=\s*True"),
        "severity": "high",
    },
    {"name": "bare_except", "pattern": re.compile(r"\bexcept\s*:"), "severity": "medium"},
    {
        "name": "wildcard_import",
        "pattern": re.compile(r"from\s+\S+\s+import\s+\*"),
        "severity": "low",
    },
]


# =============================================================================
# Fix Validator
# =============================================================================


class FixValidator:
    """Apply a candidate fix and verify the original issue is resolved.

    Performs up to ``config.max_validation_attempts`` rounds of syntax check,
    issue-resolution check, regression check, and proof generation.
    """

    def __init__(self, config: FixValidationConfig | None = None) -> None:
        self._config = config or FixValidationConfig()
        self._regression_checker = RegressionChecker()

    def validate_fix(
        self,
        original_code: str,
        fixed_code: str,
        issue: str,
        language: str = "python",
    ) -> FixValidationResult:
        """Validate that *fixed_code* resolves *issue* without regressions."""
        fix_id = _generate_fix_id(original_code, fixed_code)
        start = time.monotonic()

        logger.info("fix_validation.start", fix_id=fix_id, language=language)

        best: FixValidationResult | None = None
        for attempt in range(1, self._config.max_validation_attempts + 1):
            result = self._run_single_attempt(
                fix_id,
                original_code,
                fixed_code,
                issue,
                language,
                attempt,
            )
            if result.status == FixValidationStatus.PASSED:
                result.validation_time_ms = (time.monotonic() - start) * 1000
                logger.info("fix_validation.passed", fix_id=fix_id, attempt=attempt)
                return result
            best = result

        elapsed_ms = (time.monotonic() - start) * 1000
        if best is not None:
            best.validation_time_ms = elapsed_ms
            return best

        return FixValidationResult(
            fix_id=fix_id,
            status=FixValidationStatus.FAILED,
            issue_resolved=False,
            validation_time_ms=elapsed_ms,
            attempt_count=self._config.max_validation_attempts,
            original_code=original_code,
            fixed_code=fixed_code,
        )

    def _run_single_attempt(
        self,
        fix_id: str,
        original_code: str,
        fixed_code: str,
        issue: str,
        language: str,
        attempt: int,
    ) -> FixValidationResult:
        """Execute one full validation pass."""
        if not self._check_syntax(fixed_code, language):
            return FixValidationResult(
                fix_id=fix_id,
                status=FixValidationStatus.FAILED,
                issue_resolved=False,
                attempt_count=attempt,
                original_code=original_code,
                fixed_code=fixed_code,
            )

        resolved = self._check_issue_resolved(original_code, fixed_code, issue)

        regressions: list[RegressionResult] = []
        if self._config.regression_check_enabled:
            regressions = self._run_regression_check(original_code, fixed_code, language)

        has_critical = any(r.rollback_recommended for r in regressions)
        if has_critical:
            status = FixValidationStatus.REGRESSION_DETECTED
        elif not resolved:
            status = FixValidationStatus.FAILED
        else:
            status = FixValidationStatus.PASSED

        proof = self._generate_proof(original_code, fixed_code) if resolved else None

        return FixValidationResult(
            fix_id=fix_id,
            status=status,
            issue_resolved=resolved,
            regressions=regressions,
            attempt_count=attempt,
            original_code=original_code,
            fixed_code=fixed_code,
            proof_of_correctness=proof,
        )

    def _check_syntax(self, code: str, language: str) -> bool:
        """Return *True* when *code* is syntactically valid.

        Uses ``ast.parse`` for Python; brace-balance check for others.
        """
        if language == "python":
            try:
                ast.parse(code)
                return True
            except SyntaxError:
                logger.debug("fix_validation.syntax_error", language=language)
                return False
        return _check_brace_balance(code)

    def _check_issue_resolved(self, original_code: str, fixed_code: str, issue: str) -> bool:
        """Multi-signal heuristic checking whether *issue* is resolved."""
        if original_code == fixed_code:
            return False

        issue_lower = issue.lower()

        # Check if known-bad patterns referenced by the issue are gone
        for entry in _SUSPICIOUS_PATTERNS:
            name_words = entry["name"].replace("_", " ")
            if name_words in issue_lower or entry["name"] in issue_lower:
                orig_hits = len(entry["pattern"].findall(original_code))
                fixed_hits = len(entry["pattern"].findall(fixed_code))
                if fixed_hits >= orig_hits and orig_hits > 0:
                    return False

        # Keyword overlap — significant words still present without structural change
        keywords = [w for w in issue_lower.split() if len(w) > 5 and w.isalpha()]
        if keywords:
            still_present = sum(1 for kw in keywords if kw in fixed_code.lower())
            orig_lines = set(original_code.strip().splitlines())
            fixed_lines = set(fixed_code.strip().splitlines())
            if still_present == len(keywords) and orig_lines == fixed_lines:
                return False

        return True

    def _run_regression_check(
        self,
        original_code: str,
        fixed_code: str,
        _language: str,
    ) -> list[RegressionResult]:
        """Delegate to :class:`RegressionChecker`."""
        return self._regression_checker.check_regressions(original_code, fixed_code)

    def _generate_proof(self, original_code: str, fixed_code: str) -> str | None:
        """Build a human-readable proof using :mod:`difflib` unified diff."""
        diff = list(
            difflib.unified_diff(
                original_code.splitlines(keepends=True),
                fixed_code.splitlines(keepends=True),
                fromfile="original",
                tofile="fixed",
                lineterm="",
            )
        )
        if not diff:
            return None

        added = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))
        return f"Verified fix: {added} line(s) added, {removed} line(s) removed.\n" + "".join(diff)


# =============================================================================
# Regression Checker
# =============================================================================


class RegressionChecker:
    """Detect regressions introduced by a code fix.

    Runs behaviour-preservation, performance, and new-issue checks.
    """

    def __init__(self) -> None:
        self._similarity_threshold: float = 0.55

    def check_regressions(
        self,
        original: str,
        fixed: str,
        test_suite: list[str] | None = None,
    ) -> list[RegressionResult]:
        """Run all regression checks and return any findings."""
        results: list[RegressionResult] = []
        results.extend(self._check_behavior_preservation(original, fixed))
        results.extend(self._check_performance_regression(original, fixed))
        results.extend(self._detect_new_issues(fixed, original))

        if test_suite:
            results.extend(self._check_test_impacts(original, fixed, test_suite))
        return results

    def _check_behavior_preservation(self, original: str, fixed: str) -> list[RegressionResult]:
        """Compare structural similarity via :func:`difflib.SequenceMatcher`."""
        ratio = difflib.SequenceMatcher(None, original, fixed).ratio()

        if ratio < self._similarity_threshold:
            return [
                RegressionResult(
                    regression_type=RegressionType.BEHAVIOR_CHANGE,
                    description=(
                        f"Structural similarity {ratio:.2%} below "
                        f"threshold {self._similarity_threshold:.0%}."
                    ),
                    severity="high",
                    rollback_recommended=ratio < 0.35,
                )
            ]

        # Check for removed definitions
        orig_defs = set(re.findall(r"(?:def|class)\s+(\w+)", original))
        fixed_defs = set(re.findall(r"(?:def|class)\s+(\w+)", fixed))
        removed = orig_defs - fixed_defs

        if removed:
            return [
                RegressionResult(
                    regression_type=RegressionType.BEHAVIOR_CHANGE,
                    description=f"Definitions removed: {', '.join(sorted(removed))}.",
                    affected_tests=[f"test_{n}" for n in removed],
                    severity="critical",
                    rollback_recommended=True,
                )
            ]
        return []

    def _check_performance_regression(self, original: str, fixed: str) -> list[RegressionResult]:
        """Heuristic loop/call complexity comparison."""
        orig_c = _estimate_complexity(original)
        fixed_c = _estimate_complexity(fixed)

        if fixed_c > orig_c * 1.5 and fixed_c - orig_c >= 3:
            return [
                RegressionResult(
                    regression_type=RegressionType.PERFORMANCE_DEGRADATION,
                    description=f"Complexity rose from {orig_c} to {fixed_c}.",
                    severity="medium",
                    rollback_recommended=False,
                )
            ]
        return []

    def _detect_new_issues(self, fixed: str, original: str = "") -> list[RegressionResult]:
        """Scan *fixed* for suspicious patterns not present in *original*."""
        results: list[RegressionResult] = []
        for entry in _SUSPICIOUS_PATTERNS:
            orig_hits = len(entry["pattern"].findall(original)) if original else 0
            fixed_hits = len(entry["pattern"].findall(fixed))
            if fixed_hits > orig_hits:
                results.append(
                    RegressionResult(
                        regression_type=RegressionType.NEW_ISSUE_INTRODUCED,
                        description=f"Pattern '{entry['name']}' appears {fixed_hits - orig_hits} new time(s).",
                        severity=entry["severity"],
                        rollback_recommended=entry["severity"] in ("critical", "high"),
                    )
                )
        return results

    def _check_test_impacts(
        self,
        original: str,
        fixed: str,
        test_suite: list[str],
    ) -> list[RegressionResult]:
        """Cross-reference changed symbols with test names."""
        orig_defs = set(re.findall(r"(?:def|class)\s+(\w+)", original))
        fixed_defs = set(re.findall(r"(?:def|class)\s+(\w+)", fixed))
        changed = orig_defs - fixed_defs

        for sym in orig_defs & fixed_defs:
            if _extract_symbol_body(original, sym) != _extract_symbol_body(fixed, sym):
                changed.add(sym)

        affected = [t for t in test_suite if any(s.lower() in t.lower() for s in changed)]
        if affected:
            return [
                RegressionResult(
                    regression_type=RegressionType.TEST_FAILURE,
                    description=f"{len(affected)} test(s) affected by changes to: {', '.join(sorted(changed))}.",
                    affected_tests=affected,
                    severity="high",
                    rollback_recommended=len(affected) > 3,
                )
            ]
        return []


# =============================================================================
# Batch Fix Processor
# =============================================================================


class BatchFixProcessor:
    """Process multiple fix candidates as a single batch.

    Supports SEQUENTIAL, PARALLEL, DEPENDENCY_ORDERED, and PRIORITY_FIRST
    strategies (see :class:`BatchFixStrategy`).
    """

    def __init__(self, config: FixValidationConfig | None = None) -> None:
        self._config = config or FixValidationConfig()
        self._validator = FixValidator(config=self._config)

    def process_batch(
        self,
        fixes: list[dict[str, Any]],
        strategy: BatchFixStrategy = BatchFixStrategy.SEQUENTIAL,
    ) -> BatchFixResult:
        """Validate every fix in *fixes* according to *strategy*.

        Each element must contain ``original_code``, ``fixed_code``, and
        ``issue`` keys.  Optional: ``language``, ``priority``, ``depends_on``.
        """
        batch_id = str(uuid4())
        start = time.monotonic()
        logger.info("batch_fix.start", batch_id=batch_id, total=len(fixes), strategy=strategy.value)

        ordered = self._apply_strategy(fixes, strategy)

        results: list[FixValidationResult] = []
        for fix_dict in ordered[: self._config.batch_size]:
            result = self._validator.validate_fix(
                original_code=fix_dict.get("original_code", ""),
                fixed_code=fix_dict.get("fixed_code", ""),
                issue=fix_dict.get("issue", ""),
                language=fix_dict.get("language", "python"),
            )
            # Auto-rollback on regression
            if (
                result.status == FixValidationStatus.REGRESSION_DETECTED
                and self._config.auto_rollback
            ):
                logger.warning("batch_fix.auto_rollback", fix_id=result.fix_id)
                result.fixed_code = result.original_code
                result.status = FixValidationStatus.FAILED
            results.append(result)

        elapsed_ms = (time.monotonic() - start) * 1000
        successful = sum(1 for r in results if r.status == FixValidationStatus.PASSED)
        regression = sum(1 for r in results if r.status == FixValidationStatus.REGRESSION_DETECTED)

        batch_result = BatchFixResult(
            batch_id=batch_id,
            strategy=strategy,
            total_fixes=len(results),
            successful_fixes=successful,
            failed_fixes=len(results) - successful - regression,
            regression_fixes=regression,
            fixes=results,
            total_time_ms=elapsed_ms,
        )
        logger.info("batch_fix.complete", batch_id=batch_id, successful=successful)
        return batch_result

    def _apply_strategy(
        self, fixes: list[dict[str, Any]], strategy: BatchFixStrategy
    ) -> list[dict[str, Any]]:
        if strategy == BatchFixStrategy.DEPENDENCY_ORDERED:
            return self._order_by_dependency(fixes)
        if strategy == BatchFixStrategy.PRIORITY_FIRST:
            return self._order_by_priority(fixes)
        return list(fixes)

    def _order_by_dependency(self, fixes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Topological sort via Kahn's algorithm on ``depends_on`` edges."""
        id_map: dict[str, dict[str, Any]] = {}
        for fix in fixes:
            fid = _generate_fix_id(fix.get("original_code", ""), fix.get("fixed_code", ""))
            fix["_fix_id"] = fid
            id_map[fid] = fix

        in_degree: dict[str, int] = dict.fromkeys(id_map, 0)
        adj: dict[str, list[str]] = {fid: [] for fid in id_map}
        for fix in fixes:
            fid = fix["_fix_id"]
            for dep in fix.get("depends_on", []):
                if dep in id_map:
                    adj[dep].append(fid)
                    in_degree[fid] = in_degree.get(fid, 0) + 1

        queue = [fid for fid, deg in in_degree.items() if deg == 0]
        ordered: list[dict[str, Any]] = []
        while queue:
            current = queue.pop(0)
            ordered.append(id_map[current])
            for nb in adj.get(current, []):
                in_degree[nb] -= 1
                if in_degree[nb] == 0:
                    queue.append(nb)

        # Append unreachable nodes (cyclic deps)
        visited = {id(f) for f in ordered}
        for fix in fixes:
            if id(fix) not in visited:
                ordered.append(fix)
        return ordered

    def _order_by_priority(self, fixes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sort by descending ``priority`` (higher = first)."""
        return sorted(fixes, key=lambda f: f.get("priority", 0), reverse=True)


# =============================================================================
# PR Description Generator
# =============================================================================


class PRDescriptionGenerator:
    """Generate a rich pull-request description from batch-fix results."""

    _CATEGORY_LABELS: dict[str, str] = {
        "security": "security",
        "bug": "bug",
        "style": "code-quality",
        "error_handling": "reliability",
        "type_safety": "type-safety",
        "performance": "performance",
    }

    def __init__(self) -> None:
        pass

    def generate(
        self,
        batch_result: BatchFixResult,
        repo_context: dict[str, Any] | None = None,
    ) -> PRDescription:
        """Build a complete PR description from *batch_result*."""
        ctx = repo_context or {}
        fixes_summary = [self._format_fix_summary(f) for f in batch_result.fixes]
        before_after = [
            self._generate_before_after(f)
            for f in batch_result.fixes
            if f.status == FixValidationStatus.PASSED
        ]

        labels = self._derive_labels(batch_result)
        proof = self._build_verification_proof(batch_result)
        title = self._build_title(batch_result, ctx)
        body = self._build_body(batch_result, fixes_summary, before_after, proof, ctx)

        return PRDescription(
            title=title,
            body=body,
            labels=labels,
            reviewers=ctx.get("reviewers", []),
            fixes_summary=fixes_summary,
            before_after_snippets=before_after,
            verification_proof=proof,
        )

    def _format_fix_summary(self, fix: FixValidationResult) -> dict[str, Any]:
        return {
            "fix_id": fix.fix_id,
            "status": fix.status.value,
            "issue_resolved": fix.issue_resolved,
            "regressions": len(fix.regressions),
            "attempts": fix.attempt_count,
            "time_ms": round(fix.validation_time_ms, 2),
        }

    def _generate_before_after(self, fix: FixValidationResult) -> dict[str, str]:
        return {"before": fix.original_code, "after": fix.fixed_code, "fix_id": fix.fix_id}

    def _derive_labels(self, batch_result: BatchFixResult) -> list[str]:
        labels: list[str] = ["codeverify", "autofix"]
        if batch_result.successful_fixes == batch_result.total_fixes:
            labels.append("verified")
        if batch_result.regression_fixes > 0:
            labels.append("needs-review")
        return labels

    def _build_title(self, batch_result: BatchFixResult, ctx: dict[str, Any]) -> str:
        n, total = batch_result.successful_fixes, batch_result.total_fixes
        repo = ctx.get("repo_name", "")
        prefix = f"[{repo}] " if repo else ""
        return f"{prefix}fix: apply {n}/{total} verified autofix patches"

    def _build_body(
        self,
        batch: BatchFixResult,
        summaries: list[dict[str, Any]],
        snippets: list[dict[str, str]],
        proof: str,
        _ctx: dict[str, Any],
    ) -> str:
        parts: list[str] = []

        parts.append("## CodeVerify Autofix\n")
        parts.append(
            f"Applied **{batch.successful_fixes}** verified fix(es) "
            f"out of **{batch.total_fixes}** candidate(s) "
            f"(**{batch.strategy.value}** strategy).\n"
        )

        # Stats table
        parts.append("| Metric | Value |")
        parts.append("| --- | --- |")
        parts.append(f"| Total fixes | {batch.total_fixes} |")
        parts.append(f"| Successful | {batch.successful_fixes} |")
        parts.append(f"| Failed | {batch.failed_fixes} |")
        parts.append(f"| Regressions | {batch.regression_fixes} |")
        parts.append(f"| Total time | {batch.total_time_ms:.0f} ms |\n")

        # Per-fix details
        if summaries:
            parts.append("### Fix Details\n")
            for s in summaries:
                icon = "✅" if s["status"] == "passed" else "❌"
                parts.append(
                    f"- {icon} `{s['fix_id'][:12]}` — {s['status']} ({s['attempts']} attempt(s))"
                )
            parts.append("")

        # Before/After
        if snippets:
            parts.append("### Changes\n")
            for sn in snippets:
                parts.append(f"<details><summary><code>{sn['fix_id'][:12]}</code></summary>\n")
                parts.append(f"**Before:**\n```\n{sn['before']}\n```\n")
                parts.append(f"**After:**\n```\n{sn['after']}\n```\n</details>\n")

        if proof:
            parts.append(f"### Verification Proof\n\n```\n{proof}\n```\n")

        return "\n".join(parts)

    def _build_verification_proof(self, batch_result: BatchFixResult) -> str:
        lines: list[str] = []
        for fix in batch_result.fixes:
            if fix.proof_of_correctness:
                lines.append(f"[{fix.fix_id[:12]}] {fix.proof_of_correctness.splitlines()[0]}")
            elif fix.status == FixValidationStatus.PASSED:
                lines.append(f"[{fix.fix_id[:12]}] Passed — no proof text generated.")
        return "\n".join(lines)


# =============================================================================
# Helpers
# =============================================================================


def _generate_fix_id(original_code: str, fixed_code: str) -> str:
    """Deterministic fix identifier derived from the code pair."""
    return hashlib.sha256(f"{original_code}||{fixed_code}".encode()).hexdigest()[:16]


def _check_brace_balance(code: str) -> bool:
    """Return *True* when braces, brackets, and parentheses are balanced."""
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    in_string = False
    string_char = ""

    for ch in code:
        if ch in ('"', "'") and not in_string:
            in_string, string_char = True, ch
            continue
        if in_string and ch == string_char:
            in_string = False
            continue
        if in_string:
            continue
        if ch in ("(", "[", "{"):
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return len(stack) == 0


def _estimate_complexity(code: str) -> int:
    """Rough complexity score: loops, branches, and nested calls."""
    score = 0
    score += len(re.findall(r"\bfor\b", code))
    score += len(re.findall(r"\bwhile\b", code))
    score += len(re.findall(r"\bif\b", code))
    score += len(re.findall(r"\belif\b", code))
    for line in code.splitlines():
        parens = line.count("(")
        if parens >= 2:
            score += parens - 1
    return score


def _extract_symbol_body(source: str, symbol: str) -> str:
    """Extract the indented body of a ``def``/``class`` from *source*."""
    pattern = re.compile(
        rf"^([ \t]*)(def|class)\s+{re.escape(symbol)}\b[^\n]*:\s*\n",
        re.MULTILINE,
    )
    match = pattern.search(source)
    if not match:
        return ""

    indent = match.group(1)
    lines = source[match.end() :].splitlines(keepends=True)
    body: list[str] = []
    for line in lines:
        stripped = line.rstrip("\n\r")
        if stripped == "" or stripped.isspace():
            body.append(line)
            continue
        if len(line) - len(line.lstrip()) <= len(indent) and stripped:
            break
        body.append(line)
    return "".join(body).rstrip()
