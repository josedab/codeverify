"""Verification-as-Code SDK.

Provides a developer-friendly API for writing custom verification rules as code,
composing them into pipelines, and sharing via a registry.

Usage:
    from codeverify_core.verification_sdk import (
        verification_rule, check, VerificationPipeline, RuleRegistry,
    )

    @verification_rule(id="no-eval", severity="critical", languages=["python"])
    def no_eval(ctx: RuleContext) -> RuleResult:
        if "eval(" in ctx.code:
            return ctx.fail("eval() is forbidden", line=ctx.find_line("eval("))
        return ctx.pass_rule()

    @check(category="security")
    def sql_injection(ctx: RuleContext) -> RuleResult:
        # Pattern-based check
        ...

    pipeline = VerificationPipeline("security-checks")
    pipeline.add(no_eval)
    pipeline.add(sql_injection)
    results = pipeline.run(code, language="python")
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

# =============================================================================
# Core Types
# =============================================================================


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class RuleContext:
    """Context passed to each verification rule."""

    code: str
    file_path: str = ""
    language: str = "python"
    diff: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def find_line(self, pattern: str) -> int:
        """Find the line number of the first occurrence of pattern."""
        for i, line in enumerate(self.code.splitlines(), 1):
            if pattern in line:
                return i
        return 0

    def find_all_lines(self, pattern: str) -> list[int]:
        """Find all line numbers matching pattern."""
        return [i for i, line in enumerate(self.code.splitlines(), 1) if pattern in line]

    def has_pattern(self, regex: str) -> bool:
        """Check if code matches a regex pattern."""
        return bool(re.search(regex, self.code))

    def fail(
        self,
        message: str,
        line: int = 0,
        fix: str | None = None,
    ) -> RuleResult:
        """Create a failing rule result."""
        return RuleResult(
            passed=False,
            message=message,
            line=line,
            fix_suggestion=fix,
        )

    def pass_rule(self, message: str = "") -> RuleResult:
        """Create a passing rule result."""
        return RuleResult(passed=True, message=message)


@dataclass
class RuleResult:
    """Result of evaluating a single rule."""

    passed: bool
    message: str = ""
    line: int = 0
    fix_suggestion: str | None = None


@dataclass
class RuleMetadata:
    """Metadata attached to a rule via decorators."""

    id: str
    name: str = ""
    description: str = ""
    severity: Severity = Severity.MEDIUM
    category: str = "general"
    languages: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    author: str = ""
    version: str = "1.0.0"


class VerificationRuleFunc(Protocol):
    """Protocol for verification rule functions."""

    __rule_metadata__: RuleMetadata

    def __call__(self, ctx: RuleContext) -> RuleResult: ...


# =============================================================================
# Decorators
# =============================================================================


def verification_rule(
    id: str,
    severity: str = "medium",
    languages: list[str] | None = None,
    category: str = "general",
    tags: list[str] | None = None,
    description: str = "",
    author: str = "",
) -> Callable:
    """Decorator to mark a function as a verification rule.

    Usage:
        @verification_rule(id="no-eval", severity="critical", languages=["python"])
        def no_eval(ctx: RuleContext) -> RuleResult:
            ...
    """

    def decorator(func: Callable[[RuleContext], RuleResult]) -> Callable:
        func.__rule_metadata__ = RuleMetadata(  # type: ignore[attr-defined]
            id=id,
            name=func.__name__,
            description=description or func.__doc__ or "",
            severity=Severity(severity),
            category=category,
            languages=languages or [],
            tags=tags or [],
            author=author,
        )
        return func

    return decorator


def check(
    category: str = "general",
    severity: str = "medium",
    languages: list[str] | None = None,
) -> Callable:
    """Simplified decorator for quick checks.

    Usage:
        @check(category="security")
        def no_hardcoded_secrets(ctx: RuleContext) -> RuleResult:
            ...
    """

    def decorator(func: Callable[[RuleContext], RuleResult]) -> Callable:
        func.__rule_metadata__ = RuleMetadata(  # type: ignore[attr-defined]
            id=func.__name__,
            name=func.__name__,
            description=func.__doc__ or "",
            severity=Severity(severity),
            category=category,
            languages=languages or [],
        )
        return func

    return decorator


# =============================================================================
# Pipeline
# =============================================================================


@dataclass
class PipelineResult:
    """Aggregated results from a pipeline run."""

    pipeline_name: str
    rules_run: int = 0
    rules_passed: int = 0
    rules_failed: int = 0
    rules_skipped: int = 0
    findings: list[dict[str, Any]] = field(default_factory=list)
    execution_time_ms: float = 0.0

    @property
    def passed(self) -> bool:
        return self.rules_failed == 0

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.pipeline_name}: "
            f"{self.rules_passed} passed, {self.rules_failed} failed, "
            f"{self.rules_skipped} skipped ({self.execution_time_ms:.0f}ms)"
        )


class VerificationPipeline:
    """Composes multiple verification rules into an executable pipeline.

    Usage:
        pipeline = VerificationPipeline("security")
        pipeline.add(no_eval)
        pipeline.add(sql_injection)
        result = pipeline.run(code, language="python")
    """

    def __init__(self, name: str, fail_fast: bool = False) -> None:
        self._name = name
        self._rules: list[Callable] = []
        self._fail_fast = fail_fast

    @property
    def name(self) -> str:
        return self._name

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    def add(self, rule: Callable) -> VerificationPipeline:
        """Add a rule to the pipeline. Returns self for chaining."""
        self._rules.append(rule)
        return self

    def run(
        self,
        code: str,
        language: str = "python",
        file_path: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Execute all rules in the pipeline against the given code."""
        ctx = RuleContext(
            code=code,
            file_path=file_path,
            language=language,
            metadata=metadata or {},
        )

        start = time.time()
        result = PipelineResult(pipeline_name=self._name)

        for rule in self._rules:
            meta: RuleMetadata | None = getattr(rule, "__rule_metadata__", None)

            # Skip if rule doesn't apply to this language
            if meta and meta.languages and language not in meta.languages:
                result.rules_skipped += 1
                continue

            try:
                rule_result = rule(ctx)
                result.rules_run += 1

                if rule_result.passed:
                    result.rules_passed += 1
                else:
                    result.rules_failed += 1
                    result.findings.append(
                        {
                            "rule_id": meta.id if meta else rule.__name__,
                            "severity": meta.severity.value if meta else "medium",
                            "category": meta.category if meta else "general",
                            "message": rule_result.message,
                            "line": rule_result.line,
                            "fix_suggestion": rule_result.fix_suggestion,
                            "file_path": file_path,
                        }
                    )

                    if self._fail_fast:
                        break

            except Exception as e:
                result.rules_run += 1
                result.rules_failed += 1
                result.findings.append(
                    {
                        "rule_id": meta.id if meta else rule.__name__,
                        "severity": "high",
                        "category": "internal_error",
                        "message": f"Rule raised exception: {e}",
                        "line": 0,
                    }
                )

        result.execution_time_ms = (time.time() - start) * 1000
        return result

    def compose(self, other: VerificationPipeline) -> VerificationPipeline:
        """Compose two pipelines together into a new pipeline."""
        combined = VerificationPipeline(f"{self._name}+{other._name}")
        combined._rules = self._rules + other._rules
        return combined


# =============================================================================
# Rule Registry
# =============================================================================


class RuleRegistry:
    """Registry for sharing and discovering verification rules.

    Usage:
        registry = RuleRegistry()
        registry.register(no_eval)
        registry.register(sql_injection)

        # Get all security rules
        security_rules = registry.get_by_category("security")

        # Build pipeline from registry
        pipeline = registry.to_pipeline("my-checks", categories=["security"])
    """

    def __init__(self) -> None:
        self._rules: dict[str, Callable] = {}

    def register(self, rule: Callable) -> None:
        """Register a rule in the registry."""
        meta: RuleMetadata | None = getattr(rule, "__rule_metadata__", None)
        if meta is None:
            raise ValueError(
                f"Function {rule.__name__} must be decorated with @verification_rule or @check"
            )
        self._rules[meta.id] = rule

    def get(self, rule_id: str) -> Callable | None:
        """Get a rule by ID."""
        return self._rules.get(rule_id)

    def list_rules(self) -> list[dict[str, Any]]:
        """List all registered rules with their metadata."""
        result = []
        for rule_id, rule in self._rules.items():
            meta: RuleMetadata = rule.__rule_metadata__
            result.append(
                {
                    "id": meta.id,
                    "name": meta.name,
                    "severity": meta.severity.value,
                    "category": meta.category,
                    "languages": meta.languages,
                    "tags": meta.tags,
                    "description": meta.description,
                }
            )
        return result

    def get_by_category(self, category: str) -> list[Callable]:
        """Get all rules in a category."""
        return [
            rule for rule in self._rules.values() if rule.__rule_metadata__.category == category
        ]

    def get_by_language(self, language: str) -> list[Callable]:
        """Get all rules applicable to a language."""
        return [
            rule
            for rule in self._rules.values()
            if not rule.__rule_metadata__.languages or language in rule.__rule_metadata__.languages
        ]

    def to_pipeline(
        self,
        name: str,
        categories: list[str] | None = None,
        languages: list[str] | None = None,
        severity_min: str = "info",
    ) -> VerificationPipeline:
        """Build a pipeline from registered rules matching filters."""
        severity_order = ["info", "low", "medium", "high", "critical"]
        min_idx = severity_order.index(severity_min) if severity_min in severity_order else 0

        pipeline = VerificationPipeline(name)
        for rule in self._rules.values():
            meta: RuleMetadata = rule.__rule_metadata__

            if categories and meta.category not in categories:
                continue
            if languages and meta.languages and not set(languages) & set(meta.languages):
                continue

            rule_idx = (
                severity_order.index(meta.severity.value)
                if meta.severity.value in severity_order
                else 0
            )
            if rule_idx < min_idx:
                continue

            pipeline.add(rule)

        return pipeline

    @property
    def count(self) -> int:
        return len(self._rules)


# Module-level default registry
_default_registry: RuleRegistry | None = None


def get_rule_registry() -> RuleRegistry:
    """Get the global rule registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = RuleRegistry()
    return _default_registry


def reset_rule_registry() -> None:
    """Reset the global rule registry (for testing)."""
    global _default_registry
    _default_registry = None
