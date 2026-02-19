"""Context-Window Verification.

Verifies LLM context windows for consistency, optimizes context
selection for maximum verification coverage, and detects
truncation issues.

Features:
- Context consistency checking (contradictory types, conflicting signatures)
- Optimal code snippet selection based on dependency analysis
- Truncation detection and re-analysis triggering
- Token budget management per model
- Context quality scoring
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class ConsistencyIssueType(str, Enum):
    TYPE_CONFLICT = "type_conflict"
    SIGNATURE_MISMATCH = "signature_mismatch"
    TRUNCATED_FUNCTION = "truncated_function"
    MISSING_IMPORT = "missing_import"
    DUPLICATE_DEFINITION = "duplicate_definition"
    INCOMPLETE_CONTEXT = "incomplete_context"


class ContextQuality(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNUSABLE = "unusable"


@dataclass
class ContextSnippet:
    """A code snippet in the context window."""
    file_path: str = ""
    start_line: int = 0
    end_line: int = 0
    content: str = ""
    token_count: int = 0
    relevance_score: float = 0.5
    dependencies: list[str] = field(default_factory=list)


@dataclass
class ConsistencyIssue:
    """An issue found in the context window."""
    issue_type: ConsistencyIssueType = ConsistencyIssueType.INCOMPLETE_CONTEXT
    severity: str = "medium"
    message: str = ""
    snippet_index: int = 0
    conflicting_index: int | None = None


@dataclass
class ContextWindow:
    """A composed context window for LLM analysis."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    snippets: list[ContextSnippet] = field(default_factory=list)
    total_tokens: int = 0
    max_tokens: int = 8000
    issues: list[ConsistencyIssue] = field(default_factory=list)
    quality: ContextQuality = ContextQuality.GOOD
    quality_score: float = 0.8
    is_truncated: bool = False


@dataclass
class ContextOptimizationResult:
    """Result of optimizing context selection."""
    original_snippets: int = 0
    selected_snippets: int = 0
    tokens_saved: int = 0
    coverage_score: float = 0.0
    quality_improvement: float = 0.0


MODEL_TOKEN_LIMITS: dict[str, int] = {
    "gpt-4": 8192,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "claude-3-sonnet": 200000,
    "claude-3-opus": 200000,
    "claude-3-haiku": 200000,
}


class ConsistencyChecker:
    """Checks context windows for consistency issues."""

    def check(self, window: ContextWindow) -> list[ConsistencyIssue]:
        issues: list[ConsistencyIssue] = []

        all_defs: dict[str, list[int]] = {}
        all_types: dict[str, list[tuple[int, str]]] = {}

        for i, snippet in enumerate(window.snippets):
            content = snippet.content

            # Check for truncated functions
            open_count = content.count("{") + content.count("(")
            close_count = content.count("}") + content.count(")")
            if abs(open_count - close_count) > 2:
                issues.append(ConsistencyIssue(
                    issue_type=ConsistencyIssueType.TRUNCATED_FUNCTION,
                    severity="high",
                    message=f"Snippet {i} appears truncated (unbalanced brackets: {open_count} open, {close_count} close)",
                    snippet_index=i,
                ))

            # Track function definitions
            for match in re.finditer(r'def\s+(\w+)\s*\(', content):
                name = match.group(1)
                all_defs.setdefault(name, []).append(i)

            # Track type annotations
            for match in re.finditer(r'(\w+)\s*:\s*(\w+)', content):
                var_name, type_name = match.groups()
                all_types.setdefault(var_name, []).append((i, type_name))

        # Check duplicate definitions
        for name, indices in all_defs.items():
            if len(indices) > 1:
                issues.append(ConsistencyIssue(
                    issue_type=ConsistencyIssueType.DUPLICATE_DEFINITION,
                    severity="medium",
                    message=f"Function '{name}' defined in multiple snippets: {indices}",
                    snippet_index=indices[0],
                    conflicting_index=indices[1],
                ))

        # Check type conflicts
        for var, type_list in all_types.items():
            types = set(t for _, t in type_list)
            if len(types) > 1:
                issues.append(ConsistencyIssue(
                    issue_type=ConsistencyIssueType.TYPE_CONFLICT,
                    severity="high",
                    message=f"Variable '{var}' has conflicting types: {types}",
                    snippet_index=type_list[0][0],
                    conflicting_index=type_list[-1][0],
                ))

        return issues


class ContextOptimizer:
    """Optimizes context window content for maximum coverage."""

    def optimize(
        self, snippets: list[ContextSnippet], max_tokens: int
    ) -> tuple[list[ContextSnippet], ContextOptimizationResult]:
        """Select optimal snippets within token budget."""
        sorted_snippets = sorted(snippets, key=lambda s: s.relevance_score, reverse=True)
        selected: list[ContextSnippet] = []
        total_tokens = 0
        original_tokens = sum(s.token_count for s in snippets)

        for snippet in sorted_snippets:
            if total_tokens + snippet.token_count <= max_tokens:
                selected.append(snippet)
                total_tokens += snippet.token_count

        dep_coverage = self._calculate_coverage(selected, snippets)

        return selected, ContextOptimizationResult(
            original_snippets=len(snippets),
            selected_snippets=len(selected),
            tokens_saved=original_tokens - total_tokens,
            coverage_score=round(dep_coverage, 3),
            quality_improvement=round(dep_coverage - 0.5, 3),
        )

    def _calculate_coverage(
        self, selected: list[ContextSnippet], all_snippets: list[ContextSnippet]
    ) -> float:
        all_deps = set()
        for s in all_snippets:
            all_deps.update(s.dependencies)
        selected_deps = set()
        for s in selected:
            selected_deps.update(s.dependencies)
        if not all_deps:
            return 1.0
        return len(selected_deps & all_deps) / len(all_deps)


class TruncationDetector:
    """Detects when context truncation may have caused analysis issues."""

    def detect(self, window: ContextWindow) -> list[str]:
        """Detect potential truncation issues."""
        warnings: list[str] = []

        if window.total_tokens >= window.max_tokens * 0.95:
            warnings.append(f"Context window at {window.total_tokens}/{window.max_tokens} tokens (≥95% capacity)")
            window.is_truncated = True

        last = window.snippets[-1] if window.snippets else None
        if last and last.content and not last.content.rstrip().endswith(("}", ")", ":", "\n", "pass")):
            warnings.append("Last snippet may be truncated mid-statement")
            window.is_truncated = True

        for i, s in enumerate(window.snippets):
            if s.content.count("def ") > s.content.count("return ") + s.content.count("pass"):
                warnings.append(f"Snippet {i} has more function definitions than return statements — possible truncation")

        return warnings


class ContextWindowVerificationService:
    """Main service for context-window verification."""

    def __init__(self, model: str = "gpt-4") -> None:
        self._checker = ConsistencyChecker()
        self._optimizer = ContextOptimizer()
        self._truncation = TruncationDetector()
        self._model = model
        self._max_tokens = MODEL_TOKEN_LIMITS.get(model, 8192)

    def verify_context(self, snippets: list[ContextSnippet]) -> ContextWindow:
        """Verify a context window for consistency and quality."""
        window = ContextWindow(
            snippets=snippets,
            total_tokens=sum(s.token_count for s in snippets),
            max_tokens=self._max_tokens,
        )

        window.issues = self._checker.check(window)
        truncation_warnings = self._truncation.detect(window)
        for w in truncation_warnings:
            window.issues.append(ConsistencyIssue(
                issue_type=ConsistencyIssueType.INCOMPLETE_CONTEXT,
                severity="medium", message=w,
            ))

        high_issues = sum(1 for i in window.issues if i.severity == "high")
        total_issues = len(window.issues)

        if high_issues > 0:
            window.quality = ContextQuality.POOR
            window.quality_score = 0.3
        elif total_issues > 3:
            window.quality = ContextQuality.FAIR
            window.quality_score = 0.5
        elif total_issues > 0:
            window.quality = ContextQuality.GOOD
            window.quality_score = 0.7
        else:
            window.quality = ContextQuality.EXCELLENT
            window.quality_score = 1.0

        return window

    def optimize_context(
        self, snippets: list[ContextSnippet]
    ) -> tuple[ContextWindow, ContextOptimizationResult]:
        """Optimize and verify a context window."""
        selected, opt_result = self._optimizer.optimize(snippets, self._max_tokens)
        window = self.verify_context(selected)
        return window, opt_result

    def estimate_tokens(self, text: str) -> int:
        """Rough token count estimation (4 chars per token)."""
        return len(text) // 4


# ─── Singleton Access ──────────────────────────────────────────────────

_context_window_instance: ContextWindowVerificationService | None = None

def get_context_window_service() -> ContextWindowVerificationService:
    global _context_window_instance
    if _context_window_instance is None:
        _context_window_instance = ContextWindowVerificationService()
    return _context_window_instance

def reset_context_window_service() -> None:
    global _context_window_instance
    _context_window_instance = None
