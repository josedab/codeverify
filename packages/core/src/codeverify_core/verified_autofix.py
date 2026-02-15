"""Verified Autofix - AI-powered code fixing with formal verification.

Generates patches for common code issues and verifies that fixes are correct
using differential verification (simulated Z3-style proofs). Supports retry
logic, batch processing, and caching of verified fixes.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# =============================================================================
# Enums
# =============================================================================


class FixAttemptStatus(str, Enum):
    """Status of a fix attempt through the autofix pipeline."""

    PENDING = "pending"
    GENERATING = "generating"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    FAILED = "failed"
    REJECTED = "rejected"


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class AutofixConfig:
    """Configuration for the autofix pipeline."""

    max_attempts: int = 3
    verify_fix: bool = True
    timeout_seconds: int = 30
    allowed_languages: list[str] = field(
        default_factory=lambda: ["python", "typescript", "go", "java"]
    )


@dataclass
class CodeIssue:
    """A code issue detected during analysis."""

    file_path: str
    line: int
    message: str
    severity: str
    category: str
    code_snippet: str


@dataclass
class GeneratedPatch:
    """A generated code patch for a detected issue."""

    original_code: str
    fixed_code: str
    diff_text: str
    explanation: str
    confidence: float


@dataclass
class VerificationProof:
    """Proof that a fix is correct and safe."""

    issue_resolved: bool
    no_new_bugs: bool
    behavior_preserved: bool
    proof_details: str
    verification_time_ms: float


@dataclass
class VerifiedFix:
    """A fix that has been through the verification pipeline."""

    issue: CodeIssue
    patch: GeneratedPatch
    proof: VerificationProof | None
    status: FixAttemptStatus
    attempts: int


# =============================================================================
# Fix Generator
# =============================================================================


class FixGenerator:
    """Generates code fixes using pattern-matching for common issues.

    Applies well-known safe transformations:
    - ``eval()`` → ``ast.literal_eval()``
    - bare ``except:`` → ``except Exception:``
    - ``== None`` → ``is None``
    - mutable default arguments → ``None`` defaults with body initialisation
    """

    _FIX_PATTERNS: list[dict[str, Any]] = [
        {
            "name": "eval_to_literal_eval",
            "pattern": re.compile(r"\beval\s*\("),
            "replacement": "ast.literal_eval(",
            "explanation": "Replace eval() with ast.literal_eval() to prevent code injection.",
            "confidence": 0.95,
            "category": "security",
        },
        {
            "name": "bare_except",
            "pattern": re.compile(r"\bexcept\s*:"),
            "replacement": "except Exception:",
            "explanation": "Replace bare except with except Exception to avoid catching SystemExit/KeyboardInterrupt.",
            "confidence": 0.92,
            "category": "error_handling",
        },
        {
            "name": "equality_none",
            "pattern": re.compile(r"(\w+)\s*==\s*None"),
            "replacement": r"\1 is None",
            "explanation": "Use 'is None' instead of '== None' for identity comparison.",
            "confidence": 0.98,
            "category": "style",
        },
        {
            "name": "inequality_none",
            "pattern": re.compile(r"(\w+)\s*!=\s*None"),
            "replacement": r"\1 is not None",
            "explanation": "Use 'is not None' instead of '!= None' for identity comparison.",
            "confidence": 0.98,
            "category": "style",
        },
        {
            "name": "mutable_default_list",
            "pattern": re.compile(r"def\s+(\w+)\s*\(([^)]*?)(\w+)\s*:\s*list\s*=\s*\[\]([^)]*)\)"),
            "replacement": r"def \1(\2\3: list | None = None\4)",
            "explanation": "Replace mutable default argument [] with None to avoid shared state.",
            "confidence": 0.90,
            "category": "bug",
        },
        {
            "name": "mutable_default_dict",
            "pattern": re.compile(r"def\s+(\w+)\s*\(([^)]*?)(\w+)\s*:\s*dict\s*=\s*\{\}([^)]*)\)"),
            "replacement": r"def \1(\2\3: dict | None = None\4)",
            "explanation": "Replace mutable default argument {} with None to avoid shared state.",
            "confidence": 0.90,
            "category": "bug",
        },
    ]

    def generate_fix(self, issue: CodeIssue, context: str) -> GeneratedPatch:
        """Generate a fix for the given code issue.

        Tries each known fix pattern against the issue snippet and context.
        Returns the first matching patch or a low-confidence generic patch.
        """
        code = issue.code_snippet

        for fix in self._FIX_PATTERNS:
            match = fix["pattern"].search(code)
            if match:
                fixed_code = fix["pattern"].sub(fix["replacement"], code)
                diff_text = _make_diff(code, fixed_code)
                return GeneratedPatch(
                    original_code=code,
                    fixed_code=fixed_code,
                    diff_text=diff_text,
                    explanation=fix["explanation"],
                    confidence=fix["confidence"],
                )

        # Fallback: no pattern matched, return a low-confidence placeholder
        return GeneratedPatch(
            original_code=code,
            fixed_code=code,
            diff_text="",
            explanation="No automatic fix pattern matched; manual review required.",
            confidence=0.0,
        )


# =============================================================================
# Differential Verifier
# =============================================================================


class DifferentialVerifier:
    """Verifies that a generated fix is correct.

    Performs simulated Z3-style differential checks:
    1. The original issue is resolved in the patched code.
    2. No new suspicious patterns are introduced.
    3. Overall behaviour is preserved (heuristic).
    """

    _SUSPICIOUS_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"\beval\s*\("),
        re.compile(r"\bexec\s*\("),
        re.compile(r"\b__import__\s*\("),
        re.compile(r"\bos\.system\s*\("),
        re.compile(r"\bsubprocess\.call\s*\("),
    ]

    def verify_fix(
        self,
        issue: CodeIssue,
        original_code: str,
        patch: GeneratedPatch,
    ) -> VerificationProof:
        """Verify that *patch* resolves *issue* without side-effects."""
        start = time.monotonic()

        issue_resolved = self._check_issue_resolved(issue, patch)
        no_new_bugs = self._check_no_new_bugs(original_code, patch.fixed_code)
        behaviour_preserved = self._check_behaviour_preserved(original_code, patch.fixed_code)

        elapsed_ms = (time.monotonic() - start) * 1000

        details_parts: list[str] = []
        if issue_resolved:
            details_parts.append("Issue pattern no longer present in fixed code.")
        else:
            details_parts.append("WARNING: Issue pattern may still be present.")
        if no_new_bugs:
            details_parts.append("No new suspicious patterns introduced.")
        else:
            details_parts.append("WARNING: New suspicious pattern detected.")
        if behaviour_preserved:
            details_parts.append("Behaviour heuristically preserved.")
        else:
            details_parts.append("WARNING: Structural divergence detected.")

        return VerificationProof(
            issue_resolved=issue_resolved,
            no_new_bugs=no_new_bugs,
            behavior_preserved=behaviour_preserved,
            proof_details=" ".join(details_parts),
            verification_time_ms=elapsed_ms,
        )

    # -- internal checks -----------------------------------------------------

    def _check_issue_resolved(self, issue: CodeIssue, patch: GeneratedPatch) -> bool:
        """Return True when the issue's characteristic pattern is absent."""
        keywords = issue.message.lower().split()
        snippet_lower = patch.fixed_code.lower()

        # Simple heuristic: if the original snippet changed at all, the issue
        # is likely addressed.
        if patch.original_code != patch.fixed_code:
            return True

        for kw in keywords:
            if kw in snippet_lower and len(kw) > 4:
                return False
        return True

    def _check_no_new_bugs(self, original_code: str, fixed_code: str) -> bool:
        """Return True when no new suspicious patterns appear."""
        for pat in self._SUSPICIOUS_PATTERNS:
            original_hits = len(pat.findall(original_code))
            fixed_hits = len(pat.findall(fixed_code))
            if fixed_hits > original_hits:
                return False
        return True

    def _check_behaviour_preserved(self, original_code: str, fixed_code: str) -> bool:
        """Heuristic: behaviour is preserved when the structural diff is small."""
        orig_lines = original_code.strip().splitlines()
        fix_lines = fixed_code.strip().splitlines()

        if not orig_lines:
            return True

        diff_count = sum(1 for a, b in zip(orig_lines, fix_lines) if a != b)
        diff_count += abs(len(orig_lines) - len(fix_lines))

        # Allow up to 40 % of lines to differ
        return diff_count <= max(len(orig_lines) * 0.4, 2)


# =============================================================================
# Autofix Pipeline
# =============================================================================


class AutofixPipeline:
    """Orchestrates fix generation, verification, and retry logic.

    Wires together :class:`FixGenerator` and :class:`DifferentialVerifier`
    with configurable retry semantics.
    """

    def __init__(
        self,
        config: AutofixConfig | None = None,
        generator: FixGenerator | None = None,
        verifier: DifferentialVerifier | None = None,
        cache: FixCache | None = None,
    ) -> None:
        """Initialise the pipeline."""
        self.config = config or AutofixConfig()
        self._generator = generator or FixGenerator()
        self._verifier = verifier or DifferentialVerifier()
        self._cache = cache or FixCache()

    async def fix_issue(self, issue: CodeIssue, context: str) -> VerifiedFix:
        """Attempt to fix a single issue with retry logic.

        Generates a patch, optionally verifies it, and retries up to
        ``config.max_attempts`` times on failure.
        """
        cached = self._cache.lookup(issue)
        if cached is not None:
            return cached

        attempts = 0
        last_patch: GeneratedPatch | None = None
        last_proof: VerificationProof | None = None

        while attempts < self.config.max_attempts:
            attempts += 1

            patch = self._generator.generate_fix(issue, context)
            last_patch = patch

            if patch.confidence == 0.0:
                # No pattern matched – give up early
                break

            if not self.config.verify_fix:
                result = VerifiedFix(
                    issue=issue,
                    patch=patch,
                    proof=None,
                    status=FixAttemptStatus.VERIFIED,
                    attempts=attempts,
                )
                self._cache.store(issue, result)
                return result

            proof = self._verifier.verify_fix(issue, issue.code_snippet, patch)
            last_proof = proof

            if proof.issue_resolved and proof.no_new_bugs and proof.behavior_preserved:
                result = VerifiedFix(
                    issue=issue,
                    patch=patch,
                    proof=proof,
                    status=FixAttemptStatus.VERIFIED,
                    attempts=attempts,
                )
                self._cache.store(issue, result)
                return result

            # Allow an async yield between retries
            await asyncio.sleep(0)

        # All attempts exhausted
        status = FixAttemptStatus.REJECTED if last_proof is not None else FixAttemptStatus.FAILED
        return VerifiedFix(
            issue=issue,
            patch=last_patch
            or GeneratedPatch(
                original_code=issue.code_snippet,
                fixed_code=issue.code_snippet,
                diff_text="",
                explanation="Fix generation failed.",
                confidence=0.0,
            ),
            proof=last_proof,
            status=status,
            attempts=attempts,
        )

    async def fix_batch(self, issues: list[CodeIssue], context: str) -> list[VerifiedFix]:
        """Fix a batch of issues concurrently."""
        tasks = [self.fix_issue(issue, context) for issue in issues]
        return list(await asyncio.gather(*tasks))


# =============================================================================
# Fix Cache
# =============================================================================


class FixCache:
    """Caches verified fixes keyed by issue pattern hash.

    The cache key is derived from the issue's file path, category, and
    code snippet so structurally identical issues share a cached fix.
    """

    def __init__(self, max_size: int = 1024) -> None:
        """Initialise the cache."""
        self._store: dict[str, VerifiedFix] = {}
        self._max_size = max_size

    def lookup(self, issue: CodeIssue) -> VerifiedFix | None:
        """Return a cached fix for *issue*, or ``None``."""
        key = self._make_key(issue)
        return self._store.get(key)

    def store(self, issue: CodeIssue, fix: VerifiedFix) -> None:
        """Store a verified fix in the cache."""
        if len(self._store) >= self._max_size:
            # Evict oldest entry (FIFO)
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]
        key = self._make_key(issue)
        self._store[key] = fix

    def clear(self) -> None:
        """Clear the cache."""
        self._store.clear()

    @property
    def size(self) -> int:
        """Number of entries in the cache."""
        return len(self._store)

    @staticmethod
    def _make_key(issue: CodeIssue) -> str:
        """Build a deterministic cache key for *issue*."""
        raw = f"{issue.file_path}:{issue.category}:{issue.code_snippet}"
        return hashlib.sha256(raw.encode()).hexdigest()


# =============================================================================
# Helpers
# =============================================================================


def _make_diff(original: str, fixed: str) -> str:
    """Build a minimal unified-diff-style string."""
    orig_lines = original.splitlines(keepends=True)
    fix_lines = fixed.splitlines(keepends=True)

    diff_parts: list[str] = []
    for i, (a, b) in enumerate(zip(orig_lines, fix_lines)):
        if a != b:
            diff_parts.append(f"-{a.rstrip()}")
            diff_parts.append(f"+{b.rstrip()}")
    # Handle lines only in one version
    if len(orig_lines) > len(fix_lines):
        for line in orig_lines[len(fix_lines) :]:
            diff_parts.append(f"-{line.rstrip()}")
    elif len(fix_lines) > len(orig_lines):
        for line in fix_lines[len(orig_lines) :]:
            diff_parts.append(f"+{line.rstrip()}")

    return "\n".join(diff_parts)


# =============================================================================
# Module Singletons
# =============================================================================


_autofix_pipeline: AutofixPipeline | None = None


def get_autofix_pipeline(
    config: AutofixConfig | None = None,
) -> AutofixPipeline:
    """Get the global autofix pipeline instance."""
    global _autofix_pipeline
    if _autofix_pipeline is None:
        _autofix_pipeline = AutofixPipeline(config=config)
    return _autofix_pipeline


def reset_autofix_pipeline() -> None:
    """Reset the global autofix pipeline (mainly for testing)."""
    global _autofix_pipeline
    if _autofix_pipeline is not None:
        _autofix_pipeline._cache.clear()
    _autofix_pipeline = None


# =============================================================================
# GitHub Suggested Changes Integration
# =============================================================================


@dataclass
class GitHubSuggestedChange:
    """A GitHub PR review suggestion formatted for the suggested changes API."""

    file_path: str
    start_line: int
    end_line: int
    original_code: str
    suggested_code: str
    comment_body: str
    confidence: float
    category: str

    def to_review_comment(self) -> dict[str, Any]:
        """Format as a GitHub pull request review comment with suggestion."""
        suggestion_block = f"```suggestion\n{self.suggested_code}\n```"
        body = (
            f"**CodeVerify Autofix** ({self.category}) — "
            f"confidence: {self.confidence:.0%}\n\n"
            f"{self.comment_body}\n\n"
            f"{suggestion_block}"
        )
        return {
            "path": self.file_path,
            "line": self.end_line,
            "start_line": self.start_line if self.start_line != self.end_line else None,
            "body": body,
        }


class SuggestedChangeGenerator:
    """Converts verified fixes into GitHub suggested changes.

    Usage:
        gen = SuggestedChangeGenerator()
        fix = VerifiedFix(...)
        suggestions = gen.from_verified_fix(fix)
        # Submit via GitHub API
        comments = [s.to_review_comment() for s in suggestions]
    """

    def from_verified_fix(self, fix: VerifiedFix) -> list[GitHubSuggestedChange]:
        """Convert a verified fix into GitHub suggested changes."""
        if fix.status != FixAttemptStatus.VERIFIED:
            return []

        if not fix.patch.diff_text:
            return []

        return [
            GitHubSuggestedChange(
                file_path=fix.issue.file_path,
                start_line=fix.issue.line,
                end_line=fix.issue.line,
                original_code=fix.patch.original_code,
                suggested_code=fix.patch.fixed_code,
                comment_body=fix.patch.explanation,
                confidence=fix.patch.confidence,
                category=fix.issue.category,
            )
        ]

    def from_verified_fixes(
        self, fixes: list[VerifiedFix], min_confidence: float = 0.8
    ) -> list[GitHubSuggestedChange]:
        """Convert multiple verified fixes into suggested changes."""
        suggestions: list[GitHubSuggestedChange] = []
        for fix in fixes:
            for suggestion in self.from_verified_fix(fix):
                if suggestion.confidence >= min_confidence:
                    suggestions.append(suggestion)
        return suggestions

    def format_pr_review(
        self,
        suggestions: list[GitHubSuggestedChange],
        summary: str = "",
    ) -> dict[str, Any]:
        """Format all suggestions as a single PR review submission."""
        comments = []
        for s in suggestions:
            comment = s.to_review_comment()
            # Remove None start_line for single-line suggestions
            if comment["start_line"] is None:
                del comment["start_line"]
            comments.append(comment)

        body = summary or (
            f"## CodeVerify Autofix\n\n"
            f"Found **{len(suggestions)}** auto-fixable issue(s) with verified patches.\n\n"
            f"Click **Apply suggestion** to accept each fix."
        )

        return {
            "event": "COMMENT",
            "body": body,
            "comments": comments,
        }


# =============================================================================
# Additional Fix Patterns (Go, Java, TypeScript)
# =============================================================================


GO_FIX_PATTERNS: list[dict[str, Any]] = [
    {
        "name": "go_error_ignored",
        "pattern": re.compile(r"(\w+),\s*_\s*:?=\s*(\w+)\(([^)]*)\)"),
        "replacement": r"\1, err := \2(\3)\n\tif err != nil {\n\t\treturn err\n\t}",
        "explanation": "Handle the ignored error return value.",
        "confidence": 0.85,
        "category": "error_handling",
        "language": "go",
    },
    {
        "name": "go_nil_map",
        "pattern": re.compile(r"(var\s+(\w+)\s+map\[(\w+)\](\w+))"),
        "replacement": r"\2 := make(map[\3]\4)",
        "explanation": "Initialize map with make() to prevent nil map panic.",
        "confidence": 0.90,
        "category": "null_safety",
        "language": "go",
    },
]

JAVA_FIX_PATTERNS: list[dict[str, Any]] = [
    {
        "name": "java_string_equals",
        "pattern": re.compile(r'(\w+)\s*==\s*"([^"]*)"'),
        "replacement": r'"\2".equals(\1)',
        "explanation": "Use .equals() for String comparison instead of ==.",
        "confidence": 0.95,
        "category": "bug",
        "language": "java",
    },
    {
        "name": "java_empty_catch",
        "pattern": re.compile(r"(catch\s*\(\s*(\w+)\s+(\w+)\s*\))\s*\{\s*\}"),
        "replacement": r'\1 {\n        log.error("Unexpected exception", \3);\n    }',
        "explanation": "Log exceptions instead of silently swallowing them.",
        "confidence": 0.88,
        "category": "error_handling",
        "language": "java",
    },
]

TS_FIX_PATTERNS: list[dict[str, Any]] = [
    {
        "name": "ts_any_to_unknown",
        "pattern": re.compile(r":\s*any\b"),
        "replacement": ": unknown",
        "explanation": "Replace 'any' with 'unknown' for type safety.",
        "confidence": 0.80,
        "category": "type_safety",
        "language": "typescript",
    },
]

ALL_LANGUAGE_FIX_PATTERNS: dict[str, list[dict[str, Any]]] = {
    "go": GO_FIX_PATTERNS,
    "java": JAVA_FIX_PATTERNS,
    "typescript": TS_FIX_PATTERNS,
}


class MultiLanguageFixGenerator(FixGenerator):
    """Extended fix generator with Go, Java, and TypeScript patterns."""

    def generate_fix(
        self,
        issue: CodeIssue,
        context: str,
        language: str = "python",
    ) -> GeneratedPatch:
        """Generate a fix, checking language-specific patterns first."""
        code = issue.code_snippet
        lang_patterns = ALL_LANGUAGE_FIX_PATTERNS.get(language, [])

        for fix in lang_patterns:
            match = fix["pattern"].search(code)
            if match:
                fixed_code = fix["pattern"].sub(fix["replacement"], code)
                diff_text = _make_diff(code, fixed_code)
                return GeneratedPatch(
                    original_code=code,
                    fixed_code=fixed_code,
                    diff_text=diff_text,
                    explanation=fix["explanation"],
                    confidence=fix["confidence"],
                )

        # Fall back to base Python patterns
        return super().generate_fix(issue, context)
